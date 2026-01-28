#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Intra-layer u1 similarity between:
  - self_attn.o_proj  (o_proj)
  - mlp.down_proj     (down_proj)

For each layer l:
  s_l = |cos(u1_o_proj(l), u1_down_proj(l))|

Outputs:
  - <out_dir>/u1_intralayer_oproj_vs_downproj.<ext>
  - <out_dir>/u1_intralayer_oproj_vs_downproj.tsv
"""

import os
import re
import argparse
from typing import Dict, Tuple, Optional, List

import torch
import matplotlib.pyplot as plt

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

try:
    from safetensors.torch import load_file as safe_load
except Exception:
    safe_load = None


# ----------------------------
# HF / IO
# ----------------------------

def ensure_local(repo_or_path: str, cache_dir: str, token: Optional[str]) -> str:
    if os.path.isdir(repo_or_path):
        return repo_or_path
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub not installed. pip install huggingface_hub")
    allow = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin",
        "adapter_model.pt",
        "*.json",
    ]
    return snapshot_download(
        repo_id=repo_or_path,
        cache_dir=cache_dir,
        token=token,
        allow_patterns=allow,
        local_files_only=False,
    )


def load_adapter_state(local_dir: str) -> Dict[str, torch.Tensor]:
    st = os.path.join(local_dir, "adapter_model.safetensors")
    if os.path.exists(st) and safe_load is not None:
        return safe_load(st)
    for fn in ["adapter_model.bin", "pytorch_model.bin", "adapter_model.pt"]:
        p = os.path.join(local_dir, fn)
        if os.path.exists(p):
            obj = torch.load(p, map_location="cpu")
            if isinstance(obj, dict) and "state_dict" in obj:
                obj = obj["state_dict"]
            if not isinstance(obj, dict):
                raise RuntimeError(f"Unexpected weight format in {p}")
            return obj
    raise RuntimeError(f"Cannot find adapter weights under: {local_dir}")


# ----------------------------
# LoRA key parsing
# ----------------------------

_LORA_A_PATTERNS = [
    re.compile(r"^(.*)\.lora_A(?:\.default)?\.weight$"),
    re.compile(r"^(.*)\.lora_A\.weight$"),
]
_LORA_B_PATTERNS = [
    re.compile(r"^(.*)\.lora_B(?:\.default)?\.weight$"),
    re.compile(r"^(.*)\.lora_B\.weight$"),
]
_LAYER_RE = re.compile(r"\.(?:layers|h)\.(\d+)\.(.*)$")  # layer idx + suffix


def collect_lora_pairs(state_dict: Dict[str, torch.Tensor], module_regex: str = "") -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    mod_re = re.compile(module_regex) if module_regex.strip() else None

    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, t in state_dict.items():
        if not isinstance(t, torch.Tensor):
            continue

        for pat in _LORA_A_PATTERNS:
            m = pat.match(k)
            if m:
                pref = m.group(1)
                if mod_re and mod_re.search(pref) is None:
                    break
                pairs.setdefault(pref, {})["A"] = t
                break

        for pat in _LORA_B_PATTERNS:
            m = pat.match(k)
            if m:
                pref = m.group(1)
                if mod_re and mod_re.search(pref) is None:
                    break
                pairs.setdefault(pref, {})["B"] = t
                break

    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for pref, ab in pairs.items():
        if "A" in ab and "B" in ab:
            out[pref] = (ab["A"], ab["B"])
    return out


def parse_layer_and_suffix(prefix: str) -> Tuple[Optional[int], str]:
    m = _LAYER_RE.search(prefix)
    if not m:
        return None, prefix
    return int(m.group(1)), m.group(2)


# ----------------------------
# QR-compressed SVD for ΔW = B @ A
# ----------------------------

def _fix_AB_shapes(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Expect A:(r,in), B:(out,r). Fix common transposes.
    if A.dim() != 2 or B.dim() != 2:
        raise RuntimeError(f"Unexpected dims: A{tuple(A.shape)} B{tuple(B.shape)}")

    rA, _ = A.shape
    _, rB = B.shape
    if rA == rB:
        return A, B

    if A.shape[1] == rB:  # A is (in,r)
        A = A.t().contiguous()
        rA, _ = A.shape
        if rA == rB:
            return A, B

    if B.shape[0] == rA:  # B is (r,out)
        B = B.t().contiguous()
        _, rB = B.shape
        if rA == rB:
            return A, B

    raise RuntimeError(f"Cannot align ranks: A{tuple(A.shape)} B{tuple(B.shape)}")


@torch.no_grad()
def u1_from_AB(A: torch.Tensor, B: torch.Tensor, device: torch.device) -> torch.Tensor:
    A, B = _fix_AB_shapes(A, B)
    A = A.to(device=device, dtype=torch.float32)
    B = B.to(device=device, dtype=torch.float32)

    r, _ = A.shape
    _, r2 = B.shape
    if r != r2:
        raise RuntimeError("Rank mismatch after fix")

    Qb, Rb = torch.linalg.qr(B, mode="reduced")       # (out,r), (r,r)
    Qa, Ra = torch.linalg.qr(A.t(), mode="reduced")   # (in,r),  (r,r)
    core = Rb @ Ra.t()                                # (r,r)

    Uc, _, _ = torch.linalg.svd(core, full_matrices=False)
    U = Qb @ Uc                                       # (out,r)
    return U[:, 0].detach().cpu().to(torch.float16)   # u1 on CPU


def cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.float()
    b = b.float()
    return float((a @ b) / (a.norm() * b.norm() + eps))


# ----------------------------
# Plotting
# ----------------------------

def apply_font_scale(font_scale: float) -> None:
    base = 10.0
    fs = base * float(font_scale)
    plt.rcParams.update({
        "font.size": fs,
        "axes.titlesize": fs * 1.1,
        "axes.labelsize": fs,
        "xtick.labelsize": fs * 0.85,
        "ytick.labelsize": fs * 0.85,
        "legend.fontsize": fs * 0.9,
        "figure.titlesize": fs * 1.15,
    })


def plot_curve(layers: List[int], sims: List[float], out_path: str, font_scale: float, dpi: int) -> None:
    apply_font_scale(font_scale)
    fig = plt.figure(figsize=(10.0, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(layers, sims, marker="o")
    ax.set_xlabel("layer")
    ax.set_ylabel("|cos(u1_o_proj, u1_down_proj)|")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.set_title("Intra-layer u1 similarity: o_proj vs down_proj")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_bar(layers: List[int], sims: List[float], out_path: str, font_scale: float, dpi: int) -> None:
    apply_font_scale(font_scale)
    fig = plt.figure(figsize=(max(12.0, 0.28 * len(layers) + 6.0), 4.8))
    ax = fig.add_subplot(111)
    xs = list(range(len(layers)))
    ax.bar(xs, sims)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in layers], rotation=0)
    ax.set_xlabel("layer")
    ax.set_ylabel("|cos(u1_o_proj, u1_down_proj)|")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Intra-layer u1 similarity (bar): o_proj vs down_proj")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, required=True, help="HF repo_id or local adapter dir")
    ap.add_argument("--out_dir", type=str, default="./u1_intralayer_out")
    ap.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.cache/hf_internal_svd"))
    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--abs", action="store_true", help="Use abs(cos) (recommended). If not set, uses signed cos.")
    ap.add_argument("--font_scale", type=float, default=2.0)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--ext", type=str, default="pdf", choices=["pdf", "png"])
    ap.add_argument("--bar", action="store_true", help="Also output a bar chart.")
    ap.add_argument("--loose_parent", action="store_true",
                    help="Do not enforce self_attn/mlp parent; any *o_proj / *down_proj are accepted.")
    args = ap.parse_args()

    token = (args.hf_token or "").strip() or None
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device if args.device != "auto" else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    local_dir = ensure_local(args.adapter, cache_dir=args.cache_dir, token=token)
    sd = load_adapter_state(local_dir)

    # filter only o_proj/down_proj
    module_regex = r"(?:^|\.)(?:o_proj|down_proj)$"
    pairs = collect_lora_pairs(sd, module_regex=module_regex)
    print(f"[INFO] Found {len(pairs)} LoRA modules matching regex: {module_regex}")

    u1_o: Dict[int, torch.Tensor] = {}
    u1_down: Dict[int, torch.Tensor] = {}
    miss = 0

    for pref, (A, B) in pairs.items():
        layer, suffix = parse_layer_and_suffix(pref)
        if layer is None:
            continue
        try:
            u1 = u1_from_AB(A, B, device=device)
        except Exception:
            miss += 1
            continue

        if suffix.endswith("o_proj"):
            if args.loose_parent or ("self_attn.o_proj" in suffix or suffix.endswith("self_attn.o_proj")):
                u1_o[layer] = u1

        if suffix.endswith("down_proj"):
            if args.loose_parent or ("mlp.down_proj" in suffix or suffix.endswith("mlp.down_proj")):
                u1_down[layer] = u1

    if miss > 0:
        print(f"[WARN] Failed to compute u1 for {miss} modules (shape mismatch / SVD issues).")

    common_layers = sorted(set(u1_o.keys()) & set(u1_down.keys()))
    if len(common_layers) < 1:
        raise SystemExit(f"[ERROR] No common layers found. "
                         f"o_proj layers={len(u1_o)}, down_proj layers={len(u1_down)}. "
                         f"Try --loose_parent.")

    sims: List[float] = []
    for l in common_layers:
        c = cos(u1_o[l], u1_down[l])
        sims.append(abs(c) if args.abs else c)

    # write tsv
    tsv_path = os.path.join(args.out_dir, "u1_intralayer_oproj_vs_downproj.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("layer\tsim\n")
        for l, s in zip(common_layers, sims):
            f.write(f"{l}\t{s:.6f}\n")
    print(f"[SAVE] {tsv_path}")

    # plot curve
    fig_path = os.path.join(args.out_dir, f"u1_intralayer_oproj_vs_downproj.{args.ext}")
    plot_curve(common_layers, sims, fig_path, font_scale=args.font_scale, dpi=args.dpi)
    print(f"[SAVE] {fig_path}")

    if args.bar:
        bar_path = os.path.join(args.out_dir, f"u1_intralayer_oproj_vs_downproj_bar.{args.ext}")
        plot_bar(common_layers, sims, bar_path, font_scale=args.font_scale, dpi=args.dpi)
        print(f"[SAVE] {bar_path}")

    # quick stats
    s_t = torch.tensor(sims, dtype=torch.float32)
    print(f"[STAT] layers={len(common_layers)}  mean={float(s_t.mean()):.4f}  std={float(s_t.std(unbiased=False)):.4f}  "
          f"min={float(s_t.min()):.4f}  max={float(s_t.max()):.4f}")
    print(f"[DONE] out_dir = {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
