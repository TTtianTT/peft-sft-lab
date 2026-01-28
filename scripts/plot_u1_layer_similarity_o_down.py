#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot inter-layer similarity of u1 (top left singular vector) for LoRA ΔW = B@A
specifically for:
  - self_attn.o_proj  (o_proj)
  - mlp.down_proj     (down_proj)

Outputs (per module):
  - heatmap: <out_dir>/<module>_u1_layer_similarity.<ext>
  - optional adjacent-layer curve: <out_dir>/<module>_u1_adjacent_similarity.<ext>
"""

import os
import re
import argparse
from typing import Dict, Tuple, Optional, Any, List

import torch
import matplotlib.pyplot as plt

# Optional HF download
try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

# Optional safetensors
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


def collect_lora_pairs(
    state_dict: Dict[str, torch.Tensor],
    module_regex: str = ""
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    mod_re = re.compile(module_regex) if module_regex.strip() else None

    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, t in state_dict.items():
        if not isinstance(t, torch.Tensor):
            continue

        # match A
        for pat in _LORA_A_PATTERNS:
            m = pat.match(k)
            if m:
                pref = m.group(1)
                if mod_re and mod_re.search(pref) is None:
                    break
                pairs.setdefault(pref, {})["A"] = t
                break

        # match B
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
    """
    base_model.model.layers.12.self_attn.o_proj -> (12, self_attn.o_proj)
    """
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

    # A is (in,r)
    if A.shape[1] == rB:
        A = A.t().contiguous()
        rA, _ = A.shape
        if rA == rB:
            return A, B

    # B is (r,out)
    if B.shape[0] == rA:
        B = B.t().contiguous()
        _, rB = B.shape
        if rA == rB:
            return A, B

    raise RuntimeError(f"Cannot align ranks: A{tuple(A.shape)} B{tuple(B.shape)}")


@torch.no_grad()
def u1_from_AB(A: torch.Tensor, B: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute u1 (top left singular vector) of ΔW = B @ A using QR-compressed SVD.
    Returns u1 on CPU float16.
    """
    A, B = _fix_AB_shapes(A, B)
    A = A.to(device=device, dtype=torch.float32)
    B = B.to(device=device, dtype=torch.float32)

    r, _ = A.shape
    out_dim, r2 = B.shape
    if r != r2:
        raise RuntimeError("Rank mismatch after fix")

    # QR(B)=Qb Rb ; QR(A^T)=Qa Ra
    Qb, Rb = torch.linalg.qr(B, mode="reduced")       # (out,r), (r,r)
    Qa, Ra = torch.linalg.qr(A.t(), mode="reduced")   # (in,r),  (r,r)

    core = Rb @ Ra.t()                                # (r,r)
    Uc, _, _ = torch.linalg.svd(core, full_matrices=False)

    U = Qb @ Uc                                       # (out,r)
    u1 = U[:, 0].detach().cpu().to(torch.float16)     # (out,)
    return u1


def cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.float()
    b = b.float()
    return float((a @ b) / (a.norm() * b.norm() + eps))


def build_u1_similarity_matrix(u1_by_layer: Dict[int, torch.Tensor], use_abs: bool = True) -> Tuple[List[int], torch.Tensor]:
    layers = sorted(u1_by_layer.keys())
    n = len(layers)
    mat = torch.empty((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(n):
            c = cos(u1_by_layer[layers[i]], u1_by_layer[layers[j]])
            mat[i, j] = abs(c) if use_abs else c
    return layers, mat


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


def plot_heatmap(mat: torch.Tensor, layers: List[int], title: str, out_path: str,
                 font_scale: float, tick_step: int, vmin: float, vmax: float, dpi: int) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    apply_font_scale(font_scale)

    fig = plt.figure(figsize=(8.8, 7.4))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat.numpy(), aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("layer")
    ax.set_ylabel("layer")

    n = len(layers)
    if n <= 1:
        ax.set_xticks([0]); ax.set_yticks([0])
        ax.set_xticklabels([str(layers[0])]); ax.set_yticklabels([str(layers[0])])
    else:
        if tick_step <= 0:
            tick_step = 4
        ticks = list(range(0, n, tick_step))
        if (n - 1) not in ticks:
            ticks.append(n - 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(layers[t]) for t in ticks], rotation=0)
        ax.set_yticklabels([str(layers[t]) for t in ticks])

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_adjacent_curve(layers: List[int], mat: torch.Tensor, title: str, out_path: str,
                        font_scale: float, dpi: int) -> None:
    """
    Plot |cos(u1_l, u1_{l+1})| vs layer index l (adjacent similarity).
    mat is similarity matrix aligned with layers.
    """
    if len(layers) < 2:
        return
    apply_font_scale(font_scale)
    xs = []
    ys = []
    for i in range(len(layers) - 1):
        xs.append(layers[i])
        ys.append(float(mat[i, i + 1].item()))

    fig = plt.figure(figsize=(8.8, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, marker="o")
    ax.set_title(title)
    ax.set_xlabel("layer")
    ax.set_ylabel("|cos(u1_l, u1_{l+1})|")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, required=True, help="HF repo_id or local adapter dir")
    ap.add_argument("--out_dir", type=str, default="./u1_layer_similarity_out")
    ap.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.cache/hf_internal_svd"))
    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--abs", action="store_true", help="Use abs(cos) (recommended). If not set, uses signed cos.")
    ap.add_argument("--font_scale", type=float, default=2.0)
    ap.add_argument("--tick_step", type=int, default=4, help="Tick step on heatmap axes (e.g., 4 => 0,4,8,...).")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--ext", type=str, default="pdf", choices=["pdf", "png"])
    ap.add_argument("--no_adjacent", action="store_true", help="Do not plot adjacent-layer curve")
    args = ap.parse_args()

    token = (args.hf_token or "").strip() or None
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.out_dir, exist_ok=True)

    local_dir = ensure_local(args.adapter, cache_dir=args.cache_dir, token=token)
    sd = load_adapter_state(local_dir)

    # Only keep o_proj / down_proj
    # This regex matches prefixes ending with ".o_proj" or ".down_proj"
    module_regex = r"(?:^|\.)(?:o_proj|down_proj)$"
    pairs = collect_lora_pairs(sd, module_regex=module_regex)
    print(f"[INFO] Found {len(pairs)} LoRA modules matching regex: {module_regex}")

    # Collect u1 per layer for each target module
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

        # suffix should look like "self_attn.o_proj" or "mlp.down_proj"
        if suffix.endswith("o_proj") and ("self_attn.o_proj" in suffix or suffix.endswith("self_attn.o_proj")):
            u1_o[layer] = u1
        elif suffix.endswith("down_proj") and ("mlp.down_proj" in suffix or suffix.endswith("mlp.down_proj")):
            u1_down[layer] = u1
        else:
            # fallback: if you really want all o_proj/down_proj regardless of parent
            # uncomment these two lines:
            # if suffix.endswith("o_proj"): u1_o[layer] = u1
            # if suffix.endswith("down_proj"): u1_down[layer] = u1
            pass

    if miss > 0:
        print(f"[WARN] Failed to compute u1 for {miss} modules (shape mismatch / SVD issues).")

    def run_one(name: str, u1_by_layer: Dict[int, torch.Tensor]) -> None:
        if len(u1_by_layer) < 2:
            print(f"[WARN] {name}: not enough layers found (got {len(u1_by_layer)}).")
            return

        layers, mat = build_u1_similarity_matrix(u1_by_layer, use_abs=bool(args.abs))
        out_heat = os.path.join(args.out_dir, f"{name}_u1_layer_similarity.{args.ext}")
        plot_heatmap(
            mat=mat,
            layers=layers,
            title=f"{name} | u1 layer similarity ({'abs-cos' if args.abs else 'cos'})",
            out_path=out_heat,
            font_scale=args.font_scale,
            tick_step=args.tick_step,
            vmin=0.0 if args.abs else -1.0,
            vmax=1.0,
            dpi=args.dpi,
        )
        print(f"[SAVE] {out_heat}")

        if not args.no_adjacent:
            out_adj = os.path.join(args.out_dir, f"{name}_u1_adjacent_similarity.{args.ext}")
            plot_adjacent_curve(
                layers=layers,
                mat=mat,
                title=f"{name} | adjacent-layer |cos(u1_l, u1_l+1)|",
                out_path=out_adj,
                font_scale=args.font_scale,
                dpi=args.dpi,
            )
            print(f"[SAVE] {out_adj}")

    run_one("o_proj", u1_o)
    run_one("down_proj", u1_down)

    print(f"[DONE] out_dir = {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
