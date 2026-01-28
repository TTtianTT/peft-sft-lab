#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch LoRA alignment plotting (u1 / v1 / subU / subV heatmaps + alignment-strength bar chart),
with larger fonts (default 2x).

Key additions vs your original:
- Supports multiple adapters in one run: --adapters (repeatable) or --adapter_list_file
- Per-adapter bar chart summarizing "alignment strength" per module group
- Font scaling: --font_scale (default 2.0), applied to titles/labels/ticks/colorbar
- Writes a TSV summary with off-diagonal means for each group
"""

import os
import re
import json
import argparse
from typing import Dict, Tuple, Optional, Any, List

import torch

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

try:
    from safetensors.torch import load_file as safe_load
except Exception:
    safe_load = None

import matplotlib.pyplot as plt


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

_LAYER_RE = re.compile(r"\.(?:layers|h)\.(\d+)\.(.*)$")  # layer index + suffix


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


def parse_layer_and_group(prefix: str) -> Tuple[Optional[int], str]:
    """
    base_model.model.layers.12.self_attn.q_proj  -> layer=12, group=self_attn.q_proj
    """
    m = _LAYER_RE.search(prefix)
    if not m:
        return None, prefix
    return int(m.group(1)), m.group(2)


def sanitize_name(s: str) -> str:
    s = s.replace("/", "_").replace(":", "_")
    s = s.replace("..", ".")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:180]


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
def svd_features_from_AB(
    A: torch.Tensor,
    B: torch.Tensor,
    top_m: int,
    device: torch.device
) -> Dict[str, Any]:
    """
    Return u1, v1, Um, Vm (columns orthonormal), plus dims/rank.
    """
    A, B = _fix_AB_shapes(A, B)
    A = A.to(device=device, dtype=torch.float32)
    B = B.to(device=device, dtype=torch.float32)

    r, in_dim = A.shape
    out_dim, r2 = B.shape
    if r != r2:
        raise RuntimeError("Rank mismatch after fix")

    # QR(B)=Qb Rb ; QR(A^T)=Qa Ra
    Qb, Rb = torch.linalg.qr(B, mode="reduced")       # (out,r), (r,r)
    Qa, Ra = torch.linalg.qr(A.t(), mode="reduced")   # (in,r),  (r,r)

    core = Rb @ Ra.t()                                # (r,r)
    Uc, _, Vh = torch.linalg.svd(core, full_matrices=False)
    Vc = Vh.t()

    U = Qb @ Uc                                       # (out,r)
    V = Qa @ Vc                                       # (in,r)

    m_eff = max(1, min(int(top_m), int(r)))

    u1 = U[:, 0].detach().cpu()
    v1 = V[:, 0].detach().cpu()
    Um = U[:, :m_eff].detach().cpu()
    Vm = V[:, :m_eff].detach().cpu()

    # re-orthonormalize for numerical robustness
    Um, _ = torch.linalg.qr(Um, mode="reduced")
    Vm, _ = torch.linalg.qr(Vm, mode="reduced")

    return {
        "rank": int(r),
        "m_eff": int(m_eff),
        "out_dim": int(out_dim),
        "in_dim": int(in_dim),
        "u1": u1.to(torch.float16),
        "v1": v1.to(torch.float16),
        "Um": Um.to(torch.float16),
        "Vm": Vm.to(torch.float16),
    }


def cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.float()
    b = b.float()
    return float((a @ b) / (a.norm() * b.norm() + eps))


def subspace_overlap(A: torch.Tensor, B: torch.Tensor) -> float:
    """
    A:(d,m), B:(d,m), columns orthonormal
    overlap = ||A^T B||_F^2 / m  in [0,1]
    """
    m = A.shape[1]
    if m == 0:
        return float("nan")
    M = A.t() @ B
    v = (M ** 2).sum().item() / float(m)
    return float(max(0.0, min(1.0, v)))


# ----------------------------
# Similarity matrices + summaries
# ----------------------------

def build_similarity_mats(
    items: Dict[int, Dict[str, Any]],
    top_m: int,
    use_abs: bool = True
) -> Dict[str, Any]:
    layers = sorted(items.keys())
    n = len(layers)
    u1 = [items[L]["u1"] for L in layers]
    v1 = [items[L]["v1"] for L in layers]

    u1_mat = torch.full((n, n), float("nan"))
    v1_mat = torch.full((n, n), float("nan"))
    subU_mat = torch.full((n, n), float("nan"))
    subV_mat = torch.full((n, n), float("nan"))
    sub_mat = torch.full((n, n), float("nan"))

    for i in range(n):
        for j in range(n):
            ci = cos(u1[i], u1[j])
            cv = cos(v1[i], v1[j])
            if use_abs:
                ci = abs(ci)
                cv = abs(cv)
            u1_mat[i, j] = ci
            v1_mat[i, j] = cv

            mi = min(
                int(top_m),
                int(items[layers[i]]["rank"]),
                int(items[layers[j]]["rank"])
            )
            if mi <= 0:
                continue
            Um_i = items[layers[i]]["Um"][:, :mi].float()
            Um_j = items[layers[j]]["Um"][:, :mi].float()
            Vm_i = items[layers[i]]["Vm"][:, :mi].float()
            Vm_j = items[layers[j]]["Vm"][:, :mi].float()
            if Um_i.shape != Um_j.shape or Vm_i.shape != Vm_j.shape:
                continue

            su = subspace_overlap(Um_i, Um_j)
            sv = subspace_overlap(Vm_i, Vm_j)
            subU_mat[i, j] = su
            subV_mat[i, j] = sv
            sub_mat[i, j] = 0.5 * (su + sv)

    return {
        "layers": layers,
        "u1": u1_mat,
        "v1": v1_mat,
        "subU": subU_mat,
        "subV": subV_mat,
        "sub": sub_mat,
    }


def offdiag_mean(mat: torch.Tensor) -> float:
    """
    Mean of off-diagonal entries, ignoring NaNs.
    """
    if mat.numel() == 0:
        return float("nan")
    n = mat.shape[0]
    mask = torch.ones((n, n), dtype=torch.bool)
    mask.fill_diagonal_(False)
    vals = mat[mask]
    vals = vals[~torch.isnan(vals)]
    if vals.numel() == 0:
        return float("nan")
    return float(vals.mean().item())


def global_mean(mats: List[torch.Tensor]) -> torch.Tensor:
    """
    elementwise mean ignoring NaN
    """
    if not mats:
        return torch.empty(0)
    stack = torch.stack(mats, dim=0)
    mask = ~torch.isnan(stack)
    num = mask.sum(dim=0).clamp(min=1)
    s = torch.where(mask, stack, torch.zeros_like(stack)).sum(dim=0)
    out = s / num
    out[mask.sum(dim=0) == 0] = float("nan")
    return out


# ----------------------------
# Plotting (with font scale)
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


def plot_heatmap(
    mat: torch.Tensor,
    layers: List[int],
    title: str,
    out_path: str,
    font_scale: float,
    vmin: float = 0.0,
    vmax: float = 1.0
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    apply_font_scale(font_scale)

    fig = plt.figure(figsize=(8.8, 7.4))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat.numpy(), aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)

    n = len(layers)
    if n <= 1:
        # 单层：只显示一个刻度
        ax.set_xticks([0])
        ax.set_yticks([0])
        ax.set_xticklabels([str(layers[0])])
        ax.set_yticklabels([str(layers[0])])
    else:
        # 只显示第一个和最后一个（位置 0 与 n-1）
        ax.set_xticks([0, n - 1])
        ax.set_yticks([0, n - 1])
        ax.set_xticklabels([str(layers[0]), str(layers[-1])], rotation=0)
        ax.set_yticklabels([str(layers[0]), str(layers[-1])])

    ax.set_xlabel("layer")
    ax.set_ylabel("layer")
    ax.tick_params(bottom=True, left=True)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)



def plot_alignment_bar(
    rows: List[Dict[str, Any]],
    out_path: str,
    font_scale: float,
    max_bars: int = 40,
    sort_by: str = "subU_offdiag_mean"
) -> None:
    """
    rows: list of dicts with fields:
      group, u1_offdiag_mean, v1_offdiag_mean, subU_offdiag_mean, subV_offdiag_mean, baseline_m_over_out
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    apply_font_scale(font_scale)

    # filter NaNs and sort
    clean = [r for r in rows if isinstance(r.get(sort_by), float) and (not (r[sort_by] != r[sort_by]))]
    clean.sort(key=lambda r: r.get(sort_by, float("-inf")), reverse=True)
    show = clean[:max_bars] if max_bars > 0 else clean

    if not show:
        return

    labels = [r["group"] for r in show]
    vals = [r[sort_by] for r in show]
    baselines = [r.get("baseline_m_over_out", float("nan")) for r in show]

    fig = plt.figure(figsize=(max(12.0, 0.35 * len(show) + 6.0), 7.2))
    ax = fig.add_subplot(111)

    x = list(range(len(show)))
    ax.bar(x, vals)

    # optional baseline (random-subspace m/d) as a thin line
    # if baselines are finite and not wildly different, plot them
    if all((b == b) for b in baselines):  # no NaN
        ax.plot(x, baselines, linewidth=2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel(sort_by)
    ax.set_title(f"Alignment strength summary (top {len(show)} groups)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ----------------------------
# Runner per adapter
# ----------------------------

def process_one_adapter(args: argparse.Namespace, adapter: str) -> None:
    token = (args.hf_token or "").strip() or None
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.out_dir, exist_ok=True)

    adapter_tag = sanitize_name(adapter.split("/", 1)[1] if "/" in adapter else os.path.basename(adapter.rstrip("/")))
    work_dir = os.path.join(args.out_dir, adapter_tag)
    os.makedirs(work_dir, exist_ok=True)

    feat_cache = os.path.join(work_dir, "features_cache.pt")
    if args.resume and os.path.exists(feat_cache):
        features = torch.load(feat_cache, map_location="cpu")
        print(f"[LOAD] {adapter_tag} features cache: {feat_cache}")
    else:
        local_dir = ensure_local(adapter, cache_dir=args.cache_dir, token=token)
        sd = load_adapter_state(local_dir)
        pairs = collect_lora_pairs(sd, module_regex=args.module_regex)
        print(f"[{adapter_tag}] Found {len(pairs)} LoRA modules (after regex).")

        features: Dict[str, Dict[str, Any]] = {}
        failed = 0
        for pref, (A, B) in pairs.items():
            try:
                feat = svd_features_from_AB(A, B, top_m=args.top_m, device=device)
                layer, group = parse_layer_and_group(pref)
                feat["layer"] = layer
                feat["group"] = group
                features[pref] = feat
            except Exception:
                failed += 1

        torch.save(features, feat_cache)
        print(f"[SAVE] {adapter_tag} {feat_cache}  (ok={len(features)}, fail={failed})")

    # group by group name (same module across layers)
    groups: Dict[str, Dict[int, Dict[str, Any]]] = {}
    meta: Dict[str, Any] = {"adapter": adapter, "top_m": args.top_m, "abs": bool(args.abs), "groups": {}}

    for pref, feat in features.items():
        layer = feat.get("layer", None)
        group = feat.get("group", pref)
        if layer is None:
            continue
        groups.setdefault(group, {})[int(layer)] = feat

    print(f"[{adapter_tag}] Groups: {len(groups)} (e.g., self_attn.q_proj, mlp.down_proj, ...).")

    out_groups = os.path.join(work_dir, "groups")
    os.makedirs(out_groups, exist_ok=True)

    # summary outputs
    tsv_path = os.path.join(work_dir, "group_summary.tsv")
    align_tsv = os.path.join(work_dir, "alignment_strength.tsv")
    bar_png = os.path.join(work_dir, "ALIGNMENT_BAR_subU.pdf")

    base_layers_ref: Optional[List[int]] = None
    u1_mats, v1_mats, sub_mats = [], [], []

    align_rows: List[Dict[str, Any]] = []

    with open(tsv_path, "w", encoding="utf-8") as fsum:
        fsum.write("group\tlayers\tout_dim\tin_dim\trank_min\trank_max\n")

        for group, items in sorted(groups.items(), key=lambda x: x[0]):
            if len(items) < 2:
                continue

            mats = build_similarity_mats(items, top_m=args.top_m, use_abs=bool(args.abs))
            layers = mats["layers"]

            out_dim = int(items[layers[0]]["out_dim"])
            in_dim = int(items[layers[0]]["in_dim"])
            rks = [int(items[L]["rank"]) for L in layers]
            rank_min, rank_max = min(rks), max(rks)

            fsum.write(f"{group}\t{len(layers)}\t{out_dim}\t{in_dim}\t{rank_min}\t{rank_max}\n")

            gdir = os.path.join(out_groups, sanitize_name(group))
            os.makedirs(gdir, exist_ok=True)

            # save mats
            torch.save(
                {"group": group, "layers": layers,
                 "u1": mats["u1"], "v1": mats["v1"],
                 "sub": mats["sub"], "subU": mats["subU"], "subV": mats["subV"]},
                os.path.join(gdir, "mats.pt")
            )

            # heatmaps (bigger fonts)
            plot_heatmap(
                mats["u1"], layers, f"{group} | u1 abs-cos",
                os.path.join(gdir, "heatmap_u1.pdf"),
                font_scale=args.font_scale
            )
            plot_heatmap(
                mats["v1"], layers, f"{group} | v1 abs-cos",
                os.path.join(gdir, "heatmap_v1.pdf"),
                font_scale=args.font_scale
            )
            plot_heatmap(
                mats["subU"], layers, f"{group} | top-m U-subspace",
                os.path.join(gdir, "heatmap_subU.pdf"),
                font_scale=args.font_scale
            )
            plot_heatmap(
                mats["subV"], layers, f"{group} | top-m V-subspace",
                os.path.join(gdir, "heatmap_subV.pdf"),
                font_scale=args.font_scale
            )

            # alignment strength summary (off-diagonal means)
            u1_off = offdiag_mean(mats["u1"])
            v1_off = offdiag_mean(mats["v1"])
            subU_off = offdiag_mean(mats["subU"])
            subV_off = offdiag_mean(mats["subV"])
            # random-subspace baseline ~ m/out_dim (for U-space); use effective m from any layer
            m_eff = int(items[layers[0]].get("m_eff", min(args.top_m, rank_min)))
            baseline = float(m_eff) / float(out_dim) if out_dim > 0 else float("nan")

            align_rows.append({
                "group": group,
                "layers": len(layers),
                "out_dim": out_dim,
                "in_dim": in_dim,
                "rank_min": rank_min,
                "rank_max": rank_max,
                "m_eff": m_eff,
                "baseline_m_over_out": baseline,
                "u1_offdiag_mean": u1_off,
                "v1_offdiag_mean": v1_off,
                "subU_offdiag_mean": subU_off,
                "subV_offdiag_mean": subV_off,
            })

            meta["groups"][group] = {
                "layers": layers,
                "out_dir": os.path.relpath(gdir, work_dir),
                "out_dim": out_dim,
                "in_dim": in_dim,
                "rank_min": rank_min,
                "rank_max": rank_max,
                "m_eff": m_eff,
                "baseline_m_over_out": baseline,
                "u1_offdiag_mean": u1_off,
                "v1_offdiag_mean": v1_off,
                "subU_offdiag_mean": subU_off,
                "subV_offdiag_mean": subV_off,
            }

            # global mean accumulation: only average groups that share identical layer list
            if base_layers_ref is None:
                base_layers_ref = layers
                u1_mats.append(mats["u1"])
                v1_mats.append(mats["v1"])
                sub_mats.append(mats["sub"])
            else:
                if layers == base_layers_ref:
                    u1_mats.append(mats["u1"])
                    v1_mats.append(mats["v1"])
                    sub_mats.append(mats["sub"])

    # write alignment strength TSV
    with open(align_tsv, "w", encoding="utf-8") as f:
        headers = [
            "group", "layers", "out_dim", "in_dim", "rank_min", "rank_max", "m_eff",
            "baseline_m_over_out", "u1_offdiag_mean", "v1_offdiag_mean", "subU_offdiag_mean", "subV_offdiag_mean"
        ]
        f.write("\t".join(headers) + "\n")
        for r in sorted(align_rows, key=lambda x: (x.get("subU_offdiag_mean", float("-inf"))), reverse=True):
            f.write("\t".join(str(r.get(h, "")) for h in headers) + "\n")

    # bar chart: summarize alignment strength (default = subU_offdiag_mean)
    plot_alignment_bar(
        rows=align_rows,
        out_path=bar_png,
        font_scale=args.font_scale,
        max_bars=args.max_bars,
        sort_by="subU_offdiag_mean",
    )

    # save meta.json
    with open(os.path.join(work_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[{adapter_tag}] Save: group summary -> {tsv_path}")
    print(f"[{adapter_tag}] Save: alignment strength -> {align_tsv}")
    print(f"[{adapter_tag}] Save: bar chart -> {bar_png}")

    # global mean plots
    if base_layers_ref is not None and u1_mats:
        gm_u1 = global_mean(u1_mats)
        gm_v1 = global_mean(v1_mats)
        gm_sub = global_mean(sub_mats)

        torch.save(
            {"layers": base_layers_ref, "u1": gm_u1, "v1": gm_v1, "sub": gm_sub,
             "num_groups": len(u1_mats)},
            os.path.join(work_dir, "GLOBAL_MEAN.pt")
        )

        plot_heatmap(
            gm_u1, base_layers_ref,
            f"GLOBAL_MEAN | u1 abs-cos (groups={len(u1_mats)})",
            os.path.join(work_dir, "GLOBAL_MEAN_u1.pdf"),
            font_scale=args.font_scale
        )
        plot_heatmap(
            gm_v1, base_layers_ref,
            f"GLOBAL_MEAN | v1 abs-cos (groups={len(v1_mats)})",
            os.path.join(work_dir, "GLOBAL_MEAN_v1.pdf"),
            font_scale=args.font_scale
        )
        plot_heatmap(
            gm_sub, base_layers_ref,
            f"GLOBAL_MEAN | top-m subspace (groups={len(sub_mats)})",
            os.path.join(work_dir, "GLOBAL_MEAN_subspace.pdf"),
            font_scale=args.font_scale
        )
        print(f"[{adapter_tag}] OK: GLOBAL_MEAN_* saved.")
    else:
        print(f"[{adapter_tag}] Warn: Not enough groups/layers to compute GLOBAL_MEAN.")

    print(f"[{adapter_tag}] Done -> {os.path.abspath(work_dir)}\n")


# ----------------------------
# CLI / main
# ----------------------------

def read_adapter_list_file(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", type=str, nargs="*", default=None,
                    help="One or more adapters (HF repo_id or local dirs).")
    ap.add_argument("--adapter_list_file", type=str, default="",
                    help="Text file with one adapter per line (comments with # allowed).")
    ap.add_argument("--out_dir", type=str, default="./adapter_internal_uv_subspace_batch")
    ap.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.cache/hf_internal_svd"))
    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--top_m", type=int, default=8)
    ap.add_argument("--module_regex", type=str, default="",
                    help="Filter module prefixes, e.g. 'self_attn|mlp' or 'o_proj|down_proj'")
    ap.add_argument("--abs", action="store_true", help="Use abs(cos) for u1/v1")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--resume", action="store_true", help="Reuse cached features if exists")
    ap.add_argument("--font_scale", type=float, default=2.0,
                    help="Scale factor for matplotlib fonts (default 2.0 = ~2x bigger).")
    ap.add_argument("--max_bars", type=int, default=40,
                    help="Max number of groups to show in bar chart (full list always saved to TSV).")
    args = ap.parse_args()

    adapters: List[str] = []
    if args.adapters:
        adapters.extend(args.adapters)
    if args.adapter_list_file.strip():
        adapters.extend(read_adapter_list_file(args.adapter_list_file.strip()))

    # de-dup, keep order
    seen = set()
    uniq: List[str] = []
    for a in adapters:
        if a not in seen:
            uniq.append(a)
            seen.add(a)

    if not uniq:
        raise SystemExit("No adapters provided. Use --adapters ... or --adapter_list_file ...")

    for adapter in uniq:
        process_one_adapter(args, adapter)


if __name__ == "__main__":
    main()

