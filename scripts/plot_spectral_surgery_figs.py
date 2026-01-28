# plot_spectral_surgery_figs.py
# -*- coding: utf-8 -*-
"""
Generate all figures discussed earlier from your LaTeX tables.

Outputs:
  figs/
    delta_heatmap_<model>_residual_l1_calib128.pdf
    risk_reward_residual_l1_calib128.pdf
    guided_vs_random_smooth_abs_residual_l1_calib128.pdf
    guided_vs_random_grad_direction_residual_l1_calib128.pdf
    calib_sweep_<model>_<task>_residual_l1.pdf
    energy_ablation_delta_<model>_grad_direction_residual_calib128.pdf
    locality_tradeoff_delta_<model>_grad_direction_l1_calib128.pdf
  parsed_results.csv

Run:
  python plot_spectral_surgery_figs.py
"""

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # FIXED: Added missing import
from adjustText import adjust_text  # FIXED: Added missing import

# -----------------------------
# Global Style Settings
# -----------------------------
sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks")
plt.rcParams['font.family'] = 'serif'

# -----------------------------
# 1) LaTeX Tables
# -----------------------------
TABLE_RESIDUAL_L1_SWEEP = r"""
Llama-3.1-8B & 32 & GSM8K & acc & \textbf{0.668} & 0.657 & 0.660 & 0.657 & 0.622 \\
Llama-3.1-8B & 32 & HumanEval & pass@1 & \textbf{0.494} & \textbf{0.494} & 0.476 & 0.488 & 0.476 \\
Llama-3.1-8B & 32 & IFEval & score & 0.303 & \textbf{0.315} & 0.311 & 0.307 & 0.224 \\
Llama-3.1-8B & 32 & CSQA & acc & 0.740 & 0.735 & 0.737 & 0.742 & \textbf{0.785} \\
\addlinespace
Llama-3.1-8B & 64 & GSM8K & acc & 0.659 & \textbf{0.661} & 0.654 & 0.657 & 0.621 \\
Llama-3.1-8B & 64 & HumanEval & pass@1 & 0.488 & 0.482 & 0.488 & \textbf{0.506} & 0.476 \\
Llama-3.1-8B & 64 & IFEval & score & 0.312 & 0.311 & \textbf{0.317} & 0.308 & 0.230 \\
Llama-3.1-8B & 64 & CSQA & acc & 0.740 & 0.734 & 0.738 & 0.740 & \textbf{0.782} \\
\addlinespace
Llama-3.1-8B & 128 & GSM8K & acc & 0.657 & 0.659 & \textbf{0.668} & 0.661 & 0.614 \\
Llama-3.1-8B & 128 & HumanEval & pass@1 & 0.488 & 0.488 & \textbf{0.494} & 0.488 & 0.476 \\
Llama-3.1-8B & 128 & IFEval & score & 0.305 & \textbf{0.321} & 0.312 & 0.306 & 0.223 \\
Llama-3.1-8B & 128 & CSQA & acc & 0.740 & 0.736 & 0.735 & 0.743 & \textbf{0.784} \\
\addlinespace
Llama-3.1-8B & 256 & GSM8K & acc & 0.655 & 0.657 & 0.659 & \textbf{0.660} & 0.610 \\
Llama-3.1-8B & 256 & HumanEval & pass@1 & \textbf{0.494} & 0.488 & \textbf{0.494} & 0.476 & 0.470 \\
Llama-3.1-8B & 256 & IFEval & score & 0.305 & 0.312 & \textbf{0.323} & 0.314 & 0.218 \\
Llama-3.1-8B & 256 & CSQA & acc & 0.741 & 0.740 & 0.737 & 0.740 & \textbf{0.783} \\
\midrule
Qwen3-8B & 32 & GSM8K & acc & 0.809 & 0.801 & 0.807 & 0.811 & \textbf{0.814} \\
Qwen3-8B & 32 & HumanEval & pass@1 & 0.488 & 0.512 & 0.500 & 0.494 & \textbf{0.530} \\
Qwen3-8B & 32 & IFEval & score & 0.590 & 0.525 & 0.534 & \textbf{0.603} & 0.179 \\
Qwen3-8B & 32 & CSQA & acc & 0.852 & 0.852 & 0.851 & 0.853 & \textbf{0.855} \\
\addlinespace
Qwen3-8B & 64 & GSM8K & acc & 0.807 & 0.806 & 0.805 & 0.810 & \textbf{0.811} \\
Qwen3-8B & 64 & HumanEval & pass@1 & 0.494 & 0.482 & \textbf{0.506} & 0.488 & \textbf{0.506} \\
Qwen3-8B & 64 & IFEval & score & 0.582 & 0.535 & 0.543 & \textbf{0.598} & 0.181 \\
Qwen3-8B & 64 & CSQA & acc & 0.848 & 0.851 & 0.851 & 0.853 & \textbf{0.855} \\
\addlinespace
Qwen3-8B & 128 & GSM8K & acc & 0.815 & 0.805 & 0.802 & 0.810 & \textbf{0.816} \\
Qwen3-8B & 128 & HumanEval & pass@1 & 0.488 & 0.506 & 0.506 & 0.494 & \textbf{0.512} \\
Qwen3-8B & 128 & IFEval & score & 0.590 & 0.535 & 0.552 & \textbf{0.597} & 0.173 \\
Qwen3-8B & 128 & CSQA & acc & \textbf{0.855} & 0.852 & 0.851 & 0.852 & 0.853 \\
\addlinespace
Qwen3-8B & 256 & GSM8K & acc & 0.808 & 0.805 & 0.801 & 0.805 & \textbf{0.817} \\
Qwen3-8B & 256 & HumanEval & pass@1 & 0.494 & \textbf{0.500} & 0.488 & 0.494 & \textbf{0.500} \\
Qwen3-8B & 256 & IFEval & score & 0.588 & 0.523 & 0.547 & \textbf{0.604} & 0.169 \\
Qwen3-8B & 256 & CSQA & acc & \textbf{0.855} & 0.853 & 0.850 & 0.852 & 0.853 \\
"""

TABLE_RESIDUAL_NONE_SWEEP = r"""
Llama-3.1-8B & 32 & GSM8K & acc & \textbf{0.666} & \textbf{0.666} & 0.661 & 0.663 & 0.615 \\
Llama-3.1-8B & 32 & HumanEval & pass@1 & \textbf{0.482} & \textbf{0.482} & \textbf{0.482} & 0.476 & \textbf{0.482} \\
Llama-3.1-8B & 32 & IFEval & score & 0.317 & \textbf{0.320} & 0.311 & 0.314 & 0.186 \\
Llama-3.1-8B & 32 & CSQA & acc & 0.740 & 0.726 & 0.727 & 0.742 & \textbf{0.785} \\
\addlinespace
Llama-3.1-8B & 64 & GSM8K & acc & \textbf{0.666} & 0.660 & 0.662 & 0.661 & 0.623 \\
Llama-3.1-8B & 64 & HumanEval & pass@1 & \textbf{0.488} & 0.476 & 0.470 & \textbf{0.488} & 0.476 \\
Llama-3.1-8B & 64 & IFEval & score & 0.314 & 0.315 & \textbf{0.318} & 0.317 & 0.187 \\
Llama-3.1-8B & 64 & CSQA & acc & 0.742 & 0.727 & 0.723 & 0.742 & \textbf{0.783} \\
\addlinespace
Llama-3.1-8B & 128 & GSM8K & acc & 0.659 & \textbf{0.664} & 0.662 & 0.650 & 0.621 \\
Llama-3.1-8B & 128 & HumanEval & pass@1 & \textbf{0.500} & 0.482 & 0.488 & \textbf{0.500} & 0.470 \\
Llama-3.1-8B & 128 & IFEval & score & \textbf{0.315} & 0.309 & 0.312 & 0.314 & 0.189 \\
Llama-3.1-8B & 128 & CSQA & acc & 0.740 & 0.727 & 0.725 & 0.740 & \textbf{0.782} \\
\addlinespace
Llama-3.1-8B & 256 & GSM8K & acc & 0.659 & \textbf{0.667} & 0.665 & 0.656 & 0.625 \\
Llama-3.1-8B & 256 & HumanEval & pass@1 & 0.500 & 0.482 & 0.482 & \textbf{0.506} & 0.494 \\
Llama-3.1-8B & 256 & IFEval & score & 0.314 & 0.317 & \textbf{0.321} & 0.317 & 0.193 \\
Llama-3.1-8B & 256 & CSQA & acc & 0.740 & 0.731 & 0.728 & 0.744 & \textbf{0.783} \\
\midrule
Qwen3-8B & 32 & GSM8K & acc & 0.806 & \textbf{0.807} & \textbf{0.807} & \textbf{0.807} & 0.788 \\
Qwen3-8B & 32 & HumanEval & pass@1 & 0.482 & 0.494 & \textbf{0.506} & 0.482 & 0.482 \\
Qwen3-8B & 32 & IFEval & score & 0.584 & 0.478 & 0.490 & \textbf{0.608} & 0.017 \\
Qwen3-8B & 32 & CSQA & acc & 0.852 & 0.844 & 0.850 & \textbf{0.853} & 0.838 \\
\addlinespace
Qwen3-8B & 64 & GSM8K & acc & 0.806 & 0.806 & 0.807 & \textbf{0.808} & 0.802 \\
Qwen3-8B & 64 & HumanEval & pass@1 & 0.476 & 0.488 & 0.488 & \textbf{0.500} & 0.482 \\
Qwen3-8B & 64 & IFEval & score & 0.594 & 0.481 & 0.478 & \textbf{0.610} & 0.014 \\
Qwen3-8B & 64 & CSQA & acc & 0.852 & 0.848 & 0.848 & \textbf{0.853} & 0.841 \\
\addlinespace
Qwen3-8B & 128 & GSM8K & acc & \textbf{0.819} & 0.802 & 0.807 & 0.807 & 0.798 \\
Qwen3-8B & 128 & HumanEval & pass@1 & 0.482 & 0.476 & 0.482 & \textbf{0.488} & 0.476 \\
Qwen3-8B & 128 & IFEval & score & 0.586 & 0.488 & 0.478 & \textbf{0.606} & 0.014 \\
Qwen3-8B & 128 & CSQA & acc & 0.850 & \textbf{0.853} & 0.847 & 0.850 & 0.839 \\
\addlinespace
Qwen3-8B & 256 & GSM8K & acc & 0.806 & \textbf{0.812} & 0.803 & 0.811 & 0.797 \\
Qwen3-8B & 256 & HumanEval & pass@1 & 0.476 & 0.482 & 0.494 & \textbf{0.500} & 0.470 \\
Qwen3-8B & 256 & IFEval & score & 0.583 & 0.484 & 0.486 & \textbf{0.604} & 0.016 \\
Qwen3-8B & 256 & CSQA & acc & 0.849 & 0.848 & \textbf{0.853} & \textbf{0.853} & 0.835 \\
"""

TABLE_ATTN_INPUTS_QKV_L1_CALIB128 = r"""
Llama-3.1-8B & 128 & GSM8K & acc & 0.658 & \textbf{0.666} & \textbf{0.666} & 0.650 & 0.637 \\
Llama-3.1-8B & 128 & HumanEval & pass@1 & \textbf{0.500} & 0.482 & 0.488 & 0.488 & 0.470 \\
Llama-3.1-8B & 128 & IFEval & score & 0.312 & 0.315 & 0.312 & 0.308 & \textbf{0.317} \\
Llama-3.1-8B & 128 & CSQA & acc & 0.741 & 0.743 & 0.740 & 0.742 & \textbf{0.744} \\
\midrule
Qwen3-8B & 128 & GSM8K & acc & \textbf{0.819} & 0.802 & 0.807 & 0.810 & 0.809 \\
Qwen3-8B & 128 & HumanEval & pass@1 & 0.476 & \textbf{0.500} & 0.494 & \textbf{0.500} & \textbf{0.512} \\
Qwen3-8B & 128 & IFEval & score & \textbf{0.585} & 0.566 & 0.565 & 0.572 & 0.499 \\
Qwen3-8B & 128 & CSQA & acc & 0.849 & 0.851 & 0.854 & 0.852 & \textbf{0.857} \\
"""

TABLE_ALL_MODULES_L1_CALIB128 = r"""
Llama-3.1-8B & 128 & GSM8K & acc & 0.658 & 0.660 & \textbf{0.662} & 0.660 & 0.538 \\
Llama-3.1-8B & 128 & HumanEval & pass@1 & \textbf{0.506} & 0.457 & 0.463 & 0.494 & 0.476 \\
Llama-3.1-8B & 128 & IFEval & score & 0.313 & 0.311 & \textbf{0.315} & 0.314 & 0.195 \\
Llama-3.1-8B & 128 & CSQA & acc & 0.742 & 0.690 & 0.699 & 0.742 & \textbf{0.790} \\
\midrule
Qwen3-8B & 128 & GSM8K & acc & 0.814 & 0.802 & 0.798 & \textbf{0.818} & 0.778 \\
Qwen3-8B & 128 & HumanEval & pass@1 & 0.482 & 0.488 & 0.494 & 0.482 & \textbf{0.524} \\
Qwen3-8B & 128 & IFEval & score & \textbf{0.594} & 0.511 & 0.524 & 0.582 & 0.071 \\
Qwen3-8B & 128 & CSQA & acc & \textbf{0.852} & \textbf{0.852} & \textbf{0.852} & \textbf{0.852} & 0.846 \\
"""

TABLE_UP_GATE_L1_CALIB128 = r"""
Llama-3.1-8B & 128 & GSM8K & acc & \textbf{0.663} & 0.658 & 0.659 & 0.659 & 0.592 \\
Llama-3.1-8B & 128 & HumanEval & pass@1 & \textbf{0.500} & 0.476 & 0.470 & 0.482 & 0.451 \\
Llama-3.1-8B & 128 & IFEval & score & \textbf{0.313} & 0.299 & 0.303 & \textbf{0.313} & 0.205 \\
Llama-3.1-8B & 128 & CSQA & acc & 0.741 & 0.696 & 0.710 & 0.740 & \textbf{0.787} \\
\midrule
Qwen3-8B & 128 & GSM8K & acc & 0.805 & 0.806 & 0.803 & \textbf{0.817} & 0.789 \\
Qwen3-8B & 128 & HumanEval & pass@1 & 0.482 & 0.457 & 0.488 & 0.506 & \textbf{0.567} \\
Qwen3-8B & 128 & IFEval & score & \textbf{0.583} & 0.574 & \textbf{0.583} & 0.579 & 0.462 \\
Qwen3-8B & 128 & CSQA & acc & \textbf{0.852} & \textbf{0.852} & 0.850 & 0.849 & 0.851 \\
"""

# -----------------------------
# 2) Parsing utilities
# -----------------------------
POLICIES = ["abs_select", "smooth_abs", "random_index", "grad_direction"]


def _strip_latex_wrappers(s: str) -> str:
    prev = None
    cur = s
    while prev != cur:
        prev = cur
        cur = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cur)
        cur = re.sub(r"\\underline\{([^}]*)\}", r"\1", cur)
        cur = re.sub(r"\\emph\{([^}]*)\}", r"\1", cur)
    return cur


def to_float(cell: str) -> Optional[float]:
    cell = _strip_latex_wrappers(cell)
    m = re.search(r"-?\d+(?:\.\d+)?", cell)
    if not m:
        return None
    return float(m.group(0))


@dataclass
class TableMeta:
    preserve_energy: str  # "l1" or "none"
    module_set: str  # "residual", "attn_inputs", "all_modules", "up_gate"
    source_name: str  # for debugging


def parse_latex_rows(table_str: str, meta: TableMeta) -> pd.DataFrame:
    rows = []
    for raw in table_str.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "&" not in line:
            continue
        if any(tok in line for tok in ["\\toprule", "\\midrule", "\\bottomrule", "\\addlinespace", "\\cmidrule"]):
            continue

        line = line.replace("\\\\", "").strip()
        parts = [p.strip() for p in line.split("&")]
        if len(parts) < 9:
            continue

        model, calib, task, metric = parts[0], parts[1], parts[2], parts[3]
        baseline = to_float(parts[4])
        abs_select = to_float(parts[5])
        smooth_abs = to_float(parts[6])
        random_index = to_float(parts[7])
        grad_direction = to_float(parts[8])

        if baseline is None:
            continue

        rows.append({
            "model": _strip_latex_wrappers(model).strip(),
            "calib": int(to_float(calib)),
            "task": _strip_latex_wrappers(task).strip(),
            "metric": _strip_latex_wrappers(metric).strip(),
            "preserve_energy": meta.preserve_energy,
            "module_set": meta.module_set,
            "source": meta.source_name,
            "baseline": baseline,
            "abs_select": abs_select,
            "smooth_abs": smooth_abs,
            "random_index": random_index,
            "grad_direction": grad_direction,
        })

    df = pd.DataFrame(rows)
    for c in POLICIES:
        df[c] = df[c].astype(float)
    df["baseline"] = df["baseline"].astype(float)
    return df


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for p in POLICIES:
        df[f"delta_{p}"] = df[p] - df["baseline"]
    return df


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# 3) Plot Helpers
# -----------------------------

def plot_delta_heatmap_one_model(df_cfg: pd.DataFrame, model: str, outpath: str, title: str) -> None:
    dfm = df_cfg[df_cfg["model"] == model].copy()
    dfm = add_deltas(dfm)
    tasks = ["GSM8K", "HumanEval", "IFEval", "CSQA"]
    dfm["task"] = pd.Categorical(dfm["task"], categories=tasks, ordered=True)
    dfm = dfm.sort_values("task")

    mat = np.vstack([dfm[f"delta_{p}"].values for p in POLICIES]).T  # shape (tasks, policies)

    fig = plt.figure(figsize=(7.2, 3.4))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.05, vmax=0.05)
    ax.set_title(f"{title}\n{model}")
    ax.set_xticks(np.arange(len(POLICIES)))
    ax.set_xticklabels(POLICIES, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "white" if abs(val) > 0.02 else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ vs baseline")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_risk_reward(df_cfg: pd.DataFrame, outpath: str,
                     aligned_tasks: list, constrained_task: str = "IFEval") -> None:
    """
    Risk-Reward Plot with auto-adjusting labels and clear markers.
    """
    df = add_deltas(df_cfg)
    points = []

    for model in sorted(df["model"].unique()):
        dfm = df[df["model"] == model]
        for p in POLICIES:
            dfa = dfm[dfm["task"].isin(aligned_tasks)]
            if len(dfa) == 0:
                continue
            reward = float(np.nanmean(dfa[f"delta_{p}"].values))

            dfc = dfm[dfm["task"] == constrained_task]
            if len(dfc) == 0:
                continue
            delta_c = float(dfc[f"delta_{p}"].values[0])
            risk = max(0.0, -delta_c)

            points.append({"Model": model, "Policy": p, "Reward": reward, "Risk": risk})

    plot_df = pd.DataFrame(points)

    # ---- unified font size / weight (same vibe as previous) ----
    FS = 12          # 想更大就 13/14
    FONT_W = "bold"  # 不要细线字体

    fig, ax = plt.subplots(figsize=(6, 6))

    sns.scatterplot(
        data=plot_df,
        x="Reward", y="Risk",
        hue="Model", style="Policy",
        s=170,
        alpha=0.9,
        palette="colorblind",
        markers={"abs_select": "o", "smooth_abs": "s", "random_index": "^", "grad_direction": "D"},
        ax=ax
    )

    ax.axhline(0, color='0.6', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.axvline(0, color='0.6', linestyle='--', linewidth=1.2, alpha=0.7)

    texts = []
    for _, row in plot_df.iterrows():
        if abs(row["Reward"]) > 0.001 or row["Risk"] > 0.01:
            texts.append(
                ax.text(row["Reward"], row["Risk"], f"{row['Policy']}",
                        fontsize=FS, fontweight=FONT_W, color="0.15")
            )

    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle='-', color='0.5', lw=0.8)
    )

    ax.set_xlabel(r"Reward (Mean $\Delta$ on Aligned Tasks)", fontsize=FS, fontweight=FONT_W)
    ax.set_ylabel(fr"Risk (Drop on {constrained_task})", fontsize=FS, fontweight=FONT_W)
    ax.set_title("Safety Trade-off: Policy Risk vs. Reward",
                 pad=15, fontsize=FS+2, fontweight=FONT_W)

    ax.tick_params(axis="both", labelsize=FS)

    # ---- legend inside the axes (no outside space) ----
    # 先把 seaborn 生成的 legend 挪到图内左上角，并加一个半透明背景避免遮点看不清
    sns.move_legend(
        ax, "upper left",
        bbox_to_anchor=(0.02, 0.98),  # 图内坐标(0~1)，再往右/下就调这里
        frameon=True,
        borderaxespad=0.0
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_alpha(0.85)
        leg.get_frame().set_edgecolor("0.8")
        # legend 字体也统一变大 + 加粗
        for t in leg.get_texts():
            t.set_fontsize(FS)
            t.set_fontweight(FONT_W)
        if leg.get_title() is not None:
            leg.get_title().set_fontsize(FS)
            leg.get_title().set_fontweight(FONT_W)

    sns.despine()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)  # legend 已在图内，不需要 bbox_inches='tight'
    plt.close(fig)



def plot_guided_vs_random(df_cfg: pd.DataFrame, guided_policy: str, outpath: str, title: str) -> None:
    """
    Scatter plot comparing a guided policy vs random noise.
    """
    assert guided_policy in POLICIES and guided_policy != "random_index"
    df = add_deltas(df_cfg)

    x_vals = df["delta_random_index"].values
    y_vals = df[f"delta_{guided_policy}"].values

    temp_df = df.copy()
    temp_df["x"] = x_vals
    temp_df["y"] = y_vals

    # ---- unified font size / weight ----
    FS = 12  # 你想更大就改 13/14
    FONT_W = "bold"  # 或 "semibold"

    fig, ax = plt.subplots(figsize=(6, 6))

    all_vals = np.concatenate([x_vals, y_vals])
    min_val = min(all_vals.min(), -0.05) * 1.1
    max_val = max(all_vals.max(), 0.05) * 1.1

    ax.plot([min_val, max_val], [min_val, max_val], ls="--", c="0.6", lw=1.2, zorder=0)

    sns.scatterplot(
        data=temp_df,
        x="x", y="y",
        hue="model",
        style="task",
        s=140,
        palette="deep",
        ax=ax,
        legend=False
    )

    texts = []
    for _, row in temp_df.iterrows():
        diff = abs(row["x"] - row["y"])
        dist_from_origin = np.sqrt(row["x"] ** 2 + row["y"] ** 2)

        if diff > 0.02 or dist_from_origin > 0.1:
            task_short = row['task'].replace("HumanEval", "HEval").replace("GSM8K", "GSM")
            model_short = "Qwen" if "Qwen" in row['model'] else "Llama"
            label = f"{model_short}-{task_short}"
            texts.append(
                ax.text(
                    row["x"], row["y"], label,
                    fontsize=FS, fontweight=FONT_W, color="0.15"
                )
            )

    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle='-', color='0.5', lw=0.8)
    )

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')

    ax.set_xlabel(r"$\Delta$ Random Index", fontsize=FS, fontweight=FONT_W)
    ax.set_ylabel(fr"$\Delta$ {guided_policy}", fontsize=FS, fontweight=FONT_W)

    # ---- move annotation left (and darker + bold) ----
    span = max_val - min_val
    pad_x = 0.30 * span   # 这里越大越往左（往里）挪
    pad_y = 0.06 * span

    ann_color = "0.50"  # 越小越黑，比如 0.15 更深
    ax.text(
        max_val - pad_x, min_val + pad_y,
        "Worse than\nRandom",
        ha='right', va='bottom',
        color=ann_color, fontsize=FS, fontweight=FONT_W
    )
    ax.text(
        min_val + pad_y, max_val - pad_y,
        "Better than\nRandom",
        ha='left', va='top',
        color=ann_color, fontsize=FS, fontweight=FONT_W
    )

    ax.tick_params(axis='both', labelsize=FS)
    sns.despine()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_calib_sweep(df_all: pd.DataFrame, preserve_energy: str, module_set: str,
                     model: str, task: str, outpath: str,
                     policies_to_plot: Optional[List[str]] = None) -> None:
    if policies_to_plot is None:
        policies_to_plot = ["grad_direction", "random_index", "smooth_abs"]

    df = df_all[
        (df_all["preserve_energy"] == preserve_energy) &
        (df_all["module_set"] == module_set) &
        (df_all["model"] == model) &
        (df_all["task"] == task)
        ].copy()

    if df.empty:
        return

    df = df.sort_values("calib")
    x = df["calib"].values

    fig = plt.figure(figsize=(6.6, 3.8))
    ax = fig.add_subplot(111)
    ax.plot(x, df["baseline"].values, marker="o", label="baseline")
    for p in policies_to_plot:
        if p in df.columns:
            ax.plot(x, df[p].values, marker="o", label=p)

    ax.set_xlabel("calib_samples")
    ax.set_ylabel(df["metric"].iloc[0])
    ax.set_title(f"Calibration sweep ({preserve_energy}, {module_set})\n{model} / {task}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_energy_ablation_delta(df_all: pd.DataFrame, module_set: str, calib: int,
                               model: str, policy: str, outpath: str) -> None:
    tasks = ["GSM8K", "HumanEval", "IFEval", "CSQA"]
    deltas = []
    for energy in ["l1", "none"]:
        df = df_all[
            (df_all["preserve_energy"] == energy) &
            (df_all["module_set"] == module_set) &
            (df_all["calib"] == calib) &
            (df_all["model"] == model) &
            (df_all["task"].isin(tasks))
            ].copy()
        df = add_deltas(df)
        df["task"] = pd.Categorical(df["task"], categories=tasks, ordered=True)
        df = df.sort_values("task")
        deltas.append(df[f"delta_{policy}"].values)

    if len(deltas) != 2 or len(deltas[0]) == 0 or len(deltas[1]) == 0:
        return

    l1_delta, none_delta = deltas
    x = np.arange(len(tasks))
    width = 0.38

    fig = plt.figure(figsize=(7.0, 3.8))
    ax = fig.add_subplot(111)
    ax.bar(x - width / 2, l1_delta, width, label="energy=l1 (Δ)")
    ax.bar(x + width / 2, none_delta, width, label="energy=none (Δ)")

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel(f"Δ {policy} (vs baseline)")
    ax.set_title(f"Energy ablation @ calib={calib}, module_set={module_set}\n{model}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_locality_tradeoff_delta(df_all: pd.DataFrame, calib: int, preserve_energy: str,
                                 model: str, policy: str,
                                 module_sets: List[str],
                                 outpath: str) -> None:
    tasks = ["GSM8K", "HumanEval", "IFEval", "CSQA"]
    x = np.arange(len(tasks))
    width = 0.8 / max(1, len(module_sets))

    fig = plt.figure(figsize=(8.6, 3.9))
    ax = fig.add_subplot(111)

    for k, ms in enumerate(module_sets):
        df = df_all[
            (df_all["preserve_energy"] == preserve_energy) &
            (df_all["module_set"] == ms) &
            (df_all["calib"] == calib) &
            (df_all["model"] == model) &
            (df_all["task"].isin(tasks))
            ].copy()
        if df.empty:
            continue
        df = add_deltas(df)
        df["task"] = pd.Categorical(df["task"], categories=tasks, ordered=True)
        df = df.sort_values("task")
        y = df[f"delta_{policy}"].values

        ax.bar(x - 0.4 + (k + 0.5) * width, y, width, label=ms)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel(f"Δ {policy} (vs baseline)")
    ax.set_title(f"Locality tradeoff (Δ) @ calib={calib}, energy={preserve_energy}\n{model}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# -----------------------------
# 4) Main Execution
# -----------------------------
def main() -> None:
    ensure_dir("figs")

    dfs = []
    dfs.append(parse_latex_rows(TABLE_RESIDUAL_L1_SWEEP, TableMeta("l1", "residual", "residual_l1_sweep")))
    dfs.append(parse_latex_rows(TABLE_RESIDUAL_NONE_SWEEP, TableMeta("none", "residual", "residual_none_sweep")))
    dfs.append(parse_latex_rows(TABLE_ATTN_INPUTS_QKV_L1_CALIB128, TableMeta("l1", "attn_inputs", "qkv_l1_128")))
    dfs.append(parse_latex_rows(TABLE_ALL_MODULES_L1_CALIB128, TableMeta("l1", "all_modules", "all_l1_128")))
    dfs.append(parse_latex_rows(TABLE_UP_GATE_L1_CALIB128, TableMeta("l1", "up_gate", "up_gate_l1_128")))

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.dropna(subset=POLICIES)
    df_all.to_csv("parsed_results.csv", index=False)

    df_default = df_all[
        (df_all["preserve_energy"] == "l1") &
        (df_all["module_set"] == "residual") &
        (df_all["calib"] == 128)
        ].copy()

    # ---- (A) Δ-heatmap ----
    for model in sorted(df_default["model"].unique()):
        out = f"figs/delta_heatmap_{model.replace('/', '_')}_residual_l1_calib128.pdf"
        plot_delta_heatmap_one_model(
            df_default, model, out,
            title="Δ Heatmap (policy × task) @ residual, energy=l1, calib=128"
        )

    # ---- (B) Risk–reward scatter (Optimized) ----
    aligned = ["GSM8K", "HumanEval", "CSQA"]
    plot_risk_reward(
        df_default,
        outpath="figs/risk_reward_residual_l1_calib128.pdf",
        aligned_tasks=aligned,
        constrained_task="IFEval"
    )

    # ---- (C) Guided vs Random scatter (Optimized) ----
    plot_guided_vs_random(
        df_default,
        guided_policy="smooth_abs",
        outpath="figs/guided_vs_random_smooth_abs_residual_l1_calib128.pdf",
        title="Guided vs Random: smooth_abs vs random_index"
    )
    plot_guided_vs_random(
        df_default,
        guided_policy="grad_direction",
        outpath="figs/guided_vs_random_grad_direction_residual_l1_calib128.pdf",
        title="")

    # ---- (D) Calibration sweep ----
    for model in sorted(df_all["model"].unique()):
        for task in ["GSM8K", "HumanEval", "IFEval", "CSQA"]:
            out = f"figs/calib_sweep_{model.replace('/', '_')}_{task}_residual_l1.pdf"
            plot_calib_sweep(
                df_all,
                preserve_energy="l1",
                module_set="residual",
                model=model,
                task=task,
                outpath=out,
                policies_to_plot=["grad_direction", "random_index", "smooth_abs", "abs_select"],
            )

    # ---- (E) Energy ablation ----
    for model in sorted(df_all["model"].unique()):
        out = f"figs/energy_ablation_delta_{model.replace('/', '_')}_grad_direction_residual_calib128.pdf"
        plot_energy_ablation_delta(
            df_all,
            module_set="residual",
            calib=128,
            model=model,
            policy="grad_direction",
            outpath=out,
        )

    # ---- (F) Locality tradeoff ----
    module_sets = ["attn_inputs", "up_gate", "residual", "all_modules"]
    for model in sorted(df_all["model"].unique()):
        out = f"figs/locality_tradeoff_delta_{model.replace('/', '_')}_grad_direction_l1_calib128.pdf"
        plot_locality_tradeoff_delta(
            df_all,
            calib=128,
            preserve_energy="l1",
            model=model,
            policy="grad_direction",
            module_sets=module_sets,
            outpath=out,
        )

    print("Done. Figures saved to ./figs and parsed_results.csv written.")


if __name__ == "__main__":
    main()