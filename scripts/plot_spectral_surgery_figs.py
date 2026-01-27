# plot_spectral_surgery_figs.py
# -*- coding: utf-8 -*-
"""
Generate all figures discussed earlier from your LaTeX tables.

Outputs:
  figs/
    delta_heatmap_<model>_residual_l1_calib128.png
    risk_reward_residual_l1_calib128.png
    guided_vs_random_smooth_abs_residual_l1_calib128.png
    guided_vs_random_grad_direction_residual_l1_calib128.png
    calib_sweep_<model>_<task>_residual_l1.png
    energy_ablation_delta_<model>_grad_direction_residual_calib128.png
    locality_tradeoff_delta_<model>_grad_direction_l1_calib128.png
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


# -----------------------------
# 1) Paste your LaTeX tables here
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
ALL_COLS = ["baseline"] + POLICIES


def _strip_latex_wrappers(s: str) -> str:
    # Remove common wrappers like \textbf{...}, \underline{...}
    # Repeat until stable to handle nested wrappers.
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
    # Take first float-like token
    m = re.search(r"-?\d+(?:\.\d+)?", cell)
    if not m:
        return None
    return float(m.group(0))


@dataclass
class TableMeta:
    preserve_energy: str  # "l1" or "none"
    module_set: str       # "residual", "attn_inputs", "all_modules", "up_gate"
    source_name: str      # for debugging


def parse_latex_rows(table_str: str, meta: TableMeta) -> pd.DataFrame:
    rows = []
    for raw in table_str.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "&" not in line:
            continue
        # skip formatting rows
        if any(tok in line for tok in ["\\toprule", "\\midrule", "\\bottomrule", "\\addlinespace", "\\cmidrule"]):
            continue

        # remove trailing LaTeX row end
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
    # sanity: drop rows with missing policy values (shouldn't happen, but keep safe)
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
# 3) Plot helpers
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
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(f"{title}\n{model}")
    ax.set_xticks(np.arange(len(POLICIES)))
    ax.set_xticklabels(POLICIES, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks)

    # annotate
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ vs baseline")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_risk_reward(df_cfg: pd.DataFrame, outpath: str,
                     aligned_tasks: List[str], constrained_task: str = "IFEval") -> None:
    """
    Each point: (model, policy)
      reward = mean Δ on aligned_tasks
      risk   = max(0, baseline - policy) on constrained_task  (i.e., drop magnitude)
    """
    df = add_deltas(df_cfg)
    points = []

    for model in sorted(df["model"].unique()):
        dfm = df[df["model"] == model]
        for p in POLICIES:
            # reward: mean delta on aligned tasks present
            dfa = dfm[dfm["task"].isin(aligned_tasks)]
            reward = float(np.nanmean(dfa[f"delta_{p}"].values))
            # risk: drop on constrained task
            dfc = dfm[dfm["task"] == constrained_task]
            if len(dfc) == 0:
                continue
            delta_c = float(dfc[f"delta_{p}"].values[0])
            risk = max(0.0, -delta_c)
            points.append((model, p, reward, risk))

    fig = plt.figure(figsize=(7.0, 4.2))
    ax = fig.add_subplot(111)
    for model, p, reward, risk in points:
        ax.scatter(reward, risk)
        ax.text(reward, risk, f"{model}\n{p}", fontsize=8, ha="left", va="bottom")

    ax.set_xlabel("Reward = mean Δ on aligned tasks")
    ax.set_ylabel("Risk = drop magnitude on constrained task (IFEval)")
    ax.set_title("Policy Risk–Reward (default config)")
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_guided_vs_random(df_cfg: pd.DataFrame, guided_policy: str, outpath: str, title: str) -> None:
    """
    x = Δ(random_index), y = Δ(guided_policy), points over (model, task).
    """
    assert guided_policy in POLICIES and guided_policy != "random_index"
    df = add_deltas(df_cfg)

    x = df["delta_random_index"].values
    y = df[f"delta_{guided_policy}"].values

    fig = plt.figure(figsize=(5.8, 5.2))
    ax = fig.add_subplot(111)
    ax.scatter(x, y)

    # diagonal
    lo = float(min(np.min(x), np.min(y), -0.5))
    hi = float(max(np.max(x), np.max(y), 0.5))
    ax.plot([lo, hi], [lo, hi], linewidth=1)

    # annotate lightly
    for _, r in df.iterrows():
        ax.text(r["delta_random_index"], r[f"delta_{guided_policy}"],
                f'{r["model"].split("-")[0]}-{r["task"]}', fontsize=8, ha="left", va="bottom")

    ax.set_xlabel("Δ random_index (vs baseline)")
    ax.set_ylabel(f"Δ {guided_policy} (vs baseline)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_calib_sweep(df_all: pd.DataFrame, preserve_energy: str, module_set: str,
                     model: str, task: str, outpath: str,
                     policies_to_plot: Optional[List[str]] = None) -> None:
    """
    Line plot: metric vs calib for baseline + selected policies.
    """
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
    """
    For each task: bar plot of Δ(policy) under energy=l1 vs energy=none (within same module_set/calib/model).
    """
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
    ax.bar(x - width/2, l1_delta, width, label="energy=l1 (Δ)")
    ax.bar(x + width/2, none_delta, width, label="energy=none (Δ)")

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
    """
    Grouped bars: per task, show Δ(policy) across module_sets.
    """
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
# 4) Main: parse + plot all
# -----------------------------
def main() -> None:
    ensure_dir("figs")

    # NOTE:
    # We tag your first big table as preserve_energy="l1" and module_set="residual"
    # (i.e., your default residual-writing modules). If you want different tags, change here.
    dfs = []
    dfs.append(parse_latex_rows(TABLE_RESIDUAL_L1_SWEEP, TableMeta("l1", "residual", "residual_l1_sweep")))
    dfs.append(parse_latex_rows(TABLE_RESIDUAL_NONE_SWEEP, TableMeta("none", "residual", "residual_none_sweep")))
    dfs.append(parse_latex_rows(TABLE_ATTN_INPUTS_QKV_L1_CALIB128, TableMeta("l1", "attn_inputs", "qkv_l1_128")))
    dfs.append(parse_latex_rows(TABLE_ALL_MODULES_L1_CALIB128, TableMeta("l1", "all_modules", "all_l1_128")))
    dfs.append(parse_latex_rows(TABLE_UP_GATE_L1_CALIB128, TableMeta("l1", "up_gate", "up_gate_l1_128")))

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.dropna(subset=["abs_select", "smooth_abs", "random_index", "grad_direction"])
    df_all.to_csv("parsed_results.csv", index=False)

    # Default config subset for “main plots”
    df_default = df_all[
        (df_all["preserve_energy"] == "l1") &
        (df_all["module_set"] == "residual") &
        (df_all["calib"] == 128)
    ].copy()

    # ---- (A) Δ-heatmap (policy × task), per model ----
    for model in sorted(df_default["model"].unique()):
        out = f"figs/delta_heatmap_{model.replace('/', '_')}_residual_l1_calib128.png"
        plot_delta_heatmap_one_model(
            df_default, model, out,
            title="Δ Heatmap (policy × task) @ residual, energy=l1, calib=128"
        )

    # ---- (B) Risk–reward scatter ----
    aligned = ["GSM8K", "HumanEval", "CSQA"]  # treat IFEval as constrained axis
    plot_risk_reward(
        df_default,
        outpath="figs/risk_reward_residual_l1_calib128.png",
        aligned_tasks=aligned,
        constrained_task="IFEval"
    )

    # ---- (C) Guided vs Random scatter (Smooth vs Random; Grad vs Random) ----
    plot_guided_vs_random(
        df_default,
        guided_policy="smooth_abs",
        outpath="figs/guided_vs_random_smooth_abs_residual_l1_calib128.png",
        title="Guided vs Random: smooth_abs vs random_index (Δ over baseline)"
    )
    plot_guided_vs_random(
        df_default,
        guided_policy="grad_direction",
        outpath="figs/guided_vs_random_grad_direction_residual_l1_calib128.png",
        title="Guided vs Random: grad_direction vs random_index (Δ over baseline)"
    )

    # ---- (D) Calibration sweep curves (all tasks × both models) under residual + l1 ----
    for model in sorted(df_all["model"].unique()):
        for task in ["GSM8K", "HumanEval", "IFEval", "CSQA"]:
            out = f"figs/calib_sweep_{model.replace('/', '_')}_{task}_residual_l1.png"
            plot_calib_sweep(
                df_all,
                preserve_energy="l1",
                module_set="residual",
                model=model,
                task=task,
                outpath=out,
                policies_to_plot=["grad_direction", "random_index", "smooth_abs", "abs_select"],
            )

    # ---- (E) Energy ablation (Δ grad_direction: l1 vs none), per model @ calib=128 residual ----
    for model in sorted(df_all["model"].unique()):
        out = f"figs/energy_ablation_delta_{model.replace('/', '_')}_grad_direction_residual_calib128.png"
        plot_energy_ablation_delta(
            df_all,
            module_set="residual",
            calib=128,
            model=model,
            policy="grad_direction",
            outpath=out,
        )

    # ---- (F) Locality tradeoff (Δ grad_direction across module sets), per model @ calib=128 l1 ----
    module_sets = ["attn_inputs", "up_gate", "residual", "all_modules"]
    for model in sorted(df_all["model"].unique()):
        out = f"figs/locality_tradeoff_delta_{model.replace('/', '_')}_grad_direction_l1_calib128.png"
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
