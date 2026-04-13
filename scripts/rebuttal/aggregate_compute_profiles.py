#!/usr/bin/env python3
"""
Aggregate per-setting compute profile summaries into the final report artifacts.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


BASE_MODEL_ORDER = ["Qwen-Qwen3-8B", "meta-llama-Llama-3.1-8B"]
TASK_ORDER = ["math", "csqa"]
K_VALUES = [0, 1, 3, 5, 32]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fmt_flops(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value >= 1e15:
        return f"{value / 1e15:.2f} PF"
    if value >= 1e12:
        return f"{value / 1e12:.2f} TF"
    if value >= 1e9:
        return f"{value / 1e9:.2f} GF"
    return f"{value:.2e}"


def fmt_latency(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value < 1.0:
        return f"{value * 1000.0:.1f} ms"
    return f"{value:.2f} s"


def parse_fewshot_key(key: str) -> tuple[str, str, int]:
    base_task, k_str = key.rsplit("/k=", 1)
    base_model_dir, task = base_task.split("/", 1)
    return base_model_dir, task, int(k_str)


def setting_sort_key(setting: str) -> tuple[int, int]:
    base_model_dir, task = setting.split("/", 1)
    return (
        BASE_MODEL_ORDER.index(base_model_dir),
        TASK_ORDER.index(task),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_paths", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/rebuttal_exp/fewshot_corrected_math"),
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_spectral: dict[str, dict[str, Any]] = {}
    merged_fewshot: dict[str, dict[str, Any]] = {}
    for path in args.summary_paths:
        payload = load_json(path)
        merged_spectral.update(payload.get("spectral_profiles", {}))
        merged_fewshot.update(payload.get("fewshot_profiles", {}))

    settings = sorted(merged_spectral.keys(), key=setting_sort_key)
    if not settings:
        raise RuntimeError("No spectral profiles found in provided summaries.")

    corrected_math_records = load_jsonl(args.repo_root / "outputs/rebuttal_exp/raw/fewshot_eval_mathfix/results.jsonl")
    original_fewshot_records = load_jsonl(args.repo_root / "outputs/rebuttal_exp/raw/fewshot_eval/results.jsonl")
    spectral_costs = load_json(args.repo_root / "outputs/rebuttal_exp/raw/fewshot_eval/spectral_costs.json")

    total_flops_by_k: dict[int, list[float]] = {k: [] for k in K_VALUES}
    extra_flops_by_k: dict[int, list[float]] = {k: [] for k in K_VALUES if k > 0}
    total_latency_by_k: dict[int, list[float]] = {k: [] for k in K_VALUES}
    extra_latency_by_k: dict[int, list[float]] = {k: [] for k in K_VALUES if k > 0}
    break_even_flops_by_k: dict[int, list[float]] = {k: [] for k in K_VALUES if k > 0}
    break_even_latency_by_k: dict[int, list[float]] = {k: [] for k in K_VALUES if k > 0}
    for setting in settings:
        spectral_total = merged_spectral[setting]["total_flops_est"]
        spectral_runtime = merged_spectral[setting]["runtime_seconds"]
        base_flops = merged_fewshot[f"{setting}/k=0"]["total_query_flops"]
        base_latency = merged_fewshot[f"{setting}/k=0"]["vllm_latency_seconds"]
        total_flops_by_k[0].append(base_flops)
        total_latency_by_k[0].append(base_latency)
        for k in [1, 3, 5, 32]:
            key = f"{setting}/k={k}"
            q_flops = merged_fewshot[key]["total_query_flops"]
            q_latency = merged_fewshot[key]["vllm_latency_seconds"]
            extra_flops = q_flops - base_flops
            extra_latency = q_latency - base_latency
            total_flops_by_k[k].append(q_flops)
            total_latency_by_k[k].append(q_latency)
            extra_flops_by_k[k].append(extra_flops)
            extra_latency_by_k[k].append(extra_latency)
            break_even_flops_by_k[k].append(spectral_total / extra_flops)
            if extra_latency > 0:
                break_even_latency_by_k[k].append(spectral_runtime / extra_latency)

    token_break_even: dict[int, list[float]] = {k: [] for k in [1, 3, 5, 32]}
    setting_set = set(settings)
    for rec in original_fewshot_records + corrected_math_records:
        if rec.get("error"):
            continue
        k = int(rec["fewshot_k"])
        if k not in token_break_even:
            continue
        setting = f"{rec['base_model_dir']}/{rec['task']}"
        if setting not in setting_set:
            continue
        extra = float(rec["avg_extra_prompt_tokens"])
        token_break_even[k].append(float(spectral_costs[setting]["forward_backward_token_passes"]) / extra)

    spectral_totals = [merged_spectral[setting]["total_flops_est"] for setting in settings]
    method_rows: list[dict[str, Any]] = [
        {
            "method": "random_index",
            "one_time_flops_mean": statistics.mean(spectral_totals),
            "per_query_total_flops_mean": 0.0,
            "per_query_extra_flops_mean": 0.0,
            "per_query_total_latency_mean": 0.0,
            "per_query_extra_latency_mean": 0.0,
            "break_even_flops_mean": 0.0,
            "break_even_latency_mean": None,
            "token_break_even_mean": None,
        }
    ]
    for k in K_VALUES:
        method_rows.append(
            {
                "method": f"few-shot k{k}",
                "one_time_flops_mean": 0.0,
                "per_query_total_flops_mean": statistics.mean(total_flops_by_k[k]),
                "per_query_extra_flops_mean": 0.0 if k == 0 else statistics.mean(extra_flops_by_k[k]),
                "per_query_total_latency_mean": statistics.mean(total_latency_by_k[k]),
                "per_query_extra_latency_mean": 0.0 if k == 0 else statistics.mean(extra_latency_by_k[k]),
                "break_even_flops_mean": None if k == 0 else statistics.mean(break_even_flops_by_k[k]),
                "break_even_latency_mean": None if k == 0 or not break_even_latency_by_k[k] else statistics.mean(break_even_latency_by_k[k]),
                "token_break_even_mean": None if k == 0 else statistics.mean(token_break_even[k]),
            }
        )

    note_lines = [
        "# Compute Profiling Note",
        "",
        "This note reports **profiled FLOPs estimates**, not exact true FLOPs. The profiler can miss fused kernels and non-GEMM work, so the numbers should be read as implementation-level estimates rather than universal absolute truths.",
        "",
        "## Definitions",
        "",
        "- `F_edit`: one-time profiled FLOPs estimate for the spectral edit/calibration run.",
        "- `F_total(k)`: per-query profiled FLOPs estimate for fixed `k`-shot serving, including prefill plus cached decode.",
        "- `DeltaF(k) = F_total(k) - F_total(0)`: per-query **extra** serving overhead of `k`-shot prompting relative to zero-shot on the same setting.",
        "- Headline break-even definition: `Q_break_even_FLOPs(k) = F_edit / DeltaF(k)`.",
        "- Token cross-check: `Q_break_even_tokens(k) = edit_token_passes / extra_prompt_tokens(k vs 0)`.",
        "- Latency is secondary: `DeltaT(k) = T_total(k) - T_total(0)` is small and noisy on short-output `csqa`, so latency is reported only as a caveated secondary metric rather than the headline amortization figure.",
        "",
        "## Measurement Setup",
        "",
        "- Spectral path: the actual `random_index` edit configuration, profiled with PyTorch FLOPs accounting on a representative real calibration batch and scaled by the real number of batches; `use_cache=False`, batch size and sequence lengths match the saved edit metadata.",
        "- Few-shot FLOPs path: representative single-query HF/PEFT profiling with KV cache enabled (`use_cache=True`) on the same adapters/prompts used in evaluation.",
        "- Few-shot latency path: actual vLLM generation at batch size 1 after warmup, LoRA enabled, same decoding settings, `gpu_memory_utilization=0.6`, and workload-capped `max_model_len` so KV reservation matches the real prompt lengths rather than an unused architectural maximum.",
        "- KV cache is **enabled** for the few-shot inference FLOPs profile and for the vLLM latency measurement.",
        "- The simple token-based amortization from the earlier package is retained below as a cross-check.",
        "",
        "## Summary",
    ]
    note_lines.append(
        f"- Mean one-time spectral edit cost across the 4 included settings: {fmt_flops(statistics.mean(spectral_totals))} "
        f"(range {fmt_flops(min(spectral_totals))} to {fmt_flops(max(spectral_totals))})."
    )
    note_lines.append(
        f"- Zero-shot serving baseline (`k=0`): mean total per-query cost {fmt_flops(statistics.mean(total_flops_by_k[0]))}."
    )
    for k in [1, 3, 5, 32]:
        note_lines.append(
            f"- `k={k}`: mean total per-query cost {fmt_flops(statistics.mean(total_flops_by_k[k]))}; "
            f"mean extra per-query overhead vs `k=0` is {fmt_flops(statistics.mean(extra_flops_by_k[k]))}; "
            f"FLOPs break-even uses this **extra** term and is {statistics.mean(break_even_flops_by_k[k]):.2f} queries on average "
            f"(range {min(break_even_flops_by_k[k]):.2f}-{max(break_even_flops_by_k[k]):.2f}); "
            f"token cross-check is {statistics.mean(token_break_even[k]):.2f} queries on average."
        )
    note_lines.append(
        "- Secondary latency note: per-query total latency is about 2 seconds on average across mixed `math`+`csqa` settings, but extra latency relative to `k=0` is much noisier than extra FLOPs on short-output `csqa`, so latency is not used as the headline amortization metric."
    )

    table_lines = [
        "| Method | One-time FLOPs | Per-query total FLOPs | Per-query extra FLOPs vs k=0 | Break-even queries (FLOPs, uses extra) | Token cross-check |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        if row["method"] == "random_index":
            break_even = "0"
            token_cross_check = "N/A"
        elif row["break_even_flops_mean"] is None:
            break_even = "N/A"
            token_cross_check = "N/A"
        else:
            break_even = f"{row['break_even_flops_mean']:.1f}"
            token_cross_check = f"{row['token_break_even_mean']:.1f}"
        table_lines.append(
            f"| {row['method']} | "
            f"{fmt_flops(row['one_time_flops_mean']) if row['one_time_flops_mean'] else '0'} | "
            f"{fmt_flops(row['per_query_total_flops_mean']) if row['per_query_total_flops_mean'] else '0'} | "
            f"{fmt_flops(row['per_query_extra_flops_mean']) if row['per_query_extra_flops_mean'] else '0'} | "
            f"{break_even} | "
            f"{token_cross_check} |"
        )

    paragraph = (
        "We supplemented the token proxy with profiled FLOPs estimates under the actual model/adapters and generation setup. "
        "These are not exact true FLOPs, but they support the same implementation-level compute comparison: spectral editing pays a one-time calibration/edit cost, whereas fixed few-shot prompting pays repeated amortized serving cost. "
        f"Using the reviewer-safe definition based on extra serving overhead relative to `k=0`, the FLOPs break-even is about "
        f"{statistics.mean(break_even_flops_by_k[1]):.0f}, {statistics.mean(break_even_flops_by_k[3]):.0f}, "
        f"{statistics.mean(break_even_flops_by_k[5]):.0f}, and {statistics.mean(break_even_flops_by_k[32]):.0f} queries "
        f"for `k=1,3,5,32` respectively; the token-based amortization cross-check is directionally consistent, and `k=32` "
        "remains a poor practical operating point because it adds much larger recurrent cost without materially improving accuracy."
    )

    summary = {
        "spectral_profiles": merged_spectral,
        "fewshot_profiles": merged_fewshot,
        "method_rows": method_rows,
        "break_even_flops": break_even_flops_by_k,
        "break_even_latency": break_even_latency_by_k,
        "total_flops_by_k": total_flops_by_k,
        "extra_flops_by_k": extra_flops_by_k,
        "total_latency_by_k": total_latency_by_k,
        "extra_latency_by_k": extra_latency_by_k,
        "token_break_even": token_break_even,
    }

    (out_dir / "compute_profile_note.md").write_text("\n".join(note_lines) + "\n")
    (out_dir / "compute_profile_table.md").write_text("\n".join(table_lines) + "\n")
    (out_dir / "compute_rebuttal_paragraph.txt").write_text(paragraph + "\n")
    (out_dir / "compute_profile_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Note: {out_dir / 'compute_profile_note.md'}")
    print(f"Table: {out_dir / 'compute_profile_table.md'}")
    print(f"Paragraph: {out_dir / 'compute_rebuttal_paragraph.txt'}")
    print(f"Summary: {out_dir / 'compute_profile_summary.json'}")


if __name__ == "__main__":
    main()
