#!/usr/bin/env python3
"""
Generate the rebuttal-ready few-shot vs spectral-surgery comparison package.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def fmt_pp(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2f} pp"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * value:+.1f}%"


def fmt_tokens(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    value_f = float(value)
    if abs(value_f) >= 1_000_000:
        return f"{value_f / 1_000_000:.2f}M"
    if abs(value_f) >= 1_000:
        return f"{value_f / 1_000:.1f}k"
    return f"{value_f:.0f}"


def fmt_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value >= 60.0:
        return f"{value / 60.0:.1f} min"
    return f"{value:.1f} s"


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.mean(values)


def build_method_summary(
    *,
    method_name: str,
    settings: list[str],
    metrics_by_setting: dict[str, float],
    baseline_by_setting: dict[str, float],
    one_time_costs_by_setting: dict[str, float | None],
    per_query_costs_by_setting: dict[str, float | None],
    runtimes_by_setting: dict[str, float | None] | None = None,
    extra_notes: str = "",
) -> dict[str, Any]:
    metrics = [metrics_by_setting[s] for s in settings if s in metrics_by_setting]
    deltas_pp = [
        100.0 * (metrics_by_setting[s] - baseline_by_setting[s])
        for s in settings
        if s in metrics_by_setting and s in baseline_by_setting
    ]
    one_time = [v for s, v in one_time_costs_by_setting.items() if s in settings and v is not None]
    per_query = [v for s, v in per_query_costs_by_setting.items() if s in settings and v is not None]
    runtimes = []
    if runtimes_by_setting:
        runtimes = [v for s, v in runtimes_by_setting.items() if s in settings and v is not None]
    return {
        "method": method_name,
        "settings_covered": len([s for s in settings if s in metrics_by_setting]),
        "mean_metric": mean_or_none(metrics),
        "mean_delta_pp": mean_or_none(deltas_pp),
        "mean_one_time_cost": mean_or_none(one_time),
        "mean_per_query_cost": mean_or_none(per_query),
        "mean_runtime_seconds": mean_or_none(runtimes),
        "notes": extra_notes,
    }


def draw_performance_vs_compute_plot(
    plot_rows: list[dict[str, Any]],
    *,
    out_path: Path,
    deployment_queries: int,
) -> None:
    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    colors: list[str] = []
    for row in plot_rows:
        method = row["method"]
        if row["mean_delta_pp"] is None:
            continue
        if method.startswith("fewshot_"):
            x = (row["mean_per_query_cost"] or 0.0) * deployment_queries
            color = "#D55E00"
            label = method.replace("fewshot_k", "k=")
        else:
            x = row["mean_one_time_cost"] or 0.0
            color = "#1B9E77"
            label = method
        if x <= 0:
            continue
        xs.append(x)
        ys.append(row["mean_delta_pp"])
        labels.append(label)
        colors.append(color)

    if not xs:
        return

    width = 960
    height = 600
    left = 105
    right = 40
    top = 70
    bottom = 95
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_min = 10 ** math.floor(math.log10(min(xs)))
    x_max = 10 ** math.ceil(math.log10(max(xs)))
    if x_min == x_max:
        x_max = x_min * 10

    y_min_raw = min(ys)
    y_max_raw = max(ys)
    y_min = math.floor((y_min_raw - 2.0) / 5.0) * 5
    y_max = math.ceil((y_max_raw + 2.0) / 5.0) * 5
    if y_min == y_max:
        y_max = y_min + 5

    def x_to_px(value: float) -> float:
        frac = (math.log10(value) - math.log10(x_min)) / (math.log10(x_max) - math.log10(x_min))
        return left + frac * plot_w

    def y_to_px(value: float) -> float:
        frac = (value - y_min) / (y_max - y_min)
        return top + (1.0 - frac) * plot_h

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.rectangle([left, top, left + plot_w, top + plot_h], outline="#333333", width=1)

    x_ticks: list[float] = []
    tick = x_min
    while tick <= x_max * 1.0001:
        x_ticks.append(tick)
        tick *= 10
    for tick in x_ticks:
        x = x_to_px(tick)
        draw.line([(x, top), (x, top + plot_h)], fill="#E5E5E5", width=1)
        label = f"{int(tick):,}"
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, top + plot_h + 8), label, fill="#333333", font=font)

    y_ticks = list(range(y_min, y_max + 1, 10 if (y_max - y_min) > 40 else 5))
    for tick in y_ticks:
        y = y_to_px(float(tick))
        draw.line([(left, y), (left + plot_w, y)], fill="#EAEAEA", width=1)
        label = f"{tick:+d}"
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((left - 10 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2), label, fill="#333333", font=font)

    if y_min <= 0 <= y_max:
        y0 = y_to_px(0.0)
        draw.line([(left, y0), (left + plot_w, y0)], fill="#888888", width=2)

    for x, y, label, color in zip(xs, ys, labels, colors):
        px = x_to_px(x)
        py = y_to_px(y)
        draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=color, outline="#222222", width=1)
        draw.text((px + 10, py - 9), label, fill="#222222", font=font)

    title = "Spectral Surgery vs Fixed Few-Shot Prompting"
    subtitle = f"Compute proxy at {deployment_queries:,} served queries (log-scale x-axis)"
    x_label = "Token compute proxy"
    y_label = "Mean delta vs zero-shot baseline (pp)"

    title_bbox = draw.textbbox((0, 0), title, font=font)
    draw.text(((width - (title_bbox[2] - title_bbox[0])) / 2, 18), title, fill="#111111", font=font)
    subtitle_bbox = draw.textbbox((0, 0), subtitle, font=font)
    draw.text(((width - (subtitle_bbox[2] - subtitle_bbox[0])) / 2, 36), subtitle, fill="#444444", font=font)

    x_bbox = draw.textbbox((0, 0), x_label, font=font)
    draw.text(((width - (x_bbox[2] - x_bbox[0])) / 2, height - 34), x_label, fill="#111111", font=font)
    draw.text((18, top + plot_h / 2), y_label, fill="#111111", font=font)

    legend_items = [("spectral", "#1B9E77"), ("few-shot", "#D55E00")]
    legend_x = width - 180
    legend_y = 18
    for idx, (label, color) in enumerate(legend_items):
        y = legend_y + idx * 18
        draw.rectangle([legend_x, y + 2, legend_x + 12, y + 14], fill=color, outline="#222222", width=1)
        draw.text((legend_x + 18, y), label, fill="#222222", font=font)

    image.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate few-shot comparison report.")
    parser.add_argument("--fewshot_root", type=str, default="outputs/rebuttal_exp/raw/fewshot_eval")
    parser.add_argument("--fewshot_results_jsonl", type=str, default=None)
    parser.add_argument("--correction_results_jsonl", type=str, default=None)
    parser.add_argument("--core_eval_jsonl", type=str, default="outputs/rebuttal_exp/raw/multiseed_eval/eval_results.jsonl")
    parser.add_argument("--out_dir", type=str, default="outputs/rebuttal_exp/fewshot")
    parser.add_argument("--deployment_queries", type=int, default=1000)
    args = parser.parse_args()

    fewshot_root = Path(args.fewshot_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_selection = load_json(fewshot_root / "task_selection.json")
    context_summary = load_json(fewshot_root / "context_summary.json")
    spectral_costs = load_json(fewshot_root / "spectral_costs.json")
    fewshot_results_path = Path(args.fewshot_results_jsonl) if args.fewshot_results_jsonl else (fewshot_root / "results.jsonl")
    fewshot_records = load_jsonl(fewshot_results_path)
    correction_records = load_jsonl(Path(args.correction_results_jsonl)) if args.correction_results_jsonl else []
    core_records = load_jsonl(Path(args.core_eval_jsonl))

    included_tasks = task_selection["included_tasks"]
    excluded_tasks = dict(task_selection["excluded_tasks"])
    excluded_tasks.setdefault(
        "alpaca",
        "Excluded: the current alpaca adapter is evaluated on IFEval, and prepending fixed few-shot exemplars would change the benchmark prompt format in a non-standard way.",
    )

    core_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rec in core_records:
        if int(rec.get("repeat_seed", -1)) != 42:
            continue
        if rec.get("task") not in included_tasks:
            continue
        method = rec.get("method")
        if method not in {"baseline", "random_index", "smooth_abs"}:
            continue
        core_lookup[(rec["base_model_dir"], rec["task"], method)] = rec

    fewshot_lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    for rec in fewshot_records:
        if rec.get("error"):
            continue
        fewshot_lookup[(rec["base_model_dir"], rec["task"], int(rec["fewshot_k"]))] = rec
    correction_lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    for rec in correction_records:
        if rec.get("error"):
            continue
        correction_lookup[(rec["base_model_dir"], rec["task"], int(rec["fewshot_k"]))] = rec
    fewshot_lookup.update(correction_lookup)

    settings = sorted({f"{model}/{task}" for model, task, _ in core_lookup.keys()})
    baseline_by_setting = {
        f"{model}/{task}": float(rec["metric_value"])
        for (model, task, method), rec in core_lookup.items()
        if method == "baseline"
    }
    baseline_by_setting.update(
        {
            f"{model}/{task}": float(rec["metric_value"])
            for (model, task, k), rec in correction_lookup.items()
            if k == 0
        }
    )
    random_by_setting = {
        f"{model}/{task}": float(rec["metric_value"])
        for (model, task, method), rec in core_lookup.items()
        if method == "random_index"
    }
    smooth_by_setting = {
        f"{model}/{task}": float(rec["metric_value"])
        for (model, task, method), rec in core_lookup.items()
        if method == "smooth_abs"
    }

    k_values = sorted({int(rec["fewshot_k"]) for rec in fewshot_records})
    fewshot_by_k: dict[int, dict[str, Any]] = {}
    for k in k_values:
        fewshot_by_k[k] = {
            f"{model}/{task}": rec
            for (model, task, kk), rec in fewshot_lookup.items()
            if kk == k
        }

    spectral_one_time_costs = {
        key: float(payload["forward_backward_token_passes"])
        for key, payload in spectral_costs.items()
    }

    method_rows: list[dict[str, Any]] = []
    method_rows.append(
        build_method_summary(
            method_name="random_index",
            settings=settings,
            metrics_by_setting=random_by_setting,
            baseline_by_setting=baseline_by_setting,
            one_time_costs_by_setting=spectral_one_time_costs,
            per_query_costs_by_setting={k: 0.0 for k in spectral_one_time_costs},
            runtimes_by_setting={},
            extra_notes="Primary post-hoc spectral baseline; zero serving overhead after editing.",
        )
    )
    method_rows.append(
        build_method_summary(
            method_name="smooth_abs",
            settings=settings,
            metrics_by_setting=smooth_by_setting,
            baseline_by_setting=baseline_by_setting,
            one_time_costs_by_setting=spectral_one_time_costs,
            per_query_costs_by_setting={k: 0.0 for k in spectral_one_time_costs},
            runtimes_by_setting={},
            extra_notes="Secondary conservative spectral baseline.",
        )
    )
    method_rows.append(
        build_method_summary(
            method_name="fewshot_k0",
            settings=settings,
            metrics_by_setting=baseline_by_setting,
            baseline_by_setting=baseline_by_setting,
            one_time_costs_by_setting={s: 0.0 for s in settings},
            per_query_costs_by_setting={s: 0.0 for s in settings},
            runtimes_by_setting={},
            extra_notes="Zero-shot baseline with the unedited adapter.",
        )
    )
    for k in [1, 3, 5, 32]:
        metrics_by_setting = {
            setting: float(rec["metric_value"])
            for setting, rec in fewshot_by_k.get(k, {}).items()
        }
        per_query_costs = {
            setting: float(rec["avg_extra_prompt_tokens"])
            for setting, rec in fewshot_by_k.get(k, {}).items()
            if rec.get("avg_extra_prompt_tokens") is not None
        }
        runtimes = {
            setting: float(rec["runtime_seconds"])
            for setting, rec in fewshot_by_k.get(k, {}).items()
            if rec.get("runtime_seconds") is not None
        }
        method_rows.append(
            build_method_summary(
                method_name=f"fewshot_k{k}",
                settings=settings,
                metrics_by_setting=metrics_by_setting,
                baseline_by_setting=baseline_by_setting,
                one_time_costs_by_setting={s: 0.0 for s in settings},
                per_query_costs_by_setting=per_query_costs,
                runtimes_by_setting=runtimes,
                extra_notes="Large-context stress test." if k == 32 else "",
            )
        )

    break_even_rows: list[dict[str, Any]] = []
    for k in [1, 3, 5, 32]:
        values = []
        for setting, spectral_cost in spectral_one_time_costs.items():
            rec = fewshot_by_k.get(k, {}).get(setting)
            if not rec:
                continue
            extra = rec.get("avg_extra_prompt_tokens")
            if extra is None or float(extra) <= 0:
                continue
            values.append(spectral_cost / float(extra))
        break_even_rows.append(
            {
                "method": f"fewshot_k{k}",
                "mean_break_even_queries": mean_or_none(values),
                "min_break_even_queries": min(values) if values else None,
                "max_break_even_queries": max(values) if values else None,
            }
        )

    per_setting_rows: list[dict[str, Any]] = []
    for setting in settings:
        model, task = setting.split("/", 1)
        row = {
            "setting": setting,
            "baseline": baseline_by_setting.get(setting),
            "random_index": random_by_setting.get(setting),
            "smooth_abs": smooth_by_setting.get(setting),
        }
        for k in [1, 3, 5, 32]:
            rec = fewshot_by_k.get(k, {}).get(setting)
            row[f"fewshot_k{k}"] = float(rec["metric_value"]) if rec else None
        per_setting_rows.append(row)

    method_lookup = {row["method"]: row for row in method_rows}
    corrected_settings = sorted({f"{model}/{task}" for (model, task, _k) in correction_lookup.keys()})
    corrected_math_settings = [setting for setting in corrected_settings if setting.endswith("/math")]
    csqa_settings = [setting for setting in settings if setting.endswith("/csqa")]
    math_settings = [setting for setting in settings if setting.endswith("/math")]
    no_fewshot_beats_baseline = True
    for k in [1, 3, 5, 32]:
        for setting, rec in fewshot_by_k.get(k, {}).items():
            if float(rec["metric_value"]) > baseline_by_setting[setting]:
                no_fewshot_beats_baseline = False
                break
        if not no_fewshot_beats_baseline:
            break

    spectral_runtime_note = (
        "Wall-clock edit runtime was not stored in the existing spectral metadata, so the amortization analysis "
        "uses token-based proxies plus the recorded few-shot eval runtimes."
    )
    random_row = method_lookup["random_index"]
    smooth_row = method_lookup["smooth_abs"]
    fewshot_k1_row = method_lookup["fewshot_k1"]
    fewshot_k3_row = method_lookup["fewshot_k3"]
    fewshot_k5_row = method_lookup["fewshot_k5"]
    fewshot_k32_row = method_lookup["fewshot_k32"]

    report_lines = [
        "# Few-Shot vs Spectral Surgery\n",
        "## What Was Compared\n",
        "We reused the completed seed42 rebuttal adapters and compared zero-shot spectral editing "
        "(primary: `random_index`, secondary: `smooth_abs`) against fixed-k few-shot prompting on the "
        "same unedited adapter. The few-shot runs change only the prompt: fixed train-side exemplars are "
        "prepended for k in {0, 1, 3, 5, 32}, with the same decoding settings and the same task evaluators.\n",
    ]
    if corrected_math_settings:
        report_lines.extend(
            [
                "",
                "For `math`, the few-shot numbers in this package come from a corrected rerun after fixing the "
                "earlier continuation/extraction confound. `csqa` is unchanged and reused from the original fixed-k study.\n",
            ]
        )
    report_lines.extend(
        [
        "## Included / Excluded Tasks\n",
        f"- Included: {', '.join(included_tasks)}.",
        ]
    )
    for task, reason in excluded_tasks.items():
        report_lines.append(f"- Excluded `{task}`: {reason}")
    report_lines.append("")
    report_lines.append("## Context Feasibility\n")
    for setting in settings:
        ctx = context_summary["settings"][setting]
        report_lines.append(f"### {setting}")
        for k in [0, 1, 3, 5, 32]:
            info = ctx["per_k"].get(str(k))
            if not info:
                continue
            note = " stress-test" if info["large_context_stress_test"] else ""
            report_lines.append(
                f"- k={k}: avg prompt {fmt_tokens(info['avg_prompt_tokens'])} tok, "
                f"max {fmt_tokens(info['max_prompt_tokens'])} tok, "
                f"limit {fmt_tokens(info['context_limit_tokens'])} tok, "
                f"feasible={info['feasible']}.{note}"
            )
    report_lines.append("")

    report_lines.append("## Performance Summary\n")
    if math_settings:
        report_lines.append("### Corrected Math Results")
        report_lines.append("| Setting | Baseline | random_index | smooth_abs | k=1 | k=3 | k=5 | k=32 |")
        report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in per_setting_rows:
            if row["setting"] not in math_settings:
                continue
            report_lines.append(
                "| {setting} | {baseline:.4f} | {random_index:.4f} | {smooth_abs:.4f} | {k1} | {k3} | {k5} | {k32} |".format(
                    setting=row["setting"],
                    baseline=row["baseline"],
                    random_index=row["random_index"],
                    smooth_abs=row["smooth_abs"],
                    k1="N/A" if row["fewshot_k1"] is None else f"{row['fewshot_k1']:.4f}",
                    k3="N/A" if row["fewshot_k3"] is None else f"{row['fewshot_k3']:.4f}",
                    k5="N/A" if row["fewshot_k5"] is None else f"{row['fewshot_k5']:.4f}",
                    k32="N/A" if row["fewshot_k32"] is None else f"{row['fewshot_k32']:.4f}",
                )
            )
        report_lines.append("")
    if csqa_settings:
        report_lines.append("### Existing CSQA Results")
        report_lines.append("| Setting | Baseline | random_index | smooth_abs | k=1 | k=3 | k=5 | k=32 |")
        report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in per_setting_rows:
            if row["setting"] not in csqa_settings:
                continue
            report_lines.append(
                "| {setting} | {baseline:.4f} | {random_index:.4f} | {smooth_abs:.4f} | {k1} | {k3} | {k5} | {k32} |".format(
                    setting=row["setting"],
                    baseline=row["baseline"],
                    random_index=row["random_index"],
                    smooth_abs=row["smooth_abs"],
                    k1="N/A" if row["fewshot_k1"] is None else f"{row['fewshot_k1']:.4f}",
                    k3="N/A" if row["fewshot_k3"] is None else f"{row['fewshot_k3']:.4f}",
                    k5="N/A" if row["fewshot_k5"] is None else f"{row['fewshot_k5']:.4f}",
                    k32="N/A" if row["fewshot_k32"] is None else f"{row['fewshot_k32']:.4f}",
                )
            )
    report_lines.append("")
    report_lines.append(
        f"- `random_index` stayed near the original zero-shot adapters: mean score {fmt_metric(random_row['mean_metric'])}, "
        f"{fmt_pp(random_row['mean_delta_pp'])} vs zero-shot across {random_row['settings_covered']}/{len(settings)} included settings."
    )
    report_lines.append(
        f"- `smooth_abs` was similarly conservative: mean score {fmt_metric(smooth_row['mean_metric'])}, "
        f"{fmt_pp(smooth_row['mean_delta_pp'])} vs zero-shot."
    )
    report_lines.append(
        f"- Combined across corrected `math` plus existing `csqa`, fixed-k few-shot mean deltas were "
        f"{fmt_pp(fewshot_k1_row['mean_delta_pp'])} at k=1, {fmt_pp(fewshot_k3_row['mean_delta_pp'])} at k=3, "
        f"{fmt_pp(fewshot_k5_row['mean_delta_pp'])} at k=5, and {fmt_pp(fewshot_k32_row['mean_delta_pp'])} at k=32."
    )
    if no_fewshot_beats_baseline:
        report_lines.append(
            "- Under this fixed-exemplar protocol, no few-shot operating point beat the existing zero-shot adapter on any included setting."
        )
    if corrected_math_settings:
        report_lines.append(
            "- The earlier math few-shot numbers are not used here; they were confounded by continuation/extraction behavior and are replaced by the corrected rerun."
        )
    report_lines.append("")

    report_lines.append("## Compute Summary\n")
    for setting in settings:
        spectral = spectral_costs[setting]
        report_lines.append(
            f"- {setting}: spectral edit uses Ncal={spectral['calib_samples_used']}, "
            f"{fmt_tokens(spectral['calibration_total_tokens'])} calibration tokens, "
            f"{spectral['forward_passes']} forward + {spectral['backward_passes']} backward passes, "
            f"{spectral['edited_scalars']} edited scalars, and zero extra prompt tokens at inference."
        )
    report_lines.append("")
    report_lines.append(f"- {spectral_runtime_note}")
    for k in [1, 3, 5, 32]:
        vals = [
            float(rec["avg_extra_prompt_tokens"])
            for rec in fewshot_by_k.get(k, {}).values()
            if rec.get("avg_extra_prompt_tokens") is not None
        ]
        totals = [
            float(rec["total_extra_prompt_tokens"])
            for rec in fewshot_by_k.get(k, {}).values()
            if rec.get("total_extra_prompt_tokens") is not None
        ]
        runtimes = [
            float(rec["runtime_seconds"])
            for rec in fewshot_by_k.get(k, {}).values()
            if rec.get("runtime_seconds") is not None
        ]
        if vals:
            report_lines.append(
                f"- few-shot k={k}: mean extra prompt cost is {fmt_tokens(mean_or_none(vals))} tok/query, "
                f"or {fmt_tokens(mean_or_none(totals))} extra tokens over one full eval split; "
                f"mean eval runtime was {fmt_seconds(mean_or_none(runtimes))}."
            )
    report_lines.append("")

    report_lines.append("## Amortized Comparison\n")
    report_lines.append(
        "Break-even uses a simple forward+backward token-pass proxy for spectral editing: "
        "one-time spectral cost / extra prompt tokens per served query."
    )
    report_lines.append("")
    report_lines.append("| Method | Mean break-even queries | Range across settings |")
    report_lines.append("|---|---:|---:|")
    for row in break_even_rows:
        report_lines.append(
            f"| {row['method']} | {fmt_tokens(row['mean_break_even_queries'])} | "
            f"{fmt_tokens(row['min_break_even_queries'])} - {fmt_tokens(row['max_break_even_queries'])} |"
        )
    report_lines.append("")
    report_lines.append(
        f"- Relative to repeated few-shot prompting, the mean token-proxy break-even is about "
        f"{fmt_tokens(break_even_rows[0]['mean_break_even_queries'])} served queries for k=1, "
        f"{fmt_tokens(break_even_rows[1]['mean_break_even_queries'])} for k=3, "
        f"{fmt_tokens(break_even_rows[2]['mean_break_even_queries'])} for k=5, and "
        f"{fmt_tokens(break_even_rows[3]['mean_break_even_queries'])} for k=32."
    )
    report_lines.append("")

    report_lines.append("## Conservative Conclusion\n")
    report_lines.append(
        "This comparison should be read as an operating-point tradeoff, not as a universal replacement claim. "
        "Using the corrected `math` rerun and the original `csqa` results, few-shot prompting is a mixed baseline: "
        "it can be competitive on some settings, but it always pays a recurring prompt overhead, while the spectral "
        "edits keep zero extra context at inference. The practical takeaway is therefore about amortization: when a "
        "conservative post-hoc edit is already performance-competitive, its one-time cost can be recovered after tens "
        "to a few hundred served queries rather than only at very large scale."
    )
    report_lines.append("")

    table_lines = [
        "| Method | Performance | One-time cost | Per-query cost | Best use case |",
        "|---|---|---|---|---|",
    ]

    best_use_case = {
        "random_index": "Repeated deployment after a tiny calibration pass.",
        "smooth_abs": "Same deployment regime when a smoother conservative edit is preferred.",
        "fewshot_k0": "No edit budget and no extra prompt budget.",
        "fewshot_k1": "Small query volumes where one exemplar is acceptable.",
        "fewshot_k3": "Moderate prompt budget for a stronger fixed-k baseline.",
        "fewshot_k5": "Higher prompt budget when extra context is acceptable.",
        "fewshot_k32": "Large-context stress test only; not a default serving point.",
    }

    for row in method_rows:
        method = row["method"]
        coverage = row["settings_covered"]
        coverage_note = f"{coverage}/{len(settings)} settings"
        perf = f"{fmt_metric(row['mean_metric'])} mean score, {fmt_pp(row['mean_delta_pp'])} vs zero-shot ({coverage_note})"
        if method.startswith("fewshot_"):
            one_time = "none"
            runtime = row["mean_runtime_seconds"]
            if row["mean_per_query_cost"] == 0:
                per_query = "0 tok/query"
            else:
                per_query = f"+{fmt_tokens(row['mean_per_query_cost'])} tok/query; {fmt_seconds(runtime)} / eval"
        else:
            one_time = f"{fmt_tokens(row['mean_one_time_cost'])} token-passes once"
            per_query = "0 extra prompt toks"
        table_lines.append(
            f"| {method.replace('fewshot_', 'few-shot ')} | {perf} | {one_time} | {per_query} | "
            f"{best_use_case[method]} |"
        )

    paragraph = (
        "We compared post-hoc spectral surgery to fixed few-shot prompting as two operating points on the same trained "
        "adapter. After correcting the earlier math continuation/extraction confound, the few-shot baseline remains a "
        "mixed operating point: it can be closer to the unedited adapter on some settings, but it still incurs a "
        "recurring prompt-cost penalty on every query, whereas `random_index` and `smooth_abs` return to zero extra "
        "context after a one-time edit. The rebuttal-safe takeaway is therefore about amortization rather than "
        "universal superiority: when the post-hoc edit is performance-competitive, it can become cheaper than repeated "
        "few-shot prompting after tens to a few hundred served queries, depending on k."
    )

    caveats = [
        "- `alpaca` is excluded because the current harness evaluates those adapters on IFEval, and prepending train-side exemplars would not be a clean benchmark-valid fixed-k protocol.",
        "- `k=32` should be read as a large-context stress test, not as the default prompting operating point.",
        "- The earlier math few-shot numbers were confounded by continuation/extraction behavior and should not be used; this package uses the corrected math rerun instead.",
        "- These results cover a fixed-exemplar, no-retrieval protocol only; stronger prompting schemes could behave differently, so the claim is about this operating point rather than prompting in general.",
        "- The included results are mixed rather than uniformly favorable in an absolute sense: spectral editing stays close to the original adapter, but it is not presented here as a universal accuracy improvement over the zero-shot baseline.",
    ]

    plot_rows = [row for row in method_rows if row["method"] != "fewshot_k0" and row["mean_delta_pp"] is not None]
    plot_path = out_dir / "performance_vs_compute.png"
    draw_performance_vs_compute_plot(plot_rows, out_path=plot_path, deployment_queries=args.deployment_queries)

    summary_json = {
        "included_tasks": included_tasks,
        "excluded_tasks": excluded_tasks,
        "settings": settings,
        "corrected_settings": corrected_settings,
        "method_rows": method_rows,
        "break_even_rows": break_even_rows,
        "per_setting_rows": per_setting_rows,
        "deployment_queries_for_plot": args.deployment_queries,
    }

    (out_dir / "fewshot_report.md").write_text("\n".join(report_lines))
    (out_dir / "fewshot_table.md").write_text("\n".join(table_lines) + "\n")
    (out_dir / "rebuttal_paragraph.txt").write_text(paragraph + "\n")
    (out_dir / "caveats.md").write_text("\n".join(caveats) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2))

    print(f"Report: {out_dir / 'fewshot_report.md'}")
    print(f"Table: {out_dir / 'fewshot_table.md'}")
    print(f"Plot: {plot_path}")
    print(f"Paragraph: {out_dir / 'rebuttal_paragraph.txt'}")
    print(f"Caveats: {out_dir / 'caveats.md'}")


if __name__ == "__main__":
    main()
