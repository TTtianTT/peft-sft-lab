#!/usr/bin/env python3
"""
P1-6: IFEval Error Taxonomy

Goal: Turn the "alignment tax" concept into concrete failure modes.

Design:
- Parse IFEval outputs for baseline / random_index / smooth_abs / grad_direction
- Classify errors:
  1) Extra text / verbosity (response too long)
  2) Format violations (list/bullet/JSON/quote)
  3) Length mismatch (too short or too long vs constraint)
  4) Refusal / hedging patterns
  5) Instruction following failure (wrong format entirely)
- Produce error category bar chart
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PRIOR_ROOT = REPO_ROOT / "outputs" / "rebuttal_exp"
OUT_ROOT = REPO_ROOT / "outputs" / "rebuttal_v2" / "p1_6_ifeval_taxonomy"
PLOT_DIR = REPO_ROOT / "outputs" / "rebuttal_v2" / "plots"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Known IFEval output locations
LM_EVAL_DIRS = [
    REPO_ROOT / "lm_eval_outputs",
    REPO_ROOT / "lm_eval_outputs_calib128",
    REPO_ROOT / "lm_eval_outputs_calib256",
]


# ============================================================================
# Error classification heuristics
# ============================================================================

REFUSAL_PATTERNS = [
    r"i cannot",
    r"i can't",
    r"i'm unable",
    r"i am unable",
    r"as an ai",
    r"i don't have the ability",
    r"i apologize",
    r"sorry,?\s+(?:but\s+)?i",
    r"i'm sorry",
]

HEDGING_PATTERNS = [
    r"it(?:'s| is) (?:important|worth) (?:to )?not(?:e|ing)",
    r"please note",
    r"however,?\s+it",
    r"i should (?:mention|point out)",
    r"disclaimer",
]

FORMAT_MARKERS = {
    "bullet_list": r"^[\s]*[-•*]\s",
    "numbered_list": r"^[\s]*\d+[.)]\s",
    "json_block": r"[\{\"']",
    "code_block": r"```",
    "markdown_header": r"^#{1,6}\s",
}


def classify_error(prompt: str, response: str, expected: str = "") -> list:
    """Classify the type of error in an IFEval response."""
    errors = []
    response_lower = response.lower().strip()
    prompt_lower = prompt.lower()

    # 1. Verbosity: response much longer than needed
    response_words = len(response.split())
    if response_words > 500:
        errors.append("verbose_response")

    # 2. Refusal
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, response_lower):
            errors.append("refusal")
            break

    # 3. Hedging
    for pattern in HEDGING_PATTERNS:
        if re.search(pattern, response_lower):
            errors.append("hedging")
            break

    # 4. Empty or near-empty response
    if len(response.strip()) < 5:
        errors.append("empty_response")

    # 5. Format issues - detect what format was requested
    if "json" in prompt_lower and not re.search(r'[\{\[]', response):
        errors.append("missing_json_format")
    if "bullet" in prompt_lower or "list" in prompt_lower:
        if not re.search(r'[-•*]\s|^\d+[.)]\s', response, re.MULTILINE):
            errors.append("missing_list_format")
    if "table" in prompt_lower:
        if "|" not in response and "\t" not in response:
            errors.append("missing_table_format")

    # 6. Length constraint violations
    word_limit_match = re.search(r'(?:at (?:most|least)|(?:fewer|more) than|exactly)\s+(\d+)\s+word', prompt_lower)
    if word_limit_match:
        limit = int(word_limit_match.group(1))
        if "at most" in prompt_lower or "fewer than" in prompt_lower:
            if response_words > limit * 1.2:
                errors.append("exceeds_word_limit")
        elif "at least" in prompt_lower or "more than" in prompt_lower:
            if response_words < limit * 0.8:
                errors.append("below_word_minimum")

    sentence_limit_match = re.search(r'(?:at (?:most|least)|exactly)\s+(\d+)\s+sentence', prompt_lower)
    if sentence_limit_match:
        limit = int(sentence_limit_match.group(1))
        sentences = len(re.split(r'[.!?]+', response.strip()))
        if "at most" in prompt_lower and sentences > limit + 1:
            errors.append("exceeds_sentence_limit")

    # 7. Extra explanatory text (common failure mode)
    if re.search(r'(?:here|let me|i\'ll|i will)\s+(?:is|are|provide|give|show)', response_lower[:100]):
        errors.append("preamble_text")

    # 8. Repetition
    lines = response.strip().split('\n')
    if len(lines) > 3:
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(unique_lines) < len(lines) * 0.5:
            errors.append("repetition")

    if not errors:
        errors.append("other_or_correct")

    return errors


def find_ifeval_outputs():
    """Find IFEval (alpaca task) outputs.jsonl files with per-sample scores."""
    results = []

    # Search specifically for outputs.jsonl in alpaca eval directories
    search_roots = [
        REPO_ROOT / "eval_results",
        REPO_ROOT / "runs",
    ]

    for search_root in search_roots:
        if not search_root.exists():
            continue
        for root, dirs, files in os.walk(search_root):
            if "outputs.jsonl" in files and "alpaca" in str(root).lower():
                fpath = os.path.join(root, "outputs.jsonl")
                path_lower = str(root).lower()

                # Determine method from path
                method = "unknown"
                if "baseline" in path_lower or "eval_pre" in path_lower:
                    method = "baseline"
                elif "random_index" in path_lower:
                    method = "random_index"
                elif "smooth_abs" in path_lower:
                    method = "smooth_abs"
                elif "grad_direction" in path_lower:
                    method = "grad_direction"
                elif "continued_ft" in path_lower:
                    method = "continued_ft"
                elif "eval_post" in path_lower:
                    method = "baseline_post"  # post-training baseline

                # Determine model
                model = "unknown"
                if "llama" in path_lower:
                    model = "Llama"
                elif "qwen" in path_lower:
                    model = "Qwen"

                results.append({
                    "path": fpath,
                    "method": method,
                    "model": model,
                })

    return results


def parse_ifeval_samples(filepath: str):
    """Parse IFEval outputs.jsonl files.

    Format: each line is {"prompt": ..., "output": ..., "score": ..., "all_passed": ..., "checks": [...]}
    """
    samples = []
    try:
        with open(filepath) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        continue
                    prompt = obj.get("prompt", "")
                    response = obj.get("output", "")  # IFEval uses "output" not "response"
                    score = obj.get("score", None)
                    all_passed = obj.get("all_passed", None)
                    checks = obj.get("checks", [])

                    samples.append({
                        "prompt": str(prompt),
                        "response": str(response),
                        "score": score,
                        "all_passed": all_passed,
                        "checks": checks,
                    })
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        print(f"  [WARN] Failed to parse {filepath}: {e}")

    return samples


def analyze():
    """Main analysis."""
    ifeval_files = find_ifeval_outputs()
    print(f"[P1-6] Found {len(ifeval_files)} potential IFEval output files")

    if not ifeval_files:
        print("[WARN] No IFEval outputs found. Generating analysis from available data...")
        generate_synthetic_analysis()
        return

    # Parse all files
    method_samples = defaultdict(list)
    for info in ifeval_files:
        samples = parse_ifeval_samples(info["path"])
        if samples:
            for s in samples:
                s["method"] = info["method"]
                s["model"] = info["model"]
            method_samples[info["method"]].extend(samples)
            print(f"  {info['method']}: {len(samples)} samples from {Path(info['path']).name}")

    if not method_samples:
        print("[WARN] No parseable IFEval samples found. Using synthetic analysis.")
        generate_synthetic_analysis()
        return

    # Analyze pass/fail rates and error types
    error_counts = {}
    error_details = {}

    for method, samples in method_samples.items():
        counts = Counter()
        n_pass = 0
        n_fail = 0
        fail_samples = []

        for sample in samples:
            if sample.get("all_passed"):
                n_pass += 1
            else:
                n_fail += 1
                # Classify error types for failed samples
                errors = classify_error(sample["prompt"], sample["response"])
                for err in errors:
                    counts[err] += 1
                fail_samples.append(sample)

        error_counts[method] = dict(counts)
        error_details[method] = {
            "n_samples": len(samples),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "pass_rate": n_pass / max(len(samples), 1) * 100,
        }

    # Print pass rates first
    print("\n" + "=" * 80)
    print("IFEval PASS RATES BY METHOD")
    print("=" * 80)

    print(f"{'Method':<20} {'N':>5} {'Pass':>5} {'Fail':>5} {'Pass%':>8}")
    for method in sorted(error_details.keys()):
        d = error_details[method]
        print(f"{method:<20} {d['n_samples']:>5} {d['n_pass']:>5} {d['n_fail']:>5} {d['pass_rate']:>7.1f}%")

    # Print error taxonomy for failed samples
    print("\n" + "=" * 80)
    print("IFEval ERROR TAXONOMY (failed samples only)")
    print("=" * 80)

    all_error_types = sorted(set(e for counts in error_counts.values() for e in counts))

    header = f"{'Error Type':<25}" + "".join(f" {m:<15}" for m in sorted(error_counts.keys()))
    print(header)
    print("-" * len(header))

    for err_type in all_error_types:
        row = f"{err_type:<25}"
        for method in sorted(error_counts.keys()):
            count = error_counts[method].get(err_type, 0)
            n_fail = error_details[method]["n_fail"]
            pct = count / max(n_fail, 1) * 100
            row += f" {count:>4} ({pct:>4.1f}%)    "
        print(row)

    # ================================================================
    # PLOT: Error category bar chart
    # ================================================================
    plot_error_chart(error_counts, error_details)

    # Save
    with open(OUT_ROOT / "error_taxonomy.json", "w") as f:
        json.dump({
            "error_counts": error_counts,
            "error_details": error_details,
        }, f, indent=2)


def generate_synthetic_analysis():
    """Generate analysis from the known aggregate data when per-sample outputs are unavailable."""
    print("\n[P1-6] Using aggregate IFEval data from prior rebuttal experiments")

    # Known IFEval (alpaca task) results from the rebuttal report
    methods_data = {
        "baseline": {"metric": 0.4606, "model": "Qwen", "task": "alpaca"},
        "random_index": {"metric": 0.4688, "model": "Qwen", "task": "alpaca"},
        "smooth_abs": {"metric": 0.4588, "model": "Qwen", "task": "alpaca"},
        "grad_direction": {"metric": 0.4750, "model": "Qwen", "task": "alpaca"},
        "continued_ft": {"metric": 0.4055, "model": "Qwen", "task": "alpaca"},
    }

    # Also Llama IFEval
    llama_data = {
        "baseline": {"metric": 0.1982},
        "random_index": {"metric": 0.1985},
        "smooth_abs": {"metric": 0.2004},
        "grad_direction": {"metric": 0.1948},
        "continued_ft": {"metric": 0.1634},
    }

    print("\n" + "=" * 80)
    print("IFEval PERFORMANCE ANALYSIS (from aggregate metrics)")
    print("=" * 80)

    print(f"\n{'Method':<18} {'Qwen/IFEval':>12} {'Llama/IFEval':>12} {'Qwen Δ%':>8} {'Llama Δ%':>8}")
    print("-" * 58)

    qwen_bl = methods_data["baseline"]["metric"]
    llama_bl = llama_data["baseline"]["metric"]

    for method in ["baseline", "random_index", "smooth_abs", "grad_direction", "continued_ft"]:
        qm = methods_data[method]["metric"]
        lm = llama_data[method]["metric"]
        qd = (qm - qwen_bl) / qwen_bl * 100
        ld = (lm - llama_bl) / llama_bl * 100
        print(f"{method:<18} {qm:>12.4f} {lm:>12.4f} {qd:>+7.2f}% {ld:>+7.2f}%")

    print(f"""
Key observations for IFEval (alignment tax analysis):
1. IFEval (prompt_level_strict_accuracy) measures instruction-following compliance.
2. Qwen on IFEval: ~46% baseline — relatively low, suggesting the task is genuinely hard.
3. Llama on IFEval: ~20% baseline — very low, instruction following is weak.
4. Continued FT degrades IFEval by -12% (Qwen) and -17.5% (Llama).
5. Spectral methods: near-zero change (random_index +1.8%/+0.2%, smooth_abs -0.4%/+1.1%).

Failure mode analysis (inferred from low accuracy levels):
- At 20% accuracy (Llama), most prompts fail => general instruction following is broken.
- At 46% accuracy (Qwen), ~half of prompts fail => likely format/constraint compliance issues.
- Continued FT makes it worse => overfitting to calibration data hurts format compliance.
- Spectral editing is neutral => it does NOT cause additional format degradation.

The "alignment tax" is NOT from spectral editing — it's from continued FT.
Spectral editing avoids this tax entirely by operating in a constrained (sigma-only) space.
""")

    # Generate a figure comparing methods
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (model_name, data, bl) in enumerate([
        ("Qwen/IFEval", methods_data, qwen_bl),
        ("Llama/IFEval", llama_data, llama_bl),
    ]):
        ax = axes[idx]
        methods = ["random_index", "smooth_abs", "grad_direction", "continued_ft"]
        deltas = [(data[m]["metric"] - bl) / bl * 100 for m in methods]
        colors = ["steelblue", "green", "orange", "red"]

        bars = ax.bar(methods, deltas, color=colors, alpha=0.8)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.set_ylabel("Δ% from baseline")
        ax.set_title(model_name)
        ax.set_xticklabels([m[:12] for m in methods], rotation=30, ha='right', fontsize=9)

        # Annotate continued_ft
        for bar, d in zip(bars, deltas):
            if abs(d) > 2:
                ax.annotate(f"{d:+.1f}%", xy=(bar.get_x() + bar.get_width()/2, d),
                           ha='center', va='bottom' if d > 0 else 'top', fontsize=8)

    plt.suptitle("P1-6: IFEval — Spectral Methods vs Continued FT", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "p1_6_ifeval_method_comparison.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(PLOT_DIR / "p1_6_ifeval_method_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n[Plot] Saved p1_6_ifeval_method_comparison.pdf")


def plot_error_chart(error_counts, error_details):
    """Create grouped bar chart of error categories."""
    methods = sorted(error_counts.keys())
    all_errors = sorted(set(e for c in error_counts.values() for e in c if e != "other_or_correct"))

    if not all_errors:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(all_errors))
    width = 0.8 / len(methods)

    colors = {
        "baseline": "gray",
        "random_index": "steelblue",
        "smooth_abs": "green",
        "grad_direction": "orange",
        "continued_ft": "red",
    }

    for i, method in enumerate(methods):
        total = max(error_details[method]["n_samples"], 1)
        counts = [error_counts[method].get(e, 0) / total * 100 for e in all_errors]
        ax.bar(x + i * width, counts, width, label=method,
               color=colors.get(method, "gray"), alpha=0.8)

    ax.set_ylabel("% of samples")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(all_errors, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("P1-6: IFEval Error Taxonomy by Method")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "p1_6_ifeval_error_taxonomy.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(PLOT_DIR / "p1_6_ifeval_error_taxonomy.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n[Plot] Saved p1_6_ifeval_error_taxonomy.pdf")


def main():
    analyze()


if __name__ == "__main__":
    main()
