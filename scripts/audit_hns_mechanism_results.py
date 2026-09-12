#!/usr/bin/env python3
"""Audit completeness and internal consistency of the final HNS mechanism results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


EXPECTED_MODULES = {
    "qwen_magicoder": 252,
    "qwen_commonsense": 252,
    "llama_tulu": 224,
}


def read_table(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def finite(row: dict[str, str], fields: tuple[str, ...], context: str) -> None:
    for field in fields:
        value = float(row[field])
        require(math.isfinite(value), f"non-finite {field} in {context}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate_root", required=True)
    parser.add_argument("--statistics_root", required=True)
    parser.add_argument("--localization_root", required=True)
    parser.add_argument("--utility_root", required=True)
    parser.add_argument(
        "--fc_root",
        help="Optional decisive functional-concentration intervention root.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    aggregate = Path(args.aggregate_root)
    statistics = Path(args.statistics_root)
    localization = Path(args.localization_root)
    utility = Path(args.utility_root)
    checks: dict[str, object] = {}

    required_aggregate = (
        "summary.tsv",
        "module_spectra.csv",
        "direction_response.csv",
        "layer_response.csv",
        "module_response.csv",
        "cross_task_summary.tsv",
        "cross_task_correlations.tsv",
        "integrity_checks.tsv",
    )
    missing = [name for name in required_aggregate if not (aggregate / name).is_file()]
    require(not missing, f"missing aggregate files: {missing}")

    cross = read_table(aggregate / "cross_task_summary.tsv")
    require(len(cross) == 8, f"expected 8 cross-task rows, got {len(cross)}")
    cross_keys = [(row["base_model"], row["task"]) for row in cross]
    require(len(set(cross_keys)) == 8, "duplicate cross-task checkpoint")
    for row in cross:
        finite(
            row,
            (
                "raw_top1_spectral_share",
                "functional_top1_share",
                "raw_effective_rank",
                "functional_effective_rank",
                "hns_to_lora_p99_ratio",
                "hns_gain",
            ),
            f"cross-task {row['base_model']}:{row['task']}",
        )
        require(
            float(row["functional_top1_share"]) > float(row["raw_top1_spectral_share"]),
            f"functional top-1 did not exceed raw top-1 for {row['base_model']}:{row['task']}",
        )
        require(
            0.0 < float(row["hns_to_lora_p99_ratio"]) < 1.0,
            f"HNS did not suppress p99 for {row['base_model']}:{row['task']}",
        )
    checks["cross_task"] = {
        "rows": len(cross),
        "functional_top1_exceeds_raw": len(cross),
        "hns_p99_ratio_below_one": len(cross),
    }

    integrity = read_table(aggregate / "integrity_checks.tsv")
    require(len(integrity) == 8, f"expected 8 integrity rows, got {len(integrity)}")
    for row in integrity:
        require(row["basis_check_pass"] == "True", f"basis check failed for {row}")
        require(int(row["samples"]) == 256, f"wrong sample count for {row}")
        require(int(row["unique_sample_indices"]) == 256, f"duplicate sample index for {row}")
        require(int(row["variants"]) == 6, f"wrong variant count for {row}")
        require(
            int(row["direction_rows"]) == int(row["expected_direction_rows"]),
            f"direction row mismatch for {row}",
        )
    checks["spectral_functional_integrity"] = {
        "checkpoints": len(integrity),
        "fixed_samples_each": 256,
        "basis_checks_passed": len(integrity),
        "max_basis_offdiag_fraction": max(float(row["max_basis_offdiag_fraction"]) for row in integrity),
    }

    paired = read_table(statistics / "paired_inference.tsv")
    category = read_table(statistics / "category_inference.tsv")
    base_comparisons = 8 * 5
    head_fro_comparisons = sum(row["variant"] == "head_fro_restore" for row in paired)
    expected_paired = base_comparisons + head_fro_comparisons
    require(
        len(paired) == expected_paired,
        f"expected {expected_paired} paired comparisons, got {len(paired)}",
    )
    require(len(category) >= len(paired), f"too few category comparisons: {len(category)}")
    paired_keys = [(row["base_model"], row["task"], row["variant"]) for row in paired]
    require(len(set(paired_keys)) == len(paired), "duplicate paired comparison")
    for row in paired:
        finite(row, ("delta", "bootstrap_ci_low", "bootstrap_ci_high", "paired_permutation_p"), str(paired_keys))
        require(
            float(row["bootstrap_ci_low"]) <= float(row["delta"]) <= float(row["bootstrap_ci_high"]),
            f"paired estimate outside CI for {row}",
        )
    checks["paired_inference"] = {
        "comparisons": len(paired),
        "head_fro_restore_comparisons": head_fro_comparisons,
        "category_comparisons": len(category),
        "ci_excludes_zero": sum(row["ci_excludes_zero"] == "True" for row in paired),
    }

    # The Llama commonsense suite was freshly rerun after the original aggregate
    # was built. Record (rather than hide) every stale point estimate.
    paired_lookup = {(row["base_model"], row["task"], row["variant"]): row for row in paired}
    stale_differences: list[dict[str, object]] = []
    for row in cross:
        key = (row["base_model"], row["task"])
        for variant, cross_field in (
            ("scalar_shrink", "scalar_shrink_gain"),
            ("shape_only", "shape_only_gain"),
            ("head_only", "head_only_gain"),
            ("tail_only", "tail_only_gain"),
            ("hns", "hns_gain"),
        ):
            inferred = paired_lookup[key + (variant,)]
            difference = float(inferred["delta"]) - float(row[cross_field])
            if abs(difference) > 1e-10:
                stale_differences.append(
                    {
                        "base_model": key[0],
                        "task": key[1],
                        "variant": variant,
                        "aggregate_gain": float(row[cross_field]),
                        "fresh_paired_gain": float(inferred["delta"]),
                        "difference": difference,
                    }
                )
    checks["refreshed_point_estimates"] = stale_differences

    localization_inference = read_table(localization / "inference" / "localization_paired_inference.tsv")
    require(len(localization_inference) == 54, f"expected 54 localization comparisons, got {len(localization_inference)}")
    localization_counts: dict[str, int] = {}
    for case in EXPECTED_MODULES:
        rows = read_table(localization / case / "localization_summary.tsv")
        require(len(rows) == 16, f"expected 16 localization variants for {case}, got {len(rows)}")
        require(len({row["label"] for row in rows}) == 16, f"duplicate localization label for {case}")
        localization_counts[case] = len(rows)
    checks["functional_localization"] = {
        "variants_per_case": localization_counts,
        "paired_comparisons": len(localization_inference),
    }

    feature_rows = read_table(utility / "analysis" / "module_utility_features.tsv")
    require(len(feature_rows) == sum(EXPECTED_MODULES.values()), "wrong joined utility row count")
    require(
        len({(row["case"], row["module"]) for row in feature_rows}) == len(feature_rows),
        "duplicate joined utility module",
    )
    utility_checks: dict[str, object] = {}
    for case, expected_modules in EXPECTED_MODULES.items():
        rows = read_table(utility / "utility" / case / "module_utility.tsv")
        examples = read_table(utility / "utility" / case / "module_utility_examples.tsv")
        metadata = json.loads((utility / "utility" / case / "metadata.json").read_text())
        used_samples = int(metadata["used_samples"])
        require(len(rows) == expected_modules, f"wrong utility module count for {case}")
        require(len({row["module"] for row in rows}) == expected_modules, f"duplicate utility module for {case}")
        require(len(examples) == expected_modules * used_samples, f"wrong utility example count for {case}")
        for row in rows:
            finite(row, ("utility", "utility_ci_low", "utility_ci_high"), case)
            require(
                float(row["utility_ci_low"]) <= float(row["utility"]) <= float(row["utility_ci_high"]),
                f"utility estimate outside CI for {case}:{row['module']}",
            )
        utility_checks[case] = {"modules": len(rows), "heldout_samples_used": used_samples}
    checks["module_utility"] = utility_checks
    checks["module_utility"]["joined_rows"] = len(feature_rows)

    model_rows = read_table(utility / "analysis" / "utility_models.tsv")
    correlation_rows = read_table(utility / "analysis" / "utility_correlations.tsv")
    require(len(model_rows) == 20, f"expected 20 utility model rows, got {len(model_rows)}")
    require(len(correlation_rows) == 16, f"expected 16 utility correlation rows, got {len(correlation_rows)}")
    checks["utility_analysis"] = {
        "model_rows": len(model_rows),
        "correlation_rows": len(correlation_rows),
    }

    if args.fc_root:
        fc_root = Path(args.fc_root)
        completion = json.loads((fc_root / "completion.json").read_text())
        require(completion["status"] == "complete", "decisive follow-up is incomplete")
        require(completion["gpu_policy"] == "one serial GPU allocation", "unexpected GPU policy")
        require(completion["head_fro_checkpoints"] == 8, "HeadFro coverage is not 8/8")
        require(set(completion["fc_cases"]) == set(EXPECTED_MODULES), "wrong decisive FC cases")

        fc_case_checks: dict[str, object] = {}
        for case, expected_modules in EXPECTED_MODULES.items():
            manifest = json.loads((fc_root / case / "adapters" / "manifest.json").read_text())
            variants = manifest["variants"]
            require(len(variants) == 10, f"expected 10 decisive variants for {case}")
            require(len({row["label"] for row in variants}) == 10, f"duplicate decisive variant for {case}")
            for row in variants:
                expected_selected = expected_modules // (4 if row["scope_fraction_requested"] == 0.25 else 2)
                require(row["selected_count"] == expected_selected, f"wrong selected count for {case}:{row['label']}")
                require(row["total_modules"] == expected_modules, f"wrong total module count for {case}")

            verification = json.loads((fc_root / case / "adapter_verification.json").read_text())
            require(verification["status"] == "PASS", f"adapter verification failed for {case}")
            require(verification["variants"] == 10, f"wrong verified variant count for {case}")
            require(
                all(row["tensor_mismatches"] == 0 for row in verification["results"]),
                f"tensor mismatch in decisive adapters for {case}",
            )

            localization_rows = read_table(fc_root / case / "localization_summary.tsv")
            require(len(localization_rows) == 10, f"expected 10 decisive localization rows for {case}")
            nll_rows = read_table(fc_root / case / "nll" / "adapter_nll.tsv")
            require(len(nll_rows) == 5, f"expected 5 adapter-set NLL rows for {case}")
            nll_examples = read_table(fc_root / case / "nll" / "adapter_nll_examples.tsv")
            nll_metadata = json.loads((fc_root / case / "nll" / "metadata.json").read_text())
            require(
                len(nll_examples) == 5 * int(nll_metadata["used_samples"]),
                f"wrong decisive NLL example count for {case}",
            )
            fc_case_checks[case] = {
                "modules": expected_modules,
                "variants": len(variants),
                "verified_tensor_mismatches": 0,
                "nll_sets": len(nll_rows),
                "nll_samples": int(nll_metadata["used_samples"]),
            }

        fc_inference = read_table(fc_root / "inference" / "fc_paired_inference.tsv")
        nonadditivity = read_table(fc_root / "analysis" / "nonadditivity.tsv")
        require(len(fc_inference) == 60, f"expected 60 decisive FC comparisons, got {len(fc_inference)}")
        require(len(nonadditivity) == 12, f"expected 12 nonadditivity rows, got {len(nonadditivity)}")
        for row in nonadditivity:
            finite(
                row,
                (
                    "observed_set_utility",
                    "sum_single_module_utility",
                    "interaction_observed_minus_additive",
                    "interaction_ci_low",
                    "interaction_ci_high",
                ),
                f"nonadditivity {row['case']}:{row['label']}",
            )
            require(
                float(row["interaction_ci_low"])
                <= float(row["interaction_observed_minus_additive"])
                <= float(row["interaction_ci_high"]),
                f"nonadditivity estimate outside CI for {row}",
            )
        checks["decisive_followup"] = {
            "completion": completion,
            "cases": fc_case_checks,
            "paired_comparisons": len(fc_inference),
            "nonadditivity_sets": len(nonadditivity),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "PASS", "checks": checks}, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
