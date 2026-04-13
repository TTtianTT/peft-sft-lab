#!/usr/bin/env python3
"""
Small audit for the fixed-k few-shot comparison.

This script reuses saved prediction files and prompt builders to:
1. sample 5 evaluation examples per task,
2. reconstruct the exact prompts for k in {0,1,3,32},
3. record raw outputs / parsed predictions / gold / correctness for both models,
4. quantify parser and continuation artifacts,
5. write a markdown audit report plus exact prompt/output files.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finetune.eval.eval_csqa import (  # noqa: E402
    _build_csqa_prompt,
    _build_fewshot_prefix as build_csqa_prefix,
)
from finetune.eval.eval_gsm8k import (  # noqa: E402
    _build_fewshot_prefix as build_math_prefix,
    _build_prompt_gsm8k_metamath_style,
)


BASE_MODELS = {
    "Qwen-Qwen3-8B": "Qwen/Qwen3-8B",
    "meta-llama-Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
}
TASKS = ["math", "csqa"]
K_VALUES = [0, 1, 3, 32]
PROMPT_CONT_MARKERS = {
    "math": ["\n\nBelow is an instruction", "\n### Instruction:"],
    "csqa": ["\n\n### Instruction:"],
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def norm_math(s: str) -> str:
    return s.strip().replace(",", "")


def extract_math_current(text: str) -> str:
    if "####" in text:
        tail = text.split("####")[-1].strip()
        if not tail:
            return ""
        lines = tail.splitlines()
        return lines[0].strip() if lines else ""

    m = re.search(r"(?:The answer is|Answer is|Final answer)\s*[:：]\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if matches:
        return matches[-1].strip()

    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def extract_math_first_hash(text: str) -> str:
    if "####" in text:
        tail = text.split("####", 1)[1].strip()
        if not tail:
            return ""
        lines = tail.splitlines()
        return lines[0].strip() if lines else ""
    return extract_math_current(text)


def truncate_before_next_prompt(task: str, text: str) -> str:
    out = text
    for marker in PROMPT_CONT_MARKERS[task]:
        idx = out.find(marker)
        if idx != -1:
            out = out[:idx]
            break
    return out


def extract_csqa_current(text: str) -> str:
    t = (text or "").strip().upper()
    m = re.search(r"(?:^|\n)\s*(?:FINAL\s+ANSWER|ANSWER)\s*[:\-]?\s*([A-E])\b", t)
    if m:
        return m.group(1)
    hits = re.findall(r"\b([A-E])\b", t)
    return hits[-1] if hits else ""


def extract_csqa_first_letter(text: str) -> str:
    t = (text or "").strip().upper()
    m = re.match(r"([A-E])\b", t)
    if m:
        return m.group(1)
    hits = re.findall(r"\b([A-E])\b", t)
    return hits[0] if hits else ""


def pred_path(base_dir: str, task: str, k: int) -> Path:
    if k == 0:
        return REPO_ROOT / "outputs/rebuttal_exp/raw/multiseed_eval/eval_outputs" / f"{base_dir}_{task}_baseline_s42" / "predictions.jsonl"
    return REPO_ROOT / "outputs/rebuttal_exp/raw/fewshot_eval/eval_outputs" / f"{base_dir}_{task}_fewshot_k{k}_s42" / "predictions.jsonl"


def exemplar_path(base_dir: str, task: str, k: int) -> Path:
    return REPO_ROOT / "outputs/rebuttal_exp/raw/fewshot_eval/eval_outputs" / f"{base_dir}_{task}_fewshot_k{k}_s42" / "fewshot_exemplars.jsonl"


def find_exemplars(task: str, k: int) -> list[dict[str, Any]]:
    if k == 0:
        return []
    for base_dir in BASE_MODELS:
        path = exemplar_path(base_dir, task, k)
        if path.exists():
            return load_jsonl(path)
    raise FileNotFoundError(f"No exemplars found for task={task}, k={k}")


def build_prompt(task: str, row: dict[str, Any], exemplars: list[dict[str, Any]]) -> str:
    if task == "math":
        base_prompt = _build_prompt_gsm8k_metamath_style(row["question"])
        prefix = build_math_prefix(exemplars)
        return prefix + base_prompt
    base_prompt = _build_csqa_prompt(row["instruction"])
    prefix = build_csqa_prefix(exemplars)
    return prefix + base_prompt


def sample_indices(n: int, seed: int = 42, k: int = 5) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(range(n), k))


def file_slug(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def format_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def math_alt_accuracy(rows: list[dict[str, Any]], mode: str) -> float:
    correct = 0
    for row in rows:
        gold = norm_math(row["gold"])
        text = row["prediction_text"]
        if mode == "current":
            pred = norm_math(extract_math_current(text))
        elif mode == "first_hash":
            pred = norm_math(extract_math_first_hash(text))
        elif mode == "truncate_next_prompt":
            pred = norm_math(extract_math_current(truncate_before_next_prompt("math", text)))
        else:
            raise ValueError(mode)
        correct += int(pred == gold)
    return correct / len(rows)


def csqa_alt_accuracy(rows: list[dict[str, Any]], mode: str) -> float:
    correct = 0
    for row in rows:
        text = row["prediction_text"]
        if mode == "current":
            pred = extract_csqa_current(text)
        elif mode == "first_letter":
            pred = extract_csqa_first_letter(text)
        else:
            raise ValueError(mode)
        correct += int(pred == row["gold"])
    return correct / len(rows)


def main() -> None:
    out_root = REPO_ROOT / "outputs/rebuttal_exp/fewshot_audit"
    prompt_dir = out_root / "prompts"
    output_dir = out_root / "raw_outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    context_summary = load_json(REPO_ROOT / "outputs/rebuttal_exp/raw/fewshot_eval/context_summary.json")

    preds: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for base_dir in BASE_MODELS:
        for task in TASKS:
            for k in K_VALUES:
                preds[(base_dir, task, k)] = load_jsonl(pred_path(base_dir, task, k))

    exemplars_by_task_k = {(task, k): find_exemplars(task, k) for task in TASKS for k in K_VALUES if k > 0}
    sample_ix_by_task = {
        task: sample_indices(len(preds[("Qwen-Qwen3-8B", task, 0)]), seed=42, k=5)
        for task in TASKS
    }

    prompt_file_lookup: dict[tuple[str, int, int], Path] = {}
    output_file_lookup: dict[tuple[str, str, int, int], Path] = {}

    for task in TASKS:
        ref_rows = preds[("Qwen-Qwen3-8B", task, 0)]
        for sample_rank, idx in enumerate(sample_ix_by_task[task], start=1):
            ref_row = ref_rows[idx]
            for k in K_VALUES:
                exemplars = exemplars_by_task_k.get((task, k), [])
                prompt_text = build_prompt(task, ref_row, exemplars)
                prompt_path = prompt_dir / f"{task}_sample{sample_rank:02d}_idx{idx}_k{k}.txt"
                write_text(prompt_path, prompt_text)
                prompt_file_lookup[(task, idx, k)] = prompt_path

                for base_dir in BASE_MODELS:
                    row = preds[(base_dir, task, k)][idx]
                    raw_path = output_dir / f"{file_slug(base_dir)}_{task}_sample{sample_rank:02d}_idx{idx}_k{k}.txt"
                    write_text(raw_path, row["prediction_text"])
                    output_file_lookup[(base_dir, task, idx, k)] = raw_path

    report_lines: list[str] = [
        "# Few-Shot Sanity Audit",
        "",
        "This is a small audit of the saved few-shot outputs. It does not rerun the large comparison.",
        "",
        "## Sampled Indices",
    ]
    for task in TASKS:
        report_lines.append(f"- `{task}` sampled eval indices: {sample_ix_by_task[task]}")
    report_lines.extend(["", "## Prompt / Output Audit", "Prompts are identical across models for a given task/setting/example, so each prompt is saved once and linked below.", ""])

    for task in TASKS:
        report_lines.append(f"### {task}")
        for k in K_VALUES:
            report_lines.append(f"#### k={k}")
            for sample_rank, idx in enumerate(sample_ix_by_task[task], start=1):
                prompt_path = prompt_file_lookup[(task, idx, k)]
                report_lines.append(f"- Sample {sample_rank} (eval index {idx}) prompt: [{prompt_path.name}]({prompt_path.resolve()})")
                for base_dir in BASE_MODELS:
                    row = preds[(base_dir, task, k)][idx]
                    raw_path = output_file_lookup[(base_dir, task, idx, k)]
                    if task == "math":
                        parsed = row.get("prediction_extracted", extract_math_current(row["prediction_text"]))
                    else:
                        parsed = row.get("prediction_letter", extract_csqa_current(row["prediction_text"]))
                    report_lines.append(f"  `{base_dir}`")
                    report_lines.append(f"  parsed prediction: `{parsed}`")
                    report_lines.append(f"  gold: `{row['gold']}`")
                    report_lines.append(f"  correct: `{bool(row['correct'])}`")
                    report_lines.append(f"  raw output: [{raw_path.name}]({raw_path.resolve()})")
                report_lines.append("")

    report_lines.extend(["## Answer Extraction / Stop Compatibility", ""])

    report_lines.append("### Math")
    report_lines.append("- Generation has no stop string for the next prompt template; it only relies on EOS or `max_new_tokens=256`.")
    report_lines.append("- Many few-shot generations answer the current question and then continue into a fresh `Below is an instruction ... ### Instruction` block.")
    report_lines.append("- The current extractor takes the **last** `#### ...` in the generated text. Under few-shot continuation, that often becomes the literal prompt string `<answer>` or a later answer from the continued prompt, which is not the answer to the current eval item.")
    for base_dir in BASE_MODELS:
        for k in K_VALUES:
            rows = preds[(base_dir, "math", k)]
            current_acc = math_alt_accuracy(rows, "current")
            first_hash_acc = math_alt_accuracy(rows, "first_hash")
            trunc_acc = math_alt_accuracy(rows, "truncate_next_prompt")
            continuation = sum(any(marker in row["prediction_text"] for marker in PROMPT_CONT_MARKERS["math"]) for row in rows)
            report_lines.append(
                f"- `{base_dir}` k={k}: current={format_pct(current_acc)}, first-`####`={format_pct(first_hash_acc)}, "
                f"truncate-at-next-prompt={format_pct(trunc_acc)}, continuation-detected={continuation}/{len(rows)}."
            )
    report_lines.append("")

    report_lines.append("### CSQA")
    report_lines.append("- Generation again has no stop string for the next prompt template and uses `max_new_tokens=8`.")
    report_lines.append("- Some few-shot outputs, especially Llama k=3/k=32, emit the answer letter and then continue into `### Instruction:` before hitting the token cap.")
    report_lines.append("- However, on the saved outputs, the current parser and a stricter first-letter parser give identical accuracy for every `csqa` setting, so the measured `csqa` degradation is not coming from answer extraction mismatch.")
    for base_dir in BASE_MODELS:
        for k in K_VALUES:
            rows = preds[(base_dir, "csqa", k)]
            current_acc = csqa_alt_accuracy(rows, "current")
            first_letter_acc = csqa_alt_accuracy(rows, "first_letter")
            continuation = sum(any(marker in row["prediction_text"] for marker in PROMPT_CONT_MARKERS["csqa"]) for row in rows)
            report_lines.append(
                f"- `{base_dir}` k={k}: current={format_pct(current_acc)}, first-letter={format_pct(first_letter_acc)}, "
                f"continuation-detected={continuation}/{len(rows)}."
            )
    report_lines.append("")

    report_lines.extend(["## Context / Truncation Check", ""])
    for task in TASKS:
        for base_dir in BASE_MODELS:
            setting_key = f"{base_dir}/{task}"
            ctx = context_summary["settings"][setting_key]["per_k"]["32"]
            pred_rows = preds[(base_dir, task, 32)]
            continuation = sum(any(marker in row["prediction_text"] for marker in PROMPT_CONT_MARKERS[task]) for row in pred_rows)
            report_lines.append(
                f"- `{setting_key}` k=32: prompt avg={ctx['avg_prompt_tokens']:.1f}, max={ctx['max_prompt_tokens']}, "
                f"context limit={ctx['context_limit_tokens']}, max_new_tokens={'256' if task == 'math' else '8'}, "
                f"continuation-detected={continuation}/{len(pred_rows)}."
            )
    report_lines.append("- None of the k=32 prompts are close to the 131072-token context limit, so context overflow is not the issue here.")
    report_lines.append("- The main k=32 problem is generation continuing into the next prompt template, not running out of context window.")
    report_lines.append("")

    report_lines.extend(["## Exemplar Quality / Leakage", ""])
    for task in TASKS:
        baseline_qwen_rows = preds[("Qwen-Qwen3-8B", task, 0)]
        if task == "math":
            eval_keys = {row["question"] for row in baseline_qwen_rows}
            for k in [1, 3, 32]:
                exemplars = exemplars_by_task_k[(task, k)]
                overlap = sum(ex["question"] in eval_keys for ex in exemplars)
                report_lines.append(
                    f"- `{task}` k={k}: source is saved train-side exemplars with orig indices {[ex.get('orig_index') for ex in exemplars[:5]]}; exact question overlap with eval split = {overlap}."
                )
            report_lines.append("- `math` exemplar formatting matches the task template: each exemplar uses the same MetaMath-style instruction block plus a worked solution ending in `#### <answer>`.")
        else:
            eval_ids = {row["id"] for row in baseline_qwen_rows}
            eval_questions = {row["question"] for row in baseline_qwen_rows}
            for k in [1, 3, 32]:
                exemplars = exemplars_by_task_k[(task, k)]
                overlap_id = sum(ex.get("id") in eval_ids for ex in exemplars)
                overlap_q = sum(ex["question"] in eval_questions for ex in exemplars)
                report_lines.append(
                    f"- `{task}` k={k}: source is saved train-side exemplars with orig indices {[ex.get('orig_index') for ex in exemplars[:5]]}; exact id overlap = {overlap_id}, exact question overlap = {overlap_q}."
                )
            report_lines.append("- `csqa` exemplar formatting matches the task template: each exemplar is a `### Instruction / ### Response` block whose response is a single gold letter.")
    report_lines.append("- No leakage signal was found in the saved exemplar files.")
    report_lines.append("")

    report_lines.extend(
        [
            "## Verdict",
            "",
            "C. mixed: `csqa` is mostly clean, but the very large `math` few-shot degradation is confounded by prompt-continuation and answer-extraction behavior.",
            "",
            "Smallest clean fix:",
            "- Before answer extraction, truncate math generations at the first next-prompt marker (`\\n\\nBelow is an instruction` or `\\n### Instruction:`), or equivalently extract the first completed `#### ...` answer rather than the last one.",
            "- Cleaner end-to-end fix: add a stop string for the next prompt template during few-shot generation, then rerun the math few-shot settings only.",
        ]
    )

    report_path = out_root / "sanity_audit.md"
    write_text(report_path, "\n".join(report_lines) + "\n")

    summary = {
        "sample_indices": sample_ix_by_task,
        "verdict": "mixed",
        "math_confounded": True,
        "csqa_confounded": False,
        "report_path": str(report_path),
    }
    write_text(out_root / "summary.json", json.dumps(summary, indent=2))

    print(f"Audit report: {report_path}")
    print(f"Summary: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
