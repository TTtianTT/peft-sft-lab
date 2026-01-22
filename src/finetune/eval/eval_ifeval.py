from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from finetune.eval.generation import generate_greedy, generate_greedy_vllm_batch, load_transformers_model, save_json
from finetune.utils import seed_everything


def _as_text_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        out: list[str] = []
        for x in val:
            if x is None:
                continue
            out.append(str(x))
        return out
    return [str(val)]


def _extract_constraints(instruction_text: str) -> list[tuple[str, Any]]:
    t = instruction_text.lower()
    constraints: list[tuple[str, Any]] = []

    m = re.search(r"exactly\s+(\d+)\s+words", t)
    if m:
        constraints.append(("exact_word_count", int(m.group(1))))

    m = re.search(r"(?:at most|no more than)\s+(\d+)\s+words", t)
    if m:
        constraints.append(("max_word_count", int(m.group(1))))

    m = re.search(r"(?:at least)\s+(\d+)\s+words", t)
    if m:
        constraints.append(("min_word_count", int(m.group(1))))

    m = re.search(r'start with\s+"([^"]+)"', instruction_text, flags=re.IGNORECASE)
    if m:
        constraints.append(("starts_with", m.group(1)))

    m = re.search(r'end with\s+"([^"]+)"', instruction_text, flags=re.IGNORECASE)
    if m:
        constraints.append(("ends_with", m.group(1)))

    m = re.search(r'include the (?:word|phrase)\s+"([^"]+)"', instruction_text, flags=re.IGNORECASE)
    if m:
        constraints.append(("contains", m.group(1)))

    m = re.search(r'do not (?:include|use) the (?:word|phrase)\s+"([^"]+)"', instruction_text, flags=re.IGNORECASE)
    if m:
        constraints.append(("not_contains", m.group(1)))

    if "json" in t:
        constraints.append(("valid_json", True))

    if "bullet" in t or "bulleted" in t:
        constraints.append(("bullet_list", True))

    if "one sentence" in t:
        constraints.append(("one_sentence", True))

    return constraints


def _count_words(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def _count_sentences(text: str) -> int:
    parts = re.split(r"[.!?]+", text.strip())
    return len([p for p in parts if p.strip()])


def score_ifeval(*, instruction_text: str, output_text: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    out = output_text.strip()
    checks.append({"name": "non_empty", "passed": bool(out)})

    for name, arg in _extract_constraints(instruction_text):
        if name == "exact_word_count":
            checks.append({"name": name, "arg": arg, "passed": _count_words(out) == int(arg)})
        elif name == "max_word_count":
            checks.append({"name": name, "arg": arg, "passed": _count_words(out) <= int(arg)})
        elif name == "min_word_count":
            checks.append({"name": name, "arg": arg, "passed": _count_words(out) >= int(arg)})
        elif name == "starts_with":
            checks.append({"name": name, "arg": arg, "passed": out.startswith(str(arg))})
        elif name == "ends_with":
            checks.append({"name": name, "arg": arg, "passed": out.endswith(str(arg))})
        elif name == "contains":
            checks.append({"name": name, "arg": arg, "passed": str(arg).lower() in out.lower()})
        elif name == "not_contains":
            checks.append({"name": name, "arg": arg, "passed": str(arg).lower() not in out.lower()})
        elif name == "valid_json":
            ok = False
            try:
                json.loads(out)
                ok = True
            except Exception:
                ok = False
            checks.append({"name": name, "passed": ok})
        elif name == "bullet_list":
            ok = any(line.strip().startswith(("-", "*")) for line in out.splitlines())
            checks.append({"name": name, "passed": ok})
        elif name == "one_sentence":
            checks.append({"name": name, "passed": _count_sentences(out) == 1})

    passed = sum(1 for c in checks if c.get("passed"))
    total = len(checks)
    return {
        "passed": passed,
        "total": total,
        "score": (passed / total if total else 0.0),
        "all_passed": bool(total and passed == total),
        "checks": checks,
    }


def _load_ifeval_dataset(split: str):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(f"datasets is required: {exc}") from exc

    tried = []
    for ds_id in ["google/ifeval", "google/IFEval", "HuggingFaceH4/ifeval"]:
        try:
            return load_dataset(ds_id, split=split)
        except Exception as exc:
            tried.append((ds_id, str(exc)))

    msg = "Failed to load IFEval dataset. Tried:\n" + "\n".join([f"- {k}: {v}" for k, v in tried])
    raise RuntimeError(msg)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate IFEval (minimal rule-based scoring).")
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--adapter_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = out_dir / "outputs.jsonl"

    ds = _load_ifeval_dataset(args.split)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    loaded = None
    if not args.use_vllm:
        loaded = load_transformers_model(
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            dtype=args.dtype,
            device_map="auto",
        )

    total = 0
    strict = 0
    score_sum = 0.0
    checks_sum = 0

    with outputs_path.open("w", encoding="utf-8") as f:
        if args.use_vllm:
            prompts: list[str] = []
            records: list[dict[str, str]] = []
            for ex in ds:
                prompt = ""
                if "prompt" in ex:
                    prompt = str(ex["prompt"])
                elif "input" in ex:
                    prompt = str(ex["input"])
                elif "question" in ex:
                    prompt = str(ex["question"])
                else:
                    prompt = str(ex)

                instruction_parts = _as_text_list(ex.get("instruction")) + _as_text_list(ex.get("instructions"))
                instruction_text = "\n".join([p for p in instruction_parts if p.strip()]) or prompt
                prompts.append(prompt)
                records.append({"prompt": prompt, "instruction_text": instruction_text})

            generations = generate_greedy_vllm_batch(
                base_model=args.base_model,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                adapter_dir=args.adapter_dir,
                tensor_parallel_size=args.tensor_parallel_size,
            )

            for rec, gen in zip(records, generations):
                scored = score_ifeval(instruction_text=rec["instruction_text"], output_text=gen)
                total += 1
                strict += int(scored["all_passed"])
                score_sum += float(scored["score"])
                checks_sum += int(scored["total"])

                f.write(
                    json.dumps(
                        {
                            "prompt": rec["prompt"],
                            "output": gen,
                            "score": scored["score"],
                            "all_passed": scored["all_passed"],
                            "checks": scored["checks"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        else:
            for ex in ds:
                prompt = ""
                if "prompt" in ex:
                    prompt = str(ex["prompt"])
                elif "input" in ex:
                    prompt = str(ex["input"])
                elif "question" in ex:
                    prompt = str(ex["question"])
                else:
                    prompt = str(ex)

                instruction_parts = _as_text_list(ex.get("instruction")) + _as_text_list(ex.get("instructions"))
                instruction_text = "\n".join([p for p in instruction_parts if p.strip()]) or prompt

                gen = generate_greedy(
                    model=loaded.model, tokenizer=loaded.tokenizer, prompt=prompt, max_new_tokens=args.max_new_tokens
                )

                scored = score_ifeval(instruction_text=instruction_text, output_text=gen)
                total += 1
                strict += int(scored["all_passed"])
                score_sum += float(scored["score"])
                checks_sum += int(scored["total"])

                f.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "output": gen,
                            "score": scored["score"],
                            "all_passed": scored["all_passed"],
                            "checks": scored["checks"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    metrics = {
        "ifeval_strict_accuracy": (strict / total if total else 0.0),
        "ifeval_avg_score": (score_sum / total if total else 0.0),
        "avg_checks_per_example": (checks_sum / total if total else 0.0),
        "total": total,
    }
    save_json(out_dir / "metrics.json", metrics)


if __name__ == "__main__":
    main()
