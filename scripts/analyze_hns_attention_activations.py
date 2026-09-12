#!/usr/bin/env python3
"""Compare LoRA/HNS attention and hidden states on HNS pass/fail flips."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from finetune.eval.eval_humaneval import build_humaneval_chat_user_prompt
from finetune.eval.eval_mbpp import build_mbpp_chat_user_prompt
from finetune.eval.generation import load_eval_tokenizer, render_chat_prompt


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_flip_tasks(
    reference: dict[str, bool],
    hns: dict[str, bool],
    *,
    max_flip: int,
    max_stable: int,
) -> dict[str, str]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for task_id in sorted(reference):
        pair = reference[task_id], hns[task_id]
        category = {
            (False, True): "gain",
            (True, False): "loss",
            (True, True): "stable_pass",
            (False, False): "stable_fail",
        }[pair]
        buckets[category].append(task_id)
    chosen: dict[str, str] = {}
    for category, task_ids in buckets.items():
        limit = max_flip if category in {"gain", "loss"} else max_stable
        for task_id in task_ids[:limit]:
            chosen[task_id] = category
    return chosen


def reduce_attention(
    attention: torch.Tensor,
    *,
    prefix_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-head normalized entropy, prompt mass, and top-1 mass."""
    probs = attention[0].float()
    seq_len = probs.shape[-2]
    query_start = min(max(1, prefix_len), max(1, seq_len - 1))
    query_indices = torch.arange(query_start, seq_len, device=probs.device)
    selected = probs[:, query_indices, :]
    entropy = -(selected * selected.clamp_min(1e-12).log()).sum(dim=-1)
    normalizer = (query_indices + 1).float().log().clamp_min(1.0)
    normalized_entropy = (entropy / normalizer.unsqueeze(0)).mean(dim=-1)
    prompt_mass = selected[..., : min(prefix_len, seq_len)].sum(dim=-1).mean(dim=-1)
    top1_mass = selected.max(dim=-1).values.mean(dim=-1)
    return normalized_entropy, prompt_mass, top1_mass


def target_nll(logits: torch.Tensor, input_ids: torch.Tensor, prefix_len: int) -> float:
    shift_logits = logits[:, :-1].float()
    shift_labels = input_ids[:, 1:]
    positions = torch.arange(1, input_ids.shape[1], device=input_ids.device)
    mask = positions >= prefix_len
    if not bool(mask.any()):
        return float("nan")
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape_as(shift_labels)
    return float(losses[:, mask].mean().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--hns_path", required=True)
    parser.add_argument("--middle_hns_path", required=True)
    parser.add_argument("--qkv_hns_path", required=True)
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--humaneval_path", required=True)
    parser.add_argument("--mbpp_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_flip", type=int, default=8)
    parser.add_argument("--max_stable", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--focus_layers", type=int, nargs="+", default=[16, 17, 18])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_root = Path(args.eval_root)
    adapters = {
        "lora": args.lora_path,
        "hns_all": args.hns_path,
        "hns_middle": args.middle_hns_path,
        "hns_qkv": args.qkv_hns_path,
    }

    human_rows = {str(row["task_id"]): row for row in pq.read_table(args.humaneval_path).to_pylist()}
    mbpp_rows = {str(row["task_id"]): row for row in pq.read_table(args.mbpp_path).to_pylist()}
    benchmark_specs = {}
    for benchmark, rows in (("humaneval", human_rows), ("mbpp", mbpp_rows)):
        if benchmark == "humaneval":
            result_name = "samples_lora.jsonl_results.jsonl"
        else:
            result_name = "outputs_lora.jsonl"
        ref = {
            str(row["task_id"]): bool(row["passed"])
            for row in read_jsonl(eval_root / "control-lora" / benchmark / result_name)
        }
        hns = {
            str(row["task_id"]): bool(row["passed"])
            for row in read_jsonl(eval_root / "control-hns4p1-allmods" / benchmark / result_name)
        }
        benchmark_specs[benchmark] = (
            rows,
            select_flip_tasks(ref, hns, max_flip=args.max_flip, max_stable=args.max_stable),
        )

    tokenizer = load_eval_tokenizer(base_model=args.base_model, adapter_dir=args.lora_path)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to("cuda")
    model = PeftModel.from_pretrained(base, args.lora_path, adapter_name="lora").to("cuda")
    for adapter_name, adapter_path in adapters.items():
        if adapter_name != "lora":
            model.load_adapter(adapter_path, adapter_name=adapter_name, is_trainable=False)
    model.eval()
    model.config.use_cache = False

    layer_acc: dict[tuple, list[dict[str, float]]] = defaultdict(list)
    head_acc: dict[tuple, list[tuple[float, float, float]]] = defaultdict(list)
    task_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    with torch.inference_mode():
        for benchmark, (problems, selected) in benchmark_specs.items():
            for task_index, (task_id, category) in enumerate(selected.items(), start=1):
                problem = problems[task_id]
                if benchmark == "humaneval":
                    user = build_humaneval_chat_user_prompt(problem["prompt"], style="opencompass")
                    target = str(problem["canonical_solution"])
                else:
                    tests = list(problem["test_list"])
                    user = build_mbpp_chat_user_prompt(str(problem["prompt"]), str(tests[0]))
                    target = str(problem["code"])
                prefix = render_chat_prompt(
                    tokenizer=tokenizer,
                    base_model=args.base_model,
                    user_content=user,
                    chat_template_mode="non_thinking",
                )
                prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
                full_ids = tokenizer(prefix + target + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
                full_ids = full_ids[: args.max_seq_len]
                prefix_len = min(len(prefix_ids), max(1, len(full_ids) - 1))
                input_ids = torch.tensor([full_ids], dtype=torch.long, device="cuda")
                attention_mask = torch.ones_like(input_ids)
                adapter_cache: dict[str, dict[str, Any]] = {}

                for adapter_name in adapters:
                    model.set_adapter(adapter_name)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=True,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    if not outputs.attentions or outputs.attentions[0] is None:
                        raise RuntimeError("Model did not return attention weights with eager attention")
                    means, rms_values = [], []
                    token_slice = slice(prefix_len, input_ids.shape[1])
                    for hidden in outputs.hidden_states:
                        target_hidden = hidden[0, token_slice].float()
                        means.append(target_hidden.mean(dim=0).cpu())
                        rms_values.append(float(target_hidden.square().mean().sqrt().item()))
                    attention_values = []
                    for layer, attention in enumerate(outputs.attentions):
                        entropy, prompt_mass, top1 = reduce_attention(attention, prefix_len=prefix_len)
                        attention_values.append((entropy.cpu(), prompt_mass.cpu(), top1.cpu()))
                        if layer in args.focus_layers:
                            for head in range(entropy.numel()):
                                head_acc[(benchmark, category, adapter_name, layer, head)].append(
                                    (float(entropy[head]), float(prompt_mass[head]), float(top1[head]))
                                )
                    adapter_cache[adapter_name] = {
                        "means": means,
                        "rms": rms_values,
                        "attention": attention_values,
                        "nll": target_nll(outputs.logits, input_ids, prefix_len),
                    }
                    del outputs

                reference_means = adapter_cache["lora"]["means"]
                for adapter_name, values in adapter_cache.items():
                    for layer, (mean_hidden, rms, attn) in enumerate(
                        zip(values["means"], values["rms"], values["attention"])
                    ):
                        ref = reference_means[min(layer + 1, len(reference_means) - 1)]
                        current = mean_hidden if layer + 1 >= len(values["means"]) else values["means"][layer + 1]
                        cosine = float(F.cosine_similarity(current, ref, dim=0).item())
                        relative_l2 = float((current - ref).norm().div(ref.norm().clamp_min(1e-12)).item())
                        entropy, prompt_mass, top1 = attn
                        layer_acc[(benchmark, category, adapter_name, layer)].append(
                            {
                                "entropy": float(entropy.mean()),
                                "prompt_mass": float(prompt_mass.mean()),
                                "top1": float(top1.mean()),
                                "activation_rms": float(values["rms"][min(layer + 1, len(values["rms"]) - 1)]),
                                "cosine_to_lora": cosine,
                                "relative_l2_to_lora": relative_l2,
                            }
                        )
                    task_rows.append(
                        {
                            "benchmark": benchmark,
                            "task_id": task_id,
                            "category": category,
                            "adapter": adapter_name,
                            "target_nll": values["nll"],
                            "tokens": len(full_ids),
                            "target_tokens": len(full_ids) - prefix_len,
                        }
                    )
                manifest.append({"benchmark": benchmark, "task_id": task_id, "category": category})
                print(f"[{benchmark}] {task_index}/{len(selected)} {task_id} {category}", flush=True)

    def avg(items: list[float]) -> float:
        return sum(items) / len(items) if items else float("nan")

    layer_lines = [
        "benchmark\tcategory\tadapter\tlayer\tn\tattention_entropy\tprompt_mass\ttop1_mass\tactivation_rms\tcosine_to_lora\trelative_l2_to_lora"
    ]
    for key, values in sorted(layer_acc.items()):
        benchmark, category, adapter, layer = key
        layer_lines.append(
            f"{benchmark}\t{category}\t{adapter}\t{layer}\t{len(values)}\t"
            + "\t".join(
                str(avg([row[field] for row in values]))
                for field in ("entropy", "prompt_mass", "top1", "activation_rms", "cosine_to_lora", "relative_l2_to_lora")
            )
        )
    (out_dir / "layer_summary.tsv").write_text("\n".join(layer_lines) + "\n", encoding="utf-8")

    head_lines = ["benchmark\tcategory\tadapter\tlayer\thead\tn\tentropy\tprompt_mass\ttop1_mass"]
    for key, values in sorted(head_acc.items()):
        benchmark, category, adapter, layer, head = key
        head_lines.append(
            f"{benchmark}\t{category}\t{adapter}\t{layer}\t{head}\t{len(values)}\t"
            f"{avg([x[0] for x in values])}\t{avg([x[1] for x in values])}\t{avg([x[2] for x in values])}"
        )
    (out_dir / "head_summary.tsv").write_text("\n".join(head_lines) + "\n", encoding="utf-8")

    task_lines = ["benchmark\ttask_id\tcategory\tadapter\ttarget_nll\ttokens\ttarget_tokens"]
    task_lines.extend("\t".join(str(row[k]) for k in ("benchmark", "task_id", "category", "adapter", "target_nll", "tokens", "target_tokens")) for row in task_rows)
    (out_dir / "task_nll.tsv").write_text("\n".join(task_lines) + "\n", encoding="utf-8")
    (out_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[Done] {out_dir}")


if __name__ == "__main__":
    main()
