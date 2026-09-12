#!/usr/bin/env python3
"""Measure RLOO reward gradients along three predeclared MetaMath spectral paths."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from finetune.spectral_edit.io import (  # noqa: E402
    get_scaling_for_module,
    load_adapter_config,
    load_lora_state_dict,
    parse_lora_ab_key,
)
from finetune.spectral_edit.mechanism import (  # noqa: E402
    align_edited_spectrum_to_reference,
    build_causal_control_spectra,
)
from finetune.spectral_edit.svd import lowrank_svd_from_ba  # noqa: E402


PATHS = ("full_hns", "head_only", "scalar_shrink")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", required=True)
    p.add_argument("--lora_path", required=True)
    p.add_argument("--hns_path", required=True)
    p.add_argument("--rollouts", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--expected_rollouts", type=int, default=4)
    p.add_argument("--smoke_samples", type=int, default=8)
    p.add_argument("--finite_difference_steps", type=float, nargs="+", default=(0.01, 0.05))
    p.add_argument("--finite_difference_rtol", type=float, default=0.35)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260910)
    return p.parse_args()


def canonical(prefix: str) -> str:
    position = prefix.find("layers.")
    if position < 0:
        raise ValueError(prefix)
    return prefix[position:]


def collect_pairs(path: str) -> tuple[dict[str, dict[str, tuple[str, torch.Tensor, str | None]]], dict]:
    state, _ = load_lora_state_dict(path)
    config = load_adapter_config(path)
    pairs: dict[str, dict[str, tuple[str, torch.Tensor, str | None]]] = defaultdict(dict)
    for key, tensor in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is None:
            continue
        prefix, which, adapter = parsed
        pairs[canonical(prefix)][which] = (key, tensor, adapter)
    return {name: pair for name, pair in pairs.items() if {"A", "B"} <= pair.keys()}, config


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty TSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def comparable_vllm_logprob(row: dict, temperature: float) -> float:
    """Return only documented sampling-policy scores comparable to replay.

    Legacy V1 rollouts stored raw T=1 logprobs. They cannot quantify a
    precision gap against T=0.7 replay or supply importance-weight denominators.
    """
    if row.get("vllm_logprobs_mode") != "processed_logprobs":
        return math.nan
    sampling = row.get("sampling_distribution", {})
    expected = {
        "temperature": temperature,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }
    if any(key not in sampling or not math.isclose(float(sampling[key]), value)
           for key, value in expected.items()):
        return math.nan
    score = row.get("cumulative_logprob_vllm")
    return float(score) if score is not None else math.nan


class SpectralPathGate:
    """Add exact first-order LoRA-basis path deltas to PEFT linear outputs."""

    def __init__(self, gate: torch.Tensor, modules: dict[str, dict]) -> None:
        self.gate = gate
        self.modules = modules
        self.handles: list = []

    def register(self) -> None:
        if self.handles:
            raise RuntimeError("spectral path hooks already registered")
        for name, spec in self.modules.items():
            def hook(module, inputs, output, *, prefix=name):
                if not torch.is_tensor(output):
                    raise TypeError(f"expected tensor output for {prefix}, got {type(output)}")
                x = inputs[0]
                item = self.modules[prefix]
                # Keep the rank-r path arithmetic in fp32.  The final cast is
                # the same dtype boundary that an edited bf16 LoRA output sees.
                dsigma = self.gate @ item["path_deltas"]
                projected = x.float() @ item["V"]
                delta = (projected * dsigma) @ item["U"].T
                delta = delta * item["scaling"]
                return output + delta.to(dtype=output.dtype)

            self.handles.append(spec["module"].register_forward_hook(hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def encode_rollout_batch(records: list[dict], pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    sequences = [row["prompt_token_ids"] + row["response_token_ids"] for row in records]
    prompt_lengths = [len(row["prompt_token_ids"]) for row in records]
    if any(not row["response_token_ids"] for row in records):
        raise RuntimeError("empty response token sequence cannot define a sequence log probability")
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    response_mask = torch.zeros((len(sequences), max_len - 1), dtype=torch.float32, device=device)
    for index, (sequence, prompt_len) in enumerate(zip(sequences, prompt_lengths)):
        length = len(sequence)
        input_ids[index, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[index, :length] = 1
        response_mask[index, prompt_len - 1 : length - 1] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
    }


def sequence_logprobs(model, batch: dict[str, torch.Tensor], temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    result = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    logits = result.logits
    token_logprobs = F.log_softmax(logits[:, :-1].float() / temperature, dim=-1)
    targets = batch["input_ids"][:, 1:]
    selected = token_logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (selected * batch["response_mask"]).sum(dim=1), logits


def build_path_modules(model, lora_path: str, hns_path: str) -> tuple[dict[str, dict], list[dict]]:
    lora_pairs, config = collect_pairs(lora_path)
    hns_pairs, _ = collect_pairs(hns_path)
    if set(lora_pairs) != set(hns_pairs):
        raise RuntimeError("LoRA/HNS module sets differ")
    lookup = dict(model.named_modules())
    modules: dict[str, dict] = {}
    stats: list[dict] = []
    for name in sorted(lora_pairs):
        candidates = [loaded for loaded in lookup if loaded.endswith(name)]
        if len(candidates) != 1:
            raise RuntimeError(f"expected one loaded module for {name}, got {candidates}")
        pair = lora_pairs[name]
        a = pair["A"][1].to("cuda")
        b = pair["B"][1].to("cuda")
        U, sigma, Vh, V = lowrank_svd_from_ba(b, a)
        hns_sigma, basis = align_edited_spectrum_to_reference(
            U,
            Vh,
            hns_pairs[name]["B"][1].to("cuda"),
            hns_pairs[name]["A"][1].to("cuda"),
        )
        controls, control_stats = build_causal_control_spectra(sigma, hns_sigma)
        path_targets = {
            "full_hns": hns_sigma,
            "head_only": controls["head_only"],
            "scalar_shrink": controls["scalar_shrink"],
        }
        deltas = torch.stack([path_targets[path] - sigma for path in PATHS]).float()
        adapter = pair["A"][2] or pair["B"][2]
        modules[name] = {
            "module": lookup[candidates[0]],
            "U": U.detach().float(),
            "V": V.detach().float(),
            "path_deltas": deltas.detach(),
            "scaling": float(get_scaling_for_module(config, name)),
        }
        stats.append(
            {
                "module": name,
                "layer": int(name.split(".")[1]),
                "module_type": name.rsplit(".", 1)[-1],
                "rank": len(sigma),
                "lora_fro": control_stats["lora_fro"],
                "hns_fro": control_stats["hns_fro"],
                "hns_to_lora_fro_ratio": control_stats["hns_to_lora_fro_ratio"],
                "full_hns_path_norm": float(torch.linalg.vector_norm(deltas[0]).item()),
                "head_only_path_norm": float(torch.linalg.vector_norm(deltas[1]).item()),
                "scalar_shrink_path_norm": float(torch.linalg.vector_norm(deltas[2]).item()),
                **basis,
            }
        )
    return modules, stats


def smoke_check(
    model,
    gate: torch.Tensor,
    path_gate: SpectralPathGate,
    records: list[dict],
    pad_id: int,
    args: argparse.Namespace,
) -> dict:
    # One shortest nonempty rollout per question prevents a single question
    # from dominating this implementation-only check.
    first_by_question: dict[str, dict] = {}
    for row in sorted(records, key=lambda value: value["response_tokens"]):
        first_by_question.setdefault(row["question_id"], row)
    chosen = list(first_by_question.values())[: args.smoke_samples]
    if len(chosen) < args.smoke_samples:
        raise RuntimeError(f"only {len(chosen)} distinct nonempty smoke samples")
    batch = encode_rollout_batch(chosen, pad_id, gate.device)

    path_gate.remove()
    with torch.inference_mode():
        no_hook_scores, no_hook_logits = sequence_logprobs(model, batch, args.temperature)
    path_gate.register()
    with torch.no_grad():
        gate.zero_()
    with torch.inference_mode():
        zero_scores, zero_logits = sequence_logprobs(model, batch, args.temperature)
    zero_logit_error = float((zero_logits - no_hook_logits).abs().max().item())
    zero_score_error = float((zero_scores - no_hook_scores).abs().max().item())
    vllm_scores = np.asarray(
        [comparable_vllm_logprob(row, args.temperature) for row in chosen]
    )
    hf_scores = no_hook_scores.detach().cpu().double().numpy()
    comparable = np.isfinite(vllm_scores)
    vllm_hf_abs = np.abs(vllm_scores[comparable] - hf_scores[comparable])

    with torch.no_grad():
        gate.zero_()
    autograd_scores, _ = sequence_logprobs(model, batch, args.temperature)
    auto = torch.autograd.grad(autograd_scores.mean(), gate)[0].detach().cpu().double().numpy()
    finite_rows: list[dict] = []
    best_relative_error: dict[str, float] = {}
    best_sign_match: dict[str, bool] = {}
    for path_index, path in enumerate(PATHS):
        path_errors: list[float] = []
        path_signs: list[bool] = []
        for epsilon in args.finite_difference_steps:
            with torch.no_grad():
                gate.zero_()
                gate[path_index] = epsilon
                plus = float(sequence_logprobs(model, batch, args.temperature)[0].mean().item())
                gate[path_index] = -epsilon
                minus = float(sequence_logprobs(model, batch, args.temperature)[0].mean().item())
                gate.zero_()
            finite = (plus - minus) / (2.0 * epsilon)
            denominator = max(abs(finite), abs(float(auto[path_index])), 1e-6)
            relative = abs(finite - float(auto[path_index])) / denominator
            sign_match = (
                abs(finite) < 1e-5 and abs(float(auto[path_index])) < 1e-5
            ) or math.copysign(1.0, finite) == math.copysign(1.0, float(auto[path_index]))
            finite_rows.append(
                {
                    "path": path,
                    "epsilon": epsilon,
                    "autograd": float(auto[path_index]),
                    "finite_difference": finite,
                    "relative_error": relative,
                    "sign_match": sign_match,
                }
            )
            path_errors.append(relative)
            path_signs.append(sign_match)
        best_relative_error[path] = min(path_errors)
        best_sign_match[path] = path_signs[int(np.argmin(path_errors))]
    passed = (
        zero_logit_error <= 1e-5
        and zero_score_error <= 1e-5
        and all(best_sign_match.values())
        and all(value <= args.finite_difference_rtol for value in best_relative_error.values())
    )
    return {
        "passed": passed,
        "smoke_samples": len(chosen),
        "sample_ids": [row["question_id"] for row in chosen],
        "response_token_lengths": [row["response_tokens"] for row in chosen],
        "zero_gate_max_logit_abs_error": zero_logit_error,
        "zero_gate_max_sequence_logprob_abs_error": zero_score_error,
        "vllm_vs_teacher_forced_comparable_samples": int(comparable.sum()),
        "vllm_logprob_comparison_requirement": (
            "Explicit processed_logprobs with the same temperature and no "
            "other sampling transforms; missing/legacy raw scores are excluded. "
            "Numerical gradient checks do not establish sampling-policy equivalence."
        ),
        "vllm_vs_teacher_forced_sequence_logprob_abs_error_median": (
            float(np.median(vllm_hf_abs)) if len(vllm_hf_abs) else None
        ),
        "vllm_vs_teacher_forced_sequence_logprob_abs_error_max": (
            float(vllm_hf_abs.max()) if len(vllm_hf_abs) else None
        ),
        "finite_difference_relative_tolerance": args.finite_difference_rtol,
        "best_relative_error": best_relative_error,
        "best_sign_match": best_sign_match,
        "finite_difference": finite_rows,
    }


def bootstrap_ci(values: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(Path(args.rollouts))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        grouped[row["question_id"]].append(row)
    if any(len(rows) != args.expected_rollouts for rows in grouped.values()):
        raise RuntimeError("each question must have exactly expected_rollouts records")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model = PeftModel.from_pretrained(base, args.lora_path, is_trainable=False).to("cuda")
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    modules, module_stats = build_path_modules(model, args.lora_path, args.hns_path)
    gate = torch.zeros(len(PATHS), dtype=torch.float32, device="cuda", requires_grad=True)
    path_gate = SpectralPathGate(gate, modules)
    path_gate.register()
    numerical = smoke_check(model, gate, path_gate, records, tokenizer.pad_token_id, args)
    (output / "numerical_check.json").write_text(json.dumps(numerical, indent=2) + "\n")
    write_tsv(output / "path_module_deltas.tsv", module_stats)
    print(json.dumps(numerical, indent=2), flush=True)
    if not numerical["passed"]:
        raise RuntimeError("spectral gate numerical check failed; full reward-gradient measurement aborted")

    question_rows: list[dict] = []
    rollout_score_rows: list[dict] = []
    ordered_questions = sorted(grouped, key=lambda key: grouped[key][0]["order"])
    for number, question_id in enumerate(ordered_questions, start=1):
        rows = sorted(grouped[question_id], key=lambda value: value["rollout_index"])
        rewards = torch.tensor([row["reward"] for row in rows], dtype=torch.float32, device="cuda")
        count = len(rows)
        baselines = (rewards.sum() - rewards) / (count - 1)
        advantages = rewards - baselines
        batch = encode_rollout_batch(rows, tokenizer.pad_token_id, gate.device)
        with torch.no_grad():
            gate.zero_()
        scores, _ = sequence_logprobs(model, batch, args.temperature)
        objective = (advantages * scores).mean()
        gradient = torch.autograd.grad(objective, gate)[0].detach().cpu().double().numpy()
        for path, value in zip(PATHS, gradient):
            question_rows.append(
                {
                    "question_id": question_id,
                    "source_index": rows[0]["source_index"],
                    "order": rows[0]["order"],
                    "half": rows[0]["half"],
                    "reward_mean": float(rewards.mean().item()),
                    "mixed_outcomes": bool(0 < rewards.sum().item() < count),
                    "path": path,
                    "reward_gradient": float(value),
                }
            )
        for row, score, baseline, advantage in zip(rows, scores.detach().cpu(), baselines.cpu(), advantages.cpu()):
            rollout_score_rows.append(
                {
                    "question_id": question_id,
                    "rollout_index": row["rollout_index"],
                    "half": row["half"],
                    "reward": row["reward"],
                    "loo_baseline": float(baseline.item()),
                    "advantage": float(advantage.item()),
                    "teacher_forced_sequence_logprob": float(score.item()),
                    "vllm_cumulative_logprob": row.get("cumulative_logprob_vllm"),
                    "vllm_logprobs_mode": row.get("vllm_logprobs_mode", "legacy_unspecified"),
                    "vllm_logprob_comparable": math.isfinite(
                        comparable_vllm_logprob(row, args.temperature)
                    ),
                    "response_tokens": row["response_tokens"],
                    "finish_reason": row["finish_reason"],
                }
            )
        print(f"[Gradient] {number}/{len(ordered_questions)} {question_id}", flush=True)

    write_tsv(output / "question_path_gradients.tsv", question_rows)
    write_tsv(output / "rollout_scores.tsv", rollout_score_rows)
    rng = np.random.default_rng(args.seed)
    summary_rows: list[dict] = []
    for path in PATHS:
        subset = [row for row in question_rows if row["path"] == path]
        values = np.asarray([row["reward_gradient"] for row in subset], dtype=float)
        half_a = np.asarray([row["reward_gradient"] for row in subset if row["half"] == "A"])
        half_b = np.asarray([row["reward_gradient"] for row in subset if row["half"] == "B"])
        ci_low, ci_high = bootstrap_ci(values, args.bootstrap, rng)
        absolute = np.abs(values)
        order = np.sort(absolute)[::-1]
        absolute_sum = max(float(absolute.sum()), 1e-30)
        leave_top1 = np.delete(values, int(np.argmax(absolute))).mean() if len(values) > 1 else math.nan
        summary_rows.append(
            {
                "path": path,
                "questions": len(values),
                "reward_gradient_mean": float(values.mean()),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "half_A_mean": float(half_a.mean()),
                "half_B_mean": float(half_b.mean()),
                "half_sign_agreement": bool(np.sign(half_a.mean()) == np.sign(half_b.mean())),
                "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                "top1_absolute_contribution_share": float(order[:1].sum() / absolute_sum),
                "top5_absolute_contribution_share": float(order[:5].sum() / absolute_sum),
                "leave_top1_out_mean": float(leave_top1),
                "nonzero_question_contributions": int(np.count_nonzero(values)),
            }
        )
    write_tsv(output / "path_signal_summary.tsv", summary_rows)

    rewards = np.asarray([row["reward"] for row in records], dtype=float)
    mixed = sum(0 < sum(row["reward"] for row in grouped[key]) < len(grouped[key]) for key in grouped)
    metadata = {
        "status": "complete",
        "definition": "mean_x mean_k (R_xk - mean_{l!=k} R_xl) d log pi_T(y_xk|x) / d lambda_path at lambda=0",
        "path_parameterization": "sigma(lambda)=sigma_LoRA+lambda*(sigma_target-sigma_LoRA)",
        "paths": {
            "full_hns": "observed aligned HNS spectrum",
            "head_only": "min(sigma_LoRA, sigma_HNS) per direction",
            "scalar_shrink": "per-module scalar times LoRA spectrum matching HNS module Frobenius norm",
        },
        "temperature": args.temperature,
        "teacher_forcing_dtype": args.dtype,
        "questions": len(grouped),
        "rollouts": len(records),
        "rollouts_per_question": args.expected_rollouts,
        "reward_mean": float(rewards.mean()),
        "mixed_outcome_questions": mixed,
        "all_correct_questions": sum(sum(r["reward"] for r in rows) == len(rows) for rows in grouped.values()),
        "all_wrong_questions": sum(sum(r["reward"] for r in rows) == 0 for rows in grouped.values()),
        "truncated_rollouts": sum(row["finish_reason"] == "length" for row in records),
        "modules": len(modules),
        "numerical_check": numerical,
        "summary": summary_rows,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    path_gate.remove()


if __name__ == "__main__":
    main()
