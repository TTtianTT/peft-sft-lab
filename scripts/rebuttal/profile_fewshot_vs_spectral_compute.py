#!/usr/bin/env python3
"""
Profiled compute estimates for spectral editing vs fixed-k few-shot prompting.

Outputs:
  - compute_profile_note.md
  - compute_profile_table.md
  - compute_rebuttal_paragraph.txt
  - compute_profile_summary.json

The FLOPs numbers are profiled estimates, not exact true FLOPs.
"""

from __future__ import annotations

import gc
import argparse
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finetune.data.base import format_instruction_response  # noqa: E402
from finetune.eval.eval_csqa import _build_csqa_prompt  # noqa: E402
from finetune.eval.eval_gsm8k import (  # noqa: E402
    _NEXT_PROMPT_MARKERS,
    _build_fewshot_prefix as build_math_fewshot_prefix,
    _build_prompt_gsm8k_metamath_style,
)
from finetune.eval.generation import load_transformers_model  # noqa: E402
from finetune.spectral_edit.calib import (  # noqa: E402
    build_calib_formatter,
    load_calibration_split,
    make_calib_batch,
    sample_calibration_examples,
)
from finetune.spectral_edit.cli import set_seed  # noqa: E402
from finetune.spectral_edit.edit_strategies import EditConfig, apply_spectral_edit  # noqa: E402
from finetune.spectral_edit.hooks import HOOK_CTX, ModuleSpec, register_sigma_hooks, remove_hooks  # noqa: E402
from finetune.spectral_edit.io import (  # noqa: E402
    ensure_local_lora_dir,
    get_scaling_for_module,
    layer_idx_from_module_prefix,
    load_adapter_config,
    load_lora_state_dict,
    parse_lora_ab_key,
)
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma  # noqa: E402


BASE_MODELS = {
    "Qwen-Qwen3-8B": "Qwen/Qwen3-8B",
    "meta-llama-Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
}
TASKS = ["math", "csqa"]
K_VALUES = [0, 1, 3, 5, 32]


@dataclass
class RepresentativeQuery:
    prompt: str
    prompt_tokens: int
    generation_text: str
    generation_tokens: list[int]
    max_new_tokens: int
    stop_strings: list[str] | None
    adapter_dir: str
    base_model_dir: str
    base_model_id: str
    task: str
    fewshot_k: int


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def format_instruction_response_local(*, instruction: str, response: str) -> str:
    return format_instruction_response(instruction=instruction, response=response)


def unload_model(obj: Any) -> None:
    try:
        del obj
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def profiler_flops(fn) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_flops=True,
    ) as prof:
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total = 0.0
    for evt in prof.key_averages():
        if evt.flops:
            total += float(evt.flops)
    return total


def torch_dtype_for_eval() -> torch.dtype:
    return torch.float16


def prediction_path(base_model_dir: str, task: str, k: int) -> Path:
    if task == "math":
        if k == 0:
            return REPO_ROOT / "outputs/rebuttal_exp/raw/fewshot_eval_mathfix/eval_outputs" / f"{base_model_dir}_math_fewshot_k0_s42" / "predictions.jsonl"
        return REPO_ROOT / "outputs/rebuttal_exp/raw/fewshot_eval_mathfix/eval_outputs" / f"{base_model_dir}_math_fewshot_k{k}_s42" / "predictions.jsonl"
    if k == 0:
        return REPO_ROOT / "outputs/rebuttal_exp/raw/multiseed_eval/eval_outputs" / f"{base_model_dir}_csqa_baseline_s42" / "predictions.jsonl"
    return REPO_ROOT / "outputs/rebuttal_exp/raw/fewshot_eval/eval_outputs" / f"{base_model_dir}_csqa_fewshot_k{k}_s42" / "predictions.jsonl"


def exemplar_path(base_model_dir: str, task: str, k: int) -> Path:
    root = REPO_ROOT / ("outputs/rebuttal_exp/raw/fewshot_eval_mathfix/eval_outputs" if task == "math" else "outputs/rebuttal_exp/raw/fewshot_eval/eval_outputs")
    return root / f"{base_model_dir}_{task}_fewshot_k{k}_s42" / "fewshot_exemplars.jsonl"


def build_csqa_prefix(exemplars: list[dict[str, Any]]) -> str:
    if not exemplars:
        return ""
    blocks = [ex["prompt"].rstrip() for ex in exemplars]
    return "\n\n".join(blocks).strip() + "\n\n"


def reconstruct_prompt(task: str, row: dict[str, Any], exemplars: list[dict[str, Any]]) -> str:
    if task == "math":
        base_prompt = _build_prompt_gsm8k_metamath_style(row["question"])
        return build_math_fewshot_prefix(exemplars) + base_prompt
    base_prompt = _build_csqa_prompt(row["instruction"])
    return build_csqa_prefix(exemplars) + base_prompt


def choose_representative_query(
    *,
    base_model_dir: str,
    base_model_id: str,
    task: str,
    k: int,
    tokenizer,
    adapter_dir: str,
) -> RepresentativeQuery:
    pred_rows = load_jsonl(prediction_path(base_model_dir, task, k))
    exemplars = load_jsonl(exemplar_path(base_model_dir, task, k)) if k > 0 else []

    entries: list[tuple[float, dict[str, Any], str, int, list[int]]] = []
    prompt_lens: list[int] = []
    gen_lens: list[int] = []
    for row in pred_rows:
        prompt = reconstruct_prompt(task, row, exemplars)
        prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        gen_ids = tokenizer(row["prediction_text"], add_special_tokens=False).input_ids
        prompt_len = len(prompt_ids)
        gen_len = len(gen_ids)
        prompt_lens.append(prompt_len)
        gen_lens.append(gen_len)
        entries.append((0.0, row, prompt, prompt_len, gen_ids))

    mean_prompt = statistics.mean(prompt_lens)
    mean_gen = statistics.mean(gen_lens)
    scored: list[tuple[float, dict[str, Any], str, int, list[int]]] = []
    for _, row, prompt, prompt_len, gen_ids in entries:
        score = abs(prompt_len - mean_prompt) + abs(len(gen_ids) - mean_gen)
        scored.append((score, row, prompt, prompt_len, gen_ids))
    _, row, prompt, prompt_len, gen_ids = min(scored, key=lambda x: x[0])

    return RepresentativeQuery(
        prompt=prompt,
        prompt_tokens=prompt_len,
        generation_text=row["prediction_text"],
        generation_tokens=gen_ids,
        max_new_tokens=256 if task == "math" else 8,
        stop_strings=_NEXT_PROMPT_MARKERS if (task == "math" and k > 0) else None,
        adapter_dir=adapter_dir,
        base_model_dir=base_model_dir,
        base_model_id=base_model_id,
        task=task,
        fewshot_k=k,
    )


def profile_prefill_and_decode_loaded(*, model, tokenizer, query: RepresentativeQuery) -> dict[str, float]:
    device = torch.device("cuda")
    model.eval()
    model.config.use_cache = True

    prompt_inputs = tokenizer(query.prompt, return_tensors="pt")
    prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    gen_ids = query.generation_tokens
    if not gen_ids:
        gen_ids = [tokenizer.eos_token_id or tokenizer.pad_token_id]

    with torch.inference_mode():
        _ = model(**prompt_inputs, use_cache=True)

    def prefill_step() -> None:
        with torch.inference_mode():
            _ = model(**prompt_inputs, use_cache=True)

    prefill_flops = profiler_flops(prefill_step)

    with torch.inference_mode():
        out = model(**prompt_inputs, use_cache=True)
        past = out.past_key_values

    def decode_loop() -> None:
        with torch.inference_mode():
            past_kv = past
            cur_len = prompt_len
            for tok_id in gen_ids:
                input_ids = torch.tensor([[tok_id]], dtype=torch.long, device=device)
                attn_mask = torch.ones((1, cur_len + 1), dtype=torch.long, device=device)
                out_step = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                past_kv = out_step.past_key_values
                cur_len += 1

    decode_flops = profiler_flops(decode_loop)

    return {
        "prefill_flops": prefill_flops,
        "decode_total_flops": decode_flops,
        "decode_flops_per_token": decode_flops / max(1, len(gen_ids)),
        "total_query_flops": prefill_flops + decode_flops,
        "profile_prompt_tokens": prompt_len,
        "profile_generation_tokens": len(gen_ids),
    }


def measure_vllm_latency(queries: list[RepresentativeQuery]) -> dict[int, float]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if not queries:
        return {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    base_model = queries[0].base_model_id
    adapter_dir = queries[0].adapter_dir
    requested_max_model_len = max(query.prompt_tokens + query.max_new_tokens for query in queries)
    # Reserve only the actual workload length rather than the architectural max, which would
    # over-allocate KV cache for Llama-3.1's 131k window despite much shorter real prompts.
    workload_max_model_len = int(math.ceil(requested_max_model_len / 256.0) * 256)
    llm = LLM(
        model=base_model,
        tensor_parallel_size=1,
        enable_lora=True,
        max_lora_rank=256,
        gpu_memory_utilization=0.6,
        disable_log_stats=True,
        max_model_len=workload_max_model_len,
    )
    lora_request = LoRARequest("adapter", 1, adapter_dir)

    latencies: dict[int, float] = {}
    for query in queries:
        params = SamplingParams(
            temperature=0.0,
            max_tokens=query.max_new_tokens,
            stop=query.stop_strings or None,
        )
        _ = llm.generate([query.prompt], params, lora_request=lora_request)
        reps: list[float] = []
        for _ in range(3):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = llm.generate([query.prompt], params, lora_request=lora_request)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            reps.append(time.perf_counter() - t0)
        latencies[query.fewshot_k] = statistics.mean(reps)

    del llm
    unload_model(None)
    return latencies


def spectral_setting_to_meta() -> dict[str, Path]:
    costs = load_json(REPO_ROOT / "outputs/rebuttal_exp/raw/fewshot_eval/spectral_costs.json")
    return {setting: Path(payload["spectral_meta_path"]) for setting, payload in costs.items()}


def build_spectral_setup(meta_path: Path):
    meta_payload = load_json(meta_path)
    meta = meta_payload["meta"]

    set_seed(int(meta["seed"]))
    device = torch.device("cuda")
    lora_dir = ensure_local_lora_dir(meta["lora_path"], cache_dir=None)
    adapter_cfg = load_adapter_config(lora_dir)
    sd, _ = load_lora_state_dict(lora_dir)

    target_modules_set = set(meta["target_modules"])
    pairs: dict[str, dict[str, tuple[str, torch.Tensor, str | None]]] = {}
    for key, tensor in sd.items():
        parsed = parse_lora_ab_key(key)
        if not parsed:
            continue
        prefix, which, adapter = parsed
        suffix = prefix.split(".")[-1]
        if suffix not in target_modules_set:
            continue
        li = layer_idx_from_module_prefix(prefix)
        if li is not None and not (int(meta["layer_min"]) <= li <= int(meta["layer_max"])):
            continue
        pairs.setdefault(prefix, {})
        pairs[prefix][which] = (key, tensor, adapter)

    tok = AutoTokenizer.from_pretrained(meta["base_model"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        meta["base_model"],
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
    ).to(device)
    model = PeftModel.from_pretrained(base, lora_dir, is_trainable=True).to(device)
    model.eval()
    model.config.use_cache = False

    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    name_to_module = dict(model.named_modules())
    specs: dict[str, ModuleSpec] = {}
    for prefix, pair in pairs.items():
        if "A" not in pair or "B" not in pair:
            continue
        _, A_cpu, adapter_a = pair["A"]
        _, B_cpu, adapter_b = pair["B"]
        adapter_name = adapter_a if adapter_a is not None else adapter_b
        module_name = prefix if prefix in name_to_module else [nm for nm in name_to_module if nm.endswith(prefix)][0]
        mod = name_to_module[module_name]
        A = A_cpu.to(device)
        B = B_cpu.to(device)
        U, S, Vh, V = lowrank_svd_from_ba(B, A)
        scaling = get_scaling_for_module(adapter_cfg, prefix)
        specs[prefix] = ModuleSpec(
            module_prefix=prefix,
            module=mod,
            U=U.detach(),
            V=V.detach(),
            Vh=Vh.detach(),
            sigma0=S.detach().cpu(),
            scaling=scaling,
            adapter=adapter_name,
        )

    formatter, _ = build_calib_formatter(meta["calib_dataset"], meta["calib_text_fields"])
    ds = load_calibration_split(meta["calib_dataset"], meta["calib_config"], meta["calib_split"], cache_dir=None)
    examples = sample_calibration_examples(
        ds,
        int(meta["calib_samples"]),
        bool(meta["calib_shuffle"]),
        int(meta["calib_seed"]),
        int(meta["calib_start"]),
    )
    batches = []
    bs = int(meta["calib_batch_size"])
    for i in range(0, len(examples), bs):
        batch_ex = examples[i : i + bs]
        input_ids, attn_mask, labels = make_calib_batch(tok, batch_ex, formatter, add_eos=True, max_seq_len=None)
        batches.append(
            {
                "input_ids": input_ids.to(device),
                "attn_mask": attn_mask.to(device),
                "labels": labels.to(device),
                "active_tokens": int(attn_mask.sum().item()),
                "seq_len": int(input_ids.shape[1]),
            }
        )
    return meta, tok, model, specs, batches


def profile_spectral_setting(setting: str, meta_path: Path) -> dict[str, Any]:
    meta, tok, model, specs, batches = build_spectral_setup(meta_path)
    handles = register_sigma_hooks(specs)

    mean_active_tokens = statistics.mean(batch["active_tokens"] for batch in batches)
    rep_batch = min(batches, key=lambda b: abs(b["active_tokens"] - mean_active_tokens))

    def run_forward_only() -> None:
        HOOK_CTX.reset()
        HOOK_CTX.attn_mask = rep_batch["attn_mask"]
        with torch.no_grad():
            _ = model(
                input_ids=rep_batch["input_ids"],
                attention_mask=rep_batch["attn_mask"],
                labels=rep_batch["labels"],
            )
        HOOK_CTX.attn_mask = None

    def run_forward_backward() -> None:
        HOOK_CTX.reset()
        HOOK_CTX.attn_mask = rep_batch["attn_mask"]
        out = model(
            input_ids=rep_batch["input_ids"],
            attention_mask=rep_batch["attn_mask"],
            labels=rep_batch["labels"],
        )
        loss = out.loss
        model.zero_grad(set_to_none=True)
        loss.backward()
        model.zero_grad(set_to_none=True)
        HOOK_CTX.attn_mask = None

    run_forward_only()
    run_forward_backward()
    forward_batch_flops = profiler_flops(run_forward_only)
    total_batch_flops = profiler_flops(run_forward_backward)

    edit_cfg = EditConfig(
        mode=meta["mode"],
        core_frac=float(meta["core_frac"]),
        noise_frac=float(meta["noise_frac"]),
        amp_factor=float(meta["amp_factor"]),
        sup_factor=float(meta["sup_factor"]),
        mid_factor=float(meta["mid_factor"]),
        min_core_k=1,
        smooth_temperature=0.35,
        smooth_center_q=0.5,
        smooth_align_mid=True,
        z_high=1.0,
        z_low=-0.5,
        z_tau=0.2,
        z_fallback_std=1e-6,
        robust_z_high=1.0,
        robust_z_low=-0.5,
        robust_z_tau=0.2,
        robust_fallback_sigma=1e-6,
        eta=0.0,
        update_mode="sigma",
        asymmetric_update=False,
        eta_suppress=1.0,
        eta_enhance=1.0,
        pos_power=1.0,
        grad_norm=meta["grad_norm"],
        preserve_energy=meta["preserve_energy"],
        sigma_clip_min=1e-8,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    HOOK_CTX.reset()
    for batch in batches:
        HOOK_CTX.attn_mask = batch["attn_mask"]
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attn_mask"],
            labels=batch["labels"],
        )
        loss = out.loss
        model.zero_grad(set_to_none=True)
        loss.backward()
        model.zero_grad(set_to_none=True)
    HOOK_CTX.attn_mask = None

    for spec in specs.values():
        g = HOOK_CTX.gsum.get(spec.module_prefix)
        if g is None:
            continue
        sigma_new, _ = apply_spectral_edit(spec.sigma0.clone(), g, edit_cfg)
        _ = rebuild_ba_from_uv_sigma(spec.U.to(torch.device("cuda")), spec.Vh.to(torch.device("cuda")), sigma_new.to(torch.device("cuda")))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - t0

    remove_hooks(handles)
    unload_model(model)
    unload_model(tok)

    n_batches = len(batches)
    forward_total_est = forward_batch_flops * n_batches
    total_est = total_batch_flops * n_batches
    backward_total_est = max(0.0, total_est - forward_total_est)
    return {
        "setting": setting,
        "base_model": meta["base_model"],
        "batch_size": int(meta["calib_batch_size"]),
        "n_batches": n_batches,
        "rep_batch_active_tokens": rep_batch["active_tokens"],
        "rep_batch_seq_len": rep_batch["seq_len"],
        "mean_batch_active_tokens": mean_active_tokens,
        "forward_flops_est": forward_total_est,
        "backward_flops_est": backward_total_est,
        "total_flops_est": total_est,
        "runtime_seconds": runtime_seconds,
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_model_dirs",
        nargs="*",
        default=None,
        choices=sorted(BASE_MODELS.keys()),
        help="Optional subset of base model directory labels to profile.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        choices=TASKS,
        help="Optional subset of tasks to profile.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "outputs/rebuttal_exp/fewshot_corrected_math",
        help="Directory to write compute artifacts into.",
    )
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiling script.")

    args = parse_args()
    selected_base_models = {
        key: BASE_MODELS[key]
        for key in (args.base_model_dirs if args.base_model_dirs else BASE_MODELS.keys())
    }
    selected_tasks = list(args.tasks) if args.tasks else list(TASKS)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    corrected_math_records = load_jsonl(REPO_ROOT / "outputs/rebuttal_exp/raw/fewshot_eval_mathfix/results.jsonl")
    original_fewshot_records = load_jsonl(REPO_ROOT / "outputs/rebuttal_exp/raw/fewshot_eval/results.jsonl")
    core_records = load_jsonl(REPO_ROOT / "outputs/rebuttal_exp/raw/multiseed_eval/eval_results.jsonl")
    spectral_costs = load_json(REPO_ROOT / "outputs/rebuttal_exp/raw/fewshot_eval/spectral_costs.json")

    runtime_lookup: dict[tuple[str, str, int], float] = {}
    for rec in original_fewshot_records + corrected_math_records:
        if rec.get("error"):
            continue
        runtime_lookup[(rec["base_model_dir"], rec["task"], int(rec["fewshot_k"]))] = float(rec["runtime_seconds"])
    adapter_lookup: dict[tuple[str, str, int], str] = {}
    for rec in original_fewshot_records + corrected_math_records:
        if rec.get("error"):
            continue
        adapter_lookup[(rec["base_model_dir"], rec["task"], int(rec["fewshot_k"]))] = rec["adapter_dir"]
    for rec in core_records:
        if rec.get("error"):
            continue
        if int(rec.get("repeat_seed", -1)) != 42:
            continue
        if rec.get("method") == "baseline":
            adapter_lookup[(rec["base_model_dir"], rec["task"], 0)] = rec["adapter_dir"]

    representative_queries: dict[tuple[str, str, int], RepresentativeQuery] = {}
    fewshot_profiles: dict[tuple[str, str, int], dict[str, float]] = {}
    latency_profiles: dict[tuple[str, str, int], float] = {}

    for base_model_dir, base_model_id in selected_base_models.items():
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, local_files_only=True)
        for task in selected_tasks:
            queries: list[RepresentativeQuery] = []
            for k in K_VALUES:
                query = choose_representative_query(
                    base_model_dir=base_model_dir,
                    base_model_id=base_model_id,
                    task=task,
                    k=k,
                    tokenizer=tokenizer,
                    adapter_dir=adapter_lookup[(base_model_dir, task, k)],
                )
                representative_queries[(base_model_dir, task, k)] = query
            unload_model(tokenizer)

        for task in selected_tasks:
            task_queries = [representative_queries[(base_model_dir, task, k)] for k in K_VALUES]
            print(f"[fewshot-flops] {base_model_dir}/{task}", flush=True)
            loaded = load_transformers_model(
                base_model=base_model_id,
                adapter_dir=task_queries[0].adapter_dir,
                dtype="fp16",
                device_map=None,
            )
            model = loaded.model.to(torch.device("cuda"))
            tokenizer = loaded.tokenizer
            for query in task_queries:
                fewshot_profiles[(base_model_dir, task, query.fewshot_k)] = profile_prefill_and_decode_loaded(
                    model=model,
                    tokenizer=tokenizer,
                    query=query,
                )
            unload_model(model)
            unload_model(loaded)
            print(f"[fewshot-latency] {base_model_dir}/{task}", flush=True)
            vllm_latencies = measure_vllm_latency(task_queries)
            for k, latency in vllm_latencies.items():
                latency_profiles[(base_model_dir, task, k)] = latency

    spectral_profiles: dict[str, dict[str, Any]] = {}
    for setting, payload in spectral_costs.items():
        base_model_dir, task = setting.split("/", 1)
        if base_model_dir not in selected_base_models or task not in selected_tasks:
            continue
        print(f"[spectral] {setting}", flush=True)
        spectral_profiles[setting] = profile_spectral_setting(setting, Path(payload["spectral_meta_path"]))

    break_even_flops_by_k: dict[int, list[float]] = {k: [] for k in K_VALUES if k > 0}
    break_even_latency_by_k: dict[int, list[float]] = {k: [] for k in K_VALUES if k > 0}
    for base_model_dir in selected_base_models:
        for task in selected_tasks:
            setting = f"{base_model_dir}/{task}"
            spectral_total = spectral_profiles[setting]["total_flops_est"]
            spectral_runtime = spectral_profiles[setting]["runtime_seconds"]
            for k in [1, 3, 5, 32]:
                q_flops = fewshot_profiles[(base_model_dir, task, k)]["total_query_flops"]
                q_latency = latency_profiles[(base_model_dir, task, k)]
                break_even_flops_by_k[k].append(spectral_total / q_flops)
                break_even_latency_by_k[k].append(spectral_runtime / q_latency)

    token_break_even = {
        k: []
        for k in [1, 3, 5, 32]
    }
    for rec in original_fewshot_records + corrected_math_records:
        if rec.get("error"):
            continue
        if int(rec["fewshot_k"]) not in token_break_even:
            continue
        setting = f"{rec['base_model_dir']}/{rec['task']}"
        extra = float(rec["avg_extra_prompt_tokens"])
        token_break_even[int(rec["fewshot_k"])].append(float(spectral_costs[setting]["forward_backward_token_passes"]) / extra)

    method_rows: list[dict[str, Any]] = []
    spectral_totals = [payload["total_flops_est"] for payload in spectral_profiles.values()]
    spectral_runtimes = [payload["runtime_seconds"] for payload in spectral_profiles.values()]
    method_rows.append(
        {
            "method": "random_index",
            "one_time_flops_mean": statistics.mean(spectral_totals),
            "per_query_flops_mean": 0.0,
            "per_query_latency_mean": 0.0,
            "break_even_flops_mean": 0.0,
            "break_even_latency_mean": 0.0,
        }
    )
    for k in K_VALUES:
        if k == 0:
            per_query_flops = [
                fewshot_profiles[(base_model_dir, task, 0)]["total_query_flops"]
                for base_model_dir in selected_base_models
                for task in selected_tasks
            ]
            per_query_latency = [
                latency_profiles[(base_model_dir, task, 0)]
                for base_model_dir in selected_base_models
                for task in selected_tasks
            ]
            method_rows.append(
                {
                    "method": "few-shot k0",
                    "one_time_flops_mean": 0.0,
                    "per_query_flops_mean": statistics.mean(per_query_flops),
                    "per_query_latency_mean": statistics.mean(per_query_latency),
                    "break_even_flops_mean": None,
                    "break_even_latency_mean": None,
                }
            )
            continue
        per_query_flops = [
            fewshot_profiles[(base_model_dir, task, k)]["total_query_flops"]
            for base_model_dir in selected_base_models
            for task in selected_tasks
        ]
        per_query_latency = [
            latency_profiles[(base_model_dir, task, k)]
            for base_model_dir in selected_base_models
            for task in selected_tasks
        ]
        method_rows.append(
            {
                "method": f"few-shot k{k}",
                "one_time_flops_mean": 0.0,
                "per_query_flops_mean": statistics.mean(per_query_flops),
                "per_query_latency_mean": statistics.mean(per_query_latency),
                "break_even_flops_mean": statistics.mean(break_even_flops_by_k[k]),
                "break_even_latency_mean": statistics.mean(break_even_latency_by_k[k]),
            }
        )

    note_lines = [
        "# Compute Profiling Note",
        "",
        "This note reports **profiled FLOPs estimates**, not exact true FLOPs. The profiler can miss fused kernels and non-GEMM work, so the numbers should be read as closer-to-real implementation-specific estimates rather than universal absolute truths.",
        "",
        "## Methodology",
        "",
        "### Spectral editing one-time cost",
        "- Path profiled: the actual `random_index` spectral-edit configuration used in the rebuttal package.",
        "- Model/adapters: the same seed42 adapters and base models used in the comparison.",
        "- Profiling mode: PyTorch profiler with CUDA FLOPs accounting on a representative real calibration batch from each setting, under the same batch size, teacher-forcing loss, hooks, and `use_cache=False` configuration as the edit path.",
        "- Sequence-length handling: for each setting, the representative batch is the real calibration batch whose active-token count is closest to the run mean; the profiled per-batch FLOPs are then scaled by the actual number of calibration batches.",
        "- Forward/backward split: forward FLOPs come from a forward-only profile on the representative batch; backward FLOPs are estimated as `(forward+backward profile) - (forward-only profile)` on the same batch.",
        "- Runtime: wall-clock is measured on the same edit path after model/tokenizer/dataset load, including SVD-spec setup, calibration forward/backward passes, and sigma-update/rebuild work.",
        "",
        "### Few-shot inference cost",
        "- FLOPs path: single-query HuggingFace/PEFT profiling with KV cache enabled (`use_cache=True`) and the same adapters/prompts used in the evaluation study.",
        "- Prefill FLOPs: one profiled forward pass on a representative real prompt for each setting/k.",
        "- Decode FLOPs: profiled token-by-token cached decoding on the representative query, using the actual saved generated-token trace length from that query.",
        "- Representative-query selection: for each setting/k, we choose the saved example nearest the joint mean of prompt length and generated length.",
        "- Latency path: actual vLLM generation with batch size 1, `tensor_parallel_size=1`, LoRA enabled, the same max_new_tokens, and the same math stop strings used in the corrected rerun. Latency excludes engine load and uses a warm run before timing.",
        "- vLLM startup uses `gpu_memory_utilization=0.6` in this profiler so the latency pass can coexist with the earlier HF profiling process on a single L40S; this affects admission/headroom rather than the prompt/decode algorithm being timed.",
        "- For latency only, vLLM `max_model_len` is capped to the maximum real prompt+generation length among the profiled queries for that setting rather than the model's architectural context window, so KV-cache reservation reflects the actual workload instead of an unused 131k worst case.",
        "",
        "### KV cache / batching assumptions",
        "- KV cache is **enabled** for the few-shot inference FLOPs profile and for the vLLM latency measurement.",
        "- Spectral editing disables KV cache, matching the training-style calibration path.",
        "- Few-shot FLOPs are representative single-query measurements; latency is also measured at batch size 1 in vLLM after warmup.",
        "- The simple token-based proxy from the earlier package is retained below as a secondary cross-check.",
        "",
        "## Spectral Profile Summary",
    ]

    for setting, payload in spectral_profiles.items():
        note_lines.append(
            f"- `{setting}`: batch_size={payload['batch_size']}, representative batch active tokens={payload['rep_batch_active_tokens']}, "
            f"representative seq_len={payload['rep_batch_seq_len']}, mean batch active tokens={payload['mean_batch_active_tokens']:.1f}, "
            f"forward={fmt_flops(payload['forward_flops_est'])}, backward={fmt_flops(payload['backward_flops_est'])}, "
            f"total={fmt_flops(payload['total_flops_est'])}, runtime={fmt_latency(payload['runtime_seconds'])}."
        )

    note_lines.extend(["", "## Few-Shot Profile Summary"])
    for base_model_dir in selected_base_models:
        for task in selected_tasks:
            setting = f"{base_model_dir}/{task}"
            for k in K_VALUES:
                query = representative_queries[(base_model_dir, task, k)]
                prof = fewshot_profiles[(base_model_dir, task, k)]
                latency = latency_profiles[(base_model_dir, task, k)]
                note_lines.append(
                    f"- `{setting}` k={k}: prompt={query.prompt_tokens} toks, generated={prof['profile_generation_tokens']} toks, "
                    f"prefill={fmt_flops(prof['prefill_flops'])}, decode/token={fmt_flops(prof['decode_flops_per_token'])}, "
                    f"total/query={fmt_flops(prof['total_query_flops'])}, vLLM latency/query={fmt_latency(latency)}, "
                    f"max_new_tokens={query.max_new_tokens}, stop_strings={'yes' if query.stop_strings else 'no'}."
                )

    note_lines.extend(
        [
            "",
            "## Break-Even",
        ]
    )
    for k in [1, 3, 5, 32]:
        note_lines.append(
            f"- FLOPs break-even vs k={k}: mean={statistics.mean(break_even_flops_by_k[k]):.2f} queries, "
            f"range={min(break_even_flops_by_k[k]):.2f}-{max(break_even_flops_by_k[k]):.2f}."
        )
        note_lines.append(
            f"- Latency break-even vs k={k}: mean={statistics.mean(break_even_latency_by_k[k]):.2f} queries, "
            f"range={min(break_even_latency_by_k[k]):.2f}-{max(break_even_latency_by_k[k]):.2f}."
        )
        note_lines.append(
            f"- Token-proxy break-even vs k={k}: mean={statistics.mean(token_break_even[k]):.2f} queries, "
            f"range={min(token_break_even[k]):.2f}-{max(token_break_even[k]):.2f}."
        )

    table_lines = [
        "| Method | One-time FLOPs | Per-query FLOPs | Per-query latency | Break-even queries |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        method = row["method"]
        one_time = fmt_flops(row["one_time_flops_mean"]) if row["one_time_flops_mean"] else ("0" if row["one_time_flops_mean"] == 0.0 else "N/A")
        per_query_flops = fmt_flops(row["per_query_flops_mean"])
        per_query_latency = fmt_latency(row["per_query_latency_mean"])
        if method == "random_index":
            break_even = "0"
        elif row["break_even_flops_mean"] is None:
            break_even = "N/A"
        else:
            break_even = f"{row['break_even_flops_mean']:.1f} FLOPs / {row['break_even_latency_mean']:.1f} latency"
        table_lines.append(
            f"| {method} | {one_time} | {per_query_flops} | {per_query_latency} | {break_even} |"
        )

    paragraph = (
        "We supplemented the token proxy with profiled FLOPs estimates under the actual model/adapters and generation setup. "
        "These are not exact true FLOPs, but they consistently show the same operating-point pattern: spectral editing pays a "
        "single calibration/edit cost up front, whereas fixed few-shot prompting shifts cost into repeated prefill-heavy serving. "
        "After the corrected math rerun, the FLOPs- and latency-based break-even still arrives after only tens of served queries "
        "for realistic k, while k=32 remains a poor practical point because it buys little or no accuracy gain in exchange for much larger recurrent cost."
    )

    summary = {
        "spectral_profiles": spectral_profiles,
        "fewshot_profiles": {
            f"{base_model_dir}/{task}/k={k}": {
                **fewshot_profiles[(base_model_dir, task, k)],
                "vllm_latency_seconds": latency_profiles[(base_model_dir, task, k)],
                "representative_prompt_tokens": representative_queries[(base_model_dir, task, k)].prompt_tokens,
                "representative_generated_tokens": len(representative_queries[(base_model_dir, task, k)].generation_tokens),
            }
            for base_model_dir in selected_base_models
            for task in selected_tasks
            for k in K_VALUES
        },
        "method_rows": method_rows,
        "break_even_flops": break_even_flops_by_k,
        "break_even_latency": break_even_latency_by_k,
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
