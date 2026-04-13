from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        text = re.sub(r"\n```$", "", text.strip())
    return text.strip()


def _pick_tokenizer_source(base_model: str, adapter_dir: str | None) -> str:
    if adapter_dir is None:
        return base_model
    p = Path(adapter_dir)
    if (p / "tokenizer.json").exists() or (p / "tokenizer.model").exists():
        return adapter_dir
    return base_model


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any


def _active_adapter_names(model: Any) -> list[str]:
    try:
        active = model.active_adapters
        if callable(active):
            active = active()
    except Exception:
        active = None

    if active is None:
        active = getattr(model, "active_adapter", None)
    if active is None:
        return []
    if isinstance(active, str):
        return [active]
    return [str(name) for name in active]


def load_transformers_model(
    *,
    base_model: str,
    adapter_dir: str | None,
    dtype: str = "auto",
    device_map: str | dict[str, int] | None = "auto",
) -> LoadedModel:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        torch_dtype = None

    tokenizer = AutoTokenizer.from_pretrained(_pick_tokenizer_source(base_model, adapter_dir), use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )

    if adapter_dir is not None:
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError(
                f"Adapter requested but peft is missing: {exc}\nInstall: pip install -U peft"
            ) from exc
        model = PeftModel.from_pretrained(model, adapter_dir)
        if not getattr(model, "peft_config", None):
            raise RuntimeError(f"Adapter load failed: no PEFT config found for {adapter_dir}")
        active_adapters = _active_adapter_names(model)
        if not active_adapters:
            raise RuntimeError(f"Adapter load failed: no active adapter after loading {adapter_dir}")

    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer)


def _model_input_device(model: Any):
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _truncate_at_stop_strings(text: str, stop_strings: list[str] | None) -> str:
    if not text or not stop_strings:
        return text
    cut_idx: int | None = None
    for marker in stop_strings:
        idx = text.find(marker)
        if idx == -1:
            continue
        if cut_idx is None or idx < cut_idx:
            cut_idx = idx
    if cut_idx is None:
        return text
    return text[:cut_idx].rstrip()


def generate_greedy(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    stop_strings: list[str] | None = None,
) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    device = _model_input_device(model)
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return _truncate_at_stop_strings(text, stop_strings)


def generate_greedy_vllm_batch(
    *,
    base_model: str,
    prompts: list[str],
    max_new_tokens: int,
    adapter_dir: str | None = None,
    tensor_parallel_size: int = 1,
    stop_strings: list[str] | None = None,
) -> list[str]:
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise RuntimeError(
            f"vLLM requested but not available: {exc}\nInstall: pip install -U vllm"
        ) from exc

    lora_request = None
    if adapter_dir is not None:
        try:
            from vllm.lora.request import LoRARequest
        except Exception as exc:
            raise RuntimeError(f"vLLM LoRA support not available: {exc}") from exc
        lora_request = LoRARequest("adapter", 1, adapter_dir)

    if not prompts:
        return []

    llm = LLM(
        model=base_model,
        tensor_parallel_size=tensor_parallel_size,
        enable_lora=adapter_dir is not None,
        max_lora_rank=256,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop=stop_strings or None,
    )
    outputs = llm.generate(prompts, params, lora_request=lora_request)
    if not outputs:
        return ["" for _ in prompts]

    texts: list[str] = []
    for req_out in outputs:
        if not getattr(req_out, "outputs", None):
            texts.append("")
        else:
            texts.append(_truncate_at_stop_strings(req_out.outputs[0].text.strip(), stop_strings))

    if len(texts) < len(prompts):
        texts.extend([""] * (len(prompts) - len(texts)))
    elif len(texts) > len(prompts):
        texts = texts[: len(prompts)]

    return texts


def generate_greedy_vllm(
    *,
    base_model: str,
    prompt: str,
    max_new_tokens: int,
    adapter_dir: str | None = None,
    tensor_parallel_size: int = 1,
    stop_strings: list[str] | None = None,
) -> str:
    outputs = generate_greedy_vllm_batch(
        base_model=base_model,
        prompts=[prompt],
        max_new_tokens=max_new_tokens,
        adapter_dir=adapter_dir,
        tensor_parallel_size=tensor_parallel_size,
        stop_strings=stop_strings,
    )
    return outputs[0] if outputs else ""


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
