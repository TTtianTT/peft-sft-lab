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

    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer)


def _model_input_device(model: Any):
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def generate_greedy(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
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
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_greedy_vllm(
    *,
    base_model: str,
    prompt: str,
    max_new_tokens: int,
    adapter_dir: str | None = None,
    tensor_parallel_size: int = 1,
) -> str:
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

    llm = LLM(
        model=base_model,
        tensor_parallel_size=tensor_parallel_size,
        enable_lora=adapter_dir is not None,
        max_lora_rank=256,
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate([prompt], params, lora_request=lora_request)
    if not outputs:
        return ""
    req_out = outputs[0]
    if not getattr(req_out, "outputs", None):
        return ""
    return req_out.outputs[0].text.strip()


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

