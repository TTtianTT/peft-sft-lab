from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finetune.data.chat_sft import chat_template_kwargs, ensure_chat_template


_CODE_FENCE_RE = re.compile(
    r"```[ \t]*(?P<language>[A-Za-z0-9_+.-]*)[ \t]*\r?\n(?P<code>.*?)```",
    flags=re.DOTALL,
)


def strip_outer_blank_lines(text: str) -> str:
    """Remove blank edge lines without destroying first-line code indentation."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "".join(lines).rstrip("\n")


def extract_first_python_code_block(text: str) -> str | None:
    """Return the first Python/unlabelled fenced block, preserving indentation."""
    matches = list(_CODE_FENCE_RE.finditer(text))
    for match in matches:
        language = match.group("language").lower()
        if language in {"", "python", "py", "python3"}:
            return strip_outer_blank_lines(match.group("code"))
    return None


def find_parseable_python_segment(text: str, *, prefix: str = "") -> str | None:
    """Find the least-trimmed line segment that forms valid Python with prefix."""
    text = strip_outer_blank_lines(text)
    if not text.strip():
        return None

    import ast

    lines = text.splitlines(keepends=True)
    for start in range(len(lines)):
        for end in range(len(lines), start, -1):
            candidate = strip_outer_blank_lines("".join(lines[start:end]))
            if not candidate.strip():
                continue
            try:
                ast.parse(prefix + candidate)
            except (SyntaxError, ValueError, TypeError):
                continue
            return candidate
    return None


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


def load_eval_tokenizer(
    *,
    base_model: str,
    adapter_dir: str | None,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_pick_tokenizer_source(base_model, adapter_dir), use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_transformers_model(
    *,
    base_model: str,
    adapter_dir: str | None,
    dtype: str = "auto",
    device_map: str | dict[str, int] | None = "auto",
) -> LoadedModel:
    import torch
    from transformers import AutoModelForCausalLM

    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        torch_dtype = None

    tokenizer = load_eval_tokenizer(base_model=base_model, adapter_dir=adapter_dir)

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


def render_chat_prompt(
    *,
    tokenizer: Any,
    base_model: str,
    user_content: str,
    system_content: str | None = None,
    chat_template_mode: str = "auto",
) -> str:
    ensure_chat_template(tokenizer, base_model)

    messages: list[dict[str, str]] = []
    if system_content is not None and system_content.strip():
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs(chat_template_mode),
    )


def _model_input_device(model: Any):
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _gpu_requires_disabling_flashinfer_sampler() -> bool:
    try:
        import torch
    except Exception:
        return False

    if not torch.cuda.is_available():
        return False

    try:
        for device_idx in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(device_idx)
            if (major, minor) < (7, 5):
                return True
    except Exception:
        return False

    return False


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


def generate_greedy_vllm_batch(
    *,
    base_model: str,
    prompts: list[str],
    max_new_tokens: int,
    adapter_dir: str | None = None,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    attention_backend: str | None = None,
    disable_flashinfer_sampler: bool = False,
    request_batch_size: int | None = None,
) -> list[str]:
    if disable_flashinfer_sampler or _gpu_requires_disabling_flashinfer_sampler():
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

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
        **({} if max_model_len is None else {"max_model_len": max_model_len}),
        **(
            {}
            if gpu_memory_utilization is None
            else {"gpu_memory_utilization": gpu_memory_utilization}
        ),
        **({} if attention_backend is None else {"attention_backend": attention_backend}),
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    chunk_size = len(prompts)
    if request_batch_size is not None and request_batch_size > 0:
        chunk_size = min(int(request_batch_size), len(prompts))

    texts: list[str] = []
    for start in range(0, len(prompts), chunk_size):
        prompt_chunk = prompts[start : start + chunk_size]
        outputs = llm.generate(prompt_chunk, params, lora_request=lora_request)
        if not outputs:
            texts.extend([""] * len(prompt_chunk))
            continue

        chunk_texts: list[str] = []
        for req_out in outputs:
            if not getattr(req_out, "outputs", None):
                chunk_texts.append("")
            else:
                chunk_texts.append(req_out.outputs[0].text.strip())

        if len(chunk_texts) < len(prompt_chunk):
            chunk_texts.extend([""] * (len(prompt_chunk) - len(chunk_texts)))
        elif len(chunk_texts) > len(prompt_chunk):
            chunk_texts = chunk_texts[: len(prompt_chunk)]

        texts.extend(chunk_texts)

    return texts[: len(prompts)]


def generate_greedy_vllm(
    *,
    base_model: str,
    prompt: str,
    max_new_tokens: int,
    adapter_dir: str | None = None,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    attention_backend: str | None = None,
    disable_flashinfer_sampler: bool = False,
    request_batch_size: int | None = None,
) -> str:
    outputs = generate_greedy_vllm_batch(
        base_model=base_model,
        prompts=[prompt],
        max_new_tokens=max_new_tokens,
        adapter_dir=adapter_dir,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        attention_backend=attention_backend,
        disable_flashinfer_sampler=disable_flashinfer_sampler,
        request_batch_size=request_batch_size,
    )
    return outputs[0] if outputs else ""


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
