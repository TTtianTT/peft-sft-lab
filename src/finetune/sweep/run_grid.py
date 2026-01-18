from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _bool_flag(key: str, val: Any) -> list[str]:
    return [f"--{key}"] if bool(val) else []


def _kv_flag(key: str, val: Any) -> list[str]:
    if val is None:
        return []
    return [f"--{key}", str(val)]


def _is_done(output_dir: str) -> bool:
    p = Path(output_dir)
    return (p / "adapter_config.json").exists() or (p / "adapter_model.safetensors").exists()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a configs.jsonl sweep serially via accelerate launch.")
    p.add_argument("--configs_jsonl", type=str, required=True)
    p.add_argument("--num_processes", type=int, default=1)
    p.add_argument("--accelerate_config", type=str, default=None, help="Optional accelerate config file.")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_done", action="store_true", help="Skip configs whose output_dir already has an adapter.")
    p.add_argument("--continue_on_error", action="store_true")
    return p


def _config_to_train_args(cfg: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key in ["base_model", "task", "peft_method", "output_dir"]:
        args += _kv_flag(key, cfg.get(key))

    for key in [
        "max_steps",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "lr",
        "warmup_ratio",
        "weight_decay",
        "max_seq_len",
        "global_train_batch_size",
        "min_lr_ratio",
        "grad_clip",
        "adam_beta1",
        "adam_beta2",
        "lr_scheduler_type",
        "seed",
        "max_train_samples",
        "dataset_seed",
        "target_modules",
        "r",
        "lora_alpha",
        "lora_dropout",
        "pissa_init_mode",
        "init_r",
        "target_r",
        "loraplus_lr_ratio",
        "loraplus_lr_embedding",
        "train_profile",
    ]:
        args += _kv_flag(key, cfg.get(key))

    for key in ["bf16", "fp16", "gradient_checkpointing", "use_qlora"]:
        args += _bool_flag(key, cfg.get(key, False))

    return args


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg_path = Path(args.configs_jsonl)
    if not cfg_path.exists():
        raise FileNotFoundError(str(cfg_path))

    lines = [ln.strip() for ln in cfg_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    configs = [json.loads(ln) for ln in lines]

    for idx, cfg in enumerate(configs):
        out_dir = cfg.get("output_dir")
        if args.skip_done and isinstance(out_dir, str) and _is_done(out_dir):
            print(f"[{idx+1}/{len(configs)}] skip_done: {out_dir}")
            continue

        cmd = ["accelerate", "launch", "--num_processes", str(args.num_processes)]
        if args.accelerate_config:
            cmd += ["--config_file", args.accelerate_config]
        cmd += ["-m", "finetune.train_sft_peft"]
        cmd += _config_to_train_args(cfg)

        print(f"[{idx+1}/{len(configs)}] {' '.join(cmd)}")
        if args.dry_run:
            continue

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Run failed with exit_code={exc.returncode}")
            if not args.continue_on_error:
                raise


if __name__ == "__main__":
    main()
