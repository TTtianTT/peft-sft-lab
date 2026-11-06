import pytest

from finetune.train_sft_peft import _resolve_warmup_steps


def test_explicit_warmup_survives_installed_transformers_api(tmp_path):
    from transformers import TrainingArguments
    steps = _resolve_warmup_steps(157, .03, 1561)
    args = TrainingArguments(output_dir=str(tmp_path), warmup_steps=steps, report_to=[])
    assert args.to_dict()['warmup_steps'] == 157
    assert args.get_warmup_steps(1561) == 157
    assert _resolve_warmup_steps(0, .1, 1561) == 0


def test_ratio_converts_to_explicit_steps_with_fractional_last_step():
    assert _resolve_warmup_steps(None, .1, 1561) == 157
    with pytest.raises(ValueError):
        _resolve_warmup_steps(-1, .1, 1561)
