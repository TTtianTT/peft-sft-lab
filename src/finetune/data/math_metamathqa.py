from __future__ import annotations

from typing import Any

from finetune.data.base import TaskPlugin, first_present


# MetaMathQA "Model Usage" prompting template (prompt部分)
_METAMATH_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response: Let's think step by step."
)


def format_metamath_instruction_response(*, instruction: str, response: str) -> str:
    """
    Build one SFT training sample following MetaMathQA Model Usage template.

    - instruction: use MetaMathQA `query` (preferred) as the instruction text.
    - response: the ground-truth solution text to supervise.
    """
    instruction = instruction.strip()
    response = response.strip()

    prompt = _METAMATH_PROMPT_TEMPLATE.format(instruction=instruction)

    # Important: add a newline before the supervised response, so the model learns to continue from the prompt.
    return f"{prompt}\n{response}"


class MetaMathQATask(TaskPlugin):
    name = "math"
    dataset_id = "meta-math/MetaMathQA"

    def load(self, split: str):
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError(f"datasets is required: {exc}") from exc

        try:
            return load_dataset(self.dataset_id, split=split)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {self.dataset_id} split={split}: {exc}\n"
                "Try: pip install -U datasets\n"
                "Or verify the dataset id on Hugging Face."
            ) from exc

    def format_example(self, example: dict[str, Any]) -> str:
        # Per dataset card: use `query` to fill {instruction}
        instruction = first_present(
            example,
            ["query", "original_question", "question", "instruction", "prompt"],
        )
        response = first_present(example, ["response", "answer", "output", "solution"])
        if instruction is None or response is None:
            raise ValueError(
                f"MetaMathQA example missing required fields. Keys: {sorted(example.keys())}. "
                "Expected something like (query,response)."
            )
        return format_metamath_instruction_response(instruction=instruction, response=response)
