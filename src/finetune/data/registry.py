from __future__ import annotations

from finetune.data.base import TaskPlugin
from finetune.data.code_magicoder import MagicoderTask
from finetune.data.csqa import CommonsenseQATask
from finetune.data.general_alpaca import AlpacaTask
from finetune.data.math_metamathqa import MetaMathQATask


_TASKS: dict[str, type[TaskPlugin]] = {
    "math": MetaMathQATask,
    "metamath": MetaMathQATask,
    "metamathqa": MetaMathQATask,
    "code": MagicoderTask,
    "magicoder": MagicoderTask,
    "alpaca": AlpacaTask,
    "general": AlpacaTask,
    "csqa": CommonsenseQATask,
    "commonsenseqa": CommonsenseQATask,
}


def get_task_plugin(name: str) -> TaskPlugin:
    key = name.strip().lower()
    if key not in _TASKS:
        known = ", ".join(sorted(_TASKS))
        raise ValueError(f"Unknown --task {name!r}. Known tasks: {known}")
    return _TASKS[key]()
