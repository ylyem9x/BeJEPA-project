import dataclasses
import datetime
import hashlib
import random
import string
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

N_RANDOM_CHARACTERS = 4


def load_optimizer_state_preserving_hyperparameters(optimizer: torch.optim.Optimizer, state_dict: dict) -> None:
    """Restore optimizer moments while keeping hyperparameters from the current config."""
    current_group_options = [
        {key: value for (key, value) in group.items() if key != "params"} for group in optimizer.param_groups
    ]
    optimizer.load_state_dict(state_dict)
    if len(optimizer.param_groups) != len(current_group_options):
        raise ValueError(
            "Checkpoint optimizer parameter-group count does not match the optimizer built from the current config."
        )
    for group, current_options in zip(optimizer.param_groups, current_group_options):
        group.update(current_options)


class EveryNStepsChecker:
    def __init__(self, current_step: int, every_n_steps: int, step_zero_should_trigger: bool = True):
        self.step_zero_should_trigger = step_zero_should_trigger
        self.last_step = current_step
        self.every_n_steps = every_n_steps

    def check(self, step: int) -> bool:
        if step - self.last_step >= self.every_n_steps or (self.step_zero_should_trigger and step == 0):
            return True
        else:
            return False

    def update_last_step(self, step: int):
        self.last_step = step


def dict_to_config(source: Mapping, target: Any):
    target_fields = {field.name for field in dataclasses.fields(target)}
    for field in target_fields:
        if field in source.keys() and dataclasses.is_dataclass(getattr(target, field)):
            dict_to_config(source[field], getattr(target, field))
        elif field in source.keys():
            setattr(target, field, source[field])
        else:
            print(f"[WARNING] field {field} not found in source config")


def get_default_torch_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def config_from_dict(source: Dict, config_class: Any) -> dataclasses.dataclass:
    target = config_class()
    dict_to_config(source, target)
    return target


def all_subclasses(cls):
    """Get all subclasses of cls recursively."""
    subs = set(cls.__subclasses__())
    return subs | {s for c in subs for s in all_subclasses(c)}


def get_unique_name() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    timestamp = f"{now.year}-{now.month}-{now.day}-{now.hour}:{now.minute}:{now.second}"
    random_letters = "".join(
        (random.choice(string.ascii_uppercase + string.digits) for _ in range(N_RANDOM_CHARACTERS))
    )
    return f"{timestamp}-{random_letters}"


def get_local_workdir(name: str = "") -> str:
    return str(Path.cwd() / "workdir" / name / get_unique_name())


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_md5_of_file(filepath: str) -> str:
    file_bytes = open(filepath, "rb").read()
    digest = hashlib.md5(file_bytes).hexdigest()
    return digest
