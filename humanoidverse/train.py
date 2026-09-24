import os

_ORIGINAL_CUDA_VISIBLE_DEVICES = os.environ.get(
    "BFM_ORIGINAL_CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES")
)
_CUDA_VISIBLE_ISOLATED = os.environ.get("BFM_CUDA_VISIBLE_ISOLATED") == "1"
from humanoidverse.agents.envs.humanoidverse_isaac import (
    EXPERT_OBSERVATION_PIPELINE_VERSION,
    HumanoidVerseIsaacConfig,
    load_expert_trajectories_from_motion_lib,
)
from humanoidverse.agents.evaluations.goal_eval import HumanoidVerseIsaacGoalEvaluationConfig
from humanoidverse.agents.evaluations.humanoidverse_isaac import (
    HumanoidVerseIsaacTrackingEvaluationConfig,
    temporary_reference_motion_dataset,
)

os.environ["OMP_NUM_THREADS"] = "1"


def _normalize_proxy_env() -> None:
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        value = os.environ.get(key)
        if value and "://" not in value:
            os.environ[key] = f"http://{value}"


_normalize_proxy_env()
import torch

torch.set_float32_matmul_precision("high")
import contextlib
import copy
import datetime
import json
import time
import typing as tp
from pathlib import Path
from typing import Dict, List

import exca as xk
import gymnasium
import numpy as np
import pydantic
import torch
import torch.distributed as dist
import wandb
from packaging.version import Version
from torch.utils._pytree import tree_map
from tqdm import tqdm

from humanoidverse.agents.base import BaseConfig
from humanoidverse.agents.buffers.load_data import load_expert_trajectories
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.buffers.transition import DictBuffer, dtype_numpytotorch_lower_precision
from humanoidverse.agents.jepa_successor.agent import JepaSuccessorCprAuxAgentConfig
from humanoidverse.agents.misc.loggers import CSVLogger
from humanoidverse.agents.utils import EveryNStepsChecker, get_local_workdir, set_seed_everywhere

TRAIN_LOG_FILENAME = "train_log.txt"
REWARD_EVAL_LOG_FILENAME = "reward_eval_log.csv"
TRACKING_EVAL_LOG_FILENAME = "tracking_eval_log.csv"
CHECKPOINT_DIR_NAME = "checkpoint"
RESUME_CHANGES_FILENAME = "resume_changes.md"
_ENC_CONFIG_TO_EXPERT_DATA_OBS_MAPPER = {HumanoidVerseIsaacConfig: None}
Evaluation = tp.Annotated[
    tp.Union[HumanoidVerseIsaacTrackingEvaluationConfig, HumanoidVerseIsaacGoalEvaluationConfig],
    pydantic.Field(discriminator="name"),
]
Agent = JepaSuccessorCprAuxAgentConfig


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _dist_available() -> bool:
    return dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)


def _dist_rank() -> int:
    return dist.get_rank() if _dist_available() else 0


def _dist_world_size() -> int:
    return dist.get_world_size() if _dist_available() else 1


def _dist_is_rank0() -> bool:
    return _dist_rank() == 0


def _dist_barrier() -> None:
    if _dist_available():
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def _rank_print(message: str) -> None:
    rank = _dist_rank()
    world_size = _dist_world_size()
    print(f"[rank{rank}/{world_size}] {message}", flush=True)


def _copy_env_leaf_to_device(value, device: str | torch.device) -> torch.Tensor:
    """Copy one environment output leaf without a GPU -> NumPy -> GPU round trip."""
    dtype = dtype_numpytotorch_lower_precision(value.dtype)
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype, non_blocking=True).clone()
    return torch.as_tensor(value, dtype=dtype, device=device)


def _index_env_leaf(value, indexes):
    """Index either a NumPy or torch environment leaf at reduced precision."""
    if torch.is_tensor(value):
        dtype = dtype_numpytotorch_lower_precision(value.dtype)
        return value.to(dtype=dtype)[indexes]
    dtype = np.float32 if value.dtype == np.float64 else value.dtype
    return value.astype(dtype, copy=False)[indexes]


_MISSING_CONFIG_VALUE = object()


def _read_json_if_exists(path: Path) -> dict[str, tp.Any] | None:
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def _canonicalize_run_config(config: dict[str, tp.Any]) -> dict[str, tp.Any]:
    config = copy.deepcopy(config)
    tags = dict(config.get("tags", {}))
    if "distributed_global_batch_size" in tags:
        config["agent"]["train"]["batch_size"] = tags["distributed_global_batch_size"]
    if "distributed_global_online_parallel_envs" in tags:
        config["online_parallel_envs"] = tags["distributed_global_online_parallel_envs"]
    if "distributed_global_buffer_size" in tags:
        config["buffer_size"] = tags["distributed_global_buffer_size"]
    for key in list(tags):
        if key.startswith("distributed_"):
            tags.pop(key)
    config["tags"] = tags
    return config


def _runtime_config(config: dict[str, tp.Any]) -> dict[str, tp.Any]:
    tags = config.get("tags", {})
    local_batch_size = tags.get(
        "distributed_local_batch_size", config.get("agent", {}).get("train", {}).get("batch_size")
    )
    local_num_envs = tags.get("distributed_local_online_parallel_envs", config.get("online_parallel_envs"))
    local_buffer_size = tags.get("distributed_local_buffer_size", config.get("buffer_size"))
    return {
        "world_size": tags.get("distributed_world_size", 1),
        "local_batch_size": local_batch_size,
        "local_online_parallel_envs": local_num_envs,
        "local_buffer_size": local_buffer_size,
    }


def _collect_config_changes(previous: tp.Any, current: tp.Any, prefix: str = "") -> list[tuple[str, tp.Any, tp.Any]]:
    if isinstance(previous, dict) and isinstance(current, dict):
        changes = []
        for key in sorted(set(previous) | set(current)):
            path = f"{prefix}.{key}" if prefix else key
            changes.extend(
                _collect_config_changes(
                    previous.get(key, _MISSING_CONFIG_VALUE), current.get(key, _MISSING_CONFIG_VALUE), path
                )
            )
        return changes
    if previous != current:
        return [(prefix, previous, current)]
    return []


def _format_config_value(value: tp.Any) -> str:
    if value is _MISSING_CONFIG_VALUE:
        return "<unset>"
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return rendered.replace("|", "\\|")


def _write_resume_changes(
    work_dir: Path,
    previous_run_config: dict[str, tp.Any] | None,
    checkpoint_agent_config: dict[str, tp.Any],
    current_config: "TrainConfig",
) -> None:
    current = current_config.model_dump(mode="json")
    if previous_run_config is None:
        previous = {"agent": checkpoint_agent_config}
        current = {"agent": current["agent"]}
    else:
        previous = _canonicalize_run_config(previous_run_config)
        current_without_runtime_tags = _canonicalize_run_config(current)
        previous["runtime"] = _runtime_config(previous_run_config)
        current_without_runtime_tags["runtime"] = _runtime_config(current)
        current = current_without_runtime_tags
    changes = _collect_config_changes(previous, current)
    lines = ["# Resume configuration changes", ""]
    if changes:
        lines.extend(["| Parameter | Previous run | Current run |", "| --- | --- | --- |"])
        lines.extend(
            (
                f"| `{path}` | `{_format_config_value(old)}` | `{_format_config_value(new)}` |"
                for (path, old, new) in changes
            )
        )
    else:
        lines.append("No parameter changes.")
    (work_dir / RESUME_CHANGES_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


class TrainConfig(BaseConfig):
    agent: Agent = pydantic.Field(discriminator="name")
    motions: str | None = None
    motions_root: str | None = None
    env: HumanoidVerseIsaacConfig = pydantic.Field(discriminator="name")
    work_dir: str = pydantic.Field(default_factory=lambda: get_local_workdir("g1mujoco_train"))
    seed: int = 0
    online_parallel_envs: int = 50
    log_every_updates: int = 100000
    num_env_steps: int = 30000000
    update_agent_every: int = 500
    num_seed_steps: int = 50000
    num_agent_updates: int = 50
    checkpoint_every_steps: int = 5000000
    checkpoint_buffer: bool = True
    prioritization: bool = False
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 5
    prioritization_scale: float = 2
    prioritization_mode: str = "bin"
    padding_beginning: int = 0
    padding_end: int = 0
    use_trajectory_buffer: bool = False
    buffer_size: int = 5000000
    use_wandb: bool = True
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = None
    load_isaac_expert_data: bool = True
    expert_observation_pipeline_version: tp.Literal["isaacsim_usd_replay_v1"] = EXPERT_OBSERVATION_PIPELINE_VERSION
    buffer_device: str = "cpu"
    buffer_pin_memory: bool = False
    isaac_zero_copy: bool = False
    disable_tqdm: bool = True
    distributed: bool = False
    distributed_auto_detect: bool = True
    distributed_backend: str = "nccl"
    distributed_timeout_s: int = 600
    evaluations: Dict[str, Evaluation] | List[Evaluation] = pydantic.Field(default_factory=lambda: [])
    eval_every_steps: int = 1000000
    disable_initial_eval: bool = False
    tags: dict = pydantic.Field(default_factory=lambda: {})
    infra: xk.TaskInfra = xk.TaskInfra(version="1")

    def model_post_init(self, context):
        if self.distributed_timeout_s <= 0:
            raise ValueError("distributed_timeout_s must be > 0")
        if self.load_isaac_expert_data and (not isinstance(self.env, HumanoidVerseIsaacConfig)):
            raise ValueError("Loading expert isaac data is only supported for HumanoidVerseIsaacConfig")
        if self.prioritization:
            has_prioritization_eval = False
            for eval_type in self.evaluations:
                if isinstance(eval_type, HumanoidVerseIsaacTrackingEvaluationConfig):
                    has_prioritization_eval = True
                    break
            if not has_prioritization_eval:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")
        if self.motions is None or self.motions_root is None:
            if self.prioritization:
                raise ValueError("Prioritization requires expert data to be provided (motions and motions_root)")
        if isinstance(self.evaluations, list):
            log_names = set()
            for eval_cfg in self.evaluations:
                if eval_cfg.name_in_logs in log_names:
                    raise ValueError(
                        f"Duplicate evaluation name_in_logs found: {eval_cfg.name}. These should be unique so we do not overwrite any logs"
                    )
                log_names.add(eval_cfg.name_in_logs)

    def build(self):
        """In case of cluster run, use exca and process instead of explivit build"""
        return Workspace(self)


def _validate_expert_observation_resume_compatibility(
    cfg: TrainConfig, previous_run_config: dict[str, tp.Any] | None
) -> None:
    if not cfg.load_isaac_expert_data:
        return
    previous_version = (
        previous_run_config.get("expert_observation_pipeline_version") if previous_run_config is not None else None
    )
    if previous_version == cfg.expert_observation_pipeline_version:
        return
    previous_label = previous_version or "legacy_motionlib_mjcf_manual_v0"
    raise RuntimeError(
        f"This checkpoint uses an incompatible expert-observation pipeline ({previous_label!r}); the current run requires {cfg.expert_observation_pipeline_version!r}. The old discriminator/CPR critic may already encode the expert-vs-online source shortcut. Start training in a new work_dir instead of resuming this checkpoint."
    )


def create_agent_or_load_checkpoint(work_dir: Path, cfg: TrainConfig, agent_build_kwargs: dict[str, tp.Any]):
    checkpoint_dir = work_dir / CHECKPOINT_DIR_NAME
    checkpoint_time = 0
    if checkpoint_dir.exists():
        previous_run_config = _read_json_if_exists(work_dir / "config.json")
        _validate_expert_observation_resume_compatibility(cfg, previous_run_config)
        with (checkpoint_dir / "train_status.json").open("r") as f:
            train_status = json.load(f)
        checkpoint_time = train_status["time"]
        with (checkpoint_dir / "config.json").open("r") as f:
            checkpoint_agent_config = json.load(f)
        checkpoint_agent_name = checkpoint_agent_config.get("name")
        if checkpoint_agent_name != cfg.agent.name:
            raise ValueError(
                f"Checkpoint agent mismatch in {checkpoint_dir}: checkpoint has {checkpoint_agent_name}, but current config requests {cfg.agent.name}. Use a different --work-dir or the matching train entry."
            )
        _rank_print(f"loading agent checkpoint at time {checkpoint_time}")
        agent = cfg.agent.object_class.load(
            checkpoint_dir,
            device=cfg.agent.model.device,
            config=cfg.agent,
            obs_space=agent_build_kwargs["obs_space"],
            action_dim=agent_build_kwargs["action_dim"],
            strict=True,
            preserve_optimizer_hyperparameters=True,
        )
        if _dist_is_rank0():
            _write_resume_changes(
                work_dir=work_dir,
                previous_run_config=previous_run_config,
                checkpoint_agent_config=checkpoint_agent_config,
                current_config=cfg,
            )
    else:
        _rank_print("building agent")
        agent = cfg.agent.build(**agent_build_kwargs)
        _rank_print("finished agent build")
    return (agent, cfg, checkpoint_time)


def init_wandb(cfg: TrainConfig):
    wandb_name = Path(cfg.work_dir).name
    wandb_config = cfg.model_dump()
    wandb_root_dir = Path(cfg.work_dir).resolve()
    wandb_root_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        mode="offline",
        entity=cfg.wandb_ename,
        project=cfg.wandb_pname,
        group=cfg.wandb_gname,
        name=wandb_name,
        config=wandb_config,
        dir=str(wandb_root_dir),
    )
    print(f"wandb root dir: {wandb_root_dir / 'wandb'}")
    if run is not None:
        print(f"wandb run dir: {run.dir}")


def _divide_for_rank(value: int, world_size: int, name: str) -> int:
    if value % world_size != 0:
        raise ValueError(f"{name}={value} must be divisible by distributed world_size={world_size}.")
    local_value = value // world_size
    if local_value <= 0:
        raise ValueError(f"{name}/{world_size} must be > 0.")
    return local_value


def _init_distributed_and_localize_config(cfg: TrainConfig) -> TrainConfig:
    env_world_size = _env_int("WORLD_SIZE", 1)
    should_init = cfg.distributed or (cfg.distributed_auto_detect and env_world_size > 1)
    if not should_init:
        return cfg
    if env_world_size <= 1:
        raise ValueError("Distributed training requested, but WORLD_SIZE is not set by torchrun.")
    local_rank = _env_int("LOCAL_RANK", 0)
    if torch.cuda.is_available():
        device_index = 0 if _CUDA_VISIBLE_ISOLATED else local_rank
        torch.cuda.set_device(device_index)
        rank_device = f"cuda:{device_index}"
        model_device = "cuda"
    else:
        rank_device = "cpu"
        model_device = "cpu"
    if not dist.is_initialized():
        backend = cfg.distributed_backend if rank_device != "cpu" else "gloo"
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=cfg.distributed_timeout_s))
    world_size = _dist_world_size()
    local_batch_size = _divide_for_rank(cfg.agent.train.batch_size, world_size, "agent.train.batch_size")
    local_num_envs = _divide_for_rank(cfg.online_parallel_envs, world_size, "online_parallel_envs")
    local_buffer_size = _divide_for_rank(cfg.buffer_size, world_size, "buffer_size")
    train_cfg = cfg.agent.train.model_copy(update={"batch_size": local_batch_size})
    model_cfg = cfg.agent.model.model_copy(update={"device": model_device})
    agent_cfg = cfg.agent.model_copy(update={"train": train_cfg, "model": model_cfg})
    env_cfg = cfg.env.model_copy(update={"device": rank_device})
    buffer_device = rank_device if torch.device(cfg.buffer_device).type == "cuda" else cfg.buffer_device
    tags = dict(cfg.tags)
    tags.update(
        {
            "distributed_world_size": world_size,
            "distributed_local_rank": local_rank,
            "distributed_cuda_visible_isolated": _CUDA_VISIBLE_ISOLATED,
            "distributed_original_cuda_visible_devices": _ORIGINAL_CUDA_VISIBLE_DEVICES,
            "distributed_global_batch_size": cfg.agent.train.batch_size,
            "distributed_local_batch_size": local_batch_size,
            "distributed_global_online_parallel_envs": cfg.online_parallel_envs,
            "distributed_local_online_parallel_envs": local_num_envs,
            "distributed_global_buffer_size": cfg.buffer_size,
            "distributed_local_buffer_size": local_buffer_size,
        }
    )
    return cfg.model_copy(
        update={
            "agent": agent_cfg,
            "env": env_cfg,
            "online_parallel_envs": local_num_envs,
            "buffer_size": local_buffer_size,
            "buffer_device": buffer_device,
            "tags": tags,
        }
    )


class Workspace:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = _init_distributed_and_localize_config(cfg)
        self.rank = _dist_rank()
        self.world_size = _dist_world_size()
        self.is_rank0 = _dist_is_rank0()
        self.global_step_increment = self.cfg.online_parallel_envs * self.world_size
        _rank_print(
            f"localized config: model_device={self.cfg.agent.model.device}, env_device={self.cfg.env.device}, buffer_device={self.cfg.buffer_device}, local_envs={self.cfg.online_parallel_envs}, local_batch={self.cfg.agent.train.batch_size}, local_buffer_size={self.cfg.buffer_size}, isaac_zero_copy={self.cfg.isaac_zero_copy}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )
        if isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
            from omegaconf import OmegaConf

            _rank_print("building HumanoidVerseIsaac env")
            (self.train_env, self.train_env_info) = self.cfg.env.build(num_envs=self.cfg.online_parallel_envs)
            _rank_print("finished HumanoidVerseIsaac env build")
            self.obs_space = self.train_env.single_observation_space
            self.action_space = self.train_env.single_action_space
        else:
            _rank_print("building sample env")
            (sample_env, _) = self.cfg.env.build(num_envs=1)
            _rank_print("finished sample env build")
            self.obs_space = sample_env.observation_space
            self.action_space = sample_env.action_space
        assert "time" in self.obs_space.keys(), (
            "Observation space must contain 'obs' and 'time' (TimeAwareObservation wrapper)"
        )
        assert len(self.action_space.shape) == 1, "Only 1D action space is supported (first dim should be vector env)"
        del self.obs_space.spaces["time"]
        self.action_dim = self.action_space.shape[0]
        _rank_print(f"Workdir: {self.cfg.work_dir}")
        self.work_dir = Path(self.cfg.work_dir)
        if self.is_rank0:
            self.work_dir.mkdir(exist_ok=True, parents=True)
        while not self.work_dir.exists():
            time.sleep(0.1)
        if self.is_rank0 and isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
            with open(self.work_dir / "config.yaml", "w") as file:
                OmegaConf.save(self.train_env_info["unresolved_conf"], file)
        self.train_logger = CSVLogger(filename=self.work_dir / TRAIN_LOG_FILENAME) if self.is_rank0 else None
        set_seed_everywhere(self.cfg.seed + self.rank)
        (self.agent, self.cfg, self._checkpoint_time) = create_agent_or_load_checkpoint(
            self.work_dir, self.cfg, agent_build_kwargs=dict(obs_space=self.obs_space, action_dim=self.action_dim)
        )
        _rank_print("agent built/loaded; waiting before initial model sync")
        _dist_barrier()
        _rank_print("syncing initial model state")
        if hasattr(self.agent, "sync_distributed_state"):
            self.agent.sync_distributed_state(src=0)
        _dist_barrier()
        _rank_print("initial model state synced")
        self.agent._model.train()
        if self.is_rank0 and isinstance(self.cfg.evaluations, list):
            self.evaluations = {eval_cfg.name_in_logs: eval_cfg.build() for eval_cfg in self.cfg.evaluations}
        elif self.is_rank0:
            self.evaluations = {eval_cfg: eval_cfg.build() for (name, eval_cfg) in self.cfg.evaluations.items()}
        else:
            self.evaluations = {}
        self.evaluate = len(self.cfg.evaluations) > 0
        self.eval_loggers = (
            {name: CSVLogger(filename=self.work_dir / f"{name}.csv") for name in self.evaluations.keys()}
            if self.is_rank0
            else {}
        )
        if self.is_rank0 and self.cfg.use_wandb:
            init_wandb(self.cfg)
        if self.is_rank0:
            with (self.work_dir / "config.json").open("w") as f:
                f.write(self.cfg.model_dump_json(indent=4))
        _dist_barrier()
        self.priorization_eval_name = None
        if self.cfg.prioritization:
            if isinstance(self.cfg.evaluations, list):
                for eval_cfg in self.cfg.evaluations:
                    if isinstance(eval_cfg, HumanoidVerseIsaacTrackingEvaluationConfig):
                        self.priorization_eval_name = eval_cfg.name_in_logs
                        break
            else:
                for name, eval_cfg in self.cfg.evaluations.items():
                    if isinstance(eval_cfg, HumanoidVerseIsaacTrackingEvaluationConfig):
                        self.priorization_eval_name = name
                        break
            if self.priorization_eval_name is None:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")
        self.training_with_expert_data = True
        self.manager = None

    def _reduce_metrics(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not _dist_available():
            return metrics
        keys = sorted(metrics)
        values = []
        for key in keys:
            value = metrics[key]
            if torch.is_tensor(value):
                values.append(value.detach().float().mean().to(self.agent.device))
            else:
                values.append(torch.tensor(float(value), device=self.agent.device))
        packed = torch.stack(values)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed.div_(self.world_size)
        return {key: packed[index].detach() for (index, key) in enumerate(keys)}

    @staticmethod
    def _motion_id(value: tp.Any, *, source: str) -> int:
        """Convert a scalar motion id without silently accepting fractional values."""
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"{source} must be a scalar motion id, got shape {tuple(value.shape)}.")
            value = value.item()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{source} must be an integer motion id, got {value!r}.")
        return int(value)

    def _build_expert_priority_update(self, eval_metrics, replay_buffer) -> tuple[torch.Tensor, torch.Tensor]:
        """Build one priority per motion, averaging repeated evaluation episodes."""
        try:
            tracking_metrics = eval_metrics[self.priorization_eval_name]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Missing prioritization evaluation {self.priorization_eval_name!r}.") from error
        if not tracking_metrics:
            raise ValueError("Prioritization evaluation returned no motion metrics.")
        emd_by_motion_id: dict[int, list[float]] = {}
        for metric_name, metric in tracking_metrics.items():
            if "motion_id" not in metric or "emd" not in metric:
                raise ValueError(f"Prioritization metric {metric_name!r} must contain motion_id and emd.")
            motion_id = self._motion_id(metric["motion_id"], source=f"metric {metric_name!r} motion_id")
            emd = float(metric["emd"])
            if not np.isfinite(emd):
                raise ValueError(f"Prioritization metric {metric_name!r} has non-finite emd={emd!r}.")
            emd_by_motion_id.setdefault(motion_id, []).append(emd)
        expert_motion_ids = {
            self._motion_id(value, source="expert buffer motion_id")
            for value in replay_buffer["expert_slicer"].motion_ids
        }
        evaluated_motion_ids = set(emd_by_motion_id)
        if evaluated_motion_ids != expert_motion_ids:
            missing_from_eval = sorted(expert_motion_ids - evaluated_motion_ids)
            missing_from_expert = sorted(evaluated_motion_ids - expert_motion_ids)
            raise ValueError(
                f"Prioritization motion-id mismatch: missing_from_eval={missing_from_eval[:10]} missing_from_expert={missing_from_expert[:10]}."
            )
        ordered_motion_ids = sorted(evaluated_motion_ids)
        averaged_emd = [
            sum(emd_by_motion_id[motion_id]) / len(emd_by_motion_id[motion_id]) for motion_id in ordered_motion_ids
        ]
        priorities = (
            torch.clamp(
                torch.tensor(averaged_emd, dtype=torch.float32, device=self.agent.device),
                min=self.cfg.prioritization_min_val,
                max=self.cfg.prioritization_max_val,
            )
            * self.cfg.prioritization_scale
        )
        if self.cfg.prioritization_mode == "lin":
            pass
        elif self.cfg.prioritization_mode == "exp":
            priorities = 2**priorities
        elif self.cfg.prioritization_mode == "bin":
            bins = torch.floor(priorities)
            for i in range(int(bins.min().item()), int(bins.max().item()) + 1):
                mask = bins == i
                n = mask.sum().item()
                if n > 0:
                    priorities[mask] = 1 / n
        else:
            raise ValueError(f"Unsupported prioritization mode {self.cfg.prioritization_mode}")
        if not torch.isfinite(priorities).all() or torch.any(priorities < 0) or priorities.sum() <= 0:
            raise ValueError("Prioritization produced invalid motion priorities.")
        motion_ids = torch.tensor(ordered_motion_ids, dtype=torch.long, device=self.agent.device)
        return (priorities, motion_ids)

    def _expand_expert_buffer_priorities(
        self, replay_buffer, priorities: torch.Tensor, motion_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand motion probabilities to every expert trajectory.

        Repeated trajectories for one motion split that motion's probability so
        their combined sampling mass remains equal to the motion-level value.
        """
        expert_motion_ids = [
            self._motion_id(value, source="expert buffer motion_id")
            for value in replay_buffer["expert_slicer"].motion_ids
        ]
        if not expert_motion_ids:
            raise ValueError("Cannot prioritize an empty expert buffer.")
        ordered_motion_ids = motion_ids.detach().cpu().tolist()
        priority_index = {motion_id: index for (index, motion_id) in enumerate(ordered_motion_ids)}
        local_motion_ids = set(expert_motion_ids)
        synchronized_motion_ids = set(priority_index)
        if local_motion_ids != synchronized_motion_ids:
            raise ValueError(
                f"Expert-buffer motion ids differ from the synchronized prioritization ids: local_only={sorted(local_motion_ids - synchronized_motion_ids)[:10]} synchronized_only={sorted(synchronized_motion_ids - local_motion_ids)[:10]}."
            )
        counts: dict[int, int] = {}
        for motion_id in expert_motion_ids:
            counts[motion_id] = counts.get(motion_id, 0) + 1
        priority_indexes = torch.tensor(
            [priority_index[motion_id] for motion_id in expert_motion_ids], dtype=torch.long, device=priorities.device
        )
        duplicate_counts = torch.tensor(
            [counts[motion_id] for motion_id in expert_motion_ids], dtype=priorities.dtype, device=priorities.device
        )
        buffer_priorities = priorities[priority_indexes] / duplicate_counts
        buffer_idxs = torch.arange(len(expert_motion_ids), dtype=torch.long, device=priorities.device)
        return (buffer_priorities, buffer_idxs)

    def _expert_motion_file_names(self, replay_buffer) -> dict[int, tp.Any] | None:
        expert_buffer = replay_buffer["expert_slicer"]
        if not hasattr(expert_buffer, "file_names"):
            return None
        if len(expert_buffer.file_names) != len(expert_buffer.motion_ids):
            raise ValueError("expert_slicer.file_names and motion_ids must have the same length.")
        file_names: dict[int, tp.Any] = {}
        for raw_motion_id, file_name in zip(expert_buffer.motion_ids, expert_buffer.file_names):
            motion_id = self._motion_id(raw_motion_id, source="expert buffer motion_id")
            previous = file_names.setdefault(motion_id, file_name)
            if previous != file_name:
                raise ValueError(f"Motion ID {motion_id} maps to multiple expert file names.")
        return file_names

    def _sync_and_apply_expert_priorities(self, eval_metrics, replay_buffer) -> None:
        build_error: Exception | None = None
        if self.is_rank0:
            try:
                (priorities, motion_ids) = self._build_expert_priority_update(eval_metrics, replay_buffer)
            except Exception as error:
                build_error = error
        if _dist_available():
            if build_error is not None:
                priority_count = -1
            elif self.is_rank0:
                priority_count = len(priorities)
            else:
                priority_count = 0
            count_tensor = torch.tensor([priority_count], dtype=torch.long, device=self.agent.device)
            dist.broadcast(count_tensor, src=0)
            priority_count = int(count_tensor.item())
            if priority_count < 0:
                if build_error is not None:
                    raise build_error
                raise RuntimeError("Rank 0 failed to build expert priorities; see the rank-0 traceback.")
            if not self.is_rank0:
                priorities = torch.empty(priority_count, dtype=torch.float32, device=self.agent.device)
                motion_ids = torch.empty(priority_count, dtype=torch.long, device=self.agent.device)
            dist.broadcast(priorities, src=0)
            dist.broadcast(motion_ids, src=0)
        elif build_error is not None:
            raise build_error
        (buffer_priorities, buffer_idxs) = self._expand_expert_buffer_priorities(replay_buffer, priorities, motion_ids)
        name_in_buffer = self._expert_motion_file_names(replay_buffer)
        train_env = self.train_env
        train_env._env._motion_lib.update_sampling_weight_by_id(
            priorities=priorities.detach().cpu().tolist(),
            motions_id=motion_ids.detach().cpu().tolist(),
            file_name=name_in_buffer,
        )
        replay_buffer["expert_slicer"].update_priorities(
            priorities=buffer_priorities.to(self.cfg.buffer_device), idxs=buffer_idxs.to(self.cfg.buffer_device)
        )

    def train(self):
        self.start_time = time.time()
        self.train_online()

    def train_online(self) -> None:
        if self.training_with_expert_data:
            if self.cfg.load_isaac_expert_data:
                expert_buffer = load_expert_trajectories_from_motion_lib(
                    self.train_env,
                    self.cfg.agent,
                    device=self.cfg.buffer_device,
                    pin_memory=self.cfg.buffer_pin_memory,
                    privileged_state_world_frame=self.cfg.env.privileged_state_world_frame,
                )
            else:
                print("Loading expert trajectories")
                expert_buffer = load_expert_trajectories(
                    self.cfg.motions,
                    self.cfg.motions_root,
                    seq_length=self.agent.cfg.model.seq_length,
                    device=self.cfg.buffer_device,
                    pin_memory=self.cfg.buffer_pin_memory,
                    obs_dict_mapper=_ENC_CONFIG_TO_EXPERT_DATA_OBS_MAPPER[self.cfg.env.__class__],
                )
        print("Creating the training environment")
        if isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
            train_env = self.train_env
            train_env_info = self.train_env_info
        else:
            (train_env, train_env_info) = self.cfg.env.build(num_envs=self.cfg.online_parallel_envs)
        isaac_zero_copy = isinstance(self.cfg.env, HumanoidVerseIsaacConfig) and self.cfg.isaac_zero_copy
        print("Allocating buffers")
        replay_buffer = {}
        checkpoint_dir = self.work_dir / CHECKPOINT_DIR_NAME
        if (checkpoint_dir / "buffers/train").exists():
            print("Loading checkpointed buffer")
            if self.cfg.use_trajectory_buffer:
                replay_buffer["train"] = TrajectoryDictBufferMultiDim.load(
                    checkpoint_dir / "buffers/train",
                    device=self.cfg.buffer_device,
                    pin_memory=self.cfg.buffer_pin_memory,
                )
            else:
                replay_buffer["train"] = DictBuffer.load(
                    checkpoint_dir / "buffers/train",
                    device=self.cfg.buffer_device,
                    pin_memory=self.cfg.buffer_pin_memory,
                )
            print(f"Loaded buffer of size {len(replay_buffer['train'])}")
        elif self.cfg.use_trajectory_buffer:
            rollout_keys = ["z"]
            output_key_t = ["observation", "action", *rollout_keys, "terminated", "truncated", "step_count", "reward"]
            output_key_t.append("aux_rewards")
            replay_buffer["train"] = TrajectoryDictBufferMultiDim(
                capacity=self.cfg.buffer_size // self.cfg.online_parallel_envs,
                device=self.cfg.buffer_device,
                pin_memory=self.cfg.buffer_pin_memory,
                n_dim=2,
                end_key="truncated",
                output_key_t=output_key_t,
                output_key_tp1=["observation", "terminated"],
            )
        else:
            replay_buffer["train"] = DictBuffer(
                capacity=self.cfg.buffer_size, device=self.cfg.buffer_device, pin_memory=self.cfg.buffer_pin_memory
            )
        if self.training_with_expert_data:
            replay_buffer["expert_slicer"] = expert_buffer
        print("Starting training")
        progb = tqdm(total=self.cfg.num_env_steps, disable=self.cfg.disable_tqdm or not self.is_rank0)
        (td, info) = train_env.reset(to_numpy=False) if isaac_zero_copy else train_env.reset()
        if isaac_zero_copy:
            terminated = torch.zeros(self.cfg.online_parallel_envs, dtype=torch.bool, device=self.agent.device)
            truncated = torch.zeros_like(terminated)
            done = torch.zeros_like(terminated)
            seed_action_low = torch.as_tensor(train_env.action_space.low, device=self.agent.device, dtype=torch.float32)
            seed_action_high = torch.as_tensor(
                train_env.action_space.high, device=self.agent.device, dtype=torch.float32
            )
        else:
            terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
            truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
            done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
            seed_action_low = None
            seed_action_high = None
        (total_metrics, context) = (None, None)
        start_time = time.time()
        fps_start_time = time.time()
        checkpoint_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.checkpoint_every_steps)
        eval_time_checker = EveryNStepsChecker(
            self._checkpoint_time, self.cfg.eval_every_steps, step_zero_should_trigger=not self.cfg.disable_initial_eval
        )
        update_agent_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.update_agent_every)
        log_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.log_every_updates)
        if isinstance(self.cfg.evaluations, list):
            uses_humanoidverse_eval = any(
                (
                    isinstance(
                        eval_cfg, (HumanoidVerseIsaacTrackingEvaluationConfig, HumanoidVerseIsaacGoalEvaluationConfig)
                    )
                    for eval_cfg in self.cfg.evaluations
                )
            )
        else:
            uses_humanoidverse_eval = any(
                (
                    isinstance(
                        eval_cfg, (HumanoidVerseIsaacTrackingEvaluationConfig, HumanoidVerseIsaacGoalEvaluationConfig)
                    )
                    for eval_cfg in self.cfg.evaluations.values()
                )
            )
        for t in range(
            self._checkpoint_time, self.cfg.num_env_steps + self.global_step_increment, self.global_step_increment
        ):
            if t != self._checkpoint_time and checkpoint_time_checker.check(t):
                checkpoint_time_checker.update_last_step(t)
                if self.is_rank0:
                    self.save(t, replay_buffer)
                _dist_barrier()
            if self.evaluate and (
                eval_time_checker.check(t) or (t == self._checkpoint_time and (not self.cfg.disable_initial_eval))
            ):
                eval_metrics = self.eval(t, replay_buffer=replay_buffer) if self.is_rank0 else None
                eval_time_checker.update_last_step(t)
                _dist_barrier()
                if uses_humanoidverse_eval:
                    (td, info) = train_env.reset(to_numpy=False) if isaac_zero_copy else train_env.reset()
                    if isaac_zero_copy:
                        terminated.zero_()
                        truncated.zero_()
                        done.zero_()
                    else:
                        terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                        truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                        done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                if self.cfg.prioritization:
                    self._sync_and_apply_expert_priorities(eval_metrics, replay_buffer)
            with torch.no_grad():
                obs = tree_map(lambda x: _copy_env_leaf_to_device(x, self.agent.device), td)
                step_count = obs.pop("time")
                history_context = None
                if "history" in obs:
                    if len(obs["history"]["action"]) == 0:
                        history_context = self.agent._model._context_encoder.get_initial_context(
                            self.cfg.online_parallel_envs
                        )
                    else:
                        history_context = self.agent.history_inference(
                            obs=obs["history"]["observation"], action=obs["history"]["action"]
                        )[:, -1].clone()
                context = self.agent.maybe_update_rollout_context(
                    z=context, step_count=step_count, replay_buffer=replay_buffer, obs=obs
                )
                replay_context_fields = self.agent.rollout_replay_fields(context)
                if t < self.cfg.num_seed_steps:
                    if isaac_zero_copy:
                        action = seed_action_low + torch.rand_like(seed_action_low) * (
                            seed_action_high - seed_action_low
                        )
                    else:
                        action = train_env.action_space.sample().astype(np.float32)
                else:
                    if history_context is not None:
                        action = self.agent.act(obs=obs, z=context, context=history_context, mean=False)
                    else:
                        action = self.agent.act(obs=obs, z=context, mean=False)
                    if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                        action = action.cpu().detach().numpy()
            if isaac_zero_copy:
                (new_td, new_reward, new_terminated, new_truncated, new_info) = train_env.step(action, to_numpy=False)
            else:
                (new_td, new_reward, new_terminated, new_truncated, new_info) = train_env.step(action)
            next_t = t + self.global_step_increment
            if self.evaluate and (
                eval_time_checker.check(next_t)
                or (next_t == self._checkpoint_time and (not self.cfg.disable_initial_eval))
            ):
                if isinstance(self.cfg.env, HumanoidVerseIsaacConfig) and uses_humanoidverse_eval:
                    if torch.is_tensor(new_truncated):
                        new_truncated = torch.ones_like(new_truncated, dtype=torch.bool)
                        truncated = torch.ones_like(new_truncated, dtype=torch.bool)
                    else:
                        new_truncated = np.ones_like(new_truncated, dtype=bool)
                        truncated = np.ones_like(new_truncated, dtype=bool)
            if Version(gymnasium.__version__) >= Version("1.0"):
                if self.cfg.use_trajectory_buffer:
                    data = {
                        "observation": tree_map(lambda x: x[None, ...], obs),
                        "action": action[None, ...],
                        "terminated": terminated[None, ..., None],
                        "truncated": truncated[None, ..., None],
                        "step_count": step_count[None, ..., None],
                        "reward": new_reward[None, ..., None],
                    }
                    data["observation"].pop("history", None)
                    for key, value in replay_context_fields.items():
                        data[key] = tree_map(lambda x: x[None, ...], value)
                    if history_context is not None:
                        data["history_context"] = history_context[None, ...]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][None, ...]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][None, ...]
                    if "aux_rewards" in new_info:
                        data["aux_rewards"] = {
                            k: v[None, ..., None] for (k, v) in new_info["aux_rewards"].items() if not k.startswith("_")
                        }
                else:
                    indexes = ~done
                    real_next_obs = tree_map(lambda x: _index_env_leaf(x, indexes), new_td)
                    _ = real_next_obs.pop("time")
                    _ = real_next_obs.pop("history", None)
                    data = {
                        "observation": tree_map(lambda x: x[indexes], obs),
                        "action": action[indexes],
                        "step_count": step_count[indexes],
                        "reward": new_reward[indexes].reshape(-1, 1),
                        "next": {
                            "observation": real_next_obs,
                            "terminated": new_terminated[indexes].reshape(-1, 1),
                            "truncated": new_truncated[indexes].reshape(-1, 1),
                        },
                    }
                    data["observation"].pop("history", None)
                    for key, value in replay_context_fields.items():
                        data[key] = tree_map(lambda x: _index_env_leaf(x, indexes), value)
                    if history_context is not None:
                        data["history_context"] = history_context[indexes]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][indexes]
                        data["next"]["qpos"] = new_info["qpos"][indexes]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][indexes]
                        data["next"]["qvel"] = new_info["qvel"][indexes]
                    if "aux_rewards" in new_info:
                        data["aux_rewards"] = {
                            k: v[indexes].reshape(-1, 1)
                            for (k, v) in new_info["aux_rewards"].items()
                            if not k.startswith("_")
                        }
            else:
                raise NotImplementedError("still some work to do for gymnasium < 1.0")
            replay_buffer["train"].extend(data)
            if len(replay_buffer["train"]) > 0 and t > self.cfg.num_seed_steps and update_agent_time_checker.check(t):
                update_agent_time_checker.update_last_step(t)
                for _ in range(self.cfg.num_agent_updates):
                    metrics = self.agent.update(replay_buffer, t)
                    if total_metrics is None:
                        num_metrics_updates = 1
                        total_metrics = {k: metrics[k].float().clone() for k in metrics.keys()}
                    else:
                        num_metrics_updates += 1
                        total_metrics = {k: total_metrics[k] + metrics[k].float() for k in metrics.keys()}
            if log_time_checker.check(t) and total_metrics is not None:
                log_time_checker.update_last_step(t)
                averaged_metrics = self._reduce_metrics(
                    {key: value / num_metrics_updates for (key, value) in total_metrics.items()}
                )
                m_dict = {}
                for k in sorted(averaged_metrics):
                    tmp = averaged_metrics[k]
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                m_dict["duration [minutes]"] = (time.time() - start_time) / 60
                m_dict["FPS"] = (1 if t == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                if self.is_rank0 and self.cfg.use_wandb:
                    wandb.log({f"train/{k}": v for (k, v) in m_dict.items()}, step=t)
                if self.is_rank0:
                    print(m_dict)
                total_metrics = None
                fps_start_time = time.time()
                m_dict["timestep"] = t
                if self.is_rank0 and self.train_logger is not None:
                    self.train_logger.log(m_dict)
            progb.update(self.global_step_increment)
            td = new_td
            terminated = new_terminated
            truncated = new_truncated
            if torch.is_tensor(new_terminated):
                done = torch.logical_or(new_terminated.ravel(), new_truncated.ravel())
            else:
                done = np.logical_or(new_terminated.ravel(), new_truncated.ravel())
            info = new_info
        train_env.close()

    def eval(self, t, replay_buffer):
        print(f"Starting evaluation at time {t}", flush=True)
        evaluation_results = {}
        evaluation_results = {}
        shared_reference_path = None
        if isinstance(self.cfg.env, HumanoidVerseIsaacConfig) and self.evaluations:
            reference_paths = [
                getattr(evaluation.cfg, "reference_motion_path", None) for evaluation in self.evaluations.values()
            ]
            if all((path is not None for path in reference_paths)):
                resolved_paths = {str(Path(path).expanduser().resolve()) for path in reference_paths}
                if len(resolved_paths) == 1:
                    shared_reference_path = resolved_paths.pop()
        dataset_context = (
            temporary_reference_motion_dataset(self.train_env, shared_reference_path)
            if shared_reference_path is not None
            else contextlib.nullcontext()
        )
        with dataset_context:
            for evaluation_name in self.evaluations.keys():
                logger = self.eval_loggers[evaluation_name]
                evaluation = self.evaluations[evaluation_name]
                evaluation_started = time.monotonic()
                _rank_print(f"starting {evaluation_name}")
                if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                    self.agent._model.to("cpu")
                self.agent._model.train(False)
                if isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                    (evaluation_metrics, wandb_dict) = evaluation.run(
                        timestep=t,
                        agent_or_model=self.agent,
                        replay_buffer=replay_buffer,
                        logger=logger,
                        env=self.train_env,
                    )
                else:
                    (evaluation_metrics, wandb_dict) = evaluation.run(
                        timestep=t, agent_or_model=self.agent, replay_buffer=replay_buffer, logger=logger
                    )
                if self.cfg.use_wandb and wandb_dict is not None:
                    wandb.log({f"eval/{evaluation_name}/{k}": v for (k, v) in wandb_dict.items()}, step=t)
                evaluation_results[evaluation_name] = evaluation_metrics
                _rank_print(f"completed {evaluation_name} in {time.monotonic() - evaluation_started:.1f}s")
        if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
            self.agent._model.to(self.cfg.agent.model.device)
        self.agent._model.train()
        return evaluation_results

    def save(self, time: int, replay_buffer: Dict[str, tp.Any]) -> None:
        print(f"Checkpointing at time {time}")
        self.agent.save(str(self.work_dir / CHECKPOINT_DIR_NAME))
        if self.cfg.checkpoint_buffer:
            replay_buffer["train"].save(self.work_dir / CHECKPOINT_DIR_NAME / "buffers" / "train")
        with (self.work_dir / CHECKPOINT_DIR_NAME / "train_status.json").open("w+") as f:
            json.dump({"time": time}, f, indent=4)
