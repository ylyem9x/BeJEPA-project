import dataclasses
import typing as tp
from typing import Any, Dict

import torch
from torch.utils._pytree import tree_map
from tqdm import tqdm

from humanoidverse.agents.envs.humanoidverse_isaac import (
    ROOT_POSE_SCALE,
    HumanoidVerseIsaacConfig,
    HumanoidVerseVectorEnv,
)
from humanoidverse.agents.root_pose_goal import align_single_goal_root_pose

from .base import BaseEvalConfig, extract_model
from .humanoidverse_isaac import get_backward_observation, temporary_reference_motion_dataset


def _goal_input_keys(model) -> list[str]:
    explicit_keys = getattr(model, "goal_observation_keys", None)
    if explicit_keys is not None:
        return list(explicit_keys)
    archi = getattr(getattr(model, "cfg", None), "archi", None)
    backward_archi = getattr(archi, "b", None)
    input_filter = getattr(backward_archi, "input_filter", None)
    key = getattr(input_filter, "key", None)
    if key is None:
        return ["state", "privileged_state", "root_pose"]
    return list(key) if isinstance(key, (list, tuple)) else [key]


def _privileged_state_body_count(model) -> int | None:
    archi = getattr(getattr(model, "cfg", None), "archi", None)
    body_names = getattr(archi, "latent_reconstruction_body_names", None)
    return len(body_names) if body_names else None


def _l2_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x.float() - y.to(device=x.device, dtype=torch.float32), dim=-1)


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _root_error(
    root_pose: torch.Tensor, goal_root_pose: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    goal_root_pose = goal_root_pose.to(device=root_pose.device, dtype=root_pose.dtype)
    xy_error = torch.linalg.norm((root_pose[:, :2] - goal_root_pose[:, :2]) * ROOT_POSE_SCALE, dim=-1)
    yaw = torch.atan2(root_pose[:, 2], root_pose[:, 3])
    goal_yaw = torch.atan2(goal_root_pose[:, 2], goal_root_pose[:, 3])
    yaw_error = torch.abs(_wrap(yaw - goal_yaw))
    combined = torch.sqrt(xy_error.pow(2) + yaw_error.pow(2))
    return (combined, xy_error, yaw_error)


@dataclasses.dataclass(frozen=True)
class _WristEvalLayout:
    body_indices: tuple[int, ...]
    num_bodies: int
    prefix_dim: int


def _rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    x_axis = torch.nn.functional.normalize(rotation_6d[..., :3].float(), dim=-1, eps=1e-06)
    z_axis = torch.nn.functional.normalize(rotation_6d[..., 3:6].float(), dim=-1, eps=1e-06)
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis = torch.nn.functional.normalize(y_axis, dim=-1, eps=1e-06)
    z_axis = torch.linalg.cross(x_axis, y_axis, dim=-1)
    return torch.stack((x_axis, y_axis, z_axis), dim=-1)


def _rotation_6d_error_rad(rotation_6d: torch.Tensor, target_rotation_6d: torch.Tensor) -> torch.Tensor:
    rotation = _rotation_6d_to_matrix(rotation_6d)
    target_rotation = _rotation_6d_to_matrix(target_rotation_6d.to(device=rotation_6d.device, dtype=rotation_6d.dtype))
    relative_rotation = torch.matmul(rotation.transpose(-2, -1), target_rotation)
    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    return torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))


def _resolve_wrist_eval_layout(
    env: HumanoidVerseVectorEnv, privileged_state: torch.Tensor, wrist_body_names: list[str]
) -> _WristEvalLayout | None:
    if not wrist_body_names:
        return None
    body_names = getattr(env._env.simulator, "_body_list", None)
    if body_names is None:
        body_names = getattr(env._env.simulator, "body_names", None)
    if body_names is None:
        raise ValueError("Wrist goal evaluation requires simulator body names.")
    body_names = list(body_names)
    missing_body_names = [name for name in wrist_body_names if name not in body_names]
    if missing_body_names:
        raise ValueError(f"Wrist body names not found in simulator body list: {missing_body_names}")
    body_indices = tuple((body_names.index(name) for name in wrist_body_names))
    min_num_bodies = max(body_indices) + 1
    feature_dim = privileged_state.shape[-1]
    for num_bodies in range(len(body_names), min_num_bodies - 1, -1):
        body_feature_dim = 15 * num_bodies - 3
        prefix_dim = feature_dim - body_feature_dim
        if prefix_dim >= 0:
            return _WristEvalLayout(body_indices=body_indices, num_bodies=num_bodies, prefix_dim=prefix_dim)
    raise ValueError(
        f"Cannot infer max_local_self layout for wrist eval: feature_dim={feature_dim}, num_bodies={len(body_names)}, wrist_indices={body_indices}."
    )


def _extract_max_local_body_pose(
    privileged_state: torch.Tensor, layout: _WristEvalLayout
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_start = layout.prefix_dim
    rot_start = pos_start + 3 * (layout.num_bodies - 1)
    body_positions = []
    body_rotations = []
    for body_index in layout.body_indices:
        if body_index == 0:
            body_pos = torch.zeros(
                (*privileged_state.shape[:-1], 3), device=privileged_state.device, dtype=privileged_state.dtype
            )
        else:
            body_pos_start = pos_start + 3 * (body_index - 1)
            body_pos = privileged_state[..., body_pos_start : body_pos_start + 3]
        body_rot_start = rot_start + 6 * body_index
        body_rot = privileged_state[..., body_rot_start : body_rot_start + 6]
        body_positions.append(body_pos)
        body_rotations.append(body_rot)
    return (torch.stack(body_positions, dim=-2), torch.stack(body_rotations, dim=-2))


def _wrist_pose_error(
    observation: dict[str, torch.Tensor],
    goal_observation: dict[str, torch.Tensor],
    layout: _WristEvalLayout,
    *,
    pos_scale_m: float,
    rot_scale_rad: float,
    pos_weight: float,
    rot_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (wrist_pos, wrist_rot) = _extract_max_local_body_pose(observation["privileged_state"], layout)
    (goal_wrist_pos, goal_wrist_rot) = _extract_max_local_body_pose(
        goal_observation["privileged_state"].to(device=wrist_pos.device, dtype=wrist_pos.dtype), layout
    )
    pos_error_m = torch.linalg.norm(wrist_pos.float() - goal_wrist_pos.float(), dim=-1)
    rot_error_rad = _rotation_6d_error_rad(wrist_rot, goal_wrist_rot)
    wrist_score = torch.sqrt(
        pos_weight * (pos_error_m / pos_scale_m).pow(2) + rot_weight * (rot_error_rad / rot_scale_rad).pow(2)
    )
    return (pos_error_m.mean(dim=-1), rot_error_rad.mean(dim=-1), wrist_score.mean(dim=-1))


def _obs_error(observation: dict[str, torch.Tensor], goal_observation: dict[str, torch.Tensor], model) -> torch.Tensor:
    current_values = []
    goal_values = []
    for key in _goal_input_keys(model):
        if key == "root_pose" or key not in observation or key not in goal_observation:
            continue
        current = observation[key]
        goal = goal_observation[key].to(device=current.device, dtype=current.dtype)
        if current.shape != goal.shape:
            continue
        current_values.append(current.reshape(current.shape[0], -1).float())
        goal_values.append(goal.reshape(goal.shape[0], -1).float())
    if not current_values:
        raise ValueError("No common non-root goal observation keys were found for obs_error.")
    return _l2_error(torch.cat(current_values, dim=-1), torch.cat(goal_values, dim=-1))


def _reset_to_origin_random_pose(env: HumanoidVerseVectorEnv) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    env.reset(to_numpy=False)
    isaac_env = env._env
    root_state = isaac_env.simulator.robot_root_states.clone().detach()
    dof_state = isaac_env.simulator.dof_state.clone().detach()
    root_state[:, 0:2] = 0.0
    root_state[:, 7:13] = 0.0
    dof_state[..., 1] = 0.0
    return env.reset(to_numpy=False, target_states={"root_states": root_state, "dof_states": dof_state})


def _sample_expert_goal(
    replay_buffer: dict,
    batch_size: int,
    model,
    current_root_pose: torch.Tensor,
    max_goal_dist_m: float,
    max_goal_yaw_rad: float,
) -> dict[str, torch.Tensor]:
    if replay_buffer is None or "expert_slicer" not in replay_buffer:
        raise ValueError("Goal evaluation requires replay_buffer['expert_slicer'].")
    batch = replay_buffer["expert_slicer"].sample(batch_size, seq_length=1)
    goal_obs = tree_map(lambda x: x.to(model.device, dtype=torch.float32), batch["next"]["observation"])
    current_root_pose = current_root_pose.to(model.device, dtype=torch.float32)
    return align_single_goal_root_pose(
        goal_obs,
        current_root_pose,
        max_dist_m=max_goal_dist_m,
        max_yaw_rad=max_goal_yaw_rad,
        privileged_state_world_frame=bool(getattr(model.cfg, "privileged_state_world_frame", False)),
        body_count=_privileged_state_body_count(model),
    )


@torch.no_grad()
def _sample_reference_motion_goal(
    env: HumanoidVerseVectorEnv,
    batch_size: int,
    model,
    current_root_pose: torch.Tensor,
    max_goal_dist_m: float,
    max_goal_yaw_rad: float,
    sampled_frames: list[dict[str, int]] | None = None,
) -> dict[str, torch.Tensor]:
    """Sample goal frames uniformly from the active evaluation motion library."""
    motion_count = int(env._env._motion_lib._num_unique_motions)
    if motion_count <= 0:
        raise ValueError("reference evaluation motion library is empty")
    device = current_root_pose.device
    sampled_motion_ids = torch.randint(motion_count, (batch_size,), device=device)
    goal_observation: dict[str, torch.Tensor] = {}
    for motion_id in sampled_motion_ids.unique().tolist():
        (trajectory, _) = get_backward_observation(
            env._env,
            int(motion_id),
            include_last_action=env.include_last_action,
            privileged_state_world_frame=env.privileged_state_world_frame,
        )
        batch_indices = (sampled_motion_ids == motion_id).nonzero(as_tuple=False).flatten()
        trajectory_length = next(iter(trajectory.values())).shape[0]
        frame_indices = torch.randint(trajectory_length, (batch_indices.numel(),), device=device)
        if sampled_frames is not None:
            sampled_frames.extend(
                (
                    {"batch_index": int(batch_index), "motion_id": int(motion_id), "frame_id": int(frame_index)}
                    for (batch_index, frame_index) in zip(batch_indices.tolist(), frame_indices.tolist())
                )
            )
        for key, value in trajectory.items():
            if not torch.is_tensor(value):
                continue
            if key not in goal_observation:
                goal_observation[key] = torch.empty(
                    (batch_size, *value.shape[1:]), device=value.device, dtype=value.dtype
                )
            goal_observation[key].index_copy_(
                0, batch_indices.to(value.device), value.index_select(0, frame_indices.to(value.device))
            )
    return align_single_goal_root_pose(
        goal_observation,
        current_root_pose.to(model.device, dtype=torch.float32),
        max_dist_m=max_goal_dist_m,
        max_yaw_rad=max_goal_yaw_rad,
        privileged_state_world_frame=bool(getattr(model.cfg, "privileged_state_world_frame", False)),
        body_count=_privileged_state_body_count(model),
    )


@dataclasses.dataclass
class _GoalEvalRolloutStats:
    wrist_layout: _WristEvalLayout | None = None
    wrist_pos_scale_m: float = 0.05
    wrist_rot_scale_rad: float = 0.17453292519943295
    wrist_pos_weight: float = 1.0
    wrist_rot_weight: float = 1.0
    joint_pos_error: list[torch.Tensor] = dataclasses.field(default_factory=list)
    joint_vel_error: list[torch.Tensor] = dataclasses.field(default_factory=list)
    obs_error: list[torch.Tensor] = dataclasses.field(default_factory=list)
    root_error: list[torch.Tensor] = dataclasses.field(default_factory=list)
    root_xy_error_m: list[torch.Tensor] = dataclasses.field(default_factory=list)
    root_yaw_error_rad: list[torch.Tensor] = dataclasses.field(default_factory=list)
    wrist_pos_error_m: list[torch.Tensor] = dataclasses.field(default_factory=list)
    wrist_rot_error_rad: list[torch.Tensor] = dataclasses.field(default_factory=list)
    wrist_pose_score: list[torch.Tensor] = dataclasses.field(default_factory=list)
    terminated: list[torch.Tensor] = dataclasses.field(default_factory=list)
    truncated: list[torch.Tensor] = dataclasses.field(default_factory=list)

    def add(
        self,
        *,
        observation: dict[str, torch.Tensor],
        goal_observation: dict[str, torch.Tensor],
        model,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        num_dof = (goal_observation["state"].shape[-1] - 6) // 2
        self.joint_pos_error.append(
            _l2_error(observation["state"][:, :num_dof], goal_observation["state"][:, :num_dof])
        )
        self.joint_vel_error.append(
            _l2_error(
                observation["state"][:, num_dof : 2 * num_dof], goal_observation["state"][:, num_dof : 2 * num_dof]
            )
        )
        self.obs_error.append(_obs_error(observation, goal_observation, model))
        (root_error, root_xy_error_m, root_yaw_error_rad) = _root_error(
            observation["root_pose"], goal_observation["root_pose"]
        )
        self.root_error.append(root_error)
        self.root_xy_error_m.append(root_xy_error_m)
        self.root_yaw_error_rad.append(root_yaw_error_rad)
        if self.wrist_layout is not None:
            (wrist_pos_error_m, wrist_rot_error_rad, wrist_pose_score) = _wrist_pose_error(
                observation,
                goal_observation,
                self.wrist_layout,
                pos_scale_m=self.wrist_pos_scale_m,
                rot_scale_rad=self.wrist_rot_scale_rad,
                pos_weight=self.wrist_pos_weight,
                rot_weight=self.wrist_rot_weight,
            )
            self.wrist_pos_error_m.append(wrist_pos_error_m)
            self.wrist_rot_error_rad.append(wrist_rot_error_rad)
            self.wrist_pose_score.append(wrist_pose_score)
        self.terminated.append(terminated.float())
        self.truncated.append(truncated.float())

    def aggregate(self) -> Dict[str, float]:
        metrics = {}
        for name in ("joint_pos_error", "joint_vel_error", "obs_error", "root_error", "root_xy_error_m"):
            value = torch.stack(getattr(self, name), dim=0)
            metrics[f"{name}_mean"] = float(value.mean().item())
            metrics[f"{name}_min"] = float(value.min(dim=0).values.mean().item())
        metrics["root_yaw_error_deg_mean"] = float(
            torch.rad2deg(torch.stack(self.root_yaw_error_rad, dim=0)).mean().item()
        )
        metrics["root_yaw_error_deg_min"] = float(
            torch.rad2deg(torch.stack(self.root_yaw_error_rad, dim=0)).min(dim=0).values.mean().item()
        )
        if self.wrist_pos_error_m:
            metrics["wrist_pos_error_m_min"] = float(
                torch.stack(self.wrist_pos_error_m, dim=0).min(dim=0).values.mean().item()
            )
            metrics["wrist_rot_error_deg_min"] = float(
                torch.rad2deg(torch.stack(self.wrist_rot_error_rad, dim=0)).min(dim=0).values.mean().item()
            )
            metrics["wrist_pose_score_min"] = float(
                torch.stack(self.wrist_pose_score, dim=0).min(dim=0).values.mean().item()
            )
        metrics["terminated_fraction"] = float(torch.stack(self.terminated, dim=0).mean().item())
        metrics["truncated_fraction"] = float(torch.stack(self.truncated, dim=0).mean().item())
        return metrics


class HumanoidVerseIsaacGoalEvaluationConfig(BaseEvalConfig):
    name: tp.Literal["goal_eval"] = "goal_eval"
    name_in_logs: str = "goal_eval"
    env: HumanoidVerseIsaacConfig | None = None
    num_envs: int = 1024
    reference_motion_path: str | None = None
    rollout_steps: int = 200
    max_goal_dist_m: float = 1.0
    max_goal_yaw_rad: float = 3.141592653589793
    wrist_body_names: list[str] = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    wrist_score_pos_scale_m: float = 0.05
    wrist_score_rot_scale_rad: float = 0.17453292519943295
    wrist_score_pos_weight: float = 1.0
    wrist_score_rot_weight: float = 1.0
    disable_tqdm: bool = True

    def build(self):
        return HumanoidVerseIsaacGoalEvaluation(self)

    @classmethod
    def requires_replay_buffer(self):
        return True


class HumanoidVerseIsaacGoalEvaluation:
    def __init__(self, config: HumanoidVerseIsaacGoalEvaluationConfig):
        self.cfg = config

    def run(
        self, *, timestep, agent_or_model, replay_buffer, logger, env: HumanoidVerseVectorEnv | None = None, **kwargs
    ):
        if self.cfg.rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be > 0, got {self.cfg.rollout_steps}.")
        if self.cfg.wrist_body_names and (
            self.cfg.wrist_score_pos_scale_m <= 0 or self.cfg.wrist_score_rot_scale_rad <= 0
        ):
            raise ValueError("wrist_score_pos_scale_m and wrist_score_rot_scale_rad must be > 0.")
        if self.cfg.wrist_score_pos_weight < 0 or self.cfg.wrist_score_rot_weight < 0:
            raise ValueError("wrist_score_pos_weight and wrist_score_rot_weight must be >= 0.")
        if env is None:
            if self.cfg.env is None:
                raise ValueError("Either env or cfg.env must be provided")
            (env, _) = self.cfg.env.build(num_envs=self.cfg.num_envs)
        elif self.cfg.env is not None:
            raise ValueError("Both env and cfg.env are provided, please provide only one.")
        with temporary_reference_motion_dataset(env, self.cfg.reference_motion_path):
            return self._run_with_selected_dataset(
                timestep=timestep, agent_or_model=agent_or_model, replay_buffer=replay_buffer, logger=logger, env=env
            )

    def _run_with_selected_dataset(
        self, *, timestep, agent_or_model, replay_buffer, logger, env: HumanoidVerseVectorEnv
    ):
        model = extract_model(agent_or_model)
        actor = agent_or_model if hasattr(agent_or_model, "act") else model
        with torch.no_grad():
            (observation, _) = _reset_to_origin_random_pose(env)
            if self.cfg.reference_motion_path is None:
                goal_observation = _sample_expert_goal(
                    replay_buffer=replay_buffer,
                    batch_size=env.num_envs,
                    model=model,
                    current_root_pose=observation["root_pose"],
                    max_goal_dist_m=self.cfg.max_goal_dist_m,
                    max_goal_yaw_rad=self.cfg.max_goal_yaw_rad,
                )
            else:
                goal_observation = _sample_reference_motion_goal(
                    env=env,
                    batch_size=env.num_envs,
                    model=model,
                    current_root_pose=observation["root_pose"],
                    max_goal_dist_m=self.cfg.max_goal_dist_m,
                    max_goal_yaw_rad=self.cfg.max_goal_yaw_rad,
                )
            goal_z = model.goal_inference(goal_observation)
            wrist_layout = _resolve_wrist_eval_layout(env, observation["privileged_state"], self.cfg.wrist_body_names)
            stats = _GoalEvalRolloutStats(
                wrist_layout=wrist_layout,
                wrist_pos_scale_m=self.cfg.wrist_score_pos_scale_m,
                wrist_rot_scale_rad=self.cfg.wrist_score_rot_scale_rad,
                wrist_pos_weight=self.cfg.wrist_score_pos_weight,
                wrist_rot_weight=self.cfg.wrist_score_rot_weight,
            )
            iterator = tqdm(range(self.cfg.rollout_steps), desc="Goal Evaluation", disable=self.cfg.disable_tqdm)
            for _ in iterator:
                action = actor.act(observation, goal_z, mean=True)
                (observation, _, terminated, truncated, _) = env.step(action, to_numpy=False)
                stats.add(
                    observation=observation,
                    goal_observation=goal_observation,
                    model=model,
                    terminated=terminated,
                    truncated=truncated,
                )
        wandb_dict = stats.aggregate()
        if logger is not None:
            row = dict(wandb_dict)
            row["timestep"] = timestep
            logger.log(row)
        env._env.set_is_training()
        return ({"goal_eval": wandb_dict}, wandb_dict)
