from __future__ import annotations

import torch

from humanoidverse.agents.encoder_observation import CLEAN_ENCODER_OBS_KEY

ROOT_POSE_SCALE = 0.5


def _infer_body_count(privileged_state: torch.Tensor) -> int:
    feature_dim = privileged_state.shape[-1]
    max_body_count = (feature_dim + 3) // 15
    for body_count in range(max_body_count, 0, -1):
        base_dim = 3 * (body_count - 1) + 6 * body_count + 3 * body_count + 3 * body_count
        if feature_dim - base_dim >= 0:
            return body_count
    raise ValueError(f"Cannot infer body count from privileged_state dim {feature_dim}.")


def _yaw(root_pose: torch.Tensor) -> torch.Tensor:
    return torch.atan2(root_pose[..., 2], root_pose[..., 3])


def _xy_m(root_pose: torch.Tensor) -> torch.Tensor:
    return root_pose[..., :2] * ROOT_POSE_SCALE


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _rotate_vectors_by_yaw(vectors: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    sin_yaw = torch.sin(yaw)
    cos_yaw = torch.cos(yaw)
    x = cos_yaw * vectors[..., 0] - sin_yaw * vectors[..., 1]
    y = sin_yaw * vectors[..., 0] + cos_yaw * vectors[..., 1]
    return torch.stack([x, y, vectors[..., 2]], dim=-1)


def transform_privileged_state_root_pose_frame(
    privileged_state: torch.Tensor,
    old_root_pose: torch.Tensor,
    new_root_pose: torch.Tensor,
    *,
    body_count: int | None = None,
) -> torch.Tensor:
    """Move anchor-frame privileged body features from old_root_pose to new_root_pose."""
    body_count = body_count or _infer_body_count(privileged_state)
    if body_count <= 0:
        raise ValueError("body_count must be positive.")
    base_dim = 3 * (body_count - 1) + 6 * body_count + 3 * body_count + 3 * body_count
    prefix_dim = privileged_state.shape[-1] - base_dim
    if prefix_dim < 0:
        raise ValueError(
            f"privileged_state dim {privileged_state.shape[-1]} is too small for {body_count} bodies (expected at least {base_dim})."
        )
    old_root_pose = old_root_pose.to(device=privileged_state.device, dtype=privileged_state.dtype)
    new_root_pose = new_root_pose.to(device=privileged_state.device, dtype=privileged_state.dtype)
    converted = privileged_state.clone()
    leading_shape = privileged_state.shape[:-1]
    pos_start = prefix_dim
    rot_start = pos_start + 3 * (body_count - 1)
    vel_start = rot_start + 6 * body_count
    ang_vel_start = vel_start + 3 * body_count
    old_xy = _xy_m(old_root_pose).unsqueeze(-2)
    new_xy = _xy_m(new_root_pose).unsqueeze(-2)
    delta_yaw = (_yaw(new_root_pose) - _yaw(old_root_pose)).unsqueeze(-1)
    pos = privileged_state[..., pos_start:rot_start].reshape(*leading_shape, body_count - 1, 3)
    rel_pos = pos.clone()
    rel_pos[..., :2] = rel_pos[..., :2] - old_xy
    new_pos = _rotate_vectors_by_yaw(rel_pos, delta_yaw)
    new_pos[..., :2] = new_pos[..., :2] + new_xy
    converted[..., pos_start:rot_start] = new_pos.reshape(*leading_shape, -1)
    rot = privileged_state[..., rot_start:vel_start].reshape(*leading_shape, body_count, 6)
    tan = _rotate_vectors_by_yaw(rot[..., :3], delta_yaw)
    norm = _rotate_vectors_by_yaw(rot[..., 3:], delta_yaw)
    converted[..., rot_start:vel_start] = torch.cat([tan, norm], dim=-1).reshape(*leading_shape, -1)
    vel = privileged_state[..., vel_start:ang_vel_start].reshape(*leading_shape, body_count, 3)
    converted[..., vel_start:ang_vel_start] = _rotate_vectors_by_yaw(vel, delta_yaw).reshape(*leading_shape, -1)
    ang_vel = privileged_state[..., ang_vel_start : ang_vel_start + 3 * body_count].reshape(
        *leading_shape, body_count, 3
    )
    converted[..., ang_vel_start : ang_vel_start + 3 * body_count] = _rotate_vectors_by_yaw(ang_vel, delta_yaw).reshape(
        *leading_shape, -1
    )
    return converted


def _clone_obs_with_aligned_root_pose(
    obs: dict[str, torch.Tensor],
    old_root_pose: torch.Tensor,
    new_root_pose: torch.Tensor,
    *,
    privileged_state_world_frame: bool,
    body_count: int | None,
) -> dict[str, torch.Tensor]:
    aligned = dict(obs)
    aligned["root_pose"] = new_root_pose
    if privileged_state_world_frame and "privileged_state" in obs:
        aligned["privileged_state"] = transform_privileged_state_root_pose_frame(
            obs["privileged_state"], old_root_pose, new_root_pose, body_count=body_count
        )
    clean_obs = obs.get(CLEAN_ENCODER_OBS_KEY)
    if isinstance(clean_obs, dict):
        clean_old_root_pose = clean_obs.get("root_pose", old_root_pose)
        aligned[CLEAN_ENCODER_OBS_KEY] = _clone_obs_with_aligned_root_pose(
            clean_obs,
            clean_old_root_pose,
            new_root_pose,
            privileged_state_world_frame=privileged_state_world_frame,
            body_count=body_count,
        )
    return aligned


def replace_goal_root_pose(
    goal_obs: dict[str, torch.Tensor],
    new_root_pose: torch.Tensor,
    *,
    privileged_state_world_frame: bool = False,
    body_count: int | None = None,
) -> dict[str, torch.Tensor]:
    if "root_pose" not in goal_obs:
        return goal_obs
    return _clone_obs_with_aligned_root_pose(
        goal_obs,
        goal_obs["root_pose"],
        new_root_pose,
        privileged_state_world_frame=privileged_state_world_frame,
        body_count=body_count,
    )


def _clamp_xy_delta(delta_xy: torch.Tensor, max_dist_m: float) -> torch.Tensor:
    max_dist = max_dist_m / ROOT_POSE_SCALE
    dist = torch.linalg.norm(delta_xy, dim=-1, keepdim=True).clamp_min(1e-06)
    scale = torch.clamp(max_dist / dist, max=1.0)
    return delta_xy * scale


def _clamp_yaw_delta(delta_yaw: torch.Tensor, max_yaw_rad: float) -> torch.Tensor:
    return torch.clamp(_wrap(delta_yaw), min=-max_yaw_rad, max=max_yaw_rad)


def align_single_goal_root_pose(
    goal_obs: dict[str, torch.Tensor],
    current_root_pose: torch.Tensor | None,
    *,
    max_dist_m: float = 1.0,
    max_yaw_rad: float = torch.pi,
    privileged_state_world_frame: bool = False,
    body_count: int | None = None,
) -> dict[str, torch.Tensor]:
    if current_root_pose is None or "root_pose" not in goal_obs:
        return goal_obs
    goal_root_pose = goal_obs["root_pose"]
    current_root_pose = current_root_pose.to(device=goal_root_pose.device, dtype=goal_root_pose.dtype)
    if current_root_pose.ndim == 1:
        current_root_pose = current_root_pose.unsqueeze(0)
    delta_xy = _clamp_xy_delta(goal_root_pose[..., :2] - current_root_pose[..., :2], max_dist_m)
    aligned_xy = current_root_pose[..., :2] + delta_xy
    current_yaw = _yaw(current_root_pose)
    aligned_yaw = current_yaw + _clamp_yaw_delta(_yaw(goal_root_pose) - current_yaw, max_yaw_rad)
    aligned_root_pose = torch.cat(
        [aligned_xy, torch.sin(aligned_yaw).unsqueeze(-1), torch.cos(aligned_yaw).unsqueeze(-1)], dim=-1
    )
    return replace_goal_root_pose(
        goal_obs, aligned_root_pose, privileged_state_world_frame=privileged_state_world_frame, body_count=body_count
    )


def align_trajectory_goal_root_pose(
    goal_obs: dict[str, torch.Tensor],
    current_root_pose: torch.Tensor | None,
    *,
    traj_length: int,
    max_start_dist_m: float = 0.1,
    max_start_yaw_rad: float = torch.pi / 8,
    randomize_start: bool = False,
    privileged_state_world_frame: bool = False,
    body_count: int | None = None,
) -> dict[str, torch.Tensor]:
    """Place a goal trajectory near the robot while preserving its SE(2) shape.

    With ``randomize_start=True``, the first goal XY is sampled uniformly by
    area inside a disk centered on the current robot and its yaw offset is
    sampled uniformly from ``[-max_start_yaw_rad, max_start_yaw_rad]``.  The
    sampled SE(2) transform is applied once to the entire trajectory.
    """
    if current_root_pose is None or "root_pose" not in goal_obs:
        return goal_obs
    if max_start_dist_m < 0:
        raise ValueError("max_start_dist_m must be non-negative")
    if not 0 <= max_start_yaw_rad <= torch.pi:
        raise ValueError("max_start_yaw_rad must be in [0, pi]")
    root_pose = goal_obs["root_pose"]
    current_root_pose = current_root_pose.to(device=root_pose.device, dtype=root_pose.dtype)
    root_seq = root_pose.view(-1, traj_length, root_pose.shape[-1])
    if current_root_pose.ndim == 1:
        current_root_pose = current_root_pose.unsqueeze(0)
    start = root_seq[:, 0]
    current_yaw = _yaw(current_root_pose)
    start_yaw = _yaw(start)
    if randomize_start:
        random_xy = torch.rand((start.shape[0], 2), device=start.device, dtype=start.dtype)
        radius = torch.sqrt(random_xy[:, 0]) * (max_start_dist_m / ROOT_POSE_SCALE)
        angle = random_xy[:, 1] * (2 * torch.pi)
        start_delta_xy = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)
        yaw_unit = torch.rand(start.shape[0], device=start.device, dtype=start.dtype)
        start_delta_yaw = (yaw_unit * 2 - 1) * max_start_yaw_rad
    else:
        start_delta_xy = _clamp_xy_delta(start[..., :2] - current_root_pose[..., :2], max_start_dist_m)
        start_delta_yaw = _clamp_yaw_delta(start_yaw - current_yaw, max_start_yaw_rad)
    new_start_xy = current_root_pose[..., :2] + start_delta_xy
    new_start_yaw = _wrap(current_yaw + start_delta_yaw)
    yaw_offset = _wrap(new_start_yaw - start_yaw)
    relative_xy = root_seq[..., :2] - start[:, None, :2]
    cos_offset = torch.cos(yaw_offset)[:, None]
    sin_offset = torch.sin(yaw_offset)[:, None]
    rotated_x = cos_offset * relative_xy[..., 0] - sin_offset * relative_xy[..., 1]
    rotated_y = sin_offset * relative_xy[..., 0] + cos_offset * relative_xy[..., 1]
    aligned_xy = new_start_xy[:, None, :] + torch.stack([rotated_x, rotated_y], dim=-1)
    aligned_yaw = _wrap(_yaw(root_seq) + yaw_offset[:, None])
    aligned_root_pose = torch.cat(
        [aligned_xy, torch.sin(aligned_yaw).unsqueeze(-1), torch.cos(aligned_yaw).unsqueeze(-1)], dim=-1
    ).reshape_as(root_pose)
    return replace_goal_root_pose(
        goal_obs, aligned_root_pose, privileged_state_world_frame=privileged_state_world_frame, body_count=body_count
    )
