from typing import Literal

import mujoco
import numpy as np
import torch

from humanoidverse.playback import _quat_wxyz_yaw


def aggregate_future_latents(
    model, z: torch.Tensor, *, future_frames: int, weighting: Literal["uniform", "gamma"], gamma: float, dim: int = 0
) -> torch.Tensor:
    if future_frames < 0:
        raise ValueError("future_frames must be >= 0.")
    if weighting not in {"uniform", "gamma"}:
        raise ValueError(f"Unknown future weighting: {weighting}")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("future_gamma must satisfy 0 < gamma <= 1.")
    if future_frames == 0:
        return z
    dim %= z.ndim
    outputs = []
    weight_cache: dict[int, torch.Tensor] = {}
    for step in range(z.shape[dim]):
        window_length = min(future_frames + 1, z.shape[dim] - step)
        window = z.narrow(dim, step, window_length)
        if window_length not in weight_cache:
            offsets = torch.arange(window_length, device=z.device, dtype=torch.float32)
            weights = torch.ones_like(offsets) if weighting == "uniform" else gamma**offsets
            weight_cache[window_length] = weights / weights.sum().clamp_min(1e-12)
        weights = weight_cache[window_length]
        aggregate = model.aggregate_latent_window(window, dim=dim, weights=weights)
        outputs.append(aggregate)
    return torch.stack(outputs, dim=dim)


def _expert_qpos_from_observation(obs_dict: dict[str, torch.Tensor]) -> np.ndarray:
    return np.concatenate(
        [
            obs_dict["ref_body_pos"][:, 0].detach().cpu().numpy(),
            np.roll(obs_dict["ref_body_rots"][:, 0].detach().cpu().numpy(), 1, axis=-1),
            obs_dict["dof_pos"].detach().cpu().numpy(),
        ],
        axis=-1,
    )


def _tracking_target_states(
    wrapped_env, obs_dict: dict[str, torch.Tensor], num_envs: int, simulator: str
) -> dict[str, torch.Tensor]:
    ref_body_rots = obs_dict["ref_body_rots"][0, 0].clone()
    if simulator == "isaacsim":
        ref_body_rots = ref_body_rots[[3, 0, 1, 2]]
    ref_root_init_state = torch.cat(
        [
            obs_dict["ref_body_pos"][0, 0],
            ref_body_rots,
            obs_dict["ref_body_vels"][0, 0],
            obs_dict["ref_body_angular_vels"][0, 0],
        ]
    )
    dof_init_state = torch.zeros_like(wrapped_env._env.simulator.dof_state.view(num_envs, -1, 2)[0])
    dof_init_state[..., 0] = obs_dict["dof_pos"][0]
    dof_init_state[..., 1] = obs_dict["ref_dof_vel"][0]
    return {
        "dof_states": dof_init_state.unsqueeze(0).repeat(num_envs, 1, 1),
        "root_states": ref_root_init_state.unsqueeze(0).repeat(num_envs, 1),
    }


def _fixed_world_alignment(
    reference_initial_qpos: np.ndarray, rollout_initial_qpos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_initial_qpos = np.asarray(reference_initial_qpos, dtype=np.float64).ravel()
    rollout_initial_qpos = np.asarray(rollout_initial_qpos, dtype=np.float64).ravel()
    yaw_delta = _quat_wxyz_yaw(rollout_initial_qpos[3:7]) - _quat_wxyz_yaw(reference_initial_qpos[3:7])
    cos_yaw = np.cos(yaw_delta)
    sin_yaw = np.sin(yaw_delta)
    rotation = np.asarray([[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    translation = rollout_initial_qpos[:3] - rotation @ reference_initial_qpos[:3]
    delta_quaternion_wxyz = np.asarray([np.cos(0.5 * yaw_delta), 0.0, 0.0, np.sin(0.5 * yaw_delta)], dtype=np.float64)
    return (rotation, translation, delta_quaternion_wxyz)


def _align_reference_qpos_to_world(
    reference_qpos: np.ndarray, rollout_initial_qpos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference_qpos = np.asarray(reference_qpos, dtype=np.float64)
    if reference_qpos.ndim != 2 or reference_qpos.shape[1] < 7 or reference_qpos.shape[0] == 0:
        raise ValueError(f"reference_qpos must have shape [T, >=7], got {reference_qpos.shape}")
    (rotation, translation, delta_quaternion) = _fixed_world_alignment(reference_qpos[0], rollout_initial_qpos)
    aligned = reference_qpos.copy()
    aligned[:, :3] = reference_qpos[:, :3] @ rotation.T + translation
    for frame in range(aligned.shape[0]):
        mujoco.mju_mulQuat(aligned[frame, 3:7], delta_quaternion, reference_qpos[frame, 3:7])
    return (aligned.astype(np.float32), rotation, translation, delta_quaternion)


def _align_reference_positions(positions: torch.Tensor, rotation: np.ndarray, translation: np.ndarray) -> torch.Tensor:
    rotation_t = torch.as_tensor(rotation, device=positions.device, dtype=positions.dtype)
    translation_t = torch.as_tensor(translation, device=positions.device, dtype=positions.dtype)
    return torch.matmul(positions, rotation_t.T) + translation_t


def _align_reference_xyzw_quaternions(quaternions: torch.Tensor, delta_quaternion_wxyz: np.ndarray) -> torch.Tensor:
    original_shape = quaternions.shape
    flat = quaternions.detach().cpu().numpy().reshape(-1, 4)
    output = np.empty_like(flat)
    for index, quaternion_xyzw in enumerate(flat):
        result_wxyz = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(result_wxyz, delta_quaternion_wxyz, np.roll(quaternion_xyzw, 1))
        output[index] = np.roll(result_wxyz, -1)
    return torch.as_tensor(output.reshape(original_shape), device=quaternions.device, dtype=quaternions.dtype)
