import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import torch
from torch.utils._pytree import tree_map
from tqdm import tqdm

from humanoidverse.agents.evaluations.humanoidverse_isaac import (
    GROOT_MPJPE_BODY_NAMES,
    UPPER_BODY_MPJPE_BODY_NAMES,
    WRIST_TRACKING_BODY_NAMES,
    compute_cartesian_mpjpe_metrics,
    compute_upper_body_mpjpe_l,
    compute_wrist_tracking_metrics,
    root_se2_tracking_metrics,
)
from humanoidverse.playback import (
    GOAL_ALPHA,
    MujocoGoalGhostRenderer,
    VideoWriter,
    _check_actor_observation,
    _sync_goal_viewer,
    close_playback,
    load_playback,
    qpos_from_env,
    select_motion,
    validate_render_options,
)
from humanoidverse.tracking_utils import (
    _align_reference_positions,
    _align_reference_qpos_to_world,
    _align_reference_xyzw_quaternions,
    _expert_qpos_from_observation,
    _tracking_target_states,
    aggregate_future_latents,
)
from humanoidverse.utils.helpers import get_backward_observation


@torch.no_grad()
def main(
    model_folder: Path,
    data_path: Path | None = None,
    device: Literal["cpu", "cuda"] = "cpu",
    env_device: Literal["cpu", "cuda"] | None = None,
    simulator: Literal["mujoco", "isaacsim"] = "mujoco",
    headless: bool = True,
    motion_ids: tuple[int, ...] = (0,),
    episode_len: int = 450,
    future_frames: int | None = None,
    future_weighting: Literal["uniform", "gamma"] = "gamma",
    future_gamma: float | None = None,
    output_folder: Path | None = None,
    render_size: int = 256,
    render_camera_distance: float = 4.0,
    fps: int = 50,
    ghost_alpha: float = GOAL_ALPHA,
    save_mp4: bool = True,
    disable_dr: bool = True,
    disable_obs_noise: bool = True,
    seed: int = 0,
):
    if not motion_ids or episode_len < 0:
        raise ValueError("motion_ids must not be empty and episode_len must be non-negative")
    if future_frames is not None and future_frames < 0:
        raise ValueError("future_frames must be non-negative")
    if future_gamma is not None and not 0 < future_gamma <= 1:
        raise ValueError("future_gamma must be in (0, 1]")
    validate_render_options(fps, render_size, ghost_alpha)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output = output_folder or model_folder / "play_tracking"
    model, wrapped_env, root_height_obs = load_playback(
        model_folder, data_path, device, env_device, simulator, headless, disable_dr, disable_obs_noise
    )
    renderer = None
    try:
        env = wrapped_env._env
        if future_frames is None:
            future_frames = max(model.cfg.seq_length - 1, 0)
            future_weighting = "gamma" if model.cfg.seq_aggregation_discounted else "uniform"
        gamma = model.cfg.successor_discount if future_gamma is None else future_gamma
        body_ids = [env.body_names.index(name) for name in GROOT_MPJPE_BODY_NAMES]
        wrist_ids = [env.body_names.index(name) for name in WRIST_TRACKING_BODY_NAMES]
        upper_ids = [GROOT_MPJPE_BODY_NAMES.index(name) for name in UPPER_BODY_MPJPE_BODY_NAMES]
        if save_mp4:
            renderer = MujocoGoalGhostRenderer(render_size=render_size, camera_distance=render_camera_distance)
        viewer = getattr(env.simulator, "viewer", None)
        ghost = mujoco.MjData(env.simulator.model) if viewer is not None else None
        output.mkdir(parents=True, exist_ok=True)
        reports = []
        for motion_id in motion_ids:
            select_motion(env, motion_id)
            reference, states = get_backward_observation(env, 0, use_root_height_obs=root_height_obs)
            if reference["root_pose"].shape[0] < 2:
                raise ValueError(f"motion {motion_id} must have at least two frames")
            available_steps = reference["root_pose"].shape[0] - 1
            steps = min(episode_len, available_steps) if episode_len else available_steps
            end = steps + future_frames + 1
            latent = model.backward_map(tree_map(lambda x: x[1:end].to(model.device, dtype=torch.float32), reference))
            latent = aggregate_future_latents(
                model, latent, future_frames=future_frames, weighting=future_weighting, gamma=gamma
            )
            observation, _ = wrapped_env.reset(
                to_numpy=False, target_states=_tracking_target_states(wrapped_env, states, 1, simulator)
            )
            _check_actor_observation(model, observation)
            initial_qpos = qpos_from_env(wrapped_env)
            targets, rotation, translation, quaternion = _align_reference_qpos_to_world(
                _expert_qpos_from_observation(states), initial_qpos
            )
            target_pos = _align_reference_positions(states["ref_body_pos"], rotation, translation).cpu()
            target_rot = _align_reference_xyzw_quaternions(states["ref_body_rots"], quaternion).cpu()
            poses, body_pos, body_rot, root_pose, actions = [], [], [], [], []

            def record():
                poses.append(qpos_from_env(wrapped_env))
                body_pos.append(env.simulator._rigid_body_pos[0].cpu().clone())
                body_rot.append(env.simulator._rigid_body_rot[0].cpu().clone())
                root_pose.append(observation["root_pose"][0].cpu().clone())

            record()
            with VideoWriter(output / f"tracking_{motion_id}.mp4", fps, save_mp4) as video:
                video.append(renderer, poses[0], targets[0], ghost_alpha)
                for step in tqdm(range(steps), desc=f"tracking {motion_id}"):
                    _sync_goal_viewer(viewer, getattr(env.simulator, "model", None), ghost, targets[step + 1], False)
                    action = model.act(observation, latent[step : step + 1], mean=True)
                    observation, _, terminated, truncated, _ = wrapped_env.step(action, to_numpy=False)
                    if torch.as_tensor(terminated).any() or torch.as_tensor(truncated).any():
                        raise RuntimeError("Playback reset unexpectedly; check environment termination settings")
                    actions.append(action.cpu().numpy()[0])
                    record()
                    video.append(renderer, poses[-1], targets[step + 1], ghost_alpha)
            body_pos, body_rot, root_pose = map(torch.stack, (body_pos, body_rot, root_pose))
            target_pos, target_rot = target_pos[: steps + 1], target_rot[: steps + 1]
            body_metrics = compute_cartesian_mpjpe_metrics(body_pos[:, body_ids], target_pos[:, body_ids])
            metrics = {key.removeprefix("SONIC_"): value for key, value in body_metrics.items()}
            metrics["upper_body_mpjpe_l"] = compute_upper_body_mpjpe_l(
                body_pos[:, body_ids][:, upper_ids], target_pos[:, body_ids][:, upper_ids]
            )
            metrics.update(root_se2_tracking_metrics(root_pose, reference["root_pose"][: steps + 1].cpu()))
            metrics.update(
                compute_wrist_tracking_metrics(
                    body_pos[:, wrist_ids], body_rot[:, wrist_ids], target_pos[:, wrist_ids], target_rot[:, wrist_ids]
                )
            )
            report = {"motion_id": motion_id, "steps": steps, **{key: float(value) for key, value in metrics.items()}}
            reports.append(report)
            np.savez_compressed(
                output / f"tracking_{motion_id}.npz",
                qpos=np.stack(poses),
                target_qpos=targets[: steps + 1],
                action=np.stack(actions),
                root_pose=root_pose.numpy(),
                control_dt_s=float(env.dt),
            )
            (output / f"tracking_{motion_id}.json").write_text(json.dumps(report, indent=2) + "\n")
        summary = {
            "future_frames": future_frames,
            "future_weighting": future_weighting,
            "future_gamma": gamma,
            "motions": reports,
        }
        (output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved tracking playback to {output}")
    finally:
        close_playback(wrapped_env, renderer)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
