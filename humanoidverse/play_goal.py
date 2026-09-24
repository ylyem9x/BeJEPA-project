import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import torch
from tqdm import tqdm

from humanoidverse.agents.root_pose_goal import replace_goal_root_pose
from humanoidverse.playback import (
    GOAL_ALPHA,
    ROOT_POSE_SCALE,
    MujocoGoalGhostRenderer,
    PoseGoal,
    VideoWriter,
    _check_actor_observation,
    _goal_frame_to_mujoco_qpos,
    _goal_root_pose_to_world_qpos,
    _root_pose_metrics,
    _single_frame_observation,
    _sync_goal_viewer,
    close_playback,
    load_playback,
    qpos_from_env,
    reset_at_origin,
    select_motion,
    validate_render_options,
)
from humanoidverse.utils.helpers import get_backward_observation


def build_goals(model, env, root_height_obs, motion_ids, frame_ids, x_m, y_m, yaw_deg, velocity_multiplier):
    yaw = np.deg2rad(yaw_deg)
    root_pose = torch.tensor(
        [[x_m / ROOT_POSE_SCALE, y_m / ROOT_POSE_SCALE, np.sin(yaw), np.cos(yaw)]],
        device=model.device,
        dtype=torch.float32,
    )
    goals = []
    for motion_id in motion_ids:
        select_motion(env, motion_id)
        reference, states = get_backward_observation(
            env, 0, use_root_height_obs=root_height_obs, velocity_multiplier=velocity_multiplier
        )
        for frame_id in frame_ids:
            if not 0 <= frame_id < reference["root_pose"].shape[0]:
                raise ValueError(f"frame_id {frame_id} is outside motion {motion_id}")
            observation = replace_goal_root_pose(
                _single_frame_observation(reference, frame_id, model.device),
                root_pose,
                privileged_state_world_frame=env.privileged_state_world_frame,
                body_count=len(model.cfg.archi.latent_reconstruction_body_names) or None,
            )
            goals.append(
                PoseGoal(
                    name=f"motion_{motion_id}_frame_{frame_id}",
                    z=model.goal_inference(observation),
                    observation=observation,
                    qpos=_goal_frame_to_mujoco_qpos(states, frame_id),
                )
            )
    return goals


@torch.no_grad()
def main(
    model_folder: Path,
    data_path: Path | None = None,
    device: Literal["cpu", "cuda"] = "cpu",
    env_device: Literal["cpu", "cuda"] | None = None,
    simulator: Literal["mujoco", "isaacsim"] = "mujoco",
    headless: bool = True,
    source_motion_ids: tuple[int, ...] = (0,),
    source_frame_ids: tuple[int, ...] = (0,),
    goal_x_m: float = 1.0,
    goal_y_m: float = 0.0,
    goal_yaw_deg: float = 0.0,
    source_velocity_multiplier: float = 0.0,
    episode_len: int = 500,
    goal_hold_steps: int = 150,
    reset_on_goal_switch: bool = False,
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
    if not source_motion_ids or not source_frame_ids:
        raise ValueError("source_motion_ids and source_frame_ids must not be empty")
    if goal_hold_steps <= 0 or episode_len < 0:
        raise ValueError("goal_hold_steps must be positive and episode_len must be non-negative")
    validate_render_options(fps, render_size, ghost_alpha)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output = output_folder or model_folder / "play_goal"
    model, wrapped_env, root_height_obs = load_playback(
        model_folder, data_path, device, env_device, simulator, headless, disable_dr, disable_obs_noise
    )
    env = wrapped_env._env
    renderer = None
    try:
        goals = build_goals(
            model,
            env,
            root_height_obs,
            source_motion_ids,
            source_frame_ids,
            goal_x_m,
            goal_y_m,
            goal_yaw_deg,
            source_velocity_multiplier,
        )
        env.set_is_training()
        observation, _ = reset_at_origin(wrapped_env)
        _check_actor_observation(model, observation)
        anchor = qpos_from_env(wrapped_env)
        targets = [_goal_root_pose_to_world_qpos(anchor, goal.qpos, goal.observation["root_pose"]) for goal in goals]
        if save_mp4:
            renderer = MujocoGoalGhostRenderer(render_size=render_size, camera_distance=render_camera_distance)
        viewer = getattr(env.simulator, "viewer", None)
        ghost = mujoco.MjData(env.simulator.model) if viewer is not None else None
        metrics, poses, actions = [], [anchor], []
        steps = episode_len or goal_hold_steps * len(goals)
        with VideoWriter(output / "goal.mp4", fps, save_mp4) as video:
            video.append(renderer, anchor, targets[0], ghost_alpha)
            for step in tqdm(range(steps), desc="goal"):
                index = (step // goal_hold_steps) % len(goals)
                goal = goals[index]
                if reset_on_goal_switch and step > 0 and step % goal_hold_steps == 0:
                    observation, _ = reset_at_origin(wrapped_env)
                    anchor = qpos_from_env(wrapped_env)
                    targets = [_goal_root_pose_to_world_qpos(anchor, g.qpos, g.observation["root_pose"]) for g in goals]
                _sync_goal_viewer(viewer, getattr(env.simulator, "model", None), ghost, targets[index], False)
                action = model.act(observation, goal.z, mean=True)
                observation, _, terminated, truncated, _ = wrapped_env.step(action, to_numpy=False)
                if torch.as_tensor(terminated).any() or torch.as_tensor(truncated).any():
                    raise RuntimeError("Playback reset unexpectedly; check environment termination settings")
                metrics.append(
                    {
                        "step": step + 1,
                        "time_s": (step + 1) * float(env.dt),
                        "goal": goal.name,
                        **_root_pose_metrics(observation, goal.observation),
                    }
                )
                poses.append(qpos_from_env(wrapped_env))
                actions.append(action.cpu().numpy()[0])
                video.append(renderer, poses[-1], targets[index], ghost_alpha)
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        np.savez_compressed(
            output / "rollout.npz",
            qpos=np.stack(poses),
            action=np.stack(actions),
            goal_qpos=np.stack(targets),
            goal_z=torch.cat([g.z for g in goals]).cpu().numpy(),
            control_dt_s=float(env.dt),
        )
        print(f"Saved goal playback to {output}")
    finally:
        close_playback(wrapped_env, renderer)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
