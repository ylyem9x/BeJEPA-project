import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from humanoidverse.agents.encoder_observation import ensure_clean_encoder_obs_enabled_for_model
from humanoidverse.agents.envs.humanoidverse_isaac import ROOT_POSE_SCALE, HumanoidVerseIsaacConfig
from humanoidverse.agents.jepa_successor.model import (
    JepaSuccessorCprAuxModel,
    JepaSuccessorCprModel,
    JepaSuccessorModel,
)
from humanoidverse.prepare_data import ensure_motion_data

ASSET_PATH = Path(__file__).parent / "data/robots/g1/scene_29dof_freebase_mujoco.xml"
DEFAULT_DATA_PATH = Path(__file__).parent / "data/lafan_29dof.pkl"
GOAL_TINT = np.array([80.0, 190.0, 255.0], dtype=np.float32)
GOAL_ALPHA = 0.35


@dataclass
class PoseGoal:
    name: str
    z: torch.Tensor
    observation: dict[str, torch.Tensor]
    qpos: np.ndarray | None = None
    align_visual_root_to_current: bool = False


def _single_frame_observation(observation: Mapping[str, object], frame_idx: int, device: str) -> dict[str, object]:
    def _slice(value: object, path: str) -> object:
        if isinstance(value, torch.Tensor):
            return value[frame_idx : frame_idx + 1].to(device=device, dtype=torch.float32).clone()
        if isinstance(value, Mapping):
            return {key: _slice(child, f"{path}.{key}") for (key, child) in value.items()}
        raise TypeError(
            f"Observation value at '{path}' must be a torch.Tensor or nested mapping, got {type(value).__name__}."
        )

    return {key: _slice(value, str(key)) for (key, value) in observation.items()}


def _goal_frame_to_mujoco_qpos(goal_obs_dict: dict[str, torch.Tensor], frame_idx: int) -> np.ndarray:
    root_pos = goal_obs_dict["ref_body_pos"][frame_idx, 0].detach().cpu().numpy()
    root_rot_xyzw = goal_obs_dict["ref_body_rots"][frame_idx, 0].detach().cpu().numpy()
    dof_pos = goal_obs_dict["dof_pos"][frame_idx].detach().cpu().numpy()
    return np.concatenate([root_pos, np.roll(root_rot_xyzw, 1), dof_pos], axis=-1)


def _quat_wxyz_yaw(quat: np.ndarray) -> float:
    (w, x, y, z) = np.asarray(quat, dtype=np.float64)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _yaw_to_wxyz(yaw: float) -> np.ndarray:
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float32)


def _goal_root_pose_to_world_qpos(
    anchor_qpos: np.ndarray, goal_qpos: np.ndarray | None, goal_root_pose: torch.Tensor | None
) -> np.ndarray | None:
    if goal_qpos is None or goal_root_pose is None:
        return None
    qpos = np.asarray(goal_qpos, dtype=np.float32).copy()
    anchor_qpos = np.asarray(anchor_qpos, dtype=np.float32).ravel()
    goal_pose = goal_root_pose.detach().cpu().numpy().reshape(-1)
    anchor_yaw = _quat_wxyz_yaw(anchor_qpos[3:7])
    x_rel = float(goal_pose[0]) * ROOT_POSE_SCALE
    y_rel = float(goal_pose[1]) * ROOT_POSE_SCALE
    yaw_rel = float(np.arctan2(goal_pose[2], goal_pose[3]))
    world_dx = np.cos(anchor_yaw) * x_rel - np.sin(anchor_yaw) * y_rel
    world_dy = np.sin(anchor_yaw) * x_rel + np.cos(anchor_yaw) * y_rel
    qpos[0] = anchor_qpos[0] + world_dx
    qpos[1] = anchor_qpos[1] + world_dy
    qpos[3:7] = _sync_world_yaw(qpos[3:7], _yaw_to_wxyz(anchor_yaw + yaw_rel))
    return qpos


def _sync_world_yaw(goal_quat: np.ndarray, ref_quat: np.ndarray) -> np.ndarray:
    (goal_w, goal_x, goal_y, goal_z) = np.asarray(goal_quat, dtype=np.float64)
    (ref_w, ref_x, ref_y, ref_z) = np.asarray(ref_quat, dtype=np.float64)
    goal_yaw = np.arctan2(2.0 * (goal_w * goal_z + goal_x * goal_y), 1.0 - 2.0 * (goal_y * goal_y + goal_z * goal_z))
    ref_yaw = np.arctan2(2.0 * (ref_w * ref_z + ref_x * ref_y), 1.0 - 2.0 * (ref_y * ref_y + ref_z * ref_z))
    yaw_delta_quat = np.zeros(4, dtype=np.float64)
    aligned_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_axisAngle2Quat(yaw_delta_quat, np.array([0.0, 0.0, 1.0]), ref_yaw - goal_yaw)
    mujoco.mju_mulQuat(aligned_quat, yaw_delta_quat, np.asarray(goal_quat, dtype=np.float64))
    return aligned_quat.astype(np.float32)


def _sync_goal_viewer(
    viewer, model, goal_data: mujoco.MjData | None, goal_qpos: np.ndarray | None, align_root_to_current: bool
) -> None:
    if viewer is None or goal_data is None or goal_qpos is None or (getattr(viewer, "user_scn", None) is None):
        return
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        goal_data.qpos[:] = np.asarray(goal_qpos).ravel()
        if align_root_to_current:
            goal_data.qpos[0:3] = viewer.d.qpos[0:3]
            goal_data.qpos[3:7] = _sync_world_yaw(goal_data.qpos[3:7], viewer.d.qpos[3:7])
        goal_data.qvel[:] = 0
        mujoco.mj_forward(model, goal_data)
        mujoco.mjv_addGeoms(model, goal_data, viewer.opt, viewer.perturb, int(mujoco.mjtCatBit.mjCAT_ALL), scene)
        for i in range(scene.ngeom):
            geom = scene.geoms[i]
            if geom.objtype != int(mujoco.mjtObj.mjOBJ_GEOM) or geom.objid < 0 or model.geom_bodyid[geom.objid] <= 0:
                geom.rgba[3] = 0.0
                continue
            geom.rgba[:3] = 0.35 * geom.rgba[:3] + 0.65 * (GOAL_TINT / 255.0)
            geom.rgba[3] = GOAL_ALPHA


def _root_pose_metrics(
    observation: dict[str, torch.Tensor], goal_observation: dict[str, torch.Tensor]
) -> dict[str, float]:
    if "root_pose" not in observation or "root_pose" not in goal_observation:
        return {}
    current = observation["root_pose"][:1].detach()
    target = goal_observation["root_pose"][:1].to(current.device).detach()
    xy_error = (current[:, :2] - target[:, :2]) * ROOT_POSE_SCALE
    (sin_current, cos_current) = (current[:, 2], current[:, 3])
    (sin_target, cos_target) = (target[:, 2], target[:, 3])
    yaw_error = torch.atan2(
        sin_current * cos_target - cos_current * sin_target, cos_current * cos_target + sin_current * sin_target
    )
    return {
        "root_x_error_m": float(xy_error[0, 0].item()),
        "root_y_error_m": float(xy_error[0, 1].item()),
        "root_xy_error_m": float(torch.linalg.norm(xy_error, dim=-1)[0].item()),
        "root_yaw_error_deg": float(torch.rad2deg(yaw_error)[0].item()),
    }


def _actor_input_keys(model) -> list[str]:
    explicit_keys = getattr(model, "actor_observation_keys", None)
    if explicit_keys is not None:
        return list(explicit_keys)
    archi = getattr(getattr(model, "cfg", None), "archi", None)
    actor = getattr(archi, "actor", None)
    input_filter = getattr(actor, "input_filter", None)
    key = getattr(input_filter, "key", None)
    if key is None:
        return list(model.obs_space.keys()) if hasattr(model.obs_space, "keys") else []
    return list(key) if isinstance(key, (list, tuple)) else [key]


def _check_actor_observation(model, observation: dict[str, torch.Tensor]) -> None:
    missing = [key for key in _actor_input_keys(model) if key not in observation]
    if missing:
        raise KeyError(f"Actor observation is missing keys {missing}; available keys are {list(observation.keys())}.")
    for key in _actor_input_keys(model):
        expected_dim = model.obs_space[key].shape[0]
        actual_dim = observation[key].shape[-1]
        if actual_dim != expected_dim:
            raise ValueError(f"Actor observation key '{key}' has dim {actual_dim}, expected {expected_dim}.")


def _append_env_overrides(
    config: dict,
    headless: bool,
    simulator: str,
    disable_dr: bool,
    disable_obs_noise: bool,
    env_device: str | None = None,
) -> None:
    config["env"].setdefault("hydra_overrides", [])
    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=1000000000")
    config["env"]["hydra_overrides"].append(f"env.config.headless={headless}")
    config["env"]["hydra_overrides"].append(f"simulator={simulator}")
    config["env"]["hydra_overrides"].extend(
        [
            "env.config.termination.terminate_by_contact=False",
            "env.config.termination.terminate_by_gravity=False",
            "env.config.termination.terminate_by_low_height=False",
            "env.config.termination.terminate_when_motion_end=False",
            "env.config.termination.terminate_when_motion_far=False",
            "env.config.termination.terminate_when_close_to_dof_pos_limit=False",
            "env.config.termination.terminate_when_close_to_dof_vel_limit=False",
            "env.config.termination.terminate_when_close_to_torque_limit=False",
        ]
    )
    config["env"]["disable_domain_randomization"] = disable_dr
    config["env"]["disable_obs_noise"] = disable_obs_noise
    if env_device is not None:
        config["env"]["device"] = env_device


def load_playback(model_folder, data_path, device, env_device, simulator, headless, disable_dr, disable_obs_noise):
    model_folder = Path(model_folder)
    model_dir = model_folder / "checkpoint/model"
    model_config = json.loads((model_dir / "config.json").read_text())
    classes = {cls.__name__: cls for cls in (JepaSuccessorModel, JepaSuccessorCprModel, JepaSuccessorCprAuxModel)}
    if model_config["name"] not in classes:
        raise ValueError(f"Unsupported model: {model_config['name']}")
    model = classes[model_config["name"]].load(model_dir, device=device)
    model.to(device).eval()
    config = json.loads((model_folder / "config.json").read_text())
    config["env"] = {key: value for key, value in config["env"].items() if key in HumanoidVerseIsaacConfig.model_fields}
    motion_path = Path(data_path).expanduser().resolve() if data_path is not None else DEFAULT_DATA_PATH
    ensure_motion_data(motion_path)
    config["env"]["lafan_tail_path"] = str(motion_path)
    config["env"]["expert_motion_limit"] = None
    _append_env_overrides(
        config,
        headless,
        simulator,
        disable_dr,
        disable_obs_noise,
        env_device or ("cpu" if simulator == "mujoco" else "cuda"),
    )
    config["env"]["max_episode_length_s"] = 1e9
    ensure_clean_encoder_obs_enabled_for_model(config["env"], model)
    wrapped_env, _ = HumanoidVerseIsaacConfig(**config["env"]).build(num_envs=1)
    return model, wrapped_env, config["env"].get("root_height_obs", False)


def validate_render_options(fps, render_size, ghost_alpha):
    if fps <= 0 or render_size <= 0:
        raise ValueError("fps and render_size must be positive")
    if not 0 <= ghost_alpha <= 1:
        raise ValueError("ghost_alpha must be between 0 and 1")


def select_motion(env, motion_id):
    if not 0 <= motion_id < len(env._motion_lib._motion_data_keys):
        raise ValueError(f"motion_id {motion_id} is outside the motion archive")
    env.set_is_evaluating(motion_id)


def qpos_from_env(wrapped_env):
    return wrapped_env._get_qpos_qvel(to_numpy=True)[0][0].copy()


def reset_at_origin(wrapped_env):
    root_state = wrapped_env._env.base_init_state.reshape(1, -1).clone()
    if wrapped_env._env.config.simulator.config.name == "isaacsim":
        root_state[:, 3:7] = root_state[:, [6, 3, 4, 5]]
    root_state[:, :2] = 0
    root_state[:, 7:13] = 0
    dof_state = wrapped_env._env.simulator.dof_state.clone()
    dof_state[..., 0] = wrapped_env._env.default_dof_pos
    dof_state[..., 1] = 0
    return wrapped_env.reset(to_numpy=False, target_states={"root_states": root_state, "dof_states": dof_state})


def close_playback(wrapped_env, renderer):
    if renderer is not None:
        renderer.close()
    simulator = wrapped_env._env.simulator
    for resource in (getattr(simulator, "viewer", None), getattr(simulator, "renderer", None)):
        if resource is not None:
            resource.close()
    wrapped_env.close()


class VideoWriter:
    def __init__(self, path, fps, enabled):
        self.path = Path(path)
        self.writer = None
        self.enabled = enabled
        self.fps = fps

    def __enter__(self):
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(str(self.path), fps=self.fps, codec="libx264", macro_block_size=1)
        return self

    def append(self, renderer, policy_qpos, target_qpos, alpha):
        if self.writer is not None:
            self.writer.append_data(renderer.render(policy_qpos, target_qpos, alpha))

    def __exit__(self, *args):
        if self.writer is not None:
            self.writer.close()


class MujocoGoalGhostRenderer:
    def __init__(
        self,
        render_size: int = 256,
        camera: str | None = None,
        camera_distance: float = 6.0,
        camera_azimuth: float = 135.0,
        camera_elevation: float = -18.0,
        camera_margin: float = 1.6,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(ASSET_PATH))
        self.model.vis.global_.offwidth = render_size
        self.model.vis.global_.offheight = render_size
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=render_size, width=render_size)
        self.goal_data = mujoco.MjData(self.model)
        self.perturb = mujoco.MjvPerturb()
        self.camera = camera
        self.camera_distance = camera_distance
        self.camera_azimuth = camera_azimuth
        self.camera_elevation = camera_elevation
        self.camera_margin = camera_margin

    def _render_camera(self, policy_qpos: np.ndarray, goal_qpos: np.ndarray | None):
        if self.camera is not None:
            return self.camera
        policy_root = np.asarray(policy_qpos, dtype=np.float64).ravel()[:3]
        if goal_qpos is None:
            goal_root = policy_root
        else:
            goal_root = np.asarray(goal_qpos, dtype=np.float64).ravel()[:3]
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.fixedcamid = -1
        camera.trackbodyid = -1
        camera.lookat[:] = 0.5 * (policy_root + goal_root)
        camera.lookat[2] = max(float(camera.lookat[2]), 0.9)
        root_span = float(np.linalg.norm(policy_root[:2] - goal_root[:2]))
        camera.distance = max(float(self.camera_distance), root_span * float(self.camera_margin) + 2.0)
        camera.azimuth = float(self.camera_azimuth)
        camera.elevation = float(self.camera_elevation)
        return camera

    def _set_qpos(self, data: mujoco.MjData, qpos: np.ndarray) -> None:
        qpos = np.asarray(qpos, dtype=np.float64).ravel()
        if qpos.size != self.model.nq:
            raise ValueError(f"Expected qpos with {self.model.nq} values, got {qpos.size}.")
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)

    def render(self, policy_qpos: np.ndarray, goal_qpos: np.ndarray | None, ghost_alpha: float) -> np.ndarray:
        renderer = self.renderer
        self._set_qpos(self.data, policy_qpos)
        renderer.update_scene(self.data, camera=self._render_camera(policy_qpos, goal_qpos))
        if goal_qpos is not None:
            scene = renderer.scene
            first_goal_geom = scene.ngeom
            self._set_qpos(self.goal_data, goal_qpos)
            mujoco.mjv_addGeoms(
                self.model, self.goal_data, renderer._scene_option, self.perturb, int(mujoco.mjtCatBit.mjCAT_ALL), scene
            )
            tint = GOAL_TINT / 255.0
            for geom_id in range(first_goal_geom, scene.ngeom):
                geom = scene.geoms[geom_id]
                if (
                    geom.objtype != int(mujoco.mjtObj.mjOBJ_GEOM)
                    or geom.objid < 0
                    or self.model.geom_bodyid[geom.objid] <= 0
                ):
                    geom.rgba[3] = 0.0
                    continue
                geom.rgba[:3] = 0.35 * geom.rgba[:3] + 0.65 * tint
                geom.rgba[3] = ghost_alpha
        return renderer.render()

    def close(self):
        self.renderer.close()
