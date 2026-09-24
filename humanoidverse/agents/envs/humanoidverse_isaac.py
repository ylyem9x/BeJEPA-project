import os

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
import typing as tp
from typing import Any, Dict, Tuple, Union

import gymnasium
import hydra
import numpy as np
import pydantic
import torch
from gymnasium import Env
from gymnasium.vector import VectorEnv
from omegaconf import OmegaConf
from torch.utils._pytree import tree_map

import humanoidverse
from humanoidverse.envs.legged_robot_motions.legged_robot_motions import LeggedRobotMotions
from humanoidverse.utils.helpers import pre_process_config
from humanoidverse.utils.torch_utils import calc_heading, wrap_to_pi, xyzw_to_wxyz

from ..base import BaseConfig
from ..buffers.trajectory import TrajectoryDictBuffer
from ..encoder_observation import CLEAN_ENCODER_OBS_KEY
from .utils.history_handler import HistoryHandler

if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = os.path.dirname(humanoidverse.__file__)
else:
    HUMANOIDVERSE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HYDRA_CONFIG_DIR = os.path.join(HUMANOIDVERSE_DIR, "config")
HYDRA_CONFIG_REL_PATH = os.path.join("exp", "bfm_zero", "bfm_zero")
ROOT_POSE_SCALE = 0.5
ROOT_POSE_OBS_DIM = 4
EXPERT_OBSERVATION_PIPELINE_VERSION = "isaacsim_usd_replay_v1"
KEYPOINT_WRIST_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def make_keypoint_state(
    body_pos: torch.Tensor,
    body_rot: torch.Tensor,
    body_vel: torch.Tensor,
    body_ang_vel: torch.Tensor,
    wrist_indices: list[int] | tuple[int, int],
) -> torch.Tensor:
    """Base world state plus both wrists' base-relative position and velocities.

    Layout: base xyz(3), quaternion xyzw(4), linear velocity(3), angular
    velocity(3), followed by each wrist's relative position(3), relative
    linear velocity(3), and relative angular velocity(3).
    """
    if len(wrist_indices) != 2:
        raise ValueError(f"Expected exactly two wrist indices, got {wrist_indices}.")
    base_pos = body_pos[:, 0]
    base_rot = body_rot[:, 0]
    base_vel = body_vel[:, 0]
    base_ang_vel = body_ang_vel[:, 0]
    wrists_pos = body_pos[:, wrist_indices] - base_pos.unsqueeze(1)
    wrists_vel = body_vel[:, wrist_indices] - base_vel.unsqueeze(1)
    wrists_ang_vel = body_ang_vel[:, wrist_indices] - base_ang_vel.unsqueeze(1)
    wrists = torch.cat([wrists_pos, wrists_vel, wrists_ang_vel], dim=-1).reshape(body_pos.shape[0], -1)
    return torch.cat([base_pos, base_rot, base_vel, base_ang_vel, wrists], dim=-1)


def _body_indices(body_names, requested_names: tuple[str, ...]) -> list[int]:
    names = list(body_names)
    missing = [name for name in requested_names if name not in names]
    if missing:
        raise ValueError(f"Missing keypoint bodies {missing}; available bodies: {names}.")
    return [names.index(name) for name in requested_names]


def make_root_pose_observation(
    root_pos: torch.Tensor, root_rot: torch.Tensor, anchor_pos: torch.Tensor, anchor_yaw: torch.Tensor
) -> torch.Tensor:
    """Root SE(2) pose relative to an anchor: x, y in anchor heading frame and yaw as sin/cos."""
    root_xy = root_pos[:, :2]
    anchor_xy = anchor_pos[:, :2]
    root_yaw = calc_heading(root_rot)
    yaw_rel = wrap_to_pi(root_yaw - anchor_yaw)
    delta_xy = root_xy - anchor_xy
    cos_yaw = torch.cos(anchor_yaw)
    sin_yaw = torch.sin(anchor_yaw)
    x_rel = cos_yaw * delta_xy[:, 0] + sin_yaw * delta_xy[:, 1]
    y_rel = -sin_yaw * delta_xy[:, 0] + cos_yaw * delta_xy[:, 1]
    return torch.stack(
        [x_rel / ROOT_POSE_SCALE, y_rel / ROOT_POSE_SCALE, torch.sin(yaw_rel), torch.cos(yaw_rel)], dim=-1
    )


def make_reference_root_pose_observation(ref_body_pos: torch.Tensor, ref_body_rots: torch.Tensor) -> torch.Tensor:
    anchor_pos = ref_body_pos[0:1, 0].repeat(ref_body_pos.shape[0], 1)
    anchor_yaw = calc_heading(ref_body_rots[0:1, 0]).repeat(ref_body_pos.shape[0])
    return make_root_pose_observation(ref_body_pos[:, 0], ref_body_rots[:, 0], anchor_pos, anchor_yaw)


def _rotate_vectors_by_root_pose_yaw(vectors: torch.Tensor, root_pose: torch.Tensor) -> torch.Tensor:
    root_pose = root_pose.to(device=vectors.device, dtype=vectors.dtype)
    sin_yaw = root_pose[..., 2]
    cos_yaw = root_pose[..., 3]
    x = cos_yaw * vectors[..., 0] - sin_yaw * vectors[..., 1]
    y = sin_yaw * vectors[..., 0] + cos_yaw * vectors[..., 1]
    return torch.stack([x, y, vectors[..., 2]], dim=-1)


def privileged_state_to_root_pose_frame(
    privileged_state: torch.Tensor, root_pose: torch.Tensor, *, body_count: int, root_height_obs: bool
) -> torch.Tensor:
    """Express max_local_self body positions/rotations in the reset-relative root_pose frame."""
    if body_count <= 0:
        raise ValueError("body_count must be positive.")
    base_dim = 3 * (body_count - 1) + 6 * body_count + 3 * body_count + 3 * body_count
    prefix_dim = privileged_state.shape[-1] - base_dim
    if prefix_dim < 0:
        raise ValueError(
            f"privileged_state dim {privileged_state.shape[-1]} is too small for {body_count} bodies (expected at least {base_dim})."
        )
    root_pose = root_pose.to(device=privileged_state.device, dtype=privileged_state.dtype)
    converted = privileged_state.clone()
    leading_shape = privileged_state.shape[:-1]
    pos_start = prefix_dim
    rot_start = pos_start + 3 * (body_count - 1)
    vel_start = rot_start + 6 * body_count
    ang_vel_start = vel_start + 3 * body_count
    local_pos = privileged_state[..., pos_start:rot_start].reshape(*leading_shape, body_count - 1, 3)
    pos_in_root_pose_frame = _rotate_vectors_by_root_pose_yaw(local_pos, root_pose.unsqueeze(-2))
    pos_in_root_pose_frame[..., :2] += root_pose[..., None, :2] * ROOT_POSE_SCALE
    if root_height_obs and prefix_dim > 0:
        pos_in_root_pose_frame[..., 2] += privileged_state[..., None, 0]
    converted[..., pos_start:rot_start] = pos_in_root_pose_frame.reshape(*leading_shape, -1)
    local_rot = privileged_state[..., rot_start:vel_start].reshape(*leading_shape, body_count, 6)
    tan = _rotate_vectors_by_root_pose_yaw(local_rot[..., :3], root_pose.unsqueeze(-2))
    norm = _rotate_vectors_by_root_pose_yaw(local_rot[..., 3:], root_pose.unsqueeze(-2))
    converted[..., rot_start:vel_start] = torch.cat([tan, norm], dim=-1).reshape(*leading_shape, -1)
    local_vel = privileged_state[..., vel_start:ang_vel_start].reshape(*leading_shape, body_count, 3)
    converted[..., vel_start:ang_vel_start] = _rotate_vectors_by_root_pose_yaw(
        local_vel, root_pose.unsqueeze(-2)
    ).reshape(*leading_shape, -1)
    local_ang_vel = privileged_state[..., ang_vel_start : ang_vel_start + 3 * body_count].reshape(
        *leading_shape, body_count, 3
    )
    converted[..., ang_vel_start : ang_vel_start + 3 * body_count] = _rotate_vectors_by_root_pose_yaw(
        local_ang_vel, root_pose.unsqueeze(-2)
    ).reshape(*leading_shape, -1)
    return converted


def _require_expert_simulator_replay_wrapper(env):
    """Return the wrapper and Isaac Sim environment for expert replay.

    Expert and online observations use the same simulator state and wrapper
    to keep their kinematics and observation layout consistent.
    """
    if not hasattr(env, "_env") or not hasattr(env, "_get_g1env_observation"):
        raise TypeError(
            "Expert observation loading requires HumanoidVerseVectorEnv, not the raw environment. The wrapper is required so expert and online samples use the same observation implementation."
        )
    raw_env = env._env
    simulator_name = str(raw_env.config.simulator.config.name).lower()
    if simulator_name != "isaacsim":
        raise ValueError(
            f"load_expert_trajectories_from_motion_lib requires the Isaac Sim/USD backend; got {simulator_name!r}. Refusing to fall back to MotionLib/MJCF body observations."
        )
    return (env, raw_env)


def _concat_tensor_trees(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if not chunks:
        raise ValueError("Cannot concatenate an empty list of observation chunks.")
    return tree_map(lambda *values: torch.cat(values, dim=0), chunks[0], *chunks[1:])


def _past_observation_history(values: torch.Tensor, history_length: int) -> torch.Tensor:
    """Reproduce the history returned by the online wrapper after an episode reset.

    A reset observation is not inserted into HumanoidVerse's history buffer.
    Consequently frames 0 and 1 both have empty history; frame 2 first sees
    frame 1.  Keeping this detail prevents an otherwise subtle expert/online
    boundary cue when history observations are enabled.
    """
    if history_length < 0:
        raise ValueError("history_length must be non-negative")
    history = values.new_zeros(values.shape[0], history_length, values.shape[-1])
    for lag in range(1, history_length + 1):
        destination_start = lag + 1
        if destination_start < values.shape[0]:
            history[destination_start:, lag - 1] = values[1 : values.shape[0] - lag]
    return history.reshape(values.shape[0], -1)


def _make_history_actor_observation(
    components: dict[str, torch.Tensor], history_config, *, include_actions: bool
) -> torch.Tensor:
    history_parts = []
    for key in sorted(history_config.keys()):
        if not include_actions and key in ("action", "actions"):
            continue
        if key not in components:
            raise KeyError(f"Missing processed observation component {key!r} required by history_actor")
        history_parts.append(_past_observation_history(components[key], int(history_config[key])))
    if not history_parts:
        first = next(iter(components.values()))
        return first.new_zeros(first.shape[0], 0)
    return torch.cat(history_parts, dim=-1)


def _make_isaacsim_target_states(raw_env, motion_res: dict[str, torch.Tensor], env_ids: torch.Tensor):
    """Map MotionLib root/joint generalized coordinates into Isaac Sim state tensors."""
    count = env_ids.numel()
    required = ("root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel")
    missing = [key for key in required if key not in motion_res]
    if missing:
        raise KeyError(f"MotionLib state is missing fields required for USD replay: {missing}")
    if motion_res["dof_pos"].shape[-1] != raw_env.num_dof:
        raise ValueError(
            f"MotionLib has {motion_res['dof_pos'].shape[-1]} DOFs but the USD simulator has {raw_env.num_dof}."
        )
    root_states = raw_env.target_robot_root_states.clone()
    dof_states = raw_env.target_robot_dof_state.clone()
    root_states[env_ids, :3] = motion_res["root_pos"][:count] + raw_env.env_origins[env_ids]
    root_states[env_ids, 3:7] = xyzw_to_wxyz(motion_res["root_rot"][:count])
    root_states[env_ids, 7:10] = motion_res["root_vel"][:count]
    root_states[env_ids, 10:13] = motion_res["root_ang_vel"][:count]
    dof_states[env_ids, :, 0] = motion_res["dof_pos"][:count]
    dof_states[env_ids, :, 1] = motion_res["dof_vel"][:count]
    return (root_states, dof_states)


@torch.no_grad()
def load_expert_trajectories_from_motion_lib(
    env,
    agent_cfg,
    device="cpu",
    add_history_noaction: bool = False,
    pin_memory: bool = False,
    privileged_state_world_frame: bool = False,
):
    """Load expert trajectories using USD simulator observation replay.

    MotionLib supplies root and joint states. Each frame is replayed in the
    Isaac Sim articulation and processed by the online observation wrapper.
    """
    (wrapper, raw_env) = _require_expert_simulator_replay_wrapper(env)
    expert_motion_limit = getattr(raw_env, "expert_motion_limit", None)
    if expert_motion_limit is None:
        raw_env._motion_lib.load_motions_for_training()
    else:
        raw_env._motion_lib.load_motions_for_training(max_num_seqs=min(int(expert_motion_limit), raw_env.num_envs))
    episodes = []
    file_names = []
    history_config = raw_env.config.obs.obs_auxiliary.get("history_actor")
    if bool(privileged_state_world_frame) != bool(wrapper.privileged_state_world_frame):
        raise ValueError(
            "privileged_state_world_frame must match the live wrapper because expert observations are now constructed by that wrapper."
        )
    all_env_ids = torch.arange(raw_env.num_envs, device=raw_env.device, dtype=torch.long)
    try:
        loaded_motion_count = len(raw_env._motion_lib._motion_lengths)
        loaded_motion_ids = getattr(raw_env._motion_lib, "_curr_motion_ids", None)
        for motion_index in range(loaded_motion_count):
            motion_length = int(np.ceil((raw_env._motion_lib._motion_lengths[motion_index] / raw_env.dt).cpu()))
            motion_times = torch.arange(motion_length, device=raw_env.device) * raw_env.dt
            motion_ids = torch.full((motion_length,), motion_index, device=raw_env.device, dtype=torch.long)
            motion_res = raw_env._motion_lib.get_motion_state(motion_ids, motion_times)
            source_motion_index = (
                int(loaded_motion_ids[motion_index].item()) if loaded_motion_ids is not None else motion_index
            )
            file_names.append(raw_env._motion_lib._motion_data_keys[source_motion_index])
            observation_chunks = []
            history_component_chunks = {key: [] for key in history_config.keys()} if history_config is not None else {}
            anchor_pos = None
            anchor_yaw = None
            for start in range(0, motion_length, raw_env.num_envs):
                stop = min(start + raw_env.num_envs, motion_length)
                count = stop - start
                env_ids = all_env_ids[:count]
                chunk_motion = {key: value[start:stop] for (key, value) in motion_res.items()}
                (root_states, dof_states) = _make_isaacsim_target_states(raw_env, chunk_motion, env_ids)
                raw_env.actions[env_ids] = 0.0
                raw_env.last_actions[env_ids] = 0.0
                raw_env.actions_after_delay[env_ids] = 0.0
                raw_env.history_handler.reset(env_ids)
                raw_env.episode_length_buf[env_ids] = 0
                raw_env.simulator.set_actor_root_state_tensor(env_ids, root_states)
                raw_env.simulator.set_dof_state_tensor(env_ids, dof_states)
                raw_env._refresh_sim_tensors()
                raw_env._pre_compute_observations_callback()
                raw_env._compute_observations()
                raw_env._post_compute_observations_callback()
                simulator_root = raw_env.simulator.robot_root_states[env_ids]
                local_root_pos = simulator_root[:, :3].clone()
                local_root_pos[:, :2] -= raw_env.env_origins[env_ids, :2]
                root_rot = simulator_root[:, 3:7]
                if anchor_pos is None:
                    anchor_pos = local_root_pos[0:1].clone()
                    anchor_yaw = calc_heading(root_rot[0:1]).clone()
                chunk_root_pose = make_root_pose_observation(
                    local_root_pos, root_rot, anchor_pos.expand(count, -1), anchor_yaw.expand(count)
                )
                full_root_pose = chunk_root_pose.new_zeros(raw_env.num_envs, ROOT_POSE_OBS_DIM)
                full_root_pose[:, 3] = 1.0
                full_root_pose[env_ids] = chunk_root_pose
                full_observation = wrapper._get_g1env_observation(to_numpy=False, clean_root_pose=full_root_pose)
                observation_chunks.append(tree_map(lambda value: value[env_ids].clone(), full_observation))
                if history_config is not None:
                    processed_actor_obs = raw_env.obs_buf_dict_raw["actor_obs"]
                    for key in history_config.keys():
                        if key not in processed_actor_obs:
                            raise KeyError(f"Online actor observation does not expose history component {key!r}")
                        history_component_chunks[key].append(processed_actor_obs[key][env_ids].clone())
            observation = _concat_tensor_trees(observation_chunks)
            if "time" in observation:
                observation["time"] = torch.arange(
                    motion_length, device=observation["time"].device, dtype=observation["time"].dtype
                ).unsqueeze(-1)
            if history_config is not None and (
                wrapper.include_history_actor or wrapper.include_history_noaction or add_history_noaction
            ):
                history_components = {
                    key: torch.cat(chunks, dim=0) for (key, chunks) in history_component_chunks.items()
                }
                root_pose_history = _past_observation_history(
                    observation["root_pose"], wrapper.actor_root_pose_history_length
                )
                if wrapper.include_history_actor:
                    observation["history_actor"] = torch.cat(
                        [
                            _make_history_actor_observation(history_components, history_config, include_actions=True),
                            root_pose_history,
                        ],
                        dim=-1,
                    )
                if wrapper.include_history_noaction or add_history_noaction:
                    observation["history_noaction"] = torch.cat(
                        [
                            _make_history_actor_observation(history_components, history_config, include_actions=False),
                            root_pose_history,
                        ],
                        dim=-1,
                    )
            truncated = torch.zeros(motion_length, dtype=torch.bool, device=raw_env.device)
            truncated[-1] = True
            for key, value in observation.items():
                leaves = tree_map(lambda tensor: tensor.shape[0], value)
                leaf_lengths = list(leaves.values()) if isinstance(leaves, dict) else [leaves]
                if any((length != motion_length for length in leaf_lengths)):
                    raise ValueError(
                        f"Expert observation {key!r} for {file_names[-1]!r} has inconsistent length {leaf_lengths}; expected {motion_length}."
                    )
            episodes.append(
                {
                    "observation": observation,
                    "terminated": torch.zeros(motion_length, dtype=torch.bool, device=raw_env.device),
                    "truncated": truncated,
                    "motion_id": torch.full((motion_length,), motion_index, dtype=torch.long, device=raw_env.device),
                }
            )
    finally:
        wrapper.reset(to_numpy=False)
    expert_buffer = TrajectoryDictBuffer(
        episodes=episodes, seq_length=agent_cfg.model.seq_length, device=device, pin_memory=pin_memory
    )
    assert expert_buffer.storage["observation"]["state"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    if wrapper.include_last_action:
        assert (
            expert_buffer.storage["observation"]["last_action"].shape[0] == expert_buffer.storage["truncated"].shape[0]
        )
    assert (
        expert_buffer.storage["observation"]["privileged_state"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    )
    assert expert_buffer.storage["observation"]["root_pose"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert (
        expert_buffer.storage["observation"]["keypoint_state"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    )
    assert expert_buffer.storage["terminated"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["motion_id"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    expert_buffer.file_names = file_names
    expert_buffer.observation_pipeline_version = EXPERT_OBSERVATION_PIPELINE_VERSION
    return expert_buffer


def get_enabled_dr_dynamics_obs_names(env: LeggedRobotMotions) -> list[str]:
    """Get list of dr_* raw observations that are actually being varied in the code (e.g., if ctrl_delay is randomized, then include dr_ctrl_delay observation in the list)

    Useful for picking not including dr_* observations that are not actually varied in the environment."""
    isaac_config = env.config
    dr_dynamics = []
    if isaac_config.domain_rand.randomize_ctrl_delay:
        dr_dynamics.append("dr_ctrl_delay")
    if isaac_config.domain_rand.randomize_pd_gain:
        dr_dynamics.append("dr_pd_gains")
    if isaac_config.domain_rand.randomize_base_com:
        dr_dynamics.append("dr_coms")
    if isaac_config.domain_rand.randomize_link_mass:
        dr_dynamics.append("dr_masses")
    if isaac_config.domain_rand.randomize_friction:
        dr_dynamics.append("dr_frictions")
    if isaac_config.domain_rand.randomize_torque_rfi:
        dr_dynamics.append("dr_torque_rfi")
    if isaac_config.domain_rand.randomize_rfi_lim:
        dr_dynamics.append("dr_rfi_lim")
    if isaac_config.obs.noise_scales is not None:
        for key, noise_scale in isaac_config.obs.noise_scales.items():
            if noise_scale > 0.0:
                dr_dynamics.append("dr_obs_noise_scales")
                break
    return dr_dynamics


class HumanoidVerseVectorEnv(VectorEnv):
    """Wrapper to make the humanoidverse env compatible with gymnasium's VectorEnv interface."""

    def __init__(
        self,
        env: LeggedRobotMotions,
        add_time_aware_observation: bool = True,
        include_last_action: bool = True,
        context_length: int | None = None,
        include_dr_info: bool = False,
        included_dr_obs_names: list[str] | None = None,
        include_history_actor: bool = True,
        include_history_noaction: bool = False,
        include_clean_encoder_obs: bool = False,
        privileged_state_world_frame: bool = False,
        actor_root_pose_noise: float = 0.0,
        actor_root_pose_history_length: int = 0,
    ):
        super().__init__()
        if actor_root_pose_noise < 0.0:
            raise ValueError("actor_root_pose_noise must be non-negative.")
        if actor_root_pose_history_length < 0:
            raise ValueError("actor_root_pose_history_length must be non-negative.")
        if actor_root_pose_history_length > 0 and (not include_history_actor):
            raise ValueError("actor_root_pose_history_length requires include_history_actor=True.")
        self._env = env
        self.spec = self._env.spec
        self.num_envs = self._env.num_envs
        self.add_time_aware_observation = add_time_aware_observation
        self.include_last_action = include_last_action
        self.context_length = context_length
        self.history_handler = None
        self.include_dr_info = include_dr_info
        self.include_history_actor = include_history_actor
        self.include_history_noaction = include_history_noaction
        self.include_clean_encoder_obs = include_clean_encoder_obs
        self.privileged_state_world_frame = privileged_state_world_frame
        self.actor_root_pose_noise = actor_root_pose_noise
        self.actor_root_pose_history_length = actor_root_pose_history_length
        self._env._humanoidverse_actor_root_pose_history_length = actor_root_pose_history_length
        self._root_pose_anchor_pos = None
        self._root_pose_anchor_yaw = None
        self._actor_root_pose_history = torch.zeros(
            self.num_envs, self.actor_root_pose_history_length, ROOT_POSE_OBS_DIM, device=self.device
        )
        self._keypoint_wrist_indices = None
        if included_dr_obs_names is not None:
            self.included_dr_obs_names = included_dr_obs_names
        else:
            self.included_dr_obs_names = get_enabled_dr_dynamics_obs_names(self._env)
        if self.include_dr_info and len(self.included_dr_obs_names) == 0:
            raise ValueError(
                "include_dr_info is True, but no dr_* observations are enabled in the environment. Check the domain randomization settings in the environment config."
            )
        (example_observation, _) = self.reset()
        observation_spaces = {}
        for key, value in example_observation.items():
            if key not in ("history", CLEAN_ENCODER_OBS_KEY):
                observation_spaces[key] = gymnasium.spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=value.shape, dtype=np.float32
                )
        if self.include_history_noaction:
            history_length = 0
            for k, v in self._env.config.obs.obs_auxiliary["history_actor"].items():
                if k not in ["action", "actions"]:
                    history_length += v * self._env.config.obs.obs_dims[k]
            history_length += self.actor_root_pose_history_length * ROOT_POSE_OBS_DIM
            observation_spaces["history_noaction"] = gymnasium.spaces.Box(
                low=-float("inf"), high=float("inf"), shape=(self.num_envs, history_length), dtype=np.float32
            )
        if add_time_aware_observation:
            observation_spaces["time"] = gymnasium.spaces.Box(low=0.0, high=float("inf"), shape=(1,), dtype=np.int32)
        self.observation_space = gymnasium.spaces.Dict(observation_spaces)
        self.single_action_space = self._env.get_wrapper_attr("single_action_space")
        action_space_shape = (self.num_envs,) + self.single_action_space.shape
        self.action_space = gymnasium.spaces.Box(
            low=np.tile(self.single_action_space.low, (self.num_envs, 1)),
            high=np.tile(self.single_action_space.high, (self.num_envs, 1)),
            shape=action_space_shape,
            dtype=self.single_action_space.dtype,
        )
        if self.context_length:
            dims = {"action": self.single_action_space.shape[0]}
            for k, v in self.single_observation_space.spaces.items():
                dims[k] = 1 if len(v.shape) == 0 else v.shape[0]
            self.history_handler = HistoryHandler(
                self.num_envs, context_length=self.context_length, keys_dims=dims, device=self.device
            )
        _target_dof_pos = self._env.default_dof_pos.clone().unsqueeze(0).repeat(self.num_envs, 1, 2)
        _target_dof_pos[..., 1] = 0.0
        self._default_pose_target_reset = {"dof_states": _target_dof_pos, "root_states": self._env.base_init_state}

    def _root_positions_for_pose_obs(self) -> torch.Tensor:
        root_pos = self._env.simulator.robot_root_states[:, :3].clone().detach()
        env_origins = getattr(self._env, "env_origins", None)
        if env_origins is not None:
            root_pos[:, :2] = root_pos[:, :2] - env_origins[:, :2]
        return root_pos

    def _root_rotations_for_pose_obs(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 3:7].clone().detach()

    def _set_root_pose_anchor(self, env_ids: torch.Tensor | None = None) -> None:
        root_pos = self._root_positions_for_pose_obs()
        root_yaw = calc_heading(self._root_rotations_for_pose_obs())
        if self._root_pose_anchor_pos is None:
            self._root_pose_anchor_pos = root_pos.clone()
            self._root_pose_anchor_yaw = root_yaw.clone()
        elif env_ids is None:
            self._root_pose_anchor_pos[:] = root_pos
            self._root_pose_anchor_yaw[:] = root_yaw
        else:
            self._root_pose_anchor_pos[env_ids] = root_pos[env_ids]
            self._root_pose_anchor_yaw[env_ids] = root_yaw[env_ids]

    def _get_root_pose_observation(self) -> torch.Tensor:
        if self._root_pose_anchor_pos is None or self._root_pose_anchor_yaw is None:
            self._set_root_pose_anchor()
        return make_root_pose_observation(
            self._root_positions_for_pose_obs(),
            self._root_rotations_for_pose_obs(),
            self._root_pose_anchor_pos,
            self._root_pose_anchor_yaw,
        )

    def _actor_root_pose_noise_scale(self) -> float:
        noise_scale = self.actor_root_pose_noise
        if bool(getattr(self._env, "add_noise_currculum", False)):
            noise_scale *= float(self._env.current_noise_curriculum_value)
        return noise_scale

    def _get_actor_root_pose_observation(self, clean_root_pose: torch.Tensor) -> torch.Tensor:
        noise_scale = self._actor_root_pose_noise_scale()
        if noise_scale == 0.0:
            return clean_root_pose.clone()
        noise = (torch.rand_like(clean_root_pose) * 2.0 - 1.0) * noise_scale
        return clean_root_pose + noise

    def _reset_actor_root_pose_history(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._actor_root_pose_history.zero_()
        elif len(env_ids) > 0:
            self._actor_root_pose_history[env_ids] = 0.0

    def _get_actor_root_pose_history(self, like: torch.Tensor) -> torch.Tensor:
        history = self._actor_root_pose_history.reshape(self.num_envs, -1)
        return history.to(device=like.device, dtype=like.dtype)

    def _update_actor_root_pose_history(self, actor_root_pose: torch.Tensor) -> None:
        if self.actor_root_pose_history_length == 0:
            return
        if self.actor_root_pose_history_length > 1:
            self._actor_root_pose_history[:, 1:] = self._actor_root_pose_history[:, :-1].clone()
        self._actor_root_pose_history[:, 0] = actor_root_pose.to(
            device=self._actor_root_pose_history.device, dtype=self._actor_root_pose_history.dtype
        )

    def _append_actor_root_pose_history(self, history_actor: torch.Tensor) -> torch.Tensor:
        if self.actor_root_pose_history_length == 0:
            return history_actor
        return torch.cat([history_actor, self._get_actor_root_pose_history(history_actor)], dim=-1)

    def _privileged_state_body_count(self) -> int:
        body_names = getattr(self._env.simulator, "_body_list", None)
        if body_names is not None:
            return len(body_names)
        body_names = getattr(self._env.simulator, "body_names", None)
        if body_names is not None:
            return len(body_names)
        return int(self._env.config.robot.num_bodies + self._env.config.robot.motion.get("nums_extend_bodies", 0))

    def _get_keypoint_state(self) -> torch.Tensor:
        simulator = self._env.simulator
        if self._keypoint_wrist_indices is None:
            body_names = getattr(simulator, "_body_list", getattr(simulator, "body_names", None))
            if body_names is None:
                raise ValueError("Simulator does not expose body names required by the keypoint encoder.")
            self._keypoint_wrist_indices = _body_indices(body_names, KEYPOINT_WRIST_BODY_NAMES)
        return (
            make_keypoint_state(
                simulator._rigid_body_pos,
                simulator._rigid_body_rot,
                simulator._rigid_body_vel,
                simulator._rigid_body_ang_vel,
                self._keypoint_wrist_indices,
            )
            .clone()
            .detach()
        )

    def _maybe_convert_privileged_state(self, privileged_state: torch.Tensor, root_pose: torch.Tensor) -> torch.Tensor:
        if not self.privileged_state_world_frame:
            return privileged_state
        return privileged_state_to_root_pose_frame(
            privileged_state,
            root_pose,
            body_count=self._privileged_state_body_count(),
            root_height_obs=bool(self._env.config.obs.get("root_height_obs", True)),
        )

    @property
    def single_observation_space(self):
        """Return the observation space for a single environment."""
        single_obs_spaces = {}
        for key, space in self.observation_space.spaces.items():
            single_obs_spaces[key] = gymnasium.spaces.Box(
                low=space.low[0], high=space.high[0], shape=space.shape[1:], dtype=space.dtype
            )
        return gymnasium.spaces.Dict(single_obs_spaces)

    @property
    def device(self):
        return self.base_env.device

    @property
    def base_env(self) -> Env:
        return self._env.unwrapped

    @property
    def unwrapped(self):
        return self.base_env

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        to_numpy: bool = True,
        reset_to_default_pose: bool = False,
        target_states: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if reset_to_default_pose and target_states is not None:
            raise ValueError("Cannot specify both reset_to_default_pose and target_states.")
        target_states = target_states or (self._default_pose_target_reset if reset_to_default_pose else None)
        (_, info) = self.base_env.reset_all(target_states=target_states)
        self._set_root_pose_anchor()
        self._reset_actor_root_pose_history()
        observation = self._get_g1env_observation(to_numpy=to_numpy)
        (qpos, qvel) = self._get_qpos_qvel(to_numpy=to_numpy)
        if self.history_handler is not None:
            self._update_history(torch.arange(self.num_envs, device=self.device), observation, None)
            observation["history"] = {"observation": [], "action": []}
        self.env_to_reset = []
        info = {"qpos": qpos, "qvel": qvel}
        return (observation, info)

    def _get_qpos_qvel(self, to_numpy: bool = True):
        base_pos = self._env.simulator.robot_root_states[:, [0, 1, 2, 6, 3, 4, 5]].clone().detach()
        qpos = self._env.simulator.dof_pos.clone().detach()
        mujoco_qpos = torch.cat([base_pos, qpos], dim=-1)
        qvel = self._env.simulator.dof_vel.clone().detach()
        base_lin_vel = self._env.simulator.robot_root_states[:, 7:10].clone().detach()
        base_ang_vel = self._env._get_obs_base_ang_vel().clone().detach()
        mujoco_qvel = torch.cat([base_lin_vel, base_ang_vel, qvel], dim=-1)
        if to_numpy:
            mujoco_qpos = mujoco_qpos.cpu().numpy()
            mujoco_qvel = mujoco_qvel.cpu().numpy()
        return (mujoco_qpos, mujoco_qvel)

    def _get_g1env_observation(
        self,
        to_numpy: bool = True,
        update_actor_root_pose_history: bool = False,
        clean_root_pose: torch.Tensor | None = None,
    ):
        """Turn current Isaac sim state into a G1Env-like observation."""
        processed_obs = self._env.obs_buf_dict_raw["actor_obs"]
        g1env_state = torch.cat(
            [
                processed_obs["dof_pos"],
                processed_obs["dof_vel"],
                processed_obs["projected_gravity"],
                processed_obs["base_ang_vel"],
            ],
            dim=-1,
        )
        last_action = processed_obs["actions"]
        clean_root_pose = self._get_root_pose_observation() if clean_root_pose is None else clean_root_pose
        actor_root_pose = self._get_actor_root_pose_observation(clean_root_pose)
        privileged_state = self._maybe_convert_privileged_state(processed_obs["max_local_self"], clean_root_pose)
        observation = {
            "state": g1env_state,
            "privileged_state": privileged_state,
            "root_pose": actor_root_pose,
            "keypoint_state": self._get_keypoint_state(),
        }
        if self.include_clean_encoder_obs:
            observation[CLEAN_ENCODER_OBS_KEY] = self._get_clean_encoder_observation(root_pose=clean_root_pose)
        if self.include_last_action:
            observation["last_action"] = last_action
        if self.add_time_aware_observation:
            observation["time"] = self._env.episode_length_buf.unsqueeze(-1)
        if self.include_history_actor:
            observation["history_actor"] = self._append_actor_root_pose_history(processed_obs["history_actor"])
        if self.include_history_noaction:
            act_hlen = self._env.config.obs.obs_auxiliary["history_actor"]["actions"]
            history_noaction = processed_obs["history_actor"][:, last_action.shape[-1] * act_hlen :]
            observation["history_noaction"] = self._append_actor_root_pose_history(history_noaction)
        if self.include_dr_info:
            dr_obs = torch.cat([processed_obs[key] for key in self.included_dr_obs_names], dim=-1)
            observation["dr_dynamics"] = dr_obs
        if update_actor_root_pose_history:
            self._update_actor_root_pose_history(actor_root_pose)
        if to_numpy:
            observation = tree_map(lambda x: x.cpu().numpy(), observation)
        return observation

    def _get_scaled_obs_component(self, obs_key: str) -> torch.Tensor:
        getter = getattr(self._env, f"_get_obs_{obs_key}")
        obs_scale = self._env.config.obs.obs_scales[obs_key]
        return getter().clone().detach() * obs_scale

    def _get_clean_encoder_observation(self, root_pose: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        root_pose = self._get_root_pose_observation() if root_pose is None else root_pose
        privileged_state = self._maybe_convert_privileged_state(
            self._get_scaled_obs_component("max_local_self"), root_pose
        )
        clean_state = torch.cat(
            [
                self._get_scaled_obs_component("dof_pos"),
                self._get_scaled_obs_component("dof_vel"),
                self._get_scaled_obs_component("projected_gravity"),
                self._get_scaled_obs_component("base_ang_vel"),
            ],
            dim=-1,
        )
        return {"state": clean_state, "privileged_state": privileged_state, "root_pose": root_pose}

    def get_episodic_dr_info(self) -> Dict[str, np.ndarray]:
        """Get the episodic domain randomization information."""
        if not self.include_dr_info:
            raise ValueError("include_dr_info is False, cannot get episodic DR info.")
        dr_info = {}
        for key in self.included_dr_obs_names:
            dr_info[key] = self._env.obs_buf_dict_raw["actor_obs"][key].cpu().numpy()

    def step(
        self, actions: Union[torch.Tensor, Dict], to_numpy: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        self._update_history(None, None, actions)
        if to_numpy:
            actions = (
                torch.tensor(actions, device=self._env.device, dtype=torch.float32)
                if isinstance(actions, np.ndarray)
                else actions
            )
        actions = {"actions": actions}
        (_, reward, reset, new_info) = self.base_env.step(actions)
        (qpos, qvel) = self._get_qpos_qvel(to_numpy=to_numpy)
        reset = reset.bool()
        env_to_reset = reset.nonzero(as_tuple=False).flatten()
        if len(env_to_reset) > 0:
            self._set_root_pose_anchor(env_to_reset)
            self._reset_actor_root_pose_history(env_to_reset)
        time_outs = new_info["time_outs"].bool()
        terminated = torch.logical_and(reset, ~time_outs)
        truncated = time_outs
        observation = self._get_g1env_observation(to_numpy=to_numpy, update_actor_root_pose_history=True)
        observation = self._add_history_to_observation(observation, to_numpy=to_numpy)
        self._update_history(env_to_reset, observation, None)
        if len(env_to_reset) > 0 and self.history_handler is not None:
            assert len(env_to_reset) == self.num_envs, "Partial resets not supported with history"
            observation["history"] = {"observation": [], "action": []}
        if to_numpy:
            reward = reward.cpu().numpy()
            terminated = terminated.cpu().numpy()
            truncated = truncated.cpu().numpy()
        new_info["qpos"] = qpos
        new_info["qvel"] = qvel
        return (observation, reward, terminated, truncated, new_info)

    def close(self):
        return self.base_env.close()

    def _add_history_to_observation(
        self, observation: dict[str, torch.Tensor], to_numpy: bool = True
    ) -> dict[str, torch.Tensor]:
        if self.context_length and self.history_handler is not None:
            _history = {"observation": {}, "action": self.history_handler.query("action", filter_by_length=True)}
            for k in observation.keys():
                _history["observation"][k] = self.history_handler.query(k, filter_by_length=True)
            if to_numpy:
                _history = tree_map(lambda x: x.cpu().numpy(), _history)
            observation["history"] = _history
        return observation

    def _update_history(self, reset_ids, observation, action):
        if self.history_handler is not None:
            if reset_ids is not None and len(reset_ids) > 0:
                self.history_handler.reset(reset_ids=reset_ids)
            if observation is not None:
                for key, value in observation.items():
                    if key != "history":
                        self.history_handler.add(key, value)
            if action is not None:
                self.history_handler.add("action", action)

    def render(self):
        return self.base_env.simulator.render()


_ISAAC_SIM_INITIALIZED = False


def instantiate_isaac_sim(num_envs: int, enable_cameras: bool = False, headless: bool = True):
    global _ISAAC_SIM_INITIALIZED
    if _ISAAC_SIM_INITIALIZED:
        return
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="")
    AppLauncher.add_app_launcher_args(parser)
    (args_cli, _) = parser.parse_known_args()
    args_cli.num_envs = num_envs
    args_cli.enable_cameras = enable_cameras
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    _ = app_launcher.app
    _ISAAC_SIM_INITIALIZED = True


_humanoidverse_env_singleton = None


class HumanoidVerseIsaacConfig(BaseConfig):
    name: tp.Literal["humanoidverse_isaac"] = "humanoidverse_isaac"
    device: str = "cuda:0"
    lafan_tail_path: str
    expert_motion_limit: int | None = None
    enable_cameras: bool = False
    camera_render_save_dir: str = "isaac_videos"
    max_episode_length_s: float | None = None
    disable_obs_noise: bool = False
    disable_domain_randomization: bool = False
    relative_config_path: str = HYDRA_CONFIG_REL_PATH
    include_last_action: bool = True
    hydra_overrides: tp.List[str] = pydantic.Field(default_factory=list)
    aux_reward_names: tp.List[str] = pydantic.Field(default_factory=list)
    context_length: int | None = None
    include_dr_info: bool = False
    included_dr_obs_names: tp.List[str] | None = None
    include_history_actor: bool = False
    include_history_noaction: bool = False
    include_clean_encoder_obs: bool = False
    privileged_state_world_frame: bool = False
    actor_root_pose_noise: float = pydantic.Field(default=0.0, ge=0.0)
    actor_root_pose_history_length: int = pydantic.Field(default=0, ge=0)
    make_config_g1env_compatible: bool = False
    root_height_obs: bool = False

    def model_post_init(self, context) -> None:
        pass

    def build(self, num_envs: int = 1) -> tp.Tuple[HumanoidVerseVectorEnv, tp.Any]:
        global _humanoidverse_env_singleton
        assert num_envs >= 1
        if _humanoidverse_env_singleton is not None:
            if num_envs != _humanoidverse_env_singleton.num_envs:
                raise ValueError(
                    f"HumanoidVerse env was already created with num_envs={_humanoidverse_env_singleton.num_envs}, but requested num_envs={num_envs}. You can only spin up one HumanoidVerse per Python instance"
                )
            elif self != _humanoidverse_env_singleton._creation_config:
                raise ValueError(
                    f"HumanoidVerse env was already created with different config: {_humanoidverse_env_singleton._creation_config} vs {self}. You can only spin up one HumanoidVerse per Python instance"
                )
            return (_humanoidverse_env_singleton, {})
        hydra_overrides = self.hydra_overrides.copy()
        with hydra.initialize_config_dir(config_dir=HYDRA_CONFIG_DIR):
            cfg = hydra.compose(config_name=self.relative_config_path, overrides=hydra_overrides or [])
        unresolved_conf = OmegaConf.to_container(cfg, resolve=False)
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", lambda x: eval(x))
        cfg.num_envs = num_envs
        cfg.exp_base = "__no_exp_base__"
        if not any(["env.config.headless" in k for k in hydra_overrides]):
            cfg.env.config.headless = True
        cfg.env.config.save_rendering_dir = self.camera_render_save_dir
        cfg.robot.asset.asset_root = cfg.robot.asset.asset_root.replace("humanoidverse", HUMANOIDVERSE_DIR)
        cfg.robot.motion.asset.assetRoot = cfg.robot.motion.asset.assetRoot.replace("humanoidverse", HUMANOIDVERSE_DIR)
        cfg.robot.motion.motion_file = self.lafan_tail_path
        pre_process_config(cfg)
        OmegaConf.set_struct(cfg, False)
        if self.make_config_g1env_compatible:
            cfg.robot.motion.num_extend_bodies = 0
            cfg.robot.motion.extend_config = []
            cfg.robot.motion.motion_tracking_link = []
            cfg.robot.motion.upper_body_link = [
                "left_shoulder_pitch_link",
                "left_shoulder_roll_link",
                "left_shoulder_yaw_link",
                "left_elbow_link",
                "right_shoulder_pitch_link",
                "right_shoulder_roll_link",
                "right_shoulder_yaw_link",
                "right_elbow_link",
            ]
            cfg.robot.motion.joint_matches = [
                ["pelvis", "Pelvis"],
                ["left_hip_pitch_link", "L_Hip"],
                ["left_knee_link", "L_Knee"],
                ["left_ankle_roll_link", "L_Ankle"],
                ["right_hip_pitch_link", "R_Hip"],
                ["right_knee_link", "R_Knee"],
                ["right_ankle_roll_link", "R_Ankle"],
                ["left_shoulder_roll_link", "L_Shoulder"],
                ["left_elbow_link", "L_Elbow"],
                ["right_shoulder_roll_link", "R_Shoulder"],
                ["right_elbow_link", "R_Elbow"],
            ]
        if self.enable_cameras:
            cfg.simulator.enable_cameras = True
        if self.disable_obs_noise:
            for key in cfg.obs.noise_scales.keys():
                cfg.obs.noise_scales[key] = 0.0
        cfg.obs.root_height_obs = self.root_height_obs
        cfg.env.config.aux_reward_names = list(self.aux_reward_names)
        if self.disable_domain_randomization:
            cfg.domain_rand.randomize_ctrl_delay = False
            cfg.domain_rand.randomize_pd_gain = False
            cfg.domain_rand.randomize_base_com = False
            cfg.domain_rand.randomize_link_mass = False
            cfg.domain_rand.randomize_friction = False
            cfg.domain_rand.randomize_torque_rfi = False
            cfg.domain_rand.randomize_rfi_lim = False
            cfg.domain_rand.push_robots = False
            cfg.domain_rand.randomize_default_dof_pos = False
        assert cfg.env.config.termination.terminate_when_close_to_dof_pos_limit is False
        assert cfg.env.config.termination.terminate_when_close_to_dof_vel_limit is False
        assert cfg.env.config.termination.terminate_when_close_to_torque_limit is False
        assert cfg.env.config.termination.terminate_by_contact is False
        assert cfg.env.config.termination.terminate_by_gravity is False
        assert cfg.env.config.termination.terminate_by_low_height is False
        assert cfg.env.config.termination.terminate_when_motion_end is False
        assert cfg.env.config.termination.terminate_when_motion_far is False
        assert cfg.env.config.robot.control.normalize_action_to == cfg.env.config.robot.control.action_clip_value, (
            "normalize_action_to and action_clip_value must be the same"
        )
        if self.max_episode_length_s is not None:
            cfg.env.config.max_episode_length_s = self.max_episode_length_s
        simulator_type = cfg.simulator["_target_"].split(".")[-1]
        if simulator_type == "IsaacSim":
            instantiate_isaac_sim(num_envs, enable_cameras=self.enable_cameras, headless=cfg.env.config.headless)
        isaac_env = LeggedRobotMotions(cfg.env.config, device=self.device)
        isaac_env.expert_motion_limit = self.expert_motion_limit
        isaac_env.privileged_state_world_frame = self.privileged_state_world_frame
        env = HumanoidVerseVectorEnv(
            isaac_env,
            include_last_action=self.include_last_action,
            include_dr_info=self.include_dr_info,
            included_dr_obs_names=self.included_dr_obs_names,
            context_length=self.context_length,
            include_history_actor=self.include_history_actor,
            include_history_noaction=self.include_history_noaction,
            include_clean_encoder_obs=self.include_clean_encoder_obs,
            privileged_state_world_frame=self.privileged_state_world_frame,
            actor_root_pose_noise=0.0 if self.disable_obs_noise else self.actor_root_pose_noise,
            actor_root_pose_history_length=self.actor_root_pose_history_length,
        )
        env._creation_config = self
        _humanoidverse_env_singleton = env
        return (env, {"unresolved_conf": unresolved_conf})
