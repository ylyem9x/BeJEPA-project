import collections
import contextlib
import copy
import dataclasses
import numbers
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import ot
import torch
from torch.utils._pytree import tree_map
from tqdm import tqdm

from ..envs.humanoidverse_isaac import (
    ROOT_POSE_SCALE,
    HumanoidVerseIsaacConfig,
    HumanoidVerseVectorEnv,
    make_reference_root_pose_observation,
    privileged_state_to_root_pose_frame,
)
from .base import BaseEvalConfig, extract_model

ROOT_SE2_XY_SCALE_M = 0.5
ROOT_SE2_YAW_SCALE_RAD = float(np.deg2rad(30.0))


def _reload_reference_motion_dataset(env: HumanoidVerseVectorEnv, motion_file: Path, *, evaluation: bool) -> None:
    """Reload reference motions in the existing Isaac environment.

    Training and evaluation share one simulator instance per process.
    """
    raw_env = env._env
    motion_lib = raw_env._motion_lib
    if motion_lib.smpl_data is not None:
        raise RuntimeError("temporary reference-dataset switching does not support smpl_motion_file")
    phase = "evaluation" if evaluation else "training"
    started = time.monotonic()
    print(f"[reference-data] loading {phase} motions from {motion_file}", flush=True)
    fix_height = motion_lib.fix_height
    multi_thread = motion_lib.multi_thread
    motion_lib.m_cfg.motion_file = str(motion_file)
    motion_lib.load_data(str(motion_file))
    motion_lib.setup_constants(fix_height=fix_height, multi_thread=multi_thread)
    motion_lib.all_motions_loaded = False
    if evaluation:
        motion_lib.load_all_motions()
    else:
        motion_lib.load_motions_for_training(max_num_seqs=raw_env.num_envs)
    raw_env.motion_dt = motion_lib._motion_dt
    raw_env.num_motions = motion_lib._num_unique_motions
    motion_num_frames = getattr(motion_lib, "_motion_num_frames", None)
    frame_summary = f" / {int(motion_num_frames.sum().item())} frames" if motion_num_frames is not None else ""
    print(
        f"[reference-data] loaded {motion_lib._num_unique_motions} {phase} motions{frame_summary} in {time.monotonic() - started:.1f}s",
        flush=True,
    )


@contextlib.contextmanager
def temporary_reference_motion_dataset(env: HumanoidVerseVectorEnv, motion_file: str | None):
    """Use the evaluation motion dataset temporarily, then restore training data."""
    if motion_file is None:
        yield env
        return
    motion_lib = env._env._motion_lib
    original_path = Path(str(motion_lib.m_cfg.motion_file)).expanduser().resolve()
    evaluation_path = Path(motion_file).expanduser().resolve()
    if evaluation_path == original_path:
        yield env
        return
    _reload_reference_motion_dataset(env, evaluation_path, evaluation=True)
    try:
        yield env
    finally:
        _reload_reference_motion_dataset(env, original_path, evaluation=False)
        env._env.reset_all()


def get_next(field: str, data: Any):
    if "next" in data and field in data["next"]:
        return data["next"][field]
    elif f"next_{field}" in data:
        return data[f"next_{field}"]
    else:
        raise ValueError(f"No next of {field} found in data.")


@dataclasses.dataclass
class Episode:
    storage: Dict | None = None

    def initialise(self, observation: np.ndarray | Dict[str, np.ndarray], info: Dict) -> None:
        if self.storage is None:
            self.storage = defaultdict(list)
        if isinstance(observation, Mapping):
            self.storage["observation"] = {k: [copy.deepcopy(v)] for (k, v) in observation.items()}
        else:
            self.storage["observation"].append(copy.deepcopy(observation))
        self.storage["info"] = {
            k: [copy.deepcopy(v)]
            for (k, v) in info.items()
            if not k.startswith("_") and k not in ["final_observation", "final_info"]
        }

    def add(
        self,
        observation: np.ndarray | Dict[str, np.ndarray],
        reward: np.ndarray,
        action: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        info: Dict,
    ) -> None:
        if isinstance(observation, Mapping):
            for k, v in observation.items():
                self.storage["observation"][k].append(copy.deepcopy(v))
        else:
            self.storage["observation"].append(copy.deepcopy(observation))
        for k, v in info.items():
            if not k.startswith("_") and k not in ["final_observation", "final_info"]:
                if k in self.storage["info"]:
                    self.storage["info"][k].append(copy.deepcopy(v))
                else:
                    self.storage["info"][k] = [copy.deepcopy(v)]
        self.storage["reward"].append(copy.deepcopy(reward))
        self.storage["action"].append(copy.deepcopy(action))
        self.storage["terminated"].append(copy.deepcopy(terminated))
        self.storage["truncated"].append(copy.deepcopy(truncated))

    def get(self) -> Dict[str, np.ndarray]:
        output = {}
        for k, v in self.storage.items():
            if k in ["observation", "info"]:
                if isinstance(v, Mapping):
                    output[k] = {}
                    for k2, v2 in v.items():
                        output[k][k2] = np.array(v2)
                else:
                    output[k] = np.array(v)
            else:
                output[k] = np.array(v)
        return output


GROOT_MPJPE_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
UPPER_BODY_MPJPE_BODY_NAMES = [
    "pelvis",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
WRIST_TRACKING_BODY_NAMES = ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link")


def get_backward_observation(
    env,
    motion_id,
    include_last_action,
    velocity_multiplier: float = 1.0,
    privileged_state_world_frame: bool | None = None,
) -> torch.Tensor:
    import numpy as np

    from humanoidverse.envs.legged_robot_motions.legged_robot_motions import (
        compute_humanoid_observations_max,
        compute_humanoid_observations_max_with_contact,
    )
    from humanoidverse.utils.torch_utils import quat_rotate_inverse

    motion_times = (
        torch.arange(int(np.ceil((env._motion_lib._motion_lengths[motion_id] / env.dt).cpu()))).to(env.device) * env.dt
    )
    motion_state = env._motion_lib.get_motion_state(motion_id, motion_times)
    ref_body_pos = motion_state["rg_pos_t"]
    ref_body_rots = motion_state["rg_rot_t"]
    ref_body_vels = motion_state["body_vel_t"] * velocity_multiplier
    ref_body_angular_vels = motion_state["body_ang_vel_t"] * velocity_multiplier
    ref_dof_pos = motion_state["dof_pos"] - env.default_dof_pos[0]
    ref_dof_vel = motion_state["dof_vel"] * velocity_multiplier

    def process_body_data(data_dict, device="cuda"):
        def extract_and_concat_vec3(arr: torch.Tensor) -> torch.Tensor:
            arr = arr.float()
            return torch.cat(
                [
                    arr[:, 0:13],
                    torch.zeros_like(arr[:, 14:16]),
                    arr[:, 13:18],
                    arr[:, 21:25],
                    arr[:, 20:21],
                    arr[:, 27:28],
                    arr[:, 12:13],
                ],
                dim=1,
            )

        def extract_and_concat_quat_torch(arr: torch.Tensor) -> torch.Tensor:
            arr = arr.float()
            seg1 = arr[:, 0:13]
            seg2 = (
                torch.tensor([1, 0, 0, 0], dtype=arr.dtype, device=arr.device).view(1, 1, 4).repeat(arr.shape[0], 2, 1)
            )
            seg3 = arr[:, 13:18]
            seg4 = arr[:, 21:25]
            seg7 = arr[:, 12:13]
            seg5 = arr[:, 20:21]
            seg6 = arr[:, 27:28]
            quat_wxyz = torch.cat([seg1, seg2, seg3, seg4, seg5, seg6, seg7], dim=1)
            quat_xyzw = torch.cat([quat_wxyz[..., 1:], quat_wxyz[..., :1]], dim=-1)
            return quat_xyzw

        body_pos = torch.from_numpy(data_dict["body_pos_w"]).to(device)
        body_quat = torch.from_numpy(data_dict["body_quat_w"]).to(device)
        body_lin_vel = torch.from_numpy(data_dict["body_lin_vel_w"]).to(device)
        body_ang_vel = torch.from_numpy(data_dict["body_ang_vel_w"]).to(device)
        joint_pos = torch.from_numpy(data_dict["joint_pos"]).to(device)[:, 1:]
        joint_vel = torch.from_numpy(data_dict["joint_vel"]).to(device)[:, 1:]
        ref_body_pos = extract_and_concat_vec3(body_pos)
        ref_body_vels = extract_and_concat_vec3(body_lin_vel)
        ref_body_angular_vels = extract_and_concat_vec3(body_ang_vel)
        ref_body_rots = extract_and_concat_quat_torch(body_quat)
        ref_dof_pos = joint_pos.float() - env.default_dof_pos[0]
        ref_dof_vel = joint_vel.float()
        return (ref_body_pos, ref_body_rots, ref_body_vels, ref_body_angular_vels, ref_dof_pos, ref_dof_vel)

    if env.use_contact_in_obs_max:
        contact_binary = env.foot_contact_detect(ref_body_pos, ref_body_vels)
        obs_dict = compute_humanoid_observations_max_with_contact(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=env.config.obs.root_height_obs,
            contact_binary=contact_binary,
        )
    else:
        obs_dict = compute_humanoid_observations_max(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=env.config.obs.root_height_obs,
        )
    max_local_self_obs = torch.cat([v for v in obs_dict.values()], dim=-1)
    root_pose = make_reference_root_pose_observation(ref_body_pos, ref_body_rots)
    if privileged_state_world_frame is None:
        privileged_state_world_frame = bool(getattr(env, "privileged_state_world_frame", False))
    if privileged_state_world_frame:
        max_local_self_obs = privileged_state_to_root_pose_frame(
            max_local_self_obs,
            root_pose,
            body_count=ref_body_pos.shape[1],
            root_height_obs=env.config.obs.root_height_obs,
        )
    if env.config.obs.use_obs_filter:
        base_quat = ref_body_rots[:, 0]
        ref_ang_vel = ref_body_angular_vels[:, 0]
        projected_gravity = quat_rotate_inverse(
            base_quat, env.gravity_vec[0:1].repeat(max_local_self_obs.shape[0], 1), w_last=True
        )
        bogus_actions = ref_dof_pos
        bogus_history_actor = torch.cat(
            [bogus_actions, ref_ang_vel, ref_dof_pos, ref_dof_vel, projected_gravity], dim=-1
        ).repeat(1, 4)
        ref_dict = {
            "actions": bogus_actions,
            "ref_ang_vel": ref_ang_vel,
            "ref_dof_pos": ref_dof_pos,
            "dof_pos": motion_state["dof_pos"],
            "ref_dof_vel": ref_dof_vel,
            "dof_vel": motion_state["dof_vel"],
            "fake_history": bogus_history_actor,
            "max_local_self_obs": max_local_self_obs,
            "projected_gravity": projected_gravity,
            "ref_body_pos": ref_body_pos,
            "ref_body_rots": ref_body_rots,
            "ref_body_vels": ref_body_vels,
            "ref_body_angular_vels": ref_body_angular_vels,
            "root_pos": motion_state["root_pos"],
            "root_rot": motion_state["root_rot"],
            "root_vel": motion_state["root_vel"],
            "root_ang_vel": motion_state["root_ang_vel"],
        }
    else:
        ref_dict = {"max_local_self_obs": max_local_self_obs}
    projected_gravity = quat_rotate_inverse(
        base_quat, env.gravity_vec[0:1].repeat(max_local_self_obs.shape[0], 1), w_last=True
    )
    g1env_state = torch.cat([ref_dof_pos, ref_dof_vel, projected_gravity, ref_ang_vel], dim=-1)
    bogus_actions = ref_dof_pos
    g1env_last_action = bogus_actions
    g1env_privileged_obs = ref_dict["max_local_self_obs"]
    g1env_obs = {"state": g1env_state, "privileged_state": g1env_privileged_obs, "root_pose": root_pose}
    if include_last_action:
        g1env_obs["last_action"] = g1env_last_action * 0
    return (g1env_obs, ref_dict)


class HumanoidVerseIsaacTrackingEvaluationConfig(BaseEvalConfig):
    name_in_logs: str = "humanoidverse_tracking_eval"
    env: HumanoidVerseIsaacConfig | None = None
    num_envs: int = 1024
    reference_motion_path: str | None = None
    n_episodes_per_motion: int = 1
    include_results_from_all_envs: bool = False
    max_steps_per_motion: int | None = None
    emd_max_points: int | None = None
    disable_tqdm: bool = True

    def build(self):
        return HumanoidVerseIsaacTrackingEvaluation(self)


class HumanoidVerseIsaacTrackingEvaluation:
    def __init__(self, config: HumanoidVerseIsaacTrackingEvaluationConfig):
        self.cfg = config

    def run(
        self, *, timestep, agent_or_model, logger, env: HumanoidVerseVectorEnv | None = None, **kwargs
    ) -> Dict[str, Any]:
        if self.cfg.max_steps_per_motion is not None and self.cfg.max_steps_per_motion <= 0:
            raise ValueError("max_steps_per_motion must be > 0 or None")
        if self.cfg.emd_max_points is not None and self.cfg.emd_max_points <= 1:
            raise ValueError("emd_max_points must be > 1 or None")
        if env is None:
            if self.cfg.env is None:
                raise ValueError("Either env or cfg.env must be provided")
            (env, _) = self.cfg.env.build(num_envs=self.cfg.num_envs)
        elif self.cfg.env is not None:
            raise ValueError(
                "Both env and cfg.env are provided, please provide only one (which one you want to evaluate with?)"
            )
        with temporary_reference_motion_dataset(env, self.cfg.reference_motion_path):
            return self._run_with_selected_dataset(
                timestep=timestep, agent_or_model=agent_or_model, logger=logger, env=env, **kwargs
            )

    def _run_with_selected_dataset(
        self, *, timestep, agent_or_model, logger, env: HumanoidVerseVectorEnv, **kwargs
    ) -> Dict[str, Any]:
        del kwargs
        self.motion_ids = list(range(env._env._motion_lib._num_unique_motions))
        n_envs = env._env.num_envs
        metrics = {}
        for repetition_i in range(self.cfg.n_episodes_per_motion):
            run_metrics = {}
            for motion_id_chunk_start in range(0, len(self.motion_ids), n_envs):
                motion_id_chunk = self.motion_ids[motion_id_chunk_start : motion_id_chunk_start + n_envs]
                motion_chunk_results = _async_tracking_worker(
                    (motion_id_chunk, 0, agent_or_model),
                    env=env,
                    disable_tqdm=self.cfg.disable_tqdm,
                    include_results_from_all_envs=self.cfg.include_results_from_all_envs,
                    max_steps_per_motion=self.cfg.max_steps_per_motion,
                    emd_max_points=self.cfg.emd_max_points,
                )
                for k, v in motion_chunk_results.items():
                    assert k not in run_metrics, "Tried to override existing metric"
                    run_metrics[k] = v
            if self.cfg.n_episodes_per_motion == 1:
                metrics = run_metrics
            else:
                for k, v in run_metrics.items():
                    metrics[f"{k}_repetition#{repetition_i}"] = v
        aggregate = collections.defaultdict(list)
        wandb_dict = {}
        for _, metr in metrics.items():
            for k, v in metr.items():
                if isinstance(v, numbers.Number):
                    aggregate[k].append(v)
        for k, v in aggregate.items():
            wandb_dict[k] = np.mean(v)
            wandb_dict[f"{k}#std"] = np.std(v)
        mpjpe_weights = np.asarray(
            [metr["SONIC_mpjpe_num_frames"] for metr in metrics.values() if "SONIC_mpjpe_num_frames" in metr],
            dtype=np.float64,
        )
        if mpjpe_weights.size:
            for key in ("SONIC_mpjpe_g", "SONIC_mpjpe_l", "SONIC_mpjpe_pa"):
                values = np.asarray([metr[key] for metr in metrics.values() if key in metr], dtype=np.float64)
                if values.size != mpjpe_weights.size:
                    continue
                mean = np.average(values, weights=mpjpe_weights)
                variance = np.average((values - mean) ** 2, weights=mpjpe_weights)
                wandb_dict[key] = mean
                wandb_dict[f"{key}#std"] = np.sqrt(variance)
        if logger is not None:
            for k, v in metrics.items():
                v["motion_name"] = k
                v["timestep"] = timestep
                logger.log(v)
        env._env.set_is_training()
        return (metrics, wandb_dict)

    def close(self) -> None:
        if self.mp_manager is not None:
            self.mp_manager.shutdown()


def group_assign_motions_to_envs_with_map(motion_ids, num_envs, device=None):
    motion_ids = torch.tensor(motion_ids, device=device)
    num_motions = len(motion_ids)
    env_idxs = list(range(num_envs))
    random.shuffle(env_idxs)
    shuffled_env_idxs = torch.tensor(env_idxs, device=device, dtype=torch.long)
    assigned = motion_ids[shuffled_env_idxs % num_motions]
    motion_to_envs = {}
    for env_id, motion_id in enumerate(assigned.tolist()):
        motion_to_envs.setdefault(motion_id, []).append(env_id)
    return (assigned, motion_to_envs)


def _async_tracking_worker(
    inputs,
    env: HumanoidVerseVectorEnv,
    disable_tqdm: bool = False,
    include_results_from_all_envs=False,
    max_steps_per_motion: int | None = None,
    emd_max_points: int | None = None,
):
    (motion_ids, pos, agent) = inputs
    model = extract_model(agent)
    isaac_env = env._env
    if not isaac_env._motion_lib.all_motions_loaded:
        isaac_env._motion_lib.all_motions_loaded = True
        isaac_env._motion_lib.load_motions(
            random_sample=False, num_motions_to_load=isaac_env._motion_lib._num_unique_motions, start_idx=0
        )
    metrics = {}
    num_envs = isaac_env.num_envs
    missing_mpjpe_bodies = [name for name in GROOT_MPJPE_BODY_NAMES if name not in isaac_env.body_names]
    if missing_mpjpe_bodies:
        raise ValueError(f"GR00T MPJPE bodies missing from simulator: {missing_mpjpe_bodies}")
    groot_mpjpe_body_indices = [isaac_env.body_names.index(name) for name in GROOT_MPJPE_BODY_NAMES]
    missing_wrist_bodies = [name for name in WRIST_TRACKING_BODY_NAMES if name not in isaac_env.body_names]
    if missing_wrist_bodies:
        raise ValueError(f"Wrist tracking bodies missing from simulator: {missing_wrist_bodies}")
    wrist_tracking_body_indices = [isaac_env.body_names.index(name) for name in WRIST_TRACKING_BODY_NAMES]
    (assigned_motions, motion_to_envs) = group_assign_motions_to_envs_with_map(motion_ids, num_envs)
    if include_results_from_all_envs:
        metric_env_ids_list = list(range(num_envs))
    else:
        metric_env_ids_list = [env_ids[0] for env_ids in motion_to_envs.values()]
    metric_env_index = {env_id: index for (index, env_id) in enumerate(metric_env_ids_list)}
    metric_env_ids = torch.tensor(metric_env_ids_list, dtype=torch.long, device=isaac_env.device)
    ctx_dict = {}
    tracking_targets = {}
    tracking_body_pos = {}
    tracking_wrist_body_pos = {}
    tracking_wrist_body_rot = {}
    tracking_joint_pos = {}
    tracking_full_lengths = {}
    dof_states_list = [None] * num_envs
    root_states_list = [None] * num_envs
    for m_id in motion_ids:
        (tracking_target, tracking_target_dict) = get_backward_observation(
            env._env,
            m_id,
            include_last_action=env.include_last_action,
            privileged_state_world_frame=env.privileged_state_world_frame,
        )
        full_length = int(tracking_target["state"].shape[0])
        tracking_full_lengths[m_id] = full_length
        if max_steps_per_motion is not None:
            selected_length = min(full_length, max_steps_per_motion + 1)

            def _slice_reference(value):
                if torch.is_tensor(value) and value.ndim > 0 and (value.shape[0] == full_length):
                    return value[:selected_length]
                return value

            tracking_target = tree_map(_slice_reference, tracking_target)
            tracking_target_dict = tree_map(_slice_reference, tracking_target_dict)
        tracking_goal = tree_map(lambda x: x[1:], tracking_target)
        ctx = model.tracking_inference(tracking_goal).clone()
        ctx_dict[m_id] = ctx
        tracking_targets[m_id] = tree_map(lambda x: x.cpu(), tracking_target)
        tracking_joint_pos[m_id] = tracking_target_dict["dof_pos"].clone()
        tracking_body_pos[m_id] = tracking_target_dict["ref_body_pos"][:, groot_mpjpe_body_indices].clone()
        tracking_wrist_body_pos[m_id] = tracking_target_dict["ref_body_pos"][:, wrist_tracking_body_indices].clone()
        tracking_wrist_body_rot[m_id] = tracking_target_dict["ref_body_rots"][:, wrist_tracking_body_indices].clone()
        ref_body_rots = tracking_target_dict["ref_body_rots"][0, 0]
        ref_root_init_state = torch.cat(
            [
                tracking_target_dict["ref_body_pos"][0, 0],
                ref_body_rots,
                tracking_target_dict["ref_body_vels"][0, 0],
                tracking_target_dict["ref_body_angular_vels"][0, 0],
            ]
        )
        for env_id in motion_to_envs[m_id]:
            dof_init_state = torch.zeros_like(isaac_env.simulator.dof_state.view(num_envs, -1, 2)[0])
            dof_init_state[..., 0] = tracking_target_dict["dof_pos"][0]
            dof_init_state[..., 1] = tracking_target_dict["ref_dof_vel"][0]
            dof_states_list[env_id] = dof_init_state
            root_states_list[env_id] = ref_root_init_state
    for i in range(num_envs):
        if dof_states_list[i] is None:
            dof_states_list[i] = torch.zeros_like(isaac_env.simulator.dof_state.view(num_envs, -1, 2)[0])
        if root_states_list[i] is None:
            root_states_list[i] = torch.zeros_like(root_states_list[0])
    target_states = {"dof_states": torch.stack(dof_states_list), "root_states": torch.stack(root_states_list)}
    env_ids = list(range(num_envs))
    env_ids = torch.tensor(env_ids, dtype=torch.long, device=isaac_env.device)
    with torch.no_grad():
        (observation, _) = env.reset(target_states=target_states, to_numpy=False)
        assert torch.allclose(env._env.simulator.dof_pos.clone()[env_ids], target_states["dof_states"][..., 0])
        if env.context_length:
            env._update_history(torch.arange(num_envs, device=isaac_env.device), observation, None)
            observation["history"] = {"observation": [], "action": []}
        metric_observation_keys = ("state", "root_pose")
        observation_log = {
            key: [observation[key][metric_env_ids].detach().cpu()]
            for key in metric_observation_keys
            if key in observation
        }
        xpos_log = [isaac_env.simulator._rigid_body_pos.reshape(num_envs, -1, 3)[metric_env_ids].clone()]
        wrist_xpos_log = [isaac_env.simulator._rigid_body_pos[metric_env_ids][:, wrist_tracking_body_indices].clone()]
        wrist_xrot_log = [isaac_env.simulator._rigid_body_rot[metric_env_ids][:, wrist_tracking_body_indices].clone()]
        joint_pos = [isaac_env.simulator.dof_state.view(num_envs, -1, 2)[metric_env_ids, :, 0].clone()]
        max_ctx_len = max([ctx.shape[0] for ctx in ctx_dict.values()])
        full_max_ctx_len = max((length - 1 for length in tracking_full_lengths.values()))
        print(
            f"[tracking-eval] motions={len(motion_ids)}, simulator_envs={num_envs}, metric_envs={len(metric_env_ids_list)}, rollout_steps={max_ctx_len}, full_dataset_max_steps={full_max_ctx_len}, configured_cap={max_steps_per_motion}",
            flush=True,
        )
        rollout_started = time.monotonic()
        for step in tqdm(range(max_ctx_len), desc="Tracking Evaluation", disable=disable_tqdm):
            ctx_batch = []
            for env_id in range(num_envs):
                m_id = assigned_motions[env_id].item()
                ctx = ctx_dict[m_id]
                ctx_t = ctx[step % ctx.shape[0]]
                ctx_batch.append(ctx_t)
            ctx_batch = torch.stack(ctx_batch)
            action = agent.act(observation, ctx_batch, mean=True)
            (observation, _, _, _, _) = env.step(action, to_numpy=False)
            joint_pos.append(isaac_env.simulator.dof_state.view(num_envs, -1, 2)[metric_env_ids, :, 0].clone())
            xpos_log.append(isaac_env.simulator._rigid_body_pos.reshape(num_envs, -1, 3)[metric_env_ids].clone())
            wrist_xpos_log.append(
                isaac_env.simulator._rigid_body_pos[metric_env_ids][:, wrist_tracking_body_indices].clone()
            )
            wrist_xrot_log.append(
                isaac_env.simulator._rigid_body_rot[metric_env_ids][:, wrist_tracking_body_indices].clone()
            )
            for key in observation_log:
                observation_log[key].append(observation[key][metric_env_ids].detach().cpu())
        episode_observation = {key: torch.stack(values).numpy() for (key, values) in observation_log.items()}
        body_pos = torch.stack(xpos_log)
        wrist_body_pos = torch.stack(wrist_xpos_log)
        wrist_body_rot = torch.stack(wrist_xrot_log)
        joint_pos = torch.stack(joint_pos)
        for m_id, envs_with_current_motion in motion_to_envs.items():
            for motion_repetition, env_id in enumerate(envs_with_current_motion):
                selected_env_id = metric_env_index[env_id]
                _joint_pos = joint_pos[: ctx_dict[m_id].shape[0] + 1, selected_env_id]
                _target_joint_pos = tracking_joint_pos[m_id]
                _body_pos = body_pos[: ctx_dict[m_id].shape[0] + 1, selected_env_id, groot_mpjpe_body_indices]
                _target_body_pos = tracking_body_pos[m_id] + isaac_env.env_origins[env_id].view(1, 1, 3)
                _wrist_body_pos = wrist_body_pos[: ctx_dict[m_id].shape[0] + 1, selected_env_id]
                _wrist_body_rot = wrist_body_rot[: ctx_dict[m_id].shape[0] + 1, selected_env_id]
                local_metrics = _calc_metrics(
                    {
                        "tracking_target": tracking_targets[m_id],
                        "motion_id": m_id,
                        "motion_file": isaac_env._motion_lib.curr_motion_keys[m_id],
                        "observation": tree_map(
                            lambda x: x[0 : ctx_dict[m_id].shape[0] + 1, selected_env_id], episode_observation
                        ),
                        "joint_pos": _joint_pos,
                        "target_joint_pos": _target_joint_pos,
                        "body_pos": _body_pos,
                        "target_body_pos": _target_body_pos,
                        "wrist_body_pos": _wrist_body_pos,
                        "wrist_body_rot": _wrist_body_rot,
                        "target_wrist_body_pos": tracking_wrist_body_pos[m_id],
                        "target_wrist_body_rot": tracking_wrist_body_rot[m_id],
                        "tracking_reference_full_length": tracking_full_lengths[m_id],
                    },
                    model=model,
                    emd_max_points=emd_max_points,
                )
                assert len(local_metrics) == 1
                metric_key = list(local_metrics.keys())[0]
                if include_results_from_all_envs:
                    metrics[f"{metric_key}_repetition#{motion_repetition}"] = local_metrics[metric_key]
                else:
                    metrics[metric_key] = local_metrics[metric_key]
                    break
        print(f"[tracking-eval] rollout+metrics completed in {time.monotonic() - rollout_started:.1f}s", flush=True)
    return metrics


QPOS_START = 23 + 3
QPOS_END = 23 + 3 + 23
QVEL_IDX = 23


def distance_matrix(X: torch.Tensor, Y: torch.Tensor):
    X_norm = X.pow(2).sum(1).reshape(-1, 1)
    Y_norm = Y.pow(2).sum(1).reshape(1, -1)
    val = X_norm + Y_norm - 2 * torch.matmul(X, Y.T)
    return torch.sqrt(torch.clamp(val, min=0))


def _uniformly_subsample_rows(value: torch.Tensor, max_points: int | None) -> torch.Tensor:
    if max_points is None or value.shape[0] <= max_points:
        return value
    indexes = torch.linspace(0, value.shape[0] - 1, steps=max_points, device=value.device).round().long()
    return value[indexes]


def emd_numpy(next_obs: torch.Tensor, tracking_target: torch.Tensor, prefix="", max_points: int | None = None):
    agent_obs = _uniformly_subsample_rows(next_obs, max_points).to("cpu")
    tracked_obs = _uniformly_subsample_rows(tracking_target, max_points).to("cpu")
    cost_matrix = distance_matrix(agent_obs, tracked_obs).cpu().detach().numpy()
    X_pot = np.ones(agent_obs.shape[0]) / agent_obs.shape[0]
    Y_pot = np.ones(tracked_obs.shape[0]) / tracked_obs.shape[0]
    transport_cost = ot.emd2(X_pot, Y_pot, cost_matrix, numItermax=100000)
    return {f"{prefix}emd": transport_cost}


def distance_proximity(
    next_obs: torch.Tensor, tracking_target: torch.Tensor, bound: float = 2.0, margin: float = 2, prefix=""
):
    stats = {}
    dist = torch.norm(next_obs - tracking_target, dim=-1)
    in_bounds_mask = dist <= bound
    out_bounds_mask = dist > bound + margin
    stats[f"{prefix}proximity"] = (
        in_bounds_mask + (bound + margin - dist) / margin * ~in_bounds_mask * ~out_bounds_mask
    ).mean()
    stats[f"{prefix}distance"] = dist.mean()
    return stats


def root_se2_tracking_metrics(root_pose: torch.Tensor, target_root_pose: torch.Tensor) -> Dict[str, torch.Tensor]:
    root_xy = root_pose[..., :2] * ROOT_POSE_SCALE
    target_xy = target_root_pose[..., :2] * ROOT_POSE_SCALE
    xy_error = torch.norm(root_xy - target_xy, dim=-1)
    root_yaw = torch.atan2(root_pose[..., 2], root_pose[..., 3])
    target_yaw = torch.atan2(target_root_pose[..., 2], target_root_pose[..., 3])
    yaw_delta = root_yaw - target_yaw
    yaw_error = torch.abs(torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)))
    se2_score = torch.sqrt((xy_error / ROOT_SE2_XY_SCALE_M).pow(2) + (yaw_error / ROOT_SE2_YAW_SCALE_RAD).pow(2)).mean()
    return {
        "root_se2_score": se2_score,
        "root_xy_error_m": xy_error.mean(),
        "root_yaw_error_deg": torch.rad2deg(yaw_error).mean(),
    }


def _xyzw_quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = torch.nn.functional.normalize(quaternion.float(), dim=-1, eps=1e-08)
    (x, y, z, w) = quaternion.unbind(dim=-1)
    two = 2.0
    return torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1.0 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def _heading_rotation_matrix(rotation: torch.Tensor) -> torch.Tensor:
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    sin_yaw = torch.sin(yaw)
    cos_yaw = torch.cos(yaw)
    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)
    return torch.stack((cos_yaw, -sin_yaw, zeros, sin_yaw, cos_yaw, zeros, zeros, zeros, ones), dim=-1).reshape(
        *yaw.shape, 3, 3
    )


def _rotation_angle_rad(rotation: torch.Tensor, target_rotation: torch.Tensor) -> torch.Tensor:
    relative = rotation.transpose(-2, -1) @ target_rotation
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    return torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))


def compute_wrist_tracking_metrics(body_pos, body_rot, target_body_pos, target_body_rot):
    """Compute two-wrist errors in root-heading and initial-root-aligned world frames.

    Body order is pelvis, left wrist, right wrist. The reset frame is used only
    to align the reference world's xyz origin and yaw to the rollout; the same
    fixed transform is then applied to every reference frame.
    """
    length = min(body_pos.shape[0], body_rot.shape[0], target_body_pos.shape[0], target_body_rot.shape[0])
    if length <= 1:
        raise ValueError("Wrist tracking metrics require at least one post-reset frame")
    body_pos = body_pos[:length].float()
    body_rot_matrix = _xyzw_quaternion_to_matrix(body_rot[:length].to(device=body_pos.device))
    target_body_pos = target_body_pos[:length].to(device=body_pos.device, dtype=body_pos.dtype)
    target_body_rot_matrix = _xyzw_quaternion_to_matrix(
        target_body_rot[:length].to(device=body_pos.device, dtype=body_pos.dtype)
    )
    body_heading = _heading_rotation_matrix(body_rot_matrix[:, 0])
    target_body_heading = _heading_rotation_matrix(target_body_rot_matrix[:, 0])
    body_relative_pos = torch.matmul(
        body_heading.transpose(-2, -1)[:, None], (body_pos[:, 1:] - body_pos[:, :1]).unsqueeze(-1)
    ).squeeze(-1)
    target_relative_pos = torch.matmul(
        target_body_heading.transpose(-2, -1)[:, None], (target_body_pos[:, 1:] - target_body_pos[:, :1]).unsqueeze(-1)
    ).squeeze(-1)
    body_relative_rot = body_heading.transpose(-2, -1)[:, None] @ body_rot_matrix[:, 1:]
    target_relative_rot = target_body_heading.transpose(-2, -1)[:, None] @ target_body_rot_matrix[:, 1:]
    world_alignment_rot = body_heading[0] @ target_body_heading[0].transpose(-2, -1)
    world_alignment_translation = body_pos[0, 0] - world_alignment_rot @ target_body_pos[0, 0]
    aligned_target_pos = (
        torch.matmul(world_alignment_rot, target_body_pos.unsqueeze(-1)).squeeze(-1) + world_alignment_translation
    )
    aligned_target_rot = world_alignment_rot @ target_body_rot_matrix
    post_reset = slice(1, None)
    relative_pos_error = torch.linalg.norm(body_relative_pos[post_reset] - target_relative_pos[post_reset], dim=-1)
    relative_rot_error = _rotation_angle_rad(body_relative_rot[post_reset], target_relative_rot[post_reset])
    world_pos_error = torch.linalg.norm(body_pos[post_reset, 1:] - aligned_target_pos[post_reset, 1:], dim=-1)
    world_rot_error = _rotation_angle_rad(body_rot_matrix[post_reset, 1:], aligned_target_rot[post_reset, 1:])
    return {
        "wrist_relative_pos_error_m": relative_pos_error.mean(),
        "wrist_relative_rot_error_deg": torch.rad2deg(relative_rot_error).mean(),
        "wrist_world_pos_error_m": world_pos_error.mean(),
        "wrist_world_rot_error_deg": torch.rad2deg(world_rot_error).mean(),
    }


def _calc_metrics(ep, model=None, emd_max_points: int | None = None):
    metr = {}
    next_obs = torch.tensor(ep["observation"]["state"][:, :QVEL_IDX], dtype=torch.float32)
    tracking_target = torch.tensor(ep["tracking_target"]["state"][:, :QVEL_IDX], dtype=torch.float32)
    dist_prox_res = distance_proximity(next_obs=next_obs, tracking_target=tracking_target, prefix="obs_state_")
    metr.update(dist_prox_res)
    emd_res = emd_numpy(
        next_obs=next_obs, tracking_target=tracking_target, prefix="obs_state_", max_points=emd_max_points
    )
    metr.update(emd_res)
    root_pose = torch.as_tensor(ep["observation"]["root_pose"], dtype=torch.float32)
    target_root_pose = torch.as_tensor(ep["tracking_target"]["root_pose"], dtype=torch.float32)
    metr.update(root_se2_tracking_metrics(root_pose=root_pose, target_root_pose=target_root_pose))
    phc_metrics = compute_joint_pos_metrics(
        joint_pos=ep["joint_pos"], target_joint_pos=ep["target_joint_pos"], emd_max_points=emd_max_points
    )
    metr.update(phc_metrics)
    metr.update(compute_cartesian_mpjpe_metrics(body_pos=ep["body_pos"], target_body_pos=ep["target_body_pos"]))
    upper_indices = [GROOT_MPJPE_BODY_NAMES.index(name) for name in UPPER_BODY_MPJPE_BODY_NAMES]
    upper_body_pos = ep["body_pos"][:, upper_indices]
    upper_target_body_pos = ep["target_body_pos"][:, upper_indices]
    metr["upper_body_mpjpe_l"] = compute_upper_body_mpjpe_l(upper_body_pos, upper_target_body_pos)
    upper_body_pos_post = upper_body_pos[1:]
    upper_target_post = upper_target_body_pos[1:].to(device=upper_body_pos_post.device, dtype=upper_body_pos_post.dtype)
    metr["upper_body_mpjpe_g"] = (
        torch.linalg.norm(upper_body_pos_post[:, 1:] - upper_target_post[:, 1:], dim=-1).mean() * 1000.0
    )
    metr.update(
        compute_wrist_tracking_metrics(
            body_pos=ep["wrist_body_pos"],
            body_rot=ep["wrist_body_rot"],
            target_body_pos=ep["target_wrist_body_pos"],
            target_body_rot=ep["target_wrist_body_rot"],
        )
    )
    if model is not None and hasattr(model, "tracking_goal_alignment"):
        metr.update(model.tracking_goal_alignment(ep["observation"], ep["tracking_target"]))
    for k, v in metr.items():
        if isinstance(v, torch.Tensor):
            metr[k] = v.tolist()
    metr["motion_id"] = ep["motion_id"]
    metr["motion_file"] = ep["motion_file"]
    metr["tracking_evaluated_steps"] = max(0, int(next_obs.shape[0]) - 1)
    metr["tracking_reference_full_steps"] = max(0, int(ep["tracking_reference_full_length"]) - 1)
    metr["tracking_coverage_fraction"] = float(next_obs.shape[0] / ep["tracking_reference_full_length"])
    return {ep["motion_file"]: metr}


def compute_joint_pos_metrics(joint_pos, target_joint_pos, emd_max_points: int | None = None):
    stats = {}
    qpos_l2_mrad = torch.norm(joint_pos - target_joint_pos, dim=-1).mean(-1) * 1000
    stats["mpjpe_l"] = qpos_l2_mrad
    stats["qpos_l2_mrad"] = qpos_l2_mrad
    vel_gt = target_joint_pos[:, 1:] - target_joint_pos[:, :-1]
    vel_pred = joint_pos[:, 1:] - joint_pos[:, :-1]
    stats["vel_dist"] = torch.norm(vel_pred - vel_gt, dim=-1).mean(-1) * 1000
    accel_gt = target_joint_pos[:, :-2] - 2 * target_joint_pos[:, 1:-1] + target_joint_pos[:, 2:]
    accel_pred = joint_pos[:, :-2] - 2 * joint_pos[:, 1:-1] + joint_pos[:, 2:]
    stats["accel_dist"] = torch.norm(accel_pred - accel_gt, dim=-1).mean(-1) * 100
    stats.update(distance_proximity(next_obs=joint_pos, tracking_target=target_joint_pos, prefix=""))
    stats.update(emd_numpy(next_obs=joint_pos, tracking_target=target_joint_pos, prefix="", max_points=emd_max_points))
    return stats


def compute_cartesian_mpjpe_metrics(body_pos, target_body_pos):
    """Match GR00T-WBC's 14-body global/local MPJPE definitions."""
    length = min(body_pos.shape[0], target_body_pos.shape[0])
    body_count = min(body_pos.shape[1], target_body_pos.shape[1])
    body_pos = body_pos[1:length, :body_count]
    target_body_pos = target_body_pos[1:length, :body_count].to(device=body_pos.device, dtype=body_pos.dtype)
    if body_pos.shape[0] == 0:
        raise ValueError("Cartesian MPJPE requires at least one post-reset frame")
    global_error = torch.linalg.norm(body_pos - target_body_pos, dim=-1)
    body_pos_relative = body_pos - body_pos[:, :1]
    target_body_pos_relative = target_body_pos - target_body_pos[:, :1]
    root_relative_error = torch.linalg.norm(body_pos_relative - target_body_pos_relative, dim=-1)
    mpjpe_g = global_error.mean() * 1000.0
    mpjpe_l = root_relative_error.mean() * 1000.0
    mpjpe_pa = _procrustes_mpjpe_mm(body_pos_relative, target_body_pos_relative)
    return {
        "SONIC_mpjpe_g": mpjpe_g,
        "SONIC_mpjpe_l": mpjpe_l,
        "SONIC_mpjpe_pa": mpjpe_pa,
        "SONIC_mpjpe_num_frames": body_pos.shape[0],
    }


def compute_upper_body_mpjpe_l(body_pos, target_body_pos):
    """Pelvis-aligned upper-body MPJPE in millimetres, excluding the pelvis anchor."""
    length = min(body_pos.shape[0], target_body_pos.shape[0])
    body_count = min(body_pos.shape[1], target_body_pos.shape[1])
    if body_count < 2:
        raise ValueError("Upper-body MPJPE requires pelvis plus at least one upper body")
    body_pos = body_pos[1:length, :body_count]
    target_body_pos = target_body_pos[1:length, :body_count].to(device=body_pos.device, dtype=body_pos.dtype)
    if body_pos.shape[0] == 0:
        raise ValueError("Upper-body MPJPE requires at least one post-reset frame")
    body_relative = body_pos - body_pos[:, :1]
    target_relative = target_body_pos - target_body_pos[:, :1]
    return torch.linalg.norm(body_relative[:, 1:] - target_relative[:, 1:], dim=-1).mean() * 1000.0


def _procrustes_mpjpe_mm(predicted, target):
    """Per-frame similarity-aligned MPJPE, matching SMPLSim p_mpjpe."""
    mu_target = target.mean(dim=1, keepdim=True)
    mu_predicted = predicted.mean(dim=1, keepdim=True)
    target_centered = target - mu_target
    predicted_centered = predicted - mu_predicted
    target_norm = torch.sqrt(target_centered.square().sum(dim=(1, 2), keepdim=True)).clamp_min(1e-08)
    predicted_norm = torch.sqrt(predicted_centered.square().sum(dim=(1, 2), keepdim=True)).clamp_min(1e-08)
    target_normalized = target_centered / target_norm
    predicted_normalized = predicted_centered / predicted_norm
    covariance = target_normalized.transpose(1, 2) @ predicted_normalized
    (u, singular_values, vh) = torch.linalg.svd(covariance)
    v = vh.transpose(1, 2)
    det_sign = torch.sign(torch.linalg.det(v @ u.transpose(1, 2)))
    det_sign = torch.where(det_sign == 0, torch.ones_like(det_sign), det_sign)
    v = v.clone()
    singular_values = singular_values.clone()
    v[:, :, -1] *= det_sign[:, None]
    singular_values[:, -1] *= det_sign
    rotation = v @ u.transpose(1, 2)
    scale = singular_values.sum(dim=1, keepdim=True)[:, :, None] * target_norm / predicted_norm
    translation = mu_target - scale * (mu_predicted @ rotation)
    predicted_aligned = scale * (predicted @ rotation) + translation
    return torch.linalg.norm(predicted_aligned - target, dim=-1).mean() * 1000.0
