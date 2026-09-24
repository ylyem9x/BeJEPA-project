import copy
import os
from typing import Any, Dict, List

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from torch import nn


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def pre_process_config(config) -> None:
    for name, resolver in {"eval": eval, "len": len}.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)
    obs_dim_dict = dict()
    _obs_key_list = config.env.config.obs.obs_dict
    _aux_obs_key_list = config.env.config.obs.obs_auxiliary
    assert set(config.env.config.obs.noise_scales.keys()) == set(config.env.config.obs.obs_scales.keys())
    from omegaconf import ListConfig

    if isinstance(config.env.config.obs.obs_dims, ListConfig):
        each_dict_obs_dims = {k: v for d in config.env.config.obs.obs_dims for (k, v) in d.items()}
        config.env.config.obs.obs_dims = each_dict_obs_dims
    logger.info(f"obs_dims: {config.env.config.obs.obs_dims}")
    auxiliary_obs_dims = {}
    for aux_obs_key, aux_config in _aux_obs_key_list.items():
        auxiliary_obs_dims[aux_obs_key] = 0
        for _key, _num in aux_config.items():
            try:
                assert _key in config.env.config.obs.obs_dims.keys()
                auxiliary_obs_dims[aux_obs_key] += config.env.config.obs.obs_dims[_key] * _num
            except (AssertionError, KeyError):
                logger.warning(f"aux_obs_key: {aux_obs_key} not found in obs_dims")
    logger.info(f"auxiliary_obs_dims: {auxiliary_obs_dims}")
    for obs_key, obs_config in _obs_key_list.items():
        obs_dim_dict[obs_key] = 0
        for key in obs_config:
            if key.endswith("_raw"):
                key = key[:-4]
            if key in config.env.config.obs.obs_dims.keys():
                obs_dim_dict[obs_key] += config.env.config.obs.obs_dims[key]
                logger.info(f"{obs_key}: {key} has dim: {config.env.config.obs.obs_dims[key]}")
            else:
                obs_dim_dict[obs_key] += auxiliary_obs_dims[key]
                logger.info(f"{obs_key}: {key} has dim: {auxiliary_obs_dims[key]}")
    config.robot.algo_obs_dim_dict = obs_dim_dict
    OmegaConf.set_struct(config.env.config.obs.obs_dims, False)
    config.env.config.obs.obs_dims.update(auxiliary_obs_dims)
    logger.info(f"algo_obs_dim_dict: {config.robot.algo_obs_dim_dict}")


def parse_observation(
    cls: Any,
    key_list: List,
    buf_dict: Dict,
    obs_scales: Dict,
    noise_scales: Dict,
    current_noise_curriculum_value: Any,
    use_noise: bool = True,
) -> None:
    """Parse observations for the legged_robot_base class"""
    for obs_key in key_list:
        if use_noise:
            obs_noise = noise_scales[obs_key] * current_noise_curriculum_value
        else:
            obs_noise = 0.0
        actor_obs = getattr(cls, f"_get_obs_{obs_key}")().clone()
        obs_scale = obs_scales[obs_key]
        buf_dict[obs_key] = (actor_obs + (torch.rand_like(actor_obs) * 2.0 - 1.0) * obs_noise) * obs_scale


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, "memory_a"):
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "policy_1.pt")
        model = copy.deepcopy(actor_critic.actor).to("cpu")
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer("hidden_state", torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        (out, (h, c)) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "policy_lstm_1.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


def _past_reference_history(value: torch.Tensor, length: int) -> torch.Tensor:
    """Match HumanoidVerse's most-recent-first, past-only history layout."""
    if length <= 0:
        return value.new_zeros((value.shape[0], 0))
    history = value.new_zeros((value.shape[0], length, value.shape[-1]))
    for offset in range(1, length + 1):
        history[offset:, offset - 1] = value[:-offset]
    return history.flatten(1)


def _reference_history_actor(
    history_config, sources: dict[str, torch.Tensor], root_pose: torch.Tensor, root_pose_history_length: int
) -> torch.Tensor:
    fields = []
    for key in sorted(history_config.keys()):
        source_key = "actions" if key == "action" else key
        if source_key not in sources:
            raise KeyError(f"Reference history cannot synthesize field {key!r}")
        fields.append(_past_reference_history(sources[source_key], int(history_config[key])))
    if root_pose_history_length > 0:
        fields.append(_past_reference_history(root_pose, root_pose_history_length))
    return torch.cat(fields, dim=-1)


def _reference_motion_times(motion_length, dt, device, minimum_frames=0):
    """Keep the sampling range and optionally include the final control frame."""
    count = max(int(np.ceil((motion_length / dt).cpu())), minimum_frames)
    return torch.arange(count, device=device) * dt


def get_backward_observation(
    env,
    motion_id,
    use_root_height_obs: bool = False,
    velocity_multiplier: float = 1.0,
    privileged_state_world_frame: bool | None = None,
    minimum_frames: int = 0,
) -> torch.Tensor:
    from humanoidverse.agents.envs.humanoidverse_isaac import (
        make_reference_root_pose_observation,
        privileged_state_to_root_pose_frame,
    )
    from humanoidverse.envs.legged_robot_motions.legged_robot_motions import (
        compute_humanoid_observations_max,
        compute_humanoid_observations_max_with_contact,
    )
    from humanoidverse.utils.torch_utils import quat_rotate_inverse

    motion_times = _reference_motion_times(
        env._motion_lib._motion_lengths[motion_id], env.dt, env.device, minimum_frames
    )
    motion_state = env._motion_lib.get_motion_state(motion_id, motion_times)
    ref_body_pos = motion_state["rg_pos_t"]
    ref_body_rots = motion_state["rg_rot_t"]
    ref_body_vels = motion_state["body_vel_t"] * velocity_multiplier
    ref_body_angular_vels = motion_state["body_ang_vel_t"] * velocity_multiplier
    ref_dof_pos = motion_state["dof_pos"] - env.default_dof_pos[0]
    ref_dof_vel = motion_state["dof_vel"] * velocity_multiplier
    if env.use_contact_in_obs_max:
        contact_binary = env.foot_contact_detect(ref_body_pos, ref_body_vels)
        obs_dict = compute_humanoid_observations_max_with_contact(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=use_root_height_obs,
            contact_binary=contact_binary,
        )
    else:
        obs_dict = compute_humanoid_observations_max(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=use_root_height_obs,
        )
    max_local_self_obs = torch.cat([v for v in obs_dict.values()], dim=-1)
    root_pose = make_reference_root_pose_observation(ref_body_pos, ref_body_rots)
    if privileged_state_world_frame is None:
        privileged_state_world_frame = bool(getattr(env, "privileged_state_world_frame", False))
    if privileged_state_world_frame:
        max_local_self_obs = privileged_state_to_root_pose_frame(
            max_local_self_obs, root_pose, body_count=ref_body_pos.shape[1], root_height_obs=use_root_height_obs
        )
    if env.config.obs.use_obs_filter:
        base_quat = ref_body_rots[:, 0]
        ref_ang_vel = ref_body_angular_vels[:, 0]
        projected_gravity = quat_rotate_inverse(
            base_quat, env.gravity_vec[0:1].repeat(max_local_self_obs.shape[0], 1), w_last=True
        )
        bogus_actions = ref_dof_pos
        history_sources = {
            "actions": bogus_actions,
            "base_ang_vel": ref_ang_vel,
            "dof_pos": ref_dof_pos,
            "dof_vel": ref_dof_vel,
            "projected_gravity": projected_gravity,
        }
        bogus_history_actor = _reference_history_actor(
            env.config.obs.obs_auxiliary["history_actor"],
            history_sources,
            root_pose,
            int(getattr(env, "_humanoidverse_actor_root_pose_history_length", 0)),
        )
        ref_dict = {
            "actions": bogus_actions,
            "ref_ang_vel": ref_ang_vel,
            "ref_dof_pos": ref_dof_pos,
            "ref_dof_vel": ref_dof_vel,
            "dof_pos": motion_state["dof_pos"],
            "fake_history": bogus_history_actor,
            "max_local_self_obs": max_local_self_obs,
            "projected_gravity": projected_gravity,
            "ref_body_pos": ref_body_pos,
            "ref_body_rots": ref_body_rots,
            "ref_body_vels": ref_body_vels,
            "ref_body_angular_vels": ref_body_angular_vels,
        }
        state = torch.cat([ref_dof_pos, ref_dof_vel, projected_gravity, ref_ang_vel], dim=-1)
        last_action = bogus_actions
        obs_scales = getattr(getattr(env, "config", None), "obs_scales", None)
        if obs_scales is None:
            obs_scales = getattr(getattr(env.config, "obs", None), "obs_scales", {})
        clean_state = torch.cat(
            [
                ref_dof_pos * float(obs_scales.get("dof_pos", 1.0)),
                ref_dof_vel * float(obs_scales.get("dof_vel", 1.0)),
                projected_gravity * float(obs_scales.get("projected_gravity", 1.0)),
                ref_ang_vel * float(obs_scales.get("base_ang_vel", 1.0)),
            ],
            dim=-1,
        )
        clean_privileged_state = max_local_self_obs * float(obs_scales.get("max_local_self", 1.0))
        clean_encoder_observation = {
            "state": clean_state,
            "privileged_state": clean_privileged_state,
            "root_pose": root_pose,
        }
        bfmzero_obs = {
            "state": state,
            "last_action": last_action,
            "history_actor": bogus_history_actor,
            "privileged_state": max_local_self_obs,
            "root_pose": root_pose,
            "encoder_observation": clean_encoder_observation,
        }
        return (bfmzero_obs, ref_dict)
    else:
        ref_dict = {"max_local_self_obs": max_local_self_obs}
        return (max_local_self_obs, ref_dict)


def actor_observation_dim(inference_model) -> int:
    """Size of the deployable observation before any policy-state encoder."""
    if hasattr(inference_model, "actor_observation_dim"):
        return int(inference_model.actor_observation_dim)
    return int(inference_model._actor.input_filter.output_space.shape[0])


def export_meta_policy_as_onnx(
    inference_model, path, exported_policy_name, example_obs_dict, z_dim, history: bool = False, use_29dof: bool = True
):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)
    inference_model = inference_model.eval()
    actor = copy.deepcopy(inference_model).to("cpu")

    class PPOWrapper(nn.Module):
        def __init__(self, actor, history):
            """
            model: The original PyTorch model.
            input_keys: List of input names as keys for the input dictionary.
            """
            super(PPOWrapper, self).__init__()
            self.actor = actor
            self.history = history
            actor_filter = actor.cfg.archi.actor.input_filter
            self.actor_keys = (
                list(actor_filter.key) if isinstance(actor_filter.key, (list, tuple)) else [actor_filter.key]
            )
            self.key_dims = {key: actor.obs_space[key].shape[0] for key in self.actor_keys}

        def forward(self, actor_obs):
            """
            Dynamically creates a dictionary from the input keys and args.
            """
            (actor_obs, ctx) = (actor_obs[:, :-z_dim], actor_obs[:, -z_dim:])
            actor_dict = {}
            start = 0
            for key in self.actor_keys:
                end = start + self.key_dims[key]
                actor_dict[key] = actor_obs[:, start:end]
                start = end
            return self.actor.act(actor_dict, ctx)

    wrapper = PPOWrapper(actor, history=history)
    example_input_list = example_obs_dict["actor_obs"]
    torch.onnx.export(
        wrapper,
        example_input_list,
        path,
        verbose=True,
        input_names=["actor_obs"],
        output_names=["action"],
        opset_version=13,
    )
