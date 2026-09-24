from __future__ import annotations

from typing import Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def resolve_dist_fn(distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"):
    dist_fn = math_utils.sample_uniform
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise ValueError(f"Unrecognized distribution {distribution}")
    return dist_fn


def randomize_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    distribution_params: tuple[float, float] | tuple[torch.Tensor, torch.Tensor],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    num_envs: int = 1,
):
    """Randomize the com of the bodies by adding, scaling or setting random values.

    This function allows randomizing the center of mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales or sets the values into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    coms = asset.root_physx_view.get_coms()
    coms[env_ids[:, None], body_ids] = env.default_coms[env_ids[:, None], body_ids].clone()
    dist_fn = resolve_dist_fn(distribution)
    if isinstance(distribution_params[0], torch.Tensor):
        distribution_params = (distribution_params[0].to(coms.device), distribution_params[1].to(coms.device))
    env.base_com_bias[env_ids, :] = dist_fn(
        *distribution_params, (env_ids.shape[0], env.base_com_bias.shape[1]), device=coms.device
    )
    if operation == "add":
        coms[env_ids[:, None], body_ids, :3] += env.base_com_bias[env_ids[:, None], :]
    elif operation == "abs":
        coms[env_ids[:, None], body_ids, :3] = env.base_com_bias[env_ids[:, None], :]
    elif operation == "scale":
        coms[env_ids[:, None], body_ids, :3] *= env.base_com_bias[env_ids[:, None], :]
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    asset.root_physx_view.set_coms(coms, env_ids)
