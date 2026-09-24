"""Shared latent reconstruction heads and physical-target extraction."""

import typing as tp

import pydantic
import torch

from .base import BaseConfig
from .root_pose_goal import ROOT_POSE_SCALE

LatentReconstructionTarget = tp.Literal["root_pose", "body"]
LatentReconstructionBodyField = tp.Literal["pos", "rot", "pose", "vel", "ang_vel"]
LatentReconstructionBodyPositionFrame = tp.Literal["root", "anchor"]
LatentReconstructionSide = tp.Literal["goal", "policy", "phi", "psi", "both"]
LatentReconstructionLoss = tp.Literal["mse", "smooth_l1"]


class LatentReconstructionHeadConfig(BaseConfig):
    target: LatentReconstructionTarget = "root_pose"
    coef: float = 0.0
    backprop_to_encoder: bool = True
    weight: list[float] | None = None
    hidden_dim: int = 128
    hidden_layers: int = 2
    side: LatentReconstructionSide = "goal"
    bodies: list[str] = pydantic.Field(default_factory=list)
    fields: list[LatentReconstructionBodyField] = pydantic.Field(default_factory=list)
    body_position_frame: LatentReconstructionBodyPositionFrame = "root"
    loss: LatentReconstructionLoss = "mse"

    def model_post_init(self, context):
        if self.coef < 0:
            raise ValueError("latent reconstruction coef must be >= 0.")
        if self.hidden_dim <= 0:
            raise ValueError("latent reconstruction hidden_dim must be > 0.")
        if self.hidden_layers < 0:
            raise ValueError("latent reconstruction hidden_layers must be >= 0.")
        if self.target == "body":
            if len(self.bodies) == 0:
                raise ValueError("body latent reconstruction requires at least one body.")
            if len(self.fields) == 0:
                raise ValueError("body latent reconstruction requires at least one field.")
            if "pose" in self.fields and ("pos" in self.fields or "rot" in self.fields):
                raise ValueError("Use either field='pose' or explicit 'pos'/'rot', not both.")
        elif self.body_position_frame != "root":
            raise ValueError("body_position_frame is only supported for body latent reconstruction.")

    @property
    def output_dim(self) -> int:
        if self.target == "root_pose":
            return 4
        dim = 0
        for field in self.fields:
            if field == "pose":
                dim += 9
            elif field == "rot":
                dim += 6
            else:
                dim += 3
        return len(self.bodies) * dim

    def build(self, z_dim: int) -> torch.nn.Module:
        return LatentReconstructionHead(
            z_dim=z_dim, output_dim=self.output_dim, hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers
        )


class LatentReconstructionHead(torch.nn.Module):
    def __init__(self, z_dim: int, output_dim: int, hidden_dim: int, hidden_layers: int):
        super().__init__()
        layers: list[torch.nn.Module] = []
        input_dim = z_dim
        for _ in range(hidden_layers):
            layers += [torch.nn.Linear(input_dim, hidden_dim), torch.nn.LayerNorm(hidden_dim), torch.nn.Mish()]
            input_dim = hidden_dim
        layers += [torch.nn.Linear(input_dim, output_dim)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LatentReconstructionTargetMixin:
    """Physical-target extraction shared by JEPA and FB decoder losses."""

    @staticmethod
    def _latent_reconstruction_body_fields(head_cfg: LatentReconstructionHeadConfig) -> list[str]:
        fields = []
        for field in head_cfg.fields:
            if field == "pose":
                fields.extend(["pos", "rot"])
            else:
                fields.append(field)
        return fields

    def _latent_reconstruction_body_layout(self, privileged_state: torch.Tensor) -> dict[str, int]:
        body_names = self.cfg.model.archi.latent_reconstruction_body_names
        if len(body_names) == 0:
            raise ValueError("Body latent reconstruction requires cfg.model.archi.latent_reconstruction_body_names.")
        num_bodies = len(body_names)
        base_dim = 3 * (num_bodies - 1) + 6 * num_bodies + 3 * num_bodies + 3 * num_bodies
        prefix_dim = privileged_state.shape[-1] - base_dim
        if prefix_dim < 0:
            raise ValueError(
                f"privileged_state dim {privileged_state.shape[-1]} is too small for {num_bodies} bodies (expected at least {base_dim})."
            )
        pos_start = prefix_dim
        rot_start = pos_start + 3 * (num_bodies - 1)
        vel_start = rot_start + 6 * num_bodies
        ang_vel_start = vel_start + 3 * num_bodies
        return {
            "prefix_dim": prefix_dim,
            "pos_start": pos_start,
            "rot_start": rot_start,
            "vel_start": vel_start,
            "ang_vel_start": ang_vel_start,
        }

    def _latent_reconstruction_body_position_target(
        self,
        head_cfg: LatentReconstructionHeadConfig,
        encoder_obs: dict[str, torch.Tensor],
        position: torch.Tensor,
        layout: dict[str, int],
    ) -> torch.Tensor:
        source_frame = "anchor" if self.cfg.model.privileged_state_world_frame else "root"
        if head_cfg.body_position_frame == source_frame:
            return position
        if "root_pose" not in encoder_obs:
            raise ValueError(
                f"Body position reconstruction in the '{head_cfg.body_position_frame}' frame requires 'root_pose' in encoder observation."
            )
        root_pose = encoder_obs["root_pose"].to(device=position.device, dtype=position.dtype)
        root_xy = root_pose[..., :2] * ROOT_POSE_SCALE
        sin_yaw = root_pose[..., 2]
        cos_yaw = root_pose[..., 3]
        root_height = None
        if layout["prefix_dim"] > 0:
            root_height = encoder_obs["privileged_state"][..., 0].to(device=position.device, dtype=position.dtype)
        (x, y, z) = position.unbind(dim=-1)
        if head_cfg.body_position_frame == "anchor":
            converted_x = cos_yaw * x - sin_yaw * y + root_xy[..., 0]
            converted_y = sin_yaw * x + cos_yaw * y + root_xy[..., 1]
            converted_z = z if root_height is None else z + root_height
        else:
            delta_x = x - root_xy[..., 0]
            delta_y = y - root_xy[..., 1]
            converted_x = cos_yaw * delta_x + sin_yaw * delta_y
            converted_y = -sin_yaw * delta_x + cos_yaw * delta_y
            converted_z = z if root_height is None else z - root_height
        return torch.stack([converted_x, converted_y, converted_z], dim=-1)

    def _latent_reconstruction_body_target(
        self, head_cfg: LatentReconstructionHeadConfig, obs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if "privileged_state" not in obs:
            raise ValueError("Body latent reconstruction requires 'privileged_state' in encoder observation.")
        privileged_state = obs["privileged_state"]
        body_names = self.cfg.model.archi.latent_reconstruction_body_names
        layout = self._latent_reconstruction_body_layout(privileged_state)
        targets = []
        for body in head_cfg.bodies:
            try:
                body_idx = body_names.index(body)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown latent reconstruction body '{body}'. Available bodies: {body_names}."
                ) from exc
            for field in self._latent_reconstruction_body_fields(head_cfg):
                if field == "pos":
                    if body_idx == 0:
                        raise ValueError(
                            "Root body local position is removed from privileged_state and cannot be reconstructed as 'pos'."
                        )
                    start = layout["pos_start"] + 3 * (body_idx - 1)
                    stop = start + 3
                elif field == "rot":
                    start = layout["rot_start"] + 6 * body_idx
                    stop = start + 6
                elif field == "vel":
                    start = layout["vel_start"] + 3 * body_idx
                    stop = start + 3
                elif field == "ang_vel":
                    start = layout["ang_vel_start"] + 3 * body_idx
                    stop = start + 3
                else:
                    raise ValueError(f"Unsupported latent reconstruction body field '{field}'.")
                target = privileged_state[..., start:stop]
                if field == "pos":
                    target = self._latent_reconstruction_body_position_target(head_cfg, obs, target, layout)
                targets.append(target)
        return torch.cat(targets, dim=-1)

    def _latent_reconstruction_target(
        self, head_cfg: LatentReconstructionHeadConfig, obs: torch.Tensor | dict[str, torch.Tensor]
    ) -> torch.Tensor:
        encoder_obs = self._model._encoder_obs(obs)
        if not isinstance(encoder_obs, dict):
            raise ValueError("Latent reconstruction requires dict observations.")
        if head_cfg.target == "root_pose":
            if "root_pose" not in encoder_obs:
                raise ValueError("Root-pose latent reconstruction requires 'root_pose' in encoder observation.")
            return encoder_obs["root_pose"]
        if head_cfg.target == "body":
            return self._latent_reconstruction_body_target(head_cfg, encoder_obs)
        raise ValueError(f"Unsupported latent reconstruction target '{head_cfg.target}'.")
