import json
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Literal, Tuple

import safetensors
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ..base import BaseConfig
from ..encoder_observation import CLEAN_ENCODER_OBS_KEY
from ..envs.utils.gym_spaces import json_to_space, space_to_json
from ..latent_reconstruction import LatentReconstructionTargetMixin
from ..root_pose_goal import align_single_goal_root_pose, align_trajectory_goal_root_pose
from ..utils import load_optimizer_state_preserving_hyperparameters
from .model import LatentReconstructionHeadConfig

JepaNetworkSide = Literal["phi", "psi"]


def _device_type(device: str | torch.device) -> str:
    return torch.device(device).type


def _to_device(value, device: str | torch.device):
    return tree_map(lambda x: x.to(device=device, non_blocking=True) if torch.is_tensor(x) else x, value)


def _distributed_enabled() -> bool:
    return dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)


def _distributed_world_size() -> int:
    return dist.get_world_size() if _distributed_enabled() else 1


def _distributed_label() -> str:
    if not _distributed_enabled():
        return "[rank0/1]"
    return f"[rank{dist.get_rank()}/{dist.get_world_size()}]"


def _tensor_buckets(tensors, bucket_cap_mb: float) -> list[list[torch.Tensor]]:
    """Group same-device/dtype tensors into bounded communication buckets."""
    cap_bytes = max(1, int(bucket_cap_mb * 1024 * 1024))
    buckets: list[list[torch.Tensor]] = []
    current: list[torch.Tensor] = []
    current_bytes = 0
    current_key = None
    for tensor in tensors:
        if tensor.numel() == 0:
            continue
        key = (tensor.device, tensor.dtype, tensor.layout)
        tensor_bytes = tensor.numel() * tensor.element_size()
        if current and (key != current_key or current_bytes + tensor_bytes > cap_bytes):
            buckets.append(current)
            current = []
            current_bytes = 0
        current.append(tensor)
        current_bytes += tensor_bytes
        current_key = key
        if tensor.is_sparse or tensor_bytes >= cap_bytes:
            buckets.append(current)
            current = []
            current_bytes = 0
            current_key = None
    if current:
        buckets.append(current)
    return buckets


@torch.no_grad()
def _all_reduce_tensors(tensors, bucket_cap_mb: float, *, average: bool) -> None:
    """All-reduce tensors with one collective per bucket instead of per parameter."""
    divisor = _distributed_world_size() if average else 1
    for bucket in _tensor_buckets(tensors, bucket_cap_mb):
        if len(bucket) == 1 or bucket[0].is_sparse:
            dist.all_reduce(bucket[0], op=dist.ReduceOp.SUM)
            if divisor != 1:
                bucket[0].div_(divisor)
            continue
        flat = torch.cat([tensor.reshape(-1) for tensor in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        if divisor != 1:
            flat.div_(divisor)
        offset = 0
        for tensor in bucket:
            next_offset = offset + tensor.numel()
            tensor.copy_(flat[offset:next_offset].view_as(tensor))
            offset = next_offset


@torch.no_grad()
def _broadcast_tensors(tensors, bucket_cap_mb: float, *, src: int) -> None:
    """Broadcast model state with bounded flat buffers."""
    for bucket in _tensor_buckets(tensors, bucket_cap_mb):
        if len(bucket) == 1 or bucket[0].is_sparse:
            dist.broadcast(bucket[0], src=src)
            continue
        flat = torch.cat([tensor.reshape(-1) for tensor in bucket])
        dist.broadcast(flat, src=src)
        offset = 0
        for tensor in bucket:
            next_offset = offset + tensor.numel()
            tensor.copy_(flat[offset:next_offset].view_as(tensor))
            offset = next_offset


@contextmanager
def _freeze_module_params(*modules: torch.nn.Module):
    params: list[torch.nn.Parameter] = []
    states: list[bool] = []
    for module in modules:
        for param in module.parameters():
            params.append(param)
            states.append(param.requires_grad)
            param.requires_grad_(False)
    try:
        yield
    finally:
        for param, requires_grad in zip(params, states):
            param.requires_grad_(requires_grad)


class JepaAgentTrainConfig(BaseConfig):
    lr_predictor: float = 0.0003
    lr_latent_reconstruction: float | None = None
    lr_keypoint_encoder: float = 0.0003
    lr_phi: float = 1e-05
    lr_psi: float = 1e-05
    lr_actor: float = 0.0003
    weight_decay: float = 0.0
    clip_grad_norm: float = 0.0
    fused_optimizer: bool = False
    distributed_bucket_cap_mb: float = 25.0
    encoder_target_ema: bool = False
    jepa_symmetric: Literal[False] = False
    jepa_symmetric_side: Literal["phi"] = "phi"
    parallel_jepa_branches: bool = True
    encoder_target_tau: float = 0.005
    predictor_target_tau: float = 0.005
    phi_ortho_coef: float = 0.0
    psi_ortho_coef: float = 0.0
    code_usage_kl_coef: float = 0.0
    code_usage_temperature: float = 0.25
    train_goal_ratio: float = 0.2
    expert_goal_ratio: float = 0.6
    unshuffled_goal_ratio: float = 0.0
    next_obs_goal_ratio: float = 0.0
    relabel_ratio: float | None = 0.8
    predictor_pessimism_penalty: float = 0.0
    actor_pessimism_penalty: float = 0.5
    actor_code_confidence_cap: float = 1.0
    stddev_clip: float = 0.3
    batch_size: int = 512
    discount: float = 0.98
    short_discount: float = 0.5
    short_predictor_loss_coef: float = 1.0
    use_mix_rollout: bool = True
    update_z_every_step: int = 100
    z_buffer_size: int = 8192
    rollout_expert_trajectories: bool = True
    rollout_expert_trajectories_length: int = 250
    rollout_expert_trajectories_percentage: float = 0.5
    tracking_goal_randomize_start: bool = False
    tracking_goal_start_xy_radius_m: float = 0.1
    tracking_goal_start_yaw_limit_rad: float = torch.pi / 8
    bc_coeff: float = 0.0
    exclude_terminal_next_obs_from_train: bool = False
    terminal_batch_sample_max_attempts: int = 64

    def model_post_init(self, context):
        if self.train_goal_ratio + self.expert_goal_ratio > 1:
            raise ValueError("train_goal_ratio + expert_goal_ratio must be <= 1")
        for name, ratio in (
            ("unshuffled_goal_ratio", self.unshuffled_goal_ratio),
            ("next_obs_goal_ratio", self.next_obs_goal_ratio),
        ):
            if not 0 <= ratio <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.lr_latent_reconstruction is not None and self.lr_latent_reconstruction <= 0:
            raise ValueError("lr_latent_reconstruction must be > 0 when set")
        if self.lr_keypoint_encoder <= 0:
            raise ValueError("lr_keypoint_encoder must be > 0")
        if self.code_usage_kl_coef < 0:
            raise ValueError("code_usage_kl_coef must be >= 0")
        if self.code_usage_temperature <= 0:
            raise ValueError("code_usage_temperature must be > 0")
        if not 0 < self.actor_code_confidence_cap <= 1:
            raise ValueError("actor_code_confidence_cap must satisfy 0 < cap <= 1")
        if self.terminal_batch_sample_max_attempts <= 0:
            raise ValueError("terminal_batch_sample_max_attempts must be > 0")
        if self.distributed_bucket_cap_mb <= 0:
            raise ValueError("distributed_bucket_cap_mb must be > 0")
        if self.tracking_goal_start_xy_radius_m < 0:
            raise ValueError("tracking_goal_start_xy_radius_m must be non-negative")
        if not 0 <= self.tracking_goal_start_yaw_limit_rad <= torch.pi:
            raise ValueError("tracking_goal_start_yaw_limit_rad must be in [0, pi]")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must satisfy 0 < discount <= 1")


class JepaAgent(LatentReconstructionTargetMixin):
    @classmethod
    def supported_evaluations(cls):
        return ["reward", "tracking"]

    @property
    def device(self):
        return self._model.device

    def _adam_kwargs(self, *, weight_decay: float | None = None) -> dict:
        """Common Adam options, enabling the CUDA fused implementation safely."""
        kwargs = {
            "capturable": self.cfg.cudagraphs and (not self.cfg.compile),
            "weight_decay": self.cfg.train.weight_decay if weight_decay is None else weight_decay,
        }
        if self.cfg.train.fused_optimizer and torch.device(self.device).type == "cuda":
            kwargs["fused"] = True
        return kwargs

    def act(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        return self._model.act(obs, z, mean)

    @property
    def _active_jepa_side(self) -> JepaNetworkSide:
        return self._model.goal_side

    @property
    def _policy_jepa_side(self) -> JepaNetworkSide:
        return self._model.policy_side

    def _encoder_for_side(self, side: JepaNetworkSide):
        return self._model._encoder_for_side(side)

    def _target_encoder_for_side(self, side: JepaNetworkSide):
        return self._model._target_encoder_for_side(side)

    def _use_target_encoder_for_td(self) -> bool:
        return self.cfg.train.encoder_target_ema

    def _grad_norm(self, parameters) -> torch.Tensor:
        grad_norms = [torch.linalg.vector_norm(p.grad.detach().float()) for p in parameters if p.grad is not None]
        if len(grad_norms) == 0:
            return torch.zeros((), device=self.device, dtype=torch.float32)
        return torch.linalg.vector_norm(torch.stack(grad_norms))

    @torch.no_grad()
    def sync_distributed_state(self, src: int = 0) -> None:
        if not _distributed_enabled():
            return
        state = [param.data for param in self._model.parameters()]
        state.extend((buffer.data for buffer in self._model.buffers()))
        _broadcast_tensors(state, self.cfg.train.distributed_bucket_cap_mb, src=src)

    @torch.no_grad()
    def _sync_gradients(self, parameters) -> None:
        if not _distributed_enabled():
            return
        gradients = [param.grad for param in parameters if param.grad is not None]
        _all_reduce_tensors(gradients, self.cfg.train.distributed_bucket_cap_mb, average=True)

    @torch.no_grad()
    def _sync_model_buffers(self) -> None:
        if not _distributed_enabled():
            return
        floating_buffers = [buffer for buffer in self._model.buffers() if buffer.is_floating_point()]
        other_buffers = [buffer for buffer in self._model.buffers() if not buffer.is_floating_point()]
        _all_reduce_tensors(floating_buffers, self.cfg.train.distributed_bucket_cap_mb, average=True)
        _broadcast_tensors(other_buffers, self.cfg.train.distributed_bucket_cap_mb, src=0)

    def _step_optimizer_if_finite(
        self, optimizer: torch.optim.Optimizer, parameters, clip_grad_norm: float | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._step_optimizers_if_finite([(optimizer, parameters, clip_grad_norm)])[0]

    def _step_optimizers_if_finite(self, specs) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Prepare several optimizers, then make one host finite-check sync."""
        prepared = []
        for optimizer, parameters, clip_grad_norm in specs:
            parameters = tuple(parameters)
            self._sync_gradients(parameters)
            if clip_grad_norm is None:
                grad_norm = self._grad_norm(parameters)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
            prepared.append((optimizer, grad_norm, torch.isfinite(grad_norm)))
        step_values = torch.stack([step_ok for (_, _, step_ok) in prepared]).tolist()
        results = []
        for (optimizer, grad_norm, step_ok), should_step in zip(prepared, step_values):
            if should_step:
                optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)
            results.append((grad_norm.detach(), step_ok.detach().to(dtype=torch.float32)))
        return results

    def _encode_side(
        self,
        side: JepaNetworkSide,
        obs: torch.Tensor | dict[str, torch.Tensor],
        *,
        target: bool = False,
        quantize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        encoder = self._target_encoder_for_side(side) if target else self._encoder_for_side(side)
        pre_quant = encoder(self._model._encoder_obs(obs))
        (z, codes) = self._model.quantize_latent(pre_quant) if quantize else (pre_quant, None)
        return (pre_quant, z, codes)

    def _goal_encoder(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return self._encode_side(self._active_jepa_side, obs)[1]

    def _code_usage_loss(self, pre_quant: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = torch.zeros((), device=pre_quant.device, dtype=pre_quant.dtype)
        return (zero, zero, zero)

    def _latent_reconstruction_sides(self, head_cfg: LatentReconstructionHeadConfig) -> tuple[JepaNetworkSide, ...]:
        if head_cfg.side == "goal":
            return (self._active_jepa_side,)
        if head_cfg.side == "policy":
            return (self._policy_jepa_side,)
        if head_cfg.side == "both":
            return tuple(self._model._jepa_sides)
        return (head_cfg.side,)

    def _latent_reconstruction_loss(
        self, side: JepaNetworkSide, enc: torch.Tensor, obs: torch.Tensor | dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = torch.zeros((), device=enc.device, dtype=enc.dtype)
        if len(self._model._latent_reconstruction_heads) == 0:
            return (zero, {})
        total_loss = zero
        metrics = {}
        for name, head in self._model._latent_reconstruction_heads.items():
            head_cfg = self.cfg.model.archi.latent_reconstruction[name]
            if head_cfg.coef <= 0 or side not in self._latent_reconstruction_sides(head_cfg):
                continue
            target = self._latent_reconstruction_target(head_cfg, obs).detach().to(device=enc.device, dtype=enc.dtype)
            head_input = enc if head_cfg.backprop_to_encoder else enc.detach()
            pred = head(head_input)
            if pred.shape != target.shape:
                raise RuntimeError(
                    f"Latent reconstruction head '{name}' produced shape {tuple(pred.shape)}, but target shape is {tuple(target.shape)}."
                )
            if head_cfg.loss == "smooth_l1":
                element_loss = F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
            else:
                element_loss = F.mse_loss(pred.float(), target.float(), reduction="none")
            if head_cfg.weight is not None:
                element_loss = element_loss * element_loss.new_tensor(head_cfg.weight)
            loss = element_loss.mean().to(dtype=enc.dtype)
            weighted_loss = head_cfg.coef * loss
            total_loss = total_loss + weighted_loss
            metric_prefix = f"latent_recon/{side}/{name}"
            metrics[f"{metric_prefix}_weighted_loss"] = weighted_loss.detach()
        return (total_loss, metrics)

    def history_inference(self, obs, action):
        raise NotImplementedError("JepaAgent does not use the optional recurrent history interface.")

    @torch.no_grad()
    def encode_expert(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        with autocast(device_type=_device_type(self.device), dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            z = self._goal_encoder(next_obs).detach()
            z = z.view(self.cfg.train.batch_size // self.cfg.model.seq_length, self.cfg.model.seq_length, z.shape[-1])
            z = self._model.aggregate_latent_window(z, dim=1)
            return torch.repeat_interleave(z, self.cfg.model.seq_length, dim=0)

    def _orth_loss(self, enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cov = torch.matmul(enc, enc.T)
        diag = -cov.diag().mean()
        offdiag = 0.5 * (cov * self.off_diag).pow(2).sum() / self.off_diag_sum
        return (offdiag + diag, diag, offdiag)

    def _current_root_pose(self, obs: torch.Tensor | dict[str, torch.Tensor] | None) -> torch.Tensor | None:
        if isinstance(obs, dict):
            clean_obs = obs.get(CLEAN_ENCODER_OBS_KEY)
            if isinstance(clean_obs, dict) and "root_pose" in clean_obs:
                return clean_obs["root_pose"].to(self.device, non_blocking=True)
            if "root_pose" in obs:
                return obs["root_pose"].to(self.device, non_blocking=True)
        return None

    def _privileged_state_body_count(self) -> int | None:
        body_names = self.cfg.model.archi.latent_reconstruction_body_names
        return len(body_names) if body_names else None

    def _encode_goal_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        obs = self._model._normalize(obs)
        return self._goal_encoder(obs)

    def _sample_tracking_z(
        self, replay_buffer, batch_dim: int, traj_length: int, current_root_pose: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch = replay_buffer["expert_slicer"].sample(batch_dim * traj_length, seq_length=traj_length)
        goal_obs = _to_device(batch["next"]["observation"], self.device)
        goal_obs = align_trajectory_goal_root_pose(
            goal_obs,
            current_root_pose,
            traj_length=traj_length,
            max_start_dist_m=self.cfg.train.tracking_goal_start_xy_radius_m,
            max_start_yaw_rad=self.cfg.train.tracking_goal_start_yaw_limit_rad,
            randomize_start=self.cfg.train.tracking_goal_randomize_start,
            privileged_state_world_frame=self.cfg.model.privileged_state_world_frame,
            body_count=self._privileged_state_body_count(),
        )
        z = self._encode_goal_obs(goal_obs)
        z = z.view(batch_dim, traj_length, z.shape[-1])
        return self._model.aggregate_latent_sequence(z, dim=1)

    @torch.no_grad()
    def _sample_rollout_z(
        self, batch_size: int, replay_buffer=None, obs: torch.Tensor | dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        current_root_pose = self._current_root_pose(obs)
        if current_root_pose is None or replay_buffer is None:
            if self.cfg.train.use_mix_rollout and (not self.z_buffer.empty()):
                return self.z_buffer.sample(batch_size, device=self.device)
            return self._model.sample_z(batch_size, device=self.device)
        online_z = None
        if "train" in replay_buffer and (not replay_buffer["train"].empty()) and (self.cfg.train.train_goal_ratio > 0):
            try:
                batch = replay_buffer["train"].sample(batch_size)
            except ValueError:
                batch = None
            if batch is not None:
                goal_obs = _to_device(batch["next"]["observation"], self.device)
                goal_obs = align_single_goal_root_pose(
                    goal_obs,
                    current_root_pose,
                    max_dist_m=1.0,
                    max_yaw_rad=torch.pi,
                    privileged_state_world_frame=self.cfg.model.privileged_state_world_frame,
                    body_count=self._privileged_state_body_count(),
                )
                online_z = self._encode_goal_obs(goal_obs)
        expert_z = None
        if "expert_slicer" in replay_buffer and self.cfg.train.expert_goal_ratio > 0:
            batch = replay_buffer["expert_slicer"].sample(batch_size)
            goal_obs = _to_device(batch["next"]["observation"], self.device)
            goal_obs = align_single_goal_root_pose(
                goal_obs,
                current_root_pose,
                max_dist_m=1.0,
                max_yaw_rad=torch.pi,
                privileged_state_world_frame=self.cfg.model.privileged_state_world_frame,
                body_count=self._privileged_state_body_count(),
            )
            expert_z = self._encode_goal_obs(goal_obs)
        z = self._model.sample_z(batch_size, device=self.device)
        if online_z is not None and self.cfg.train.train_goal_ratio > 0:
            mask = torch.rand((batch_size, 1), device=self.device) < self.cfg.train.train_goal_ratio
            z = torch.where(mask, online_z, z)
        if expert_z is not None and self.cfg.train.expert_goal_ratio > 0:
            mask = torch.rand((batch_size, 1), device=self.device) < self.cfg.train.expert_goal_ratio
            z = torch.where(mask, expert_z, z)
        return z

    def maybe_update_rollout_context(
        self,
        z: torch.Tensor | None,
        step_count: torch.Tensor,
        replay_buffer=None,
        obs: torch.Tensor | dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        current_root_pose = self._current_root_pose(obs)
        if z is not None:
            mask_reset_z = step_count.to(self.device, non_blocking=True) % self.cfg.train.update_z_every_step == 0
            if mask_reset_z.ndim == 1:
                mask_reset_z = mask_reset_z.unsqueeze(-1)
            new_z = self._sample_rollout_z(z.shape[0], replay_buffer=replay_buffer, obs=obs)
            z = torch.where(mask_reset_z, new_z, z.to(self.device, non_blocking=True))
            if (
                self.cfg.train.rollout_expert_trajectories
                and replay_buffer is not None
                and ("expert_slicer" in replay_buffer)
            ):
                idxs = step_count.to(self.device, non_blocking=True) % self.cfg.train.rollout_expert_trajectories_length
                if torch.any(idxs == 0):
                    n_elem = int(self.cfg.train.rollout_expert_trajectories_percentage * step_count.shape[0])
                    self.env_idx_with_expert_rollout = torch.randint(
                        0, step_count.shape[0], size=(n_elem,), device=self.device
                    )
                    tracking_root_pose = (
                        current_root_pose[self.env_idx_with_expert_rollout] if current_root_pose is not None else None
                    )
                    self.tracking_z = self._sample_tracking_z(
                        replay_buffer,
                        n_elem,
                        self.cfg.train.rollout_expert_trajectories_length,
                        current_root_pose=tracking_root_pose,
                    )
                if self.env_idx_with_expert_rollout is not None:
                    mod_time = idxs[self.env_idx_with_expert_rollout].ravel()
                    z[self.env_idx_with_expert_rollout] = self.tracking_z[
                        torch.arange(len(self.env_idx_with_expert_rollout), device=self.device), mod_time
                    ]
        else:
            z = self._sample_rollout_z(step_count.shape[0], replay_buffer=replay_buffer, obs=obs)
            if (
                self.cfg.train.rollout_expert_trajectories
                and replay_buffer is not None
                and ("expert_slicer" in replay_buffer)
            ):
                n_elem = int(self.cfg.train.rollout_expert_trajectories_percentage * step_count.shape[0])
                self.env_idx_with_expert_rollout = torch.randint(
                    0, step_count.shape[0], size=(n_elem,), device=self.device
                )
                tracking_root_pose = (
                    current_root_pose[self.env_idx_with_expert_rollout] if current_root_pose is not None else None
                )
                self.tracking_z = self._sample_tracking_z(
                    replay_buffer,
                    n_elem,
                    self.cfg.train.rollout_expert_trajectories_length,
                    current_root_pose=tracking_root_pose,
                )
                z[self.env_idx_with_expert_rollout] = self.tracking_z[:, 0]
        return z

    def rollout_replay_fields(self, context):
        """Store the rollout goal latent in the replay buffer."""
        return {} if context is None else {"z": context}

    def _next_terminated_mask(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        terminated = batch["next"]["terminated"].bool()
        return terminated.reshape(terminated.shape[0], -1).any(dim=1)

    def _index_batch(self, batch: Dict[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return tree_map(lambda x: x.index_select(0, idx.to(x.device)), batch)

    def _concat_batch_chunks(self, chunks: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if len(chunks) == 1:
            return chunks[0]
        return tree_map(lambda *xs: torch.cat(xs, dim=0), *chunks)

    def _sample_train_batch(self, replay_buffer) -> Dict[str, torch.Tensor]:
        if not self.cfg.train.exclude_terminal_next_obs_from_train:
            return replay_buffer["train"].sample(self.cfg.train.batch_size)
        chunks = []
        num_selected = 0
        attempts = 0
        batch_size = self.cfg.train.batch_size
        while num_selected < batch_size and attempts < self.cfg.train.terminal_batch_sample_max_attempts:
            candidate = replay_buffer["train"].sample(batch_size)
            valid_idx = (~self._next_terminated_mask(candidate)).nonzero(as_tuple=False).flatten()
            attempts += 1
            if valid_idx.numel() == 0:
                continue
            valid_idx = valid_idx[: batch_size - num_selected]
            chunks.append(self._index_batch(candidate, valid_idx))
            num_selected += int(valid_idx.numel())
        if num_selected < batch_size:
            raise RuntimeError(
                "Could not sample a train batch without terminal next observations. The replay buffer may contain too many auto-reset transitions."
            )
        return self._concat_batch_chunks(chunks)

    def _prepare_train_batch(self, replay_buffer):
        batch = self._sample_train_batch(replay_buffer)
        obs = _to_device(batch["observation"], self.device)
        action = batch["action"].to(self.device, non_blocking=True)
        next_obs = _to_device(batch["next"]["observation"], self.device)
        discount = self.cfg.train.discount * ~batch["next"]["terminated"].to(self.device, non_blocking=True)
        self._model._obs_normalizer(obs)
        self._model._obs_normalizer(next_obs)
        obs = self._model._normalize(obs)
        next_obs = self._model._normalize(next_obs)
        return (batch, obs, action, next_obs, discount)

    def _prepare_expert_batch(self, replay_buffer):
        if "expert_slicer" not in replay_buffer:
            return (None, None, None, None)
        expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
        expert_obs = _to_device(expert_batch["observation"], self.device)
        expert_next_obs = _to_device(expert_batch["next"]["observation"], self.device)
        expert_obs = self._model._normalize(expert_obs)
        expert_next_obs = self._model._normalize(expert_next_obs)
        expert_z = self.encode_expert(expert_next_obs)
        return (expert_batch, expert_obs, expert_next_obs, expert_z)

    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = 0
        preds_mean = preds.mean(dim=dim)
        if preds.shape[dim] <= 1:
            preds_unc = torch.zeros_like(preds_mean)
        else:
            preds_uns = preds.unsqueeze(dim=dim)
            preds_uns2 = preds.unsqueeze(dim=dim + 1)
            preds_diffs = torch.abs(preds_uns - preds_uns2)
            num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
            preds_unc = preds_diffs.sum(dim=(dim, dim + 1)) / num_parallel_scaling
        return (preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc)

    @classmethod
    def load(
        cls,
        path: str,
        device: str | None = None,
        *,
        config: BaseConfig | None = None,
        obs_space=None,
        action_dim: int | None = None,
        strict: bool = False,
        preserve_optimizer_hyperparameters: bool = False,
    ):
        path = Path(path)
        if config is None:
            with (path / "config.json").open() as f:
                loaded_config = json.load(f)
            if device is not None:
                loaded_config["model"]["device"] = device
            config = cls.config_class(**loaded_config)
            if (path / "init_kwargs.pkl").exists():
                with (path / "init_kwargs.pkl").open("rb") as f:
                    args = pickle.load(f)
                obs_space = args["obs_space"]
                action_dim = args["action_dim"]
            else:
                with (path / "init_kwargs.json").open("r") as f:
                    args = json.load(f)
                obs_space = json_to_space(args["obs_space"])
                action_dim = args["action_dim"]
        else:
            if obs_space is None or action_dim is None:
                raise ValueError("obs_space and action_dim are required when loading with the current agent config.")
            if device is not None and config.model.device != device:
                config = config.model_copy(update={"model": config.model.model_copy(update={"device": device})})
        agent = config.build(obs_space, action_dim)
        try:
            safetensors.torch.load_model(agent._model, path / "model/model.safetensors", device=device, strict=strict)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint network parameters do not strictly match the network built from the current code config."
            ) from exc
        optimizers = torch.load(str(path / "optimizers.pth"), weights_only=True, map_location=device)
        for k, v in optimizers.items():
            optimizer = getattr(agent, k, None)
            if optimizer is not None:
                if preserve_optimizer_hyperparameters:
                    load_optimizer_state_preserving_hyperparameters(optimizer, v)
                else:
                    optimizer.load_state_dict(v)
        agent._model.train()
        agent._model.requires_grad_(True)
        return agent

    def save(self, output_folder: str) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
        with (output_folder / "config.json").open("w+") as f:
            json.dump(self.cfg.model_dump(), f, indent=4)
        torch.save(self.optimizer_dict, output_folder / "optimizers.pth")
        model_folder = output_folder / "model"
        model_folder.mkdir(exist_ok=True)
        self._model.save(output_folder=str(model_folder))
        init_kwargs = {"obs_space": space_to_json(self.obs_space), "action_dim": self.action_dim}
        with (output_folder / "init_kwargs.json").open("w") as f:
            json.dump(init_kwargs, f, indent=4)

    def _mark_compiled_step_begin(self) -> None:
        if self.cfg.compile:
            torch.compiler.cudagraph_mark_step_begin()
