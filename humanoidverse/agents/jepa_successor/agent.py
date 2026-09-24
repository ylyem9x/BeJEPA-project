"""Training agent for goal-conditioned continuous successor occupancy."""

import math
from typing import Dict, Literal

import pydantic
import torch
import torch.nn.functional as F
from torch import autograd
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ..base import BaseConfig
from ..jepa.agent import (
    JepaAgent,
    JepaAgentTrainConfig,
    _all_reduce_tensors,
    _device_type,
    _distributed_enabled,
    _distributed_label,
    _freeze_module_params,
    _to_device,
)
from ..misc.zbuffer import ZBuffer
from ..nn_models import _soft_update_params, weight_init
from ..pytree_utils import tree_get_batch_size
from .model import (
    JepaSuccessorCprAuxModel,
    JepaSuccessorCprAuxModelConfig,
    JepaSuccessorCprModel,
    JepaSuccessorCprModelConfig,
    JepaSuccessorModel,
    JepaSuccessorModelConfig,
    compose_continuous_successor_value,
)


def _validate_actor_successor_convention(model_cfg, train_cfg) -> None:
    pass


def _validate_dynamics_rollout(model_cfg, train_cfg) -> None:
    pass


class JepaSuccessorAgentTrainConfig(JepaAgentTrainConfig):
    """Optimization settings specific to the JEPA-successor scheme."""

    tracking_goal_randomize_start: bool = True
    tracking_goal_start_xy_radius_m: float = 0.5
    tracking_goal_start_yaw_limit_rad: float = torch.pi
    lr_dynamics: float = 0.0003
    lr_successor: float = 0.0003
    lr_decay_function: Literal["none", "linear", "cosine", "exponential"] = "none"
    lr_decay_networks: list[
        Literal[
            "encoder",
            "predictor",
            "dynamics",
            "successor",
            "actor",
            "critic",
            "discriminator",
            "policy_state_encoder",
            "latent_reconstruction",
        ]
    ] = pydantic.Field(default_factory=list)
    lr_decay_start_step: int = 0
    lr_decay_end_step: int = 100000000
    lr_decay_final_ratio: float = 0.1
    fused_optimizer: bool = True
    dynamics_loss_coef: float = 1.0
    dynamics_loss_coef_schedule_function: Literal["log", "cosine"] = "log"
    dynamics_loss_coef_schedule_start: tuple[int, float] | None = None
    dynamics_loss_coef_schedule_end: tuple[int, float] | None = None
    dynamics_rollout_horizon: Literal[1] = 1
    successor_loss_coef: float = 1.0
    dynamics_backprop_to_encoder: bool = True
    successor_backprop_to_encoder: bool = True
    successor_target_tau: float = 0.005
    random_goal_ratio: float | None = None
    successor_td_use_g_train: bool = False
    actor_use_unscaled_next_z: Literal[False] = False
    target_delta_metrics_every_updates: int = 100

    @property
    def lr_policy_state_encoder(self) -> float:
        """Canonical name for the inherited checkpoint-compatible LR field."""
        return self.lr_keypoint_encoder

    @pydantic.field_validator("dynamics_loss_coef_schedule_start", "dynamics_loss_coef_schedule_end", mode="before")
    @classmethod
    def _restore_schedule_tuple(cls, value):
        return tuple(value) if isinstance(value, list) else value

    def model_post_init(self, context):
        super().model_post_init(context)
        for name in ("train_goal_ratio", "expert_goal_ratio"):
            ratio = getattr(self, name)
            if not 0 <= ratio <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.random_goal_ratio is None:
            if self.train_goal_ratio + self.expert_goal_ratio > 1.000001:
                raise ValueError(
                    "JEPA-successor train_goal_ratio + expert_goal_ratio must be <= 1; the remaining mass is the random-goal ratio"
                )
        else:
            if not 0 <= self.random_goal_ratio <= 1:
                raise ValueError("random_goal_ratio must be in [0, 1]")
            ratio_sum = self.train_goal_ratio + self.expert_goal_ratio + self.random_goal_ratio
            if abs(ratio_sum - 1.0) > 1e-06:
                raise ValueError("JEPA-successor online/expert/random goal ratios must sum to one")
        if self.relabel_ratio is not None and (not 0 <= self.relabel_ratio <= 1):
            raise ValueError("JEPA-successor relabel_ratio must be None or in [0, 1]")
        if self.lr_dynamics <= 0:
            raise ValueError("lr_dynamics must be > 0")
        if self.lr_successor <= 0:
            raise ValueError("lr_successor must be > 0")
        if self.lr_decay_start_step < 0:
            raise ValueError("lr_decay_start_step must be >= 0")
        if self.lr_decay_end_step <= self.lr_decay_start_step:
            raise ValueError("lr_decay_end_step must exceed lr_decay_start_step")
        if not 0 < self.lr_decay_final_ratio <= 1:
            raise ValueError("lr_decay_final_ratio must satisfy 0 < ratio <= 1")
        if len(set(self.lr_decay_networks)) != len(self.lr_decay_networks):
            raise ValueError("lr_decay_networks must not contain duplicates")
        if self.dynamics_loss_coef <= 0:
            raise ValueError("dynamics_loss_coef must be > 0")
        schedule_start = self.dynamics_loss_coef_schedule_start
        schedule_end = self.dynamics_loss_coef_schedule_end
        if (schedule_start is None) != (schedule_end is None):
            raise ValueError("dynamics loss coefficient schedule requires both start and end endpoints")
        if schedule_start is not None and schedule_end is not None:
            (start_step, start_coef) = schedule_start
            (end_step, end_coef) = schedule_end
            if start_step < 0:
                raise ValueError("dynamics loss coefficient schedule start timestep must be >= 0")
            if end_step <= start_step:
                raise ValueError("dynamics loss coefficient schedule end timestep must exceed start timestep")
            if start_coef <= 0 or end_coef <= 0:
                raise ValueError("dynamics loss coefficient schedule endpoint values must be > 0")
        if self.successor_loss_coef <= 0:
            raise ValueError("successor_loss_coef must be > 0")
        if not 0 < self.successor_target_tau <= 1:
            raise ValueError("successor_target_tau must satisfy 0 < tau <= 1")
        if self.target_delta_metrics_every_updates < 0:
            raise ValueError("target_delta_metrics_every_updates must be >= 0")
        if self.discount >= 1:
            raise ValueError("JEPA-successor uses a normalized occupancy target and requires discount < 1")

    def dynamics_loss_coef_at(self, step: int) -> float:
        """Return the scheduled dynamics-loss coefficient at ``step``.

        ``log`` interpolates linearly in log-coefficient space; ``cosine``
        applies cosine easing directly in coefficient space.  Both preserve
        the configured endpoint values and clamp outside the schedule range.
        """
        schedule_start = self.dynamics_loss_coef_schedule_start
        schedule_end = self.dynamics_loss_coef_schedule_end
        if schedule_start is None or schedule_end is None:
            return self.dynamics_loss_coef
        (start_step, start_coef) = schedule_start
        (end_step, end_coef) = schedule_end
        if step <= start_step:
            return start_coef
        if step >= end_step:
            return end_coef
        progress = (step - start_step) / (end_step - start_step)
        if self.dynamics_loss_coef_schedule_function == "log":
            return math.exp(math.log(start_coef) + progress * (math.log(end_coef) - math.log(start_coef)))
        if self.dynamics_loss_coef_schedule_function == "cosine":
            return end_coef + (start_coef - end_coef) * 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(
            f"Unsupported dynamics_loss_coef_schedule_function: {self.dynamics_loss_coef_schedule_function}"
        )

    def lr_decay_factor_at(self, step: int) -> float:
        """Return the multiplicative factor for the configured LR schedule."""
        if self.lr_decay_function == "none" or not self.lr_decay_networks:
            return 1.0
        if step <= self.lr_decay_start_step:
            return 1.0
        progress = min(
            1.0, max(0.0, (step - self.lr_decay_start_step) / (self.lr_decay_end_step - self.lr_decay_start_step))
        )
        final_ratio = self.lr_decay_final_ratio
        if self.lr_decay_function == "linear":
            return 1.0 + progress * (final_ratio - 1.0)
        if self.lr_decay_function == "cosine":
            return final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        if self.lr_decay_function == "exponential":
            return math.exp(math.log(final_ratio) * progress)
        raise ValueError(f"Unsupported lr_decay_function: {self.lr_decay_function}")


class JepaSuccessorAgentConfig(BaseConfig):
    name: Literal["JepaSuccessorAgent"] = "JepaSuccessorAgent"
    model: JepaSuccessorModelConfig = JepaSuccessorModelConfig()
    train: JepaSuccessorAgentTrainConfig = JepaSuccessorAgentTrainConfig()
    cudagraphs: bool = False
    compile: bool = False

    def model_post_init(self, context):
        if abs(self.model.successor_discount - self.train.discount) > 1e-09:
            raise ValueError("model.successor_discount and train.discount must match")
        _validate_actor_successor_convention(self.model, self.train)
        _validate_dynamics_rollout(self.model, self.train)

    def build(self, obs_space, action_dim: int) -> "JepaSuccessorAgent":
        return JepaSuccessorAgent(obs_space, action_dim, self)

    @property
    def object_class(self):
        return JepaSuccessorAgent


class JepaSuccessorAgent(JepaAgent):
    """Goal-conditioned continuous successor agent."""

    config_class = JepaSuccessorAgentConfig

    def __init__(self, obs_space, action_dim: int, cfg: JepaSuccessorAgentConfig):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = cfg
        if cfg.train.batch_size % cfg.model.seq_length != 0:
            raise ValueError("batch_size must be divisible by seq_length")
        self._model: JepaSuccessorModel = cfg.model.build(obs_space, action_dim)
        self._model.to(self.device)
        self.setup_training()
        self.setup_compile()
        self.env_idx_with_expert_rollout = None

    def _setup_lr_decay(self) -> None:
        """Register available optimizers and preserve their configured initial LRs."""
        encoder = self._unique_optimizers(getattr(self, "encoder_optimizer", None))
        successor = self._unique_optimizers(getattr(self, "successor_optimizer", None))
        policy_state_encoder = self._unique_optimizers(getattr(self, "policy_state_encoder_optimizer", None))
        latent_reconstruction = self._unique_optimizers(getattr(self, "latent_reconstruction_optimizer", None))
        self._lr_decay_optimizers = {
            "encoder": encoder,
            "predictor": getattr(self, "predictor_optimizer", None),
            "dynamics": self._unique_optimizers(getattr(self, "dynamics_optimizer", None)),
            "successor": successor,
            "actor": self._unique_optimizers(getattr(self, "actor_optimizer", None)),
            "critic": self._unique_optimizers(
                getattr(self, "critic_optimizer", None), getattr(self, "aux_critic_optimizer", None)
            ),
            "discriminator": self._unique_optimizers(getattr(self, "discriminator_optimizer", None)),
            "policy_state_encoder": policy_state_encoder,
            "latent_reconstruction": latent_reconstruction,
        }
        self._lr_decay_optimizers = {
            name: optimizer for (name, optimizer) in self._lr_decay_optimizers.items() if optimizer
        }
        for optimizer in self._lr_decay_optimizers.values():
            if not isinstance(optimizer, (tuple, list)):
                optimizer = (optimizer,)
            for optimizer_instance in optimizer:
                for group in optimizer_instance.param_groups:
                    group.setdefault("initial_lr", group["lr"])

    def _validate_lr_decay_networks(self) -> None:
        unavailable = [name for name in self.cfg.train.lr_decay_networks if name not in self._lr_decay_optimizers]
        if unavailable:
            mode = "split"
            raise ValueError(
                f"lr_decay_networks contains unavailable network(s) {unavailable} for pretrain_mode={mode!r}"
            )

    def _update_learning_rates(self, step: int) -> None:
        cfg = self.cfg.train
        if cfg.lr_decay_function == "none" or not cfg.lr_decay_networks:
            return
        self._setup_lr_decay()
        self._validate_lr_decay_networks()
        factor = cfg.lr_decay_factor_at(step)
        for network in cfg.lr_decay_networks:
            optimizers = self._lr_decay_optimizers.get(network)
            if optimizers is None:
                continue
            if not isinstance(optimizers, (tuple, list)):
                optimizers = (optimizers,)
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    initial_lr = group.setdefault("initial_lr", group["lr"])
                    group["lr"] = initial_lr * factor

    def _mark_compiled_step_begin(self, step: int | None = None) -> None:
        """Apply the LR schedule once per agent update and preserve compile hooks."""
        if step is not None:
            self._update_learning_rates(step)
        super()._mark_compiled_step_begin()

    @staticmethod
    def _unique_optimizers(*optimizers):
        result = []
        seen = set()
        for optimizer in optimizers:
            if optimizer is not None and id(optimizer) not in seen:
                result.append(optimizer)
                seen.add(id(optimizer))
        return tuple(result)

    @property
    def optimizer_dict(self):
        optimizers = {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
        }
        if self.predictor_optimizer is not None:
            optimizers["predictor_optimizer"] = self.predictor_optimizer.state_dict()
        else:
            optimizers["dynamics_optimizer"] = self.dynamics_optimizer.state_dict()
            optimizers["successor_optimizer"] = self.successor_optimizer.state_dict()
        if self.policy_state_encoder_optimizer is not None:
            optimizers["policy_state_encoder_optimizer"] = self.policy_state_encoder_optimizer.state_dict()
        if self.latent_reconstruction_optimizer is not None:
            optimizers["latent_reconstruction_optimizer"] = self.latent_reconstruction_optimizer.state_dict()
        return optimizers

    def _dynamics_loss_coef_tensor(self, step: int) -> torch.Tensor:
        return torch.tensor(self.cfg.train.dynamics_loss_coef_at(step), device=self.device, dtype=torch.float32)

    def _active_dynamics_rollout_horizon(self) -> int:
        """Return the one-step dynamics horizon."""
        return 1

    def _prepare_dynamics_rollout_batch(self, replay_buffer):
        """Sample a trajectory-aligned batch used only by multi-step dynamics."""
        horizon = self._active_dynamics_rollout_horizon()
        if horizon == 1:
            return None
        train_buffer = replay_buffer["train"]
        try:
            batch = train_buffer.sample(self.cfg.train.batch_size, seq_length=horizon)
        except TypeError as error:
            raise TypeError(
                "dynamics_rollout_horizon > 1 requires a trajectory replay buffer whose sample method accepts seq_length"
            ) from error
        obs = _to_device(batch["observation"], self.device)
        action = batch["action"].to(self.device, non_blocking=True)
        next_obs = _to_device(batch["next"]["observation"], self.device)
        discount = self.cfg.train.discount * ~batch["next"]["terminated"].to(self.device, non_blocking=True)
        return {
            "obs": self._model._normalize(obs),
            "action": action,
            "next_obs": self._model._normalize(next_obs),
            "discount": discount,
        }

    def _paired_dynamics_step(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Roll each ensemble member through its own previous prediction."""
        dynamics = self._model._dynamics
        if dynamics is None:
            raise RuntimeError("paired dynamics rollout requires pretrain_mode='split'")
        if latent.ndim == 2:
            return dynamics(latent, action)
        if latent.shape[0] != len(dynamics.heads):
            raise ValueError("recursive latent ensemble does not match dynamics head count")
        return torch.stack(
            [head(latent[head_index], action) for (head_index, head) in enumerate(dynamics.heads)], dim=0
        )

    def _multistep_dynamics_loss(
        self,
        initial_z: torch.Tensor,
        action: torch.Tensor,
        target_z: torch.Tensor,
        target_current_z: torch.Tensor | None,
        target_codes: torch.Tensor | None,
        discount: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute a normalized recursive K-step dynamics objective.

        The flat action/target tensors contain sequence-major contiguous
        transitions.  A terminal transition itself remains a valid target;
        all later targets in that sampled sequence are masked.
        """
        flat_batch = action.shape[0]
        if flat_batch % horizon != 0:
            raise ValueError("dynamics rollout batch must be divisible by its horizon")
        sequence_count = flat_batch // horizon
        action_seq = action.reshape(sequence_count, horizon, *action.shape[1:])
        target_z_seq = target_z.reshape(sequence_count, horizon, target_z.shape[-1])
        target_current_z.reshape(
            sequence_count, horizon, target_current_z.shape[-1]
        ) if target_current_z is not None else None
        target_codes.reshape(sequence_count, horizon, target_codes.shape[-1]) if target_codes is not None else None
        nonterminal_seq = discount.reshape(flat_batch, -1)[:, 0].reshape(sequence_count, horizon) > 0
        predicted = initial_z
        alive = torch.ones(sequence_count, device=initial_z.device, dtype=torch.bool)
        weighted_loss_sum = initial_z.new_zeros((), dtype=torch.float32)
        weighted_mse_sum = initial_z.new_zeros((), dtype=torch.float32)
        weighted_accuracy_sum = initial_z.new_zeros((), dtype=torch.float32)
        total_weight = initial_z.new_zeros((), dtype=torch.float32)
        metrics: Dict[str, torch.Tensor] = {}
        for rollout_step in range(horizon):
            dynamics_output = self._paired_dynamics_step(predicted, action_seq[:, rollout_step])
            valid = alive.to(dtype=torch.float32)
            horizon_weight = float(self.cfg.train.discount**rollout_step)
            sample_weight = valid * horizon_weight
            num_heads = dynamics_output.shape[0]
            current_prediction = predicted
            next_prediction = self._model.dynamics_output_to_next_z(current_prediction, dynamics_output)
            residual = next_prediction.float() - target_z_seq[:, rollout_step].float().unsqueeze(0)
            per_sample_mse = residual.square().mean(dim=-1)
            predicted = next_prediction
            per_sample_loss = residual.square().mean(dim=-1)
            per_sample_accuracy = torch.zeros_like(per_sample_loss)
            head_weight = sample_weight.unsqueeze(0).expand(num_heads, -1)
            step_denominator = head_weight.sum().clamp_min(1.0)
            step_loss = (per_sample_loss * head_weight).sum() / step_denominator
            step_mse = (per_sample_mse * head_weight).sum() / step_denominator
            step_accuracy = (per_sample_accuracy * head_weight).sum() / step_denominator
            metrics[f"dynamics_rollout_loss_h{rollout_step + 1}"] = step_loss.detach()
            metrics[f"dynamics_rollout_mse_h{rollout_step + 1}"] = step_mse.detach()
            metrics[f"dynamics_rollout_code_accuracy_h{rollout_step + 1}"] = step_accuracy.detach()
            metrics[f"dynamics_rollout_valid_fraction_h{rollout_step + 1}"] = valid.mean().detach()
            weighted_loss_sum = weighted_loss_sum + (per_sample_loss * head_weight).sum()
            weighted_mse_sum = weighted_mse_sum + (per_sample_mse * head_weight).sum()
            weighted_accuracy_sum = weighted_accuracy_sum + (per_sample_accuracy * head_weight).sum()
            total_weight = total_weight + head_weight.sum()
            alive = alive & nonterminal_seq[:, rollout_step]
        denominator = total_weight.clamp_min(1.0)
        metrics["dynamics_rollout_horizon"] = initial_z.new_tensor(float(horizon)).detach()
        return (
            weighted_loss_sum / denominator,
            weighted_accuracy_sum / denominator,
            weighted_mse_sum / denominator,
            metrics,
        )

    @staticmethod
    def _loss_gradient_cosine(
        first_loss: torch.Tensor, second_loss: torch.Tensor, parameters: tuple[torch.nn.Parameter, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cosine and norms for two loss gradients on one parameter set."""
        first_gradients = (
            autograd.grad(first_loss, parameters, retain_graph=True, allow_unused=True)
            if first_loss.requires_grad
            else (None,) * len(parameters)
        )
        second_gradients = (
            autograd.grad(second_loss, parameters, retain_graph=True, allow_unused=True)
            if second_loss.requires_grad
            else (None,) * len(parameters)
        )
        zero = first_loss.new_zeros((), dtype=torch.float32)
        dot = zero.clone()
        first_norm_sq = zero.clone()
        second_norm_sq = zero.clone()
        for first, second in zip(first_gradients, second_gradients):
            if first is not None:
                first_norm_sq = first_norm_sq + first.detach().float().square().sum()
            if second is not None:
                second_norm_sq = second_norm_sq + second.detach().float().square().sum()
            if first is not None and second is not None:
                dot = dot + (first.detach().float() * second.detach().float()).sum()
        first_norm = first_norm_sq.sqrt()
        second_norm = second_norm_sq.sqrt()
        denominator = first_norm * second_norm
        cosine = torch.where(denominator > 0, dot / denominator.clamp_min(torch.finfo(torch.float32).eps), zero)
        return (cosine.detach(), first_norm.detach(), second_norm.detach())

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()
        common = self._adam_kwargs()
        self.encoder_optimizer = torch.optim.Adam(
            self._model._phi_encoder.parameters(), lr=self.cfg.train.lr_phi, **common
        )
        self.phi_encoder_optimizer = self.encoder_optimizer
        self.dynamics_optimizer = None
        self.successor_optimizer = None
        self.predictor_optimizer = None
        self.dynamics_optimizer = torch.optim.Adam(
            self._model._dynamics.parameters(), lr=self.cfg.train.lr_dynamics, **common
        )
        self.successor_optimizer = torch.optim.Adam(
            self._model._successor.parameters(), lr=self.cfg.train.lr_successor, **common
        )
        self.actor_optimizer = torch.optim.Adam(self._model._actor.parameters(), lr=self.cfg.train.lr_actor, **common)
        self.latent_reconstruction_optimizer = None
        if len(self._model._latent_reconstruction_heads) > 0:
            fallback_lr = self.cfg.train.lr_successor
            lr = self.cfg.train.lr_latent_reconstruction or fallback_lr
            self.latent_reconstruction_optimizer = torch.optim.Adam(
                self._model._latent_reconstruction_heads.parameters(), lr=lr, **common
            )
        self.policy_state_encoder_optimizer = None
        if self._model._policy_state_encoder is not None:
            self.policy_state_encoder_optimizer = torch.optim.Adam(
                self._model._policy_state_encoder.parameters(), lr=self.cfg.train.lr_policy_state_encoder, **common
            )
        self.keypoint_encoder_optimizer = self.policy_state_encoder_optimizer
        self._setup_lr_decay()
        self._phi_encoder_paramlist = tuple(self._model._phi_encoder.parameters())
        self._target_phi_encoder_paramlist = tuple(self._model._target_phi_encoder.parameters())
        self._predictor_paramlist = (
            tuple(self._model._predictor.parameters()) if self._model._predictor is not None else None
        )
        self._target_predictor_paramlist = (
            tuple(self._model._target_predictor.parameters()) if self._model._target_predictor is not None else None
        )
        self._successor_paramlist = (
            tuple(self._model._successor.parameters()) if self._model._successor is not None else None
        )
        self._target_successor_paramlist = (
            tuple(self._model._target_successor.parameters()) if self._model._target_successor is not None else None
        )
        batch_size = self.cfg.train.batch_size
        self.off_diag = 1 - torch.eye(batch_size, batch_size, device=self.device)
        self.off_diag_sum = self.off_diag.sum()
        self.z_buffer = ZBuffer(self.cfg.train.z_buffer_size, self.cfg.model.archi.z_dim, self.device)

    def setup_compile(self) -> None:
        print(f"{_distributed_label()} compile {self.cfg.compile}", flush=True)
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self.update_world_model = torch.compile(self.update_world_model, mode=mode)
            self.update_actor = torch.compile(self.update_actor, mode=mode)

    @staticmethod
    def continuous_successor_target(
        immediate_z: torch.Tensor,
        target_successor_output: torch.Tensor,
        discount: torch.Tensor,
        *,
        normalized_value: bool = False,
    ) -> torch.Tensor:
        """Return a discounted sum or normalized successor target."""
        batch_size = immediate_z.shape[0]
        gamma = discount.reshape(batch_size, -1)[:, 0].view(1, batch_size, 1).to(immediate_z)
        target_successor = target_successor_output.mean(dim=0, keepdim=True)
        return compose_continuous_successor_value(
            immediate_z.unsqueeze(0), target_successor, gamma, normalized_value=normalized_value
        )

    def _snap_goal_z(self, goal_z: torch.Tensor) -> torch.Tensor:
        goal_z = goal_z.to(self.device, non_blocking=True).float().detach()
        return self._model.project_z(goal_z)

    def _behavior_goal_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "z" not in batch:
            raise KeyError(
                "JEPA-successor requires replay transitions to contain the rollout actor goal under key 'z'."
            )
        return self._snap_goal_z(batch["z"])

    def _effective_random_goal_ratio(self) -> float:
        """Return the BFM random-goal mass, preserving dual-latent overrides."""
        configured = self.cfg.train.random_goal_ratio
        if configured is not None:
            return float(configured)
        return max(0.0, 1.0 - float(self.cfg.train.train_goal_ratio) - float(self.cfg.train.expert_goal_ratio))

    @torch.no_grad()
    def _sample_relabel_candidate(
        self, *, online_goal_obs, expert_z: torch.Tensor | None, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one categorical online/expert/random candidate per row."""
        online_ratio = float(self.cfg.train.train_goal_ratio)
        expert_ratio = float(self.cfg.train.expert_goal_ratio)
        random_ratio = self._effective_random_goal_ratio()
        probabilities = torch.tensor(
            [online_ratio, expert_ratio, random_ratio], dtype=torch.float32, device=self.device
        )
        if not torch.isclose(probabilities.sum(), probabilities.new_tensor(1.0), atol=1e-06):
            raise ValueError(
                f"JEPA-successor online/expert/random goal ratios must sum to one; got {online_ratio}, {expert_ratio}, {random_ratio}"
            )
        if online_ratio > 0 and online_goal_obs is None:
            raise ValueError("online goal relabeling requires next-observation goals")
        if expert_ratio > 0 and expert_z is None:
            raise ValueError("expert goal relabeling requires expert_slicer")
        if expert_z is not None and expert_z.shape[0] < batch_size:
            raise ValueError(f"expert goal batch is too small: {expert_z.shape[0]} < {batch_size}")
        random_goal = self._model.sample_z(batch_size, device=self.device).float().detach()
        online_goal = torch.zeros_like(random_goal)
        if online_ratio > 0:
            online_index = torch.randperm(batch_size, device=self.device)
            selected_online_obs = tree_map(lambda x: x.index_select(0, online_index.to(x.device)), online_goal_obs)
            online_goal = self._goal_encoder(selected_online_obs).float().detach()
        expert_goal = torch.zeros_like(random_goal)
        if expert_ratio > 0:
            expert_index = torch.randperm(expert_z.shape[0], device=self.device)[:batch_size]
            expert_goal = expert_z.index_select(0, expert_index).to(random_goal).detach()
        source = torch.multinomial(probabilities, batch_size, replacement=True)
        candidate = random_goal
        candidate = torch.where((source == 0).unsqueeze(-1), online_goal, candidate)
        candidate = torch.where((source == 1).unsqueeze(-1), expert_goal, candidate)
        return (self._snap_goal_z(candidate), source)

    @torch.no_grad()
    def _sample_g_train(
        self,
        *,
        behavior_goal: torch.Tensor,
        behavior_valid: torch.Tensor,
        online_goal_obs,
        expert_z: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply BFM-style partial relabeling to a replay-aligned goal batch."""
        batch_size = behavior_goal.shape[0]
        (candidate, source) = self._sample_relabel_candidate(
            online_goal_obs=online_goal_obs, expert_z=expert_z, batch_size=batch_size
        )
        candidate = self._snap_goal_z(candidate)
        if self.cfg.train.use_mix_rollout and hasattr(self, "z_buffer"):
            self.z_buffer.add(candidate)
        relabel_ratio = self.cfg.train.relabel_ratio
        if relabel_ratio is None or relabel_ratio <= 0:
            relabel = torch.zeros((batch_size,), device=self.device, dtype=torch.bool)
        elif relabel_ratio >= 1:
            relabel = torch.ones((batch_size,), device=self.device, dtype=torch.bool)
        else:
            relabel = torch.rand((batch_size,), device=self.device) < relabel_ratio
        behavior_goal = behavior_goal.to(self.device).detach()
        behavior_valid = behavior_valid.to(self.device).bool().reshape(batch_size)
        g_train = torch.where(relabel.unsqueeze(-1), candidate, behavior_goal).detach()
        g_train_valid = relabel | behavior_valid
        metrics = {
            "behavior_goal_valid_fraction": behavior_valid.float().mean(),
            "g_train_valid_fraction": g_train_valid.float().mean(),
            "g_train_relabel_fraction": relabel.float().mean(),
            "g_train_behavior_fraction": (~relabel).float().mean(),
            "g_train_online_fraction": (relabel & (source == 0)).float().mean(),
            "g_train_expert_fraction": (relabel & (source == 1)).float().mean(),
            "g_train_random_fraction": (relabel & (source == 2)).float().mean(),
            "successor_td_uses_g_train": behavior_goal.new_tensor(float(self.cfg.train.successor_td_use_g_train)),
        }
        return (g_train, g_train_valid, metrics)

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        (train_batch, obs, action, next_obs, discount) = self._prepare_train_batch(replay_buffer)
        dynamics_rollout = self._prepare_dynamics_rollout_batch(replay_buffer)
        (_, _, _, expert_z) = self._prepare_expert_batch(replay_buffer)
        behavior_goal_z = self._behavior_goal_from_batch(train_batch)
        self._mark_compiled_step_begin(step)
        (g_train, _, goal_metrics) = self._sample_g_train(
            behavior_goal=behavior_goal_z,
            behavior_valid=torch.ones(behavior_goal_z.shape[0], device=self.device, dtype=torch.bool),
            online_goal_obs=next_obs,
            expert_z=expert_z,
        )
        successor_goal_z = g_train if self.cfg.train.successor_td_use_g_train else behavior_goal_z
        metrics = self.update_world_model(
            obs=obs,
            action=action,
            discount=discount,
            next_obs=next_obs,
            successor_goal_z=successor_goal_z,
            dynamics_loss_coef=self._dynamics_loss_coef_tensor(step),
            dynamics_rollout=dynamics_rollout,
        )
        metrics.update(goal_metrics)
        metrics.update(self.update_policy_state_encoder(obs))
        metrics.update(self.update_actor(obs=obs, action=action, goal_z=g_train))
        self._soft_update_targets()
        self._sync_model_buffers()
        return metrics

    def update_world_model(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        successor_goal_z: torch.Tensor,
        dynamics_loss_coef: torch.Tensor | None = None,
        dynamics_rollout: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if dynamics_loss_coef is None:
            dynamics_loss_coef = self._dynamics_loss_coef_tensor(0)
        with autocast(device_type=_device_type(self.device), dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            zero = torch.zeros((), device=successor_goal_z.device, dtype=successor_goal_z.dtype)
            dynamics_rollout_metrics: Dict[str, torch.Tensor] = {}
            use_target_encoder = self._use_target_encoder_for_td()
            with torch.no_grad():
                (_, next_z, next_codes) = self._encode_side("phi", next_obs, target=use_target_encoder)
                target_current_z = None
            (pre_quant, z_t, current_codes) = self._encode_side("phi", obs)
            successor_z_t = z_t
            if not self.cfg.train.successor_backprop_to_encoder:
                successor_z_t = successor_z_t.detach()
                successor_goal_z = successor_goal_z.detach()
            td_immediate_z = next_z.detach()
            td_immediate_codes = next_codes
            if use_target_encoder:
                with torch.no_grad():
                    (_, td_immediate_z, td_immediate_codes) = self._encode_side("phi", obs, target=True)
            else:
                td_immediate_z = z_t.detach()
            with torch.no_grad():
                target_successor_output = self._model.successor(next_z.detach(), successor_goal_z, target=True)
            successor_output = self._model.successor(successor_z_t, successor_goal_z)
            rollout_horizon = 1
            dynamics_initial_z = z_t
            dynamics_action = action
            dynamics_target = next_z.detach()
            dynamics_target_current = target_current_z.detach() if target_current_z is not None else None
            dynamics_target_codes = next_codes
            dynamics_discount = discount
            if dynamics_rollout is not None:
                rollout_horizon = self._active_dynamics_rollout_horizon()
                rollout_action = dynamics_rollout["action"]
                rollout_batch = rollout_action.shape[0]
                if rollout_batch % rollout_horizon != 0:
                    raise ValueError("sampled dynamics rollout batch is not divisible by its horizon")
                sequence_count = rollout_batch // rollout_horizon
                first_rollout_obs = tree_map(
                    lambda x: x.reshape(sequence_count, rollout_horizon, *x.shape[1:])[:, 0], dynamics_rollout["obs"]
                )
                (_, dynamics_initial_z, _) = self._encode_side("phi", first_rollout_obs)
                with torch.no_grad():
                    dynamics_target_current = None
                    (_, dynamics_target, dynamics_target_codes) = self._encode_side(
                        "phi", dynamics_rollout["next_obs"], target=use_target_encoder
                    )
                dynamics_action = rollout_action
                dynamics_discount = dynamics_rollout["discount"]
            if not self.cfg.train.dynamics_backprop_to_encoder:
                dynamics_initial_z = dynamics_initial_z.detach()
            (dynamics_loss, dynamics_accuracy, dynamics_next_mse, dynamics_rollout_metrics) = (
                self._multistep_dynamics_loss(
                    dynamics_initial_z,
                    dynamics_action,
                    dynamics_target.detach(),
                    dynamics_target_current.detach() if dynamics_target_current is not None else None,
                    dynamics_target_codes,
                    dynamics_discount,
                    rollout_horizon,
                )
            )
            td_target = self.continuous_successor_target(
                td_immediate_z, target_successor_output, discount, normalized_value=True
            )
            successor_loss = (successor_output - td_target).pow(2).mean(dim=-1).mean()
            successor_goal_probability = (successor_output * successor_goal_z.unsqueeze(0)).sum(dim=-1).mean()
            orth_loss = zero
            orth_diag = zero
            orth_offdiag = zero
            if self.cfg.train.phi_ortho_coef > 0:
                (orth_loss, orth_diag, orth_offdiag) = self._orth_loss(z_t)
            (code_usage_loss, code_usage_entropy, code_dead_rate) = self._code_usage_loss(pre_quant)
            (latent_recon_loss, latent_recon_metrics) = self._latent_reconstruction_loss("phi", z_t, obs)
            dynamics_loss_total = dynamics_loss_coef * dynamics_loss
            successor_loss_total = self.cfg.train.successor_loss_coef * successor_loss
            world_model_loss = (
                dynamics_loss_total
                + successor_loss_total
                + self.cfg.train.phi_ortho_coef * orth_loss
                + self.cfg.train.code_usage_kl_coef * code_usage_loss
                + latent_recon_loss
            )
            (dynamic_successor_encoder_grad_cosine, dynamic_encoder_grad_norm, successor_encoder_grad_norm) = (
                self._loss_gradient_cosine(dynamics_loss, successor_loss, self._phi_encoder_paramlist)
            )
        prediction_optimizer = None
        optimizers = (
            [prediction_optimizer, self.encoder_optimizer]
            if prediction_optimizer is not None
            else [self.dynamics_optimizer, self.successor_optimizer, self.encoder_optimizer]
        )
        if self.latent_reconstruction_optimizer is not None:
            optimizers.append(self.latent_reconstruction_optimizer)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        world_model_loss.backward()
        clip = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None
        if prediction_optimizer is not None:
            step_specs = [
                (prediction_optimizer, self._model._predictor.parameters(), clip),
                (self.encoder_optimizer, self._model._phi_encoder.parameters(), clip),
            ]
        else:
            step_specs = [
                (self.dynamics_optimizer, self._model._dynamics.parameters(), clip),
                (self.successor_optimizer, self._model._successor.parameters(), clip),
                (self.encoder_optimizer, self._model._phi_encoder.parameters(), clip),
            ]
        if self.latent_reconstruction_optimizer is not None:
            step_specs.append(
                (self.latent_reconstruction_optimizer, self._model._latent_reconstruction_heads.parameters(), clip)
            )
        step_results = self._step_optimizers_if_finite(step_specs)
        if prediction_optimizer is not None:
            (successor_grad_norm, successor_step) = step_results[0]
            (encoder_grad_norm, encoder_step) = step_results[1]
            dynamics_grad_norm = zero.detach()
            dynamics_step = zero.new_ones(())
        else:
            (dynamics_grad_norm, dynamics_step) = step_results[0]
            (successor_grad_norm, successor_step) = step_results[1]
            (encoder_grad_norm, encoder_step) = step_results[2]
        self._model._update_z_stats(z_t.detach())
        with torch.no_grad():
            latent_float = z_t.detach().float()
            latent_std = latent_float.std(dim=0, unbiased=False)
        metrics = {
            "world_model_loss": world_model_loss.detach(),
            "dynamics_jepa_loss": dynamics_loss.detach(),
            "dynamics_loss_total": dynamics_loss_total.detach(),
            "dynamics_loss_coef_effective": dynamics_loss_coef.detach(),
            "dynamics_predict_delta": zero.new_tensor(float(False)),
            "successor_td_loss": successor_loss.detach(),
            "predictor_td_loss": successor_loss.detach(),
            "successor_loss_total": successor_loss_total.detach(),
            "dynamics_code_accuracy": dynamics_accuracy.detach(),
            "dynamics_next_mse": dynamics_next_mse.detach(),
            "successor_behavior_goal_score": successor_goal_probability.detach(),
            "encoder_norm": torch.linalg.vector_norm(z_t, dim=-1).mean().detach(),
            "dynamics_grad_norm": dynamics_grad_norm,
            "successor_grad_norm": successor_grad_norm,
            "encoder_grad_norm": encoder_grad_norm,
            "dynamic_successor_encoder_grad_cosine": dynamic_successor_encoder_grad_cosine,
            "dynamic_encoder_grad_norm_for_cosine": dynamic_encoder_grad_norm,
            "successor_encoder_grad_norm_for_cosine": successor_encoder_grad_norm,
            "dynamics_step_finite": dynamics_step,
            "successor_step_finite": successor_step,
            "predictor_grad_norm": successor_grad_norm,
            "predictor_step_finite": successor_step,
            "encoder_step_finite": encoder_step,
            "dynamics_backprop_to_encoder": latent_float.new_tensor(float(self.cfg.train.dynamics_backprop_to_encoder)),
            "successor_backprop_to_encoder": latent_float.new_tensor(
                float(self.cfg.train.successor_backprop_to_encoder)
            ),
            "dynamics_uses_target_encoder": latent_float.new_tensor(float(use_target_encoder)),
            "phi_orth_loss": orth_loss.detach(),
            "phi_orth_loss_diag": orth_diag.detach(),
            "phi_orth_loss_offdiag": orth_offdiag.detach(),
            "code_usage_kl": code_usage_loss.detach(),
            "code_usage_entropy": code_usage_entropy.detach(),
            "code_dead_rate": code_dead_rate.detach(),
            "latent_recon_loss": latent_recon_loss.detach(),
            "latent_mean_abs": latent_float.mean(dim=0).abs().mean(),
            "latent_std_mean": latent_std.mean(),
            "latent_std_min": latent_std.min(),
            "latent_std_max": latent_std.max(),
        }
        metrics.update(dynamics_rollout_metrics)
        metrics.update(latent_recon_metrics)
        return metrics

    @torch.compiler.disable
    def update_policy_state_encoder(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Supervise the deployable actor-state encoder with the JEPA latent."""
        encoder = self._model._policy_state_encoder
        optimizer = self.policy_state_encoder_optimizer
        if encoder is None or optimizer is None:
            return {}
        encoder_cfg = self.cfg.model.archi.policy_state_encoder
        encoder_input = encoder_cfg.select_input(obs)
        with torch.no_grad():
            (_, target_z, target_codes) = self._encode_side(self._active_jepa_side, obs)
        with autocast(device_type=_device_type(self.device), dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            output = encoder(encoder_input)
            predicted_z = self._model.project_z(output)
            loss = F.mse_loss(predicted_z.float(), target_z.float())
            cosine_similarity = F.cosine_similarity(predicted_z.float(), target_z.float(), dim=-1).mean()
            code_accuracy = torch.zeros((), device=loss.device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None
        (grad_norm, step_ok) = self._step_optimizer_if_finite(optimizer, encoder.parameters(), clip)
        objective_name = "policy_state_encoder_mse"
        legacy_name = "keypoint_encoder_mse"
        return {
            "policy_state_encoder_loss": loss.detach(),
            objective_name: loss.detach(),
            "policy_state_encoder_cosine_similarity": cosine_similarity.detach(),
            "policy_state_encoder_code_accuracy": code_accuracy.detach(),
            "policy_state_encoder_grad_norm": grad_norm,
            "policy_state_encoder_step_finite": step_ok,
            "keypoint_encoder_loss": loss.detach(),
            legacy_name: loss.detach(),
        }

    def update_keypoint_encoder(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compatibility alias for the renamed policy-state encoder update."""
        return self.update_policy_state_encoder(obs)

    def _actor_chain_values(
        self, z_t: torch.Tensor, goal_z: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        dynamics_output = self._model._dynamics(z_t, action)
        gamma = self.cfg.train.discount
        predicted_next = self._model.dynamics_output_to_next_z(z_t, dynamics_output)
        (num_dynamics, batch_size) = predicted_next.shape[:2]
        flat_next = predicted_next.reshape(num_dynamics * batch_size, -1)
        flat_goal = (
            goal_z.to(device=flat_next.device, dtype=flat_next.dtype)
            .unsqueeze(0)
            .expand(num_dynamics, -1, -1)
            .reshape(num_dynamics * batch_size, -1)
        )
        successor_output = self._model._successor(flat_next, flat_goal)
        successor_feature = successor_output.view(successor_output.shape[0], num_dynamics, batch_size, -1)
        composed = self._compose_continuous_actor_value(
            predicted_next.unsqueeze(0), successor_feature, gamma, current_feature=z_t.view(1, 1, batch_size, -1)
        )
        qs = (composed * goal_z.view(1, 1, batch_size, -1)).sum(dim=-1).flatten(0, 1)
        immediate_feature = predicted_next
        immediate_feature = z_t.unsqueeze(0).expand_as(predicted_next)
        metrics = {
            "Q_actor_one_step": (immediate_feature * goal_z.unsqueeze(0)).sum(dim=-1).mean().detach(),
            "Q_actor_successor_only": (successor_feature * goal_z.view(1, 1, batch_size, -1))
            .sum(dim=-1)
            .mean()
            .detach(),
            "actor_cap_dim": torch.zeros((), device=goal_z.device),
            "actor_predicted_next_disagreement": predicted_next.std(dim=0, unbiased=False).abs().mean().detach(),
        }
        (_, uncertainty, pessimistic_q) = self.get_targets_uncertainty(qs, self.cfg.train.actor_pessimism_penalty)
        metrics.update(
            {
                "Q_successor": pessimistic_q.mean().detach(),
                "Q_successor_ensemble_mean": qs.mean().detach(),
                "Q_successor_uncertainty": uncertainty.mean().detach(),
            }
        )
        return (pessimistic_q, qs, metrics)

    def _compose_continuous_actor_value(
        self,
        next_feature: torch.Tensor,
        successor_feature: torch.Tensor,
        discount: float | torch.Tensor,
        *,
        current_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compose the actor value without changing the successor TD target."""
        immediate_feature = next_feature
        if current_feature is None:
            raise ValueError("current_feature is required for current-state successor occupancy")
        immediate_feature = current_feature
        normalized_value = True
        return compose_continuous_successor_value(
            immediate_feature, successor_feature, discount, normalized_value=normalized_value
        )

    def _normalize_continuous_actor_loss(self, actor_loss: torch.Tensor) -> torch.Tensor:
        """Remove the latent dot-product dimension scale from direct actor losses."""
        return actor_loss / float(self.cfg.model.archi.z_dim)

    def update_actor(
        self, obs: torch.Tensor | dict[str, torch.Tensor], action: torch.Tensor, goal_z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=_device_type(self.device), dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            with torch.no_grad():
                (_, z_t, _) = self._encode_side("phi", obs)
            dist = self._model.actor_from_normalized_obs(obs, goal_z, self._model.cfg.actor_std)
            actor_action = dist.sample(clip=self.cfg.train.stddev_clip)
            with _freeze_module_params(*self._actor_world_modules()):
                (q, qs, chain_metrics) = self._actor_chain_values(z_t, goal_z, actor_action)
            actor_loss = -q.mean()
            bc_error = torch.zeros((), device=goal_z.device)
            if self.cfg.train.bc_coeff > 0:
                bc_error = F.mse_loss(actor_action, action)
                actor_loss = actor_loss / qs.abs().mean().detach().clamp_min(1e-06) + self.cfg.train.bc_coeff * bc_error
            else:
                actor_loss = self._normalize_continuous_actor_loss(actor_loss)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        clip = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None
        (actor_grad_norm, actor_step) = self._step_optimizer_if_finite(
            self.actor_optimizer, self._model._actor.parameters(), clip
        )
        metrics = {
            "actor_loss": actor_loss.detach(),
            "bc_error": bc_error.detach(),
            "actor_chain_grad_norm": actor_grad_norm,
            "actor_step_finite": actor_step,
        }
        metrics.update(chain_metrics)
        return metrics

    def _soft_update_targets(self) -> None:
        with torch.no_grad():
            if self._use_target_encoder_for_td():
                _soft_update_params(
                    self._phi_encoder_paramlist, self._target_phi_encoder_paramlist, self.cfg.train.encoder_target_tau
                )
            if self._predictor_paramlist is not None:
                _soft_update_params(
                    self._predictor_paramlist, self._target_predictor_paramlist, self.cfg.train.successor_target_tau
                )
            else:
                _soft_update_params(
                    self._successor_paramlist, self._target_successor_paramlist, self.cfg.train.successor_target_tau
                )

    def _target_delta_metrics(self) -> Dict[str, torch.Tensor]:
        interval = self.cfg.train.target_delta_metrics_every_updates
        if interval == 0:
            return {}
        calls = getattr(self, "_target_delta_metric_calls", 0)
        cache = getattr(self, "_target_delta_metric_cache", None)
        if cache is not None and calls % interval != 0:
            self._target_delta_metric_calls = calls + 1
            return cache
        metrics = {}
        with torch.no_grad():
            for name, (online, target) in self._target_delta_paramlists().items():
                metrics[name] = torch.stack(
                    [
                        (parameter - target_parameter).float().abs().mean()
                        for (parameter, target_parameter) in zip(online, target)
                    ]
                ).mean()
        self._target_delta_metric_cache = metrics
        self._target_delta_metric_calls = calls + 1
        return metrics

    def _target_delta_paramlists(self):
        paramlists = (
            {"ema_predictor_delta": (self._predictor_paramlist, self._target_predictor_paramlist)}
            if self._predictor_paramlist is not None
            else {"ema_successor_delta": (self._successor_paramlist, self._target_successor_paramlist)}
        )
        if self._use_target_encoder_for_td():
            paramlists["ema_phi_delta"] = (self._phi_encoder_paramlist, self._target_phi_encoder_paramlist)
        return paramlists

    def _actor_world_modules(self) -> tuple[torch.nn.Module, ...]:
        return tuple(
            (
                module
                for module in (self._model._dynamics, self._model._successor, self._model._predictor)
                if module is not None
            )
        )


class JepaSuccessorCprAgentTrainConfig(JepaSuccessorAgentTrainConfig):
    """Optimization settings for the CPR discriminator and value branch."""

    lr_discriminator: float = 1e-05
    lr_critic: float = 0.0001
    critic_target_tau: float = 0.005
    critic_pessimism_penalty: float = 0.5
    reg_coeff: float = 0.0005
    scale_reg: bool = True
    actor_gradient_projection_enabled: Literal[False] = False
    actor_gradient_projection_strength: float = 1.0
    grad_penalty_discriminator: float = 10.0
    weight_decay_discriminator: float = 0.0
    cpr_critic_use_behavior_goal: bool = False

    def model_post_init(self, context):
        super().model_post_init(context)
        for name in ("lr_discriminator", "lr_critic"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if not 0 < self.critic_target_tau <= 1:
            raise ValueError("critic_target_tau must satisfy 0 < tau <= 1")
        if not 0 <= self.actor_gradient_projection_strength <= 1:
            raise ValueError("actor_gradient_projection_strength must be in [0, 1]")
        for name in (
            "critic_pessimism_penalty",
            "reg_coeff",
            "grad_penalty_discriminator",
            "weight_decay_discriminator",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


class JepaSuccessorCprAgentConfig(BaseConfig):
    name: Literal["JepaSuccessorCprAgent"] = "JepaSuccessorCprAgent"
    model: JepaSuccessorCprModelConfig = JepaSuccessorCprModelConfig()
    train: JepaSuccessorCprAgentTrainConfig = JepaSuccessorCprAgentTrainConfig()
    cudagraphs: bool = False
    compile: bool = False

    def model_post_init(self, context):
        if abs(self.model.successor_discount - self.train.discount) > 1e-09:
            raise ValueError("model.successor_discount and train.discount must match")
        _validate_actor_successor_convention(self.model, self.train)
        _validate_dynamics_rollout(self.model, self.train)

    def build(self, obs_space, action_dim: int) -> "JepaSuccessorCprAgent":
        return JepaSuccessorCprAgent(obs_space, action_dim, self)

    @property
    def object_class(self):
        return JepaSuccessorCprAgent


class JepaSuccessorCprAgent(JepaSuccessorAgent):
    """Goal-conditioned successor agent regularized by CPR."""

    config_class = JepaSuccessorCprAgentConfig

    def __init__(self, obs_space, action_dim: int, cfg: JepaSuccessorCprAgentConfig):
        super().__init__(obs_space, action_dim, cfg)
        self._model: JepaSuccessorCprModel

    @property
    def optimizer_dict(self):
        optimizers = super().optimizer_dict
        optimizers.update(
            {
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            }
        )
        return optimizers

    def setup_training(self) -> None:
        super().setup_training()
        common = self._adam_kwargs()
        self.critic_optimizer = torch.optim.Adam(
            self._model._critic.parameters(), lr=self.cfg.train.lr_critic, **common
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self._model._discriminator.parameters(),
            lr=self.cfg.train.lr_discriminator,
            **self._adam_kwargs(weight_decay=self.cfg.train.weight_decay_discriminator),
        )
        self._critic_paramlist = tuple(self._model._critic.parameters())
        self._target_critic_paramlist = tuple(self._model._target_critic.parameters())
        self._setup_lr_decay()

    def setup_compile(self) -> None:
        super().setup_compile()
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self.update_discriminator = torch.compile(self.update_discriminator, mode=mode)
            self.update_critic = torch.compile(self.update_critic, mode=mode)

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        (train_batch, obs, action, next_obs, discount) = self._prepare_train_batch(replay_buffer)
        dynamics_rollout = self._prepare_dynamics_rollout_batch(replay_buffer)
        (_, expert_obs, _, expert_z) = self._prepare_expert_batch(replay_buffer)
        if expert_obs is None or expert_z is None:
            raise ValueError("JepaSuccessorCprAgent requires expert_slicer in replay_buffer.")
        behavior_goal_z = self._behavior_goal_from_batch(train_batch)
        self._mark_compiled_step_begin(step)
        grad_penalty = (
            self.cfg.train.grad_penalty_discriminator if self.cfg.train.grad_penalty_discriminator > 0 else None
        )
        metrics = self.update_discriminator(
            expert_obs=expert_obs, expert_z=expert_z, train_obs=obs, train_z=behavior_goal_z, grad_penalty=grad_penalty
        )
        (g_train, _, goal_metrics) = self._sample_g_train(
            behavior_goal=behavior_goal_z,
            behavior_valid=torch.ones(behavior_goal_z.shape[0], device=self.device, dtype=torch.bool),
            online_goal_obs=next_obs,
            expert_z=expert_z,
        )
        successor_goal_z = g_train if self.cfg.train.successor_td_use_g_train else behavior_goal_z
        metrics.update(
            self.update_world_model(
                obs=obs,
                action=action,
                discount=discount,
                next_obs=next_obs,
                successor_goal_z=successor_goal_z,
                dynamics_loss_coef=self._dynamics_loss_coef_tensor(step),
                dynamics_rollout=dynamics_rollout,
            )
        )
        metrics.update(goal_metrics)
        metrics.update(self.update_policy_state_encoder(obs))
        critic_goal_z = behavior_goal_z if self.cfg.train.cpr_critic_use_behavior_goal else g_train
        metrics.update(
            self.update_critic(obs=obs, action=action, discount=discount, next_obs=next_obs, z=critic_goal_z)
        )
        metrics.update(self.update_actor(obs=obs, action=action, goal_z=g_train))
        self._soft_update_targets()
        metrics.update(self._target_delta_metrics())
        self._sync_model_buffers()
        return metrics

    def _aux_reward_from_batch(
        self, train_batch: Dict[str, torch.Tensor], metrics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if "aux_rewards" not in train_batch:
            raise KeyError("JepaSuccessorCprAuxAgent requires replay transitions to contain 'aux_rewards'.")
        aux_reward = torch.zeros((self.cfg.train.batch_size, 1), device=self.device, dtype=torch.float32)
        for name in self.cfg.aux_rewards:
            if name not in train_batch["aux_rewards"]:
                raise KeyError(f"Aux reward {name!r} is missing from the replay transition.")
            value = train_batch["aux_rewards"][name]
            metrics[f"aux_rew/{name}"] = value.mean().detach()
            aux_reward.add_(
                value.to(self.device, non_blocking=True, dtype=torch.float32), alpha=self.cfg.aux_rewards_scaling[name]
            )
        return self._model._aux_reward_normalizer(aux_reward)

    @torch.compiler.disable
    def gradient_penalty_wgan(
        self,
        real_obs: torch.Tensor | dict[str, torch.Tensor],
        real_z: torch.Tensor,
        fake_obs: torch.Tensor | dict[str, torch.Tensor],
        fake_z: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = tree_get_batch_size(real_obs)
        alpha = torch.rand(batch_size, 1, device=real_z.device)
        real_discriminator_obs = self._model._discriminator.input_filter(real_obs)
        fake_discriminator_obs = self._model._discriminator.input_filter(fake_obs)
        if not torch.is_tensor(real_discriminator_obs) or not torch.is_tensor(fake_discriminator_obs):
            raise TypeError("The discriminator input filter must return a tensor for gradient penalty.")
        if real_discriminator_obs.shape != fake_discriminator_obs.shape:
            raise ValueError(
                f"Real and fake discriminator observations must match: {real_discriminator_obs.shape} vs {fake_discriminator_obs.shape}."
            )
        if real_z.shape != fake_z.shape:
            raise ValueError(f"Real and fake latent goals must match: {real_z.shape} vs {fake_z.shape}.")
        if real_discriminator_obs.shape[0] != batch_size or real_z.shape[0] != batch_size:
            raise ValueError("Observation and latent-goal batch sizes must match for gradient penalty.")
        obs_alpha = alpha.view(batch_size, *[1] * (real_discriminator_obs.ndim - 1))
        z_alpha = alpha.view(batch_size, *[1] * (real_z.ndim - 1))
        interpolated_obs = (
            obs_alpha * real_discriminator_obs + (1.0 - obs_alpha) * fake_discriminator_obs
        ).requires_grad_(True)
        interpolated_z = (z_alpha * real_z + (1.0 - z_alpha) * fake_z).requires_grad_(True)
        logits = self._model._discriminator.compute_logits(interpolated_obs, interpolated_z)
        (obs_gradient, z_gradient) = autograd.grad(
            outputs=logits,
            inputs=(interpolated_obs, interpolated_z),
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        joint_gradient = torch.cat((obs_gradient.reshape(batch_size, -1), z_gradient.reshape(batch_size, -1)), dim=1)
        return joint_gradient.norm(2, dim=1).sub(1.0).pow(2).mean()

    def update_discriminator(
        self,
        expert_obs: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        train_obs: torch.Tensor | dict[str, torch.Tensor],
        train_z: torch.Tensor,
        grad_penalty: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=_device_type(self.device), dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            expert_logits = self._model._discriminator.compute_logits(expert_obs, expert_z)
            train_logits = self._model._discriminator.compute_logits(train_obs, train_z)
            expert_loss = -F.logsigmoid(expert_logits)
            train_loss = F.softplus(train_logits)
            discriminator_loss = (expert_loss + train_loss).mean()
            gradient_penalty = None
            if grad_penalty is not None:
                gradient_penalty = self.gradient_penalty_wgan(expert_obs, expert_z, train_obs, train_z)
                discriminator_loss = discriminator_loss + grad_penalty * gradient_penalty
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        discriminator_loss.backward()
        clip = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None
        (grad_norm, step_ok) = self._step_optimizer_if_finite(
            self.discriminator_optimizer, self._model._discriminator.parameters(), clip
        )
        metrics = {
            "disc_loss": discriminator_loss.detach(),
            "disc_expert_loss": expert_loss.mean().detach(),
            "disc_train_loss": train_loss.mean().detach(),
            "discriminator_grad_norm": grad_norm,
            "discriminator_step_finite": step_ok,
        }
        if gradient_penalty is not None:
            metrics["disc_wgan_gp_loss"] = gradient_penalty.detach()
        return metrics

    def update_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=_device_type(self.device), dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            num_parallel = self.cfg.model.archi.critic.num_parallel
            with torch.no_grad():
                reward = self._model._discriminator.compute_reward(obs, z)
                next_dist = self._model.actor_from_normalized_obs(next_obs, z, self._model.cfg.actor_std)
                next_action = next_dist.sample(clip=self.cfg.train.stddev_clip)
                next_qs = self._model._target_critic(next_obs, z, next_action)
                (q_mean, q_uncertainty, next_value) = self.get_targets_uncertainty(
                    next_qs, self.cfg.train.critic_pessimism_penalty
                )
                target_q = reward + discount * next_value
                expanded_target = target_q.expand(num_parallel, -1, -1)
            qs = self._model._critic(obs, z, action)
            critic_loss = 0.5 * num_parallel * F.mse_loss(qs, expanded_target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        clip = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None
        (grad_norm, step_ok) = self._step_optimizer_if_finite(
            self.critic_optimizer, self._model._critic.parameters(), clip
        )
        return {
            "target_Q": target_q.mean().detach(),
            "Q1": qs.mean().detach(),
            "mean_next_Q": q_mean.mean().detach(),
            "unc_Q": q_uncertainty.mean().detach(),
            "critic_loss": critic_loss.detach(),
            "mean_disc_reward": reward.mean().detach(),
            "critic_grad_norm": grad_norm,
            "critic_step_finite": step_ok,
        }

    def update_aux_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        aux_reward: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=_device_type(self.device), dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            num_parallel = self.cfg.model.archi.aux_critic.num_parallel
            with torch.no_grad():
                next_dist = self._model.actor_from_normalized_obs(next_obs, z, self._model.cfg.actor_std)
                next_action = next_dist.sample(clip=self.cfg.train.stddev_clip)
                next_qs = self._model._target_aux_critic(next_obs, z, next_action)
                (q_mean, q_uncertainty, next_value) = self.get_targets_uncertainty(
                    next_qs, self.cfg.train.aux_critic_pessimism_penalty
                )
                target_q = aux_reward + discount * next_value
                expanded_target = target_q.expand(num_parallel, -1, -1)
            qs = self._model._aux_critic(obs, z, action)
            aux_critic_loss = 0.5 * num_parallel * F.mse_loss(qs, expanded_target)
        self.aux_critic_optimizer.zero_grad(set_to_none=True)
        aux_critic_loss.backward()
        clip = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None
        (grad_norm, step_ok) = self._step_optimizer_if_finite(
            self.aux_critic_optimizer, self._model._aux_critic.parameters(), clip
        )
        return {
            "target_auxQ": target_q.mean().detach(),
            "auxQ1": qs.mean().detach(),
            "mean_next_auxQ": q_mean.mean().detach(),
            "unc_auxQ": q_uncertainty.mean().detach(),
            "aux_critic_loss": aux_critic_loss.detach(),
            "mean_aux_reward": aux_reward.mean().detach(),
            "aux_critic_grad_norm": grad_norm,
            "aux_critic_step_finite": step_ok,
        }

    def _backward_actor_objectives(
        self, actor_loss: torch.Tensor, primary_loss: torch.Tensor, regularizer_losses: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Backpropagate actor objectives, optionally protecting the primary gradient."""
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        return {}

    @torch.no_grad()
    def _sync_actor_objective_gradients(self, gradients: tuple[torch.Tensor | None, ...]) -> None:
        if _distributed_enabled():
            _all_reduce_tensors(
                [gradient for gradient in gradients if gradient is not None],
                self.cfg.train.distributed_bucket_cap_mb,
                average=True,
            )

    def update_actor(
        self, obs: torch.Tensor | dict[str, torch.Tensor], action: torch.Tensor, goal_z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=_device_type(self.device), dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            with torch.no_grad():
                (_, z_t, _) = self._encode_side("phi", obs)
            actor_dist = self._model.actor_from_normalized_obs(obs, goal_z, self._model.cfg.actor_std)
            actor_action = actor_dist.sample(clip=self.cfg.train.stddev_clip)
            with _freeze_module_params(*self._actor_world_modules(), self._model._critic):
                (q_successor, qs_successor, chain_metrics) = self._actor_chain_values(z_t, goal_z, actor_action)
                critic_qs = self._model._critic(obs, goal_z, actor_action)
                (_, _, q_cpr) = self.get_targets_uncertainty(critic_qs, self.cfg.train.actor_pessimism_penalty)
            regularizer_scale = q_successor.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
            primary_loss = -q_successor.mean()
            cpr_loss = -self.cfg.train.reg_coeff * regularizer_scale * q_cpr.mean()
            bc_error = torch.zeros((), device=goal_z.device)
            if self.cfg.train.bc_coeff > 0:
                bc_error = F.mse_loss(actor_action, action)
                actor_scale = qs_successor.abs().mean().detach().clamp_min(1e-06)
                primary_loss = primary_loss / actor_scale + self.cfg.train.bc_coeff * bc_error
                cpr_loss = cpr_loss / actor_scale
            else:
                primary_loss = self._normalize_continuous_actor_loss(primary_loss)
                cpr_loss = self._normalize_continuous_actor_loss(cpr_loss)
            actor_loss = primary_loss + cpr_loss
        projection_metrics = self._backward_actor_objectives(
            actor_loss, primary_loss, {"cpr": cpr_loss} if self.cfg.train.reg_coeff > 0 else {}
        )
        clip = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None
        (grad_norm, step_ok) = self._step_optimizer_if_finite(
            self.actor_optimizer, self._model._actor.parameters(), clip
        )
        metrics = {
            "actor_loss": actor_loss.detach(),
            "bc_error": bc_error.detach(),
            "Q_discriminator": q_cpr.mean().detach(),
            "actor_chain_grad_norm": grad_norm,
            "actor_step_finite": step_ok,
            "actor_regularizer_scale": torch.as_tensor(regularizer_scale, device=goal_z.device).detach(),
        }
        metrics.update(chain_metrics)
        metrics.update(projection_metrics)
        return metrics

    def _soft_update_targets(self) -> None:
        super()._soft_update_targets()
        with torch.no_grad():
            _soft_update_params(self._critic_paramlist, self._target_critic_paramlist, self.cfg.train.critic_target_tau)

    def _target_delta_paramlists(self):
        paramlists = super()._target_delta_paramlists()
        paramlists["ema_critic_delta"] = (self._critic_paramlist, self._target_critic_paramlist)
        return paramlists


class JepaSuccessorCprAuxAgentTrainConfig(JepaSuccessorCprAgentTrainConfig):
    """Auxiliary-critic settings layered on top of successor CPR."""

    lr_aux_critic: float = 0.0001
    aux_critic_pessimism_penalty: float = 0.5
    reg_coeff_aux: float = 0.0025
    aux_critic_use_behavior_goal: bool = False

    def model_post_init(self, context):
        super().model_post_init(context)
        if self.lr_aux_critic <= 0:
            raise ValueError("lr_aux_critic must be > 0")
        if self.aux_critic_pessimism_penalty < 0:
            raise ValueError("aux_critic_pessimism_penalty must be >= 0")
        if self.reg_coeff_aux < 0:
            raise ValueError("reg_coeff_aux must be >= 0")


class JepaSuccessorCprAuxAgentConfig(JepaSuccessorCprAgentConfig):
    name: Literal["JepaSuccessorCprAuxAgent"] = "JepaSuccessorCprAuxAgent"
    model: JepaSuccessorCprAuxModelConfig = JepaSuccessorCprAuxModelConfig()
    train: JepaSuccessorCprAuxAgentTrainConfig = JepaSuccessorCprAuxAgentTrainConfig()
    aux_rewards: list[str] = pydantic.Field(default_factory=list)
    aux_rewards_scaling: dict[str, float] = pydantic.Field(default_factory=dict)

    def model_post_init(self, context):
        super().model_post_init(context)
        missing = sorted(set(self.aux_rewards) - set(self.aux_rewards_scaling))
        if missing:
            raise ValueError(f"Missing aux reward scaling for: {', '.join(missing)}")

    def build(self, obs_space, action_dim: int) -> "JepaSuccessorCprAuxAgent":
        return JepaSuccessorCprAuxAgent(obs_space, action_dim, self)

    @property
    def object_class(self):
        return JepaSuccessorCprAuxAgent


class JepaSuccessorCprAuxAgent(JepaSuccessorCprAgent):
    """Goal-conditioned successor agent regularized by CPR and Aux critics."""

    config_class = JepaSuccessorCprAuxAgentConfig

    def __init__(self, obs_space, action_dim: int, cfg: JepaSuccessorCprAuxAgentConfig):
        super().__init__(obs_space, action_dim, cfg)
        self._model: JepaSuccessorCprAuxModel

    @property
    def optimizer_dict(self):
        optimizers = super().optimizer_dict
        optimizers["aux_critic_optimizer"] = self.aux_critic_optimizer.state_dict()
        return optimizers

    def setup_training(self) -> None:
        super().setup_training()
        self.aux_critic_optimizer = torch.optim.Adam(
            self._model._aux_critic.parameters(), lr=self.cfg.train.lr_aux_critic, **self._adam_kwargs()
        )
        self._aux_critic_paramlist = tuple(self._model._aux_critic.parameters())
        self._target_aux_critic_paramlist = tuple(self._model._target_aux_critic.parameters())
        self._setup_lr_decay()

    def setup_compile(self) -> None:
        super().setup_compile()
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self.update_aux_critic = torch.compile(self.update_aux_critic, mode=mode)

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        (train_batch, obs, action, next_obs, discount) = self._prepare_train_batch(replay_buffer)
        dynamics_rollout = self._prepare_dynamics_rollout_batch(replay_buffer)
        (_, expert_obs, _, expert_z) = self._prepare_expert_batch(replay_buffer)
        if expert_obs is None or expert_z is None:
            raise ValueError("JepaSuccessorCprAuxAgent requires expert_slicer in replay_buffer.")
        behavior_goal_z = self._behavior_goal_from_batch(train_batch)
        self._mark_compiled_step_begin(step)
        grad_penalty = (
            self.cfg.train.grad_penalty_discriminator if self.cfg.train.grad_penalty_discriminator > 0 else None
        )
        metrics = self.update_discriminator(
            expert_obs=expert_obs, expert_z=expert_z, train_obs=obs, train_z=behavior_goal_z, grad_penalty=grad_penalty
        )
        (g_train, _, goal_metrics) = self._sample_g_train(
            behavior_goal=behavior_goal_z,
            behavior_valid=torch.ones(behavior_goal_z.shape[0], device=self.device, dtype=torch.bool),
            online_goal_obs=next_obs,
            expert_z=expert_z,
        )
        successor_goal_z = g_train if self.cfg.train.successor_td_use_g_train else behavior_goal_z
        metrics.update(
            self.update_world_model(
                obs=obs,
                action=action,
                discount=discount,
                next_obs=next_obs,
                successor_goal_z=successor_goal_z,
                dynamics_loss_coef=self._dynamics_loss_coef_tensor(step),
                dynamics_rollout=dynamics_rollout,
            )
        )
        metrics.update(goal_metrics)
        metrics.update(self.update_policy_state_encoder(obs))
        critic_goal_z = behavior_goal_z if self.cfg.train.cpr_critic_use_behavior_goal else g_train
        metrics.update(
            self.update_critic(obs=obs, action=action, discount=discount, next_obs=next_obs, z=critic_goal_z)
        )
        aux_reward = self._aux_reward_from_batch(train_batch, metrics)
        metrics.update(
            self.update_aux_critic(
                obs=obs,
                action=action,
                discount=discount,
                aux_reward=aux_reward,
                next_obs=next_obs,
                z=behavior_goal_z if self.cfg.train.aux_critic_use_behavior_goal else g_train,
            )
        )
        metrics.update(self.update_actor(obs=obs, action=action, goal_z=g_train))
        self._soft_update_targets()
        metrics.update(self._target_delta_metrics())
        self._sync_model_buffers()
        return metrics

    def update_actor(
        self, obs: torch.Tensor | dict[str, torch.Tensor], action: torch.Tensor, goal_z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=_device_type(self.device), dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            with torch.no_grad():
                (_, z_t, _) = self._encode_side("phi", obs)
            actor_dist = self._model.actor_from_normalized_obs(obs, goal_z, self._model.cfg.actor_std)
            actor_action = actor_dist.sample(clip=self.cfg.train.stddev_clip)
            with _freeze_module_params(*self._actor_world_modules(), self._model._critic, self._model._aux_critic):
                (q_successor, qs_successor, chain_metrics) = self._actor_chain_values(z_t, goal_z, actor_action)
                critic_qs = self._model._critic(obs, goal_z, actor_action)
                (_, _, q_cpr) = self.get_targets_uncertainty(critic_qs, self.cfg.train.actor_pessimism_penalty)
                aux_qs = self._model._aux_critic(obs, goal_z, actor_action)
                (_, _, q_aux) = self.get_targets_uncertainty(aux_qs, self.cfg.train.actor_pessimism_penalty)
            regularizer_scale = q_successor.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
            primary_loss = -q_successor.mean()
            cpr_loss = -self.cfg.train.reg_coeff * regularizer_scale * q_cpr.mean()
            aux_loss = -self.cfg.train.reg_coeff_aux * regularizer_scale * q_aux.mean()
            bc_error = torch.zeros((), device=goal_z.device)
            if self.cfg.train.bc_coeff > 0:
                bc_error = F.mse_loss(actor_action, action)
                actor_scale = qs_successor.abs().mean().detach().clamp_min(1e-06)
                primary_loss = primary_loss / actor_scale + self.cfg.train.bc_coeff * bc_error
                cpr_loss = cpr_loss / actor_scale
                aux_loss = aux_loss / actor_scale
            else:
                primary_loss = self._normalize_continuous_actor_loss(primary_loss)
                cpr_loss = self._normalize_continuous_actor_loss(cpr_loss)
                aux_loss = self._normalize_continuous_actor_loss(aux_loss)
            actor_loss = primary_loss + cpr_loss + aux_loss
        regularizer_losses = {}
        if self.cfg.train.reg_coeff > 0:
            regularizer_losses["cpr"] = cpr_loss
        if self.cfg.train.reg_coeff_aux > 0:
            regularizer_losses["aux"] = aux_loss
        projection_metrics = self._backward_actor_objectives(actor_loss, primary_loss, regularizer_losses)
        clip = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None
        (grad_norm, step_ok) = self._step_optimizer_if_finite(
            self.actor_optimizer, self._model._actor.parameters(), clip
        )
        metrics = {
            "actor_loss": actor_loss.detach(),
            "bc_error": bc_error.detach(),
            "Q_discriminator": q_cpr.mean().detach(),
            "Q_aux": q_aux.mean().detach(),
            "actor_chain_grad_norm": grad_norm,
            "actor_step_finite": step_ok,
            "actor_regularizer_scale": torch.as_tensor(regularizer_scale, device=goal_z.device).detach(),
        }
        metrics.update(chain_metrics)
        metrics.update(projection_metrics)
        return metrics

    def _soft_update_targets(self) -> None:
        super()._soft_update_targets()
        with torch.no_grad():
            _soft_update_params(
                self._aux_critic_paramlist, self._target_aux_critic_paramlist, self.cfg.train.critic_target_tau
            )

    def _target_delta_paramlists(self):
        paramlists = super()._target_delta_paramlists()
        paramlists["ema_aux_critic_delta"] = (self._aux_critic_paramlist, self._target_aux_critic_paramlist)
        return paramlists
