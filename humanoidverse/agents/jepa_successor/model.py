"""Continuous goal-conditioned dynamics, successor, and policy networks."""

import copy
import math
import typing as tp
from typing import Literal

import pydantic
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast

from ..base import BaseConfig
from ..base_model import BaseModel
from ..encoder_observation import (
    CLEAN_ENCODER_OBS_KEY,
    normalize_observation_with_clean_encoder_obs,
    select_encoder_observation,
)
from ..jepa.model import (
    KeypointEncoder,
    KeypointEncoderConfig,
    LatentReconstructionHeadConfig,
    aggregate_continuous_latent_window,
    temporal_aggregation_weights,
)
from ..nn_models import (
    ActorArchiConfig,
    BackwardArchiConfig,
    DiscriminatorArchiConfig,
    ForwardArchiConfig,
    RewardNormalizerConfig,
    eval_mode,
)
from ..normalizers import ObsNormalizerConfig
from ..root_pose_goal import ROOT_POSE_SCALE


def _device_type(device: str | torch.device) -> str:
    return torch.device(device).type


def compose_continuous_successor_value(
    next_feature: torch.Tensor,
    successor_feature: torch.Tensor,
    discount: float | torch.Tensor,
    *,
    normalized_value: bool,
) -> torch.Tensor:
    """Compose one-step and bootstrapped continuous successor features.

    The normalized form weights the immediate feature by ``1 - discount``;
    the unnormalized form uses unit weight.
    """
    immediate_weight = 1.0 - discount if normalized_value else 1.0
    return immediate_weight * next_feature + discount * successor_feature


class _LatentResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Mish())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _LatentMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, hidden_layers: int, residual: bool):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish()]
        if residual:
            layers.extend((_LatentResidualBlock(hidden_dim) for _ in range(hidden_layers)))
        else:
            for _ in range(hidden_layers):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish()])
        layers.extend([nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim)])
        self.net = nn.Sequential(*layers)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat(inputs, dim=-1))


class LatentPredictorEnsemble(nn.Module):
    """Independent MLP ensemble with a stable ``ensemble, ..., feature`` output."""

    def __init__(self, input_dim: int, output_dim: int, cfg: "LatentPredictorConfig"):
        super().__init__()
        self.num_parallel = cfg.num_parallel
        self.heads = nn.ModuleList(
            [
                _LatentMLP(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=cfg.hidden_dim,
                    hidden_layers=cfg.hidden_layers,
                    residual=cfg.model == "residual",
                )
                for _ in range(cfg.num_parallel)
            ]
        )

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(*inputs) for head in self.heads], dim=0)


class LatentPredictorConfig(BaseConfig):
    """Architecture used by both the one-step and successor predictors."""

    hidden_dim: int = 1024
    hidden_layers: int = 4
    model: tp.Literal["simple", "residual"] = "residual"
    num_parallel: int = 2

    def model_post_init(self, context):
        if self.hidden_dim <= 0:
            raise ValueError("latent predictor hidden_dim must be > 0")
        if self.hidden_layers < 0:
            raise ValueError("latent predictor hidden_layers must be >= 0")
        if self.num_parallel <= 0:
            raise ValueError("latent predictor num_parallel must be > 0")

    def build(self, input_dim: int, output_dim: int) -> LatentPredictorEnsemble:
        return LatentPredictorEnsemble(input_dim, output_dim, self)


class PolicyStateEncoder(KeypointEncoder):
    """Actor-side encoder for deployable observations."""


class PolicyStateEncoderConfig(KeypointEncoderConfig):
    """Deployable observation encoder supervised by the shared state latent."""

    name: tp.Literal["PolicyStateEncoderConfig", "KeypointEncoderConfig"] = "PolicyStateEncoderConfig"

    def build(self, obs_space, z_dim: int, num_levels: int, *, actor_input_filter=None) -> PolicyStateEncoder:
        encoder = super().build(obs_space, z_dim, num_levels, actor_input_filter=actor_input_filter)
        return PolicyStateEncoder(backbone=encoder.backbone, z_dim=encoder.z_dim, num_levels=encoder.num_levels)


class JepaSuccessorModelArchiConfig(BaseConfig):
    z_dim: int = 256
    norm_z: bool = True
    fsq_num_levels: Literal[0] = 0
    fsq_input_scale: float = 1.0
    actor_input_mode: Literal["policy_latent"] = "policy_latent"
    actor_direct_action_history_length: int = 0
    latent_reconstruction_body_names: list[str] = pydantic.Field(default_factory=list)
    latent_reconstruction: dict[str, LatentReconstructionHeadConfig] = pydantic.Field(default_factory=dict)
    policy_state_encoder: PolicyStateEncoderConfig | KeypointEncoderConfig = pydantic.Field(
        default=PolicyStateEncoderConfig(enabled=True, use_actor_obs=True),
        validation_alias=pydantic.AliasChoices("policy_state_encoder", "keypoint_encoder"),
    )
    pretrain_mode: Literal["split"] = "split"
    encoder: BackwardArchiConfig = BackwardArchiConfig()
    dynamics: LatentPredictorConfig = LatentPredictorConfig()
    successor: LatentPredictorConfig = LatentPredictorConfig()
    predictor: LatentPredictorConfig = LatentPredictorConfig()
    actor: ActorArchiConfig = ActorArchiConfig()

    @pydantic.model_validator(mode="before")
    @classmethod
    def _migrate_legacy_actor_input_mode(cls, data):
        """Restore the actor input mode recorded by older checkpoint formats."""
        if (
            isinstance(data, dict)
            and "actor_input_mode" not in data
            and ("keypoint_encoder" in data)
            and ("policy_state_encoder" not in data)
        ):
            data = dict(data)
            data["actor_input_mode"] = "raw_obs"
        return data

    def model_post_init(self, context):
        if self.z_dim <= 0:
            raise ValueError("z_dim must be > 0")
        if self.fsq_input_scale <= 0:
            raise ValueError("fsq_input_scale must be > 0")
        if self.actor_direct_action_history_length < 0:
            raise ValueError("actor_direct_action_history_length must be non-negative")
        for name in self.latent_reconstruction:
            if not name or "." in name:
                raise ValueError("latent reconstruction item names must be non-empty and must not contain '.'")
        if not self.policy_state_encoder.enabled:
            raise ValueError("actor_input_mode='policy_latent' requires policy_state_encoder.enabled=True.")
        if not self.policy_state_encoder.use_actor_obs:
            raise ValueError(
                "actor_input_mode='policy_latent' requires the policy-state encoder to use deployable actor observations (use_actor_obs=True)."
            )

    @property
    def keypoint_encoder(self) -> PolicyStateEncoderConfig | KeypointEncoderConfig:
        """Compatibility alias for pre-policy-state-encoder callers."""
        return self.policy_state_encoder

    @property
    def fsq_enabled(self) -> bool:
        return False

    @property
    def predictor_output_dim(self) -> int:
        return self.z_dim


class JepaSuccessorModelConfig(BaseConfig):
    name: tp.Literal["JepaSuccessorModel"] = "JepaSuccessorModel"
    device: tp.Literal["cpu", "cuda"] = "cuda"
    archi: JepaSuccessorModelArchiConfig = JepaSuccessorModelArchiConfig()
    obs_normalizer: ObsNormalizerConfig = ObsNormalizerConfig()
    inference_batch_size: int = 500000
    seq_length: int = 1
    seq_aggregation_discounted: bool = False
    actor_std: float = 0.05
    successor_discount: float = 0.98
    successor_normalized_value_loss: Literal[True] = True
    successor_occupancy_start: Literal["current_state"] = "current_state"
    dynamics_predict_delta: Literal[False] = False
    amp: bool = False
    use_clean_encoder_obs: bool = False
    clean_encoder_obs_key: str = CLEAN_ENCODER_OBS_KEY
    privileged_state_world_frame: bool = False

    def model_post_init(self, context):
        if self.seq_length <= 0:
            raise ValueError("seq_length must be > 0")
        if self.actor_std <= 0:
            raise ValueError("actor_std must be > 0")
        if not 0 < self.successor_discount < 1:
            raise ValueError("successor_discount must satisfy 0 < discount < 1")

    def build(self, obs_space, action_dim: int) -> "JepaSuccessorModel":
        return JepaSuccessorModel(obs_space, action_dim, self)


class JepaSuccessorModel(BaseModel):
    config_class = JepaSuccessorModelConfig

    def __init__(self, obs_space, action_dim: int, cfg: JepaSuccessorModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        self.cfg: JepaSuccessorModelConfig = cfg
        self.device = cfg.device
        self.amp_dtype = torch.bfloat16
        arch = cfg.archi
        self._fsq = None
        self._jepa_sides = ("phi",)
        self._phi_encoder = arch.encoder.build(obs_space, arch.z_dim)
        self._dynamics: LatentPredictorEnsemble | None = None
        self._successor: LatentPredictorEnsemble | None = None
        self._predictor: LatentPredictorEnsemble | None = None
        self._dynamics = arch.dynamics.build(arch.z_dim + action_dim, arch.predictor_output_dim)
        self._successor = arch.successor.build(2 * arch.z_dim, arch.predictor_output_dim)
        self._actor_direct_action_history_length = arch.actor_direct_action_history_length
        self._actor_direct_action_context_dim = (
            action_dim * (self._actor_direct_action_history_length + 1)
            if self._actor_direct_action_history_length > 0
            else 0
        )
        if self._actor_direct_action_history_length > 0:
            self._validate_actor_direct_action_observation(obs_space, action_dim)
        self._actor = arch.actor.build_from_latent(
            arch.z_dim, action_dim, state_extra_dim=self._actor_direct_action_context_dim
        )
        self._latent_reconstruction_heads = nn.ModuleDict(
            {name: head_cfg.build(arch.z_dim) for (name, head_cfg) in arch.latent_reconstruction.items()}
        )
        self._obs_normalizer = cfg.obs_normalizer.build(obs_space)
        self._policy_state_encoder = (
            arch.policy_state_encoder.build(obs_space, arch.z_dim, 0, actor_input_filter=arch.actor.input_filter)
            if arch.policy_state_encoder.enabled
            else None
        )
        self.register_buffer("_z_cov", torch.eye(arch.z_dim))
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    def _validate_actor_direct_action_observation(self, obs_space, action_dim: int) -> None:
        """Validate the BFM-style flattened action-history contract."""
        required = ("last_action", "history_actor")
        spaces = getattr(obs_space, "spaces", {})
        missing = [key for key in required if key not in spaces]
        if missing:
            raise ValueError(
                f"actor_direct_action_history_length requires dictionary observations with {list(required)}; missing {missing}"
            )
        if spaces["last_action"].shape != (action_dim,):
            raise ValueError(
                f"last_action dimension must equal action_dim when direct actor action history is enabled: last_action={spaces['last_action'].shape}, action_dim={action_dim}"
            )
        required_history_dim = self._actor_direct_action_history_length * action_dim
        if spaces["history_actor"].shape[0] < required_history_dim:
            raise ValueError(
                f"history_actor is too short for actor_direct_action_history_length: need at least {required_history_dim}, got {spaces['history_actor'].shape[0]}"
            )
        configured_keys = getattr(self.cfg.archi.actor.input_filter, "key", None)
        if isinstance(configured_keys, str):
            configured_keys = (configured_keys,)
        if configured_keys is not None:
            missing_filter_keys = [key for key in required if key not in configured_keys]
            if missing_filter_keys:
                raise ValueError(
                    f"actor_direct_action_history_length reuses normalized actor observations; actor.input_filter must include {list(required)}, missing {missing_filter_keys}"
                )

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "_target_phi_encoder"):
            self._target_phi_encoder.train(False)
        if getattr(self, "_target_successor", None) is not None:
            self._target_successor.train(False)
        if getattr(self, "_target_predictor", None) is not None:
            self._target_predictor.train(False)
        return self

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)
        if hasattr(self, "_target_phi_encoder"):
            self._target_phi_encoder.requires_grad_(False)
        if getattr(self, "_target_successor", None) is not None:
            self._target_successor.requires_grad_(False)
        if getattr(self, "_target_predictor", None) is not None:
            self._target_predictor.requires_grad_(False)
        return self

    def _prepare_for_train(self) -> None:
        self._target_phi_encoder = copy.deepcopy(self._phi_encoder)
        self._target_successor = copy.deepcopy(self._successor) if self._successor is not None else None
        self._target_predictor = copy.deepcopy(self._predictor) if self._predictor is not None else None
        self._target_phi_encoder.requires_grad_(False)
        self._target_phi_encoder.train(False)
        for module in (self._target_successor, self._target_predictor):
            if module is not None:
                module.requires_grad_(False)
                module.train(False)

    @property
    def _keypoint_encoder(self):
        """Compatibility alias; use ``_policy_state_encoder`` in new code."""
        return self._policy_state_encoder

    @property
    def actor_observation_filter(self):
        """Filter for the real/deployable observation in front of the actor."""
        if self._policy_state_encoder is None:
            raise RuntimeError("policy-state encoder is disabled in policy-latent actor mode")
        return self._policy_state_encoder.backbone.input_filter

    @property
    def actor_observation_dim(self) -> int:
        return self.actor_observation_filter.output_space.shape[0]

    @property
    def actor_direct_action_history_length(self) -> int:
        """Number of older ``history_actor`` action frames bypassed to the actor."""
        return self._actor_direct_action_history_length

    @property
    def actor_direct_action_context_dim(self) -> int:
        """Direct actor side-channel size, including the newest ``last_action``."""
        return self._actor_direct_action_context_dim

    @property
    def goal_side(self) -> tp.Literal["phi"]:
        return "phi"

    @property
    def policy_side(self) -> tp.Literal["phi"]:
        return "phi"

    def _encoder_for_side(self, side: str):
        if side != "phi":
            raise RuntimeError(f"JEPA-successor uses one shared encoder; requested side={side!r}")
        return self._phi_encoder

    def _target_encoder_for_side(self, side: str):
        if side != "phi":
            raise RuntimeError(f"JEPA-successor uses one shared target encoder; requested side={side!r}")
        return self._target_phi_encoder

    def _normalize(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return normalize_observation_with_clean_encoder_obs(
                obs, self._obs_normalizer, clean_encoder_obs_key=self.cfg.clean_encoder_obs_key
            )

    def _encoder_obs(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        return select_encoder_observation(
            obs,
            use_clean_encoder_obs=self.cfg.use_clean_encoder_obs,
            clean_encoder_obs_key=self.cfg.clean_encoder_obs_key,
        )

    @torch.no_grad()
    def _update_z_stats(self, z: torch.Tensor) -> None:
        z = z.float()
        self._z_cov.mul_(0.995).add_(0.005 * torch.matmul(z.T, z) / z.shape[0])

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        if self.cfg.archi.norm_z:
            return math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def aggregate_latent_window(
        self, z: torch.Tensor, *, dim: int, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Aggregate one future-state window and remove its sequence axis."""
        dim = dim % z.ndim
        if z.shape[dim] == 1:
            return z.select(dim, 0)
        aggregate = aggregate_continuous_latent_window(
            z,
            dim=dim,
            discounted=self.cfg.seq_aggregation_discounted,
            discount=self.cfg.successor_discount,
            weights=weights,
        )
        return self.project_z(aggregate)

    def aggregate_latent_sequence(self, z: torch.Tensor, *, dim: int) -> torch.Tensor:
        """Replace every state by its aggregate over the next seq_length states."""
        dim = dim % z.ndim
        if self.cfg.seq_length == 1:
            return z
        outputs = []
        weight_cache = {}
        for step in range(z.shape[dim]):
            window_length = min(self.cfg.seq_length, z.shape[dim] - step)
            window = z.narrow(dim, step, window_length)
            if window_length not in weight_cache:
                weight_cache[window_length] = temporal_aggregation_weights(
                    window_length,
                    discounted=self.cfg.seq_aggregation_discounted,
                    discount=self.cfg.successor_discount,
                    device=z.device,
                )
            outputs.append(self.aggregate_latent_window(window, dim=dim, weights=weight_cache[window_length]))
        return torch.stack(outputs, dim=dim)

    @property
    def fsq_enabled(self) -> bool:
        return self._fsq is not None

    @property
    def fsq_num_levels(self) -> int:
        return 0

    def quantize_latent(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._fsq is None:
            return (self.project_z(z), None)
        return self._fsq(z)

    def predictor_expected_z(self, output: torch.Tensor) -> torch.Tensor:
        return output

    def dynamics(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self._dynamics is None:
            raise RuntimeError("dynamics exists only when pretrain_mode='split'")
        return self._dynamics(z, action)

    def dynamics_output_to_next_z(self, current_z: torch.Tensor, dynamics_output: torch.Tensor) -> torch.Tensor:
        """Project the predicted next state to the normalized latent space."""
        next_pre_quant = dynamics_output
        return self.project_z(next_pre_quant)

    def dynamics_next_z(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Run split dynamics and return the decoded absolute next latent."""
        return self.dynamics_output_to_next_z(z, self.dynamics(z, action))

    def successor(self, z: torch.Tensor, goal_z: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        module = self._target_successor if target else self._successor
        if module is None:
            raise RuntimeError("successor exists only when pretrain_mode='split'")
        return module(z, goal_z)

    def compose_continuous_successor_value(
        self, next_feature: torch.Tensor, successor_feature: torch.Tensor, discount: float | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply the checkpoint-serialized continuous successor convention."""
        return compose_continuous_successor_value(
            next_feature,
            successor_feature,
            self.cfg.successor_discount if discount is None else discount,
            normalized_value=True,
        )

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        with autocast(device_type=_device_type(self.device), dtype=self.amp_dtype, enabled=self.cfg.amp):
            normalized = self._normalize(obs)
            return self.project_z(self._phi_encoder(self._encoder_obs(normalized)))

    @torch.no_grad()
    def phi(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return self.backward_map(obs)

    @torch.no_grad()
    def psi(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return self.backward_map(obs)

    @torch.no_grad()
    def forward_map(self, obs: torch.Tensor | dict[str, torch.Tensor], goal_z: torch.Tensor, action: torch.Tensor):
        """Return one-step Bellman-composed successor features for diagnostics.

        Existing inference tools expect an ensemble of latent features and
        compute a dot product with ``goal_z``. Categorical training uses the
        corresponding probabilities directly in the agent; this decoded
        feature view keeps those tools operational.
        """
        with autocast(device_type=_device_type(self.device), dtype=self.amp_dtype, enabled=self.cfg.amp):
            normalized = self._normalize(obs)
            z_t = self.project_z(self._phi_encoder(self._encoder_obs(normalized)))
            goal_z = self.project_z(goal_z.to(device=z_t.device, dtype=z_t.dtype))
            dynamics_output = self.dynamics(z_t, action)
            predicted_next = self.dynamics_output_to_next_z(z_t, dynamics_output)
            (num_dynamics, batch_size) = predicted_next.shape[:2]
            goal_z = goal_z.to(device=predicted_next.device, dtype=predicted_next.dtype)
            flat_next = predicted_next.reshape(num_dynamics * batch_size, -1)
            flat_goal = goal_z.unsqueeze(0).expand(num_dynamics, -1, -1).reshape(num_dynamics * batch_size, -1)
            successor_output = self.successor(flat_next, flat_goal)
            successor_feature = self.predictor_expected_z(successor_output).view(
                successor_output.shape[0], num_dynamics, batch_size, -1
            )
            gamma = self.cfg.successor_discount
            immediate_feature = predicted_next.unsqueeze(0)
            immediate_feature = z_t.view(1, 1, batch_size, -1)
            composed = self.compose_continuous_successor_value(immediate_feature, successor_feature, gamma)
            return composed.flatten(0, 1)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, std: float):
        with autocast(device_type=_device_type(self.device), dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self.actor_from_normalized_obs(self._normalize(obs), z, std)

    def act(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        dist = self.actor(obs, z, self.cfg.actor_std)
        return (dist.mean if mean else dist.sample()).float()

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        return self.project_z(torch.randn((size, self.cfg.archi.z_dim), device=device))

    @torch.no_grad()
    def policy_state_z(
        self, obs: torch.Tensor | dict[str, torch.Tensor], *, obs_is_normalized: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode deployable observations into the actor's current-state latent."""
        if self._policy_state_encoder is None:
            raise RuntimeError("policy-state encoder is disabled")
        normalized_obs = obs if obs_is_normalized else self._normalize(obs)
        encoder_cfg = self.cfg.archi.policy_state_encoder
        output = self._policy_state_encoder(encoder_cfg.select_input(normalized_obs))
        return (self.project_z(output), None)

    def actor_from_normalized_obs(self, obs: torch.Tensor | dict[str, torch.Tensor], goal_z: torch.Tensor, std: float):
        """Differentiable actor path for observations already normalized by replay."""
        (policy_z, _) = self.policy_state_z(obs, obs_is_normalized=True)
        return self.actor_from_policy_state_z(policy_z, goal_z, std, normalized_obs=obs)

    def actor_direct_action_context(self, normalized_obs: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Extract ``[last_action, older actions...]`` from normalized obs.

        HumanoidVerse constructs ``history_actor`` by sorted auxiliary key, so
        the standard BFM observation places its four flattened action frames at
        the beginning.  Keeping the extraction here gives training, inference,
        ONNX export, and MPC one checkpoint-persisted contract.
        """
        if self._actor_direct_action_history_length == 0:
            return None
        if not isinstance(normalized_obs, dict):
            raise TypeError("direct actor action history requires dictionary observations")
        missing = [key for key in ("last_action", "history_actor") if key not in normalized_obs]
        if missing:
            raise KeyError(f"direct actor action history is missing normalized observation keys {missing}")
        last_action = normalized_obs["last_action"]
        history_dim = self._actor_direct_action_history_length * self.action_dim
        history_actions = normalized_obs["history_actor"][..., :history_dim]
        if last_action.shape[-1] != self.action_dim or history_actions.shape[-1] != history_dim:
            raise ValueError(
                f"invalid direct actor action-history dimensions: last_action={last_action.shape[-1]}, history={history_actions.shape[-1]}, expected=({self.action_dim}, {history_dim})"
            )
        return torch.cat([last_action, history_actions], dim=-1)

    def actor_from_policy_state_z(
        self,
        policy_z: torch.Tensor,
        goal_z: torch.Tensor,
        std: float,
        *,
        normalized_obs: dict[str, torch.Tensor] | None = None,
        action_context: torch.Tensor | None = None,
    ):
        """Run the policy from an explicit state latent and action context.

        ``normalized_obs`` is the normal training/inference path.  MPC may pass
        a candidate-specific ``action_context`` while rolling predicted states.
        """
        if self._actor_direct_action_history_length > 0:
            if action_context is not None and normalized_obs is not None:
                raise ValueError("pass either normalized_obs or action_context, not both")
            if action_context is None:
                if normalized_obs is None:
                    raise ValueError("direct actor action history requires normalized_obs or action_context")
                action_context = self.actor_direct_action_context(normalized_obs)
            if action_context is None:
                raise RuntimeError("direct actor action context was unexpectedly empty")
            if action_context.shape[:-1] != policy_z.shape[:-1]:
                raise ValueError(
                    f"policy_z and action_context leading dimensions must match: {policy_z.shape[:-1]} vs {action_context.shape[:-1]}"
                )
            if action_context.shape[-1] != self._actor_direct_action_context_dim:
                raise ValueError(
                    f"action_context has dim {action_context.shape[-1]}, expected {self._actor_direct_action_context_dim}"
                )
            actor_state = torch.cat([policy_z, action_context.to(policy_z)], dim=-1)
        else:
            if action_context is not None:
                raise ValueError("action_context was provided but direct actor action history is disabled")
            actor_state = policy_z
        return self._actor(actor_state, goal_z, std)

    @torch.no_grad()
    def keypoint_z(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compatibility alias for the former keypoint-encoder API."""
        return self.policy_state_z(obs)

    @torch.no_grad()
    def goal_root_xy_distance(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        goal_z: torch.Tensor,
        *,
        root_pose_head: str = "root_pose",
        obs_is_normalized: bool = False,
    ) -> torch.Tensor:
        if not isinstance(obs, dict):
            raise TypeError("root XY distance requires dictionary observations")
        if root_pose_head not in self._latent_reconstruction_heads:
            raise KeyError(f"Root-pose reconstruction head {root_pose_head!r} was not found")
        normalized_obs = obs if obs_is_normalized else self._normalize(obs)
        encoder_obs = self._encoder_obs(normalized_obs)
        current_root_pose = encoder_obs["root_pose"]
        goal_root_pose = self._latent_reconstruction_heads[root_pose_head](goal_z)
        normalized_delta = current_root_pose.float() - goal_root_pose.float()
        root_pose_delta = self._obs_normalizer.denormalize_delta("root_pose", normalized_delta)
        return torch.linalg.vector_norm(root_pose_delta[..., :2] * ROOT_POSE_SCALE, dim=-1)

    def goal_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return self.backward_map(next_obs)

    def tracking_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.backward_map(next_obs)
        return self.aggregate_latent_sequence(z, dim=0)

    def reward_inference(
        self, next_obs: torch.Tensor | dict[str, torch.Tensor], reward: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        z = self.backward_map(next_obs)
        wr = reward if weight is None else reward * weight
        return self.project_z(torch.matmul(wr.to(self.device).T, z))

    def reward_wr_inference(
        self, next_obs: torch.Tensor | dict[str, torch.Tensor], reward: torch.Tensor
    ) -> torch.Tensor:
        return self.reward_inference(next_obs, reward, F.softmax(10 * reward, dim=0))


class JepaSuccessorCprModelArchiConfig(JepaSuccessorModelArchiConfig):
    """Successor architecture augmented with CPR discriminator/value heads."""

    critic: ForwardArchiConfig = ForwardArchiConfig()
    discriminator: DiscriminatorArchiConfig = DiscriminatorArchiConfig()


class JepaSuccessorCprModelConfig(JepaSuccessorModelConfig):
    name: tp.Literal["JepaSuccessorCprModel"] = "JepaSuccessorCprModel"
    archi: JepaSuccessorCprModelArchiConfig = JepaSuccessorCprModelArchiConfig()

    def build(self, obs_space, action_dim: int) -> "JepaSuccessorCprModel":
        return JepaSuccessorCprModel(obs_space, action_dim, self)


class JepaSuccessorCprModel(JepaSuccessorModel):
    """JEPA-successor model with the CPR discriminator and value head."""

    config_class = JepaSuccessorCprModelConfig

    def __init__(self, obs_space, action_dim: int, cfg: JepaSuccessorCprModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        self.cfg: JepaSuccessorCprModelConfig = cfg
        self._discriminator = cfg.archi.discriminator.build(obs_space, cfg.archi.z_dim)
        self._critic = cfg.archi.critic.build(obs_space, cfg.archi.z_dim, action_dim, output_dim=1)
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "_target_critic"):
            self._target_critic.train(False)
        return self

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)
        if hasattr(self, "_target_critic"):
            self._target_critic.requires_grad_(False)
        return self

    def _prepare_for_train(self) -> None:
        super()._prepare_for_train()
        self._target_critic = copy.deepcopy(self._critic)
        self._target_critic.requires_grad_(False)
        self._target_critic.train(False)

    @torch.no_grad()
    def critic(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        with autocast(device_type=_device_type(self.device), dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._critic(self._normalize(obs), z, action)

    @torch.no_grad()
    def discriminator(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor):
        with autocast(device_type=_device_type(self.device), dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._discriminator(self._normalize(obs), z)


class JepaSuccessorCprAuxModelArchiConfig(JepaSuccessorCprModelArchiConfig):
    """CPR successor architecture augmented with an auxiliary critic."""

    aux_critic: ForwardArchiConfig = ForwardArchiConfig()


class JepaSuccessorCprAuxModelConfig(JepaSuccessorCprModelConfig):
    name: tp.Literal["JepaSuccessorCprAuxModel"] = "JepaSuccessorCprAuxModel"
    archi: JepaSuccessorCprAuxModelArchiConfig = JepaSuccessorCprAuxModelArchiConfig()
    norm_aux_reward: RewardNormalizerConfig = RewardNormalizerConfig()

    def build(self, obs_space, action_dim: int) -> "JepaSuccessorCprAuxModel":
        return JepaSuccessorCprAuxModel(obs_space, action_dim, self)


class JepaSuccessorCprAuxModel(JepaSuccessorCprModel):
    """JEPA-successor model with CPR discriminator/value and Aux value heads."""

    config_class = JepaSuccessorCprAuxModelConfig

    def __init__(self, obs_space, action_dim: int, cfg: JepaSuccessorCprAuxModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        self.cfg: JepaSuccessorCprAuxModelConfig = cfg
        self._aux_critic = cfg.archi.aux_critic.build(obs_space, cfg.archi.z_dim, action_dim, output_dim=1)
        self._aux_reward_normalizer = cfg.norm_aux_reward.build()
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "_target_aux_critic"):
            self._target_aux_critic.train(False)
        return self

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)
        if hasattr(self, "_target_aux_critic"):
            self._target_aux_critic.requires_grad_(False)
        return self

    def _prepare_for_train(self) -> None:
        super()._prepare_for_train()
        self._target_aux_critic = copy.deepcopy(self._aux_critic)
        self._target_aux_critic.requires_grad_(False)
        self._target_aux_critic.train(False)

    @torch.no_grad()
    def aux_critic(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        with autocast(device_type=_device_type(self.device), dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._aux_critic(self._normalize(obs), z, action)
