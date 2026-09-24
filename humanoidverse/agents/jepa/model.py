import gymnasium
import torch

from .. import latent_reconstruction as _latent_reconstruction
from ..nn_models import BackwardArchiConfig

LatentReconstructionTarget = _latent_reconstruction.LatentReconstructionTarget
LatentReconstructionBodyField = _latent_reconstruction.LatentReconstructionBodyField
LatentReconstructionBodyPositionFrame = _latent_reconstruction.LatentReconstructionBodyPositionFrame
LatentReconstructionSide = _latent_reconstruction.LatentReconstructionSide
LatentReconstructionLoss = _latent_reconstruction.LatentReconstructionLoss
LatentReconstructionHeadConfig = _latent_reconstruction.LatentReconstructionHeadConfig
LatentReconstructionHead = _latent_reconstruction.LatentReconstructionHead


def _device_type(device: str | torch.device) -> str:
    return torch.device(device).type


def temporal_aggregation_weights(
    length: int, *, discounted: bool, discount: float, device: torch.device
) -> torch.Tensor:
    """Return normalized oldest-to-newest weights for a future window.

    Offset zero is the nearest future state.  Normalizing the weights keeps
    aggregation scale independent of both ``seq_length`` and a shortened
    window at the end of a trajectory.
    """
    if length <= 0:
        raise ValueError("temporal aggregation requires a non-empty window")
    if discounted:
        offsets = torch.arange(length, device=device, dtype=torch.float32)
        weights = torch.pow(torch.tensor(discount, device=device, dtype=torch.float32), offsets)
    else:
        weights = torch.ones(length, device=device, dtype=torch.float32)
    return weights / weights.sum().clamp_min(1e-12)


def aggregate_continuous_latent_window(
    z: torch.Tensor, *, dim: int, discounted: bool, discount: float, weights: torch.Tensor | None = None
) -> torch.Tensor:
    dim = dim % z.ndim
    if weights is None:
        weights = temporal_aggregation_weights(z.shape[dim], discounted=discounted, discount=discount, device=z.device)
    weights = weights.to(device=z.device, dtype=z.dtype)
    shape = [1] * z.ndim
    shape[dim] = weights.shape[0]
    return (z * weights.view(shape)).sum(dim=dim)


class KeypointEncoderConfig(BackwardArchiConfig):
    enabled: bool = False
    input_key: str = "keypoint_state"
    use_actor_obs: bool = False

    def model_post_init(self, context):
        if not self.use_actor_obs and (not self.input_key):
            raise ValueError("keypoint encoder input_key must be non-empty.")
        if self.hidden_dim <= 0:
            raise ValueError("keypoint encoder hidden_dim must be > 0.")
        if self.hidden_layers < 1:
            raise ValueError("keypoint encoder hidden_layers must be >= 1.")

    def build(self, obs_space, z_dim: int, num_levels: int, *, actor_input_filter=None) -> "KeypointEncoder":
        if self.use_actor_obs:
            if actor_input_filter is None:
                raise ValueError("keypoint encoder requires the actor input filter when use_actor_obs=True.")
            input_space = obs_space
            input_filter = actor_input_filter
        else:
            if not isinstance(obs_space, gymnasium.spaces.Dict) or self.input_key not in obs_space.spaces:
                raise ValueError(f"keypoint encoder requires observation key '{self.input_key}'.")
            input_space = obs_space[self.input_key]
            if not isinstance(input_space, gymnasium.spaces.Box) or len(input_space.shape) != 1:
                raise ValueError("keypoint encoder input must be a one-dimensional Box observation.")
            input_filter = self.input_filter
        backbone_cfg = BackwardArchiConfig(
            name="BackwardArchi",
            hidden_dim=self.hidden_dim,
            model=self.model,
            hidden_layers=self.hidden_layers,
            norm=self.norm,
            input_filter=input_filter,
        )
        output_dim = z_dim * num_levels if num_levels > 0 else z_dim
        return KeypointEncoder(backbone=backbone_cfg.build(input_space, output_dim), z_dim=z_dim, num_levels=num_levels)

    def select_input(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.use_actor_obs:
            return obs
        if not isinstance(obs, dict):
            raise ValueError("keypoint encoder requires dictionary observations.")
        if self.input_key not in obs:
            raise KeyError(f"keypoint encoder input '{self.input_key}' is missing from the observation.")
        return obs[self.input_key]


class KeypointEncoder(torch.nn.Module):
    """Predict FSQ logits or a continuous latent from the configured observation."""

    def __init__(self, backbone: torch.nn.Module, z_dim: int, num_levels: int):
        super().__init__()
        self.z_dim = z_dim
        self.num_levels = num_levels
        self.backbone = backbone

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.backbone(obs)
        if self.num_levels == 0:
            return output
        return output.view(*output.shape[:-1], self.z_dim, self.num_levels)
