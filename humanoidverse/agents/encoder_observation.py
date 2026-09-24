from collections.abc import Mapping, MutableMapping
from typing import Any

CLEAN_ENCODER_OBS_KEY = "encoder_observation"


def ensure_clean_encoder_obs_enabled_for_model(env_config: Any, model_or_cfg: Any) -> None:
    cfg = getattr(model_or_cfg, "cfg", model_or_cfg)
    use_clean_encoder_obs = bool(
        getattr(cfg, "use_clean_encoder_obs", False) or getattr(cfg, "use_clean_phi_observation", False)
    )
    if not use_clean_encoder_obs:
        return
    if isinstance(env_config, MutableMapping):
        env_config["include_clean_encoder_obs"] = True


def select_encoder_observation(
    obs: Any, *, use_clean_encoder_obs: bool, clean_encoder_obs_key: str = CLEAN_ENCODER_OBS_KEY
) -> Any:
    if not use_clean_encoder_obs or not isinstance(obs, Mapping) or clean_encoder_obs_key not in obs:
        return obs
    clean_obs = obs[clean_encoder_obs_key]
    if not isinstance(clean_obs, Mapping):
        return obs
    encoder_obs = dict(obs)
    for key, value in clean_obs.items():
        encoder_obs[key] = value
    return encoder_obs


def normalize_observation_with_clean_encoder_obs(
    obs: Any, obs_normalizer, *, clean_encoder_obs_key: str = CLEAN_ENCODER_OBS_KEY
) -> Any:
    normalized_obs = obs_normalizer(obs)
    if not isinstance(obs, Mapping) or clean_encoder_obs_key not in obs or (not isinstance(normalized_obs, Mapping)):
        return normalized_obs
    clean_obs = obs[clean_encoder_obs_key]
    if not isinstance(clean_obs, Mapping):
        return normalized_obs
    normalized_obs = dict(normalized_obs)
    clean_normalizer_input = dict(obs)
    clean_normalizer_input.update(clean_obs)
    normalized_clean_input = obs_normalizer(clean_normalizer_input)
    if isinstance(normalized_clean_input, Mapping):
        normalized_obs[clean_encoder_obs_key] = {
            key: normalized_clean_input[key] for key in clean_obs.keys() if key in normalized_clean_input
        }
    else:
        normalized_obs[clean_encoder_obs_key] = normalized_clean_input
    return normalized_obs
