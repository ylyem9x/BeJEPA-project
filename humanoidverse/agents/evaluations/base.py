from ..base import BaseConfig
from ..base_model import BaseModel


class BaseEvalConfig(BaseConfig):
    """Abstract class for evaluation configurations."""

    generate_videos: bool = False
    videos_dir: str = "videos"
    video_name_prefix: str = "unknown_agent"

    @classmethod
    def requires_replay_buffer(self):
        return False


def extract_model(agent_or_model):
    if isinstance(agent_or_model, BaseModel):
        return agent_or_model
    return agent_or_model._model
