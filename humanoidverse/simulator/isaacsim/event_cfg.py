from isaaclab.utils import configclass


@configclass
class EventCfg:
    """Configuration for events."""

    scale_body_mass = None
    random_joint_friction = None
