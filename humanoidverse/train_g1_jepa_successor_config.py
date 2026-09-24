"""BeJEPA training, architecture, and evaluation defaults.

Scalar options use --lr-phi, --batch-size, etc. Nested options use dotted names,
for example --encoder.hidden-dim 512 or --tracking-evaluation.num-envs 128.
"""

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal


@dataclass
class EncoderConfig:
    hidden_dim: int = 1024
    hidden_layers: int = 6
    model: Literal["simple", "residual"] = "residual"
    norm: bool = True
    input_keys: list[str] = field(default_factory=lambda: ["state", "privileged_state", "root_pose"])


@dataclass
class PolicyStateEncoderConfig:
    """Uses the deployable observation keys configured in ActorConfig."""

    hidden_dim: int = 512
    hidden_layers: int = 4
    model: Literal["simple", "residual"] = "residual"
    norm: bool = True


@dataclass
class PredictorConfig:
    hidden_dim: int = 2048
    hidden_layers: int = 3
    model: Literal["simple", "residual"] = "residual"
    num_parallel: int = 2


@dataclass
class ActorConfig:
    hidden_dim: int = 2048
    hidden_layers: int = 6
    embedding_layers: int = 2
    model: Literal["simple", "residual"] = "residual"
    output_scale: float = 1.0
    input_keys: list[str] = field(default_factory=lambda: ["state", "root_pose", "last_action", "history_actor"])


@dataclass
class CriticConfig:
    hidden_dim: int = 2048
    hidden_layers: int = 6
    embedding_layers: int = 2
    model: Literal["simple", "residual"] = "residual"
    num_parallel: int = 2
    ensemble_mode: Literal["batch", "seq", "vmap"] = "batch"
    input_keys: list[str] = field(
        default_factory=lambda: ["state", "privileged_state", "root_pose", "last_action", "history_actor"]
    )


@dataclass
class DiscriminatorConfig:
    hidden_dim: int = 1024
    hidden_layers: int = 3
    input_keys: list[str] = field(default_factory=lambda: ["state", "privileged_state", "root_pose"])


@dataclass
class ReconstructionHeadConfig:
    target: Literal["root_pose", "body"] = "root_pose"
    coef: float = 0.05
    backprop_to_encoder: bool = False
    weight: list[float] | None = field(default_factory=lambda: [2.0, 2.0, 0.25, 0.25])
    hidden_dim: int = 128
    hidden_layers: int = 2
    side: Literal["goal", "policy", "phi", "psi", "both"] = "goal"
    bodies: list[str] = field(default_factory=list)
    fields: list[Literal["pos", "rot", "pose", "vel", "ang_vel"]] = field(default_factory=list)
    body_position_frame: Literal["root", "anchor"] = "root"
    loss: Literal["mse", "smooth_l1"] = "mse"


def default_reconstruction_heads() -> dict[str, ReconstructionHeadConfig]:
    """Set coef=0 to disable a head; add dictionary entries for other targets."""
    return {
        "root_pose": ReconstructionHeadConfig(),
        "wrists": ReconstructionHeadConfig(
            target="body",
            coef=0.05,
            backprop_to_encoder=True,
            weight=None,
            hidden_dim=256,
            hidden_layers=4,
            side="goal",
            bodies=["left_wrist_yaw_link", "right_wrist_yaw_link"],
            fields=["pose", "vel", "ang_vel"],
            body_position_frame="root",
            loss="mse",
        ),
    }


@dataclass
class EnvironmentConfig:
    relative_config_path: str = "exp/bfm_zero/bfm_zero"
    expert_motion_limit: int | None = None
    max_episode_length_s: float | None = None
    enable_cameras: bool = False
    camera_render_save_dir: str = "isaac_videos"
    disable_obs_noise: bool = False
    disable_domain_randomization: bool = False
    include_last_action: bool = True
    include_history_actor: bool = True
    include_history_noaction: bool = False
    context_length: int | None = None
    include_dr_info: bool = False
    included_dr_obs_names: list[str] | None = None
    actor_root_pose_noise: float = 0.01
    actor_root_pose_history_length: int = 4
    root_height_obs: bool = True
    make_config_g1env_compatible: bool = False

    # Robot control and reset defaults. Extend this list to change simulator,
    # terrain, PD-gain, noise, or domain-randomization fields in the Hydra YAMLs.
    hydra_overrides: list[str] = field(
        default_factory=lambda: [
            "robot=g1/g1_29dof_hard_waist",
            "robot.control.action_scale=0.25",
            "robot.control.action_clip_value=5.0",
            "robot.control.action_rescale=true",
            "robot.control.clip_torques=true",
            "robot.control.normalize_action=true",
            "robot.control.normalize_action_from=1.0",
            "robot.control.normalize_action_to=5.0",
            "env.config.lie_down_init=True",
            "env.config.lie_down_init_prob=0.3",
        ]
    )


@dataclass
class TrackingEvaluationConfig:
    enabled: bool = True
    num_envs: int = 1024
    # None uses evaluation_motion_path from the main configuration.
    reference_motion_path: Path | None = None
    n_episodes_per_motion: int = 2
    include_results_from_all_envs: bool = False
    max_steps_per_motion: int | None = 500
    emd_max_points: int | None = 512
    name_in_logs: str = "humanoidverse_tracking_eval"
    generate_videos: bool = False
    videos_dir: str = "videos"
    video_name_prefix: str = "bejepa"
    disable_tqdm: bool = True


@dataclass
class GoalEvaluationConfig:
    enabled: bool = True
    num_envs: int = 1024
    reference_motion_path: Path | None = None
    rollout_steps: int = 250
    max_goal_dist_m: float = 1.0
    max_goal_yaw_rad: float = math.pi
    wrist_body_names: list[str] = field(default_factory=lambda: ["left_wrist_yaw_link", "right_wrist_yaw_link"])
    wrist_score_pos_scale_m: float = 0.05
    wrist_score_rot_scale_rad: float = math.pi / 18
    wrist_score_pos_weight: float = 1.0
    wrist_score_rot_weight: float = 1.0
    name_in_logs: str = "goal_eval"
    generate_videos: bool = False
    videos_dir: str = "videos"
    video_name_prefix: str = "bejepa"
    disable_tqdm: bool = True


@dataclass
class OptimizationConfig:
    """Flat CLI options inherited by G1JepaSuccessorConfig below."""

    # Optimizers. lr_phi trains the shared observation encoder;
    # lr_keypoint_encoder trains the deployable policy-state encoder.
    # lr_latent_reconstruction=None uses lr_successor.
    batch_size: int = 4096
    lr_phi: float = 1e-05
    lr_keypoint_encoder: float = 0.0003
    lr_dynamics: float = 0.0003
    lr_successor: float = 0.0003
    lr_actor: float = 0.0003
    lr_critic: float = 0.0001
    lr_aux_critic: float = 0.0001
    lr_discriminator: float = 1e-05
    lr_latent_reconstruction: float | None = 1e-05
    weight_decay: float = 0.0
    weight_decay_discriminator: float = 0.0
    clip_grad_norm: float = 10.0
    fused_optimizer: bool = True
    distributed_bucket_cap_mb: float = 25.0

    # Learning-rate schedule, measured in global environment steps.
    # The critic entry schedules both CPR and auxiliary critics.
    lr_decay_function: Literal["none", "linear", "cosine", "exponential"] = "cosine"
    lr_decay_networks: list[
        Literal[
            "encoder",
            "dynamics",
            "successor",
            "actor",
            "critic",
            "discriminator",
            "policy_state_encoder",
            "latent_reconstruction",
        ]
    ] = field(default_factory=lambda: ["encoder"])
    lr_decay_start_step: int = 50000000
    lr_decay_end_step: int = 150000000
    lr_decay_final_ratio: float = 0.01

    # World-model losses and target updates. Set both schedule endpoints to
    # None to use the constant dynamics_loss_coef throughout training.
    discount: float = 0.98
    dynamics_loss_coef: float = 0.1
    dynamics_loss_coef_schedule_function: Literal["log", "cosine"] = "cosine"
    dynamics_loss_coef_schedule_start: tuple[int, float] | None = (30000000, 0.1)
    dynamics_loss_coef_schedule_end: tuple[int, float] | None = (100000000, 0.001)
    successor_loss_coef: float = 1.0
    dynamics_backprop_to_encoder: bool = True
    successor_backprop_to_encoder: bool = True
    # False uses the online encoder for TD targets; True enables EMA targets.
    encoder_target_ema: bool = False
    encoder_target_tau: float = 0.005
    successor_target_tau: float = 0.005
    critic_target_tau: float = 0.005
    successor_td_use_g_train: bool = False
    target_delta_metrics_every_updates: int = 100

    # Orthogonality regularization on the shared observation encoder.
    phi_ortho_coef: float = 0.0001

    # Goal sampling, relabeling, and rollout collection.
    # random_goal_ratio=None assigns remaining probability to random goals.
    train_goal_ratio: float = 0.4
    expert_goal_ratio: float = 0.6
    random_goal_ratio: float | None = 0.0
    relabel_ratio: float | None = 0.8
    use_mix_rollout: bool = True
    update_z_every_step: int = 150
    z_buffer_size: int = 8192
    rollout_expert_trajectories: bool = True
    rollout_expert_trajectories_length: int = 250
    rollout_expert_trajectories_percentage: float = 0.5
    tracking_goal_randomize_start: bool = True
    tracking_goal_start_xy_radius_m: float = 0.5
    tracking_goal_start_yaw_limit_rad: float = math.pi
    exclude_terminal_next_obs_from_train: bool = False
    terminal_batch_sample_max_attempts: int = 64

    # Actor, CPR, auxiliary critic, and discriminator objectives.
    actor_pessimism_penalty: float = 0.25
    stddev_clip: float = 0.3
    bc_coeff: float = 0.0
    critic_pessimism_penalty: float = 0.0
    aux_critic_pessimism_penalty: float = 0.0
    reg_coeff: float = 0.005
    reg_coeff_aux: float = 0.005
    scale_reg: bool = True
    grad_penalty_discriminator: float = 10.0
    cpr_critic_use_behavior_goal: bool = False
    aux_critic_use_behavior_goal: bool = False


@dataclass
class G1JepaSuccessorConfig(OptimizationConfig):
    # Paths and execution. Run from the BeJEPA directory.
    work_dir: Path = Path("outputs/bejepa")
    training_motion_path: Path = Path("humanoidverse/data/lafan_locomotion_mix2000.pkl")
    evaluation_motion_path: Path = Path("humanoidverse/data/lafan_29dof.pkl")
    device: str = "cuda:0"
    seed: int = 4728
    compile: bool = False
    cudagraphs: bool = False
    amp: bool = False
    isaac_zero_copy: bool = True
    disable_tqdm: bool = True
    print_config: bool = False

    # Training scale. Counts are global across distributed workers.
    num_envs: int = 4096
    num_env_steps: int = 384_000_000
    num_agent_updates: int = 16
    # None means num_envs (4096); an explicit value overrides auto-scaling.
    update_agent_every: int | None = None
    # None means 10 * num_envs (40960).
    num_seed_steps: int | None = None
    log_every_updates: int = 384_000
    checkpoint_every_steps: int = 4_800_000
    checkpoint_buffer: bool = False

    # Replay storage and expert-motion sampling.
    buffer_size: int = 5_120_000
    # None follows device; set "cpu" to store replay outside GPU memory.
    buffer_device: str | None = None
    buffer_pin_memory: bool = False
    use_trajectory_buffer: bool = True
    padding_beginning: int = 0
    padding_end: int = 0
    prioritization: bool = False
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 2.0
    prioritization_scale: float = 2.0
    prioritization_mode: Literal["bin", "lin", "exp"] = "exp"

    # Model inputs and shared latent space. Observation switches are forwarded
    # to both the model and environment to keep their formats aligned.
    z_dim: int = 256
    norm_z: bool = True
    # Append recent actions directly to the policy latent input.
    actor_direct_action_history_length: int = 4
    actor_std: float = 0.1
    inference_batch_size: int = 500_000
    seq_length: int = 1
    seq_aggregation_discounted: bool = False
    use_clean_encoder_obs: bool = True
    privileged_state_world_frame: bool = False
    norm_aux_reward_translate: bool = False
    norm_aux_reward_scale: bool = True
    obs_normalizer_allow_mismatching_keys: bool = True
    # Per-key BatchNorm momentum. Identity keys below bypass normalization.
    obs_normalizer_momenta: dict[str, float] = field(
        default_factory=lambda: {
            "state": 0.01,
            "privileged_state": 0.01,
            "root_pose": 0.01,
            "keypoint_state": 0.01,
            "last_action": 0.01,
            "history_actor": 0.01,
        }
    )
    obs_normalizer_identity_keys: list[str] = field(default_factory=list)

    # Each network can be configured independently through its fields above.
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    policy_state_encoder: PolicyStateEncoderConfig = field(default_factory=PolicyStateEncoderConfig)
    dynamics: PredictorConfig = field(default_factory=PredictorConfig)
    successor: PredictorConfig = field(default_factory=lambda: PredictorConfig(hidden_layers=6))
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    aux_critic: CriticConfig = field(default_factory=CriticConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    latent_reconstruction: dict[str, ReconstructionHeadConfig] = field(default_factory=default_reconstruction_heads)
    # Must match the body order in privileged_state, including the pelvis.
    latent_reconstruction_body_names: list[str] = field(
        default_factory=lambda: [
            "pelvis",
            "left_hip_pitch_link",
            "left_hip_roll_link",
            "left_hip_yaw_link",
            "left_knee_link",
            "left_ankle_pitch_link",
            "left_ankle_roll_link",
            "right_hip_pitch_link",
            "right_hip_roll_link",
            "right_hip_yaw_link",
            "right_knee_link",
            "right_ankle_pitch_link",
            "right_ankle_roll_link",
            "waist_yaw_link",
            "waist_roll_link",
            "torso_link",
            "left_shoulder_pitch_link",
            "left_shoulder_roll_link",
            "left_shoulder_yaw_link",
            "left_elbow_link",
            "left_wrist_roll_link",
            "left_wrist_pitch_link",
            "left_wrist_yaw_link",
            "right_shoulder_pitch_link",
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_link",
            "right_wrist_roll_link",
            "right_wrist_pitch_link",
            "right_wrist_yaw_link",
            "head_link",
        ]
    )

    # Auxiliary reward terms. Removing a key disables collection of that term.
    # Agent and environment use the same term list, in this dictionary's order.
    aux_rewards_scaling: dict[str, float] = field(
        default_factory=lambda: {
            "penalty_torques": -1e-6,
            "penalty_action_rate": -0.0,
            "penalty_upper_body_action_rate": -0.1,
            "penalty_other_body_action_rate": -0.1,
            "limits_dof_pos": -10.0,
            "limits_torque": 0.0,
            "penalty_undesired_contact": -1.0,
            "penalty_feet_ori": -0.1,
            "penalty_ankle_roll": -4.0,
            "penalty_slippage": -2.0,
        }
    )
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    # Evaluations share the training environment. Disable either one separately
    # via its enabled flag, or disable both with disable_evaluation.
    disable_evaluation: bool = False
    eval_every_steps: int = 3_600_000
    disable_initial_eval: bool = True
    tracking_evaluation: TrackingEvaluationConfig = field(default_factory=TrackingEvaluationConfig)
    goal_evaluation: GoalEvaluationConfig = field(default_factory=GoalEvaluationConfig)

    # Distributed execution and optional logging.
    distributed: bool = False
    distributed_auto_detect: bool = True
    distributed_backend: Literal["nccl", "gloo"] = "nccl"
    distributed_timeout_s: int = 1200
    use_wandb: bool = False
    wandb_entity: str | None = None
    wandb_project: str = "BeJEPA"
    wandb_group: str | None = None
    tags: dict[str, str | float] = field(
        default_factory=lambda: {"method": "BeJEPA", "control_profile": "bfm_hard_waist"}
    )

    def to_dict(self) -> dict:
        """Assemble the runtime recipe solely from the editable fields above."""
        cfg = asdict(self)
        train = {option.name: cfg.pop(option.name) for option in fields(OptimizationConfig)}
        train["name"] = "JepaSuccessorCprAuxAgentTrainConfig"
        arch = {"name": "JepaSuccessorCprAuxModelArchiConfig"}
        for key in ("z_dim", "norm_z", "actor_direct_action_history_length", "latent_reconstruction_body_names"):
            arch[key] = cfg.pop(key)
        for key, name in (
            ("encoder", "BackwardArchi"),
            ("policy_state_encoder", "PolicyStateEncoderConfig"),
            ("dynamics", "LatentPredictorConfig"),
            ("successor", "LatentPredictorConfig"),
            ("actor", "actor"),
            ("critic", "ForwardArchi"),
            ("aux_critic", "ForwardArchi"),
            ("discriminator", "DiscriminatorArchi"),
        ):
            network = cfg.pop(key)
            if "input_keys" in network:
                network["input_filter"] = {"name": "DictInputFilterConfig", "key": network.pop("input_keys")}
            arch[key] = {"name": name, **network}
        arch["policy_state_encoder"].update(
            enabled=True, use_actor_obs=True, input_filter={"name": "IdentityInputFilterConfig"}
        )
        arch["latent_reconstruction"] = {
            key: {"name": "LatentReconstructionHeadConfig", **head}
            for key, head in cfg.pop("latent_reconstruction").items()
        }
        model = {"name": "JepaSuccessorCprAuxModel", "archi": arch, "successor_discount": self.discount}
        for key in (
            "actor_std",
            "inference_batch_size",
            "seq_length",
            "seq_aggregation_discounted",
            "amp",
            "use_clean_encoder_obs",
            "privileged_state_world_frame",
        ):
            model[key] = cfg.pop(key)
        model["norm_aux_reward"] = {
            "name": "RewardNormalizer",
            "translate": cfg.pop("norm_aux_reward_translate"),
            "scale": cfg.pop("norm_aux_reward_scale"),
        }
        model["obs_normalizer"] = {
            "name": "ObsNormalizerConfig",
            "normalizers": {
                key: {"name": "BatchNormNormalizerConfig", "momentum": momentum}
                for key, momentum in cfg.pop("obs_normalizer_momenta").items()
            },
            "allow_mismatching_keys": cfg.pop("obs_normalizer_allow_mismatching_keys"),
        }
        model["obs_normalizer"]["normalizers"].update(
            {key: {"name": "IdentityNormalizerConfig"} for key in cfg.pop("obs_normalizer_identity_keys")}
        )
        cfg["agent"] = {
            "name": "JepaSuccessorCprAuxAgent",
            "model": model,
            "train": train,
            "compile": cfg.pop("compile"),
            "cudagraphs": cfg.pop("cudagraphs"),
            "aux_rewards": list(self.aux_rewards_scaling),
            "aux_rewards_scaling": cfg.pop("aux_rewards_scaling"),
        }
        device = cfg.pop("device")
        model["device"] = device.split(":", 1)[0]
        cfg["buffer_device"] = self.buffer_device or device
        cfg["env"].update(
            name="humanoidverse_isaac",
            device=device,
            lafan_tail_path=str(cfg.pop("training_motion_path")),
            aux_reward_names=list(self.aux_rewards_scaling),
            include_clean_encoder_obs=self.use_clean_encoder_obs,
            privileged_state_world_frame=self.privileged_state_world_frame,
        )
        evaluations = []
        for key, name in (
            ("tracking_evaluation", "HumanoidVerseIsaacTrackingEvaluationConfig"),
            ("goal_evaluation", "goal_eval"),
        ):
            evaluation = cfg.pop(key)
            enabled = evaluation.pop("enabled")
            if enabled and not self.disable_evaluation:
                evaluation.update(
                    name=name,
                    reference_motion_path=str(evaluation["reference_motion_path"] or self.evaluation_motion_path),
                )
                evaluations.append(evaluation)
        cfg["evaluations"] = evaluations
        for key in ("disable_evaluation", "evaluation_motion_path", "print_config"):
            cfg.pop(key)
        cfg["online_parallel_envs"] = cfg.pop("num_envs")
        cfg["update_agent_every"] = self.num_envs if self.update_agent_every is None else self.update_agent_every
        cfg["num_seed_steps"] = 10 * self.num_envs if self.num_seed_steps is None else self.num_seed_steps
        cfg["wandb_ename"] = cfg.pop("wandb_entity")
        cfg["wandb_pname"] = cfg.pop("wandb_project")
        cfg["wandb_gname"] = cfg.pop("wandb_group")
        cfg["work_dir"] = str(self.work_dir)
        cfg["tags"].setdefault("actor_output_scale", self.actor.output_scale)
        cfg.update(name="TrainConfig", motions="", motions_root="", load_isaac_expert_data=True)
        for key in ("online_parallel_envs", "num_env_steps", "buffer_size", "update_agent_every", "num_agent_updates"):
            if cfg[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if cfg["num_seed_steps"] < 0:
            raise ValueError("num_seed_steps must be non-negative")
        if self.buffer_size < self.num_envs:
            raise ValueError("buffer_size must be at least num_envs")
        return cfg

    def validate_data(self) -> None:
        """Check only the motion files used by enabled training/evaluations."""
        from humanoidverse.prepare_data import ensure_motion_data

        cfg = self.to_dict()
        paths = [Path(cfg["env"]["lafan_tail_path"])]
        paths.extend(Path(evaluation["reference_motion_path"]) for evaluation in cfg["evaluations"])
        for path in dict.fromkeys(paths):
            ensure_motion_data(path)
