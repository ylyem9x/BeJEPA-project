# BeJEPA

This repository provides the training code for **BeJEPA: Learning a Behavior World Model for Humanoid Control via Unsupervised RL**, built on the **BFM-Zero** framework. BeJEPA learns a shared representation of whole-body posture, world position, and heading from human motion priors and robot interaction. An action-conditioned **Dynamic Predictor** predicts the next state embedding, while a goal-conditioned **Successor Predictor** summarizes expected discounted future state embeddings. Their composition guides policy learning for goal reaching and motion tracking on the Unitree G1 humanoid.

## Installation

Use Linux x86_64 and uv for Python environment management. Training requires an NVIDIA GPU and a driver compatible with Isaac Sim 4.5. CPU MuJoCo playback does not require a CUDA GPU. Installing the pinned environment requires network access and roughly 25 GB of free disk space.

Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Run the following commands from the **BeJEPA repository root**:

```bash
bash setup.sh
```

`setup.sh` runs `uv sync --locked` and unpacks the bundled motion data. It creates the project-local `.venv` using the pinned dependencies in `uv.lock`: Python 3.10, PyTorch 2.5.1, Isaac Sim 4.5.0, and Isaac Lab 2.0.2. All commands below run from the repository root and use `.venv/bin/python` directly. For manual installation, run `uv sync --locked` followed by `.venv/bin/python -m humanoidverse.prepare_data`.

Both `git clone` and GitHub's **Download ZIP** include the full data as compressed parts in `humanoidverse/data/packed/`. Setup restores `lafan_locomotion_mix2000.pkl` (2,000 training motions) and `lafan_29dof.pkl` (40 evaluation motions) in `humanoidverse/data/`. Training and playback also unpack missing default datasets automatically. G1 MJCF, meshes, and USD assets are included. No Git LFS or separate dataset download is needed. Part names and uncompressed sizes are listed in [data_manifest.json](humanoidverse/data/data_manifest.json).

Trained weights are produced by the training commands below and are not bundled.

## Training

Start a single-GPU training run:

```bash
CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=WARN .venv/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=1 \
  -m humanoidverse.train_g1_jepa_successor \
  --work-dir outputs/bejepa
```

All editable defaults are listed in [train_g1_jepa_successor_config.py](humanoidverse/train_g1_jepa_successor_config.py), including network architecture, optimization, loss schedules, goal sampling, rewards, environment settings, and evaluations. The default recipe uses 4,096 environments, a batch size of 4,096, a 256-dimensional latent space, and 384 million environment steps.

Inspect the available options or print the effective configuration before launching:

```bash
.venv/bin/python -m humanoidverse.train_g1_jepa_successor --help
.venv/bin/python -m humanoidverse.train_g1_jepa_successor --print-config
```

Edit the Python defaults or append CLI overrides such as `--batch-size 2048`, `--lr-phi 2e-5`, or `--encoder.hidden-dim 512` to the training command. [training.json](humanoidverse/config/training.json) is an exported reference snapshot; the Python configuration supplies the defaults. Use `--training-motion-path` and `--evaluation-motion-path` to select different motion archives.

Logs and checkpoints are saved under `--work-dir`. Launching again with the same directory resumes its checkpoint. Choose a new directory to start a separate run.

W&B logging is disabled by default. To enable it, authenticate once:

```bash
.venv/bin/wandb login
```

Then append `--use-wandb --wandb-entity YOUR_ENTITY --wandb-project BeJEPA` to the training command.

## Multi-GPU Training

Use one process per GPU. For two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN .venv/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m humanoidverse.train_g1_jepa_successor \
  --work-dir outputs/bejepa_2gpu
```

For four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=WARN .venv/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m humanoidverse.train_g1_jepa_successor \
  --work-dir outputs/bejepa_4gpu
```

Distributed execution is detected automatically. The launcher assigns one visible GPU to each worker and synchronizes training gradients. Set `--nproc-per-node` to the number of GPUs listed in `CUDA_VISIBLE_DEVICES`.

`--num-envs`, `--batch-size`, and `--buffer-size` are **global values** divided equally across workers; each must be divisible by the number of processes. With four GPUs, the default configuration gives each worker 1,024 environments and a local batch size of 1,024. The environment-step budget remains global.

## Playback

Play a trained checkpoint with CPU MuJoCo from the repository root:

```bash
./play.sh goal --model-folder outputs/bejepa \
  --source-motion-ids 0 --source-frame-ids 450 \
  --goal-x-m 1.0 --goal-y-m 0.0 --goal-yaw-deg 0.0

./play.sh tracking --model-folder outputs/bejepa --motion-ids 0
```

`--model-folder` is a training run containing `config.json` and `checkpoint/model/`. The default motion archive is `humanoidverse/data/lafan_29dof.pkl`; select another with `--data-path`. `MODEL_FOLDER` and `DATA_PATH` can also be set as environment variables. Motion and frame indices are zero-based.

Goal playback takes the posture from each selected motion/frame pair and replaces its root position and heading relative to the rollout's initial pose. Multiple values, such as `--source-frame-ids 0 150 300`, cycle every `--goal-hold-steps` (default 150). Use `--reset-on-goal-switch` to reset between goals, or `--episode-len 0` to visit every goal once.

Tracking follows the reference from its initial state. Future-frame aggregation defaults to the checkpoint's sequence length and discount settings. Override it with `--future-frames 6 --future-weighting gamma --future-gamma 0.98`, or use `--future-frames 0` for a single-frame target. `--episode-len 0` plays the full motion; positive values cap the number of control steps. The reference ghost is aligned once at reset so drift remains visible.

Both modes save MP4 video, JSON metrics, and NPZ trajectories under `play_goal/` or `play_tracking/` in the model folder. Set `--output-folder` to redirect outputs or `--no-save-mp4` to skip rendering. MuJoCo physics runs on CPU; `--device cuda` places the policy on GPU. Headless video rendering uses EGL by default; set `MUJOCO_GL=osmesa` if using an OSMesa installation. Use `--no-headless` for the interactive viewer on a machine with a display, or `--simulator isaacsim --env-device cuda` for Isaac Sim playback.

The equivalent Python entry points and all options are available with:

```bash
.venv/bin/python -m humanoidverse.play_goal --help
.venv/bin/python -m humanoidverse.play_tracking --help
```

## File Structure

Key files and directories are shown below. `.venv/` and `outputs/` are generated locally. Usage terms are provided in [LICENSE](LICENSE).

```text
BeJEPA/
├── README.md
├── pyproject.toml                         # Package and dependency definitions
├── uv.lock                                # Locked environment
├── .python-version                        # Python 3.10
├── .gitattributes                         # Binary data and shell line endings
├── setup.sh                               # Dependency installation and data unpacking
├── play.sh                                # Goal and tracking playback
├── LICENSE
├── humanoidverse/
│   ├── train_g1_jepa_successor.py          # Training entry point
│   ├── train_g1_jepa_successor_config.py   # Editable training defaults
│   ├── prepare_data.py                    # Restore bundled motion archives
│   ├── play_goal.py                       # Pose-goal playback
│   ├── play_tracking.py                   # Motion-tracking playback
│   ├── playback.py                        # Checkpoint loading and ghost rendering
│   ├── tracking_utils.py                  # Latent aggregation and world alignment
│   ├── train.py                           # Data collection, updates, evaluation, checkpoints
│   ├── distributed_launch.py              # GPU assignment for distributed workers
│   ├── agents/
│   │   ├── jepa_successor/
│   │   │   ├── model.py                   # Encoders, Dynamic, Successor, actor, and critics
│   │   │   └── agent.py                   # Training objectives and optimizer updates
│   │   ├── jepa/                          # Shared encoder and agent utilities
│   │   ├── buffers/                       # Transition and trajectory replay buffers
│   │   ├── envs/                          # Environment wrapper and expert-data loading
│   │   ├── evaluations/                   # Motion tracking and goal-reaching evaluation
│   │   └── latent_reconstruction.py       # Root-pose and body reconstruction heads
│   ├── config/
│   │   ├── training.json                  # Exported reference configuration
│   │   ├── exp/bfm_zero/                  # BFM-Zero environment configuration
│   │   ├── robot/                         # G1 robot and control settings
│   │   ├── obs/                           # Observation definitions
│   │   ├── rewards/                       # Auxiliary rewards
│   │   └── domain_rand/                   # Domain randomization
│   ├── envs/                              # Humanoid simulation tasks
│   ├── simulator/                         # Isaac Sim and MuJoCo backends
│   ├── data/
│   │   ├── data_manifest.json             # Data parts and byte sizes
│   │   ├── packed/                        # Complete compressed motion data
│   │   ├── lafan_locomotion_mix2000.pkl    # Training motions (unpacked locally)
│   │   ├── lafan_29dof.pkl                 # Evaluation motions (unpacked locally)
│   │   └── robots/g1/                     # G1 robot descriptions and geometry
│   └── utils/                             # Motion loading and simulation utilities
├── .venv/                                 # Created by uv sync
└── outputs/                               # Training logs and checkpoints
```
