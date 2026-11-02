# BeJEPA

[English](README.md) | [简体中文](README.zh-CN.md)

[Project Page](https://ylyem9x.github.io/BeJEPA-project/) | [Paper (arXiv)](https://arxiv.org/abs/2610.23127)

This repository provides the source code for **BeJEPA: A Goal-Conditioned Humanoid Behavior World Model via Unsupervised RL**, built on the **BFM-Zero** framework.

BeJEPA uses human motion priors and robot interaction to learn a behavior world model through unsupervised reinforcement learning, jointly representing whole-body posture, world position, and heading. An action-conditioned **Dynamic Predictor** predicts the next state embedding, while a goal-conditioned **Successor Predictor** summarizes a policy's expected discounted future state embeddings in a single vector. Together, they guide policy learning for goal reaching and motion tracking on the Unitree G1 humanoid.

## Installation

Use Linux x86_64 and uv for Python environment management. Training requires an NVIDIA GPU and a driver compatible with Isaac Sim 4.5.

Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install the environment:

```bash
git clone https://github.com/ylyem9x/BeJEPA-project.git BeJEPA
cd BeJEPA
bash setup.sh
```

`setup.sh` runs `uv sync --locked` and unpacks the bundled motion data. The environment is installed in the project-local `.venv/` directory.

Alternatively, run the installation steps manually:

```bash
uv sync --locked
.venv/bin/python -m humanoidverse.prepare_data
```

## Data and Model Weights

The complete release includes compressed parts in `humanoidverse/data/packed/`. The setup script restores them to `humanoidverse/data/`:

| File | Purpose |
| --- | --- |
| `lafan_locomotion_mix2000.pkl` | Training set with 2,000 motions |
| `lafan_29dof.pkl` | Evaluation set with 40 motions |

Training and playback also automatically attempt to unpack these parts when the default datasets are missing.

## Training

Start a single-GPU training run:

```bash
CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=WARN .venv/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=1 \
  -m humanoidverse.train_g1_jepa_successor \
  --work-dir outputs/bejepa
```

All editable defaults are listed in [train_g1_jepa_successor_config.py](humanoidverse/train_g1_jepa_successor_config.py), including network architecture, optimization, loss schedules, goal sampling, rewards, environment settings, and evaluations. The default recipe uses 4,096 environments, a batch size of 4,096, a 256-dimensional latent space, and 384 million environment steps. Training converges at around 100 million environment steps.

Inspect the available options or print the effective configuration before launching:

```bash
.venv/bin/python -m humanoidverse.train_g1_jepa_successor --help
.venv/bin/python -m humanoidverse.train_g1_jepa_successor --print-config
```

Edit the Python defaults or append CLI overrides such as `--batch-size 2048`, `--lr-phi 2e-5`, or `--encoder.hidden-dim 512` to the training command. [training.json](humanoidverse/config/training.json) is an exported reference snapshot; the Python configuration supplies the defaults. Use `--training-motion-path` and `--evaluation-motion-path` to select different motion archives.

Logs and checkpoints are saved under `--work-dir`. Launching again with the same directory resumes its checkpoint. Choose a new directory to start a separate run.

### W&B Logging

W&B logging is disabled by default. To enable it, authenticate once:

```bash
.venv/bin/wandb login
```

Then append `--use-wandb --wandb-entity YOUR_ENTITY --wandb-project BeJEPA` to the training command, replacing `YOUR_ENTITY` with your username or team name.

## Multi-GPU Training

Use one process per GPU. For two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN .venv/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m humanoidverse.train_g1_jepa_successor \
  --work-dir outputs/bejepa_2gpu
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

### Goal Reaching

The `goal` mode takes the posture from each selected motion/frame pair and sets the target root position and heading relative to the rollout's initial pose. Multiple values, such as `--source-frame-ids 0 150 300`, cycle every 150 control steps by default; adjust this with `--goal-hold-steps`. Use `--reset-on-goal-switch` to reset between goals, or `--episode-len 0` to visit every goal once.

### Motion Tracking

The `tracking` mode follows the reference from its initial state. Future-frame aggregation defaults to the checkpoint's sequence length and discount settings. Override it with `--future-frames 6 --future-weighting gamma --future-gamma 0.98`, or use `--future-frames 0` for a single-frame target. `--episode-len 0` plays the full motion; positive values cap the number of control steps. The reference ghost is aligned once at reset so drift remains visible.

### Outputs and Rendering

Both modes save MP4 video, JSON metrics, and NPZ trajectories under `play_goal/` or `play_tracking/` in the model folder. Set `--output-folder` to redirect outputs or `--no-save-mp4` to skip video rendering.

MuJoCo physics runs on CPU; `--device cuda` places policy inference on GPU. Headless video rendering uses EGL by default; set `MUJOCO_GL=osmesa` if using an OSMesa installation. Use `--no-headless` for the interactive viewer on a machine with a display, or `--simulator isaacsim --env-device cuda` for Isaac Sim playback.

The equivalent Python entry points and all options are available with:

```bash
.venv/bin/python -m humanoidverse.play_goal --help
.venv/bin/python -m humanoidverse.play_tracking --help
```

## File Structure

Key files and directories are shown below. `.venv/` and `outputs/` are generated locally:

```text
BeJEPA/
├── README.md                             # English documentation
├── README.zh-CN.md                       # Chinese documentation
├── pyproject.toml                        # Package and dependency definitions
├── uv.lock                               # Locked environment
├── .python-version                       # Python 3.10
├── .gitattributes                        # Binary data and shell line endings
├── setup.sh                              # Dependency installation and data unpacking
├── play.sh                               # Goal and tracking playback
├── LICENSE
├── humanoidverse/
│   ├── train_g1_jepa_successor.py        # Training entry point
│   ├── train_g1_jepa_successor_config.py # Editable training defaults
│   ├── prepare_data.py                   # Restore bundled motion archives
│   ├── play_goal.py                      # Pose-goal playback
│   ├── play_tracking.py                  # Motion-tracking playback
│   ├── playback.py                       # Checkpoint loading and ghost rendering
│   ├── tracking_utils.py                 # Latent aggregation and world alignment
│   ├── train.py                          # Data collection, updates, evaluation, checkpoints
│   ├── distributed_launch.py             # GPU assignment for distributed workers
│   ├── agents/
│   │   ├── jepa_successor/
│   │   │   ├── model.py                  # Encoders, Dynamic, Successor, actor, and critics
│   │   │   └── agent.py                  # Training objectives and optimizer updates
│   │   ├── jepa/                         # Shared encoder and agent utilities
│   │   ├── buffers/                      # Transition and trajectory replay buffers
│   │   ├── envs/                         # Environment wrapper and expert-data loading
│   │   ├── evaluations/                  # Motion tracking and goal-reaching evaluation
│   │   └── latent_reconstruction.py      # Root-pose and body reconstruction heads
│   ├── config/                           # Training, robot, observation, reward, and other settings
│   ├── envs/                             # Humanoid simulation tasks
│   ├── simulator/                        # Isaac Sim and MuJoCo backends
│   ├── data/
│   │   ├── data_manifest.json            # Data parts and byte sizes
│   │   ├── packed/                       # Complete compressed motion data
│   │   ├── lafan_locomotion_mix2000.pkl  # Training motions (unpacked locally)
│   │   ├── lafan_29dof.pkl               # Evaluation motions (unpacked locally)
│   │   └── robots/g1/                    # G1 robot descriptions and geometry
│   └── utils/                            # Motion loading and simulation utilities
├── .venv/                                # Created by uv sync
└── outputs/                              # Training logs and checkpoints
```

## Acknowledgements and License

BeJEPA builds on [BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero) and uses Isaac Sim, Isaac Lab, and MuJoCo for simulation. Please retain the attribution and license notices of the upstream code and assets when redistributing this repository.

The repository includes a [CC BY-NC 4.0 license](LICENSE). Motion datasets, robot assets, and third-party dependencies remain subject to their respective licenses and usage terms.

## Citation

If you use BeJEPA in your research, please cite our paper:

```bibtex
@misc{yan2026bejepa,
  title         = {{BeJEPA}: A Goal-Conditioned Humanoid Behavior World Model via Unsupervised RL},
  author        = {Liyun Yan and Haoyang Li and Yang Zhang and Jialing Zhu and Zhanxiang Cao and Zihao Zhou and Keqi Zhu and Yizhi Chen and Huang Chuxuan and Shengcheng Fu and Zhong Jiangwei and Qiao Sun and Chen Zhang and Yue Gao},
  year          = {2026},
  eprint        = {2610.23127},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2610.23127},
  url           = {https://arxiv.org/abs/2610.23127}
}
```
