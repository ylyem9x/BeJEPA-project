# BeJEPA

[English](README.md) | [简体中文](README.zh-CN.md)

[项目网页](https://ylyem9x.github.io/BeJEPA-project/) | [论文（arXiv）](https://arxiv.org/abs/2610.23127)

本仓库提供 **BeJEPA: A Goal-Conditioned Humanoid Behavior World Model via Unsupervised RL** 开源代码，基于 **BFM-Zero** 框架构建。

BeJEPA 利用人类动作先验与机器人交互，通过无监督强化学习，学习统一表示全身姿态、世界坐标位置和朝向的行为世界模型。动作条件 **Dynamic Predictor（动力学预测器）** 预测下一时刻的状态嵌入；目标条件 **Successor Predictor（后继预测器）** 将策略预期的折扣未来状态嵌入汇总为一个向量。二者共同指导策略学习，在 Unitree G1 人形机器人上实现目标到达与动作跟踪。

## 安装

使用 Linux x86_64 系统，并通过 uv 管理 Python 环境。训练需要 NVIDIA GPU，以及兼容 Isaac Sim 4.5 的驱动。

如尚未安装 uv，可运行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

克隆仓库并安装环境：

```bash
git clone https://github.com/ylyem9x/BeJEPA.git
cd BeJEPA
bash setup.sh
```

`setup.sh` 会运行 `uv sync --locked`，并解压随仓库提供的动作数据。环境安装在项目目录下的 `.venv/` 中。

也可以手动执行：

```bash
uv sync --locked
.venv/bin/python -m humanoidverse.prepare_data
```

## 数据与模型权重

完整发布包在 `humanoidverse/data/packed/` 中提供压缩分卷。安装脚本将其还原至 `humanoidverse/data/`：

| 文件 | 用途 |
| --- | --- |
| `lafan_locomotion_mix2000.pkl` | 训练动作，共 2,000 条 |
| `lafan_29dof.pkl` | 评估动作，共 40 条 |

训练和回放也会在默认数据缺失时自动尝试解压这些分卷。

## 训练

启动单 GPU 训练：

```bash
CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=WARN .venv/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=1 \
  -m humanoidverse.train_g1_jepa_successor \
  --work-dir outputs/bejepa
```

可修改的默认参数集中在 [train_g1_jepa_successor_config.py](humanoidverse/train_g1_jepa_successor_config.py)，涵盖网络结构、优化器、损失调度、目标采样、奖励、环境及评估。默认配置使用 4,096 个并行环境、4,096 的批大小、256 维潜在空间和 3.84 亿环境步。（1亿步左右即收敛完成）

启动前可查看参数或打印实际配置：

```bash
.venv/bin/python -m humanoidverse.train_g1_jepa_successor --help
.venv/bin/python -m humanoidverse.train_g1_jepa_successor --print-config
```

可以修改 Python 默认配置，也可以在训练命令后追加命令行参数，例如 `--batch-size 2048`、`--lr-phi 2e-5` 或 `--encoder.hidden-dim 512`。[training.json](humanoidverse/config/training.json) 是导出的参考配置快照，实际默认值由 Python 配置提供。使用 `--training-motion-path` 和 `--evaluation-motion-path` 可指定其他动作数据文件。

日志和检查点保存在 `--work-dir` 指定的目录中。再次使用同一目录启动时，会恢复已有检查点；如需独立训练，请指定新目录。

### W&B 日志

默认关闭 W&B。如需启用，先登录：

```bash
.venv/bin/wandb login
```

然后在训练命令后追加 `--use-wandb --wandb-entity YOUR_ENTITY --wandb-project BeJEPA`，将 `YOUR_ENTITY` 替换为自己的用户名或团队名。

## 多 GPU 训练

每张 GPU 对应一个进程。双 GPU 示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN .venv/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m humanoidverse.train_g1_jepa_successor \
  --work-dir outputs/bejepa_2gpu
```

程序会自动识别分布式运行，为每个进程分配一张可见 GPU 并同步训练梯度。`--nproc-per-node` 应与 `CUDA_VISIBLE_DEVICES` 中的 GPU 数量一致。

`--num-envs`、`--batch-size` 和 `--buffer-size` 均为**全局值**，会在各进程之间均分，因此必须能够被进程数整除。例如，四 GPU 默认配置下，每个进程使用 1,024 个环境和 1,024 的本地批大小。环境步数预算仍按全局统计。

## 回放

使用 CPU MuJoCo 回放训练得到的检查点：

```bash
./play.sh goal --model-folder outputs/bejepa \
  --source-motion-ids 0 --source-frame-ids 450 \
  --goal-x-m 1.0 --goal-y-m 0.0 --goal-yaw-deg 0.0

./play.sh tracking --model-folder outputs/bejepa --motion-ids 0
```

`--model-folder` 指向包含 `config.json` 和 `checkpoint/model/` 的训练目录。默认动作文件为 `humanoidverse/data/lafan_29dof.pkl`，可通过 `--data-path` 指定其他文件，也可设置 `MODEL_FOLDER` 和 `DATA_PATH` 环境变量。动作编号与帧编号均从 0 开始。

### 目标到达

`goal` 模式从指定动作和帧中提取姿态，并根据回放初始位姿设置目标的相对根位置与朝向。传入多个帧编号（例如 `--source-frame-ids 0 150 300`）时，默认每 150 个控制步切换目标，可通过 `--goal-hold-steps` 修改。使用 `--reset-on-goal-switch` 可在目标切换时重置环境；`--episode-len 0` 表示依次访问所有目标一次。

### 动作跟踪

`tracking` 模式从参考动作的初始状态开始跟踪。未来帧聚合默认采用检查点中的序列长度与折扣设置，可用 `--future-frames 6 --future-weighting gamma --future-gamma 0.98` 覆盖；`--future-frames 0` 表示使用单帧目标。`--episode-len 0` 播放完整动作，正数则限制控制步数。参考机器人仅在重置时对齐一次，因此回放过程中可观察到累积漂移。

### 输出与渲染

两种模式均会将 MP4 视频、JSON 指标和 NPZ 轨迹保存到模型目录下的 `play_goal/` 或 `play_tracking/`。使用 `--output-folder` 可更改输出位置，使用 `--no-save-mp4` 可跳过视频渲染。

MuJoCo 物理仿真运行在 CPU 上；`--device cuda` 可将策略推理放到 GPU。无显示器的视频渲染默认使用 EGL；如已安装 OSMesa，可设置 `MUJOCO_GL=osmesa`。在带显示器的机器上，使用 `--no-headless` 打开交互式窗口；也可使用 `--simulator isaacsim --env-device cuda` 进行 Isaac Sim 回放。

对应的 Python 入口及完整参数说明：

```bash
.venv/bin/python -m humanoidverse.play_goal --help
.venv/bin/python -m humanoidverse.play_tracking --help
```

## 文件结构

主要文件与目录如下，`.venv/` 和 `outputs/` 在本地运行时生成：

```text
BeJEPA/
├── README.md                            # 英文说明
├── README.zh-CN.md                      # 中文说明
├── pyproject.toml                       # 项目与依赖定义
├── uv.lock                              # 锁定依赖版本
├── .python-version                      # Python 3.10
├── .gitattributes                       # 二进制数据与脚本换行设置
├── setup.sh                             # 安装依赖并解压数据
├── play.sh                              # 目标到达与动作跟踪回放
├── LICENSE
├── humanoidverse/
│   ├── train_g1_jepa_successor.py        # 训练入口
│   ├── train_g1_jepa_successor_config.py # 可编辑的默认训练参数
│   ├── prepare_data.py                  # 还原动作数据
│   ├── play_goal.py                     # 目标到达回放
│   ├── play_tracking.py                 # 动作跟踪回放
│   ├── playback.py                      # 检查点加载与参考机器人渲染
│   ├── tracking_utils.py                # 潜在表示聚合与世界坐标对齐
│   ├── train.py                         # 数据收集、更新、评估与检查点
│   ├── distributed_launch.py            # 分布式进程的 GPU 分配
│   ├── agents/
│   │   ├── jepa_successor/
│   │   │   ├── model.py                 # 编码器、预测器、策略与价值网络
│   │   │   └── agent.py                 # 训练目标与优化更新
│   │   ├── jepa/                        # 共享编码器与工具
│   │   ├── buffers/                     # 经验与轨迹回放缓冲区
│   │   ├── envs/                        # 环境封装与专家数据加载
│   │   ├── evaluations/                 # 动作跟踪与目标到达评估
│   │   └── latent_reconstruction.py     # 根位姿与身体重建头
│   ├── config/                          # 训练、机器人、观测、奖励等配置
│   ├── envs/                            # 人形机器人仿真任务
│   ├── simulator/                       # Isaac Sim 与 MuJoCo 后端
│   ├── data/
│   │   ├── data_manifest.json           # 分卷清单与文件字节数
│   │   ├── packed/                      # 压缩动作数据分卷
│   │   ├── lafan_locomotion_mix2000.pkl  # 训练动作（本地解压生成）
│   │   ├── lafan_29dof.pkl               # 评估动作（本地解压生成）
│   │   └── robots/g1/                   # G1 机器人描述与几何资源
│   └── utils/                           # 动作加载与仿真工具
├── .venv/                               # uv sync 创建
└── outputs/                             # 训练日志与检查点
```

## 致谢与许可证

BeJEPA 基于 [BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero) 构建，并使用 Isaac Sim、Isaac Lab 和 MuJoCo 进行仿真。重新分发时请保留上游代码与资源的署名和许可证声明。

仓库附带 [CC BY-NC 4.0 许可证](LICENSE)。动作数据、机器人资源及第三方依赖仍受其各自许可证与使用条款约束。

## 引用

如果本项目对您的研究有帮助，请引用以下论文：

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
