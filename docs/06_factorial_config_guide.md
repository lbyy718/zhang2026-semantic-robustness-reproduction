# R0/R1、C0/C1 配置与运行指南

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + validate
- Origin Date: 2026-08-12
- Verification Status: VERIFIED（配置解析、训练、clean、PGA 集成测试）
- Version Label: factorial_interface_v1

## 四组到底控制什么

| 配置 | 任务目标 | 训练信道 | 共同测试信道 | 组间主要差异 |
| --- | --- | --- | --- | --- |
| R0 | 图像重建，MSE | 恒等信道，无噪声 | AWGN，0–20 dB | Noise=off |
| R1 | 图像重建，MSE | 10 dB AWGN | AWGN，0–20 dB | Noise=on |
| C0 | CIFAR-10 分类，cross entropy | 恒等信道，无噪声 | AWGN，0–20 dB | Task=classification，Noise=off |
| C1 | CIFAR-10 分类，cross entropy | 10 dB AWGN | AWGN，0–20 dB | Task=classification，Noise=on |

四组使用同一个 DeepJSCC encoder、相同的 768 维实数 codeword、相同功率归一化、batch size、优化器、学习率、weight decay、训练轮数和固定 45k/5k 划分。`split_seed=2026` 在换训练种子时保持不变，保证各组看到相同的训练/验证样本。

重建 decoder 有 40,931 个参数，分类 head 有 41,297 个参数，相差约 0.89%；整个 R 模型有 50,375 个参数，整个 C 模型有 50,741 个参数。容量只能做到近似匹配，不能称为严格同构。

## 如何切换配置

不需要修改 Python 代码，只替换 `--config` 后的路径。

```powershell
# R0：重建 + 无噪训练
python run_semantic.py train --config configs/factorial/r0_reconstruction_noiseless_seed2026.json --device cuda

# R1：重建 + 10 dB AWGN 训练
python run_semantic.py train --config configs/factorial/r1_reconstruction_awgn10_seed2026.json --device cuda

# C0：分类 + 无噪训练
python run_semantic.py train --config configs/factorial/c0_classification_noiseless_seed2026.json --device cuda

# C1：分类 + 10 dB AWGN 训练
python run_semantic.py train --config configs/factorial/c1_classification_awgn10_seed2026.json --device cuda
```

每个配置已经包含不同的 `output_dir`，因此顺序运行不会相互覆盖。单张 GPU 建议串行跑四组，不要同时启动四个正式训练。

## clean、攻击与画图

以 R0 为例：

```powershell
python run_semantic.py clean --config configs/factorial/r0_reconstruction_noiseless_seed2026.json --checkpoint outputs/factorial/r0_seed2026/checkpoint_best.pt --device cuda

python run_semantic.py attack --config configs/factorial/r0_reconstruction_noiseless_seed2026.json --checkpoint outputs/factorial/r0_seed2026/checkpoint_best.pt --attacks pga cw --device cuda

python run_semantic.py plot --config configs/factorial/r0_reconstruction_noiseless_seed2026.json --clean-csv outputs/factorial/r0_seed2026/clean_metrics.csv --attack-csv outputs/factorial/r0_seed2026/attack_summary.csv --output outputs/factorial/r0_seed2026/curves.png
```

C0/C1 的命令形式完全相同，只需同时替换配置路径和 checkpoint/output 路径。分类 clean CSV 报告 accuracy、cross entropy 和 logit margin；分类攻击以真实类别 margin 首次降到 0 为成功，只在 clean-correct 样本上统计成功率。

## 如何换种子

一次独立训练需要同时修改两个字段：

1. 把配置顶层 `seed` 改为新值，例如 `2027`；
2. 把 `output_dir` 改为新目录，例如 `../../outputs/factorial/r0_seed2027`。

不要修改 `data.split_seed`，否则“训练随机性变化”和“数据划分变化”会混在一起。建议种子集合固定为 `2026, 2027, 2028`，可信后再扩到五个种子。

仓库还提供 `configs/image_cifar10_seed2027.json`。它是对第一次 legacy R1 的单因素换种子复跑，仍沿用 test split 选 checkpoint，仅用于估计旧基线的跨种子波动；它不属于采用规范 train/validation 划分的确认性 2×2。

## 运行前后检查

- 开始前确认目标输出目录不存在，避免把不同运行的日志混在一起。
- 正式比较统一使用 `checkpoint_best.pt`。
- 训练结束先检查 `training_log.csv` 与 `run_manifest.json`，再跑 clean 和攻击。
- R0/C0 的“无噪”只指训练；它们在 AWGN 测试曲线上的退化正是需要比较的结果。
- 每个 seed 才是独立实验重复；10,000 张图片和多次信道 realization 不能冒充 10,000 个独立模型。
