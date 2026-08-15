# R0/R1/C0/C1 三种子批处理

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + run + validate
- Origin Date: 2026-08-12
- Verification Status: VERIFIED（12/12 正式训练完成，统一评估完成）
- Version Label: factorial_2x2_three_seed_batch_v2

> 2026-08-13 状态：本页所述 12 次训练已经全部完成，退出码均为 0。最终结果、图和方法限制见 [三种子正式结果](08_factorial_2x2_results.md)。下列命令保留用于复现或恢复，不表示仍需重新启动训练。

批处理脚本为 `scripts/run_factorial_batch.py`。默认使用三个独立训练种子：

```text
2026, 2027, 2028
```

每个种子依次完成四个实验单元：

| 单元 | 任务 | 训练信道 |
| --- | --- | --- |
| R0 | 图像重建 | 无噪声恒等信道 |
| R1 | 图像重建 | 10 dB AWGN |
| C0 | CIFAR-10 分类 | 无噪声恒等信道 |
| C1 | CIFAR-10 分类 | 10 dB AWGN |

运行顺序为：

```text
seed 2026: R0 -> R1 -> C0 -> C1
seed 2027: R0 -> R1 -> C0 -> C1
seed 2028: R0 -> R1 -> C0 -> C1
```

因此一共 12 次正式训练，且每完成一个种子就得到一个完整的 2×2 区组。单张 GPU 上严格串行，不同时训练多个模型。

## 复现命令

不带参数即可运行默认矩阵：

```powershell
python scripts/run_factorial_batch.py
```

等价的显式命令是：

```powershell
python scripts/run_factorial_batch.py --cells r0 r1 c0 c1 --seeds 2026 2027 2028 --device cuda
```

只检查任务矩阵而不训练：

```powershell
python scripts/run_factorial_batch.py --dry-run
```

## 结果位置

每次训练有独立目录：

```text
outputs/factorial/r0_seed2026/ ... r0_seed2028/
outputs/factorial/r1_seed2026/ ... r1_seed2028/
outputs/factorial/c0_seed2026/ ... c0_seed2028/
outputs/factorial/c1_seed2026/ ... c1_seed2028/
```

每个实验目录保存：

- `resolved_config.json`：实际运行的完整配置和基础配置哈希；
- `command.txt`：可复制的精确训练命令；
- `formal_train.stdout.log`、`formal_train.stderr.log`；
- `training_log.csv`；
- `checkpoint_best.pt`、`checkpoint_last.pt` 和周期 checkpoint；
- `run_manifest.json`。

批次级记录统一位于：

```text
outputs/factorial/factorial_2x2_three_seed_batch/
```

其中 `registry.csv` 记录 12 个单元的状态、PID、退出码和输出路径；`batch_manifest.json` 记录 Python 环境、Git commit、实验矩阵和执行策略。后台启动时，应将批处理进程自身的输出重定向为该目录下的 `batch.stdout.log` 与 `batch.stderr.log`。

## 与之前 R1 结果的关系

`outputs/image_cifar10_full_20260812/` 的 seed-2026 R1 使用 CIFAR-10 test split 逐 epoch 选择 checkpoint，是复现旧基线，不纳入本次公平 2×2 统计。

`outputs/image_cifar10_seed2027/` 是另一个 legacy R1，且只运行到 epoch 167，没有最终 manifest，判定为未完成，也不纳入 12 组。

本脚本中的 R1 使用与 R0/C0/C1 完全相同的固定 45k/5k train/validation 划分，所以必须重新训练。

## 恢复和失败策略

- 只有 `run_manifest.json`、1000 行 `training_log.csv`、best/last checkpoint 全部通过验证，才自动跳过；
- 发现已有但不完整的实验目录时立即停止，不覆盖现场；
- 任一子训练异常退出时停止整个批次；
- 不自动重试失败实验；
- 重新启动前先人工检查出错单元的 stderr 和文件完整性。

这保证“目录存在”不会被错误当成“训练完成”，也避免自动续跑掩盖异常。
