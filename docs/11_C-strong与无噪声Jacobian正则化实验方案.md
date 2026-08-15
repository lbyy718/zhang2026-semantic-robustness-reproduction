# C-strong 与无噪声 Jacobian 正则化实验方案

> 状态（2026-08-14）：seed 2026 已完成，但 CS0/CSJ 发生表示坍塌，联合技术门未通过；正式 Jacobian/PGA 诊断和 2027/2028 未启动。详见 [C-strong 与 Jacobian 正则化实验结果](./12_C-strong与Jacobian正则化实验结果.md)。

## 目的

这组实验只在同一个强分类架构内部比较三种训练处理，避免把模型容量差异误当成噪声训练效应：

- `CS0`：无训练 AWGN、无 Jacobian 正则化；
- `CS1`：10 dB latent AWGN 训练；
- `CSJ`：无训练 AWGN，使用 latent failure-score Jacobian penalty。

模型均为从头训练的 CIFAR ResNet-18，经 `512 -> 768` 投影和单位功率归一化进入信道，再由 `768 -> 512 -> 10` 分类头输出 logits。它不是外部冻结 ResNet，也没有借用预训练权重。

## 已锁定设置

- 数据：CIFAR-10，固定 45k/5k train/validation 划分，训练使用 RandomCrop 与 HorizontalFlip；
- 优化：batch 64、SGD、lr 0.05、momentum 0.9、weight decay 5e-4、cosine；
- 停止：最多 200 epoch，patience 30；
- 选择：按固定 10 dB validation cross-entropy 选择最佳 checkpoint；
- CSJ：第 1 epoch 仅 warm-up，同时测量 CE 和 penalty；从第 2 epoch 起固定使用 `lambda = 0.1 * mean_CE / mean_penalty`；
- test：noiseless 和 0--20 dB clean 曲线；
- 机制诊断：1000 张固定类别平衡图片、3 个共享噪声重复；记录 margin、failure gradient、局部谱、normalized spectrum、linearized distance 和 PGA；
- `normalized spectrum = raw spectral norm / centered-logit RMS`，用于排除仅缩小 logits 的伪平滑。

## 运行入口

先跑 seed 2026 三臂训练：

```powershell
python scripts/run_cstrong_pilot.py --device cuda --workers 0
```

训练完成后跑 9 个诊断作业：

```powershell
python scripts/run_cstrong_diagnostics.py --device cuda --workers 0
```

最后一次性分析、画图并生成中文结果：

```powershell
python scripts/analyze_cstrong.py
```

训练通过技术门槛后，无论方向是否有利，补跑 seed 2027/2028：

```powershell
python scripts/run_cstrong_pilot.py --seeds 2027 2028 --device cuda --workers 0
```

## 恢复与失败规则

- runner 使用 PID 锁与源码哈希快照；批次中途修改执行源码会被拒绝；
- 已完整校验的作业自动跳过；只允许恢复状态为 `running` 或 `interrupted` 的作业；
- 状态为 `failed` 的作业不自动重试，需先查明原因；
- 非空但不可验证的旧目录会阻止覆盖；
- 每个 epoch 的完整指标只写 CSV，控制台仅输出第 1、每 10 个 epoch 和结束记录。

## 技术门槛

- CS0/CS1 的 10 dB test accuracy 至少 75%；
- CSJ 至少 72%，且不比 CS0 低超过 3 个百分点；
- 诊断无 NaN，局部谱收敛率至少 95%；
- 每一个 pairwise-common-correct 条件集至少 500/1000；
- PGA 未成功样本以右删失保留，不能静默删除。

seed 2026 只作描述性 pilot。技术门槛通过只表示实验可解释，不表示研究假设已经成立。

## 产物位置

- 训练：`outputs/cstrong_pilot/{cs0,cs1,csj}_seed2026/`；
- 注册表：`outputs/cstrong_pilot/registry.csv`；
- 诊断：每臂目录下的 `diagnostics/repeat0..2/`；
- 小型结果和图：`results/cstrong_pilot_seed2026/`；
- 自动中文结论：`docs/12_C-strong与Jacobian正则化实验结果.md`。

旧 C-small、R0/R1/C0/C1 结果不会被覆盖，也不会被这套脚本重新解释。
