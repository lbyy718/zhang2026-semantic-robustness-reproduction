# C-strong 无噪声 Jacobian 正则化两种子结果

> 本报告由 `scripts/analyze_cstrong_formal.py` 从锁定训练与诊断产物生成。当前只有两个独立训练种子，只作描述性复盘，不进行显著性检验。

## 1. 实验范围

- 模型：同一 `resnet18_bottleneck` 架构。
- CS0：无 AWGN、无 Jacobian 正则化。
- CSJ：无 AWGN、加入 latent failure-score Jacobian penalty。
- 已完成种子：2026, 2027。
- 2028 两组按用户决定延期，因此本报告不是原计划的三种子终局分析。

## 2. Clean 性能

| 模型 | 两种子平均 10 dB test accuracy |
|---|---:|
| CS0 | 94.40% |
| CSJ | 93.31% |

逐种子准确率、最佳 epoch 和交叉熵见 `training_summary.csv`。

## 3. 共同正确样本上的局部与攻击比值

比值均为 `CSJ / CS0`。每个训练种子内部先对三个共享信道重复聚合，再对训练种子的对数比取平均。

| 指标 | CSJ / CS0 |
|---|---:|
| clean failure margin | 0.525x |
| failure gradient L2 | 0.378x |
| raw Jacobian spectrum | 1.003x |
| normalized Jacobian spectrum | 1.230x |
| linearized failure distance | 1.387x |
| PGA L2（两侧均成功的 complete cases） | 1.312x |

`normalized spectrum = raw spectrum / centered-logit RMS`，用于降低单纯 logits 缩放造成的伪平滑影响。

## 4. 解释规则

CSJ 提高了 PGA 距离，但 normalized spectrum 未下降；收益可能来自 margin 或非局部几何，不能归因于 Jacobian 平滑。

## 5. 结论边界

- 两个训练种子不足以支持可靠显著性检验或普适外推。
- pairwise-common-correct 是处理后的条件子集，可能存在幸存者/碰撞变量选择偏差。
- PGA 是单一固定攻击的经验估计，不是认证鲁棒性；complete-case 比值必须结合删失模式阅读。
- 本批次没有 CS1，不能直接比较 AWGN 训练与 Jacobian 正则化谁更优，也不能单独识别 AWGN 的机制。
- 即使 CSJ 改善局部谱和 PGA，也只能说明显式局部平滑在当前架构上足以产生相似现象。

## 6. 产物

- `results/cstrong_formal_lr001_two_seed/training_summary.csv`
- `results/cstrong_formal_lr001_two_seed/arm_seed_summary.csv`
- `results/cstrong_formal_lr001_two_seed/repeat_effects.csv`
- `results/cstrong_formal_lr001_two_seed/seed_effects.csv`
- `results/cstrong_formal_lr001_two_seed/aggregate_effects.csv`
- `results/cstrong_formal_lr001_two_seed/robust_curves.csv`
- `results/cstrong_formal_lr001_two_seed/clean_curves.csv`
- 三张 PNG 图和 `analysis_summary.json`
