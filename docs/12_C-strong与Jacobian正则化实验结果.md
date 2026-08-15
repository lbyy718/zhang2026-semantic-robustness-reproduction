# C-strong 与 Jacobian 正则化 seed 2026 Pilot 复盘

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-14T09:26:20.509746+00:00
- Verification Status: ANALYZED
- Version Label: cstrong_training_failure_v1

## 1. 验收结论

本轮三个训练作业都正常退出，checkpoint、CSV 和哈希校验完整；但是联合训练质量门没有通过，所以这不是一轮可用于比较鲁棒性来源的有效实验。

| 模型 | 完成 epoch | best epoch | 10 dB test accuracy | 门槛 | 是否通过 |
|---|---:|---:|---:|---:|---:|
| CS0 | 31 | 1 | 15.99% | 75% | 否 |
| CS1 | 169 | 139 | 90.74% | 75% | 是 |
| CSJ | 32 | 2 | 20.55% | 72%，且不低于 CS0 3 个百分点以上 | 否 |

因此，正式 1000 样本 Jacobian/PGA 诊断没有继续启动。CS0 与 CSJ 的准确率决定了 pairwise-common-correct 不可能达到预注册的 500/1000；强行分析只会比较失败模型。

## 2. 发生了什么

### CS0：无噪声基线发生表示坍塌

- epoch 1 已是最佳 checkpoint，validation accuracy 只有 16.34%；
- epoch 2 后迅速掉到接近随机猜测，epoch 31 因 patience 30 停止；
- last checkpoint 的 latent 跨样本坐标标准差均值为 `1.763e-09`；
- 分类头 ReLU 激活率为 `0.0000%`；
- 1000 张图全部预测成同一个类别。

这不是“过拟合”：训练准确率本身也下降到约 10%。它是优化/表示坍塌。

### CSJ：正则项走向退化的零 Jacobian 解

- warm-up 得到固定 `lambda = 897.057`；
- Jacobian penalty 从 epoch 1 的约 `2.41e-4`，在 epoch 2 降至约 `2.88e-5`，随后接近 0；
- last checkpoint 的 latent 跨样本坐标标准差均值为 `5.407e-09`，ReLU 激活率为 0；
- 模型通过输出几乎与输入无关的常量 logits，使 failure-score Jacobian 变成 0，但分类任务也一起失败。

所以“raw Jacobian 变小”本身不能证明获得了有用平滑；它可能只是网络死亡。这正是原计划要求同时看 clean performance 与 normalized spectrum 的原因。

### CS1：AWGN 臂成功训练

- best checkpoint 位于 epoch 139；
- test accuracy 为 noiseless `90.81%`、10 dB `90.74%`；
- best checkpoint 的 latent 跨样本坐标标准差均值为 `0.913`，分类头 ReLU 激活率为 `27.13%`；
- 预测覆盖 10 个类别，没有出现坍塌。

但这只能说明 AWGN 在当前优化设置下避免了坍塌并成功完成训练，不能说明它让一个“同等训练良好的 CS0”更加鲁棒，因为后者并不存在。

## 3. 为什么本轮不能回答原研究问题

原问题要求在 clean 性能合格的同架构模型之间比较 `CS1-CS0` 与 `CSJ-CS0`。实际比较变成了：

- CS1：成功训练的 90.74% 模型；
- CS0/CSJ：只会预测少数类别或单一类别的失败模型。

任何 margin、Jacobian 谱或 PGA 差异都会被训练成功与否支配。它既不能证明“AWGN 带来鲁棒性”，也不能证明“显式 Jacobian 正则化不能替代 AWGN”。

## 4. 失效原因判断

证据支持三个相互关联的问题：

1. `lr=0.05 + unit-power bottleneck + 768->512 ReLU head` 对无噪声臂存在严重优化坍塌；标准 ResNet 的超参数不能直接保证插入功率归一化瓶颈后仍稳定。
2. patience 30 在 cosine 学习率仍约 0.047 时就终止 CS0/CSJ；CS1 到约 epoch 50 后才开始持续改善，因此早停放大了不同臂的优化轨迹差异。
3. raw failure-score Jacobian penalty 有常量 logits / dead ReLU 的平凡最优方向。仅固定其初始损失占比为 10%，并不能排除这种退化。

这些是实现的实验设计问题，不是论文鲁棒性命题的正面或反面证据。

## 5. 下一轮应如何修正

先做独立于鲁棒性结论的“可训练性资格阶段”，不直接重跑三臂：

1. 只在 CS0 上做小型、预先锁定的学习率诊断，例如 0.005、0.01、0.02；固定跑完整 cosine，不使用 patience 提前截断。
2. 每 epoch 增加健康指标：latent 跨样本标准差、分类头激活率、预测类别数；出现 latent std 接近 0 或激活率接近 0 就标记 collapse。
3. 选出能让 CS0 稳定超过 75% 的单一训练协议后，原封不动地应用到 CS0/CS1/CSJ。
4. CSJ 改为尺度不变的 normalized-logit Jacobian，或同时约束 logit RMS；先用解析缩放测试证明不能靠把 logits 缩成 0 降低正则项。
5. 三臂都通过 clean 门槛后，才恢复 1000 样本、3 重复的 Jacobian/PGA 比较；随后再无条件补 2027/2028。

## 6. 统计与方法学谬误扫描

- Coverage: 11/11 checked。

| 类型 | 结论 | 说明 |
|---|---|---|
| Simpson's paradox | NOTE | 目前只有一个训练种子，无法检查跨 seed 方向反转。 |
| Ecological fallacy | CAUTION | 不能从一个 seed/一套架构外推到所有神经网络或语义通信系统。 |
| Berkson's paradox | NOTE | 尚未执行 clean-correct 条件分析；若以后执行，需披露选择机制。 |
| Collider bias | CAUTION | pairwise-common-correct 是处理后条件集，只能作机制描述。 |
| Base-rate neglect | NOTE | CIFAR-10 类别均衡，不代表部署类别先验。 |
| Regression to the mean | CAUTION | best-validation checkpoint 带选择效应；本轮更主要的问题是训练坍塌。 |
| Survivorship bias | NOTE | 3/3 作业均完成，没有作业流失；但不能只保留成功臂。 |
| Look-elsewhere effect | NOTE | 主要门槛已预锁定；坍塌诊断属于事后探索。 |
| Garden of forking paths | CAUTION | 下一轮超参数诊断必须先锁定搜索集合和选择规则，避免只保留有利配置。 |
| Correlation != causation | RED_FLAG | CS1 成功与 AWGN 同时出现，但 CS0 训练失败，不能把差异归因于鲁棒机制。 |
| Reverse causality | CAUTION | Jacobian 接近 0 与网络死亡同时出现，不能断言前者造成或解释鲁棒性。 |

## 7. 最终判断

- 数据和产物可信：三个作业完整且结果可复查。
- 原假设比较无效：技术质量门失败，整体置信等级为 `RED_FLAG`。
- 有效的新发现：当前 C-strong 实现暴露了无噪声优化坍塌，以及 raw Jacobian penalty 的退化解。
- 本轮不能进入 2027/2028，也不能作为 AWGN/语义通信鲁棒性的正面或证伪证据。

## 8. 产物

- `results/cstrong_pilot_seed2026/training_summary.csv`
- `results/cstrong_pilot_seed2026/training_milestones.csv`
- `results/cstrong_pilot_seed2026/clean_curve.csv`
- `results/cstrong_pilot_seed2026/collapse_diagnostics.csv`
- `results/cstrong_pilot_seed2026/training_analysis_summary.json`
- `results/cstrong_pilot_seed2026/training_curves.png`
- `results/cstrong_pilot_seed2026/clean_accuracy_by_snr.png`
- `results/cstrong_pilot_seed2026/csj_penalty_collapse.png`
