# CIFAR-10 任务 × 噪声 2×2 精选结果

本目录保存 12 次正式训练（R0/R1/C0/C1 × seeds 2026/2027/2028）的可审查精简证据。checkpoint、逐样本攻击数据和完整日志仍在本地 `outputs/factorial/`，不进入 Git。

## 设计

| 单元 | 任务 | 训练信道 |
| --- | --- | --- |
| R0 | 重建 | 无噪声 |
| R1 | 重建 | 10 dB AWGN |
| C0 | 分类 | 无噪声 |
| C1 | 分类 | 10 dB AWGN |

共同设置：CIFAR-10 固定 45k/5k train/validation、768 维功率归一化 latent、相同 encoder、三个训练种子。clean 测试使用 10,000 张图、0–20 dB 步长 2 dB、每 SNR 三次信道重复。

## 文件

- `training_summary.csv`：12 次训练的最佳/末轮指标；
- `clean_summary.csv`：每单元和 SNR 的跨种子均值、SD 和 t 区间；
- `paired_clean_effects.csv`：同一种子的 R1−R0、C1−C0 效应；
- `pga_10db_seed_summary.csv`：探索性 10 dB、128 样本 PGA 汇总；
- `pga_10db_paired_effects.csv`：噪声开关前后的攻击功率比；
- `pga_10db_classification_common_correct_effects.csv`：C0/C1 共同 clean-correct 样本的公平比较；
- `analysis_summary.json`：机器可读核心结论；
- `SHA256SUMS.txt`：本次精选快照的文件校验值；
- 三张 clean/training 图和一张 PGA 诊断图。

完整解释和限制见 `../../docs/08_factorial_2x2_results.md`。PGA 仅为探索性求解器审计，不是认证鲁棒性，也没有完成 C&W、多重重启、阈值匹配或独立攻击交叉验证。
