# COST2100 语义通信分轮检查结论

## 停止条件与结果

本轮在“出现可信且可观结果后停止”的条件下结束。基线与改进模型均训练 200 epoch，并在同一 20,000 条测试集上用 5 组独立 AWGN 噪声重复评估。

| SNR (dB) | 原始 3×3 基线 NMSE (dB) | 改进模型 NMSE (dB) | 改善 (dB) |
|---:|---:|---:|---:|
| 0 | -4.7947 | -6.6221 | 1.8274 |
| 5 | -8.2155 | -10.4509 | 2.2354 |
| 10 | -10.4541 | -13.1426 | 2.6886 |
| 15 | -11.5108 | -14.4804 | 2.9696 |
| 16 | -11.6274 | -14.6281 | 3.0007 |
| 20 | -11.9154 | -14.9966 | 3.0812 |

20 dB 的五次重复标准差分别为 0.00140 dB（基线）和 0.00136 dB（改进模型）。16--20 dB 的改善均超过 3 dB，曲线单调且重复波动远小于改善量，因此满足本次停止条件。

## 主要影响因素

1. **CSI 表示与物理带宽口径是首要因素。** 论文明确把 `32×32` angular-delay CSI 当作单通道图像，并定义 `N=256` channel uses。此前两通道实/虚部方案把源实维从 1024 增至 2048，却仍只发送 256 个实维，实际形成更强的 1/8 压缩并出现接近常数预测器的塌缩。
2. **全局瓶颈混合是最强的网络结构因素。** 纯局部卷积只能在有限角度-时延邻域内分配码字。加入残差式全局 MLP 后，100 epoch、10 dB 测试 NMSE 从同预算基线的 -10.068 dB 提升到 -11.488 dB。
3. **显式复符号计数继续带来明显收益。** 512 个 I/Q 实维配成 256 个复符号，总能量保持 256；100 epoch 时 10/20 dB 达到 -12.237/-13.698 dB。
4. **训练到 200 epoch 后跨过 3 dB 门槛。** 改进模型第 200 轮验证 MSE 为 `3.25708e-4`，原模型第 200 轮为 `6.52162e-4`，误差比约 0.4994。
5. **中间 Sigmoid、NMSE 损失、5×5 卷积核是次要因素。** 单独或早期组合的独立曲线改善大多低于 1 dB；逐码字去均值与取消 weight decay 均未改善最终最佳验证误差。

## 重要边界

- 这是一项可信的**消融改善结果**，不是完整论文曲线复现。论文 Fig. 7 在 20 dB 约为 -33 dB，本结果为 -14.997 dB，仍有很大差距。
- 改进模型在 20 dB 仍比论文攻击阈值 -16 dB 差约 1.00 dB，因此现在直接跑以 -16 dB 为目标的攻击会出现“clean 已失败/零扰动即成功”的退化，暂不应开始正式攻击实验。
- 训练仅使用一个确定性种子 `2026`；五次重复只覆盖信道噪声，不等价于五个独立训练种子。
- 全局 MLP、5×5 主卷积和“`2C_num` 为 I/Q 实维”的解释属于对论文未披露细节的实现选择，不能声称是作者原始代码。
- 本轮只涉及语义通信 DeepJSCC；传统 SSCC 代码未运行。

## Material Passport

- **论文材料：** `../paper.pdf`，核对了 Table I、CSI system setup、Eq. (41) 与 Fig. 7。
- **数据：** CsiNet 作者公开 COST2100 outdoor 数据；训练/验证/测试文件分别为 `DATA_Htrainout.mat`（1,473,870,791 bytes）、`DATA_Hvalout.mat`（442,171,698 bytes）、`DATA_Htestout.mat`（294,782,702 bytes），划分为 100,000/30,000/20,000。
- **任务表示：** 单通道 magnitude，训练集 min/max 为 `6.563280408e-7` / `0.5000000596`，验证和测试复用相同边界。
- **改进配置：** `configs/csi_cost2100_ablation_single_channel_mse_complex256_kernel5_global_mixing_200.json`；SHA-256 `5B42DD59F9B9AF72B89060A1B2B01166183CE9F01972436E1503265F21DBB960`。
- **最佳检查点：** `outputs/csi_cost2100_ablation_single_channel_mse_complex256_kernel5_global_mixing_200/checkpoint_best.pt`，epoch 200；SHA-256 `415E1ECEAA9DDCAEB5AFD8003D14E99D54B10214F547EE5142E400E17047352E`。
- **硬件/软件：** NVIDIA GeForce RTX 5060 Laptop GPU；PyTorch `2.13.0+cu130`；CUDA runtime `13.0`；环境 `zhang2026-gpu`。
- **训练：** AdamW，MSE，10 dB，batch 512，learning rate `1e-3`，weight decay `0.01`，seed 2026，200 epoch。
- **验证：** 20,000 个独立测试样本；SNR 0--20 dB 主曲线；最终可信性检查取 0/5/10/15/16/17/18/19/20 dB，每点 5 次独立信道噪声重复。
- **证据文件：** `csi_clean_nmse_comparison.csv`、`csi_clean_nmse_comparison.png`、两组 repeats5 `clean_metrics.csv`、训练日志及 `run_manifest.json`。

## 复现最终验证

```powershell
python scripts/plot_csi_ablation_comparison.py `
  --baseline outputs/csi_cost2100_baseline_epoch200_repeats5/clean_metrics.csv `
  --improved outputs/csi_cost2100_complex256_epoch200_repeats5/clean_metrics.csv `
  --output outputs/csi_cost2100_ablation_summary/csi_clean_nmse_comparison.png `
  --comparison-csv outputs/csi_cost2100_ablation_summary/csi_clean_nmse_comparison.csv
```
