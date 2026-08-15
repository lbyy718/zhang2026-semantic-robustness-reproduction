# R1（seed 2026）正式基线验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-12
- Verification Status: VERIFIED（训练轨迹复现）；ANALYZED（clean 曲线）
- Version Label: r1_seed2026_validation_v1

## 实验身份

- **组别：**R1，连续重建＋10 dB AWGN 训练。
- **命令：**`python run_semantic.py train --config configs/image_cifar10.json --output outputs/image_cifar10_full_20260812 --device cuda`。
- **代码版本：**启动时 Git commit `ce406012a0b51d302bda4e983468d1d497bb6ee1`。
- **环境：**Python 3.11.15，PyTorch 2.13.0+cu130，torchvision 0.28.0+cu130，NVIDIA GeForce RTX 5060 Laptop GPU。
- **训练规模：**CIFAR-10，batch size 512，1000 epoch，约 98,000 次参数更新。
- **模型选择：**10 dB 验证 MSE 最小。
- **完成状态：**1000/1000 epoch；`checkpoint_best.pt`、`checkpoint_last.pt`、每 100 epoch checkpoint、训练日志和 run manifest 均存在。

## 文件校验

| 文件 | SHA-256 |
| --- | --- |
| `configs/image_cifar10.json` | `B47AB412F268E5DE1B1C0EBE46C44315233D98B1A34B5DD47E660B75CF64B433` |
| `outputs/image_cifar10_full_20260812/checkpoint_best.pt` | `9675EA2BA3630DD714D42D02A1F0E618BFAC242F56551102FE4011310B595EA2` |
| `outputs/image_cifar10_full_20260812/training_log.csv` | `20E12ADB7C73F66F71F70B2973B35FC8B1C7F25044DD684F5A5011F6C7908D4B` |
| `outputs/image_cifar10_full_20260812/clean_metrics.csv` | `20B24950FC4AFA783F7E903B90AE66920C44D7D96138670757229ED2A5504947` |

## 训练结果

| 指标 | 数值 | 解释 |
| --- | ---: | --- |
| epoch 1 验证 MSE | 0.03216748 | 初始重建误差 |
| 最佳验证 MSE | 0.00241279 | epoch 985 |
| epoch 1000 验证 MSE | 0.00248190 | 比最佳值高 2.86% |
| 验证 MSE 降幅 | 92.50% | 相对 epoch 1 |
| 最佳 MSE 对应的聚合 PSNR | 26.17 dB | `-10 log10(mean MSE)`；不是逐样本平均 PSNR |
| 总训练时间 | 5278.14 秒（约 88.0 分钟） | 不作为复现判据 |

训练和验证误差都持续下降，最佳点靠近训练末期，没有出现明显的 train–validation 分叉。最后 15 个 epoch 有正常随机波动，因此应使用 `checkpoint_best.pt`，而不是默认使用 epoch 1000。

### 同种子可重复性

本轮与 `results/cifar10_formal/training_log.csv` 中此前 seed 2026 正式运行比较：

- 两边都恰好有 1000 个 epoch；
- 1000 个验证 MSE 逐轮完全相同；
- 训练 MSE 最大绝对差约 `1.02×10^-9`，最大相对差约 `2.42×10^-8`；
- 最佳 epoch 均为 985，最佳和最后验证 MSE 完全一致；
- 本轮墙钟时间比旧运行短约 11.78%，时间差不影响数值复现。

因此，在相同 seed、数据、硬件环境和确定性设置下，当前重建训练轨迹可判为可重复。

## clean PSNR–SNR 结果

正式最佳 checkpoint 已在 10,000 张测试图、SNR 0–20 dB（步长 1 dB）上完成一次 clean 评估：

| SNR | 平均 MSE | 平均逐样本 PSNR |
| ---: | ---: | ---: |
| 0 dB | 0.01759694 | 18.15495 dB |
| 10 dB | 0.00241275 | 26.67257 dB |
| 20 dB | 0.00157095 | 28.70047 dB |

- 21 个 SNR 点的平均 PSNR 严格单调上升；
- 0→10 dB 时 PSNR 增加 8.52 dB；
- 10→20 dB 时只增加 2.03 dB，显示高 SNR 区逐渐进入模型/压缩误差平台；
- 所有 SNR 点的平均 PSNR 都高于论文的 15 dB 失败阈值，因此没有“平均 clean 状态已经攻击成功”的退化。

这支持“当前 DeepJSCC 基线呈现随信道 SNR 提升而平滑改善、随后趋于平台”的定性 clean 趋势。由于当前每个 SNR 只有一次信道 realization，正式不确定性分析仍需提高 `channel_repeats`。

clean 命令的模型计算和 21 行输出均完成，但 Conda 包装器在回显含 Unicode 符号的 stdout 时发生 GBK `UnicodeEncodeError`。CSV/JSON 完整，因此它被记录为包装器异常，不判为评估失败。

## 当前允许的结论

1. 图像 DeepJSCC R1 基线能稳定学会 CIFAR-10 重建。
2. 同种子正式训练轨迹在当前环境中可重复。
3. clean PSNR 随测试 SNR 单调提升，并在高 SNR 区趋于平台；该定性趋势得到复现。
4. 当前结果建立了后续 R0/R1、C0/C1 因果对照的 R1 参照。

## 当前不允许的结论

- 尚未在这次正式 checkpoint 上完成 PGA/C&W 强度审计，不能声称正式攻击曲线已经复现。
- 单个 seed 不能估计跨初始化方差。
- 低 MSE 不能推出全局 Lipschitz 常数 `G` 很小。
- 单一 R1 不能区分噪声训练、重建任务、通信瓶颈和 clean margin 的贡献。
- 当前训练过程使用 CIFAR-10 test split 做逐 epoch 验证和最佳 checkpoint 选择，存在测试集参与模型选择的问题；后续确认性 2×2 应使用固定 train/validation holdout，并只在最终阶段查看 test。

## 11 类统计/方法谬误扫描

- **Coverage：11/11 checked。**

| 类型 | 严重度 | 本轮判断 | 后续处理 |
| --- | --- | --- | --- |
| Simpson 悖论 | NOTE | 当前没有分组效应推断；无法从聚合曲线判断类别内趋势 | 分类别报告可作探索分析 |
| 生态谬误 | NOTE | 没有从组均值推断单样本鲁棒性 | 保留逐样本攻击文件 |
| Berkson 悖论 | NOTE | 未按结果筛选训练样本 | 无 |
| Collider bias | NOTE | 当前没有回归控制变量 | 未来不要按同时受 clean 与攻击影响的变量筛选 |
| Base-rate neglect | NOTE | clean 评估不是诊断分类问题 | 分类实验同时报告 clean-correct 基率 |
| Regression to mean | NOTE | 没有按极端初始值选择本轮 | 无 |
| Survivorship bias | CAUTION | 攻击阶段若只汇总成功样本会产生幸存者偏差 | 必须同时报告成功率和失败处理 |
| Look-elsewhere effect | CAUTION | 未来多 SNR、多阈值、多攻击会形成多重比较 | 预先区分主比较与探索比较 |
| Garden of forking paths | CAUTION | 结构、阈值和攻击预算仍有研究者自由度 | 在正式 2×2 前冻结配置和停止规则 |
| Correlation ≠ causation | RED_FLAG | 单一 R1 的收敛/clean 曲线不能证明噪声导致鲁棒性 | 完成 R0/R1、C0/C1 因子对照 |
| Reverse causality | NOTE | 训练条件先于结果，不是主要问题；机制中介仍未识别 | 测局部 Jacobian 与攻击功率，但不把相关当因果 |

## 下一步

1. 启动 seed 2027 的同条件 R1，开始估计跨种子变化。
2. 把 R0/R1/C0/C1 的确认性训练改为固定 train/validation holdout。
3. 四组先跑 quick，确认分类 head、无噪开关、clean 指标和 latent 攻击接口。
4. 在正式 checkpoint 上先做小规模 PGA/C&W 强度审计，再决定全量攻击预算。
