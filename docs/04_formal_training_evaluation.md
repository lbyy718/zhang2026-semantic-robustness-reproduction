# CIFAR-10 正式训练结束后的验收与决策清单

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-12
- Verification Status: UNVERIFIED（训练完成并执行下述评估后再更新）
- Version Label: formal_run_followup_v1

## 当前正式复训

- **目的：**确认精简后的图像-only代码仍能稳定训练，并建立后续因果对照实验的 R1（连续重建＋10 dB 噪声训练）基线。
- **启动代码版本：**Git commit `ce40601`。
- **配置：**`configs/image_cifar10.json`。
- **随机种子：**2026。
- **训练预算：**1000 epoch。
- **输出目录：**`outputs/image_cifar10_full_20260812/`。
- **完成标志：**训练进程退出码为 0，并生成非空的 `run_manifest.json`、`training_log.csv`、`checkpoint_best.pt` 和 `checkpoint_last.pt`。

该记录只说明启动条件，不自动表示训练已完成。训练过程中已有 checkpoint 和日志属于中间产物。

本轮进程启动后，仓库才把 manifest 的 `scope` 文案从宽泛的 `semantic-only` 改成 `cifar10-image-only`。由于运行中的 Python 已加载旧代码，本轮结束时的 manifest 可能仍保留旧文案；这只影响元数据标签，不影响模型计算或指标。精选归档时应注明这一点并统一标签。

## 1000 epoch 训练本身可以回答什么

训练完成后，单看训练日志与 checkpoint，可以合理回答：

1. 当前 DeepJSCC 实现在 10 dB AWGN 下能否优化到稳定的 CIFAR-10 重建误差。
2. 训练和验证 MSE 是否继续下降、是否出现明显过拟合或后期振荡。
3. 最佳模型出现在哪个 epoch，以及 1000 epoch 是否明显超过必要训练预算。
4. 在相同代码、种子和环境下，正式训练是否具有可重复性。
5. 精简代码是否仍与已有正式 checkpoint 的网络尺寸和功率约束一致。

已有精选正式运行的参考值是：最佳验证 MSE `0.002412790957093239`，出现在 epoch 985；epoch 1000 验证 MSE 为 `0.0024818958774209024`。新运行应报告相对差异，而不是只说“差不多”。若同种子结果相差超过 5%，先检查环境、数据、代码版本和确定性设置。

## 训练本身不能回答什么

即使验证 MSE 很低，也不能仅凭训练完成声称：

- 已经复现论文的鲁棒性曲线；
- 语义通信比普通神经网络更鲁棒；
- AWGN 训练导致 decoder 的全局 Lipschitz 常数变小；
- PGA/C&W 找到了真正的最小攻击扰动；
- 观察到的攻击功率来自“语义通信”，而不是重建阈值、clean margin、噪声训练或通信瓶颈。

这些结论至少还需要 clean SNR 曲线、攻击评估、攻击强度审计和对照组。

## 训练结束后的第一轮验收

在仓库根目录执行：

```powershell
$formalOutput = "outputs/image_cifar10_full_20260812"
Test-Path "$formalOutput/run_manifest.json"

$rows = Import-Csv "$formalOutput/training_log.csv"
$best = $rows | Sort-Object {[double]$_.validation_mse} | Select-Object -First 1
$last = $rows | Select-Object -Last 1
$best
$last
```

检查：

- 是否恰好有 1000 行训练记录；
- 是否存在 NaN、Inf、突然发散或长时间停滞；
- 最佳验证 MSE、最佳 epoch 和最后一轮 MSE；
- `train_mse` 与 `validation_mse` 的差距；
- 最佳 checkpoint 是否早于最后 checkpoint。

## 第二轮：正式 clean 曲线

```powershell
$formalOutput = "outputs/image_cifar10_full_20260812"

python run_semantic.py clean `
  --config configs/image_cifar10.json `
  --checkpoint "$formalOutput/checkpoint_best.pt" `
  --output $formalOutput `
  --device cuda
```

clean 曲线至少回答：

- PSNR 是否随测试 SNR 上升；
- 低 SNR 区是否由信道噪声主导；
- 高 SNR 区是否出现压缩/模型误差平台；
- 各 SNR 下 clean PSNR 距离 15 dB 失败阈值有多远。

正式机制比较时应复制配置并把 `evaluation.channel_repeats` 提高到至少 5；一次信道 realization 只适合初步曲线。

## 第三轮：先做攻击审计，再做全量攻击

不要直接启动 10,000 张图、21 个 SNR 点和完整 C&W。先用正式 checkpoint 配合 quick 预算检查求解器：

```powershell
$formalOutput = "outputs/image_cifar10_full_20260812"

python run_semantic.py attack `
  --config configs/image_cifar10_quick.json `
  --checkpoint "$formalOutput/checkpoint_best.pt" `
  --output "$formalOutput/attack_audit_pga" `
  --attacks pga `
  --device cuda

python run_semantic.py attack `
  --config configs/image_cifar10_quick.json `
  --checkpoint "$formalOutput/checkpoint_best.pt" `
  --output "$formalOutput/attack_audit_cw" `
  --attacks cw `
  --device cuda
```

审计重点：

- 未攻击样本是否已经低于 15 dB；
- 每个 SNR 的攻击成功率；
- 失败样本是否被排除在条件均值之外；
- 总平方 L2 功率和每信道使用功率是否同时保留；
- PGA 步长、迭代数、二分精修和 C&W 搜索预算变化后，`rho*` 是否稳定。

只有小规模审计没有暴露零功率退化、低成功率或数量级敏感性，才值得扩大 PGA 样本量。完整 C&W 应视为独立的高成本验证，不是训练结束后的默认下一条命令。

## 可能得到的有用结论层级

| 证据 | 可以采用的措辞 | 仍不能声称 |
| --- | --- | --- |
| 训练/验证 MSE 稳定下降 | 图像 DeepJSCC 基线训练成功 | 已复现鲁棒性 |
| clean PSNR–SNR 趋势与论文定性一致 | clean 趋势得到复现 | 数值严格一致 |
| PGA/C&W 均需非平凡扰动且成功率充分 | 在当前实现与攻击预算下复现了经验鲁棒现象 | 找到全局最小扰动 |
| 攻击结果对重启、步长和阈值稳定 | 现象不太可能只是单一求解器设置造成 | 机制已经确定 |
| R1 相比匹配的无噪 R0 更强 | 支持噪声训练贡献 | 语义通信独有机制 |
| R0/R1 与 C0/C1 的交互跨种子稳定 | 可以讨论任务与噪声的交互 | 普适或认证鲁棒性 |

## 下一步实验顺序

当前 1000 epoch 模型是 R1 基线。资源最有效的顺序是：

1. 完成 R1 的训练日志、clean 曲线和小规模强度审计。
2. 实现显式无噪训练开关，建立匹配的 R0；不要把极高 SNR 当成确认性实验中的“严格无噪”。
3. 先对 R0/R1 跑 quick，再以相同种子和预算做正式重建噪声对照。
4. 在公共 encoder/latent/攻击位置上加入分类头，形成 C0/C1。
5. 四组全部 quick 通过后，使用种子 2026、2027、2028 运行核心 2×2。
6. 最后增加局部 Jacobian、阈值敏感性、多重重启和 clean-matched 分析。

这样可以先用一个额外模型直接检验论文最核心的 noisy-channel-training 解释，再承担分类分支和四组正式训练的成本。
