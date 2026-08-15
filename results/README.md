# 精选 CIFAR-10 复现证据

`results/` 只保存适合 Git 审查的小型图像实验产物。原始数据、checkpoint、控制台日志和完整本地输出由 `.gitignore` 排除。

## 当前内容

- `cifar10_quick/`：20 epoch quick 管线的 clean SNR 曲线、PGA 汇总、manifest 和曲线图。它证明端到端流程可运行，不是正式论文数值。
- `cifar10_formal/`：此前 1000 epoch 正式训练的逐轮日志和 run manifest。该运行最佳验证 MSE 为 `0.002412790957093239`（epoch 985），但目录目前没有正式 clean 曲线和正式攻击结果。
- `factorial_2x2/`：R0/R1/C0/C1 × 三个种子的正式任务—噪声对照，含训练、clean SNR 曲线和探索性 PGA 精简汇总。

因此，当前精选证据已经支持“噪声训练跨重建与分类改善普通信道性能，并提高探索性 PGA 所需功率”；仍不足以把该现象解释为认证鲁棒性、全局 smoothness 或语义通信特异效应。

## 新正式运行何时可以进入 results

只有满足以下条件，才从 `outputs/<run>/` 复制到新的精选结果目录：

1. 1000 行训练日志和完成态 `run_manifest.json` 齐全；
2. 明确记录最佳 epoch、最佳/最后验证 MSE、代码 commit 和配置；
3. clean PSNR–SNR 曲线覆盖预设 SNR，并说明信道重复次数；
4. 攻击结果同时报告成功率、失败样本处理、总能量和每信道使用功率；
5. 至少完成一次 PGA/C&W 小规模强度审计；
6. 结果文案没有把经验攻击功率写成认证鲁棒性或因果机制证明。

训练完成后的具体顺序见 `../docs/04_formal_training_evaluation.md`。论文未披露的实现选择见 `../ASSUMPTIONS.md`。
