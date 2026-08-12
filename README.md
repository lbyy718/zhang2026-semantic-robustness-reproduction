# CIFAR-10 DeepJSCC 鲁棒性复现

这是 Zhang et al. (2026) *Unanticipated Adversarial Robustness of Semantic Communication* 的非官方、图像传输侧最小复现。仓库只保留：

- CIFAR-10 连续图像重建；
- Table I 对应的 DeepJSCC 编码器—AWGN 信道—解码器；
- clean PSNR–SNR 曲线；
- 接收端 latent 上的 PGA 与回归版 C&W 攻击；
- 论文 Eq. (10)–(12) 的数值计算与局部 Jacobian 诊断。

其他数据模态、传统 SSCC、原始数据集和完整训练 checkpoint 均不在本仓库范围内。本实现不是作者源码，也不宣称严格复现论文所有数值。

## 目录

```text
.
├─ semantic_robustness/   核心模型、信道、攻击、指标与运行流程
├─ configs/               正式与 quick 两份 CIFAR-10 配置
├─ tests/                 单元测试和微型端到端测试
├─ results/               可提交 Git 的精选图像实验证据
├─ docs/                  学习路线、论证审计和后续对照实验计划
├─ ASSUMPTIONS.md         论文未披露细节及本地补齐假设
├─ environment.yml        推荐 GPU 环境
└─ run_semantic.py        统一命令行入口
```

数据、checkpoint、日志和原始输出由 `.gitignore` 排除。CIFAR-10 会下载到 `data/cifar10/`，训练结果默认写入 `outputs/`。

## 环境

```powershell
conda env create -f environment.yml
conda activate zhang2026-gpu
```

本机已验证 Python 3.11、PyTorch 2.13.0+cu130、torchvision 0.28.0+cu130 与 NVIDIA GPU。也可在已有兼容环境中安装：

```powershell
python -m pip install -r requirements.txt
```

先运行无需下载数据的检查：

```powershell
python run_semantic.py smoke --device cpu
python -m unittest discover -s tests -v
```

## 快速闭环

quick 配置用于确认训练、clean 评估、攻击和绘图完整可运行；其结果不能代替正式实验。

```powershell
python run_semantic.py train --config configs/image_cifar10_quick.json --device cuda

python run_semantic.py clean `
  --config configs/image_cifar10_quick.json `
  --checkpoint outputs/image_cifar10_quick/checkpoint_best.pt `
  --device cuda

python run_semantic.py attack `
  --config configs/image_cifar10_quick.json `
  --checkpoint outputs/image_cifar10_quick/checkpoint_best.pt `
  --attacks pga `
  --device cuda

python run_semantic.py plot `
  --config configs/image_cifar10_quick.json `
  --clean-csv outputs/image_cifar10_quick/clean_metrics.csv `
  --attack-csv outputs/image_cifar10_quick/attack_summary.csv `
  --output outputs/image_cifar10_quick/semantic_curves.png
```

## 正式训练与评估

正式配置使用 CIFAR-10 的 50,000/10,000 划分、batch size 512、AdamW、学习率 `1e-3`、1000 epoch、带宽比 `1/4`、训练 SNR 10 dB，以及 15 dB 攻击失败阈值。

```powershell
python run_semantic.py train --config configs/image_cifar10.json --device cuda

python run_semantic.py clean `
  --config configs/image_cifar10.json `
  --checkpoint outputs/image_cifar10/checkpoint_best.pt `
  --device cuda

python run_semantic.py attack `
  --config configs/image_cifar10.json `
  --checkpoint outputs/image_cifar10/checkpoint_best.pt `
  --attacks pga cw `
  --device cuda

python run_semantic.py plot `
  --config configs/image_cifar10.json `
  --clean-csv outputs/image_cifar10/clean_metrics.csv `
  --attack-csv outputs/image_cifar10/attack_summary.csv `
  --output outputs/image_cifar10/semantic_curves.png
```

C&W 对全部图像、全部 SNR 点和完整二分搜索运行非常耗时。调试时先复制 quick 配置并减小 `attacks.max_samples`、`evaluation.snr_db`、`binary_search_steps` 与 `max_steps`。

## 输出解释

- `training_log.csv`：逐 epoch 训练/验证 MSE；
- `run_manifest.json`：设备、随机种子、最佳验证误差和完整配置；
- `clean_metrics.csv`：各 SNR 下的 MSE 与 PSNR；
- `attack_samples.csv`：逐样本 clean/attack 失真、成功标记、步数与功率；
- `attack_summary.csv`：各攻击和 SNR 下的成功率与成功样本平均功率；
- `semantic_curves.png`：clean PSNR 曲线和攻击功率曲线。

论文对攻击功率存在总平方 L2 能量与每信道使用平均功率混用风险。本实现同时记录 `attack_power_total_l2_sq` 和 `attack_power_per_channel_use`，绘图默认使用前者。C&W 使用满足约束含义的 `relu(D* + kappa - D)`；与论文公式的符号差异见 [ASSUMPTIONS.md](ASSUMPTIONS.md)。

## 学习与后续研究

- [代码学习路线](docs/LEARNING_PATH.md)
- [鲁棒性论证审计](docs/ROBUSTNESS_CLAIM_AUDIT.md)
- [任务—噪声—通信结构对照实验计划](docs/FACTORIAL_EXPERIMENT_PLAN.md)

这些文档把“复现论文现象”和“验证其机制归因”分开：现有图像曲线只能支持特定配置下的经验趋势，不能证明语义通信具有普适或认证鲁棒性。
