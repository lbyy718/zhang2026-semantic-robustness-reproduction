# 语义通信侧代码复现

这是 Zhang et al. (2026) *Unanticipated Adversarial Robustness of Semantic Communication* 的语义通信侧独立复现。当前实现覆盖 DeepJSCC 图像传输、DeepJSCC massive-MIMO CSI 反馈、无攻击曲线、PGA、回归版 C&W、逐样本攻击功率统计，以及论文 Eq. (10)–(12) 的可执行理论计算。

> **仓库定位：**这是非官方、仅覆盖语义通信侧的概念与趋势复现。它不声称是作者源码，也不包含传统 SSCC、原始论文 PDF、数据集或完整训练 checkpoint。

传统 SSCC 相关的 BPG、LDPC、调制、vulnerable-set 和 GMS/DQN 暂未实现，代码入口也不会隐式调用这些部分。

## 仓库内容

- `semantic_robustness/`：模型、信道、攻击、指标、数据和运行时实现。
- `configs/`：正式、quick 与 COST2100 消融配置，索引见 [configs/README.md](configs/README.md)。
- `tests/`：单元与端到端小规模测试。
- `scripts/`：结果汇总和绘图脚本。
- `results/`：可提交 Git 的小型复现证据，见 [results/README.md](results/README.md)。
- `data/README.md`：数据布局、下载与校验说明；真实数据不进入 Git。

公开仓库不保存原始训练输出和中间 checkpoint。需要共享最佳权重时，应通过 GitHub Release 或科研数据仓库发布，并同时给出 SHA-256。

## 已实现内容

| 论文内容 | 本地实现 | 输出 |
| --- | --- | --- |
| Table I DeepJSCC | `semantic_robustness/model.py` | 图像 `N=768`、CSI `N=256` |
| Eq. (1) AWGN 信道 | `semantic_robustness/channel.py` | `r=|h|z+w` |
| Fig. 4 语义曲线 | `clean` 命令 | `clean_metrics.csv` |
| Eq. (35)–(36) PGA | `ProgressiveGradientAscent` | 逐样本阈值、步数、两种功率定义 |
| Eq. (38) C&W baseline | `CWRegressionAttack` | 修正符号后的最小范数搜索 |
| Fig. 5 语义曲线 | `attack` + `plot` | PGA/C&W `rho*` 曲线 |
| Fig. 7/8 语义曲线 | CSI 配置下的同一入口 | clean NMSE 与攻击功率 |
| Eq. (10)–(12) | `theory` 命令 | 语义系统功率下界 |

论文没有公开足够信息唯一确定源码。所有补齐项见 [ASSUMPTIONS.md](ASSUMPTIONS.md)，尤其是训练 SNR、卷积细节、CSI 复数表示、C&W 搜索和攻击功率归一化。

## 环境

推荐新建独立环境：

```powershell
conda env create -f environment.yml
conda activate zhang2026-gpu
```

若已有合适的 PyTorch 环境，也可使用：

```powershell
python -m pip install -r requirements.txt
```

首次检查不需要下载数据：

```powershell
python run_semantic.py smoke --task image --device cpu
python run_semantic.py smoke --task csi --device cpu
python -m unittest discover -s tests -v
```

本机已在 NVIDIA GeForce RTX 5060 Laptop GPU（8 GB、compute capability 12.0、驱动
591.91）上验证以下组合：Python 3.11.15、PyTorch 2.13.0+cu130、torchvision
0.28.0+cu130、CUDA runtime 13.0、cuDNN 9.2。实际的前向、反向传播和 AdamW
更新均已在 `cuda` 上通过。可复查：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

## 图像任务：CIFAR-10

配置严格使用论文的 50,000/10,000 划分、batch size 512、AdamW、学习率 `1e-3`、1000 epoch、带宽比 `1/4` 和攻击阈值 15 dB。当前已下载并验证 torchvision 官方 CIFAR-10 归档：

- `data/cifar10/cifar-10-python.tar.gz`
- 大小：170,498,071 字节
- MD5：`c58f30108f718f92721af3b95e74349a`
- 来源：torchvision 指向的 Toronto 原始数据；本机下载使用具有相同 MD5 的 [Zenodo 镜像](https://zenodo.org/records/10089977)

先跑约两分钟的真实 quick 配置：

```powershell
python run_semantic.py train --config configs/image_cifar10_quick.json --device cuda
python run_semantic.py clean --config configs/image_cifar10_quick.json `
  --checkpoint outputs/image_cifar10_quick/checkpoint_best.pt --device cuda
python run_semantic.py attack --config configs/image_cifar10_quick.json `
  --checkpoint outputs/image_cifar10_quick/checkpoint_best.pt --attacks pga --device cuda
python run_semantic.py plot --config configs/image_cifar10_quick.json `
  --clean-csv outputs/image_cifar10_quick/clean_metrics.csv `
  --attack-csv outputs/image_cifar10_quick/attack_summary.csv `
  --output outputs/image_cifar10_quick/semantic_curves.png
```

正式 1000-epoch 训练：

```powershell
python run_semantic.py train --config configs/image_cifar10.json --device cuda
```

得到 `outputs/image_cifar10/checkpoint_best.pt` 后，运行无攻击语义曲线：

```powershell
python run_semantic.py clean `
  --config configs/image_cifar10.json `
  --checkpoint outputs/image_cifar10/checkpoint_best.pt `
  --device cuda
```

运行论文的两种语义攻击：

```powershell
python run_semantic.py attack `
  --config configs/image_cifar10.json `
  --checkpoint outputs/image_cifar10/checkpoint_best.pt `
  --attacks pga cw `
  --device cuda
```

C&W 对 10,000 张图、21 个 SNR、9 次 `c` 搜索和每次最多 2000 步的完整运行非常耗时。调试时应复制配置，把 `attacks.max_samples`、`evaluation.snr_db`、`binary_search_steps` 和 `max_steps` 调小；正式结果再恢复论文规模。

绘制仅包含 DeepJSCC 的 Fig. 4/5 对应曲线：

```powershell
python run_semantic.py plot `
  --config configs/image_cifar10.json `
  --clean-csv outputs/image_cifar10/clean_metrics.csv `
  --attack-csv outputs/image_cifar10/attack_summary.csv `
  --output outputs/image_cifar10/semantic_curves.png
```

## CSI 任务：COST2100

论文使用 COST2100 outdoor-rural 300 MHz：`Nt=32`、`Nc=1024`，变换至 angular-delay 域后保留前 32 个 delay taps，并使用 100,000/30,000/20,000 划分。本机已下载 [CsiNet 作者公开的数据包](https://github.com/sydney222/Python_CsiNet)，并从 6,834,910,804 字节的完整 ZIP 中只解压 outdoor 三个文件：

完整 ZIP 的 SHA-256 为 `a6de66031db1f54018564927985fbece59e61b3fd5dc2a08b9ba10033c3c7ba5`，且 ZIP CRC 全量检查通过。

- `data/cost2100/original/DATA_Htrainout.mat`：`HT[100000,2048]`
- `data/cost2100/original/DATA_Hvalout.mat`：`HT[30000,2048]`
- `data/cost2100/original/DATA_Htestout.mat`：`HT[20000,2048]`

适配器按 CsiNet 的布局把 2048 维向量还原为实部/虚部两个 `32×32` 平面，减去其 `+0.5` 偏移，再取复数幅度并统一使用训练集 min-max 边界。论文目标模型是单通道 `32×32` 输入，因此“取幅度”属于明确记录的实现选择，见 [ASSUMPTIONS.md](ASSUMPTIONS.md)。

当前数据可直接运行 quick 配置：

```powershell
python run_semantic.py train --config configs/csi_cost2100_quick.json --device cuda
python run_semantic.py clean --config configs/csi_cost2100_quick.json `
  --checkpoint outputs/csi_cost2100_quick/checkpoint_best.pt --device cuda
python run_semantic.py attack --config configs/csi_cost2100_quick.json `
  --checkpoint outputs/csi_cost2100_quick/checkpoint_best.pt --attacks pga --device cuda
python run_semantic.py plot --config configs/csi_cost2100_quick.json `
  --clean-csv outputs/csi_cost2100_quick/clean_metrics.csv `
  --attack-csv outputs/csi_cost2100_quick/attack_summary.csv `
  --output outputs/csi_cost2100_quick/semantic_curves.png
```

quick 模型只训练 20 epoch，clean NMSE 尚未优于正式阈值 −16 dB，因此其攻击阈值设为 −2 dB，避免“未攻击即成功”的零功率退化。正式配置仍使用论文阈值 −16 dB。

如果另有原始复数 CSI `H[sample,1024,32]`，也可先转换：

如果已有原始复数 CSI `H[sample,1024,32]`，可先转换：

```powershell
python run_semantic.py prepare-csi `
  --input data/cost2100/raw_train.mat `
  --key H `
  --output data/cost2100/train_prepared.npy `
  --representation magnitude
```

对 validation/test 使用训练集转换时打印的 `minimum` 和 `maximum`：

```powershell
python run_semantic.py prepare-csi `
  --input data/cost2100/raw_test.mat `
  --key H `
  --output data/cost2100/test_prepared.npy `
  --representation magnitude `
  --normalization-min <训练集minimum> `
  --normalization-max <训练集maximum>
```

使用三个独立 `.npy` 文件时，把 CSI 配置中的 `path` 改成 `train_path`、`validation_path`、`test_path` 三项，并删除三个 `*_key` 项。正式配置已直接指向上述三个 `.mat`，无需预转换。

```powershell
python run_semantic.py train --config configs/csi_cost2100.json --device cuda
python run_semantic.py clean `
  --config configs/csi_cost2100.json `
  --checkpoint outputs/csi_cost2100/checkpoint_best.pt `
  --device cuda
python run_semantic.py attack `
  --config configs/csi_cost2100.json `
  --checkpoint outputs/csi_cost2100/checkpoint_best.pt `
  --attacks pga cw `
  --device cuda
```

CSI 攻击成功条件是线性 NMSE `>=10^(-16/10)`，结果表同时保留线性 NMSE 和 dB NMSE。

## 已生成的真实 quick 结果

- 图像训练：20 epoch，约 96 秒；10 dB 验证 MSE 从 0.03239 降到 0.00638。
- 图像 clean：PSNR 从 0 dB 信道下的 18.18 dB 升至 20 dB 信道下的 23.28 dB。
- COST2100 训练：20 epoch，约 76 秒；10 dB 验证 MSE 从 0.00624 降到 0.00100。
- COST2100 clean：NMSE 从 −3.82 dB 改善至 −9.48 dB。
- 两项 PGA 评估在各 11 个 SNR 点的成功率均为 100%。
- 曲线位于 `outputs/image_cifar10_quick/semantic_curves.png` 和 `outputs/csi_cost2100_quick/semantic_curves.png`；原始数值 CSV、checkpoint 和运行清单在各自输出目录中。

## 理论下界

例如直接计算论文 Eq. (11) 和 Eq. (12)：

```powershell
python run_semantic.py theory `
  --target-distortion 0.0316227766 `
  --clean-distortion 0.005 `
  --lipschitz 0.05 `
  --channel-uses 768 `
  --noise-variance 0.0630957344
```

`estimate_local_lipschitz` 还可用幂迭代估计 decoder 在给定接收点的局部 Jacobian 谱范数。它只是诊断量，不是 Eq. (4) 所需的全局 Lipschitz 证书。

## 结果语义

`attack_samples.csv` 是主审计文件。它记录每个样本的 clean/attacked distortion、质量、成功标记、步数、总平方 L2 功率和每信道使用功率。`attack_summary.csv` 只对成功样本计算平均功率；只有成功率为 100% 时才填写全样本平均，避免把失败样本当成零功率。

论文正文的 `rho` 记法存在总能量与每符号功率混用风险。本复现将 `attack_power_total_l2_sq=sum_i s_i²` 作为论文攻击曲线默认口径，同时保存 `attack_power_per_channel_use=mean_i s_i²` 供审计。
