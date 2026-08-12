# 2026 Zhang 语义通信复现代码学习路线

## 学习目标

完成本路线后，应当能够独立回答以下问题：

1. CIFAR-10 图像如何进入 DeepJSCC。
2. 编码器输出为何需要功率归一化，`N=768` 如何得到。
3. AWGN 在训练、clean 评估和攻击评估中分别怎样出现。
4. PGA/C&W 在哪里加扰动，怎样判断成功，怎样统计攻击功率。
5. 论文 Eq. (10)–(12) 的代码做了什么，以及它没有证明什么。
6. 哪些结果是正式训练，哪些只是 quick 管线检查，哪些属于本地补齐假设。

建议遵循“先运行、再追踪数据流、最后读理论”的顺序。不要一开始逐行阅读 `runtime.py`，否则容易被训练、评估和文件输出细节淹没。

## 阶段 0：建立复现边界

先读：

1. `README.md`：已经实现的论文部分、命令和已有结果。
2. `ASSUMPTIONS.md`：论文未披露而本实现必须补齐的细节。
3. `configs/README.md`：正式、quick 和消融配置的关系。
4. `results/README.md`：哪些结果可以作为现阶段证据。

这一阶段要牢牢记住：本项目是“语义通信侧概念与趋势复现”，不是作者官方源码，也没有实现传统 SSCC。当前结果不能独立验证论文的“14 倍/16 倍优于传统通信”比较。

### 阶段验收

不用看代码，先用自己的话写出：

- 已实现范围；
- 未实现范围；
- 三个最重要的实现假设；
- 为什么不能把 quick 曲线当正式论文结果。

## 阶段 1：运行最小闭环

先执行不需要真实数据的检查：

```powershell
conda activate zhang2026-gpu
python run_semantic.py smoke --device cpu
python -m unittest discover -s tests -v
```

然后用 quick 配置完成一次图像闭环：

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

### 阶段验收

确认以下文件的作用：

| 文件 | 应能说明的内容 |
| --- | --- |
| `training_log.csv` | 每轮训练/验证误差是否收敛 |
| `run_manifest.json` | 训练使用的配置、随机种子、设备和最佳指标 |
| `clean_metrics.csv` | SNR 改变时 clean PSNR 的趋势 |
| `attack_samples.csv` | 每个样本是否成功、步数、失真和扰动功率 |
| `attack_summary.csv` | 每个 SNR 的成功率和成功样本平均功率 |
| `semantic_curves.png` | clean 曲线和攻击功率曲线 |

## 阶段 2：从命令行入口追踪调用关系

按下面顺序阅读：

```text
run_semantic.py
  → semantic_robustness/cli.py
  → semantic_robustness/config.py
  → semantic_robustness/runtime.py
```

重点不是记住 argparse，而是知道每条命令最终调用哪个函数：

| 命令 | 核心运行函数 |
| --- | --- |
| `train` | `runtime.train()` |
| `clean` | `runtime.evaluate_clean()` |
| `attack` | `runtime.evaluate_attacks()` |
| `plot` | `runtime.plot_results()` |
| `theory` | `theory.py` 中的三个边界函数 |
| `smoke` | 小规模前向、反向传播和更新检查 |

同时阅读 `config.py`，理解：

- 相对路径相对于 JSON 配置文件解析；
- CIFAR-10 配置的信道使用数被校验为 768；
- 配置中的带宽比必须与网络 latent 维度一致。

### 阶段验收

从一条 `train` 命令出发，在纸上画出它最终如何建立数据集、模型、信道、优化器、checkpoint 和日志。

## 阶段 3：理解 DeepJSCC 数据流和约束

阅读顺序：

1. `semantic_robustness/data.py`
2. `semantic_robustness/model.py`
3. `semantic_robustness/channel.py`
4. `semantic_robustness/metrics.py`

需要掌握的主链路：

```text
x
→ encoder(x)
→ PowerNormalizer
→ z
→ AWGNChannel(z, snr_db)
→ r
→ decoder(r)
→ x_hat
→ MSE / PSNR
```

### 图像任务

- 输入为 `[B,3,32,32]`。
- 带宽比为 `1/4`，得到 768 个信道使用。
- 训练损失是 MSE，clean 质量使用 PSNR。
- 攻击成功阈值是 PSNR 不高于 15 dB，即 MSE 不低于约 `0.03162`。

### 功率归一化

重点阅读 `PowerNormalizer`。需要区分：

- 编码信号总能量；
- 每个信道使用的平均能量；
- 攻击总平方 L2 功率；
- 每信道使用攻击功率。

这几个口径不能混用，否则攻击曲线可能相差信道维数倍。

### 阶段验收

利用 `tests/test_core.py` 验证并解释：

- 输入/输出形状；
- 带宽比；
- 功率约束；
- PSNR 阈值与 MSE 阈值转换。

## 阶段 4：理解训练和 clean 评估

精读 `runtime.py` 的：

- `build_model()`；
- `build_dataset()`；
- `_sample_train_snr()`；
- `train()`；
- `_evaluate_dataset()`；
- `evaluate_clean()`。

重点问题：

1. 训练 SNR 是固定值还是区间采样。
2. 验证模型依据哪一个指标保存最佳 checkpoint。
3. 测试曲线每个 SNR 使用多少样本、多少次信道重复。
4. 随机种子是否只控制初始化，还是也控制数据和信道噪声。
5. clean 曲线的改善来自信道噪声减小，还是压缩误差平台。

### 动手练习

只改 JSON，不改代码，分别以 0、5、10、15 dB 固定训练 SNR 训练 quick 模型，观察：

- 训练 SNR 附近性能；
- 跨 SNR 泛化；
- clean 曲线平台；
- 后续攻击功率。

这个练习也是后续“噪声训练是否导致鲁棒性”实验的预备步骤。

## 阶段 5：理解攻击和统计口径

阅读 `semantic_robustness/attacks.py`：

### PGA

```text
r = channel(encoder(x))
s = 0
r_adv = r + s
x_adv = decoder(r_adv)
迭代增大 distortion(x, x_adv)
达到 D* 时停止
```

需要理解：

- 攻击扰动加在接收 latent 上；
- 梯度按样本归一化；
- 论文步长 `alpha=0.1` 不是扰动预算；
- “首次越过阈值”只是经验解，不保证得到全局最小扰动；
- 多重随机重启和最后二分精修会影响 `rho*`。

### C&W 回归攻击

注意本实现使用满足约束含义的修正版 hinge。需要同时看 `ASSUMPTIONS.md`，不要把修正版结果称为对作者未知源码的严格复现。

### 统计口径

主审计文件是 `attack_samples.csv`。报告结果时至少同时给出：

- 攻击成功率；
- 成功样本平均/中位攻击功率；
- 分位数；
- 全体样本口径；
- 失败样本处理方法；
- 总能量和每信道使用功率。

不能把失败样本静默记成零功率，也不能只展示条件成功均值而隐藏成功率。

## 阶段 6：最后阅读理论代码

阅读 `semantic_robustness/theory.py`，区分三类内容：

1. 论文公式的字面计算；
2. 数值合法性检查；
3. 局部 Jacobian 谱范数诊断。

`estimate_local_lipschitz()` 估计的是给定接收点附近的局部 Jacobian 谱范数，不是论文假设的全局 Lipschitz 常数，也不是认证鲁棒性下界。详细逻辑问题见 `docs/ROBUSTNESS_CLAIM_AUDIT.md`。

### 阶段验收

应能清楚解释以下三句话为什么不同：

- “这个有限参数网络是 Lipschitz 的。”
- “这个网络的全局 Lipschitz 常数很小。”
- “这个网络在数据分布附近的局部 Jacobian 较小。”

## 阶段 7：用测试反向理解设计意图

最后阅读：

- `tests/test_core.py`：图像形状、功率、指标和理论函数。
- `tests/test_attacks.py`：PGA/C&W 是否能达到重建阈值。
- `tests/test_runtime.py`：微型训练—clean—攻击闭环。

测试通过只表示实现满足当前约定，不表示论文结论已经被证明。学习完成的最终标准是：你可以指出每个测试保护了什么，同时指出它无法验证什么。

## 推荐学习产出

完成后保留四张自己的表：

1. 论文设置—本地实现—补齐假设对照表。
2. 训练/clean/attack 调用链图。
3. 图像任务的维度、功率和指标口径表。
4. 论文理论条件—可测代理量—不能推出的结论表。

这些内容将直接成为后续对照实验的实施基础。
