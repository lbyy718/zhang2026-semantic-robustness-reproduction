# 任务—噪声—通信结构鲁棒性对照实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-12
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## 实验概览

- **标题：**拆解 DeepJSCC 经验鲁棒性的任务、噪声与通信结构来源
- **目标：**判断论文报告的高攻击功率主要来自连续重建、随机噪声训练、通信瓶颈/功率约束，还是三者交互。
- **类型：**PyTorch 模型训练、白盒攻击、局部敏感性诊断和因子分析。
- **当前阶段：**实验设计；尚未实现分类分支和尚未启动新训练。

## 研究问题

### 主研究问题

在模型容量、latent 维度、攻击位置、优化预算和 clean performance 尽量匹配后，DeepJSCC 的经验鲁棒性是否仍然具有“语义通信特异性”？

### 子问题

1. 随机噪声训练是否在重建和分类任务中普遍提高鲁棒性？
2. 连续重建是否因为损失与失败阈值定义而表现出更大的最小攻击功率？
3. encoder—channel—decoder 结构、功率归一化和带宽瓶颈是否独立影响鲁棒性？
4. 局部 Jacobian 变化是否与攻击功率变化一致？
5. 原结论是否对攻击算法、重启次数、目标阈值和 clean margin 敏感？

## 假设

这些是假设，不是预设结论：

- **H1（噪声主效应）：**带噪训练会在重建和分类中降低局部敏感性并提高经验最小攻击功率。
- **H2（任务主效应）：**即使无噪训练，连续重建仍可能因损失几何和阈值定义表现出不同于分类的 `rho*`。
- **H3（交互效应）：**若 `Task × Noise` 显著，噪声训练对重建和分类的作用不同。
- **H4（通信结构效应）：**功率归一化、固定带宽瓶颈或信道层可能在控制任务后继续影响鲁棒性。
- **H5（机制一致性）：**若论文 smoothness 解释成立，噪声引起的局部 Jacobian 下降应与 `rho*` 上升跨样本相关。

## 因子设计

### 第一阶段：最小 2×2 设计

| 组别 | 任务 | 训练信道噪声 | 结构 |
| --- | --- | --- | --- |
| R0 | 连续重建 | 无噪 | 相同 encoder + latent + reconstruction decoder |
| R1 | 连续重建 | 有噪 | 同 R0，仅打开训练噪声 |
| C0 | 分类 | 无噪 | 相同 encoder + latent + classifier head |
| C1 | 分类 | 有噪 | 同 C0，仅打开训练噪声 |

这一阶段回答 `Task × Noise`，不能单独证明语义通信效应。

### 第二阶段：加入通信结构因素

增加 `Communication structure`：

- `channelized`：encoder → 功率归一化 → 带宽瓶颈 → AWGN → task head；
- `non-channelized`：参数量和 latent 维度匹配，但不使用通信功率/信道约束；必要时用普通 Gaussian augmentation 单独控制噪声。

形成：

```text
Task (reconstruction/classification)
× Noise training (off/on 或多个强度)
× Communication structure (off/on)
```

完整 2×2×2 共 8 个实验单元。若资源有限，先完成 2×2，再根据效应决定是否扩展。

### 第三阶段：噪声剂量响应

二元“有噪/无噪”无法说明剂量关系。建议从以下训练条件中选至少三个：

```text
noiseless
20 dB
10 dB
0 dB
SNR range，例如 [0,20] dB 均匀采样
```

如果鲁棒性随噪声增加呈稳定变化，同时 clean performance 出现代价，机制解释会更可信。

## 模型设计原则

### 共享部分

- CIFAR-10 数据和数据划分相同。
- 尽量共享 encoder 主干。
- latent 实维度/信道使用数相同，首轮建议沿用图像任务 `N=768`。
- 相同功率归一化策略。
- 参数量与 FLOPs 尽量匹配，并在结果中报告。
- 相同优化器、学习率、epoch、batch size、数据增强和权重衰减。

### 重建头

- 沿用当前 DeepJSCC decoder。
- 训练目标为 MSE。
- clean 指标为 PSNR、MSE，可增加 SSIM 作为辅助指标。

### 分类头

- 在相同接收 latent 上接分类 head。
- 训练目标为 cross entropy。
- clean 指标为 accuracy、NLL 和 logit margin。
- 不能直接用“输出 L2 变化”代替标签鲁棒性，必须报告决策 margin/标签翻转。

### “语义通信”操作化

不要把“分类”自动等同于“非语义通信”。本实验中的可操作因素应称为：

```text
是否具有学习式通信瓶颈、功率归一化和显式随机信道
```

这样研究对象可测量，也避免语义通信定义争议吞没实验结论。

## 训练条件和随机性

- 每个核心实验单元至少 3 个独立训练种子；正式结果建议 5 个。
- 信道重复不能代替训练种子。
- 所有组使用同一组种子集合。
- 每个 checkpoint 使用多个独立信道噪声 realization 评估。
- 保存完整配置、环境、Git commit、数据校验值和模型选择规则。
- 禁止依据测试集挑选阈值或最佳训练轮次。

首轮推荐训练种子：

```text
2026, 2027, 2028
```

可信后扩展：

```text
2026, 2027, 2028, 2029, 2030
```

## 攻击设计

### 统一攻击位置

所有主要比较均攻击接收 latent：

```text
r = channel(encoder(x))
r_adv = r + s
```

不要用“重建攻击 latent、分类攻击原图”的设置做主比较，否则攻击面不同。

### 统一功率口径

主结果至少同时保存：

```text
sum_i s_i^2
mean_i s_i^2
||s||_2
||s||_2 / ||r||_2
```

跨不同 latent 维度时，应优先使用每信道使用功率或相对信号能量，并保留总能量供论文口径对照。

### 重建失败事件

不只使用一个固定 15 dB 阈值。建议同时使用：

1. 论文阈值；
2. 多个 PSNR/MSE 阈值的敏感性曲线；
3. 相对 clean degradation；
4. 匹配 clean-to-failure margin 的阈值。

### 分类失败事件

至少报告：

- 标签首次翻转所需最小扰动；
- logit margin 到零所需最小扰动；
- 固定攻击预算下 robust accuracy；
- clean-correct 样本子集和全测试集两种口径。

### 攻击优化强度

主攻击应包含：

- 梯度上升/PGD 类攻击；
- C&W 类最小范数攻击；
- 多重随机重启；
- 成功后的二分精修；
- 步长和最大迭代敏感性检查。

建议至少用一个独立攻击实现或不同目标函数交叉验证，防止单个求解器对某一任务不公平。

## 机制测量

不能只测攻击功率，还应测可能的中介变量：

1. 接收 latent 处 task head 的局部 Jacobian 谱范数。
2. 随机方向和对抗方向的有限差分敏感性。
3. reconstruction decoder 输出变化或 classification logit/margin 变化。
4. 各层算子范数，仅作为松散诊断，不当作全局证书。
5. clean margin、重建阈值距离和 latent 信号能量。

关键检验链：

```text
Noise training
→ local sensitivity 下降？
→ rho* 上升？
→ 两者是否跨种子/样本相关？
```

如果只有 `rho*` 上升而局部敏感性不下降，论文的 smoothness 解释需要重新考虑。

## 主要指标

### Primary endpoints

1. 达到任务失败事件的经验最小每信道使用攻击功率。
2. 在预设攻击预算下的攻击成功率/robust accuracy。
3. 局部 Jacobian 谱范数估计。

### Secondary endpoints

- clean PSNR/MSE/SSIM；
- clean accuracy/NLL/margin；
- 攻击功率中位数、均值和 10/25/75/90 分位数；
- 攻击迭代数和失败率；
- 参数量、FLOPs、训练时间；
- 不同 SNR 与阈值下的完整曲线。

## clean performance 公平性

必须避免把“离失败阈值更远”误判为“局部更鲁棒”。采用三种并行分析：

1. **原始任务指标：**保留论文原阈值，便于复现。
2. **匹配分析：**只比较 clean performance/margin 接近的模型或样本。
3. **归一化分析：**把攻击功率与 clean-to-failure 距离共同报告。

若模型 clean performance 差异很大，不能只用原始 `rho*` 宣称鲁棒性机制成立。

## 统计分析计划

实验单元以独立训练种子为主要重复单位，样本和信道重复不能伪装成独立模型重复。

### 第一阶段 2×2

对每个预注册主要指标分析：

- Task 主效应；
- Noise 主效应；
- Task × Noise 交互效应；
- 效应量和置信区间。

### 第二阶段 2×2×2

加入：

- Communication structure 主效应；
- 两两交互；
- 三重交互。

若分布偏斜或存在攻击失败截尾，不应只做普通均值 t 检验；可考虑种子级汇总、bootstrap 置信区间、混合效应模型或生存/截尾分析。正式采用哪一种方法，应在看到最终结果前固定。

多 SNR、多阈值和多攻击属于多重比较，区分预注册主比较与探索性分析，必要时使用 FDR 控制。

## 结果判读矩阵

| 观察结果 | 更支持的解释 | 不能直接声称 |
| --- | --- | --- |
| R1、C1 都明显优于 R0、C0 | 一般噪声正则化 | 语义通信独有鲁棒性 |
| R0 已明显优于 C0 | 任务/损失/阈值效应 | noisy-channel training 是唯一原因 |
| 只有 channelized 组更强 | 通信瓶颈、功率或信道结构效应 | “语义”本身是原因 |
| Jacobian 下降且 rho* 上升 | smoothness 机制得到支持 | 全局认证鲁棒 |
| rho* 上升但 Jacobian 不变 | margin、阈值或攻击优化可能主导 | smoothness 已被证明 |
| 强攻击使差异消失 | 原攻击求解不足 | 所有语义通信都不鲁棒 |
| 控制后差异仍跨种子稳定 | 原论文现象被加强 | 可推广到未测试任务/信道 |

## 实施阶段

### Phase 0：锁定口径

- 写出主研究问题和主指标。
- 确定无噪/有噪定义、攻击位置、功率口径和失败事件。
- 固定种子、SNR、阈值与训练预算。
- 不训练正式模型。

### Phase 1：重构最小公共接口

- 保留当前 reconstruction task。
- 新增 classification task/head。
- 让两个任务共用 encoder、channel、功率归一化、训练框架和攻击接口。
- 新增形状、功率、分类阈值和攻击成功测试。

### Phase 2：小规模可行性检查

- 每个 2×2 单元运行 quick 配置。
- 检查所有模型都能学习，攻击成功率不是 0% 或无意义 100%。
- 检查输出 CSV/JSON 字段一致。
- 只修复实现问题，不根据结果更换假设。

### Phase 3：核心 2×2 正式训练

- 先跑 3 个独立种子。
- 生成 clean、attack、Jacobian 和阈值敏感性结果。
- 若效应远大于种子波动，再扩展到 5 个种子。

### Phase 4：通信结构与攻击强度消融

- 扩展至 2×2×2。
- 增加多重重启、精修和独立攻击交叉验证。
- 做 clean-matched 分析。

### Phase 5：理论—实验一致性检查

- 核对 Eq. (11)/(12) 条件。
- 检验局部 Jacobian 与 `rho*` 的关系。
- 明确哪些结论是经验的、条件性的或因果的。

## 预期输出

| 输出 | 建议路径 | 格式 | 成功标准 |
| --- | --- | --- | --- |
| 实验注册表 | `experiments/factorial/registry.csv` | CSV | 每个条件、种子、状态唯一可追踪 |
| 配置 | `configs/factorial/` | JSON | 因素只通过明确配置变化 |
| 训练清单 | `outputs/factorial/*/run_manifest.json` | JSON | 含 seed、commit、环境和最佳指标 |
| clean 结果 | `outputs/factorial/*/clean_metrics.csv` | CSV | 所有预设 SNR 和重复齐全 |
| 攻击样本 | `outputs/factorial/*/attack_samples.csv` | CSV | 成功率、功率和失败处理可审计 |
| Jacobian 结果 | `outputs/factorial/*/sensitivity.csv` | CSV | 样本级局部敏感性可关联攻击结果 |
| 汇总图 | `outputs/factorial_summary/` | PNG/CSV | 主效应、交互、阈值和 SNR 曲线齐全 |

## 启动正式实验前的强制检查

- [ ] 分类和重建是否攻击同一位置。
- [ ] latent 维度和功率口径是否一致。
- [ ] 训练噪声以外的变量是否固定。
- [ ] clean performance 差距是否被记录和处理。
- [ ] 攻击成功率和失败样本规则是否预先定义。
- [ ] 至少三个独立训练种子是否就绪。
- [ ] quick 四组是否全部跑通。
- [ ] 局部 Jacobian 是否被正确称为诊断量而非全局证书。
- [ ] 主比较与探索性比较是否分开。
- [ ] 结论措辞是否区分现象、机制和普适性。

## 停止规则

下列情况暂停正式扩展并先检查实现：

- 任一组没有学到非平凡 clean 性能；
- 未攻击状态已经满足攻击失败阈值；
- 攻击成功率接近 0%，导致条件均值失去意义；
- 不同任务的攻击实际发生在不同空间；
- 功率定义或 latent 维数不一致；
- 单个种子的效应与多种子方向相反；
- 更强攻击使主结论发生数量级变化。

正式研究不以“获得证伪结果”为停止条件，而以预注册核心比较完成、攻击强度检查通过和多种子不确定性可报告为完成条件。
