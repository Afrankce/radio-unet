# RadioFlow Hybrid FNO-U 单波束实验设计

## 1. 目标与实验问题

本实验检验：在不改变 RadioFlow 条件编码、五尺度注意力融合、Flow Matching 目标、数据协议和推理器的前提下，仅把时间条件速度网络中的局部双卷积特征块替换为 Fourier Neural Operator（FNO）谱算子块，是否能改善 6.7 GHz、0° 单波束无线电图预测。

模型正式名称为 **Hybrid FNO-U Lite**。它不是纯全分辨率 FNO，也不能写成“完全删除 U-Net”。它保留 U 形五尺度拓扑、池化、上采样和 skip connection，只替换绿色速度路径中的 `TwoConv` 变换。因此实验比较的是局部卷积算子与全局谱算子，而不是同时比较两套完全不同的条件通路和拓扑。

主实验包含 8×8、16×16、32×32 三种阵列。数据协议固定为 6.7 GHz、0°、按场景互斥的 560/80/160 train/val/test 划分及 seed 42。已有 U-Net Lite 和纯全分辨率 FNO 结果只作为只读对照，不覆盖、不续写其目录。

## 2. 明确不变的部分

以下部分与锁定的 U-Net Lite 基线保持一致：

1. 条件张量 `c=[Tx mask, height, beam map]`，形状 `[B,3,256,256]`。
2. 目标无线电图 `x_1`，形状 `[B,1,256,256]`。
3. 蓝色 `BasicUNetEncoder` 及特征通道 `(32,32,64,128,256)`。
4. 黄色五个现行 `CrossAttention` 模块。这里的真实实现是通道注意力、空间注意力和条件特征的 1×1 投影相加，不启用历史遗留的 Q/K/V `CrossAttention_old`。
5. 绿色路径的五尺度分辨率、MaxPool 下采样、上采样和 skip connection。
6. 128 维正弦时间编码以及 `128 -> 512 -> 512` 的共享时间 MLP。
7. 条件 Flow Matching 插值、valid-mask MSE、AdamW、warmup、EMA、梯度累积、早停和断点续训。
8. 固定哈希初始噪声、CFG=1.0 和两步 Euler 推理。
9. 数据读取、归一化、有效像素口径及 RMSE、MAE、NMSE、PSNR、SSIM 评估器。

本实验不增加坐标通道，不修改条件通道顺序，不修改注意力公式，不引入 Q/K/V，不进行 modes/width 验证集搜索，也不包含稀疏 Task 2 或跨频率实验。

## 3. 完整数据流

### 3.1 蓝色条件分支

条件编码器接收：

```text
c: [B,3,256,256]
```

输出五尺度特征：

```text
e0: [B, 32,256,256]
e1: [B, 32,128,128]
e2: [B, 64, 64, 64]
e3: [B,128, 32, 32]
e4: [B,256, 16, 16]
```

训练时沿用锁定基线的 sample-level `cfg_drop_prob=0.25`：被选中的样本只把 `e0...e4` 置零。原始条件 `c` 仍沿绿色路径的原始输入拼接进入模型。该行为虽然不是严格的全条件 unconditional 分支，但必须保留，避免同时改变基线的条件语义。正式评估固定 CFG=1.0，因此最终输出就是 conditional prediction。

### 3.2 Flow Matching 状态与绿色输入

每个样本、每个训练 micro-batch 独立采样：

```text
x_0 ~ N(0,I)
t   ~ Uniform(0,1)
x_t = (1-t)x_0 + t x_1
u_t = x_1 - x_0
```

`t` 的形状为 `[B]`。绿色路径的原始输入保持：

```text
h_in = concat(c, x_t)  -> [B,4,256,256]
```

模型输出：

```text
v_theta(x_t,t,c): [B,1,256,256]
```

监督损失保持：

```text
L = sum(valid_mask * (v_theta-u_t)^2) / sum(valid_mask)
```

### 3.3 黄色五尺度融合

每个编码尺度先由新的 FNO 算子块得到状态特征 `z_i`，再调用原有黄色模块：

```text
CA_i = ChannelAttention(z_i)
SA_i = SpatialAttention(z_i)
f_i  = z_i * CA_i + z_i * SA_i + Conv1x1_i(e_i)
```

`f_i` 与 `z_i` 的形状相同。融合后的 `f_i` 同时流向下一层编码器，并作为相同尺度的 decoder skip。这样条件信息不仅影响 bottleneck，而是在 256、128、64、32、16 五个尺度逐级引导速度场。

## 4. FNO 算子块

每个原 `TwoConv(C_in,C_out)` 替换为一个 `FNOOperatorBlock(C_in,C_out,m,p)`。外部通道和空间尺寸不变，内部固定谱宽 `w=24`：

```text
y      = Conv1x1(C_in -> 24)(h)
y_pad  = right_bottom_pad(y, p)
global = SpectralConv2d(24,24,m,m)(y_pad)
local  = Conv1x1(24 -> 24)(y_pad)
time   = Linear(512 -> 24)(SiLU(t_emb))[:, :, None, None]
q_pad  = GELU(global + local + time)
q      = crop_to_original_size(q_pad)
out    = Conv1x1(24 -> C_out)(q)
```

`SpectralConv2d` 复用已经通过 AMP 修正的实现：FFT 分支强制 float32，保留实值二维 FFT 的上下两个频率角，其余频率置零，再执行 `irfft2`。谱权重是两个独立、稠密、未因子化的复数张量 `[24,24,m,m]`。三个 1×1/Linear 映射均使用 bias，谱卷积本身无 bias；右侧和底部使用零 padding。局部 1×1 分支保留点级变换能力；GELU 在频域全局分支、局部分支和时间偏置相加之后执行。块内不增加 normalization、dropout、坐标网格、显式 residual 或第二个谱层。

右侧和底部 padding 按相对分辨率缩放，避免把全分辨率的 9 像素机械复制到 16×16 层：

| 输出尺度 | 位置 | `C_in -> C_out` | modes `m` | padding `p` |
|---:|---|---:|---:|---:|
| 256×256 | `conv_0` | 4 -> 32 | 12 | 9 |
| 128×128 | `down_1` | 32 -> 32 | 12 | 5 |
| 64×64 | `down_2` | 32 -> 64 | 12 | 3 |
| 32×32 | `down_3` | 64 -> 128 | 8 | 2 |
| 16×16 | `down_4` | 128 -> 256 | 4 | 1 |
| 32×32 | `upcat_4` 拼接后 | 256 -> 128 | 8 | 2 |
| 64×64 | `upcat_3` 拼接后 | 128 -> 64 | 12 | 3 |
| 128×128 | `upcat_2` 拼接后 | 64 -> 32 | 12 | 5 |
| 256×256 | `upcat_1` 拼接后 | 64 -> 32 | 12 | 9 |

原 MaxPool、UpSample/deconvolution 和 skip 拼接保持不变。最终仍由 `Conv1x1(32 -> 1)` 输出速度。

## 5. 参数预算与架构锁

原 U-Net Lite 总参数量为 3,994,859 个实参数。Hybrid FNO-U 的固定预算为：

| 组成 | 参数口径 |
|---|---:|
| 蓝色条件编码器 | 1,192,800 real |
| 黄色五尺度注意力 | 100,074 real |
| 共享时间 MLP | 328,704 real |
| 上采样与最终投影等固定部分 | 176,417 real |
| 九个 FNO 块的非复数参数 | 154,152 real |
| FNO 复数权重 | 1,161,216 complex = 2,322,432 real |
| **合计** | **3,113,363 tensor elements / 4,274,579 real scalars** |

以复数参数的实部和虚部分别计数时，Hybrid FNO-U 比原 Lite 多 7.0%，满足批准的 ±10% 约束。工厂函数必须同时锁定 tensor-element count 与 real-scalar count；任一数量、features、谱宽、modes、padding、attention 类型或 CFG dropout 发生变化时，preflight 直接失败。

## 6. 训练与推理协议

三个阵列运行完全相同的科学控制：

- 频率：6.7 GHz
- 波束：由 manifest 校验为 0°，不在模型中猜测 beam ID
- 场景划分：560/80/160，scene-disjoint，seed 42
- 分辨率：256×256
- 优化器：AdamW，learning rate `1e-3`，weight decay `1e-5`
- warmup：计划 optimizer steps 的 10%
- EMA：0.999
- 最多轮数：1000
- 早停 patience：20
- 最优模型及早停指标：validation dB-RMSE，越低越好
- AMP：float16；FFT 分支内部 float32
- 初始 micro-batch：2
- 梯度累积：28
- 有效 batch：56

每个阵列独占一张 GPU 并行训练，第四张 GPU 留给 smoke test、评估或故障恢复，不使用三卡 DDP 混合三个阵列的数据。三个运行使用独立目录、配置哈希、日志、`last.pt` 与 `best.pt`。已有结果目录保持只读。

如果 batch=2 在 24 GB GPU 上发生 OOM，自动降为 micro-batch=1、accumulation=56，保持有效 batch=56；该回退必须写入 `run_config.json` 和日志，不能静默改变。NaN/Inf、manifest 不匹配、参数锁失败、谱 modes 越界或 checkpoint identity 不匹配均视为硬失败，不用错误状态继续训练。

推理使用 EMA `best.pt`、固定哈希噪声、CFG=1.0 和两步 Euler：

```text
x^(0) = fixed_hash_noise
x^(1) = x^(0) + 0.5 * v_theta(x^(0), 0.0, c)
x^(2) = x^(1) + 0.5 * v_theta(x^(1), 0.5, c)
```

## 7. 代码边界

新增独立模块和入口，不把第三种骨干继续塞入已有 `train.py`：

- 新增 Hybrid FNO-U 模型模块，复用现有 `SpectralConv2d`、`BasicUNetEncoder`、`CrossAttention` 和时间编码函数。
- 新增锁定的 Hybrid FNO-U 配置与 factory 注册。
- 新增单一训练 CLI、单一评估 CLI 和单一三阵列汇总入口。
- 复用现有 same-frequency dataset、trainer primitives、checkpoint、EMA、采样与 evaluator。
- 原 `DiffUNet`、`BasicUNetDe`、纯 `ConditionalFNO2d` 及其已有 CLI 和结果保持行为不变。

运行目录使用新的 experiment identity `same_frequency_6.7_single_beam_hybrid_fno_u`，由 CLI 的 `--run-root` 解析绝对位置；不在代码中硬编码本地或服务器路径。

## 8. 测试与验证

本地提交前必须通过：

1. `FNOOperatorBlock` 在全部九种通道/尺度组合上的 shape、finite forward、finite backward 测试。
2. float16 autocast 下 FFT 分支仍以 float32 工作且复数梯度可由 GradScaler 正确缩放。
3. 五尺度蓝色输出、绿色输出和黄色融合前后 shape 精确匹配。
4. forward hook 证明一次前向恰好调用五个现行 `CrossAttention`，且从未实例化 `CrossAttention_old`。
5. Hybrid 参数锁精确等于 3,113,363 tensor elements 和 4,274,579 real scalars。
6. CFG=1.0 等于 conditional forward；embedding dropout 与原 U-Net Lite 语义一致。
7. 一次 Flow Matching optimizer step、EMA 更新、checkpoint 保存与恢复 smoke test。
8. 原 U-Net Lite 和纯 FNO 测试继续通过，防止回归。
9. 三个 CLI 的参数、路径隔离、resume 和结果汇总测试。

服务器启动正式训练前，每个阵列执行一轮、一个 optimizer step 的 GPU smoke test，检查显存、loss、梯度、EMA 和 checkpoint。smoke 目录与正式目录分离。

## 9. 结果与成功判据

测试阶段仅加载每个阵列验证集选出的 EMA `best.pt`。分别报告：

- dB-RMSE
- dB-MAE
- NMSE
- PSNR
- SSIM
- best epoch
- real-scalar parameter count
- 峰值 GPU 显存
- 每 epoch 与每样本推理时间

主比较表只包含同一 6.7 GHz、0°、560/80/160 协议下的 U-Net Lite、纯全分辨率 FNO 和 Hybrid FNO-U；不得混入 common8、多波束、跨频率或稀疏 Task 2 指标。

工程成功的最低条件是：全部测试通过、参数锁满足、三阵列训练均产生可恢复的 checkpoint 并完成 EMA best 测试评估。科学结论不预设 Hybrid 必须优于 U-Net；只有测试指标完成后才能判断谱算子是否有效。

## 10. 明确排除项

- 不修改原 attention 为 Q/K/V。
- 不重新训练或覆盖已有 U-Net Lite、纯 FNO checkpoint。
- 不做 width、modes、padding 或 CFG sweep。
- 不加入 F3 稀疏观测、硬写回或 pinned Flow Matching。
- 不做 4.9 -> 6.7 GHz 跨频率实验。
- 不改变两步 Euler 推理口径。
