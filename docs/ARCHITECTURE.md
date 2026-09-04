# 架构与数据流

## 1. 输入、目标与有效区域

每个 batch 的核心张量为：

| 符号 | 形状 | 含义 |
|---|---|---|
| `c` | `[B,3,256,256]` | `Tx mask`、归一化高度、归一化 beam map |
| `Y=x_1` | `[B,1,256,256]` | 目标无线电图，合法 dB 值线性映射到 `[0,1]` |
| `M` | `[B,1,256,256]` | `valid_mask`，只标记真实传播像素 |
| `x_0` | `[B,1,256,256]` | 与目标同形状的高斯噪声 |
| `t` | `[B]` | 每个样本独立从 `U(0,1)` 采样的连续时间 |
| `x_t` | `[B,1,256,256]` | 噪声和目标之间的中间状态 |

原始 height 为 256x256；beam map 与 radiomap 为 128x128，并在加载时分别使用
双线性插值和相应的连续/离散规则调整到 256x256。目标中的 `-300` 和 `1000`
分别是 floor/building 哨兵，不进入有效区域损失。

## 2. Conditional Flow Matching

当前实现使用 `ConditionalFlowMatcher(sigma=0.0)`。因此训练路径是确定性的直线插值：

```math
x_t=(1-t)x_0+tY,\qquad u_t=Y-x_0.
```

速度网络学习：

```math
v_\theta(x_t,t,c)\approx u_t.
```

训练损失只在有效像素上计算：

```math
\mathcal L=
\frac{\sum M\odot\lVert v_\theta(x_t,t,c)-u_t\rVert_2^2}
     {\sum M}.
```

这里的 `M` 是无线电图有效区域 mask，不是 FNO 的频率模态数 `m`。一次训练前向对
batch 中每个样本只采一个 `t`；跨 batch、跨 epoch 的随机采样共同覆盖 `[0,1]`，
并不等价于只学习一个固定时刻。

## 3. 条件编码器（蓝色路径）

`BasicUNetEncoder` 对 `c` 产生五个空间尺度：

```text
e0: [B,  32,256,256]
e1: [B,  32,128,128]
e2: [B,  64, 64, 64]
e3: [B, 128, 32, 32]
e4: [B, 256, 16, 16]
```

这些是条件特征，不是 FM 状态。每个 `e_i` 只送到相同分辨率的 encoder/decoder
FNO stage；`e_4` 送入 bottleneck。条件编码器只为条件提取多尺度特征，FM 的
当前状态仍由另一条 U 形状态路径传播。

## 4. 状态 lifting 与坐标网格

模型生成两个确定性坐标通道：

```math
g_x(r,s)=s/(W-1),\qquad g_y(r,s)=r/(H-1).
```

随后逐像素拼接：

```math
q=[x_t,c,g_x,g_y]\in\mathbb R^{B\times6\times256\times256},
\qquad z_0=P_{lift}(q)\in\mathbb R^{B\times32\times256\times256},
```

其中 `P_lift` 是 1x1 convolution。它在每个像素位置共享同一组权重，只混合通道，
不做空间卷积。坐标通道用于显式区分绝对位置，避免纯频域平移等变算子把所有位置
视为完全同质。

## 5. U 形多尺度 FNO 状态骨干（绿色路径）

状态通道与分辨率为：

```text
Enc0  32 @ 256x256  --down-->  Enc1  64 @ 128x128
  |                                  |
skip0                              skip1
  |                                  |
Dec0  32 @ 256x256  <--up--    Dec1  64 @ 128x128

Enc2 128 @ 64x64 --down--> Enc3 256 @ 32x32 --down--> Bottleneck 256 @ 16x16
  |                         |                                  |
skip2                     skip3                               up
  |                         |                                  |
Dec2 128 @ 64x64 <--up-- Dec3 256 @ 32x32 <-------------------+
```

每次下采样是 `AvgPool2d(2,2)` 后接 1x1 convolution；每次上采样是双线性插值到
对应 encoder skip 的大小，沿通道维 concat，再用 1x1 convolution 压缩。因而
encoder 和 decoder 都各有一个 FNO stage，且存在四条真实的 U-Net 式 skip
connection。全模型共有 4+1+4=9 个 FNO stage，而不是四个全分辨率 block 串联。

## 6. 单个 Attention-Conditioned FNO stage

输入为同尺度状态 `z_i`、同尺度条件特征 `e_i` 和全局时间嵌入 `tau(t)`：

```text
z_i: [B,C_i,H_i,W_i]
e_i: [B,E_i,H_i,W_i]
tau: [B,512]
```

### 6.1 CA/SA 条件融合

代码中的类名是 `CrossAttention`，但当前真实实现不是 Transformer Q/K/V attention。
它由 channel attention、spatial attention 和一个条件 1x1 projection 组成：

```math
CA(z)=\sigma(MLP(GAP(z))+MLP(GMP(z))),
```

```math
SA(z)=\sigma(Conv_{7\times7}([Mean_c(z),Max_c(z)])),
```

```math
a_i=z_i\odot CA(z_i)+z_i\odot SA(z_i)+P_i(e_i).
```

`P_i(e_i)` 把条件通道投影到 `C_i`。条件不是拿来生成 Q/K/V，而是作为同尺度加性
残差进入状态；CA/SA 权重本身由状态 `z_i` 计算。

### 6.2 频域、局部与时间三支路

先将 `a_i` 逐像素投影到固定 operator width 24：

```math
h_i=L_i(a_i)\in\mathbb R^{B\times24\times H_i\times W_i}.
```

stage 更新为：

```math
\Delta_i=GELU\left(
\mathcal F^{-1}(R_i\,\mathcal F(h_i))+W_i h_i+T_i(\tau(t))
\right),
```

```math
z_i^{out}=a_i+Q_i(\Delta_i).
```

- 频域支路：`rfft2 -> 截取低频 -> 复数可学习权重 -> irfft2`；
- 局部支路：1x1 convolution，补充逐位置通道混合；
- 时间支路：`t -> sinusoidal embedding(128) -> MLP(512) -> Linear(C=24)`，
  再 reshape 为 `[B,24,1,1]` 并广播到整个空间；
- 残差：将 24 通道 delta 投回 `C_i` 后与 `a_i` 相加。

时间分支进入每个 stage，是为了让每一尺度的速度估计都显式依赖当前 ODE 时间；
只在网络入口注入一次，经过多层变换后容易被其他幅值覆盖。

### 6.3 `m=12/8/4` 的含义

`m` 是每个尺度保留的低频 Fourier modes 数，不是层数或通道数。对应配置为：

| Scale | Resolution | modes `m` | padding |
|---|---:|---:|---:|
| level 0 | 256x256 | 12 | 9 |
| level 1 | 128x128 | 12 | 5 |
| level 2 | 64x64 | 8 | 3 |
| level 3 | 32x32 | 4 | 2 |
| level 4 | 16x16 | 4 | 1 |

二维实数 FFT 的实现分别学习频谱上、下两个纵向角区块，并保留横向 rFFT 的前
`m` 个频率。较深尺度网格更小，因此使用更少的模态。

## 7. 输出与两步 Euler

最后的 256x256 decoder 状态经过 `1x1:32->128`、GELU 和 `1x1:128->1`，得到
速度场。推理从固定噪声 `x^(0)` 开始，用 `K=2`、`Delta t=1/2`：

```math
t_k=k/K,\qquad
x^{(k+1)}=x^{(k)}+\frac{1}{K}v_{CFG}(x^{(k)},t_k,c),
\quad k=0,1.
```

因此网络在 `t=0` 和 `t=0.5` 各调用一次，`x^(2)` 作为预测图。一般 CFG 形式为：

```math
v_{CFG}=v_{uncond}+\gamma(v_{cond}-v_{uncond}).
```

本实验锁定 `gamma=1.0`，实现会直接返回 conditional 分支，数值上等价于纯条件
速度。两步是推理 ODE 的离散积分步数，与训练时每个样本采一个连续 `t` 是两个不同
概念。

## 8. 与原始 U-Net 速度骨干的关键差异

- 保留：条件 `BasicUNetEncoder`、多尺度条件、CA/SA 融合、CFM 目标、EMA、CFG、
  Euler 采样。
- 改变：绿色状态速度网络的 3x3 CNN stage 改为 spectral+local FNO stage。
- 下采样：改为平均池化 + 1x1 projection。
- 上采样：改为 bilinear resize + skip concat + 1x1 projection。
- 保留四条 encoder-to-decoder skip；并非只替换一个 `TwoConv`。

实现入口：`model/attention_multiscale_uno.py`。
