# Task 2 稀疏重建实验 v3 设计

> 2026-08-16。依据论文 2603.06401v1（Section IV-C Task 2、Table IV）与官方仓库
> `MulticonfigRadiomapDataset` 的 released benchmark、preprocessing、training 脚本核对后重写。

## 0. 现状与结论

B-long（concat_fullfm，819 点，单波束 0°，560/80/160 scene-disjoint，600 轮）测试集
missing-region dB-RMSE 13.86、观测区 dB-RMSE 13.30。观测区误差与缺失区几乎相同，说明
819 个已知观测值没有被模型实际使用；该实验是单波束 scene-disjoint 控制实验，不是论文
Task 2 的主协议。v3 要解决两件事：对齐 Task 2 协议，以及让稀疏观测真正进入预测。

## 1. 论文 Task 2 的权威定义

- 输入：稀疏测量 `X_sample` 为主输入；阵列参数 `X_array` 与环境 `X_env`（建筑高度图）为
  可选辅助输入；另外加入发射端位置二值 mask。
- 采样：从完整 radiomap 做受控采样，"uniform random 或 spatially imbalanced"；论文
  Table IV 的实验写的是 **5% sampling rate**。
- 划分：Task 1/2 均为 **instance 级随机划分**，论文写 7:1:2；官方发布代码默认
  `train/val/test = 0.8/0.1/0.1`。两者不一致，必须显式声明。
- 目标：完整 radiomap；评价 MAE(dB) 与 RMSE(dB)。
- 论文 Task 2 结果（5%）：RadioUNet+Beam MAE 5.87 / RMSE 8.86；
  RME-GAN+Beam MAE 5.56 / RMSE 8.61。

官方发布仓库的 released benchmark 把稀疏协议固定为
`random_sparse_feature_samples819`：819 个 valid 点、随机划分、feature-map 输入；
预处理 `fix_samples=819`，在 valid propagation 区域 `np.random.choice` 均匀选点；
官方 UNet 训练脚本的模型输入为 `[sparse_map, Tx, height, beam_map]`（feature 模式），
损失为全部 valid 像素上的掩码 MSE，观测 mask 不作为独立通道。

结论：主协议选 **819 点 source-aligned** 版本；另加一个 **5% 采样率** 变体用于和论文
Table IV 对齐。两版都做，报告时明确标注 819/65536 ≠ 5%。

## 2. 协议分层（不可再混用）

| 协议 | 划分 | 波束 | 观测 | 定位 |
|---|---|---|---|---|
| `random_sparse_feature_samples819` | instance 随机，7:1:2 | 8 个公共波束 | 819 | v3 主实验 |
| `random_sparse_feature_ratio5pct` | instance 随机，7:1:2 | 8 个公共波束 | 5% | 与论文 Table IV 对齐 |
| `singlebeam_feature5_samples819` | 560/80/160 scene-disjoint | 仅 0° | 819 | 保留的控制实验（已存在，不动） |

主实验 8×8：800 scene × 8 beam = 6400 records → 4480/640/1280。允许同一 scene 通过
不同 beam 跨 split 出现；record key（array+scene+beam）不得重叠。split 用固定 seed 42
的确定性 shuffle，写 `split_sha256`。

## 3. 输入与目标

4 通道（source-aligned，官方 feature 语义）：

```text
condition[0] = sparse_map  = observation_mask * normalized_target
condition[1] = tx_mask
condition[2] = normalized_height
condition[3] = beam_map
```

5 通道对照只追加显式观测位置：

```text
condition = [sparse_map, observation_mask, tx_mask, normalized_height, beam_map]
```

`valid_mask` 仅用于采样、损失与指标，不作为输入。目标为完整 normalized radiomap。
819 个点在最终 256×256 valid 网格上按 `hash(protocol, seed, scene_id)` 确定性选取
（与 array/beam 无关，使同场景不同波束共享物理采样位置；写入 mask_protocol_sha256）。

## 4. 模型变体

### 观测一致性语义（V0/V1 共用原则）

`sparse_map` 中的有效观测是无噪声真值，属于硬约束，不是待预测目标：训练与推理全程
观测点固定为其真值，模型速度场在观测点为 0，损失只覆盖缺失区。最终输出在观测点
必须逐比特等于 `sparse_map`，缺失区由 flow/回归在观测值作为边界条件下补全。当前
代码里 `build_masked_flow_pair`（multiscale_consistent 所用）已经实现该语义：
`xt = observed_map + missing * ((1-t)*noise + t*target)`、`ut = missing*(target-noise)`、
`loss_mask = valid & ~obs`；v3 把它作为 V1 的核心，而不是再依赖输出后 copy-back。

### V0 确定性回归（对齐官方 RadioUNet 范式）

- RadioFlow Lite UNet，condition 4/5 通道，直接回归 normalized dB 图。
- 损失：缺失区掩码 MSE，观测像素单独加权（×100）使网络自身复现观测值；对比官方
  全 valid MSE 消融。
- 推理：`output = where(obs_mask, sparse_map, output)` 硬回写，保证观测区逐像素等于
  `sparse_map`（回归范式下回写是必要保底，但训练加权让它不是唯一手段）。
- 这是 v3 的主变体：先证明协议正确、拿到可与论文比较的确定性基线。

### V1 条件 FM + 观测一致性（保留 RadioFlow 风格）

- 沿用条件 FM，但修复信息通路：
  1. 训练用 pinned-observation flow：`xt` 的观测点固定为 `sparse_map`，`ut` 观测点为 0，
     `loss_mask = valid & ~obs`（复用 `build_masked_flow_pair` 语义，训练/推理一致）；
  2. 每个 Euler 步之后 `x = where(obs_mask, sparse_map, x)` 硬投影，输出端不再额外回写；
  3. sparse_map/mask 经独立分支在原始分辨率注入（多尺度 skip 注入，gate 非零初始化）；
  4. 2-step Euler 与 10-step Euler 对比。
- 目标是回答：FM 范式在同等协议下能否不输给 V0。

### V2 可选对抗版（对齐 RME-GAN）

- PatchGAN 判别器 + L1 重建（L1 权重 400、对抗权重 0.5，论文口径），4 通道输入。
- 仅当 V0/V1 主结果稳定后作为补充。

## 5. 基线

- IDW/Shepard 插值（819 点直接插值）：量化"观测信息本身的天花板"。
- Task 1 稠密预测（无稀疏输入、同划分同波束）：无测量参考线。
- 4ch vs 5ch；有 beam_map vs 无 beam_map（论文核心结论的复现）。

## 6. 指标

- 主指标：valid 像素上的 MAE(dB)、RMSE(dB)（论文口径），分 overall/missing 两区。
- 观测区：硬回写后必须 ≈0；未回写版本也要报告，用于证明"是否真正读入了观测"。
- 次指标：PSNR、NMSE、标准 11×11 窗口 SSIM（沿用已修正的窗口版实现）。
- 选模：val missing-region dB-RMSE（或 MAE），best.pt + last.pt + EMA 评估。

## 7. 实施顺序与验证

1. manifest 合约：4480/640/1280、8 波束、819 点、scene-overlap audit、hash 锁定。
2. dataset 合约：channel 顺序、`obs ⊂ valid`、819 点、确定性 mask key。
3. V0 训练与 IDW 基线，8×8 先行；IDW 若不优于 V0，先排查数据/采样而不是加模型。
4. V1 一致性 FM（投影 + 加权 + 原始分辨率注入），同预算对比 V0。
5. 消融：beam_map on/off、4ch/5ch、819/5%、2-step/10-step。
6. 全部结果写 protocol + split_sha256 + mask_protocol_sha256 + 配置 hash。

验收线（参考，非承诺）：论文 5% 下 RadioUNet+Beam 为 8.86 dB RMSE；819 点信息量约为
5% 的 1/4，且我们的 Lite 骨干更小，v3 主结果应在 9-11 dB 量级才有意义；若仍 >12 dB，
优先怀疑观测注入/损失权重，而非训练量。

## 8. 明确不做的

- 不改动已存在的 `singlebeam_feature5_samples819` 控制实验与 B-long 结果目录。
- 不把 819 写成"5%"；不把 scene-disjoint 控制实验宣称为主 Task 2。
- 不修改 `training/sparse_*` legacy inpainting 语义。
