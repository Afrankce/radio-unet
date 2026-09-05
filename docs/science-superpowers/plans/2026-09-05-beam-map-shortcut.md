# Beam map 条件捷径消融 Analysis Plan

> **For agentic workers:** REQUIRED SUB-SKILL: pre-register this plan with science-superpowers:preregistering-analysis BEFORE execution. Then use science-superpowers:subagent-driven-analysis (recommended) or science-superpowers:executing-analysis to run it step-by-step. Steps use checkbox (`- [ ]`) syntax for tracking.

**Question:** 在固定 0° 单波束的多尺度 UNO-FM 中，完整 beam map 是否形成有害捷径，从而削弱模型对建筑邻域传播规律的学习？

**Design:** 受控输入消融；Full 与 Beam-zero 保持数据、网络、参数量、优化器、随机种子和推理过程一致，仅改变 beam 条件值。

**Data:** Multi-config Radiomap Dataset；每个阵列固定 560/80/160 个场景，分析单位为测试场景内的有效像素，并分别对 8x8、16x16、32x32 报告。

**Primary analysis:** 比较 Beam-zero 与已冻结 Full 基线在建筑邻域和开放区域的 dB-RMSE，并计算区域差异中的差异。

**Decision rule:** 若至少两个阵列的建筑相对效应 `S = delta_near - delta_open <= -0.25 dB`，且这两个阵列中至少一个满足 `delta_near <= -0.25 dB`，则结果支持“beam map 抑制建筑规律学习”；若至少两个阵列满足 `delta_near >= +0.25 dB`，则反驳该方向性假设；其余结果为不确定或近似冗余。

---

## 1. 固定数据流与实验身份

Full 条件：

\[
c_{full}=[T,H,B]\in\mathbb{R}^{B\times3\times256\times256}.
\]

Beam-zero 条件：

\[
c_{zero}=[T,H,0]\in\mathbb{R}^{B\times3\times256\times256}.
\]

第三通道使用 `zeros_like(beam)`，而不是删除通道。因此 BasicUNetEncoder 第一层、状态 lifting、九个 CA/SA-FNO stage 和参数量完全不变。Beam-zero 必须有独立的 experiment 名、config SHA、run root、checkpoint 和 result root，禁止续训 Full checkpoint。

原始数据只读：height 256x256；beam/radiomap 128x128 后 resize 至 256x256；radiomap 有效 dB 区间为 (-300, 0)，无效像素不进入损失或指标。

## 2. 固定训练与停止规则

- seed：42。
- AdamW：learning rate 1e-3，weight decay 1e-5。
- micro-batch：2；gradient accumulation：28；effective batch：56。
- EMA decay：0.999。
- max epochs：1000。
- early stopping patience：20，以 validation dB-RMSE 为唯一 checkpoint 选择依据。
- 不查看 test 后续训，不因某个阵列表现改变其他阵列的训练设置。
- FM：`x_t=(1-t)x_0+tY`、`u_t=Y-x_0`，训练每个样本独立采样一次 `t~U(0,1)`。
- 推理：固定哈希噪声、EMA、CFG=1.0、2-step Euler。

## 3. 建筑邻域的预定义

从 condition 的 height 通道取得归一化高度 `H`。建筑种子为 `H > 0`。使用 11x11、stride 1、padding 5 的最大池化对建筑种子膨胀，即半径 5 像素：

\[
N_5=\operatorname{MaxPool}_{11\times11}(\mathbb{1}[H>0])>0.
\]

区域固定为：

- `near_building = valid_mask AND N5`；
- `open = valid_mask AND NOT N5`。

建筑物自身通常被 valid mask 排除，因此不会计入任何指标。不得在结果出现后更换半径；其他半径只能标记为探索性分析。

## 4. 主指标与判定

所有预测先裁剪至 [0,1]，再乘 300 转换为 dB 误差。对每个阵列和区域进行像素加权聚合：

\[
\operatorname{RMSE}_r=300\sqrt{\frac{\sum_{i,p\in r}(\hat y_{ip}-y_{ip})^2}{\sum_{i,p\in r}1}}.
\]

定义：

\[
\delta_r=\operatorname{RMSE}_{zero,r}-\operatorname{RMSE}_{full,r},
\qquad
S=\delta_{near}-\delta_{open}.
\]

负的 `delta` 表示 Beam-zero 更好；负的 `S` 表示移除 beam 对建筑邻域更有利。最小实际差异固定为 0.25 dB。本实验不报告 p 值；固定 seed 的结果作为机制消融估计，不能替代多 seed 不确定性分析。

整体 valid-mask dB-RMSE 是共同主要性能指标。dB-MAE、NMSE、PSNR、SSIM、每场景差值分布和三张固定场景可视化均为次要描述性结果，不用于改变主结论。

## 5. 混杂与有效性边界

- 固定 0° beam map 在同一阵列的所有场景中相同，因此本实验只能判断它在“阵列独立单波束模型”中是否成为固定空间捷径。
- 结论不得外推为 beam map 对多波束模型无用；多波束中它承担波束身份与方向控制。
- 仅一个训练 seed 无法估计训练随机性；若差异小于 0.25 dB，应报告为近似持平，而不是宣称严格等价。
- `H>0` 邻域是建筑影响的代理，不等同于严格射线追踪得到的 NLOS 区域。

## 6. 工件与单向数据流

数据流：不可变 dataset/manifest -> Beam-zero dataset view -> 独立训练目录 -> 独立测试预测 -> 区域比较报告。

代码工件：

- 修改 `data_loaders/same_frequency.py`：增加 `full` / `beam_zero` 条件视图。
- 修改 `training/same_frequency_multiscale_uno_config.py`：记录独立实验身份。
- 修改 `training/same_frequency_trainer.py`：把条件变体传给 train/val/test dataset。
- 修改 `run_same_frequency_multiscale_uno.py`：增加显式 CLI 选择。
- 新建 `scripts/run_beam_zero_ablation.sh`：三阵列并行训练与评估。
- 新建 `evaluation/beam_ablation_regions.py`：在 Full 与 Beam-zero 预测工件上计算固定区域指标。

运行工件：

- `runs/beam_zero_ablation_6.7ghz_0deg/{8x8,16x16,32x32}`；
- `results/beam_zero_ablation_6.7ghz_0deg/{8x8,16x16,32x32}`；
- `results/beam_zero_ablation_6.7ghz_0deg/region_comparison.json`。

## 7. 执行步骤

### Task 1: 实现不改变形状的 Beam-zero 数据视图

- [ ] 在 dataset 构造器中验证条件变体只能为 `full` 或 `beam_zero`。
- [ ] 在 resize/normalize 后、拼接前把 Beam-zero 的 beam 张量替换为全零。
- [ ] 将条件变体写入 metadata。
- [ ] 用真实 manifest 的首个样本验证 Full 与 Beam-zero 的 Tx、height、target、valid mask 完全相同，Full beam 非零而 Beam-zero 第三通道严格全零。

### Task 2: 锁定配置、checkpoint 与 CLI 身份

- [ ] 将条件变体写入 Beam-zero 的 scientific payload 和 config SHA，同时保持旧 Full 配置可读。
- [ ] 确保 `with_run_root` 和 `from_json` 保留变体。
- [ ] CLI 增加 `--condition-variant {full,beam_zero}`，默认仍为 `full`。
- [ ] preflight 输出条件变体并拒绝 checkpoint 身份混用。

### Task 3: 实现固定建筑区域分析

- [ ] 在合成 16x16 height/valid mask 上验证半径 5 膨胀的像素数与手算结果一致。
- [ ] 验证 prediction 等于 target 时两个区域 RMSE 均为零。
- [ ] 验证只在 near 区域加入已知误差时，near RMSE 恢复该已知 dB 幅度且 open RMSE 为零。
- [ ] 输出每个阵列的 Full、Beam-zero、delta_near、delta_open、S 和固定判定标签。

### Task 4: 冻结代码并部署服务器

- [ ] 运行语法、配置往返、数据条件和单次 CUDA optimizer smoke validation。
- [ ] 提交实验代码并推送独立 GitHub 分支。
- [ ] 服务器拉取该精确 commit，记录 commit SHA、Python/CUDA/GPU 信息和输入路径。

### Task 5: 训练与评估

- [ ] 三个阵列分别使用独立 GPU、独立 run root，从头训练 Beam-zero。
- [ ] 训练完成后先执行 validation CFG=1.0 固定确认，再一次性执行 test。
- [ ] 保存全部 160 个测试预测、标准指标和固定三场景可视化。
- [ ] 运行区域比较脚本并按预注册规则解释；任何追加分析明确标为探索性。
