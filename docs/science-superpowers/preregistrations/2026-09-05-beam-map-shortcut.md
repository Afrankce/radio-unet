# Pre-registration: Beam map 条件捷径消融

**Frozen at commit:** ca1a4f3b9911d9693f312deb77f1190795351e89
**Question doc:** `docs/science-superpowers/questions/2026-09-05-beam-map-shortcut.md`
**Analysis plan:** `docs/science-superpowers/plans/2026-09-05-beam-map-shortcut.md`

## Hypotheses

- H0：在固定 0° 单波束的阵列独立模型中，移除 beam map 不会对建筑邻域产生比开放区域更有利的实际效应；至少两个阵列满足 `S > -0.25 dB`。
- H1（方向性）：beam map 是抑制建筑传播学习的条件捷径；至少两个阵列满足 `S <= -0.25 dB`，且其中至少一个阵列满足 `delta_near <= -0.25 dB`。

其中 `delta_r = RMSE_beam_zero,r - RMSE_full,r`，`S = delta_near - delta_open`。负值代表移除 beam 更有利。

## Primary analysis (exact)

- 模型：Attention-Conditioned Multiscale UNO-FM Lite。
- 数据：6.7 GHz、0°，8x8/16x16/32x32 分别独立训练；每个阵列 560/80/160 场景。
- Full：`[Tx,height,beam]`；Beam-zero：`[Tx,height,zeros_like(beam)]`。
- 两组张量形状均为 `[B,3,256,256]`，参数量完全一致。
- `near_building = valid_mask AND (MaxPool11x11(height > 0) > 0)`；`open = valid_mask AND NOT near_building`。
- 主量：每个阵列的 near/open 像素加权 dB-RMSE、`delta_near`、`delta_open` 与 `S`。
- prediction 在计算前固定裁剪至 [0,1]；无效像素完全排除。

## Prediction

如果 beam map 提供了压制 height 学习的全局捷径，Beam-zero 对 near-building 的改善应比 open 至少大 0.25 dB；方向预测为 `S < 0`。不预设超过 0.25 dB 的具体幅度。

## Decision rule

- 支持 H1：至少两个阵列 `S <= -0.25 dB`，且其中至少一个 `delta_near <= -0.25 dB`。
- 反驳方向性 H1：至少两个阵列 `delta_near >= +0.25 dB`。
- 近似冗余：三个阵列的 `abs(delta_near) < 0.25 dB` 且 `abs(S) < 0.25 dB`。
- 其他组合：报告为异质或不确定，不重新定义阈值。

## Sample size & stopping

- 每个阵列固定 N=560 train、80 validation、160 test；不增加、删除或更换场景。
- seed=42，max epochs=1000，early-stopping patience=20。
- checkpoint 仅由 validation dB-RMSE 选择；test 只在训练和 CFG 固定后运行一次。
- 不做中途查看后延长训练，不依据一个阵列结果改变另两个阵列。

## Multiplicity

不进行 p 值检验。三个阵列采用预定义的“至少两个阵列”一致方向规则。整体 RMSE、MAE、NMSE、PSNR、SSIM 和可视化属于次要描述性结果；新增半径、分组或指标一律标为探索性。

## Secondary & exploratory

- 次要：整体 valid-mask dB-RMSE、dB-MAE、NMSE、PSNR、SSIM。
- 次要：三个固定测试场景的 target / Full / Beam-zero / error map。
- 探索性：其他建筑邻域半径、严格 LOS/NLOS 几何、频域误差和多 seed 复验。

## Planned deviations handling

任何数据、阈值、区域、训练轮数、seed 或模型结构偏离都必须单独记录；受影响结果只能作为探索性结果，不能用于上述 H1 判定。
