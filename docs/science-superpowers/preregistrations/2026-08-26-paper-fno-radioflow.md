# Pre-registration: paper-faithful FNO for RadioFlow single-beam prediction

**Frozen at commit:** 122eb2f712498719cf0a11f11781501f05e47d34
**Question doc:** `docs/science-superpowers/questions/2026-08-26-paper-fno-radioflow-singlebeam.md`
**Analysis plan:** `docs/science-superpowers/plans/2026-08-26-paper-fno-radioflow.md`
**Architecture spec:** `docs/superpowers/specs/2026-08-26-paper-fno-radioflow-design.md`

## Hypotheses

- H0: The FNO does not improve mean test dB-RMSE by at least 0.3 dB, or improves fewer than two arrays, or degrades at least one array by more than 0.5 dB.
- H1 (directional): The FNO improves mean test dB-RMSE by at least 0.3 dB, improves at least two arrays, and degrades no array by more than 0.5 dB.

## Primary analysis (exact)

- Frozen U-Net Lite dB-RMSE references: 8x8 `11.627`, 16x16 `11.657`, 32x32 `11.700` dB.
- For array `i`, compute `Delta_i = U-Net_RMSE_i - FNO_RMSE_i` using unrounded FNO metrics and the frozen three-decimal U-Net references above.
- Compute `mean_Delta = (Delta_8x8 + Delta_16x16 + Delta_32x32) / 3`.
- Compute `n_improved = count(Delta_i > 0)`.
- Compute `worst_Delta = min(Delta_i)`.
- No arrays, scenes, valid pixels, or completed runs are excluded after evaluation.

## Fixed model and training configuration

- Backbone: dense, unfactorized FNO2d.
- Input: `[x_t, Tx mask, height, beam map, t_map, grid_x, grid_y]`.
- Output: one-channel conditional FM velocity.
- Four Fourier layers; 12x12 retained modes; hidden width 40; right/bottom padding 9; pointwise 1x1 local branch; GELU after the first three layers; no normalization; `40 -> 128 -> 1` projection.
- Sample-level condition dropout 0.25; no time FiLM, attention, U-Net, U-FNO, mode schedule, factorization, or hyperparameter sweep.
- Seed 42; AdamW; learning rate 1e-3; weight decay 1e-5; warmup 0.10; EMA 0.999; effective batch 56; maximum 1000 epochs; patience 20; AMP with float32 FFT branch.
- Sampling: EMA `best.pt`, fixed hash noise, Euler, two steps, CFG=1.0.

## Prediction

- Direction: positive `Delta_i` values, indicating lower FNO dB-RMSE.
- Smallest effect of engineering interest: mean improvement of 0.3 dB.
- No external prior effect estimate exists for this architecture transfer.

## Decision rule

- Confirm H1 if and only if `mean_Delta >= 0.3`, `n_improved >= 2`, and `worst_Delta >= -0.5` dB.
- Disconfirm H1 otherwise.
- Secondary metrics and visual appearance cannot override this rule.

## Sample size & stopping

- Fixed data: 560 train, 80 validation, and 160 test scenes independently for each of three arrays.
- Fixed runs: one seed-42 run per array.
- Training stops only at automatic patience 20 or epoch 1000. No manual performance-based stopping and no peek-and-extend.
- Infrastructure interruption may resume from strict `last.pt`; it does not change the sample, seed, optimizer trajectory, or stopping rule.

## Multiplicity

- One confirmatory composite decision is made from three predeclared array-specific dB-RMSE improvements.
- No p-values are computed and no individual-array significance claims are made.
- dB-MAE, NMSE, PSNR, SSIM, memory, runtime, and figures are secondary/descriptive.

## Secondary & exploratory analyses

- Secondary: report the fixed evaluator's dB-MAE, NMSE, PSNR, SSIM, parameter count, selected epoch, memory, and runtime for all three arrays.
- Exploratory only: any alternative width, mode count, padding, activation, optimizer, CFG value, Euler step count, random seed, attention hybrid, U-FNO, cross-frequency setting, common8 setting, or sparse-reconstruction setting.

## Planned deviations handling

Any scientific deviation requires a new uniquely named run and is reported as exploratory. A code defect discovered before test evaluation may be fixed with a new test and a new frozen source commit; all affected training runs must restart from initialization. A defect discovered after test evaluation makes the affected result exploratory and forbids silently replacing it.

