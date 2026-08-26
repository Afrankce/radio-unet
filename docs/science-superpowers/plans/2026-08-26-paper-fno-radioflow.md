# Does a Paper-Faithful FNO Improve RadioFlow Single-Beam Prediction? Analysis Plan

> **For agentic workers:** REQUIRED SUB-SKILL: pre-register this plan with science-superpowers:preregistering-analysis BEFORE execution. Then use science-superpowers:subagent-driven-analysis (recommended) or science-superpowers:executing-analysis to run it step-by-step. Steps use checkbox (`- [ ]`) syntax for tracking.

**Question:** Under the fixed 6.7 GHz, zero-degree, scene-disjoint 560/80/160 protocol, does a paper-faithful FNO2d velocity backbone improve dense radio-map test dB-RMSE relative to the existing U-Net Lite backbone across 8x8, 16x16, and 32x32 arrays?

**Design:** Controlled architecture-replacement benchmark with three independently trained array-specific models and a single predeclared joint decision rule.

**Data:** Three immutable 800-record manifests, each split into 560 training, 80 validation, and 160 held-out test scenes using `scene_split_seed42`; valid pixels in each 256x256 map are evaluated by the existing metric implementation.

**Primary analysis:** Compute the three array-specific dB-RMSE improvements `Delta_i = U-Net_RMSE_i - FNO_RMSE_i`, their arithmetic mean, the number of positive improvements, and the worst degradation.

**Decision rule:** Confirm H1 only when mean `Delta >= 0.3 dB`, at least two of three `Delta_i > 0`, and every `Delta_i >= -0.5 dB`.

---

## Immutable inputs and provenance

Server data root:

```text
/home/wys/radioflow_20260823/datasets/MultiConfigRadiomap
```

Fixed input checksums:

| Input | SHA-256 |
|---|---|
| `manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl` | `30d7cd86fd77b10fbb1d249661be0e482fdb9a2f35f8a895e0c2eba693523383` |
| `manifests/manifest_samefreq_6.7ghz_16x16_0deg.jsonl` | `1aed953055e9dd8be7db46b45fcbcbd42143ebd98e398433a6b37f99e858c47c` |
| `manifests/manifest_samefreq_6.7ghz_32x32_0deg.jsonl` | `8f8c4602b627a476ef1e91187563fe36bf6a16ec48afe1f1e189e10e1b1d84a6` |
| `manifests/height_stats_train.json` | `b2dc49cab44b4d3f8090920d0423b6fc6cf57faa1b5222fdf5cc2a0ed53c0c86` |
| `manifests/scene_split_seed42.json` | `a62ffd48065b3ff39560fd18a93455bf7eec9b6f6397edc98555f46cbdfa9e27` |

Each manifest has exactly 800 records. Training code must verify these checksums before fitting. Raw data are read-only; the experiment writes only checkpoints, logs, predictions, metrics, and figures to new output directories.

## Fixed environment and execution paths

```text
Code checkout: /home/wys/radioflow_20260823/fno-paper-singlebeam
Python:        /UserProject/wjs/anaconda3/envs/radioflow_20260823/bin/python
Environment:   /home/wys/radioflow_20260823/radioflow_remote_env.sh
Run root:      /home/wys/radioflow_20260823/results/fno_paper_samefreq_6.7ghz/runs
Result root:   /home/wys/radioflow_20260823/results/fno_paper_samefreq_6.7ghz/eval
Log root:      /home/wys/radioflow_20260823/results/fno_paper_samefreq_6.7ghz/logs
```

Observed environment before execution: Python 3.10.20, PyTorch 2.5.1+cu121, CUDA runtime 12.1, NumPy 1.26.4, pytest 9.0.3, four RTX 3090 GPUs with 24,576 MiB each. The implementation records `pip freeze`, GPU driver details, source commit, and invocation in the run root.

## Data flow

```text
immutable dataset + manifests
    -> existing SameFrequencyRadiomapDataset
    -> condition [B,3,256,256], target [B,1,256,256], valid_mask [B,1,256,256]
    -> x0 and one t per sample
    -> xt and target velocity
    -> ConditionalFNO2d velocity prediction
    -> valid-mask MSE
    -> EMA best.pt selected on 80-scene validation dB-RMSE at CFG=1.0
    -> one immutable 160-scene test evaluation
    -> metrics, predictions, runtime record, deterministic visualizations
    -> three-array summary and decision rule
```

## Confounds and validity controls

- Architecture capacity is controlled by fixing `width=40`, `modes=(12,12)`, and four layers before outcomes; real scalar degrees of freedom must be within 10 percent of 3,994,859.
- Data leakage is controlled by the existing scene-disjoint split and train-only height statistics. Test scenes are not used for stopping, configuration, width, modes, or CFG.
- Optional stopping is controlled by a maximum of 1000 epochs and automatic validation patience of 20. No manual stopping based on apparent performance is allowed.
- Randomness is controlled by seed 42, deterministic settings, fixed DataLoader generator state, full-state checkpoints, and fixed hash sampling noise.
- Evaluation flexibility is controlled by fixing CFG to 1.0, Euler to two steps, EMA `best.pt`, and the existing metric implementation.
- Boundary artifacts are controlled identically across arrays with fixed nine-pixel right/bottom padding and cropping.
- Infrastructure failures may be resumed only from the latest full-state checkpoint with the same commit/config identity. A scientific configuration change requires a new run directory and is exploratory.

## Sample size and uncertainty

The dataset size is fixed: 160 test scenes per array. No p-value or population-level power claim is planned, so alpha and frequentist power are not applicable. The smallest effect of engineering interest is fixed at a 0.3 dB improvement in the mean of the three array-specific dB-RMSE values. Single-seed variability is an acknowledged limitation; no confidence claim over random initializations will be made.

## Planned outputs

- Primary: per-array and mean dB-RMSE improvements versus the frozen U-Net Lite values.
- Secondary: dB-MAE, NMSE, PSNR, SSIM, selected epoch, parameter counts, peak GPU memory, training time, and generation runtime.
- Figures: existing deterministic first/middle/last-scene comparison and error maps for each array; training/validation curve for each FNO run; one three-array FNO-versus-U-Net metric table.
- No secondary metric can reverse the primary decision.

## Execution tasks

### Task 1: Validate code and synthetic model contracts

**Artifacts:**
- Reads: committed source and tests only
- Writes: pytest output and smoke logs under the new log root

- [ ] Run unit tests for Fourier corner truncation, output shape, coordinate/time concatenation, condition dropout, CFG=1 equivalence, float32 FFT under AMP, gradients, parameter accounting, factory identity, CLI locks, and evaluator routing.
- [ ] Validate on synthetic tensors that a forward/backward pass is finite for batch size 2 and 256x256 resolution.
- [ ] Confirm the real scalar degree count is within the fixed Lite tolerance before any real-data training.

### Task 2: Validate immutable data and one-step training

**Artifacts:**
- Reads: the five checksummed input files and source data referenced by the manifests
- Writes: `_smoke` checkpoint and smoke log only

- [ ] Recompute all five SHA-256 values and require exact matches.
- [ ] Run `--preflight-only` independently for 8x8, 16x16, and 32x32; require 560/80/160 counts, 6.7 GHz, zero degrees, and schema-selected beam IDs.
- [ ] Run one optimizer-step smoke training on one idle GPU; require finite masked loss, valid optimizer/scaler/EMA/scheduler state, and a fresh strict checkpoint.

### Task 3: Train the three fixed FNO models

**Artifacts:**
- Writes: array-specific run directory, stdout log, stderr log, `last.pt`, `best.pt`, and canonical config/identity records

- [ ] Launch 8x8 on physical GPU 1, 16x16 on physical GPU 2, and 32x32 on physical GPU 3, with each process seeing its assigned card as `cuda:0` through `CUDA_VISIBLE_DEVICES`.
- [ ] Train until automatic early stopping or epoch 1000; do not inspect test outputs.
- [ ] Resume infrastructure interruptions from `last.pt` only after strict commit/config/checkpoint identity validation.

### Task 4: Perform the single frozen test evaluation

**Artifacts:**
- Reads: each array's strict EMA `best.pt`
- Writes: a new immutable result transaction for each array

- [ ] Validate the best checkpoint and recompute its validation dB-RMSE at fixed CFG=1.0.
- [ ] Evaluate all 160 test scenes once with fixed hash noise and two-step Euler.
- [ ] Require exactly 160 prediction artifacts, one complete metric row, runtime evidence, and all deterministic visualizations before publishing each result directory.

### Task 5: Apply the registered decision rule and report

**Artifacts:**
- Reads: three completed `metrics_test.json` files and frozen U-Net reference values
- Writes: `summary.json`, `summary.csv`, comparison table, curves, and a concise report

- [ ] Compute `Delta_i`, mean `Delta`, positive-improvement count, and worst degradation without changing metric definitions or rounding before calculation.
- [ ] Apply the fixed composite decision rule exactly once.
- [ ] Label all unregistered diagnostics or follow-up configurations exploratory.
- [ ] Run the preregistration audit and archive source commit, environment record, logs, checkpoints, metrics, and figures.

