# Measurement-Consistent Sparse Radiomap Reconstruction Analysis Plan

> **For agentic workers:** REQUIRED SUB-SKILL: pre-register this plan with science-superpowers:preregistering-analysis BEFORE execution. Then use science-superpowers:subagent-driven-analysis (recommended) or science-superpowers:executing-analysis to run it step-by-step. Steps use checkbox (`- [ ]`) syntax for tracking.

**Question:** Under the fixed single-beam protocol, does mask-aware sparse conditioning plus measurement-consistent FM improve missing-region radiomap reconstruction over an environment-only baseline?

**Design:** Controlled four-arm ML experiment. All arms use the same scenes, masks, Lite capacity class, optimizer schedule, noise seeds, validation rule, and test protocol. The only registered differences are sparse representation and flow-consistency objective.

**Data:** 6.7 GHz, 0-degree beam, 8x8/16x16/32x32 arrays, exact scene-disjoint 560/80/160 split, 819 deterministic valid observations per map, 256x256 output. Raw data are immutable on `E:\datasets\MultiConfigRadiomap`.

**Primary analysis:** Compare D against A using pixel-weighted missing-region dB-RMSE on the fixed test scenes. Use scene-level paired bootstrap over the 800 common scene IDs, resampling scenes jointly across arrays with seed 42 and 10,000 replicates; report the point delta and 95% percentile interval.

**Decision rule:** Support H1 for the primary comparison if D has lower pooled missing dB-RMSE than A and the paired bootstrap 95% interval for D-A is entirely below zero. Report per-array results as secondary. Observed-region mean and maximum absolute errors for D must be <= 1e-5 in normalized units under projected sampling.

## Registered arms

- **A `environment_only`:** condition `[Tx_mask, Height, Beam_map]`; standard full-target FM; no sparse measurement input; source-equivalent Euler sampling.
- **B `concat_fullfm`:** condition `[sparse_map, observation_mask, Tx_mask, Height, Beam_map]`; standard full-target FM; direct concatenation; source-equivalent sampling.
- **C `multiscale_fullfm`:** environment condition plus a mask-aware multi-scale sparse encoder. At each encoder scale, compute `value_s = Pool(M*Y)/(Pool(M)+eps)` and `coverage_s = Pool(M)`; fuse with gated residual features. Standard full-target FM; no projection.
- **D `multiscale_consistent`:** exactly the C architecture. During FM training, observed pixels follow the fixed observed map and only missing-valid pixels have a noise-to-target path. During sampling, observed pixels are projected back after initialization and after every Euler step.

## Locked implementation

- Use the existing fixed manifest and mask protocol; do not regenerate splits or masks.
- Keep target normalization, invalid sentinels, height statistics, beam-map normalization, and 256x256 resizing identical across arms.
- Use Lite models, AdamW learning rate `1e-3`, weight decay `1e-5`, effective batch size 56, AMP float16, 2-step Euler, CFG scale 1.0, and fixed hash-derived test noise.
- Use `ema_decay=0.995`, 120 maximum epochs, and do not early-stop before 1000 optimizer steps. After the burn-in, use patience 20 on validation missing dB-RMSE.
- Select the checkpoint only from validation missing dB-RMSE; test data are never used for selection.
- Save protocol/config/model identity, metrics, per-scene metrics, and fixed visualization arrays under a new experiment root; never overwrite the previous `sparse_task2_singlebeam_feature5_samples819` runs.

## Data flow and artifacts

Raw NPY data -> shared manifest/mask dataset -> arm-specific condition tensors -> FM checkpoints -> validation-selected test predictions -> missing/observed/overall metrics and per-scene bootstrap deltas.

Planned files:

```text
data_loaders/sparse_consistent.py
training/sparse_consistent_config.py
training/sparse_consistent_model.py
training/sparse_consistent_flow.py
training/sparse_consistent_trainer.py
evaluation/sparse_consistent_sampling.py
evaluation/sparse_consistent_metrics.py
train_sparse_consistent.py
evaluate_sparse_consistent.py
summarize_sparse_consistent.py
scripts/run_sparse_consistent_8x8.ps1
tests/test_sparse_consistent_*.py
```

## Validation and execution steps

- [ ] Write the immutable question and pre-registration documents.
- [ ] Implement the shared dataset wrapper and validate exact shapes, scene counts, mask subset, and 819 observed pixels.
- [ ] Implement the four model arms and validate parameter construction, condition routing, and gated sparse features on synthetic tensors.
- [ ] Implement full-target and pinned-observation FM pairs; validate endpoint equations and missing-only masks on synthetic tensors.
- [ ] Implement source-equivalent and projected samplers; validate exact observed-value preservation for D.
- [ ] Run unit tests and a one-optimizer-step GPU smoke test for all four arms on 8x8.
- [ ] Run the registered 8x8 A/B/C/D training and validation selection.
- [ ] Run the registered 8x8 test evaluation and paired bootstrap summary.
- [ ] Run the same fixed protocol for 16x16 and 32x32 after the 8x8 pipeline passes its preflight; report any deviation as exploratory.
- [ ] Verify all result artifacts and perform protocol and rigor review before reporting conclusions.

## Confounds and validity controls

- EMA lag: fixed lower decay and burn-in rule across all arms; report raw/EMA state separately.
- Split severity: all arms use the same scene-disjoint split; comparisons are within-protocol only.
- Sampling density: fixed 819 count and same deterministic mask for each arm; no literal 5% claim.
- Architecture capacity: A/B use locked Lite DiffUNet; C/D add only the registered sparse encoder and report parameter counts.
- Projection: only D uses projection; observed-region metrics are reported separately and never substituted for missing-region metrics.

## Planned figures

- Per-arm missing dB-RMSE validation curves with EMA burn-in marker.
- Fixed test scene panels: observed mask, sparse map, ground truth, A/B/C/D predictions, missing-region absolute error.
- Per-array table of missing metrics and observed consistency audit.
- Paired scene-level D-A and C-B delta distributions.
