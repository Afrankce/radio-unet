# Attention-Conditioned Full-Resolution FNO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, verify, deploy, and launch three dense single-beam RadioFlow runs using the approved attention-conditioned full-resolution FNO velocity backbone.

**Architecture:** A new model composes the existing Lite `BasicUNetEncoder`, five-to-one full-resolution condition aggregation, four enabled RadioFlow CA/SA fusion modules, four width-40 spectral/local/time FNO blocks, and pointwise velocity projection. Dedicated configuration, trainer, CLI, evaluator registration, and server launcher preserve all previous experiment identities and outputs.

**Tech Stack:** Python 3.10, PyTorch, MONAI, pytest, CUDA AMP, Bash, SSH.

**Spec:** `docs/superpowers/specs/2026-08-27-attention-conditioned-full-resolution-fno-design.md`

## Global Constraints

- Formal protocol is 6.7 GHz, 0 degrees, 560/80/160 scenes, seed 42.
- Maximum epochs are 1000 and early-stopping patience is 20.
- Width is 40, retained modes are `(12,12)`, padding is 9, and FNO depth is 4.
- State features remain full-resolution; no state U-Net down/up path or skip is allowed.
- Existing U-Net, paper-FNO, sparse, and Hybrid FNO-U files/results remain intact.
- Formal training runs only on the `wys@222.199.196.27` server.

---

### Task 1: Lock the model behavior with failing tests

**Files:**
- Create: `tests/test_attention_fno_model.py`
- Create: `tests/test_attention_fno_factory.py`

**Interfaces:**
- Consumes: existing `SpectralConv2d`, `BasicUNetEncoder`, `CrossAttention`.
- Produces: required API for `AttentionConditionedFNO2d` and `build_attention_fno()`.

- [ ] Write a test constructing a small width-4/modes-2 model and asserting output `[B,1,H,W]`, finite state/parameter gradients, four spectral blocks, four CA/SA blocks, and five condition projections.
- [ ] Run `python -m pytest tests/test_attention_fno_model.py -q` and confirm import failure for the absent class.
- [ ] Add tests that hand-check coordinate directions and verify the raw lifting has six inputs and no replicated time-map input.
- [ ] Add tests that replace the five encoder outputs with constants and assert projected/resized summation has shape `[B,C,H,W]`.
- [ ] Add tests proving changing `t` changes output through every block's nonzero time projection and CFG scale 1 equals the conditional path.
- [ ] Add a factory test asserting exact architecture identity without changing existing paper-FNO and U-Net factories.

### Task 2: Implement the approved model

**Files:**
- Create: `model/attention_fno.py`
- Modify: `training/model_factory.py`

**Interfaces:**
- Produces: `AttentionConditionedFNO2d`, `FullResolutionFNOBlock`, `build_attention_fno`, and model-size identity `attention_fno_lite`.
- `forward(image,x,pred_type='denoise',step,embedding=None) -> Tensor`.
- `embed_model(condition) -> list[Tensor]`.
- `forward_with_cfg(image,x,step,embedding=None,cfg_scale=1.0) -> Tensor`.

- [ ] Implement `FullResolutionFNOBlock` with CA/SA, padded spectral branch, local 1x1 branch, block-specific time projection, sum, and GELU.
- [ ] Run the focused block tests until green.
- [ ] Implement normalized grids, six-channel lifting, Lite encoder, five projections/resizes/sum, shared sinusoidal time MLP, four serial blocks, and pointwise projection.
- [ ] Implement joint raw/encoded condition dropout and CFG behavior.
- [ ] Run `python -m pytest tests/test_attention_fno_model.py tests/test_attention_fno_factory.py -q`.
- [ ] Refactor only after the focused tests remain green.

### Task 3: Add immutable experiment configuration and training entry point

**Files:**
- Create: `training/same_frequency_attention_fno_config.py`
- Create: `training/same_frequency_attention_fno_trainer.py`
- Create: `train_same_frequency_attention_fno.py`
- Create: `tests/test_same_frequency_attention_fno_config.py`
- Create: `tests/test_same_frequency_attention_fno_trainer.py`
- Create: `tests/test_train_same_frequency_attention_fno_cli.py`

**Interfaces:**
- Produces: `AttentionFNOTrainConfig`, `run_same_frequency_attention_fno_training`, and a CLI matching existing dataset/run/device/resume controls.
- Reuses: `SameFrequencyTrainConfig`, `MultiConfigSRMTrainer`, strict checkpointing, `ComplexGradScaler`, and manifest inference.

- [ ] Write failing tests for locked scientific payload, serialization/hash validation, model identity, preflight, and one-step full-state smoke reload.
- [ ] Run focused tests and confirm failures are caused by missing modules.
- [ ] Implement the frozen wrapper config with a distinct run identity and measured parameter counts.
- [ ] Implement the trainer by adapting the existing paper-FNO orchestration without changing its behavior.
- [ ] Implement the CLI for `8x8|16x16|32x32`, explicit device and resume, preflight, smoke, and optional stop-after-epoch.
- [ ] Run all new config/trainer/CLI tests until green.

### Task 4: Add evaluation and server launch integration

**Files:**
- Create: `evaluate_same_frequency_attention_fno.py`
- Create: `scripts/run_same_frequency_attention_fno_server.sh`
- Create: `tests/test_evaluate_same_frequency_attention_fno_cli.py`
- Create: `tests/test_same_frequency_attention_fno_server_script.py`

**Interfaces:**
- Evaluator consumes the new config/checkpoint identity and existing same-frequency evaluator.
- Launcher supports exactly `--dry-run`, `--preflight`, `--smoke`, and `--train`.

- [ ] Write failing CLI and Bash launcher tests that assert separate code/result roots and GPU mapping `0,1,2`.
- [ ] Implement evaluator model selection while preserving existing metric and fixed-noise behavior.
- [ ] Implement launcher commands for the three manifests and independent run directories.
- [ ] Run focused integration tests until green.

### Task 5: Verify locally and on CUDA

**Files:**
- Modify only code required by failures attributable to this feature.

**Interfaces:**
- Produces fresh test and smoke evidence.

- [ ] Run all new tests with `D:/Anaconda3/envs/radioflow-win/python.exe -m pytest ... -q`.
- [ ] Run existing paper-FNO and same-frequency regression tests.
- [ ] Run a local width-4 forward/backward and a production-width CUDA inference shape test.
- [ ] Record unrelated baseline failures separately; do not fix them in this feature.
- [ ] Inspect `git diff --check`, `git status --short`, and the exact parameter counts.

### Task 6: Deploy and launch three server experiments

**Files:**
- Deploy tracked repository files to `/home/wys/radioflow_20260823/attention-fno-singlebeam`.
- Write outputs only below `/home/wys/radioflow_20260823/results/attention_fno_samefreq_6.7ghz`.

**Interfaces:**
- Consumes server environment `/home/wys/radioflow_20260823/radioflow_remote_env.sh` and existing dataset/manifests.
- Produces three independent live training processes and recoverable checkpoints.

- [ ] Sync code without copying local results, checkpoints, caches, or the unrelated untracked Hybrid FNO-U test.
- [ ] Run the server launcher in `--dry-run` and `--preflight` modes.
- [ ] Run three one-step CUDA smoke jobs and require all to exit zero with fresh reloadable checkpoints.
- [ ] Launch formal 8x8/16x16/32x32 jobs on physical GPUs 0/1/2.
- [ ] Verify PID liveness, GPU ownership, initial logs, immutable config files, and result-directory separation.
- [ ] Report exact remote paths and current status without claiming completion of training.

