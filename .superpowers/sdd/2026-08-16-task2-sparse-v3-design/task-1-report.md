# Task 1 Report — random Task 2 pinned_fm V1

Date: 2026-08-17

## Scope handled

Implemented the random-instance Task 2 V1 `mode="pinned_fm"` path while preserving the existing `mode="regression"` behavior and keeping the work scoped to the random Task 2 modules plus focused tests.

I did not modify or stage unrelated pre-existing dirty files already present in the worktree.

## Changed files

- `evaluate_random_task2.py`
- `train_random_task2.py`
- `training/random_task2_config.py`
- `training/random_task2_trainer.py`
- `training/random_task2_flow.py` (new)
- `training/random_task2_model.py` (new)
- `evaluation/random_task2_sampling.py` (new)
- `tests/test_random_task2_cli.py` (new)
- `tests/test_random_task2_flow.py` (new)
- `tests/test_random_task2_model.py` (new)
- `tests/test_random_task2_sampling.py` (new)

## What changed

### Mode/config/CLI plumbing

- Unlocked `RandomTask2TrainConfig.mode` to accept both `regression` and `pinned_fm`.
- Kept mode in the canonical payload/hash so config SHA and run directory differ by mode.
- Added `--mode {regression,pinned_fm}` to both `train_random_task2.py` and `evaluate_random_task2.py`.

### Pinned flow helper

- Added `training/random_task2_flow.py`.
- Implemented `build_random_task2_pinned_flow_pair(...)` with the required semantics:
  - `xt = sparse_map` on observed pixels.
  - `xt = (1 - t) * x0 + t * target` on valid missing pixels.
  - `ut = 0` on observed and invalid pixels.
  - `ut = target - x0` on valid missing pixels.
  - `loss_mask = valid_mask & ~observation_mask`.
- Added contract checks for shape, dtype, device, finiteness, observation subset, and zero-outside-observation sparse maps.

### V1 pinned model path

- Added `training/random_task2_model.py`.
- Implemented a Lite pinned-FM model that:
  - preserves the locked Lite feature tuple `(32, 32, 64, 128, 256, 32)`;
  - accepts both 4-channel and 5-channel conditions;
  - exposes `embed_model(condition, sparse_map, observation_mask)`;
  - injects sparse value + coverage through an independent branch at every encoder scale;
  - uses trainable nonzero-initialized sparse gates;
  - exposes standard conditioned forward and CFG forward needed by training/sampling.

### Trainer/evaluator integration

- Updated `training/random_task2_trainer.py` so:
  - `regression` still uses the existing direct prediction path;
  - `pinned_fm` samples uniform `t in [0,1]`, standard-normal `x0`, builds the pinned pair, predicts velocity, and optimizes masked velocity MSE only on missing valid pixels;
  - validation for `pinned_fm` uses a deterministic 2-step projected Euler sampler and selects on `val_missing_db_rmse`;
  - observed pixels are projected back to `sparse_map` during pinned validation/sampling.
- Updated `evaluate_random_task2.py` so test-time pinned evaluation uses the same projected 2-step Euler sampler.

### Sampler

- Added `evaluation/random_task2_sampling.py`.
- Implemented projected Euler CFG sampling that:
  - starts from `where(observation_mask, sparse_map, x0)`;
  - uses timestep sequence `0/steps, 1/steps, ...`;
  - projects observed pixels back to `sparse_map` after every Euler step;
  - returns final observed pixels exactly equal to `sparse_map`.

## TDD record

### Focused failing test run before implementation

Command:

```powershell
& 'D:\Anaconda3\envs\radioflow-win\python.exe' -m pytest -q tests/test_random_task2_flow.py tests/test_random_task2_model.py tests/test_random_task2_sampling.py tests/test_random_task2_cli.py
```

Output:

```text
....FF.FF                                                                [100%]
================================== FAILURES ===================================
FAILED tests/test_random_task2_model.py::test_pinned_model_supports_both_condition_widths_and_keeps_shape[feature4-4]
FAILED tests/test_random_task2_model.py::test_pinned_model_supports_both_condition_widths_and_keeps_shape[feature5_mask-5]
FAILED tests/test_random_task2_cli.py::test_config_accepts_both_modes_and_hashes_them_differently
FAILED tests/test_random_task2_cli.py::test_train_and_evaluate_parsers_accept_both_modes
4 failed, 5 passed in 10.12s
```

This was the expected post-red state after the new sampler/flow modules existed but mode plumbing and builder integration still did not.

### Focused suite after implementation

Command:

```powershell
& 'D:\Anaconda3\envs\radioflow-win\python.exe' -m pytest -q tests/test_random_task2_flow.py tests/test_random_task2_model.py tests/test_random_task2_sampling.py tests/test_random_task2_cli.py
```

Output:

```text
.........                                                                [100%]
9 passed in 9.95s
```

### Focused suite plus closest existing regression coverage

Command:

```powershell
& 'D:\Anaconda3\envs\radioflow-win\python.exe' -m pytest -q tests/test_random_task2_flow.py tests/test_random_task2_model.py tests/test_random_task2_sampling.py tests/test_random_task2_cli.py tests/test_sparse_task2_flow.py tests/test_sparse_task2_model_factory.py tests/test_sparse_task2_sampling.py
```

Output:

```text
.........................                                                [100%]
25 passed in 9.84s
```

## Checkpoint compatibility notes

- Regression runs remain isolated under their existing `.../<variant>/regression` run directory.
- Pinned-FM runs now live under `.../<variant>/pinned_fm`.
- Config SHA already includes `mode`, so regression and pinned-FM checkpoints cannot be resumed interchangeably.
- Checkpoint schema version remains `1`; compatibility separation is by mode-specific config hash and run path rather than a schema bump.
- I did not run a real checkpoint resume against dataset-backed training in this task; compatibility here is based on the preserved checkpoint payload format and mode/hash isolation.

## Concerns / follow-up notes

- I did not run a long GPU experiment, as requested.
- I also did not run a real dataset-backed training/evaluation smoke through the new pinned-FM path; verification here is code-path and targeted-test based.
- The pinned-FM sampler/evaluator currently uses fixed `cfg_scale=1.0` and `steps=2`, matching the brief’s deterministic 2-step projected Euler requirement.
