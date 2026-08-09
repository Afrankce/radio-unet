# 8x8 Cross-Frequency RadioFlow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separately auditable 8x8 cross-frequency RadioFlow experiment that trains and validates on 4.9 GHz and tests on 6.7 GHz, while leaving the completed fixed-frequency 6.7 GHz benchmark unchanged.

**Architecture:** Add a cross-frequency manifest/data contract and dedicated train/evaluate entry points. Reuse the locked three-channel condition, Lite `DiffUNet`, masked conditional flow matching, EMA, CFG, two-step Euler sampler, deterministic noise, and metric definitions. The first run keeps frequency implicit in the frequency-specific beam map; no fourth frequency channel is introduced.

**Tech Stack:** Python 3.10, PyTorch, NumPy, MONAI, torchcfm, pytest, the existing RadioFlow model/checkpoint/sampling utilities, and PowerShell. All Python commands use `D:\Anaconda3\envs\radioflow-win\python.exe`.

## Global Constraints

- Work only in `E:\RadioFlow-worktrees\multiconfig-srm-01x`; preserve all existing user modifications and untracked scripts.
- Do not modify the fixed-frequency manifest, trainer contract, checkpoints, or result tree under `E:\RadioFlow\results\srm_6.7ghz_common8_refmatch_lite_earlystop_eval`.
- Scientific protocol is fixed: array `8x8`, 8x8/64TR, 256x256 model tensors, seed 42, train 560 scenes at 4.9 GHz, validation 80 scenes at 4.9 GHz, test 160 scenes at 6.7 GHz, and steering angle `0.0°` only.
- The released source uses different beam IDs for the same zero-degree beam: 4.9 GHz uses beam ID `0`, while 6.7 GHz's eight-beam configuration uses beam ID `4`. The implementation must select by steering angle and record the actual ID; it must not assume that beam ID `0` means `0°`.
- Reuse the existing `scene_split_seed42.json` without regenerating it. The resulting claim is joint cross-frequency and scene-disjoint generalization, not pure frequency-only transfer.
- Conditions remain `[Tx mask, normalized height, frequency-specific normalized beam map]`; targets and valid masks retain the existing `[-300, 0]` dB mapping and sentinel handling.
- Generated manifests, checkpoints, predictions, metrics, visualizations, and runtime files stay outside the repository. The expected roots are `E:\RadioFlow\runs\srm_crossfreq_8x8_49train_67test_lite` and `E:\RadioFlow\results\srm_crossfreq_8x8_49train_67test_lite`.
- Follow red-green-refactor for every behavior change: add a focused failing test, run it and observe the expected failure, implement the smallest complete behavior, run the focused test green, then run earlier relevant tests.
- Before any completion claim, run focused tests, the non-dataset regression suite, `git diff --check`, and a real-data preflight/smoke. Never silently skip a failed check.

---

### Task 1: Add the cross-frequency protocol and manifest builder

**Files:**

- Create: `experiments/cross_frequency.py`
- Create: `tests/test_cross_frequency_manifest.py`
- Create: `prepare_cross_frequency.py`
- Create: `tests/test_prepare_cross_frequency_cli.py`

- [ ] **Step 1: Write failing manifest-selection tests**

  Use a fake `SampleInventory`-compatible object and a minimal schema configuration list. Prove that the selector chooses the unique 4.9 GHz `8x8/64TR/0°` configuration and the unique 6.7 GHz `8x8/64TR` configuration's `0°` beam ID `4`. Prove that it rejects a missing configuration, two matching configurations, an angle mismatch, and a configuration with the wrong shape or element count. Prove that the generated records contain exactly 560 train, 80 validation, and 160 test samples, with train/validation at 4.9 GHz, test at 6.7 GHz, no scene overlap, one record per scene, matching beam-map/radiomap configuration IDs, and distinct sample keys.

  Exercise the public interfaces planned for the implementation:

  ```python
  spec = cross_frequency_spec()
  selected = select_zero_degree_configurations(schema)
  records = build_cross_frequency_records(inventory, split, selected, spec)
  validate_cross_frequency_records(records, split, selected, spec)
  ```

- [ ] **Step 2: Run the focused tests and confirm the red state**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" -m pytest tests/test_cross_frequency_manifest.py -q
  ```

  Expected failure: `experiments.cross_frequency` and its public selector do not yet exist.

- [ ] **Step 3: Implement the pure selector, record builder, and strict validator**

  Define immutable constants and dataclasses for the protocol. Select configurations from the audited schema's `configurations` entries, requiring exactly one match per frequency with `rows=8`, `cols=8`, `tx_elements=64`, and the requested frequency. Select the beam by `steering_deg` with a finite-angle comparison; retain the released beam ID (`0` for 4.9 GHz and `4` for 6.7 GHz). Resolve every `(config_id, beam_id, scene_id)` through `inventory.require_unique_triplet` and create existing `ManifestRecord` objects with a sample key containing scene, array, frequency, angle, and beam ID.

  `validate_cross_frequency_records` must check exact counts, split frequencies, shape, angle, beam IDs, one-to-one source paths, no scene overlap, unique sample/logical keys, and equality with a deterministic rebuild from the same inventory. It must not call or loosen the fixed `validate_manifest` contract.

- [ ] **Step 4: Rerun the focused tests and the existing manifest tests**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" -m pytest tests/test_cross_frequency_manifest.py tests/test_multiconfig_manifest.py -q
  git diff --check
  ```

- [ ] **Step 5: Add the real-data manifest command and its parser test**

  Add `build-manifest` to `prepare_cross_frequency.py` with fixed protocol controls and arguments for `--dataset-root`, `--schema`, `--manifest-dir`, and optional output path. Load the pinned schema and scene split, build and validate the manifest, then publish canonical JSONL atomically. The command must fail if the existing output has different bytes. Its summary must include counts, selected configuration IDs/beam IDs, manifest SHA-256, scene-split SHA-256, and the dataset revision.

  The CLI test must prove the command exposes no frequency, array, beam, or split-size override and that `main([...])` dispatches to the builder.

- [ ] **Step 6: Run the CLI test and commit the manifest layer**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" -m pytest tests/test_cross_frequency_manifest.py tests/test_prepare_cross_frequency_cli.py -q
  git diff --check
  git add experiments/cross_frequency.py prepare_cross_frequency.py tests/test_cross_frequency_manifest.py tests/test_prepare_cross_frequency_cli.py
  git commit -m "feat: add 8x8 cross-frequency manifest contract"
  ```

---

### Task 2: Add a three-channel cross-frequency dataset adapter

**Files:**

- Create: `data_loaders/cross_frequency.py`
- Create: `tests/test_cross_frequency_dataset.py`

- [ ] **Step 1: Write failing dataset tests**

  Build a temporary miniature dataset with 256x256 float32 heights, 128x128 float64 beam maps, and 128x128 float32 targets containing valid cells plus `-300` and `1000` sentinels. Use manifest records from Task 1 with a small test-only expected-count override in the constructor fixture. Assert that one decoded item has condition shape `(3,256,256)`, target shape `(1,256,256)`, boolean valid mask, transmitter pixel `(127,127)`, and metadata containing frequency, actual beam ID, and angle. Assert that the target frequency and beam-map frequency must agree. Add failures for wrong array geometry, wrong split frequency, wrong beam angle, unsafe paths, unknown target values, empty valid masks, and a non-positive height maximum.

  Exercise:

  ```python
  CrossFrequencyRadiomapDataset(
      dataset_root=..., manifest_path=..., split="train",
      height_max=..., expected_frequency_hz=4_900_000_000,
      expected_counts={"train": ..., "val": ..., "test": ...},
  )
  ```

- [ ] **Step 2: Run the focused dataset tests and confirm red**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" -m pytest tests/test_cross_frequency_dataset.py -q
  ```

- [ ] **Step 3: Implement the adapter by reusing the locked decoding primitives**

  Implement a separate `CrossFrequencyRadiomapDataset` that loads `ManifestRecord`s, validates the cross-frequency record contract, reads the train-only height maximum, and reuses `_safe_path`, `_load_npy`, `normalize_db`, `prepare_target`, `resize_continuous`, `resize_valid_mask`, `build_tx_mask`, and `multiconfig_collate` from `data_loaders.multiconfig`. Keep the source shapes/dtypes from the schema metadata when supplied, with the released defaults as the strict fallback. Do not add a fourth frequency channel. Cache decoded samples using the same immutable-record semantics as the fixed loader.

  Add `load_cross_frequency_height_max` to validate the existing train-only height artifact: schema version 1, `derived_from="train"`, 560 unique train scenes, finite positive maximum, and a split hash matching `scene_split_seed42.json`. It may reuse the already computed height maximum because the cross-frequency train/validation scenes are the same fixed split; it must not recompute normalization from validation or test files.

- [ ] **Step 4: Rerun focused and baseline data tests**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" -m pytest tests/test_cross_frequency_dataset.py tests/test_multiconfig_dataset.py tests/test_radiomap_metrics.py -q
  git diff --check
  ```

- [ ] **Step 5: Commit the dataset adapter**

  ```powershell
  git add data_loaders/cross_frequency.py tests/test_cross_frequency_dataset.py
  git commit -m "feat: decode cross-frequency radiomap samples"
  ```

---

### Task 3: Add cross-frequency training configuration, preflight, and CLI

**Files:**

- Create: `training/cross_frequency_config.py`
- Create: `training/cross_frequency_trainer.py`
- Create: `train_cross_frequency.py`
- Create: `tests/test_cross_frequency_training.py`
- Create: `tests/test_train_cross_frequency_cli.py`

- [ ] **Step 1: Write failing configuration and preflight tests**

  Test that `CrossFrequencyTrainConfig` locks `array_size="8x8"`, condition channels `3`, frequencies `4.9/4.9/6.7 GHz`, counts `560/80/160`, seed `42`, resolution `256`, Lite feature choice, and the existing optimizer/EMA/flow-matching controls. Test that it rejects changes to those scientific fields. Test that preflight builds datasets with 560/80/160 samples, all decoded conditions have three channels, and the context records manifest/split/schema/archive/dataset identity hashes.

  Test the loader builder's micro-batch and optimizer-step counts without constructing a full model. Test run-config serialization/resume identity is canonical and rejects a changed manifest or protocol.

- [ ] **Step 2: Run the focused tests and confirm red**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" -m pytest tests/test_cross_frequency_training.py -q
  ```

- [ ] **Step 3: Implement the cross-frequency configuration and context**

  Create a frozen config with the same tunable/runtime fields as `MultiConfigTrainConfig`, plus fixed `manifest_path`, `height_stats_path`, `train_frequency_hz`, `val_frequency_hz`, and `test_frequency_hz`. Its `run_dir` is the supplied cross-frequency run root, and its `scientific_payload`, `config_sha256`, `to_record`, and `from_json` must include every scientific control and identity needed for strict resume. Keep `train_scale` fixed at `1.0` for this first experiment.

  Implement `preflight_cross_frequency` to load the schema, cross-frequency manifest, and train-only height statistic, construct the dedicated datasets, enforce counts and tensor shapes, verify the source identities against the pinned checkout, and return a context with all relevant hashes. Implement a loader builder with the existing Lite micro-batch/accumulation recipe and validation batch size one.

- [ ] **Step 4: Reuse the existing trainer core without changing the fixed benchmark**

  Instantiate the existing `MultiConfigSRMTrainer` with the cross-frequency config/context/loaders, because its flow-matching loss, EMA, checkpoint state, deterministic validation noise, and training loop operate on the config protocol rather than the fixed dataset class. Add only a small generic run-config writer/validator and a cross-frequency checkpoint-identity builder where the existing helper's fixed type checks or field names prevent reuse. Do not change the fixed trainer's eight-beam or 640-sample assumptions; the new path must own the 80-sample validation checks.

- [ ] **Step 5: Implement `train_cross_frequency.py`**

  Expose only `--dataset-root`, `--manifest-path`, `--height-stats-path`, `--run-root`, `--model-size` (default `lite`), `--device`, `--resume`, `--stop-after-epoch`, `--smoke-optimizer-steps`, and `--preflight-only`. Scientific values must not be command-line overrides. Support `none`, `auto`, or an explicit checkpoint path for resume, and print a JSON summary. Use the locked model factory and existing `seed_everything`, checkpoint, optimizer, and EMA utilities.

- [ ] **Step 6: Run focused, CLI, and fixed-training regression tests**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" -m pytest tests/test_cross_frequency_training.py tests/test_train_cross_frequency_cli.py tests/test_multiconfig_train_integration.py tests/test_train_multiconfig_cli.py -q
  git diff --check
  ```

- [ ] **Step 7: Commit training support**

  ```powershell
  git add training/cross_frequency_config.py training/cross_frequency_trainer.py train_cross_frequency.py tests/test_cross_frequency_training.py tests/test_train_cross_frequency_cli.py
  git commit -m "feat: train 8x8 cross-frequency RadioFlow"
  ```

---

### Task 4: Add generic frequency/angle metric grouping and evaluation

**Files:**

- Modify: `evaluation/radiomap_metrics.py`
- Create: `tests/test_cross_frequency_metrics.py`
- Create: `evaluation/cross_frequency_evaluator.py`
- Create: `evaluate_cross_frequency.py`
- Create: `tests/test_cross_frequency_evaluate_cli.py`
- Create: `tests/test_cross_frequency_evaluator.py`

- [ ] **Step 1: Write failing grouped-metric tests**

  Add tests for a `PerFrequencyMetricAccumulators` class that groups by `(frequency_hz, steering_deg)`, accumulates the same masked global metrics as `MetricAccumulator`, emits deterministic rows with `frequency_hz` and `angle_deg`, rejects unknown groups, inconsistent metadata, missing metadata, and empty groups, and preserves the existing `PerBeamMetricAccumulators` behavior unchanged.

- [ ] **Step 2: Run the focused metric test and confirm red**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" -m pytest tests/test_cross_frequency_metrics.py -q
  ```

- [ ] **Step 3: Implement the grouped accumulator**

  Add the smallest generic accumulator around `MetricAccumulator`. It must use metadata frequency and steering angle as the group key, allow an explicit expected-group set, and return rows ordered by frequency then angle. No metric formula or valid-pixel policy may diverge from `MetricAccumulator`.

- [ ] **Step 4: Write failing evaluator/CLI tests**

  With mocked loaders/models/checkpoints, test that CFG selection evaluates exactly 80 4.9 GHz validation records for each candidate, records the selected epoch and identity hashes, and freezes the selection. Test that test evaluation reads the frozen selection, evaluates exactly 160 6.7 GHz records, writes `metrics_test.json`, `metrics_per_frequency.csv`, `predictions/`, selected visualizations, runtime evidence, and a canonical `run_manifest.json`. Test that rerunning a completed result fails. Test the CLI subcommands `select-cfg` and `test` and prove they do not expose protocol overrides.

- [ ] **Step 5: Run evaluator tests and confirm red**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" -m pytest tests/test_cross_frequency_evaluator.py tests/test_cross_frequency_evaluate_cli.py -q
  ```

- [ ] **Step 6: Implement cross-frequency evaluation**

  Reuse `CFG_CANDIDATES`, the existing deterministic noise function, `euler_cfg_sample`, EMA-only checkpoint loading, `MetricAccumulator`, result transactions, visualization helpers, and runtime benchmarking where their fixed eight-beam/640-sample validation assumptions do not apply. Implement cross-frequency-specific selection payload validation for `n_samples=80`, and test payload validation for `n_samples=160`. Use CFG selection only on 4.9 GHz validation; use the frozen scale once on 6.7 GHz test.

  Write `metrics_per_frequency.csv` with at least `frequency_hz,angle_deg,n_samples,n_valid_pixels,db_rmse,db_mae,mse,nmse,psnr,ssim`. Record the actual source beam IDs and frequency/config IDs in result metadata. Render a small deterministic set of 0° test comparisons/errors, label them `6.7 GHz / 0°`, and save all test predictions for auditability.

- [ ] **Step 7: Rerun focused evaluation and all non-dataset regression tests**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" -m pytest tests/test_cross_frequency_metrics.py tests/test_cross_frequency_evaluator.py tests/test_cross_frequency_evaluate_cli.py tests/test_radiomap_metrics.py tests/test_multiconfig_evaluate_integration.py tests/test_multiconfig_evaluate_cli.py -q
  & "D:\Anaconda3\envs\radioflow-win\python.exe" -m pytest -m "not dataset and not gpu and not slow" -q
  git diff --check
  ```

- [ ] **Step 8: Commit evaluation support**

  ```powershell
  git add evaluation/radiomap_metrics.py evaluation/cross_frequency_evaluator.py evaluate_cross_frequency.py tests/test_cross_frequency_metrics.py tests/test_cross_frequency_evaluator.py tests/test_cross_frequency_evaluate_cli.py
  git commit -m "feat: evaluate cross-frequency radiomap transfer"
  ```

---

### Task 5: Build the real manifest, run preflight, and perform a GPU smoke/resume check

**Files:**

- Generated outside the repository: `E:\datasets\MultiConfigRadiomap\manifests\manifest_cross_frequency_8x8.jsonl`
- Generated outside the repository: `E:\RadioFlow\runs\srm_crossfreq_8x8_49train_67test_lite\`
- No source change is expected unless a real-data discrepancy first receives a regression test.

- [ ] **Step 1: Build and independently inspect the real manifest**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" prepare_cross_frequency.py build-manifest --dataset-root E:\datasets\MultiConfigRadiomap --schema E:\RadioFlow-worktrees\multiconfig-srm-01x\experiments\multiconfig_schema.json --manifest-dir E:\datasets\MultiConfigRadiomap\manifests
  ```

  Independently load the JSONL and verify 560/80/160 counts, train/validation `4.9 GHz`, test `6.7 GHz`, angle `0°`, actual beam IDs `{0,4}` by frequency, disjoint scene sets, and exact one-to-one path/config pairing. Record the published manifest SHA-256 in the run protocol.

- [ ] **Step 2: Run real-data preflight**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" train_cross_frequency.py --dataset-root E:\datasets\MultiConfigRadiomap --manifest-path E:\datasets\MultiConfigRadiomap\manifests\manifest_cross_frequency_8x8.jsonl --height-stats-path E:\datasets\MultiConfigRadiomap\manifests\height_stats_train.json --run-root E:\RadioFlow\runs\srm_crossfreq_8x8_49train_67test_lite --model-size lite --device cpu --resume none --preflight-only
  ```

  The preflight-only path must construct and validate all three datasets without constructing or training a model. Confirm every split decodes at least one sample and every identity hash is recorded.

- [ ] **Step 3: Run one complete Lite optimizer-step GPU smoke**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" train_cross_frequency.py --dataset-root E:\datasets\MultiConfigRadiomap --manifest-path E:\datasets\MultiConfigRadiomap\manifests\manifest_cross_frequency_8x8.jsonl --height-stats-path E:\datasets\MultiConfigRadiomap\manifests\height_stats_train.json --run-root E:\RadioFlow\runs\srm_crossfreq_8x8_49train_67test_lite --model-size lite --device cuda:0 --resume none --smoke-optimizer-steps 1
  ```

  Verify the condition is three-channel, the loss is finite, ten Lite accumulation windows produce one optimizer step for the 560-sample train split, EMA/scheduler/checkpoint state is saved, and the checkpoint reloads with the same identity.

- [ ] **Step 4: Run a pause/resume probe**

  Run one short epoch or stop point, rerun with `--resume auto`, and verify history/counters continue without duplicate rows or an identity mismatch. Keep this smoke state separate from the production full-run directory if the implementation uses immutable completion artifacts.

- [ ] **Step 5: Run all relevant regression tests before full training**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" -m pytest -m "not dataset and not gpu and not slow" -q
  git diff --check
  git status --short
  ```

  If real data exposes a format mismatch, stop, add a minimal failing fixture/test, make the narrow fix, and rerun the earlier task gates.

---

### Task 6: Train Lite for up to 1000 epochs, freeze CFG, evaluate once, and publish the report

**Files:**

- Generated outside the repository: `E:\RadioFlow\runs\srm_crossfreq_8x8_49train_67test_lite\`
- Generated outside the repository: `E:\RadioFlow\results\srm_crossfreq_8x8_49train_67test_lite\`
- Create: `docs/cross-frequency-8x8.md`

- [ ] **Step 1: Start or resume the full 1000-epoch Lite run**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" train_cross_frequency.py --dataset-root E:\datasets\MultiConfigRadiomap --manifest-path E:\datasets\MultiConfigRadiomap\manifests\manifest_cross_frequency_8x8.jsonl --height-stats-path E:\datasets\MultiConfigRadiomap\manifests\height_stats_train.json --run-root E:\RadioFlow\runs\srm_crossfreq_8x8_49train_67test_lite --model-size lite --device cuda:0 --resume auto
  ```

  Allow the locked early-stopping rule to terminate sooner only when its recorded patience condition is met. If the process is interrupted, resume with the same command and verify the last checkpoint identity before continuing. Do not claim 1000 epochs until the runtime/config state shows either `completed` at epoch 1000 or a valid early-stop event.

- [ ] **Step 2: Select CFG from the frozen 4.9 GHz validation set**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" evaluate_cross_frequency.py select-cfg --dataset-root E:\datasets\MultiConfigRadiomap --manifest-path E:\datasets\MultiConfigRadiomap\manifests\manifest_cross_frequency_8x8.jsonl --height-stats-path E:\datasets\MultiConfigRadiomap\manifests\height_stats_train.json --run-root E:\RadioFlow\runs\srm_crossfreq_8x8_49train_67test_lite --results-root E:\RadioFlow\results\srm_crossfreq_8x8_49train_67test_lite --model-size lite --device cuda:0
  ```

  Confirm all four candidates evaluate exactly 80 4.9 GHz records with deterministic noise and that the selected epoch matches `best.pt`.

- [ ] **Step 3: Evaluate the frozen 6.7 GHz test set exactly once**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" evaluate_cross_frequency.py test --dataset-root E:\datasets\MultiConfigRadiomap --manifest-path E:\datasets\MultiConfigRadiomap\manifests\manifest_cross_frequency_8x8.jsonl --height-stats-path E:\datasets\MultiConfigRadiomap\manifests\height_stats_train.json --run-root E:\RadioFlow\runs\srm_crossfreq_8x8_49train_67test_lite --results-root E:\RadioFlow\results\srm_crossfreq_8x8_49train_67test_lite --model-size lite --device cuda:0
  ```

  Validate exactly 160 predictions, test frequency `6.7 GHz`, angle `0°`, actual beam ID `4`, overall masked dB-RMSE/MAE, MSE/NMSE/PSNR/SSIM, `metrics_per_frequency.csv`, runtime evidence, selected visualizations, and canonical artifact hashes. A completed result must reject a second test run.

- [ ] **Step 4: Write the experiment report**

  Create `docs/cross-frequency-8x8.md` with the exact protocol, the joint-generalization caveat, source/config/manifest hashes, train/validation history summary, selected epoch and CFG, final overall and grouped test metrics, comparison to the fixed-frequency 6.7 GHz 8x8 Lite baseline, and links to the external run/result roots. Include condition channels and the beam-ID mapping `4.9 GHz:0 → 0°`, `6.7 GHz:4 → 0°` so the report is not ambiguous.

- [ ] **Step 5: Final verification and handoff**

  ```powershell
  & "D:\Anaconda3\envs\radioflow-win\python.exe" -m pytest -m "not dataset and not gpu and not slow" -q
  git diff --check
  git status --short
  ```

  Confirm no fixed benchmark artifact changed, no generated checkpoint/result was staged, the new report matches the JSON/CSV receipts, and the final handoff explicitly distinguishes completed, early-stopped, paused, or blocked training.

---

## Acceptance Checklist

- [ ] Cross-frequency manifest has exactly 560/80/160 records with 4.9/4.9/6.7 GHz split assignments and no scene overlap.
- [ ] The selected zero-degree beam is paired by angle, preserving actual IDs `0` at 4.9 GHz and `4` at 6.7 GHz.
- [ ] Every decoded condition is `(3,256,256)` and uses the frequency-specific beam map; no fourth channel is present.
- [ ] Fixed-frequency benchmark tests and non-dataset regression tests remain green.
- [ ] Lite smoke/resume passes with finite loss and strict checkpoint identity.
- [ ] Full training state truthfully records 1000 epochs or a valid early-stop/interruption status.
- [ ] CFG selection uses only the 80-record 4.9 GHz validation split; test uses the frozen scale on exactly 160 6.7 GHz records.
- [ ] Test outputs include overall masked metrics, frequency/angle grouped metrics, predictions, labeled visualizations, runtime evidence, and a canonical run manifest.
- [ ] The final report calls the result joint cross-frequency and scene-disjoint generalization and compares it with the existing 6.7 GHz 8x8 Lite baseline.
