# Multi-Config SRM Array Benchmark Design

## 1. Objective

Adapt the open-source RadioFlow implementation to the public dataset released with arXiv:2603.06401, then train and evaluate six independent static radio-map (SRM) models:

| Array | Lite | Large |
| --- | --- | --- |
| 8x8 UPA | one model | one model |
| 16x16 UPA | one model | one model |
| 32x32 UPA | one model | one model |

The benchmark must isolate array size as the experimental variable. All runs therefore use 6.7 GHz, the same eight steering angles, identical scene splits, identical target preprocessing, and the original 256x256 RadioFlow output resolution.

This phase covers static SRM only. It does not claim to evaluate RadioFlow's dynamic radio-map (DRM) task because the released multi-configuration dataset does not contain vehicle masks or ray-traced labels with dynamic vehicle obstruction.

## 2. Reproducibility Boundary

- RadioFlow repository: `E:\RadioFlow`.
- Python environment: `D:\Anaconda3\envs\radioflow-win` (Python 3.10, PyTorch 2.5.1+cu121).
- Target device: NVIDIA RTX A2000 Laptop GPU with 8 GiB VRAM.
- Dataset root: `E:\datasets\MultiConfigRadiomap` outside the Git repository.
- Dataset source revision: Hugging Face revision `49ca1dcebe2caa2b2112e6c862132243a992b00a`.
- Reference code revision: GitHub commit `f64e22a578933aa0ba57850ab2c7cf0695063c90`.
- Only `Dataset_20260306164917.zip` is required. The released UNet/GAN checkpoints are not RadioFlow checkpoints and will not be used.
- Every run records the RadioFlow Git commit, dataset revision, manifest checksum, split checksum, complete configuration, and random seed.

## 3. Dataset Selection

### 3.1 Fixed physical configuration

All arrays use carrier frequency 6.7 GHz and the following common steering angles:

```text
-28, -21, -14, -7, 0, 7, 14, 21 degrees
```

The corresponding released beam identifiers are:

| Array | Base configuration | Selected beam IDs |
| --- | --- | --- |
| 8x8 | 64TR, 8 beams | 00, 01, 02, 03, 04, 05, 06, 07 |
| 16x16 | 256TR, 16 beams | 00, 02, 04, 06, 08, 10, 12, 14 |
| 32x32 | 1024TR, 64 beams | 04, 11, 18, 25, 32, 39, 46, 53 |

The manifest builder must verify array rows and columns from the released configuration/settings files rather than trusting total transmitter element count alone.

Each array contributes exactly 800 scenes x 8 beams = 6,400 samples.

### 3.2 Fixed scene-disjoint split

The split ratio is fixed at 70% training, 10% validation, and 20% testing:

| Split | Scenes | Samples per array |
| --- | ---: | ---: |
| Train | 560 | 4,480 |
| Validation | 80 | 640 |
| Test | 160 | 1,280 |

Scene IDs are shuffled once with seed 42 and stored in `scene_split_seed42.json`. The same stored scene IDs are reused by all arrays and both model sizes. No scene may occur in more than one split, and different beams from one scene may not cross split boundaries.

The generated files are:

```text
E:\datasets\MultiConfigRadiomap\manifests\scene_split_seed42.json
E:\datasets\MultiConfigRadiomap\manifests\manifest_8x8.jsonl
E:\datasets\MultiConfigRadiomap\manifests\manifest_16x16.jsonl
E:\datasets\MultiConfigRadiomap\manifests\manifest_32x32.jsonl
```

The released `metadata.csv` is not used as an index. Manifests are built from extracted directories plus configuration and beam-setting files.

## 4. Sample Representation

### 4.1 Condition and target

Each dataset item returns:

```python
condition, target, valid_mask, metadata
```

The condition tensor has three channels at 256x256 resolution, ordered as:

```text
[transmitter mask, normalized building height, normalized beam map]
```

- The transmitter mask uses the released transmitter position `(127, 127)`.
- Height maps are already 256x256.
- Beam maps are linearly resized from 128x128 to 256x256.
- Radiomap targets are linearly resized from 128x128 to 256x256.
- Categorical validity/building masks are resized with nearest-neighbor interpolation.

### 4.2 Normalization and masking

- The training scenes alone determine the maximum height used for height normalization. The scalar is saved in the run configuration and reused unchanged for validation and testing.
- Valid beam-map and target values are clipped to `[-300, 0]` dB and normalized with `y = (dB + 300) / 300`.
- In released labels, `1000` denotes building cells, `-300` denotes invalid/no-label cells, and `-300 < value < 0` denotes valid propagation values.
- Building and invalid cells are set to zero in the normalized target but excluded from all losses and reported accuracy metrics by `valid_mask`.
- Metadata includes the scene ID, array dimensions, beam ID, steering angle, configuration ID, frequency, and source paths.

## 5. Model and Training

### 5.1 Model boundary

The existing `DiffUNet` remains the model family. It receives `condition` with three channels, a one-channel flow state `x_t`, and time `t`, and predicts a one-channel velocity field. Array dimensions are not encoded as image dimensions; the configuration-specific beam map provides the physical spatial condition.

The existing feature definitions remain unchanged:

```text
Lite:  (32, 32, 64, 128, 256, 32)
Large: (128, 128, 256, 512, 1024, 128)
```

### 5.2 Masked flow-matching objective

Training uses conditional flow matching and minimizes mean squared velocity error only on valid target cells:

```text
loss = sum(valid_mask * (predicted_velocity - target_velocity)^2)
       / max(sum(valid_mask), 1)
```

A batch with no valid cell is a data error and terminates the run.

### 5.3 Shared optimizer recipe

```text
optimizer: AdamW
learning rate: 1e-3
weight decay: 1e-5
warmup: first 10% of optimizer steps
scheduler: cosine decay after warmup
EMA decay: 0.999
classifier-free condition dropout: 0.25
effective batch size: 16
maximum epochs: 200
early stopping: 20 validation epochs without dB-RMSE improvement
selection metric: validation valid-region dB-RMSE
seed: 42
DataLoader workers: 2
resolution: 256x256
```

Lite uses micro-batch 2 and eight-step gradient accumulation. Large uses micro-batch 1, sixteen-step gradient accumulation, automatic mixed precision, and gradient checkpointing. Lite also uses automatic mixed precision.

No run may silently change resolution, model width, selected scenes, selected beams, or effective batch size.

### 5.4 Execution order and recovery

1. Validate manifests and one decoded sample per split/array.
2. Run CPU unit tests.
3. Run one GPU forward/backward/update smoke batch for each model size.
4. Train each Lite model for five pilot epochs and verify loss, validation, checkpoint, and resume behavior.
5. Train the three Lite models to the common stopping rule.
6. Enable checkpointing and run the Large memory smoke test.
7. Train the three Large models only if full-resolution forward/backward succeeds.

Every run saves `best.pt`, `last.pt`, optimizer/scheduler/EMA/scaler state, RNG state, epoch/step counters, `config.json`, `metrics.csv`, and manifest/split checksums. Resume must restore all these states.

If Large still exhausts 8 GiB VRAM at micro-batch 1 with AMP and checkpointing, the Large experiment is reported as hardware-blocked with the captured error and peak-memory evidence. It is not replaced by a smaller or lower-resolution model.

## 6. Evaluation

The validation set selects the best epoch and CFG scale. The fixed candidate grid is `1.0, 1.5, 2.0, 2.5`. Test data is evaluated once after all model choices are frozen.

Generation uses the best EMA checkpoint, a two-step Euler configuration, 256x256 resolution, and batch size one for latency measurement.

Primary valid-region metrics are:

- dB-RMSE;
- dB-MAE.

Compatibility and perceptual metrics are:

- normalized MSE and NMSE;
- normalized PSNR with data range 1;
- masked-map SSIM.

Efficiency measurements are:

- parameter count;
- checkpoint size;
- peak allocated VRAM;
- batch-one latency after warm-up, reported with median and 95th percentile.

Metrics are reported overall and separately for every steering angle. Visual comparisons use the same stored test scenes and angles across all six models and contain height map, beam map, ground truth, prediction, and absolute dB error.

Output layout:

```text
results/srm_6.7ghz_common8/<array>/<model_size>/
  metrics_test.json
  metrics_per_beam.csv
  predictions/
  comparisons/
  error_maps/
  runtime.json
  run_manifest.json
```

The final report contains one six-row comparison table for 8x8/16x16/32x32 Lite and Large.

## 7. Code Structure

The original RadioMapSeer entry points remain available. Multi-configuration support is added through focused modules:

```text
data_loaders/multiconfig.py
experiments/multiconfig_manifest.py
training/masked_flow_loss.py
evaluation/radiomap_metrics.py
train_multiconfig.py
evaluate_multiconfig.py
tests/
```

Existing model files receive only the changes required for three condition channels and optional Large gradient checkpointing. Dataset-specific parsing, metrics, and experiment orchestration do not go into the model module.

## 8. Failure Handling

The pipeline fails immediately for missing files, duplicate manifest keys, inconsistent configuration/beam pairing, unexpected shapes, non-finite values, empty valid masks, overlapping scene splits, checkpoint/model mismatch, or absent checkpoints. Evaluation must never continue with randomly initialized weights.

Out-of-memory handling is limited to the predeclared micro-batch, AMP, accumulation, and checkpointing strategy. Any additional change requires a new explicit experiment configuration and may not overwrite comparable results.

## 9. Verification and Acceptance Criteria

Automated tests must prove:

- exactly 6,400 samples per array and 4,480/640/1,280 per split;
- identical scene IDs across arrays and model sizes;
- pairwise-disjoint train/validation/test scene sets;
- exact selected beam IDs and angles;
- condition shape `(3, 256, 256)` and target/mask shape `(1, 256, 256)`;
- correct interpolation modes;
- reversible dB normalization on valid values;
- train-only height normalization reuse;
- invalid/building cells do not affect loss or metrics;
- masked loss agrees with a hand-computed example;
- synthetic perfect predictions yield zero dB error and expected image metrics;
- checkpoint save/resume restores optimizer, scheduler, EMA, scaler, counters, and RNG state;
- missing or incompatible checkpoints fail evaluation;
- a one-batch train-save-resume-evaluate integration path succeeds.

The implementation phase is accepted when the data adapter and manifests pass all tests, Lite completes the staged training/evaluation path, and Large either completes the same path or is accompanied by reproducible full-resolution OOM evidence under the approved memory strategy.

## 10. Non-Goals

- Dynamic vehicle-aware DRM generation or evaluation.
- Sparse-measurement reconstruction.
- Cross-array joint training or cross-configuration zero-shot generalization.
- New Sionna ray tracing or modification of released propagation labels.
- Comparison using unequal frequencies, unequal steering-angle sets, or random image-level splits.
