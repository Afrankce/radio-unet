# 8x8 Cross-Frequency RadioFlow Experiment Design

## 1. Objective

Add a separate 8x8 cross-frequency experiment to the existing Multi-config RadioFlow benchmark without changing the completed fixed-frequency 6.7 GHz baseline.

The first experiment will use the released 8x8/64TR configurations and the common 0-degree beam only:

| Split | Frequency | Scenes | Purpose |
| --- | ---: | ---: | --- |
| Train | 4.9 GHz | 560 | Fit the RadioFlow model |
| Validation | 4.9 GHz | 80 | Select epoch and CFG scale |
| Test | 6.7 GHz | 160 | Evaluate frequency transfer on held-out scenes |

Because the test scenes are also disjoint from training and validation scenes, the result is a joint cross-frequency and cross-environment generalization result. It must not be described as pure frequency-only transfer.

## 2. Scope and Non-goals

### In scope

- One Lite RadioFlow model shared by the 4.9 GHz training data.
- 8x8/64TR only.
- Steering angle 0 degrees only. The released 4.9 GHz configuration uses beam ID 0 for this angle; the released 6.7 GHz eight-beam configuration uses beam ID 4.
- Existing three-channel condition: transmitter mask, normalized height, and frequency-specific beam map.
- Existing masked flow-matching loss, EMA, classifier-free guidance, two-step Euler sampling, deterministic evaluation noise, and valid-mask metrics.
- Frequency-grouped test metrics and saved visualizations for 6.7 GHz predictions.
- Regression tests for manifest selection, frequency/target pairing, split counts, and grouped metrics.

### Out of scope

- 16x16 or 32x32 cross-frequency claims; the released data does not provide the same square array configurations across the lower-frequency bands.
- Using the 6.7 GHz nonzero beams in this first experiment.
- Adding a fourth explicit frequency channel. The first run follows the paper-faithful beam-map representation in which frequency is encoded by the frequency-specific beam map. An explicit frequency embedding can be added later as an ablation.
- Modifying or overwriting the existing 6.7 GHz fixed-frequency manifests, checkpoints, results, or benchmark contract.

## 3. Scientific Design

### 3.1 Input and target pairing

Every item must satisfy:

```text
condition = [Tx mask, height map, beam_map(4.9 or 6.7 GHz, 8x8, 0 degrees)]
target    = radiomap(same frequency, 8x8, 0 degrees, same scene)
```

The beam map and target frequency must agree. A frequency-specific beam map is treated as the spatial configuration representation; the raw frequency remains in metadata for auditing and grouping.

### 3.2 Scene split

Reuse the existing `scene_split_seed42.json` IDs. The train and validation manifests contain only 4.9 GHz records, while the test manifest contains only 6.7 GHz records. No scene ID may appear in more than one split. Each split contains exactly one record per selected scene because only the 0-degree beam is retained; the manifest preserves the source beam IDs (0 at 4.9 GHz and 4 at 6.7 GHz).

Expected counts:

```text
train: 560
val:    80
test:  160
```

### 3.3 Normalization and metrics

- Reuse the existing train-only height maximum.
- Keep the global dB mapping `[-300, 0] -> [0, 1]` for both frequencies.
- Exclude building and invalid/sentinel cells using `valid_mask` from loss and all reported metrics.
- Report overall test dB-RMSE, dB-MAE, MSE, NMSE, PSNR, and SSIM.
- Also report frequency and angle metadata in the result manifest, even though the first test has only one frequency and one angle.
- Preserve the existing masked-pixel aggregation definition so the cross-frequency result is comparable to the current baseline metrics.

## 4. Architecture and Code Boundaries

The current fixed-frequency benchmark remains the default path. The cross-frequency run uses a new experiment mode and separate manifests/results root.

### Manifest/data layer

- Add a cross-frequency manifest generator or a narrowly scoped manifest-selection helper that filters the released records by array geometry, frequency, and steering angle.
- Reuse the existing decoder and normalization logic where possible.
- Relax only the new experiment path's assumption that every split contains eight beams; do not remove the locked eight-beam validation from the existing benchmark.

### Training layer

- Add a cross-frequency configuration carrying the selected frequency groups and expected sample counts.
- Keep `condition_channels=3`; no model layer change is needed for the paper-faithful first run.
- Keep the same Lite model, optimizer, EMA, flow-matching objective, early stopping, and inference settings.
- Train only on the 4.9 GHz manifest and validate only on the 4.9 GHz validation manifest.

### Evaluation layer

- Add a cross-frequency test command or mode that loads the 6.7 GHz test manifest.
- Group metrics by frequency and steering angle using generic group keys rather than the fixed eight-beam accumulator.
- Keep the existing test evaluator unchanged for the fixed 6.7 GHz benchmark.

### Results and visualization

Write to a new root such as:

```text
E:\RadioFlow\results\srm_crossfreq_8x8_49train_67test_lite\
```

The result manifest must record the experiment protocol, manifest checksums, model configuration, selected epoch, selected CFG scale, and source dataset revision. Saved visualizations must label the test frequency as 6.7 GHz and the beam as 0 degrees.

## 5. Alternatives Considered

### A. Pure frequency transfer with overlapping scene IDs

Use the same scenes at 4.9 GHz and 6.7 GHz, changing only frequency. This isolates frequency transfer but departs from the existing scene-disjoint benchmark and is deferred to a follow-up experiment.

### B. Explicit frequency channel or embedding

Add `log10(frequency_GHz)` as a constant fourth channel or an MLP/FiLM embedding. This may improve frequency-dependent multipath learning but is an extension beyond the first paper-faithful beam-map experiment. It should be evaluated after the current run as an ablation.

### C. Mixed frequencies and array sizes

Train a single model over all released configurations. This measures joint frequency/array/beam generalization and is useful later, but it cannot support a clean 8x8 cross-frequency claim.

## 6. Acceptance Criteria

- The original fixed-frequency benchmark tests continue to pass unchanged.
- The new manifest contains exactly 560 train, 80 validation, and 160 test records.
- All records are 8x8/64TR and beam angle 0 degrees.
- Train and validation records are 4.9 GHz; test records are 6.7 GHz.
- Every record's beam-map configuration matches its radiomap configuration.
- The model receives three condition channels and the existing flow-matching path remains intact.
- A smoke run completes one optimizer step and one validation batch.
- A full Lite run can resume safely and writes a complete test metrics JSON plus frequency/angle metrics CSV.
- The final report labels the result as joint cross-frequency and scene-disjoint generalization.

## 7. Open Follow-up

After the first run, compare the paper-faithful three-channel model with an explicit log-frequency condition. Then run the same-frequency-scene-overlap protocol to separate frequency transfer from environment transfer.
