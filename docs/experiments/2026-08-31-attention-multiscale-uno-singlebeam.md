# Attention-conditioned multiscale UNO: execution record

## Status and immutable identity

- **Tracked deployment commit:** `8701bedf4a63a011efe058a46ba84331cdbf2ac9`.
- **Experiment:** `same_frequency_6.7_single_beam_attention_multiscale_uno`.
- **Model size / backbone:** `attention_multiscale_uno_lite` / `attention_conditioned_multiscale_uno2d`.
- **Protocol:** independent `8x8`, `16x16`, and `32x32` arrays; 6.7 GHz; 0 degrees; scene-disjoint seed 42 split of 560 train / 80 validation / 160 test scenes; 256 x 256; condition channels `[Tx mask, height, beam map]`; valid mask is metric/loss-only.
- **Locked architecture:** state channels `(32, 64, 128, 256, 256)`; operator width `24`; modes `(12, 12, 8, 4, 4)`; right/bottom padding `(9, 5, 3, 2, 1)`; `BasicUNetEncoder_lite` features `(32, 32, 64, 128, 256, 32)`; nine native-scale CA/SA injections; CFG dropout `0.25`; evaluation CFG candidate `1.0`.
- **Formal-run state:** launched successfully and still running; the runs are independently resumable. CFG selection and test evaluation have not run and remain pending until training ends.

Live local construction with `D:\Anaconda3\envs\radioflow-win\python.exe` verified that `build_attention_multiscale_uno()` and `build_same_frequency_backbone("attention_multiscale_uno_lite")` both return `AttentionMultiscaleUNO2d`, with:

| Quantity | Locked and observed value |
| --- | ---: |
| Tensor parameter elements (complex element counted once) | 3,059,355 |
| Independent real scalar parameters | 3,925,659 |
| `CrossAttention` modules | 9 |
| Condition channels | 3 |

The local representative `8x8` configuration (synthetic paths and `beam_id=0`, used only to exercise construction and round-trip) had config SHA-256 `75bb49db1c65af94f2da87dc2c3b2515dffcb60fe700163fb8129f32deb0e13a`. Its JSON round-trip preserved the hash. This is not a server-run identity: the real per-array hash is determined only after each server manifest supplies its validated beam ID.

## Server provenance, isolation, and GPU allocation

The new server code root is `/home/wys/radioflow_20260823/multiscale-uno-singlebeam`; results are written only below `/home/wys/radioflow_20260823/results/multiscale_uno_samefreq_6.7ghz`.

```text
CODE_ROOT=/home/wys/radioflow_20260823/multiscale-uno-singlebeam
DATASET_ROOT=/home/wys/radioflow_20260823/datasets/MultiConfigRadiomap
RESULT_ROOT=/home/wys/radioflow_20260823/results/multiscale_uno_samefreq_6.7ghz
ENV_FILE=/home/wys/radioflow_20260823/radioflow_remote_env.sh
```

The deployed tracked Git bundle has SHA-256 `39034e4df0e13ea55053569bf8871e106d3914e989d0a16b5fe653cb2a69a701` both locally and remotely. The remote clone was clean at `8701bedf4a63a011efe058a46ba84331cdbf2ac9`. The first preflight correctly rejected the bundle-path origin; only the new clone's `origin` was then corrected to `https://github.com/Hxxxz0/RadioFlow.git`.

The old Attention-FNO roots were unchanged before and after this work: code inode/mtime `68820424` / `1787797706`, and results inode/mtime `68820804` / `1787829027`.

GPU 0 was occupied by another account and was not used. Although the launcher default remains physical GPUs `0,1,2`, every live server operation used `RADIOFLOW_MULTISCALE_UNO_GPUS=1,2,3`; each child sees its assigned physical GPU through `CUDA_VISIBLE_DEVICES` and uses `--device cuda:0`.

| Array | Physical GPU | Formal PID | Manifest | Run root | Evaluation result root |
| --- | ---: | ---: | --- | --- | --- |
| `8x8` | 1 | 1913313 | `$DATASET_ROOT/manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl` | `$RESULT_ROOT/runs/8x8` | `$RESULT_ROOT/results/8x8` |
| `16x16` | 2 | 1913318 | `$DATASET_ROOT/manifests/manifest_samefreq_6.7ghz_16x16_0deg.jsonl` | `$RESULT_ROOT/runs/16x16` | `$RESULT_ROOT/results/16x16` |
| `32x32` | 3 | 1913323 | `$DATASET_ROOT/manifests/manifest_samefreq_6.7ghz_32x32_0deg.jsonl` | `$RESULT_ROOT/runs/32x32` | `$RESULT_ROOT/results/32x32` |

At `2026-09-01T09:33:31+08:00`, `nvidia-smi` mapped these exact live PIDs to GPUs 1, 2, and 3 respectively, each using about 1144 MiB.

## Live server commands

Run from the new code root after sourcing the prescribed environment. The ordering is intentional: the environment script changes the working directory, so source first and then `cd` to the new clone.

```bash
source /home/wys/radioflow_20260823/radioflow_remote_env.sh
cd /home/wys/radioflow_20260823/multiscale-uno-singlebeam

# Validate all three manifests and inferred beam/config identities.
RADIOFLOW_MULTISCALE_UNO_GPUS=1,2,3 bash scripts/run_same_frequency_multiscale_uno_server.sh --preflight

# One optimizer step per array, isolated below each run root's _smoke directory.
RADIOFLOW_MULTISCALE_UNO_GPUS=1,2,3 bash scripts/run_same_frequency_multiscale_uno_server.sh --smoke

# Formal, independently resumable training on physical GPUs 1/2/3.
RADIOFLOW_MULTISCALE_UNO_GPUS=1,2,3 bash scripts/run_same_frequency_multiscale_uno_server.sh --train

# Do not run until formal training has finished and a valid best checkpoint exists.
RADIOFLOW_MULTISCALE_UNO_GPUS=1,2,3 bash scripts/run_same_frequency_multiscale_uno_server.sh --select-cfg
RADIOFLOW_MULTISCALE_UNO_GPUS=1,2,3 bash scripts/run_same_frequency_multiscale_uno_server.sh --test
```

The train launcher uses `--resume auto`: it resumes from an array's `last.pt` when present, otherwise starts a new run. An explicit `--resume none` refuses to overwrite an existing formal `last.pt`. Smoke artifacts do not become formal-run checkpoints. The `--train` launcher takes all three `flock` locks before launch, records owned PID/birth metadata, and rejects a live owned run rather than creating a second writer.

## CUDA memory probe

The first probe attempt imported the old repository because the environment script changed the current directory and execution exited before model construction. The corrected order above was used for the successful probe on physical GPU 1 (RTX 3090): PyTorch `2.5.1+cu121`, CUDA `12.1`, float16 autocast, batch `B=2`, and 256 x 256.

| Field | Observed value |
| --- | --- |
| Command result | exit 0 |
| Loss | `0.0033314230386167765` |
| Loss and gradients | finite |
| Peak allocated | `449696256` bytes |
| Peak reserved | `528482304` bytes |
| Device total memory | `25438126080` bytes |
| OOM fallback | not used |

No lower-batch or accumulation fallback was needed.

## Preflight and scientific identities

Preflight succeeded for all arrays with split `560/80/160`, model `attention_multiscale_uno_lite`, and backbone `attention_conditioned_multiscale_uno2d`.

| Array | Inferred beam | Config ID | Manifest SHA-256 | Scientific config SHA-256 |
| --- | ---: | --- | --- | --- |
| `8x8` | 4 | `freq_6.7GHz_64TR_8beams_pattern_tr38901` | `30d7cd86fd77b10fbb1d249661be0e482fdb9a2f35f8a895e0c2eba693523383` | `f74fe23f75db2d093a0cf28c124cc714f3aa350e1648993fa4fa150114d9e6a8` |
| `16x16` | 8 | `freq_6.7GHz_256TR_16beams_pattern_tr38901` | `1aed953055e9dd8be7db46b45fcbcbd42143ebd98e398433a6b37f99e858c47c` | `f86253fdb9cd7d2a3d8e24c79725df47a9ff3cac82f6602e7352d7889ad64b01` |
| `32x32` | 32 | `freq_6.7GHz_1024TR_64beams_pattern_tr38901` | `8f8c4602b627a476ef1e91187563fe36bf6a16ec48afe1f1e189e10e1b1d84a6` | `715afb6416cb6a7cb89c7ccfb34c11f9675bdbef98badaa8bcc75278b69975ad` |

## Smoke evidence

All three smoke runs completed one optimizer step with 28 micro-batches and 56 samples. Each reported peak training allocation of `605852672` bytes. Each `smoke.pt` is `63163610` bytes and reload verified model, EMA, optimizer, scheduler, scaler, RNG, trainer state, and Git/config identity.

| Array | Smoke loss | `config.json` SHA-256 | `smoke.pt` SHA-256 |
| --- | ---: | --- | --- |
| `8x8` | `1.3182900391642092` | `34ea6a186de1248bf76b6893b0757351baac23450a58400ca3de589dff9a632c3` | `12462a9f159d6b22e98d0802f9070906a2999b63687226accb64717478365edff` |
| `16x16` | `1.3011826859981246` | `ae1f8004c1679b4037babd19f040260dd12f8f5b2cca136c4501bbc9e2765a074` | `c11fabc032fd0cc5996a0385740681b8a549485ab55c434b2deba4a50f2398b59` |
| `32x32` | `1.2656011730499659` | `dfbc89b494fb885a261bbc8b8214cafc5e199f1bed60d5a71542f5ee210b2f3a` | `c929ff6b2612868697de2a6848339b2eafc59eecee789cdd31ca94d6e7cb21c8d` |

## Formal training launch and early health

The formal launcher exited 0 at `2026-09-01T09:30:11+08:00`; PID metadata is present for each run. Its immutable formal `config.json` hashes are:

| Array | Formal config SHA-256 | First finite train loss | First finite validation RMSE |
| --- | --- | ---: | ---: |
| `8x8` | `0170ade169bd67ce5108d34a8a598256b407369914aac4f9e05ce26212f7bb895` | `1.3112550455201823` | `146.05302604196717` |
| `16x16` | `623312f2982ae280143050983cf51701dd571b4f22db1689c3d40a08a45a25224` | `1.294079223990407` | `143.39571302237355` |
| `32x32` | `1d3ae20e6dbca7098de3d692c519f4ae557c73fdd198341030e5a7c3b3e902f67` | `1.2589257919522308` | `138.24445381966008` |

At epoch 1, `last.pt` and `best.pt` were created for all arrays and training continued. These high early validation RMSE values are expected startup observations, not final results. The `2026-09-01T09:33+08:00` evidence snapshot found all owned PIDs alive: `8x8` and `16x16` had completed epoch 2, and `32x32` had completed epoch 3. These are launch-health observations only; training, CFG selection, and test evaluation are not complete.

## Regression evidence

| Scope | Result | Duration |
| --- | --- | ---: |
| Focused seven-file multiscale UNO suite | 140 passed | 71.60 s |
| Complete repository suite | 572 passed, 5 skipped, 5 failed, 2 warnings | 406.27 s |

The focused suite was fully green. The complete suite has exactly the five frozen baseline failures and no new failure names:

1. `tests/test_multiconfig_gpu_smoke.py::test_cuda_device_and_lite_complete_optimizer_step_smoke` — stale receipt contains a non-float `train_scale`.
2. `tests/test_multiconfig_gpu_smoke.py::test_large_smoke_is_complete_or_has_one_valid_global_oom_gate` — the same stale non-float `train_scale` receipt issue.
3. `tests/test_same_frequency_fno_server_script.py::test_server_launcher_dry_run_maps_three_physical_gpus` — an old WSL/Bash launcher test is invoked with a Windows path.
4. `tests/test_same_frequency_fno_server_script.py::test_server_launcher_requires_an_explicit_mode` — the same legacy Windows WSL/Bash path issue.
5. `tests/test_sparse_task2_metrics.py::test_task2_metrics_report_overall_missing_observed_and_groups` — old 4x4 fixture is smaller than the locked 11x11 SSIM window.

The frozen base recorded 432 passed, 5 skipped, and these same 5 failures in 299.38 s. The additional 140 passing tests are the completed multiscale UNO coverage. The two current warnings are `PytestUnhandledThreadExceptionWarning` instances from GBK decoding in the two already-failing legacy WSL launcher tests; no production, test, or README change was made to suppress them.
