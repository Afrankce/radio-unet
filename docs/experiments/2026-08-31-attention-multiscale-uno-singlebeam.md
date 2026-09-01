# Attention-conditioned multiscale UNO: execution record

## Status and immutable identity

- **Implementation snapshot verified:** `b45fe10cee5a32ad098afeca333fb5014758fa38` (`fix: make UNO rollback cleanup signal-safe`).
- **Experiment:** `same_frequency_6.7_single_beam_attention_multiscale_uno`.
- **Model size / backbone:** `attention_multiscale_uno_lite` / `attention_conditioned_multiscale_uno2d`.
- **Protocol:** independent `8x8`, `16x16`, and `32x32` arrays; 6.7 GHz; 0 degrees; scene-disjoint seed 42 split of 560 train / 80 validation / 160 test scenes; 256 x 256; condition channels `[Tx mask, height, beam map]`; valid mask is metric/loss-only.
- **Locked architecture:** state channels `(32, 64, 128, 256, 256)`; operator width `24`; modes `(12, 12, 8, 4, 4)`; right/bottom padding `(9, 5, 3, 2, 1)`; `BasicUNetEncoder_lite` features `(32, 32, 64, 128, 256, 32)`; nine native-scale CA/SA injections; CFG dropout `0.25`; evaluation CFG candidate `1.0`.

Live local construction with `D:\Anaconda3\envs\radioflow-win\python.exe` verified that `build_attention_multiscale_uno()` and `build_same_frequency_backbone("attention_multiscale_uno_lite")` both return `AttentionMultiscaleUNO2d`, with:

| Quantity | Locked and observed value |
| --- | ---: |
| Tensor parameter elements (complex element counted once) | 3,059,355 |
| Independent real scalar parameters | 3,925,659 |
| `CrossAttention` modules | 9 |
| Condition channels | 3 |

The local representative `8x8` configuration (synthetic paths and `beam_id=0`, used only to exercise construction and round-trip) had config SHA-256 `75bb49db1c65af94f2da87dc2c3b2515dffcb60fe700163fb8129f32deb0e13a`.  Its JSON round-trip preserved the hash. This is **not** a server-run identity: the real per-array hash is determined only after each server manifest supplies its validated beam ID.

## Hash and identity procedure

The source/model identity before a checkpoint exists is the verified Git commit plus live factory construction and the two locked parameter counts above. There is intentionally no claimed weight-file SHA-256 before the server creates a checkpoint.

For each array, the lifecycle CLI calls `infer_manifest_selection()` on the corresponding manifest before constructing `MultiscaleUNOTrainConfig`. The resulting `config_sha256` is SHA-256 of the canonical JSON bytes of `scientific_payload()`; it excludes machine-specific paths but includes the array dimensions, inferred beam ID, frozen protocol, and multiscale architecture fields. Capture it from that run's immutable `config.json` after preflight/smoke.

The runtime checkpoint identity additionally binds `array_size`, `model_size`, tensor parameter count, config SHA-256, manifest/split/schema/archive SHA-256 values, dataset revision, upstream base, Git commit, and seed. After smoke and training, record artifact digests with:

```bash
sha256sum "$RESULT_ROOT/runs/8x8/_smoke/config.json" "$RESULT_ROOT/runs/8x8/_smoke/last.pt"
sha256sum "$RESULT_ROOT/runs/16x16/_smoke/config.json" "$RESULT_ROOT/runs/16x16/_smoke/last.pt"
sha256sum "$RESULT_ROOT/runs/32x32/_smoke/config.json" "$RESULT_ROOT/runs/32x32/_smoke/last.pt"
```

## Server locations and GPU allocation

New roots only; existing Attention-FNO worktrees, result directories, and checkpoints remain read-only:

```text
CODE_ROOT=/home/wys/radioflow_20260823/multiscale-uno-singlebeam
DATASET_ROOT=/home/wys/radioflow_20260823/datasets/MultiConfigRadiomap
RESULT_ROOT=/home/wys/radioflow_20260823/results/multiscale_uno_samefreq_6.7ghz
ENV_FILE=/home/wys/radioflow_20260823/radioflow_remote_env.sh
```

| Array | Physical GPU | Manifest | Run root | Evaluation result root |
| --- | ---: | --- | --- | --- |
| `8x8` | 0 | `$DATASET_ROOT/manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl` | `$RESULT_ROOT/runs/8x8` | `$RESULT_ROOT/results/8x8` |
| `16x16` | 1 | `$DATASET_ROOT/manifests/manifest_samefreq_6.7ghz_16x16_0deg.jsonl` | `$RESULT_ROOT/runs/16x16` | `$RESULT_ROOT/results/16x16` |
| `32x32` | 2 | `$DATASET_ROOT/manifests/manifest_samefreq_6.7ghz_32x32_0deg.jsonl` | `$RESULT_ROOT/runs/32x32` | `$RESULT_ROOT/results/32x32` |

Each launcher child receives its mapped `CUDA_VISIBLE_DEVICES` value and runs the CLI as `--device cuda:0`. GPU 3 is not assigned to this experiment.

## Execution commands

Run from `$CODE_ROOT` after sourcing the prescribed server environment:

```bash
source /home/wys/radioflow_20260823/radioflow_remote_env.sh
cd /home/wys/radioflow_20260823/multiscale-uno-singlebeam

# Validate all three manifests and their inferred beam/config identities.
bash scripts/run_same_frequency_multiscale_uno_server.sh --preflight

# One optimizer step per array, written only under each run root's _smoke directory.
bash scripts/run_same_frequency_multiscale_uno_server.sh --smoke

# Formal, independently resumable training runs on GPUs 0/1/2.
bash scripts/run_same_frequency_multiscale_uno_server.sh --train

# After a valid best checkpoint exists, select the locked CFG=1.0 then test once.
bash scripts/run_same_frequency_multiscale_uno_server.sh --select-cfg
bash scripts/run_same_frequency_multiscale_uno_server.sh --test
```

The train launcher uses `--resume auto`: it resumes from the array's `last.pt` when present, otherwise starts a new run. An explicit `--resume none` refuses to overwrite an existing formal `last.pt`. Smoke commands isolate artifacts below `runs/<array>/_smoke`; they do not become formal-run checkpoints. The `--train` launcher takes all three `flock` locks before launch, records owned PID/birth metadata, and rejects a live owned run rather than launching a second writer.

The local regression commands actually executed for this record were:

```powershell
& 'D:\Anaconda3\envs\radioflow-win\python.exe' -m pytest tests/test_attention_multiscale_uno_stage.py tests/test_attention_multiscale_uno_model.py tests/test_attention_multiscale_uno_factory.py tests/test_same_frequency_multiscale_uno_config.py tests/test_same_frequency_multiscale_uno_trainer.py tests/test_same_frequency_multiscale_uno_cli.py tests/test_same_frequency_multiscale_uno_server_script.py -q
& 'D:\Anaconda3\envs\radioflow-win\python.exe' -m pytest -q
```

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

## Deferred 24 GB CUDA memory evidence — Task 8 required

**Status: PENDING.** No server transfer, CUDA memory probe, smoke run, formal training, CFG selection, or test evaluation has been claimed or performed in Task 7.

After Task 8 transfers the verified tracked source snapshot, run the following on one idle 24 GB server GPU before formal training. It constructs the locked factory model, runs one float16-autocast `B=2`, 256 x 256 forward/backward, synchronizes, and emits `torch.cuda.max_memory_allocated()`:

```bash
source /home/wys/radioflow_20260823/radioflow_remote_env.sh
cd /home/wys/radioflow_20260823/multiscale-uno-singlebeam
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import json
import torch
from training.model_factory import build_attention_multiscale_uno

assert torch.cuda.is_available()
torch.manual_seed(42)
device = torch.device("cuda:0")
model = build_attention_multiscale_uno().to(device).train()
condition = torch.randn(2, 3, 256, 256, device=device)
state = torch.randn(2, 1, 256, 256, device=device)
steps = torch.rand(2, device=device)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)
with torch.autocast(device_type="cuda", dtype=torch.float16):
    velocity = model(image=condition, x=state, step=steps)
    loss = velocity.square().mean()
loss.backward()
torch.cuda.synchronize(device)
assert torch.isfinite(loss)
assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
print(json.dumps({
    "batch_size": 2,
    "resolution": [256, 256],
    "loss": float(loss.detach().cpu()),
    "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
    "device": torch.cuda.get_device_name(device),
}, sort_keys=True))
PY
```

If and only if this OOMs, repeat with micro-batch 1 and 56 accumulation iterations (divide each loss by 56 before `backward()`); retain every architecture constant and record the fallback peak. Task 8 must amend this section with the exact GPU name, CUDA/PyTorch versions, command exit code, batch/accumulation setting, peak allocated bytes, finite-loss/gradient result, smoke evidence, per-array config/checkpoint hashes, and timestamp before formal training.

| Server evidence field | Task 7 value |
| --- | --- |
| 24 GB GPU name / CUDA / PyTorch | PENDING — Task 8 |
| B=2 peak allocated bytes | PENDING — Task 8 |
| OOM fallback used | PENDING — Task 8 |
| Preflight manifest / inferred beam IDs / hashes | PENDING — Task 8 |
| Smoke checkpoint and finite-gradient evidence | PENDING — Task 8 |
| Formal train PIDs / first metrics / `last.pt` | PENDING — Task 8 |
| CFG selection and test artifacts | PENDING — Task 8 |
