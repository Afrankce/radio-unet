# Hybrid FNO-U RadioFlow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the parameter-matched Hybrid FNO-U Lite velocity backbone, verify that the original five-scale condition encoder and CA/SA fusion remain active, and launch the 8x8, 16x16, and 32x32 6.7 GHz zero-degree runs on three server GPUs.

**Architecture:** `HybridFNODiffUNet` keeps the original blue `BasicUNetEncoder`, five yellow `CrossAttention` modules, U-shaped resolution transitions, and skip connections. Every green `TwoConv` is replaced by one width-24 spectral-plus-local `FNOOperatorBlock`; modes and right/bottom padding follow the locked five-scale schedules. Dedicated config, trainer, evaluator, summary, and launcher entry points keep the existing U-Net and pure-FNO paths unchanged.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, MONAI, `torch.fft.rfft2/irfft2`, torchcfm, pytest, Bash, Git bundle, four 24 GB CUDA GPUs.

**Spec:** `docs/superpowers/specs/2026-08-27-hybrid-fno-u-radioflow-design.md`

## Global Constraints

- Preserve condition input `[Tx mask,height,beam map]`, target shape `[B,1,256,256]`, valid-mask loss, and 560/80/160 scene split.
- Preserve the blue feature shapes `[32@256,32@128,64@64,128@32,256@16]`.
- Reuse the enabled CA/SA `CrossAttention`; never instantiate or route through `CrossAttention_old`.
- Keep U-shaped max-pooling, upsampling, skip topology, 128-to-512-to-512 time embedding, condition-drop probability 0.25, and CFG=1.0 behavior.
- Lock operator width to 24, encoder modes to `(12,12,12,8,4)`, encoder padding to `(9,5,3,2,1)`, and mirror both schedules in the decoder.
- Lock parameter counts to 3,113,363 PyTorch tensor elements and 4,274,579 real scalar degrees of freedom.
- Keep learning rate `1e-3`, weight decay `1e-5`, warmup 10%, EMA 0.999, effective batch 56, maximum 1000 epochs, patience 20, fixed hash noise, and two-step Euler.
- FFT operations execute in float32 even under CUDA float16 autocast.
- Do not change or overwrite existing U-Net Lite, pure FNO, sparse Task 2, common8, or cross-frequency code/results.
- Use experiment identity `same_frequency_6.7_single_beam_hybrid_fno_u` and model identity `hybrid_fno_u_lite`.

---

### Task 1: Implement the reusable FNO operator block

**Files:**
- Create: `model/hybrid_fno_u.py`
- Create: `tests/test_hybrid_fno_u_block.py`

**Interfaces:**
- Consumes: `model.fno.SpectralConv2d`, `model.fno.count_tensor_parameters`, and `model.fno.count_real_scalar_parameters`.
- Produces: `FNOOperatorBlock(in_channels: int, out_channels: int, *, width: int, modes: int, padding: int)`.
- Produces: `FNOOperatorBlock.forward(value: Tensor, temb: Tensor) -> Tensor` with unchanged batch/spatial dimensions and `out_channels` channels.

- [ ] **Step 1: Write failing validation, shape, formula, and gradient tests**

```python
def test_operator_block_matches_locked_formula_and_preserves_shape():
    block = FNOOperatorBlock(4, 32, width=24, modes=12, padding=9)
    value = torch.randn(1, 4, 32, 32, requires_grad=True)
    temb = torch.randn(1, 512)
    output = block(value, temb)
    output.square().mean().backward()
    assert output.shape == (1, 32, 32, 32)
    assert torch.isfinite(output).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    expected_complex = 2 * 24 * 24 * 12 * 12
    expected_real = (
        (4 * 24 + 24) + 2 * expected_complex +
        (24 * 24 + 24) + (512 * 24 + 24) + (24 * 32 + 32)
    )
    assert count_real_scalar_parameters(block) == expected_real

@pytest.mark.parametrize("field,value", [("width", 0), ("modes", 0), ("padding", -1)])
def test_operator_block_rejects_invalid_controls(field, value):
    kwargs = {"width": 24, "modes": 12, "padding": 9, field: value}
    with pytest.raises(ValueError):
        FNOOperatorBlock(4, 32, **kwargs)
```

- [ ] **Step 2: Run RED**

Run:

```powershell
& D:\Anaconda3\envs\radioflow-win\python.exe -m pytest -q tests/test_hybrid_fno_u_block.py
```

Expected: collection fails because `model.hybrid_fno_u` does not exist.

- [ ] **Step 3: Implement the minimal operator block**

```python
class FNOOperatorBlock(nn.Module):
    def forward(self, value: Tensor, temb: Tensor) -> Tensor:
        hidden = self.lifting(value)
        if self.padding:
            hidden = F.pad(hidden, (0, self.padding, 0, self.padding))
        time_bias = self.time_projection(F.silu(temb))[:, :, None, None]
        hidden = F.gelu(self.spectral(hidden) + self.local(hidden) + time_bias)
        if self.padding:
            hidden = hidden[..., :-self.padding, :-self.padding]
        return self.projection(hidden)
```

`lifting`, `local`, and `projection` are bias-enabled 1x1 convolutions. `time_projection` is a bias-enabled `Linear(512,24)`. `SpectralConv2d` supplies the two complex Fourier corners and float32 FFT guard.

- [ ] **Step 4: Add mode-fit and CUDA autocast tests**

The mode-fit test passes an undersized grid and expects `ValueError("retained modes")`. The CUDA test runs the block under float16 autocast, backpropagates through a finite loss, and asserts finite complex gradients.

- [ ] **Step 5: Run GREEN and commit**

```powershell
& D:\Anaconda3\envs\radioflow-win\python.exe -m pytest -q tests/test_hybrid_fno_u_block.py tests/test_fno_model.py
git add model/hybrid_fno_u.py tests/test_hybrid_fno_u_block.py
git commit -m "feat: add multiscale FNO operator block"
```

---

### Task 2: Build the five-scale Hybrid FNO-U velocity model

**Files:**
- Modify: `model/hybrid_fno_u.py`
- Create: `tests/test_hybrid_fno_u_model.py`

**Interfaces:**
- Consumes: `BasicUNetEncoder`, enabled `CrossAttention`, MONAI `UpSample`, max pooling, and `get_timestep_embedding`.
- Produces: `HybridFNOUNetDe(...)` with the same `forward(x, t, embeddings=None, image=None)` interface as `BasicUNetDe`.
- Produces: `HybridFNODiffUNet(con_channels=3, cfg_drop_prob=0.25, ...)` with the same `forward` and `forward_with_cfg` contract as `DiffUNet`.
- Production defaults: features `(32,32,64,128,256,32)`, width 24, modes `(12,12,12,8,4)`, padding `(9,5,3,2,1)`.

- [ ] **Step 1: Write failing topology and data-flow tests**

```python
def test_hybrid_model_keeps_blue_encoder_and_enabled_attention():
    model = HybridFNODiffUNet()
    assert type(model.embed_model) is BasicUNetEncoder
    assert isinstance(model.model.cross_attn_0, CrossAttention)
    assert isinstance(model.model.cross_attn_4, CrossAttention)
    assert not any(isinstance(module, CrossAttention_old) for module in model.modules())
    assert tuple(model.model.operator_modes) == (12, 12, 12, 8, 4)
    assert tuple(model.model.operator_padding) == (9, 5, 3, 2, 1)

def test_tiny_hybrid_forward_calls_all_five_attention_scales():
    model = make_tiny_hybrid_model().eval()
    calls = []
    hooks = [
        getattr(model.model, f"cross_attn_{index}").register_forward_hook(
            lambda _module, inputs, _output, index=index:
                calls.append((index, tuple(inputs[0].shape), tuple(inputs[1].shape)))
        )
        for index in range(5)
    ]
    condition = torch.randn(1, 3, 32, 32)
    state = torch.randn(1, 1, 32, 32)
    output = model(image=condition, x=state, step=torch.tensor([0.5]))
    for hook in hooks:
        hook.remove()
    assert output.shape == state.shape
    assert [item[0] for item in calls] == [0, 1, 2, 3, 4]
    assert all(state_shape == embedding_shape for _, state_shape, embedding_shape in calls)
```

The tiny test uses features `(16,16,16,16,16,16)`, width 4, modes `(2,2,2,1,1)`, padding `(1,1,1,1,1)`, and 32x32 input so it remains fast while preserving all five levels.

- [ ] **Step 2: Run RED**

Run `python -m pytest -q tests/test_hybrid_fno_u_model.py`; expect missing classes.

- [ ] **Step 3: Implement encoder, fusion, decoder, and wrapper**

The encoder sequence is:

```python
x0 = cross_attn_0(block_0(torch.cat([image, x], 1), temb), embeddings[0])
x1 = cross_attn_1(down_1(x0, temb), embeddings[1])
x2 = cross_attn_2(down_2(x1, temb), embeddings[2])
x3 = cross_attn_3(down_3(x2, temb), embeddings[3])
x4 = cross_attn_4(down_4(x3, temb), embeddings[4])
```

The decoder upsamples, concatenates `x3/x2/x1/x0`, and applies matching FNO blocks at 32/64/128/256. `HybridFNODiffUNet.forward` computes blue embeddings, applies sample-level embedding-only dropout exactly as `DiffUNet`, and passes both raw `image` and embeddings to the decoder. `forward_with_cfg` uses zero embeddings for the unconditional branch while retaining the raw image, matching the locked baseline.

- [ ] **Step 4: Add CFG, dropout, time, and shape rejection tests**

```python
def test_hybrid_cfg_one_equals_conditional():
    model = make_tiny_hybrid_model().eval()
    c = torch.randn(1, 3, 32, 32)
    x = torch.randn(1, 1, 32, 32)
    t = torch.tensor([0.25])
    expected = model(image=c, x=x, step=t)
    actual = model.forward_with_cfg(
        image=c, x=x, step=t, embedding=model.embed_model(c), cfg_scale=1.0,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
```

Add assertions for five embedding lengths/shapes, condition/state batch and resolution matches, one time value per sample, and finite `cfg_scale`.

- [ ] **Step 5: Run GREEN and commit**

```powershell
& D:\Anaconda3\envs\radioflow-win\python.exe -m pytest -q tests/test_hybrid_fno_u_block.py tests/test_hybrid_fno_u_model.py tests/test_radioflow_framework_lock.py
git add model/hybrid_fno_u.py tests/test_hybrid_fno_u_model.py
git commit -m "feat: add attention-conditioned Hybrid FNO-U model"
```

---

### Task 3: Lock scientific configuration and factory identity

**Files:**
- Create: `training/same_frequency_hybrid_fno_config.py`
- Modify: `training/model_factory.py`
- Create: `tests/test_same_frequency_hybrid_fno_config.py`
- Create: `tests/test_same_frequency_hybrid_fno_factory.py`

**Interfaces:**
- Produces: `HYBRID_FNO_U_MODEL_SIZE = "hybrid_fno_u_lite"`.
- Produces: `HYBRID_FNO_U_BACKBONE = "hybrid_fno_u"`.
- Produces: `HybridFNOUTrainConfig(base: SameFrequencyTrainConfig, memory_profile: Literal["standard","low_memory"]="standard")` with `scientific_payload`, `config_sha256`, `to_record`, `from_json`, and `with_run_root`.
- Produces: `build_hybrid_fno_u() -> HybridFNODiffUNet`.
- Extends: `build_same_frequency_backbone(model_size: str)` without changing `lite`, `large`, or `paper_fno_lite`.

- [ ] **Step 1: Write failing config and parameter-lock tests**

```python
def test_hybrid_config_locks_architecture_and_protocol(tmp_path):
    cfg = make_hybrid_config(tmp_path, array_size="16x16", beam_id=8)
    payload = cfg.scientific_payload()
    assert cfg.model_size == "hybrid_fno_u_lite"
    assert payload["experiment"] == "same_frequency_6.7_single_beam_hybrid_fno_u"
    assert payload["operator_width"] == 24
    assert payload["operator_modes"] == [12, 12, 12, 8, 4]
    assert payload["operator_padding"] == [9, 5, 3, 2, 1]
    assert payload["cfg_candidates"] == [1.0]
    assert cfg.memory_profile == "standard"
    assert cfg.micro_batch_size == 2
    assert cfg.accumulation_steps == 28
    assert cfg.effective_batch_size == 56

def test_low_memory_profile_preserves_effective_batch_and_optimizer_steps(tmp_path):
    standard = make_hybrid_config(tmp_path, memory_profile="standard")
    low_memory = make_hybrid_config(tmp_path, memory_profile="low_memory")
    assert (low_memory.micro_batch_size, low_memory.accumulation_steps) == (1, 56)
    assert low_memory.effective_batch_size == standard.effective_batch_size == 56
    assert low_memory.optimizer_steps_per_epoch == standard.optimizer_steps_per_epoch
    assert low_memory.config_sha256 != standard.config_sha256

def test_hybrid_factory_locks_real_parameter_budget():
    model = build_same_frequency_backbone("hybrid_fno_u_lite")
    assert isinstance(model, HybridFNODiffUNet)
    assert count_tensor_parameters(model) == 3_113_363
    assert count_real_scalar_parameters(model) == 4_274_579
```

- [ ] **Step 2: Run RED**

Run both new test files; expect missing config and factory entry points.

- [ ] **Step 3: Implement immutable wrapper config and factory lock**

`HybridFNOUTrainConfig` delegates all data/training fields to a `SameFrequencyTrainConfig(model_size="lite")`, exposes the hybrid model identity, records every architecture constant and both parameter counts, and rejects any altered field during construction or JSON round-trip. The standard profile returns micro-batch 2/accumulation 28; the low-memory profile returns 1/56. Both retain effective batch 56, ten optimizer steps per epoch, and identical warmup length. `build_hybrid_fno_u` verifies exact class identities for blue encoder and five enabled attention modules before checking counts.

- [ ] **Step 4: Test checkpoint identity separation and old factory regressions**

Prove that hybrid config hashes differ from U-Net and pure FNO hashes; prove a hybrid checkpoint identity cannot be loaded into either old model. Re-run existing U-Net/FNO factory tests unchanged.

- [ ] **Step 5: Run GREEN and commit**

```powershell
& D:\Anaconda3\envs\radioflow-win\python.exe -m pytest -q tests/test_same_frequency_hybrid_fno_config.py tests/test_same_frequency_hybrid_fno_factory.py tests/test_same_frequency_fno_config.py tests/test_same_frequency_fno_factory.py tests/test_radioflow_framework_lock.py tests/test_checkpoint_resume.py
git add training/same_frequency_hybrid_fno_config.py training/model_factory.py tests/test_same_frequency_hybrid_fno_config.py tests/test_same_frequency_hybrid_fno_factory.py
git commit -m "feat: lock Hybrid FNO-U experiment identity"
```

---

### Task 4: Add dedicated training and evaluation entry points

**Files:**
- Create: `training/same_frequency_hybrid_fno_trainer.py`
- Create: `train_same_frequency_hybrid_fno.py`
- Create: `evaluate_same_frequency_hybrid_fno.py`
- Create: `tests/test_same_frequency_hybrid_fno_trainer.py`
- Create: `tests/test_train_same_frequency_hybrid_fno_cli.py`
- Create: `tests/test_evaluate_same_frequency_hybrid_fno_cli.py`
- Create: `tests/test_same_frequency_hybrid_fno_evaluator.py`

**Interfaces:**
- Produces: `write_or_validate_hybrid_fno_run_config(cfg, controls) -> Path`.
- Produces: `run_same_frequency_hybrid_fno_training(cfg, controls, device, preflight_only=False) -> dict[str,Any]`.
- Training CLI accepts data/manifest/height/run/array/device/resume, `--memory-profile {standard,low_memory}`, plus mutually exclusive preflight, smoke, and stop controls; architecture controls are not exposed.
- Evaluation CLI exposes only `select-cfg` and `test`; both route through the existing strict same-frequency evaluator with candidates `(1.0,)`.

- [ ] **Step 1: Write failing parser and one-step trainer tests**

```python
def test_hybrid_cli_exposes_operational_controls_only():
    args = build_parser().parse_args([
        "--dataset-root", "dataset", "--manifest-path", "manifest.jsonl",
        "--height-stats-path", "height.json", "--run-root", "run",
        "--array-size", "32x32", "--device", "cuda:0",
        "--resume", "auto", "--memory-profile", "standard", "--preflight-only",
    ])
    assert args.array_size == "32x32"
    assert args.preflight_only is True
    assert args.memory_profile == "standard"
    assert not hasattr(args, "operator_width")
    assert not hasattr(args, "modes")
```

The trainer test monkeypatches the factory to a tiny hybrid model and uses the existing tiny same-frequency dataset fixture. One optimizer-step smoke must save model, EMA, optimizer, scheduler, complex-aware scaler, CPU train generator, trainer state, and hybrid identity, then reload them into a fresh instance.

- [ ] **Step 2: Run RED**

Run the four new test files; expect missing modules.

- [ ] **Step 3: Implement trainer and CLIs by reusing existing primitives**

`run_same_frequency_hybrid_fno_training` reuses `preflight_same_frequency`, `build_same_frequency_loaders`, `MultiConfigSRMTrainer`, `ComplexGradScaler`, strict checkpointing, EMA, scheduler, valid-mask FM loss, and resume logic. It writes immutable `config.json`, refuses `resume=none` over an existing `last.pt`, redirects smoke runs to `_smoke`, and fresh-reloads the smoke checkpoint.

The evaluation CLI constructs `HybridFNOUTrainConfig` after `infer_manifest_selection` and calls the existing `run_cfg_selection` or `run_test_evaluation`; no evaluator math is copied.

- [ ] **Step 4: Run trainer/evaluator regressions**

```powershell
& D:\Anaconda3\envs\radioflow-win\python.exe -m pytest -q tests/test_same_frequency_hybrid_fno_trainer.py tests/test_train_same_frequency_hybrid_fno_cli.py tests/test_evaluate_same_frequency_hybrid_fno_cli.py tests/test_same_frequency_hybrid_fno_evaluator.py tests/test_same_frequency_fno_trainer.py tests/test_same_frequency_fno_evaluator.py tests/test_same_frequency_evaluator.py tests/test_masked_flow_loss.py tests/test_gradient_accumulation.py
```

- [ ] **Step 5: Commit**

```powershell
git add training/same_frequency_hybrid_fno_trainer.py train_same_frequency_hybrid_fno.py evaluate_same_frequency_hybrid_fno.py tests/test_same_frequency_hybrid_fno_trainer.py tests/test_train_same_frequency_hybrid_fno_cli.py tests/test_evaluate_same_frequency_hybrid_fno_cli.py tests/test_same_frequency_hybrid_fno_evaluator.py
git commit -m "feat: train and evaluate Hybrid FNO-U"
```

---

### Task 5: Add strict three-array result summarization

**Files:**
- Create: `summarize_same_frequency_hybrid_fno.py`
- Create: `tests/test_summarize_same_frequency_hybrid_fno.py`

**Interfaces:**
- Produces: `collect_hybrid_results(results_root: str | Path) -> dict[str,dict[str,float]]`.
- Produces: `write_comparison(output_dir, hybrid_metrics) -> dict[str,Path]`.
- Reads exactly three completed hybrid `metrics_test.json` and `run_manifest.json` pairs.
- Writes `hybrid_fno_u_metrics.json` and `hybrid_fno_u_comparison.csv` with U-Net Lite, pure FNO, and Hybrid FNO-U rows kept under the same single-beam protocol.
- Uses the frozen dB-RMSE references `U-Net={8x8:11.627,16x16:11.657,32x32:11.700}` and `paper FNO={8x8:14.759,16x16:14.424,32x32:14.093}`; these values are labeled as prior results rather than newly evaluated checkpoints.

- [ ] **Step 1: Write failing strict-collection tests**

```python
def test_collect_requires_exact_three_array_result_set(tmp_path):
    write_hybrid_result(tmp_path, "8x8", rmse=11.0)
    with pytest.raises(HybridFNOSummaryError, match="exactly"):
        collect_hybrid_results(tmp_path)

def test_comparison_keeps_protocols_and_metrics_separate(tmp_path):
    for array_size, rmse in {"8x8": 11.0, "16x16": 11.1, "32x32": 11.2}.items():
        write_hybrid_result(tmp_path, array_size, rmse=rmse)
    metrics = collect_hybrid_results(tmp_path)
    paths = write_comparison(tmp_path / "summary", metrics)
    rows = list(csv.DictReader(paths["csv"].open(encoding="utf-8")))
    assert {row["model"] for row in rows} == {"unet_lite", "paper_fno_lite", "hybrid_fno_u_lite"}
    assert {row["array_size"] for row in rows} == {"8x8", "16x16", "32x32"}
```

- [ ] **Step 2: Run RED**

Run the new summary tests; expect module import failure.

- [ ] **Step 3: Implement atomic strict summary publication**

Validate experiment/model/split/sample-count/status fields, reject extra or missing metrics files, parse finite RMSE/MAE/NMSE/PSNR/SSIM and best epoch, then publish JSON and CSV atomically. Baseline constants are labeled as previously verified results; no statistical superiority claim is generated automatically.

- [ ] **Step 4: Run GREEN and commit**

```powershell
& D:\Anaconda3\envs\radioflow-win\python.exe -m pytest -q tests/test_summarize_same_frequency_hybrid_fno.py tests/test_summarize_same_frequency_fno.py
git add summarize_same_frequency_hybrid_fno.py tests/test_summarize_same_frequency_hybrid_fno.py
git commit -m "feat: summarize Hybrid FNO-U benchmark"
```

---

### Task 6: Add safe three-GPU server orchestration

**Files:**
- Create: `scripts/run_same_frequency_hybrid_fno_server.sh`
- Create: `tests/test_same_frequency_hybrid_fno_server_script.py`

**Interfaces:**
- Requires exactly one mode: `--dry-run`, `--preflight`, `--smoke`, `--train`, `--status`, `--select-cfg`, or `--test`.
- Uses physical GPU 1 for 8x8, GPU 2 for 16x16, GPU 3 for 32x32, leaving GPU 0 free.
- Defaults to `/home/wys/radioflow_20260823/hybrid-fno-u-singlebeam`, the existing dataset root, and a new result root `/home/wys/radioflow_20260823/results/hybrid_fno_u_samefreq_6.7ghz`.
- Never starts work without an explicit mode and never overwrites a live PID.

- [ ] **Step 1: Write failing dry-run, explicit-mode, and PID-safety tests**

```python
def test_hybrid_launcher_dry_run_maps_three_gpus(tmp_path):
    completed = run_launcher(tmp_path, "--dry-run")
    lines = [line for line in completed.stdout.splitlines() if line.startswith("TRAIN ")]
    assert len(lines) == 3
    for line, (array_size, gpu) in zip(lines, {"8x8":"1", "16x16":"2", "32x32":"3"}.items()):
        assert f"ARRAY={array_size} " in line
        assert f"CUDA_VISIBLE_DEVICES={gpu} " in line
        assert "--device cuda:0" in line
        assert "--resume auto" in line
```

- [ ] **Step 2: Run RED**

Run the server-script test; expect missing script.

- [ ] **Step 3: Implement explicit orchestration modes**

Adapt the proven pure-FNO launcher but use hybrid CLI names and environment prefix `RADIOFLOW_HYBRID_FNO_*`. `--preflight` runs serially; `--smoke` first runs three standard-profile one-step jobs and waits for all. Only a failed log containing CUDA OOM is retried with `--memory-profile low_memory`; the selected profile is atomically recorded per array and reused by `--train`. Any non-OOM smoke failure remains a hard failure. `--train` launches three `nohup` jobs with distinct logs/PIDs; `--status` validates PID liveness and tails concise loss/epoch lines; evaluation modes run one array per GPU and wait for all.

- [ ] **Step 4: Run shell syntax and test suite**

```powershell
bash -n scripts/run_same_frequency_hybrid_fno_server.sh
& D:\Anaconda3\envs\radioflow-win\python.exe -m pytest -q tests/test_same_frequency_hybrid_fno_server_script.py tests/test_same_frequency_fno_server_script.py
```

- [ ] **Step 5: Commit**

```powershell
git add scripts/run_same_frequency_hybrid_fno_server.sh tests/test_same_frequency_hybrid_fno_server_script.py
git commit -m "feat: orchestrate Hybrid FNO-U server runs"
```

---

### Task 7: Verify, deploy, smoke, and launch the formal experiment

**Files:**
- Verify only: all tracked source and tests
- Generate outside Git: Git bundle, server logs, PID files, checkpoints, metrics, and visualizations

**Interfaces:**
- Local source branch: `codex/hybrid-fno-u-singlebeam`.
- Remote checkout: `/home/wys/radioflow_20260823/hybrid-fno-u-singlebeam`.
- Remote environment bootstrap: `/home/wys/radioflow_20260823/radioflow_remote_env.sh`.
- Remote result root: `/home/wys/radioflow_20260823/results/hybrid_fno_u_samefreq_6.7ghz`.

- [ ] **Step 1: Run the complete local regression suite**

```powershell
& D:\Anaconda3\envs\radioflow-win\python.exe -m pytest -q
git status --short
git log --oneline -8
```

Expected: zero failures and no uncommitted source changes.

- [ ] **Step 2: Create and verify a Git bundle**

```powershell
git bundle create E:\RadioFlow-transfer\hybrid-fno-u-singlebeam.bundle codex/hybrid-fno-u-singlebeam
git bundle verify E:\RadioFlow-transfer\hybrid-fno-u-singlebeam.bundle
```

- [ ] **Step 3: Transfer and create the isolated remote checkout**

Copy the verified bundle through the configured SSH host, then on the server fetch `codex/hybrid-fno-u-singlebeam` into `/home/wys/radioflow_20260823/hybrid-fno-u-singlebeam`. Refuse to replace a non-clean existing checkout; if it already exists cleanly, fast-forward it from the bundle.

- [ ] **Step 4: Run remote baseline and GPU preflight**

```bash
source /home/wys/radioflow_20260823/radioflow_remote_env.sh
cd /home/wys/radioflow_20260823/hybrid-fno-u-singlebeam
pytest -q
bash scripts/run_same_frequency_hybrid_fno_server.sh --dry-run
bash scripts/run_same_frequency_hybrid_fno_server.sh --preflight
```

Expected: all tests pass; all three manifests resolve 560/80/160 scenes and a 0° beam; parameter and config hashes print without mismatch.

- [ ] **Step 5: Run and validate three GPU smoke jobs**

```bash
bash scripts/run_same_frequency_hybrid_fno_server.sh --smoke
```

Expected: three finite optimizer-step losses, EMA updates, strict fresh checkpoint reloads, and no OOM. If batch 2 OOMs, change only the documented operational fallback to micro-batch 1/accumulation 56, record it in config identity, repeat all smoke jobs, and retain effective batch 56.

- [ ] **Step 6: Launch all formal runs and verify stable progress**

```bash
bash scripts/run_same_frequency_hybrid_fno_server.sh --train
bash scripts/run_same_frequency_hybrid_fno_server.sh --status
```

Expected: three live PIDs mapped to GPUs 1/2/3, distinct run/log directories, finite training loss, and `last.pt` creation. Recheck after at least one completed epoch; if a process exits, inspect its own log, correct only the demonstrated cause, re-run the affected smoke, and resume with `--resume auto`.

- [ ] **Step 7: Keep training under bounded monitoring and evaluate automatically on completion**

Use the launcher status mode to inspect actual logs/checkpoints without loading long logs wholesale. Once each run early-stops or reaches epoch 1000, run `--select-cfg`, then `--test`, then `summarize_same_frequency_hybrid_fno.py`. Never report final metrics before all three `metrics_test.json` and `run_manifest.json` files validate as complete.

---

## Final Verification Checklist

- [ ] Hybrid model contains the original blue encoder and five enabled CA/SA fusion modules.
- [ ] No Q/K/V attention path is instantiated.
- [ ] All nine green `TwoConv` blocks are replaced by one FNO operator block each.
- [ ] Locked tensor and real-scalar parameter counts match exactly.
- [ ] Existing U-Net and pure-FNO tests still pass.
- [ ] Three remote smoke checkpoints reload strictly.
- [ ] Three formal training PIDs are live with finite progress and non-overlapping directories.
- [ ] Final evaluation, when reached, uses EMA best, CFG=1.0, fixed hash noise, and two-step Euler.
