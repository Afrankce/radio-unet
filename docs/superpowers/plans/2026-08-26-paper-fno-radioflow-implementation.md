# Paper-Faithful FNO RadioFlow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paper-faithful FNO2d velocity backbone to the fixed 6.7 GHz zero-degree same-frequency RadioFlow experiment, then run the 8x8, 16x16, and 32x32 models independently on three server GPUs.

**Architecture:** A dedicated `ConditionalFNO2d` implements pointwise lifting, four dense spectral-plus-local Fourier blocks, and pointwise projection. A separate FNO config/trainer/CLI preserves the existing U-Net entry points and checkpoint identities. The existing same-frequency evaluator is generalized only at its model-factory and CFG-candidate boundaries so U-Net behavior remains byte-for-byte compatible while FNO uses the preregistered fixed candidate `(1.0,)`.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, `torch.fft.rfft2/irfft2`, torchcfm, pytest, Bash, Git bundle, RTX 3090.

**Spec:** `docs/superpowers/specs/2026-08-26-paper-fno-radioflow-design.md`

## Global Constraints

- Do not modify the frozen preregistration `docs/science-superpowers/preregistrations/2026-08-26-paper-fno-radioflow.md`.
- Preserve all existing U-Net Lite/Large public behavior and checkpoint hashes; default same-frequency code paths remain U-Net.
- FNO architecture is locked to four layers, width 40, modes 12x12, padding 9, projection 40->128->1, no normalization, and condition dropout 0.25.
- The FNO input order is exactly `[x_t, Tx mask, height, beam map, t_map, grid_x, grid_y]`.
- The FFT branch executes in float32 even when its caller is under CUDA autocast.
- PyTorch complex tensor elements and real scalar trainable degrees of freedom are both recorded.
- Dataset, split, normalization, seed 42, optimizer, EMA, effective batch 56, max 1000 epochs, patience 20, fixed hash noise, two-step Euler, and CFG 1.0 stay locked.
- New output directories must never overwrite existing U-Net or sparse experiments.

---

### Task 1: Implement the paper-faithful conditional FNO model

**Files:**
- Create: `model/fno.py`
- Create: `tests/test_fno_model.py`

**Interfaces:**
- Produces: `SpectralConv2d(in_channels, out_channels, modes1, modes2)`.
- Produces: `ConditionalFNO2d(condition_channels=3, width=40, modes1=12, modes2=12, padding=9, cfg_drop_prob=0.25)`.
- Produces: `count_tensor_parameters(module) -> int` and `count_real_scalar_parameters(module) -> int`.
- `ConditionalFNO2d.forward(...)` and `forward_with_cfg(...)` conform to `evaluation.radioflow_sampling.RadioFlowCFGModel`.

- [ ] **Step 1: Write the failing spectral and parameter-accounting tests**

```python
def test_spectral_conv_matches_hand_built_two_corner_fft():
    layer = SpectralConv2d(1, 1, modes1=2, modes2=2)
    with torch.no_grad():
        layer.weights1.fill_(1 + 0j)
        layer.weights2.fill_(2 + 0j)
    x = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) / 64
    x_ft = torch.fft.rfft2(x)
    expected_ft = torch.zeros(1, 1, 8, 5, dtype=torch.cfloat)
    expected_ft[:, :, :2, :2] = x_ft[:, :, :2, :2]
    expected_ft[:, :, -2:, :2] = 2 * x_ft[:, :, -2:, :2]
    expected = torch.fft.irfft2(expected_ft, s=(8, 8))
    assert torch.allclose(layer(x), expected, atol=1e-6, rtol=1e-6)

def test_locked_fno_parameter_counts_use_real_complex_dofs():
    model = ConditionalFNO2d()
    assert count_tensor_parameters(model) == 1_855_457
    assert count_real_scalar_parameters(model) == 3_698_657
```

- [ ] **Step 2: Run the model tests remotely and verify RED**

Run:

```bash
source /home/wys/radioflow_20260823/radioflow_remote_env.sh
cd /home/wys/radioflow_20260823/fno-paper-singlebeam
pytest -q tests/test_fno_model.py
```

Expected: collection fails because `model.fno` does not exist.

- [ ] **Step 3: Implement the minimal spectral layer and model**

```python
class SpectralConv2d(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        output_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            value = x.float()
            x_ft = torch.fft.rfft2(value)
            out_ft = torch.zeros(
                value.shape[0], self.out_channels, value.shape[-2],
                value.shape[-1] // 2 + 1, dtype=torch.cfloat, device=value.device,
            )
            out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, :self.modes1, :self.modes2], self.weights1,
            )
            out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, -self.modes1:, :self.modes2], self.weights2,
            )
            result = torch.fft.irfft2(out_ft, s=value.shape[-2:])
        return result.to(output_dtype)
```

`ConditionalFNO2d` must append `t_map` and `[0,1]` coordinate grids, lift 7 channels to width 40, pad right/bottom by 9, execute four spectral-plus-1x1 blocks with GELU after blocks 0-2, crop, and project to one channel. `forward()` applies sample-level condition dropout only in training. `forward_with_cfg()` computes zero-condition and full-condition branches and returns the standard CFG combination.

- [ ] **Step 4: Add shape, dropout, CFG, gradient, and CUDA AMP tests**

```python
def test_fno_forward_and_cfg_shapes_and_gradients():
    model = ConditionalFNO2d(width=4, modes1=2, modes2=2, padding=1)
    condition = torch.randn(2, 3, 16, 16)
    state = torch.randn(2, 1, 16, 16, requires_grad=True)
    time = torch.tensor([0.0, 0.75])
    output = model(image=condition, x=state, step=time)
    assert output.shape == state.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()

def test_cfg_one_equals_conditional_velocity():
    model = ConditionalFNO2d(width=4, modes1=2, modes2=2, padding=1).eval()
    condition = torch.randn(1, 3, 16, 16)
    state = torch.randn(1, 1, 16, 16)
    time = torch.tensor([0.5])
    conditional = model(image=condition, x=state, step=time)
    guided = model.forward_with_cfg(
        image=condition, x=state, step=time,
        embedding=model.embed_model(condition), cfg_scale=1.0,
    )
    assert torch.allclose(guided, conditional, atol=1e-6, rtol=1e-5)
```

The CUDA test uses the locked 256x256 model under float16 autocast and asserts a finite 256x256 output, proving the 265x265 FFT actually ran in float32.

- [ ] **Step 5: Run tests and commit GREEN**

```bash
pytest -q tests/test_fno_model.py
git add model/fno.py tests/test_fno_model.py
git commit -m "feat: add paper-faithful conditional FNO2d"
```

Expected: all model tests pass.

---

### Task 2: Lock the FNO factory, scientific config, and checkpoint identity

**Files:**
- Modify: `training/model_factory.py`
- Create: `training/same_frequency_fno_config.py`
- Create: `tests/test_same_frequency_fno_config.py`
- Create: `tests/test_same_frequency_fno_factory.py`

**Interfaces:**
- Produces: `PAPER_FNO_MODEL_SIZE = "paper_fno_lite"`.
- Produces: `build_paper_fno() -> ConditionalFNO2d`.
- Produces: `build_same_frequency_backbone(model_size: str) -> nn.Module`; delegates `lite`/`large` to the unchanged locked builder.
- Produces: `PaperFNOTrainConfig(base: SameFrequencyTrainConfig)` with `model_size`, `scientific_payload`, `config_sha256`, `to_record`, `from_json`, and `with_run_root`.

- [ ] **Step 1: Write failing config/factory behavior tests**

```python
def test_fno_config_keeps_training_protocol_and_adds_architecture_identity(tmp_path):
    cfg = make_fno_config(tmp_path, array_size="16x16", beam_id=8)
    payload = cfg.scientific_payload()
    assert cfg.model_size == "paper_fno_lite"
    assert cfg.micro_batch_size == 2
    assert cfg.accumulation_steps == 28
    assert cfg.effective_batch_size == 56
    assert payload["backbone"] == "paper_fno2d"
    assert payload["fno_width"] == 40
    assert payload["fno_modes"] == [12, 12]
    assert payload["cfg_candidates"] == [1.0]

def test_same_frequency_factory_preserves_unet_and_builds_locked_fno():
    assert type(build_same_frequency_backbone("lite")) is DiffUNet
    fno = build_same_frequency_backbone("paper_fno_lite")
    assert isinstance(fno, ConditionalFNO2d)
    assert count_real_scalar_parameters(fno) == 3_698_657
```

- [ ] **Step 2: Run and verify RED**

Run `pytest -q tests/test_same_frequency_fno_config.py tests/test_same_frequency_fno_factory.py`.

Expected: imports fail because the config and factory entry points do not exist.

- [ ] **Step 3: Implement the wrapper config and locked factory**

`PaperFNOTrainConfig` delegates unchanged data/training controls to an internal `SameFrequencyTrainConfig(model_size="lite")`, but exposes `model_size="paper_fno_lite"`. Its canonical payload adds the complete FNO architecture, fixed CFG candidate list `[1.0]`, tensor parameter count, and real scalar degree count. `with_run_root(path)` returns a new wrapper around `dataclasses.replace(base, run_root=path)`.

- [ ] **Step 4: Test round-trip rejection and identity separation**

Add tests proving canonical JSON round-trips exactly, changing width/modes is rejected, and an FNO checkpoint identity cannot load as U-Net Lite because `model_size` and `config_sha256` differ.

- [ ] **Step 5: Run all related tests and commit**

```bash
pytest -q tests/test_radioflow_framework_lock.py tests/test_same_frequency_training.py tests/test_same_frequency_fno_config.py tests/test_same_frequency_fno_factory.py tests/test_checkpoint_resume.py
git add training/model_factory.py training/same_frequency_fno_config.py tests/test_same_frequency_fno_config.py tests/test_same_frequency_fno_factory.py
git commit -m "feat: lock paper FNO same-frequency configuration"
```

---

### Task 3: Add the dedicated FNO trainer and CLI

**Files:**
- Create: `training/same_frequency_fno_trainer.py`
- Create: `train_same_frequency_fno.py`
- Create: `tests/test_train_same_frequency_fno_cli.py`
- Create: `tests/test_same_frequency_fno_trainer.py`

**Interfaces:**
- Produces: `write_or_validate_fno_run_config(cfg, controls) -> Path`.
- Produces: `run_same_frequency_fno_training(cfg, controls, device, preflight_only=False) -> dict`.
- CLI accepts the same data/array/device/resume/smoke controls as `train_same_frequency.py`; it does not expose width, modes, padding, CFG, seed, optimizer, or stopping-rule overrides.

- [ ] **Step 1: Write failing parser and orchestration tests**

```python
def test_fno_cli_exposes_only_operational_controls():
    args = build_parser().parse_args([
        "--dataset-root", "dataset", "--manifest-path", "manifest.jsonl",
        "--height-stats-path", "height.json", "--run-root", "run",
        "--array-size", "32x32", "--device", "cuda:0",
        "--resume", "auto", "--preflight-only",
    ])
    assert args.array_size == "32x32"
    assert args.preflight_only is True
    assert not hasattr(args, "modes")
    assert not hasattr(args, "width")
```

The trainer test uses a real tiny dataset fixture and a tiny FNO injected at the factory boundary; it asserts that one optimizer-step smoke produces a strict fresh checkpoint containing model, EMA, optimizer, scheduler, scaler, RNG, and FNO identity.

- [ ] **Step 2: Run and verify RED**

Run `pytest -q tests/test_train_same_frequency_fno_cli.py tests/test_same_frequency_fno_trainer.py`.

Expected: module imports fail.

- [ ] **Step 3: Implement trainer reuse without copying the training loop**

`run_same_frequency_fno_training` reuses `preflight_same_frequency`, `build_same_frequency_loaders`, `MultiConfigSRMTrainer`, strict checkpointing, EMA, scheduler, masked loss, and resume logic. It differs only in config deserialization and `build_same_frequency_backbone("paper_fno_lite")`.

- [ ] **Step 4: Run smoke and regression tests**

```bash
pytest -q tests/test_train_same_frequency_fno_cli.py tests/test_same_frequency_fno_trainer.py tests/test_gradient_accumulation.py tests/test_masked_flow_loss.py tests/test_checkpoint_resume.py
```

Expected: all pass with no changes to existing trainer behavior.

- [ ] **Step 5: Commit**

```bash
git add training/same_frequency_fno_trainer.py train_same_frequency_fno.py tests/test_train_same_frequency_fno_cli.py tests/test_same_frequency_fno_trainer.py
git commit -m "feat: add paper FNO same-frequency trainer"
```

---

### Task 4: Route fixed-CFG FNO evaluation through the existing evaluator

**Files:**
- Modify: `evaluation/same_frequency_evaluator.py`
- Create: `evaluate_same_frequency_fno.py`
- Create: `tests/test_evaluate_same_frequency_fno_cli.py`
- Create: `tests/test_same_frequency_fno_evaluator.py`

**Interfaces:**
- Existing U-Net configs retain candidates `(1.0, 1.5, 2.0, 2.5)`.
- `PaperFNOTrainConfig` exposes candidates `(1.0,)`.
- `_prepare_evaluation` calls `build_same_frequency_backbone(cfg.model_size)`.
- Selection payload and validator derive the exact expected candidate list from the config and remain immutable.

- [ ] **Step 1: Write failing evaluator routing tests**

```python
def test_fno_selection_payload_is_locked_to_cfg_one(prepared_fno):
    payload = build_cfg_selection_payload(
        prepared=prepared_fno,
        candidate_metrics={1.0: finite_metrics(db_rmse=10.0)},
    )
    assert payload["candidates"] == [1.0]
    assert payload["selected_scale"] == 1.0

def test_unet_selection_grid_is_unchanged(prepared_unet):
    metrics = {scale: finite_metrics(db_rmse=10.0 + scale) for scale in (1.0, 1.5, 2.0, 2.5)}
    assert build_cfg_selection_payload(
        prepared=prepared_unet, candidate_metrics=metrics,
    )["candidates"] == [1.0, 1.5, 2.0, 2.5]
```

- [ ] **Step 2: Run and verify RED**

Expected: FNO candidate payload is rejected by the current global-grid assertion.

- [ ] **Step 3: Generalize candidate/model boundaries only**

Add a helper that returns `tuple(cfg.cfg_candidates)` when present and otherwise returns the existing global candidates. Replace only hard-coded candidate-grid and model-builder calls. Keep metric accumulation, strict EMA loading, fixed noise, Euler loop, runtime evidence, predictions, visualization, and atomic publication unchanged.

- [ ] **Step 4: Run evaluator and sampling regressions**

```bash
pytest -q tests/test_same_frequency_fno_evaluator.py tests/test_evaluate_same_frequency_fno_cli.py tests/test_same_frequency_evaluator.py tests/test_evaluate_same_frequency_cli.py tests/test_cfg_selection.py tests/test_radioflow_sampling.py
```

- [ ] **Step 5: Commit**

```bash
git add evaluation/same_frequency_evaluator.py evaluate_same_frequency_fno.py tests/test_evaluate_same_frequency_fno_cli.py tests/test_same_frequency_fno_evaluator.py
git commit -m "feat: evaluate paper FNO with fixed CFG one"
```

---

### Task 5: Add reproducible server launch and registered summary

**Files:**
- Create: `scripts/run_same_frequency_fno_server.sh`
- Create: `summarize_same_frequency_fno.py`
- Create: `tests/test_same_frequency_fno_server_script.py`
- Create: `tests/test_summarize_same_frequency_fno.py`

**Interfaces:**
- Server script supports `--dry-run`, `--preflight`, `--smoke`, and `--train`; the default is no action and an explicit mode is required.
- Summary exposes `apply_registered_decision(fno_rmse: Mapping[str,float]) -> dict` and reads exactly three completed `metrics_test.json` files.

- [ ] **Step 1: Write failing launcher and decision tests**

```python
def test_registered_decision_uses_all_three_guards():
    result = apply_registered_decision({"8x8": 11.20, "16x16": 11.30, "32x32": 11.35})
    assert result["mean_delta_db"] >= 0.3
    assert result["n_improved"] == 3
    assert result["h1_confirmed"] is True

def test_large_single_array_regression_disconfirms_h1():
    result = apply_registered_decision({"8x8": 10.5, "16x16": 10.5, "32x32": 12.3})
    assert result["worst_delta_db"] < -0.5
    assert result["h1_confirmed"] is False
```

The launcher dry-run test executes Bash with temporary roots and asserts exactly three commands map GPU 1->8x8, GPU 2->16x16, and GPU 3->32x32, all with `--device cuda:0`, `--resume auto`, and distinct run/log paths.

- [ ] **Step 2: Run and verify RED**

Expected: imports and script path fail because neither artifact exists.

- [ ] **Step 3: Implement explicit server modes and immutable paths**

The script sources `/home/wys/radioflow_20260823/radioflow_remote_env.sh`, then overrides `RADIOFLOW_CODE` to `/home/wys/radioflow_20260823/fno-paper-singlebeam`. It uses the checksummed manifests and writes only under `/home/wys/radioflow_20260823/results/fno_paper_samefreq_6.7ghz`.

- [ ] **Step 4: Implement summary parsing and exact decision output**

Use baseline constants `11.627`, `11.657`, and `11.700`; perform calculations before display rounding; reject missing/duplicate/incomplete arrays and non-finite metrics; write canonical JSON and CSV.

- [ ] **Step 5: Run tests and commit**

```bash
pytest -q tests/test_same_frequency_fno_server_script.py tests/test_summarize_same_frequency_fno.py
git add scripts/run_same_frequency_fno_server.sh summarize_same_frequency_fno.py tests/test_same_frequency_fno_server_script.py tests/test_summarize_same_frequency_fno.py
git commit -m "feat: launch and summarize paper FNO benchmark"
```

---

### Task 6: Verify, deploy, smoke-test, and launch

**Files:**
- Create locally: `artifacts/fno-paper-singlebeam.bundle` temporarily outside Git
- Create remotely: `/home/wys/radioflow_20260823/fno-paper-singlebeam`
- Write remotely: `/home/wys/radioflow_20260823/results/fno_paper_samefreq_6.7ghz/environment.txt`

**Interfaces:**
- Deployment transfers a Git bundle containing `codex/fno-paper-singlebeam`, clones it remotely, and verifies the exact source commit. It does not reuse the broken copied Windows `.git` file in the older server directory.

- [ ] **Step 1: Run complete tests on the server environment**

```bash
source /home/wys/radioflow_20260823/radioflow_remote_env.sh
cd /home/wys/radioflow_20260823/fno-paper-singlebeam
pytest -q
```

Expected: all tests pass; GPU-only FNO AMP test passes on an RTX 3090.

- [ ] **Step 2: Record environment and verify data checksums**

Record source commit, Python/PyTorch/CUDA/driver/GPU versions, `pip freeze`, disk space, and the five preregistered checksums. Abort on any mismatch.

- [ ] **Step 3: Run three preflights and one optimizer-step smoke**

```bash
bash scripts/run_same_frequency_fno_server.sh --preflight
bash scripts/run_same_frequency_fno_server.sh --smoke
```

Expected: all array identities/counts pass; smoke produces a finite strict checkpoint.

- [ ] **Step 4: Launch three independent training processes**

```bash
bash scripts/run_same_frequency_fno_server.sh --train
```

Expected: three live PIDs, one each on physical GPUs 1, 2, and 3, with new array-specific logs and no writes to existing experiment directories.

- [ ] **Step 5: Monitor without outcome peeking and evaluate after completion**

Monitor only liveness, errors, checkpoint freshness, memory, and epoch progress during training. After automatic completion, run fixed-CFG validation and the single test transaction for each array, then run `summarize_same_frequency_fno.py` and the preregistration audit.

- [ ] **Step 6: Final verification commit**

Commit only source, tests, plans, and small result summaries/figures intended for version control. Keep checkpoints and full predictions in the server result root. Report source commit, run directories, checkpoint identities, metrics, decision, and limitations.

