# Attention-Conditioned Multiscale UNO-FM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a size-matched multiscale UNO velocity backbone for the frozen RadioFlow 6.7 GHz, zero-degree, single-beam experiment without changing existing model behavior or result identities.

**Architecture:** Keep the current `BasicUNetEncoder`, strict full-condition CFG dropout, Flow Matching objective, evaluator, and two-step Euler sampler. Replace the four serial full-resolution Attention-FNO blocks with a U-shaped state path containing four encoder stages, one bottleneck, four decoder stages, native-scale CA/SA injection, fixed-width 24-channel spectral operators, deterministic resize operators, and U-Net skips.

**Tech Stack:** Python 3.10+, PyTorch, MONAI, pytest, Bash, existing RadioFlow training/checkpoint/evaluation framework.

**Spec:** `docs/superpowers/specs/2026-08-31-attention-multiscale-uno-fm-design.md`

## Global Constraints

- Start from commit `44df1ee4e0c40d5600bfe5be62ebbab6cede6431` in a new clean worktree and branch; do not implement in the dirty `hybrid-fno-u-singlebeam` or `multiconfig-srm-01x` worktrees.
- Existing U-Net, paper-FNO, full-resolution Attention-FNO, Hybrid FNO-U, sparse Task 2 models, CLIs, configs, tests, checkpoints, and result roots are read-only.
- Model input order is exactly `x_t, tx_mask, height, beam_map, grid_x, grid_y`; time remains a separate `[B]` tensor.
- External state channels are `(32,64,128,256,256)`, internal operator width is `24`, modes are `(12,12,8,4,4)`, and padding is `(9,5,3,2,1)`.
- The state path contains exactly nine enabled current `CrossAttention` modules and never instantiates `CrossAttention_old`.
- Expected parameter counts are exactly `3,059,355` tensor elements and `3,925,659` independent real scalars.
- Training remains masked conditional Flow Matching; evaluation remains EMA `best.pt`, CFG `1.0`, fixed hash noise, and two-step Euler.
- New results use a new root and never overwrite `/home/wys/radioflow_20260823/results/attention_fno_samefreq_6.7ghz`.

---

### Task 1: Create the isolated implementation worktree and verify the baseline

**Files:**
- Read: repository at `E:/RadioFlow-worktrees/hybrid-fno-u-singlebeam`
- Create worktree: `E:/RadioFlow-worktrees/multiscale-uno-singlebeam`
- Create branch: `codex/multiscale-uno-singlebeam`

**Interfaces:**
- Consumes: Git commit `44df1ee4e0c40d5600bfe5be62ebbab6cede6431`.
- Produces: a clean worktree in which every later task runs.

- [ ] **Step 1: Confirm the source commit and preserve the dirty worktree**

Run:

```powershell
git -C E:\RadioFlow-worktrees\hybrid-fno-u-singlebeam rev-parse HEAD
git -C E:\RadioFlow-worktrees\hybrid-fno-u-singlebeam status --short
```

Expected: HEAD is `44df1ee4e0c40d5600bfe5be62ebbab6cede6431`; the untracked `tests/test_hybrid_fno_u_model.py` is visible and is not moved, deleted, or added.

- [ ] **Step 2: Create the feature worktree**

Run:

```powershell
git -C E:\RadioFlow worktree add -b codex/multiscale-uno-singlebeam E:\RadioFlow-worktrees\multiscale-uno-singlebeam 44df1ee4e0c40d5600bfe5be62ebbab6cede6431
```

Expected: Git reports a new worktree at the exact commit.

- [ ] **Step 3: Run the frozen Attention-FNO regression suite before editing**

Run from the project environment:

```powershell
pytest tests/test_attention_fno_model.py tests/test_attention_fno_factory.py tests/test_same_frequency_attention_fno_config.py tests/test_same_frequency_attention_fno_trainer.py -q
```

Expected: all selected tests pass.

---

### Task 2: Implement one residual CA/SA-conditioned FNO stage

**Files:**
- Create: `model/attention_multiscale_uno.py`
- Create: `tests/test_attention_multiscale_uno_stage.py`

**Interfaces:**
- Consumes: `model.fno.SpectralConv2d`, `model.unet.basic_unet_denose.CrossAttention`, and `nonlinearity`.
- Produces: `AttentionConditionedFNOStage.forward(value, condition, time_embedding) -> Tensor` with shape-preserving external channels.

- [ ] **Step 1: Write the failing shape, residual, gradient, and validation tests**

Create tests using this exact construction:

```python
def _stage() -> AttentionConditionedFNOStage:
    return AttentionConditionedFNOStage(
        channels=8,
        embedding_channels=4,
        operator_width=4,
        modes=2,
        padding=1,
        attention_reduction=4,
    )


def test_stage_preserves_external_shape_and_backpropagates_every_branch() -> None:
    stage = _stage()
    value = torch.randn(2, 8, 16, 16, requires_grad=True)
    condition = torch.randn(2, 4, 16, 16)
    time = torch.randn(2, 512)
    output = stage(value, condition, time)
    output.square().mean().backward()
    assert output.shape == value.shape
    assert torch.isfinite(output).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert stage.spectral.weights1.grad is not None
    assert stage.local.weight.grad is not None
    assert stage.time_projection.weight.grad is not None
    assert stage.attention.embedding_proj.weight.grad is not None


def test_zero_operator_update_leaves_the_attended_residual() -> None:
    stage = _stage().eval()
    value = torch.randn(1, 8, 16, 16)
    condition = torch.randn(1, 4, 16, 16)
    time = torch.randn(1, 512)
    with torch.no_grad():
        stage.projection.weight.zero_()
        stage.projection.bias.zero_()
        expected = stage.attention(value, condition)
        actual = stage(value, condition, time)
    assert torch.equal(actual, expected)
```

Also test invalid channel, spatial, time, modes, and padding values, plus CUDA float16 autocast when CUDA is available.

- [ ] **Step 2: Run the stage tests and confirm failure**

Run:

```powershell
pytest tests/test_attention_multiscale_uno_stage.py -q
```

Expected: collection fails because `model.attention_multiscale_uno` does not exist.

- [ ] **Step 3: Implement the stage and strict input checks**

Implement this public signature:

```python
class AttentionConditionedFNOStage(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        embedding_channels: int,
        operator_width: int,
        modes: int,
        padding: int,
        time_channels: int = 512,
        attention_reduction: int = 16,
    ) -> None: ...

    def forward(
        self,
        value: Tensor,
        condition: Tensor,
        time_embedding: Tensor,
    ) -> Tensor: ...
```

The module members are exactly `attention`, `lifting`, `spectral`, `local`, `time_projection`, and `projection`. Compute the update as specified in the design document and return `attended + projected_delta`.

- [ ] **Step 4: Run the stage and existing spectral tests**

Run:

```powershell
pytest tests/test_attention_multiscale_uno_stage.py tests/test_fno_model.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the independently testable stage**

```powershell
git add model/attention_multiscale_uno.py tests/test_attention_multiscale_uno_stage.py
git commit -m "feat: add conditioned multiscale FNO stage"
```

---

### Task 3: Assemble the U-shaped multiscale velocity model

**Files:**
- Modify: `model/attention_multiscale_uno.py`
- Create: `tests/test_attention_multiscale_uno_model.py`

**Interfaces:**
- Consumes: `AttentionConditionedFNOStage` from Task 2 and `BasicUNetEncoder`.
- Produces: `AttentionMultiscaleUNO2d`, compatible with the existing `MultiConfigSRMTrainer` and evaluator model API.

- [ ] **Step 1: Write failing topology and data-flow tests**

Use a tiny model constructor that keeps five levels while reducing memory:

```python
def _tiny_model(cfg_drop_prob: float = 0.0) -> AttentionMultiscaleUNO2d:
    return AttentionMultiscaleUNO2d(
        state_channels=(8, 8, 16, 16, 16),
        operator_width=4,
        operator_modes=(2, 2, 2, 1, 1),
        operator_padding=(1, 1, 1, 1, 1),
        encoder_features=(4, 4, 8, 16, 32, 4),
        attention_reduction=4,
        cfg_drop_prob=cfg_drop_prob,
    )
```

The tests must assert:

```python
assert model.lifting.in_channels == 6
assert model.lifting.out_channels == 8
assert len(model.encoder_stages) == 4
assert len(model.decoder_stages) == 4
assert type(model.bottleneck) is AttentionConditionedFNOStage
assert sum(isinstance(m, CrossAttention) for m in model.modules()) == 9
assert not any(isinstance(m, CrossAttention_old) for m in model.modules())
```

Register hooks on all nine stages and assert one 32 x 32 forward produces the resolution sequence:

```python
[(32,32), (16,16), (8,8), (4,4), (2,2),
 (4,4), (8,8), (16,16), (32,32)]
```

Also test coordinate direction, output `[B,1,H,W]`, finite backward through every spectral stage, exact CFG=1 conditional equality, and `cfg_drop_prob=1.0` equality with the explicit unconditional branch.

- [ ] **Step 2: Run the model tests and confirm failure**

```powershell
pytest tests/test_attention_multiscale_uno_model.py -q
```

Expected: imports or missing model members fail.

- [ ] **Step 3: Implement deterministic resize modules**

Add private modules with exact behavior:

```python
class _Downsample2d(nn.Module):
    # AvgPool2d(2), then Conv2d(in_channels,out_channels,1,bias=True)

class _UpsampleFuse2d(nn.Module):
    # bilinear resize to skip.shape[-2:], concatenate, then bias-enabled 1x1
```

Validate even encoder spatial dimensions and exact skip batch/spatial compatibility.

- [ ] **Step 4: Implement `AttentionMultiscaleUNO2d`**

Use this constructor and public methods:

```python
class AttentionMultiscaleUNO2d(nn.Module):
    def __init__(
        self,
        *,
        condition_channels: int = 3,
        state_channels: Sequence[int] = (32, 64, 128, 256, 256),
        operator_width: int = 24,
        operator_modes: Sequence[int] = (12, 12, 8, 4, 4),
        operator_padding: Sequence[int] = (9, 5, 3, 2, 1),
        encoder_features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        cfg_drop_prob: float = 0.25,
        attention_reduction: int = 16,
        activation_checkpointing: bool = False,
    ) -> None: ...

    @staticmethod
    def coordinate_grid(state: Tensor) -> tuple[Tensor, Tensor]: ...

    def embed_model(self, condition: Tensor) -> list[Tensor]: ...

    def forward(
        self,
        image: Tensor | None = None,
        x: Tensor | None = None,
        pred_type: str = "denoise",
        step: Tensor | float | int | None = None,
        embedding: Sequence[Tensor] | None = None,
    ) -> Tensor: ...

    def forward_with_cfg(
        self,
        *,
        image: Tensor,
        x: Tensor,
        step: Tensor | float | int,
        embedding: Sequence[Tensor] | None = None,
        cfg_scale: float = 1.0,
    ) -> Tensor: ...
```

Reuse the finite-under-AMP dropout logic from `AttentionConditionedFNO2d`: encode the real condition first, then jointly zero the selected samples in the raw condition and all five embeddings.

- [ ] **Step 5: Run model, attention, and framework-lock tests**

```powershell
pytest tests/test_attention_multiscale_uno_model.py tests/test_attention_fno_model.py tests/test_radioflow_framework_lock.py -q
```

Expected: all tests pass and no existing test changes are needed.

- [ ] **Step 6: Commit the complete backbone**

```powershell
git add model/attention_multiscale_uno.py tests/test_attention_multiscale_uno_model.py
git commit -m "feat: assemble attention multiscale UNO backbone"
```

---

### Task 4: Register and scientifically lock the new model and configuration

**Files:**
- Modify: `training/model_factory.py`
- Create: `training/same_frequency_multiscale_uno_config.py`
- Create: `tests/test_attention_multiscale_uno_factory.py`
- Create: `tests/test_same_frequency_multiscale_uno_config.py`

**Interfaces:**
- Consumes: `AttentionMultiscaleUNO2d`.
- Produces: `build_attention_multiscale_uno()`, factory model size `attention_multiscale_uno_lite`, and `MultiscaleUNOTrainConfig`.

- [ ] **Step 1: Write the failing factory test**

```python
def test_factory_registers_multiscale_uno_without_mutating_existing_models() -> None:
    model = build_attention_multiscale_uno()
    assert MULTISCALE_UNO_MODEL_SIZE == "attention_multiscale_uno_lite"
    assert type(model) is AttentionMultiscaleUNO2d
    assert tuple(model.state_channels) == (32, 64, 128, 256, 256)
    assert model.operator_width == 24
    assert tuple(model.operator_modes) == (12, 12, 8, 4, 4)
    assert tuple(model.operator_padding) == (9, 5, 3, 2, 1)
    assert count_tensor_parameters(model) == 3_059_355
    assert count_real_scalar_parameters(model) == 3_925_659
    assert type(build_same_frequency_backbone("attention_fno_lite")) is AttentionConditionedFNO2d
    assert type(build_same_frequency_backbone("lite")) is DiffUNet
```

- [ ] **Step 2: Write the failing immutable-config tests**

Assert the scientific payload contains exactly:

```python
{
    "experiment": "same_frequency_6.7_single_beam_attention_multiscale_uno",
    "model_size": "attention_multiscale_uno_lite",
    "backbone": "attention_conditioned_multiscale_uno2d",
    "state_channels": [32, 64, 128, 256, 256],
    "operator_width": 24,
    "operator_modes": [12, 12, 8, 4, 4],
    "operator_padding": [9, 5, 3, 2, 1],
    "operator_stages": 9,
    "condition_injection": "native_scale_CA_SA_encoder_decoder",
    "downsample": "avgpool2_plus_1x1",
    "upsample": "bilinear_concat_1x1",
    "state_skip_connections": True,
    "tensor_parameter_count": 3_059_355,
    "real_scalar_parameter_count": 3_925_659,
}
```

Test JSON round-trip, path-independent config hash, rejection of width/modes/padding/CFG drift, and the unchanged 560/80/160, 1000-epoch, patience-20 protocol.

- [ ] **Step 3: Run tests and confirm failure**

```powershell
pytest tests/test_attention_multiscale_uno_factory.py tests/test_same_frequency_multiscale_uno_config.py -q
```

Expected: imports fail.

- [ ] **Step 4: Implement the factory registration**

Add `build_attention_multiscale_uno()` and one new branch in `build_same_frequency_backbone`. Lock all architecture fields, nine attention modules, CFG dropout, and both parameter counts. Do not alter existing factory branches or constants.

- [ ] **Step 5: Implement `MultiscaleUNOTrainConfig`**

Mirror the immutable wrapper and strict `from_json` semantics of `AttentionFNOTrainConfig`, but use the new constants and scientific payload. Set `model_size` to `attention_multiscale_uno_lite` while the wrapped base config remains `model_size="lite"`.

- [ ] **Step 6: Run factory/config tests and the existing registrations**

```powershell
pytest tests/test_attention_multiscale_uno_factory.py tests/test_same_frequency_multiscale_uno_config.py tests/test_attention_fno_factory.py tests/test_same_frequency_fno_factory.py tests/test_radioflow_framework_lock.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit registration and config**

```powershell
git add training/model_factory.py training/same_frequency_multiscale_uno_config.py tests/test_attention_multiscale_uno_factory.py tests/test_same_frequency_multiscale_uno_config.py
git commit -m "feat: register multiscale UNO experiment identity"
```

---

### Task 5: Add one lifecycle CLI and a resumable trainer adapter

**Files:**
- Create: `training/same_frequency_multiscale_uno_trainer.py`
- Create: `run_same_frequency_multiscale_uno.py`
- Create: `tests/test_same_frequency_multiscale_uno_trainer.py`
- Create: `tests/test_same_frequency_multiscale_uno_cli.py`

**Interfaces:**
- Consumes: `MultiscaleUNOTrainConfig`, existing same-frequency preflight/loaders, `MultiConfigSRMTrainer`, checkpoint identity, evaluator, and device resolver.
- Produces: one executable with `train`, `select-cfg`, and `test` subcommands.

- [ ] **Step 1: Write the failing trainer smoke test**

Copy the synthetic 32 x 32 dataset fixture pattern from `tests/test_same_frequency_attention_fno_trainer.py`. Monkeypatch model construction to a tiny `AttentionMultiscaleUNO2d` and assert one CPU optimizer step writes a checkpoint whose identity contains:

```python
assert payload["run_identity"]["model_size"] == "attention_multiscale_uno_lite"
assert payload["run_identity"]["config_sha256"] == cfg.config_sha256
assert payload["trainer_state"]["optimizer_step"] == 1
assert payload["optimizer"]["state"]
```

Also test config revalidation and strict rejection of a checkpoint from `attention_fno_lite`.

- [ ] **Step 2: Write the failing lifecycle CLI test**

The parser must accept exactly these subcommands:

```text
train      common data arguments + run root + resume + one operational stop/smoke control
select-cfg common data arguments + run root + results root
test       common data arguments + run root + results root
```

Assert the CLI exposes no width, modes, padding, channels, model-size, or CFG-scale override.

- [ ] **Step 3: Implement the trainer adapter**

Expose:

```python
def write_or_validate_multiscale_uno_run_config(
    cfg: MultiscaleUNOTrainConfig,
    controls: InvocationControls,
) -> Path: ...

def run_same_frequency_multiscale_uno_training(
    cfg: MultiscaleUNOTrainConfig,
    controls: InvocationControls,
    device: torch.device,
    *,
    preflight_only: bool = False,
) -> dict[str, Any]: ...
```

Use the same deterministic seed, loader, optimizer-step scheduler, EMA,
`ComplexGradScaler`, full-state checkpoint reload, resume behavior, and smoke
directory isolation as the existing Attention-FNO trainer.

- [ ] **Step 4: Implement the single lifecycle CLI**

`run_same_frequency_multiscale_uno.py` constructs one frozen config and routes:

```python
train      -> run_same_frequency_multiscale_uno_training
select-cfg -> evaluation.same_frequency_evaluator.run_cfg_selection
test       -> evaluation.same_frequency_evaluator.run_test_evaluation
```

Print one JSON result to stdout and keep all architecture controls out of argparse.

- [ ] **Step 5: Run trainer, CLI, and evaluator tests**

```powershell
pytest tests/test_same_frequency_multiscale_uno_trainer.py tests/test_same_frequency_multiscale_uno_cli.py tests/test_same_frequency_evaluator.py tests/test_radioflow_sampling.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit lifecycle integration**

```powershell
git add training/same_frequency_multiscale_uno_trainer.py run_same_frequency_multiscale_uno.py tests/test_same_frequency_multiscale_uno_trainer.py tests/test_same_frequency_multiscale_uno_cli.py
git commit -m "feat: add multiscale UNO experiment lifecycle"
```

---

### Task 6: Add the isolated three-array server launcher

**Files:**
- Create: `scripts/run_same_frequency_multiscale_uno_server.sh`
- Create: `tests/test_same_frequency_multiscale_uno_server_script.py`

**Interfaces:**
- Consumes: `run_same_frequency_multiscale_uno.py`.
- Produces: explicit `--dry-run`, `--preflight`, `--smoke`, `--train`, `--select-cfg`, and `--test` operations for all three arrays.

- [ ] **Step 1: Write the failing launcher contract tests**

Assert the script contains these isolated defaults:

```text
CODE_ROOT=/home/wys/radioflow_20260823/multiscale-uno-singlebeam
RESULT_ROOT=/home/wys/radioflow_20260823/results/multiscale_uno_samefreq_6.7ghz
ARRAYS=(8x8 16x16 32x32)
GPUS=(0 1 2)
```

Run `bash script --dry-run` and assert exactly three commands, each with one physical GPU, `--resume auto`, the matching manifest, and the new lifecycle CLI. Assert an omitted or unknown mode exits with code 2.

- [ ] **Step 2: Implement the launcher**

Reuse the PID/log/config isolation and explicit-mode style of `run_same_frequency_attention_fno_server.sh`. Keep GPU 3 free. `--smoke` must wait for all three one-step runs and fail if any exit nonzero. Formal training uses `nohup`, one log and PID file per array, and refuses a live existing PID.

- [ ] **Step 3: Run shell and CLI tests**

```powershell
pytest tests/test_same_frequency_multiscale_uno_server_script.py tests/test_same_frequency_multiscale_uno_cli.py tests/test_same_frequency_attention_fno_server_script.py -q
```

Expected: all tests pass and the old launcher remains byte-for-byte unchanged.

- [ ] **Step 4: Commit the launcher**

```powershell
git add scripts/run_same_frequency_multiscale_uno_server.sh tests/test_same_frequency_multiscale_uno_server_script.py
git commit -m "feat: add multiscale UNO server launcher"
```

---

### Task 7: Run regression, verify the parameter lock, and document execution

**Files:**
- Create: `docs/experiments/2026-08-31-attention-multiscale-uno-singlebeam.md`
- Modify only if required by test discovery: `README.md`

**Interfaces:**
- Consumes: all earlier tasks.
- Produces: a verified implementation ready for server preflight and smoke testing.

- [ ] **Step 1: Run the focused new suite**

```powershell
pytest tests/test_attention_multiscale_uno_stage.py tests/test_attention_multiscale_uno_model.py tests/test_attention_multiscale_uno_factory.py tests/test_same_frequency_multiscale_uno_config.py tests/test_same_frequency_multiscale_uno_trainer.py tests/test_same_frequency_multiscale_uno_cli.py tests/test_same_frequency_multiscale_uno_server_script.py -q
```

Expected: all pass.

- [ ] **Step 2: Run the complete repository suite**

```powershell
pytest -q
```

Expected: all CPU tests pass; CUDA-only tests are skipped only when CUDA is unavailable.

- [ ] **Step 3: Run a CUDA forward/backward memory probe**

On one 24 GB server GPU, instantiate the locked model under float16 autocast, run one `B=2` 256 x 256 forward/backward, and record `torch.cuda.max_memory_allocated()`. If it OOMs, use micro-batch 1 with accumulation 56; do not change architecture constants.

- [ ] **Step 4: Write the experiment execution record**

Record the commit hash, model/config hashes, parameter counts, protocol, code/result roots, GPU mapping, preflight command, smoke command, formal train command, evaluation command, and the measured micro-batch memory result in `docs/experiments/2026-08-31-attention-multiscale-uno-singlebeam.md`.

- [ ] **Step 5: Commit verification documentation**

```powershell
git add docs/experiments/2026-08-31-attention-multiscale-uno-singlebeam.md README.md
git commit -m "docs: record multiscale UNO experiment protocol"
```

Do not add `README.md` if it did not require a change.

---

### Task 8: Deploy, preflight, smoke, and launch without touching old results

**Files:**
- Server code root: `/home/wys/radioflow_20260823/multiscale-uno-singlebeam`
- Server result root: `/home/wys/radioflow_20260823/results/multiscale_uno_samefreq_6.7ghz`

**Interfaces:**
- Consumes: the verified feature commit from Task 7 and the existing remote environment file `/home/wys/radioflow_20260823/radioflow_remote_env.sh`.
- Produces: three isolated resumable training runs.

- [ ] **Step 1: Transfer the committed source snapshot**

Transfer only tracked files from the clean feature worktree to the new server code root. Verify the remote Git/source hash and confirm the old Attention-FNO code and result roots are unchanged.

- [ ] **Step 2: Run three manifest preflights**

```bash
bash scripts/run_same_frequency_multiscale_uno_server.sh --preflight
```

Expected: each array reports 560/80/160 samples, 6.7 GHz, zero steering, and the correct inferred beam ID.

- [ ] **Step 3: Run and inspect three one-step CUDA smoke tests**

```bash
bash scripts/run_same_frequency_multiscale_uno_server.sh --smoke
```

Expected: three fresh reloadable full-state checkpoints, finite loss/gradients, correct model/config identities, and no OOM. Do not start formal training if any smoke run fails.

- [ ] **Step 4: Launch three independent formal runs**

```bash
bash scripts/run_same_frequency_multiscale_uno_server.sh --train
```

Expected: one live PID and growing log/config/checkpoint activity for each of 8x8, 16x16, and 32x32 on GPUs 0, 1, and 2 respectively.

- [ ] **Step 5: Verify launch evidence**

Check `nvidia-smi`, all three PID files, the first finite training metric, immutable `config.json`, and `last.pt` creation. Record the evidence in the experiment document; do not claim completion until training and test evaluation actually finish.

