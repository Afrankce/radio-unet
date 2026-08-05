from __future__ import annotations

import copy
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from train import ModelEMA
from training.optimization import build_optimizer_step_scheduler


def _checkpoint_module():
    from training import checkpointing

    return checkpointing


def _identity(model: torch.nn.Module):
    module = _checkpoint_module()
    return module.CheckpointIdentity(
        array_size="8x8",
        model_size="lite",
        condition_channels=3,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        manifest_sha256="1" * 64,
        split_sha256="2" * 64,
        schema_sha256="3" * 64,
        config_sha256="4" * 64,
        archive_sha256="5" * 64,
        dataset_revision="6" * 40,
        radioflow_upstream_base="7" * 40,
        git_commit="8" * 40,
        seed=42,
    )


def _trainer_state(completed: int):
    module = _checkpoint_module()
    history = tuple(
        {
            "epoch": index + 1,
            "train_loss": 0.5 / (index + 1),
            "val_db_rmse": 10.0 / (index + 1),
        }
        for index in range(completed)
    )
    return module.TrainerState(
        completed_epochs=completed,
        next_epoch_index=completed,
        optimizer_step=completed,
        micro_batches_seen=completed * 2,
        samples_seen=completed * 4,
        best_val_db_rmse=10.0 / max(completed, 1),
        epochs_without_improvement=0,
        history=history,
    )


def _seed_all() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def _bundle():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = build_optimizer_step_scheduler(
        optimizer,
        total_steps=10,
        warmup_steps=2,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    ema = ModelEMA(model, decay=0.999)
    generator = torch.Generator(device="cpu").manual_seed(4242)
    return model, ema, optimizer, scheduler, scaler, generator


def _training_step(bundle) -> None:
    model, ema, optimizer, scheduler, _scaler, generator = bundle
    python_value = random.random()
    numpy_value = float(np.random.random())
    cpu_value = float(torch.rand(()).item())
    loader_value = float(torch.rand((), generator=generator).item())
    if torch.cuda.is_available():
        _ = torch.rand((), device="cuda")
    x = torch.tensor([[python_value + numpy_value, cpu_value + loader_value]])
    target = torch.tensor([[0.25]])
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(x), target).backward()
    optimizer.step()
    ema.update(model)
    scheduler.step()


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _next_rng_values(generator: torch.Generator) -> dict[str, Any]:
    values: dict[str, Any] = {
        "python": random.random(),
        "numpy": np.random.random(3),
        "torch_cpu": torch.rand(3),
        "loader": torch.rand(3, generator=generator),
    }
    if torch.cuda.is_available():
        values["torch_cuda"] = torch.rand(3, device="cuda").cpu()
    return values


def test_atomic_resume_matches_continuous_training_and_all_rngs(tmp_path: Path) -> None:
    module = _checkpoint_module()
    _seed_all()
    continuous = _bundle()
    _training_step(continuous)
    _training_step(continuous)
    continuous_rng = _next_rng_values(continuous[-1])

    _seed_all()
    interrupted = _bundle()
    _training_step(interrupted)
    path = tmp_path / "last.pt"
    module.save_checkpoint_atomic(
        path,
        model=interrupted[0],
        ema=interrupted[1],
        optimizer=interrupted[2],
        scheduler=interrupted[3],
        scaler=interrupted[4],
        trainer_state=_trainer_state(1),
        identity=_identity(interrupted[0]),
        train_generator=interrupted[5],
    )
    assert path.is_file()
    assert not path.with_name(path.name + ".tmp").exists()

    fresh = _bundle()
    restored = module.load_checkpoint_strict(
        path,
        model=fresh[0],
        ema=fresh[1],
        optimizer=fresh[2],
        scheduler=fresh[3],
        scaler=fresh[4],
        expected_identity=_identity(fresh[0]),
        train_generator=fresh[5],
    )
    assert restored == _trainer_state(1)
    _training_step(fresh)
    resumed_rng = _next_rng_values(fresh[-1])

    assert _nested_equal(continuous[0].state_dict(), fresh[0].state_dict())
    assert _nested_equal(
        continuous[1].ema_model.state_dict(), fresh[1].ema_model.state_dict()
    )
    assert _nested_equal(continuous[2].state_dict(), fresh[2].state_dict())
    assert continuous[3].state_dict() == fresh[3].state_dict()
    assert continuous[4].state_dict() == fresh[4].state_dict()
    assert _nested_equal(continuous_rng, resumed_rng)


def _valid_checkpoint(tmp_path: Path):
    module = _checkpoint_module()
    _seed_all()
    bundle = _bundle()
    _training_step(bundle)
    path = tmp_path / "valid.pt"
    identity = _identity(bundle[0])
    module.save_checkpoint_atomic(
        path,
        model=bundle[0],
        ema=bundle[1],
        optimizer=bundle[2],
        scheduler=bundle[3],
        scaler=bundle[4],
        trainer_state=_trainer_state(1),
        identity=identity,
        train_generator=bundle[5],
    )
    return module, path, identity


def _load_fresh(module, path: Path, identity):
    bundle = _bundle()
    return module.load_checkpoint_strict(
        path,
        model=bundle[0],
        ema=bundle[1],
        optimizer=bundle[2],
        scheduler=bundle[3],
        scaler=bundle[4],
        expected_identity=identity,
        train_generator=bundle[5],
    )


def test_checkpoint_load_rejects_missing_and_truncated_files(tmp_path: Path) -> None:
    module = _checkpoint_module()
    model = torch.nn.Linear(2, 1)

    with pytest.raises(module.CheckpointError, match="does not exist"):
        _load_fresh(module, tmp_path / "missing.pt", _identity(model))
    truncated = tmp_path / "truncated.pt"
    truncated.write_bytes(b"not a torch checkpoint")
    with pytest.raises(module.CheckpointError, match="cannot load checkpoint"):
        _load_fresh(module, truncated, _identity(model))


@pytest.mark.parametrize(
    "field",
    [
        "array_size",
        "model_size",
        "config_sha256",
        "manifest_sha256",
        "split_sha256",
        "schema_sha256",
        "archive_sha256",
    ],
)
def test_checkpoint_load_rejects_identity_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    module, path, identity = _valid_checkpoint(tmp_path)
    replacement = "changed" if field in {"array_size", "model_size"} else "f" * 64
    wrong = replace(identity, **{field: replacement})

    with pytest.raises(module.CheckpointIdentityError, match=field):
        _load_fresh(module, path, wrong)


def test_checkpoint_load_rejects_missing_top_key_and_wrong_state_keys(
    tmp_path: Path,
) -> None:
    module, path, identity = _valid_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.pop("optimizer")
    missing = tmp_path / "missing-key.pt"
    torch.save(payload, missing)

    with pytest.raises(module.CheckpointError, match="top-level keys"):
        _load_fresh(module, missing, identity)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["model"].pop(next(iter(payload["model"])))
    wrong_state = tmp_path / "wrong-state.pt"
    torch.save(payload, wrong_state)
    with pytest.raises(module.CheckpointError, match="model state keys"):
        _load_fresh(module, wrong_state, identity)


def test_checkpoint_load_rejects_nonfinite_best_metric(tmp_path: Path) -> None:
    module, path, identity = _valid_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["trainer_state"]["best_val_db_rmse"] = float("inf")
    invalid = tmp_path / "nonfinite.pt"
    torch.save(payload, invalid)

    with pytest.raises(module.CheckpointError, match="best_val_db_rmse"):
        _load_fresh(module, invalid, identity)


def test_evaluation_loader_uses_only_strict_ema_state(tmp_path: Path) -> None:
    module, path, identity = _valid_checkpoint(tmp_path)
    target = torch.nn.Linear(2, 1)

    state = module.load_ema_for_evaluation(
        path,
        model=target,
        expected_identity=identity,
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert _nested_equal(target.state_dict(), payload["ema"])
    assert state == _trainer_state(1)


def test_metrics_csv_is_rebuilt_exactly_from_checkpoint_history(tmp_path: Path) -> None:
    module = _checkpoint_module()
    state = _trainer_state(2)
    path = tmp_path / "metrics.csv"

    module.rebuild_metrics_csv(path, state.history)
    first = path.read_bytes()
    module.rebuild_metrics_csv(path, state.history)

    assert path.read_bytes() == first
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[0] == "epoch,train_loss,val_db_rmse"

