from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from train import ModelEMA
from training.optimization import build_optimizer_step_scheduler


LEGACY_IDENTITY_KEYS = {
    "array_size",
    "model_size",
    "condition_channels",
    "parameter_count",
    "manifest_sha256",
    "split_sha256",
    "schema_sha256",
    "config_sha256",
    "archive_sha256",
    "dataset_revision",
    "radioflow_upstream_base",
    "git_commit",
    "seed",
}

SPARSE_IDENTITY_KEYS = {
    "experiment",
    "array_size",
    "variant",
    "model_size",
    "condition_channels",
    "parameter_count",
    "config_sha256",
    "mask_protocol_sha256",
}


def _checkpoint_module():
    from training import checkpointing

    return checkpointing


def _legacy_identity(model: torch.nn.Module):
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


def _sparse_identity(
    model: torch.nn.Module,
    *,
    experiment: str = "sparse_same_frequency_6.7_single_beam",
    array_size: str = "8x8",
    variant: str = "beam_masked",
    condition_channels: int = 5,
    config_sha256: str = "9" * 64,
    mask_protocol_sha256: str = "a" * 64,
):
    module = _checkpoint_module()
    return module.CheckpointIdentity(
        experiment=experiment,
        array_size=array_size,
        variant=variant,
        model_size="lite",
        condition_channels=condition_channels,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        config_sha256=config_sha256,
        mask_protocol_sha256=mask_protocol_sha256,
    )


def _trainer_state():
    module = _checkpoint_module()
    return module.TrainerState(
        completed_epochs=1,
        next_epoch_index=1,
        optimizer_step=1,
        micro_batches_seen=2,
        samples_seen=4,
        best_val_db_rmse=1.0,
        epochs_without_improvement=0,
        history=({"epoch": 1, "train_loss": 0.5, "val_db_rmse": 1.0},),
    )


def _seed_all() -> None:
    import random

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
    x = torch.rand(1, 2, generator=generator)
    target = torch.tensor([[0.25]])
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(x), target).backward()
    optimizer.step()
    ema.update(model)
    scheduler.step()


def _save_checkpoint(path: Path, identity) -> None:
    module = _checkpoint_module()
    _seed_all()
    bundle = _bundle()
    _training_step(bundle)
    module.save_checkpoint_atomic(
        path,
        model=bundle[0],
        ema=bundle[1],
        optimizer=bundle[2],
        scheduler=bundle[3],
        scaler=bundle[4],
        trainer_state=_trainer_state(),
        identity=identity(bundle[0]),
        train_generator=bundle[5],
    )


def _load_checkpoint(path: Path, expected_identity):
    module = _checkpoint_module()
    bundle = _bundle()
    return module.load_checkpoint_strict(
        path,
        model=bundle[0],
        ema=bundle[1],
        optimizer=bundle[2],
        scheduler=bundle[3],
        scaler=bundle[4],
        expected_identity=expected_identity(bundle[0]),
        train_generator=bundle[5],
    )


def test_legacy_identity_keys_remain_unchanged() -> None:
    identity = _legacy_identity(torch.nn.Linear(2, 1))

    assert set(identity.to_dict()) == LEGACY_IDENTITY_KEYS


def test_legacy_identity_preserves_historical_positional_argument_order() -> None:
    module = _checkpoint_module()
    model = torch.nn.Linear(2, 1)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    identity = module.CheckpointIdentity(
        "8x8",
        "lite",
        3,
        parameter_count,
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
        "6" * 40,
        "7" * 40,
        "8" * 40,
        42,
    )

    assert identity.array_size == "8x8"
    assert identity.model_size == "lite"
    assert identity.condition_channels == 3
    assert identity.parameter_count == parameter_count
    assert identity.manifest_sha256 == "1" * 64
    assert identity.split_sha256 == "2" * 64
    assert identity.schema_sha256 == "3" * 64
    assert identity.config_sha256 == "4" * 64
    assert identity.archive_sha256 == "5" * 64
    assert identity.dataset_revision == "6" * 40
    assert identity.radioflow_upstream_base == "7" * 40
    assert identity.git_commit == "8" * 40
    assert identity.seed == 42
    identity.validate()
    assert tuple(identity.to_dict()) == module.CheckpointIdentity.LEGACY_KEYS


def test_sparse_identity_uses_distinct_key_schema() -> None:
    identity = _sparse_identity(torch.nn.Linear(2, 1))

    assert set(identity.to_dict()) == SPARSE_IDENTITY_KEYS


def test_sparse_checkpoint_round_trips_without_changing_checkpoint_api(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sparse.pt"
    _save_checkpoint(path, _sparse_identity)

    restored = _load_checkpoint(path, _sparse_identity)

    assert restored == _trainer_state()


@pytest.mark.parametrize(
    ("field", "saved_value", "expected_value"),
    [
        ("condition_channels", 4, 5),
        ("variant", "no_beam_masked", "beam_masked"),
        ("array_size", "16x16", "8x8"),
        ("config_sha256", "b" * 64, "c" * 64),
        ("mask_protocol_sha256", "d" * 64, "e" * 64),
    ],
)
def test_sparse_checkpoint_load_rejects_identity_mismatch_before_restore(
    tmp_path: Path,
    field: str,
    saved_value,
    expected_value,
) -> None:
    module = _checkpoint_module()
    path = tmp_path / f"{field}.pt"
    _save_checkpoint(
        path,
        lambda model: replace(_sparse_identity(model), **{field: saved_value}),
    )

    with pytest.raises(module.CheckpointIdentityError, match=field):
        _load_checkpoint(
            path,
            lambda model: replace(_sparse_identity(model), **{field: expected_value}),
        )


def test_sparse_run_rejects_legacy_checkpoint_before_state_restore(
    tmp_path: Path,
) -> None:
    module = _checkpoint_module()
    path = tmp_path / "legacy.pt"
    _save_checkpoint(path, _legacy_identity)

    with pytest.raises(module.CheckpointIdentityError, match="schema"):
        _load_checkpoint(path, _sparse_identity)
