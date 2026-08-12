from __future__ import annotations

import copy
import csv
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


CHECKPOINT_KEYS = {
    "schema_version",
    "model",
    "ema",
    "optimizer",
    "scheduler",
    "scaler",
    "trainer_state",
    "rng_state",
    "run_identity",
}


class CheckpointError(RuntimeError):
    """A local full-state checkpoint is absent, corrupt, or incomplete."""


class CheckpointIdentityError(CheckpointError):
    """A checkpoint belongs to a different scientific run."""


@dataclass(frozen=True)
class CheckpointIdentity:
    array_size: str
    model_size: str
    condition_channels: int
    parameter_count: int
    config_sha256: str
    manifest_sha256: str | None = None
    split_sha256: str | None = None
    schema_sha256: str | None = None
    archive_sha256: str | None = None
    dataset_revision: str | None = None
    radioflow_upstream_base: str | None = None
    git_commit: str | None = None
    seed: int | None = None
    experiment: str | None = None
    variant: str | None = None
    mask_protocol_sha256: str | None = None

    LEGACY_KEYS = (
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
    )
    SPARSE_KEYS = (
        "experiment",
        "array_size",
        "variant",
        "model_size",
        "condition_channels",
        "parameter_count",
        "config_sha256",
        "mask_protocol_sha256",
    )

    def _mode(self) -> str:
        has_sparse = any(
            getattr(self, field) is not None
            for field in ("experiment", "variant", "mask_protocol_sha256")
        )
        has_legacy = any(
            getattr(self, field) is not None
            for field in (
                "manifest_sha256",
                "split_sha256",
                "schema_sha256",
                "archive_sha256",
                "dataset_revision",
                "radioflow_upstream_base",
                "git_commit",
                "seed",
            )
        )
        if has_sparse and has_legacy:
            raise CheckpointIdentityError(
                "run identity schema mismatch: cannot mix legacy and sparse fields"
            )
        if has_sparse:
            return "sparse"
        if has_legacy:
            return "legacy"
        raise CheckpointIdentityError(
            "run identity schema mismatch: identity is missing legacy and sparse fields"
        )

    def to_dict(self) -> dict[str, Any]:
        mode = self._mode()
        keys = self.LEGACY_KEYS if mode == "legacy" else self.SPARSE_KEYS
        return {key: getattr(self, key) for key in keys}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CheckpointIdentity":
        payload_keys = set(payload)
        legacy_keys = set(cls.LEGACY_KEYS)
        sparse_keys = set(cls.SPARSE_KEYS)
        if payload_keys == legacy_keys:
            schema = "legacy"
        elif payload_keys == sparse_keys:
            schema = "sparse"
        else:
            raise CheckpointIdentityError(
                "run identity schema mismatch: "
                f"legacy_missing={sorted(legacy_keys - payload_keys)}, "
                f"legacy_extra={sorted(payload_keys - legacy_keys)}, "
                f"sparse_missing={sorted(sparse_keys - payload_keys)}, "
                f"sparse_extra={sorted(payload_keys - sparse_keys)}"
            )
        try:
            common = dict(
                array_size=str(payload["array_size"]),
                model_size=str(payload["model_size"]),
                condition_channels=int(payload["condition_channels"]),
                parameter_count=int(payload["parameter_count"]),
                config_sha256=str(payload["config_sha256"]),
            )
            if schema == "legacy":
                identity = cls(
                    **common,
                    manifest_sha256=str(payload["manifest_sha256"]),
                    split_sha256=str(payload["split_sha256"]),
                    schema_sha256=str(payload["schema_sha256"]),
                    archive_sha256=str(payload["archive_sha256"]),
                    dataset_revision=str(payload["dataset_revision"]),
                    radioflow_upstream_base=str(payload["radioflow_upstream_base"]),
                    git_commit=str(payload["git_commit"]),
                    seed=int(payload["seed"]),
                )
            else:
                identity = cls(
                    **common,
                    experiment=str(payload["experiment"]),
                    variant=str(payload["variant"]),
                    mask_protocol_sha256=str(payload["mask_protocol_sha256"]),
                )
        except (KeyError, TypeError, ValueError) as error:
            raise CheckpointIdentityError(f"invalid run identity: {error}") from error
        identity.validate()
        return identity

    def validate(self) -> None:
        if not self.array_size or not self.model_size:
            raise CheckpointIdentityError("array_size and model_size must be non-empty")
        if self.parameter_count <= 0:
            raise CheckpointIdentityError("parameter_count must be positive")
        mode = self._mode()
        if len(self.config_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.config_sha256
        ):
            raise CheckpointIdentityError("config_sha256 must be a lowercase SHA-256")
        if mode == "legacy":
            if self.condition_channels != 3:
                raise CheckpointIdentityError("condition_channels must equal 3")
            for field in (
                "manifest_sha256",
                "split_sha256",
                "schema_sha256",
                "archive_sha256",
            ):
                value = getattr(self, field)
                if not isinstance(value, str) or len(value) != 64 or any(
                    character not in "0123456789abcdef" for character in value
                ):
                    raise CheckpointIdentityError(f"{field} must be a lowercase SHA-256")
            for field in ("dataset_revision", "radioflow_upstream_base", "git_commit"):
                value = getattr(self, field)
                if not isinstance(value, str) or len(value) != 40 or any(
                    character not in "0123456789abcdef" for character in value
                ):
                    raise CheckpointIdentityError(f"{field} must be a lowercase Git commit")
            if self.seed != 42:
                raise CheckpointIdentityError("seed must equal 42")
            return
        if not self.experiment or not self.variant:
            raise CheckpointIdentityError(
                "experiment and variant must be non-empty for sparse checkpoints"
            )
        if self.variant not in {"no_beam_masked", "beam_masked"}:
            raise CheckpointIdentityError(
                "variant must be one of {'no_beam_masked', 'beam_masked'}"
            )
        if self.condition_channels not in {4, 5}:
            raise CheckpointIdentityError("condition_channels must equal 4 or 5")
        value = self.mask_protocol_sha256
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise CheckpointIdentityError(
                "mask_protocol_sha256 must be a lowercase SHA-256"
            )


@dataclass(frozen=True)
class TrainerState:
    completed_epochs: int
    next_epoch_index: int
    optimizer_step: int
    micro_batches_seen: int
    samples_seen: int
    best_val_db_rmse: float
    epochs_without_improvement: int
    history: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_epochs": self.completed_epochs,
            "next_epoch_index": self.next_epoch_index,
            "optimizer_step": self.optimizer_step,
            "micro_batches_seen": self.micro_batches_seen,
            "samples_seen": self.samples_seen,
            "best_val_db_rmse": self.best_val_db_rmse,
            "epochs_without_improvement": self.epochs_without_improvement,
            "history": [dict(row) for row in self.history],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainerState":
        expected = {
            "completed_epochs",
            "next_epoch_index",
            "optimizer_step",
            "micro_batches_seen",
            "samples_seen",
            "best_val_db_rmse",
            "epochs_without_improvement",
            "history",
        }
        if set(payload) != expected:
            raise CheckpointError(
                "trainer_state keys mismatch: "
                f"missing={sorted(expected - set(payload))}, "
                f"extra={sorted(set(payload) - expected)}"
            )
        try:
            raw_history = payload["history"]
            if not isinstance(raw_history, (list, tuple)):
                raise TypeError("history must be a list")
            history = tuple(
                dict(row) if isinstance(row, Mapping) else _raise_history_row()
                for row in raw_history
            )
            state = cls(
                completed_epochs=int(payload["completed_epochs"]),
                next_epoch_index=int(payload["next_epoch_index"]),
                optimizer_step=int(payload["optimizer_step"]),
                micro_batches_seen=int(payload["micro_batches_seen"]),
                samples_seen=int(payload["samples_seen"]),
                best_val_db_rmse=float(payload["best_val_db_rmse"]),
                epochs_without_improvement=int(payload["epochs_without_improvement"]),
                history=history,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CheckpointError(f"invalid trainer_state: {error}") from error
        state.validate()
        return state

    def validate(self) -> None:
        counters = {
            "completed_epochs": self.completed_epochs,
            "next_epoch_index": self.next_epoch_index,
            "optimizer_step": self.optimizer_step,
            "micro_batches_seen": self.micro_batches_seen,
            "samples_seen": self.samples_seen,
            "epochs_without_improvement": self.epochs_without_improvement,
        }
        for name, value in counters.items():
            if value < 0:
                raise CheckpointError(f"trainer_state {name} must be non-negative")
        if self.next_epoch_index != self.completed_epochs:
            raise CheckpointError(
                "next_epoch_index must equal completed_epochs at an epoch boundary"
            )
        if len(self.history) != self.completed_epochs:
            raise CheckpointError("history length must equal completed_epochs")
        if not math.isfinite(self.best_val_db_rmse) or self.best_val_db_rmse < 0.0:
            raise CheckpointError("best_val_db_rmse must be finite and non-negative")
        for index, row in enumerate(self.history, start=1):
            if not row:
                raise CheckpointError(f"history row {index} is empty")
            if "epoch" in row and row["epoch"] != index:
                raise CheckpointError("history epoch sequence is not contiguous")


def _raise_history_row():
    raise TypeError("each history row must be an object")


def _ema_model(ema: Any) -> torch.nn.Module:
    candidate = getattr(ema, "ema_model", ema)
    if not isinstance(candidate, torch.nn.Module):
        raise CheckpointError("EMA object must be a module or expose ema_model")
    return candidate


def _capture_rng_state(train_generator: torch.Generator) -> dict[str, Any]:
    if not isinstance(train_generator, torch.Generator):
        raise CheckpointError("train_generator must be a torch.Generator")
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "train_generator": train_generator.get_state(),
    }


def _validate_rng_state(payload: Mapping[str, Any]) -> None:
    expected = {"python", "numpy", "torch_cpu", "torch_cuda", "train_generator"}
    if set(payload) != expected:
        raise CheckpointError("rng_state keys mismatch")
    try:
        python_probe = random.Random()
        python_probe.setstate(payload["python"])
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(payload["numpy"])
    except (TypeError, ValueError) as error:
        raise CheckpointError(f"invalid Python/NumPy RNG state: {error}") from error
    for key in ("torch_cpu", "train_generator"):
        value = payload[key]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise CheckpointError(f"invalid {key} RNG tensor")
    cuda_states = payload["torch_cuda"]
    if not isinstance(cuda_states, list):
        raise CheckpointError("torch_cuda RNG state must be a list")
    expected_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(cuda_states) != expected_cuda:
        raise CheckpointError(
            f"CUDA RNG device count mismatch: expected {expected_cuda}, got {len(cuda_states)}"
        )
    if any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 1
        for value in cuda_states
    ):
        raise CheckpointError("invalid CUDA RNG tensor")


def _restore_rng_state(
    payload: Mapping[str, Any],
    train_generator: torch.Generator,
) -> None:
    _validate_rng_state(payload)
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["torch_cuda"])
    train_generator.set_state(payload["train_generator"])


def _validate_state_dict(
    label: str,
    payload: Any,
    expected: Mapping[str, Tensor],
) -> None:
    if not isinstance(payload, Mapping):
        raise CheckpointError(f"{label} state must be an object")
    if set(payload) != set(expected):
        raise CheckpointError(
            f"{label} state keys mismatch: missing={sorted(set(expected) - set(payload))}, "
            f"extra={sorted(set(payload) - set(expected))}"
        )
    for key, wanted in expected.items():
        value = payload[key]
        if not isinstance(value, Tensor):
            raise CheckpointError(f"{label} state {key!r} is not a tensor")
        if value.shape != wanted.shape or value.dtype != wanted.dtype:
            raise CheckpointError(f"{label} state tensor contract mismatch at {key!r}")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise CheckpointError(f"{label} state contains non-finite tensor {key!r}")


def _validate_optimizer_state(
    payload: Any,
    optimizer: torch.optim.Optimizer,
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != {"state", "param_groups"}:
        raise CheckpointError("optimizer state keys mismatch")
    current = optimizer.state_dict()
    groups = payload["param_groups"]
    if not isinstance(groups, list) or len(groups) != len(current["param_groups"]):
        raise CheckpointError("optimizer parameter-group count mismatch")
    for saved, live in zip(groups, current["param_groups"]):
        if not isinstance(saved, Mapping) or "params" not in saved:
            raise CheckpointError("optimizer parameter group is invalid")
        if len(saved["params"]) != len(live["params"]):
            raise CheckpointError("optimizer parameter count mismatch")
    state = payload["state"]
    if not isinstance(state, Mapping):
        raise CheckpointError("optimizer state payload is invalid")
    valid_parameter_ids = {
        parameter_id for group in groups for parameter_id in group["params"]
    }
    if not set(state).issubset(valid_parameter_ids):
        raise CheckpointError("optimizer state references an unknown parameter")
    for parameter_state in state.values():
        if not isinstance(parameter_state, Mapping):
            raise CheckpointError("optimizer parameter state is invalid")
        for value in parameter_state.values():
            if isinstance(value, Tensor) and value.is_floating_point():
                if not bool(torch.isfinite(value).all()):
                    raise CheckpointError("optimizer state contains a non-finite tensor")


def _load_payload(path: Path) -> Mapping[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise CheckpointError(f"cannot load checkpoint {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint root must be an object")
    if set(payload) != CHECKPOINT_KEYS:
        raise CheckpointError(
            "checkpoint top-level keys mismatch: "
            f"missing={sorted(CHECKPOINT_KEYS - set(payload))}, "
            f"extra={sorted(set(payload) - CHECKPOINT_KEYS)}"
        )
    if payload["schema_version"] != 1:
        raise CheckpointError("checkpoint schema_version must equal 1")
    return payload


def _validate_identity(
    payload: Any,
    expected: CheckpointIdentity,
    model: torch.nn.Module,
) -> CheckpointIdentity:
    if not isinstance(payload, Mapping):
        raise CheckpointIdentityError("run_identity must be an object")
    actual = CheckpointIdentity.from_dict(payload)
    expected.validate()
    actual_mode = actual._mode()
    expected_mode = expected._mode()
    if actual_mode != expected_mode:
        raise CheckpointIdentityError(
            f"run identity schema mismatch: expected {expected_mode}, got {actual_mode}"
        )
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != expected.parameter_count:
        raise CheckpointIdentityError(
            "live model parameter_count mismatch: "
            f"expected {expected.parameter_count}, got {actual_parameters}"
        )
    for field in CheckpointIdentity.__dataclass_fields__:
        actual_value = getattr(actual, field)
        expected_value = getattr(expected, field)
        if actual_value != expected_value:
            raise CheckpointIdentityError(
                f"run identity {field} mismatch: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )
    return actual


def save_checkpoint_atomic(
    path: Path,
    *,
    model: torch.nn.Module,
    ema: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    trainer_state: TrainerState,
    identity: CheckpointIdentity,
    train_generator: torch.Generator,
) -> None:
    path = Path(path)
    identity.validate()
    trainer_state.validate()
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != identity.parameter_count:
        raise CheckpointIdentityError(
            f"parameter_count mismatch: expected {identity.parameter_count}, "
            f"got {actual_parameters}"
        )
    ema_model = _ema_model(ema)
    _validate_state_dict("model", model.state_dict(), model.state_dict())
    _validate_state_dict("EMA", ema_model.state_dict(), model.state_dict())
    rng_state = _capture_rng_state(train_generator)
    payload = {
        "schema_version": 1,
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "trainer_state": trainer_state.to_dict(),
        "rng_state": rng_state,
        "run_identity": identity.to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception as error:
        raise CheckpointError(f"cannot save checkpoint {path}: {error}") from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _prevalidate_full_checkpoint(
    payload: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    ema: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    expected_identity: CheckpointIdentity,
) -> TrainerState:
    _validate_identity(payload["run_identity"], expected_identity, model)
    trainer_state_payload = payload["trainer_state"]
    if not isinstance(trainer_state_payload, Mapping):
        raise CheckpointError("trainer_state must be an object")
    trainer_state = TrainerState.from_dict(trainer_state_payload)
    _validate_state_dict("model", payload["model"], model.state_dict())
    _validate_state_dict("EMA", payload["ema"], _ema_model(ema).state_dict())
    _validate_optimizer_state(payload["optimizer"], optimizer)
    if not isinstance(payload["scheduler"], Mapping):
        raise CheckpointError("scheduler state must be an object")
    if not isinstance(payload["scaler"], Mapping):
        raise CheckpointError("scaler state must be an object")
    rng = payload["rng_state"]
    if not isinstance(rng, Mapping):
        raise CheckpointError("rng_state must be an object")
    _validate_rng_state(rng)
    return trainer_state


def load_checkpoint_strict(
    path: Path,
    *,
    model: torch.nn.Module,
    ema: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    expected_identity: CheckpointIdentity,
    train_generator: torch.Generator,
) -> TrainerState:
    payload = _load_payload(path)
    trainer_state = _prevalidate_full_checkpoint(
        payload,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_identity=expected_identity,
    )
    ema_model = _ema_model(ema)
    backups = {
        "model": copy.deepcopy(model.state_dict()),
        "ema": copy.deepcopy(ema_model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "scaler": copy.deepcopy(scaler.state_dict()),
        "rng": _capture_rng_state(train_generator),
    }
    try:
        model.load_state_dict(payload["model"], strict=True)
        ema_model.load_state_dict(payload["ema"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        _restore_rng_state(payload["rng_state"], train_generator)
    except Exception as error:
        model.load_state_dict(backups["model"], strict=True)
        ema_model.load_state_dict(backups["ema"], strict=True)
        optimizer.load_state_dict(backups["optimizer"])
        scheduler.load_state_dict(backups["scheduler"])
        scaler.load_state_dict(backups["scaler"])
        _restore_rng_state(backups["rng"], train_generator)
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError(f"cannot restore checkpoint state: {error}") from error
    return trainer_state


def load_ema_for_evaluation(
    path: Path,
    *,
    model: torch.nn.Module,
    expected_identity: CheckpointIdentity,
) -> TrainerState:
    payload = _load_payload(path)
    _validate_identity(payload["run_identity"], expected_identity, model)
    trainer_payload = payload["trainer_state"]
    if not isinstance(trainer_payload, Mapping):
        raise CheckpointError("trainer_state must be an object")
    trainer_state = TrainerState.from_dict(trainer_payload)
    _validate_state_dict("EMA", payload["ema"], model.state_dict())
    backup = copy.deepcopy(model.state_dict())
    try:
        model.load_state_dict(payload["ema"], strict=True)
    except Exception as error:
        model.load_state_dict(backup, strict=True)
        raise CheckpointError(f"cannot restore EMA checkpoint state: {error}") from error
    return trainer_state


def rebuild_metrics_csv(
    path: Path,
    history: Sequence[Mapping[str, Any]],
) -> None:
    rows = tuple(dict(row) for row in history)
    if not rows:
        raise CheckpointError("cannot rebuild metrics.csv from empty history")
    columns = tuple(rows[0])
    if not columns or any(tuple(row) != columns for row in rows):
        raise CheckpointError("history rows do not share one fixed column order")
    if "epoch" in columns:
        epochs = [row["epoch"] for row in rows]
        if epochs != list(range(1, len(rows) + 1)):
            raise CheckpointError("history contains duplicate or non-contiguous epochs")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception as error:
        raise CheckpointError(f"cannot rebuild metrics CSV {path}: {error}") from error
    finally:
        if temporary.exists():
            temporary.unlink()
