from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from data_loaders.cross_frequency import (
    CrossFrequencyRadiomapDataset,
    load_cross_frequency_height_max,
)
from data_loaders.multiconfig import multiconfig_collate
from experiments.multiconfig_manifest import (
    DatasetSchemaLock,
    canonical_json_bytes,
    load_manifest_jsonl,
    load_schema_lock,
)
from experiments.provenance import (
    DATASET_REVISION,
    RADIOFLOW_UPSTREAM_BASE,
    assert_radioflow_checkout,
    sha256_file,
)
from training.checkpointing import CheckpointIdentity
from training.config import InvocationControls
from training.cross_frequency_config import (
    CrossFrequencyTrainConfig,
    CrossFrequencyTrainConfigError,
)
from training.model_factory import build_locked_radioflow
from training.multiconfig_trainer import (
    REPO_ROOT,
    MultiConfigSRMTrainer,
    TrainerContractError,
    _validate_smoke_checkpoint_fresh,
    resolve_device,
    seed_everything,
)


SCHEMA_PATH = REPO_ROOT / "experiments" / "multiconfig_schema.json"


class CrossFrequencyTrainerContractError(RuntimeError):
    """Cross-frequency preflight or orchestration state is invalid."""


@dataclass(frozen=True)
class CrossFrequencyContext:
    train_dataset: Any
    val_dataset: Any
    test_dataset: Any
    manifest_path: Path
    split_path: Path
    schema_path: Path
    manifest_sha256: str
    split_sha256: str
    schema_sha256: str
    archive_sha256: str
    dataset_revision: str
    git_commit: str
    height_max: float
    schema: DatasetSchemaLock | Any | None = None


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    expected = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as output:
            output.write(expected)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_identity_hex(value: str, length: int, label: str) -> None:
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise CrossFrequencyTrainerContractError(
            f"{label} must be lowercase hexadecimal with {length} characters"
        )


def preflight_cross_frequency(
    cfg: CrossFrequencyTrainConfig,
) -> CrossFrequencyContext:
    try:
        checkout = assert_radioflow_checkout(REPO_ROOT)
    except Exception as error:
        raise CrossFrequencyTrainerContractError(
            f"RadioFlow checkout provenance failed: {error}"
        ) from error
    dataset_root = cfg.dataset_root.resolve()
    manifest_path = cfg.manifest_path.resolve()
    stats_path = cfg.height_stats_path.resolve()
    if not manifest_path.is_file():
        raise CrossFrequencyTrainerContractError(
            f"cross-frequency manifest is missing: {manifest_path}"
        )
    if not stats_path.is_file():
        raise CrossFrequencyTrainerContractError(
            f"height statistics are missing: {stats_path}"
        )
    split_path = manifest_path.parent / "scene_split_seed42.json"
    if not split_path.is_file():
        raise CrossFrequencyTrainerContractError(
            f"fixed scene split is missing: {split_path}"
        )
    try:
        schema = load_schema_lock(SCHEMA_PATH)
    except Exception as error:
        raise CrossFrequencyTrainerContractError(
            f"cannot load schema lock {SCHEMA_PATH}: {error}"
        ) from error
    try:
        height_max = load_cross_frequency_height_max(
            stats_path,
            split_path=split_path,
        )
        all_records = load_manifest_jsonl(manifest_path)
    except Exception as error:
        raise CrossFrequencyTrainerContractError(
            f"cross-frequency data contract could not be loaded: {error}"
        ) from error
    if len(all_records) != cfg.train_samples + cfg.val_samples + cfg.test_samples:
        raise CrossFrequencyTrainerContractError(
            f"manifest must contain {cfg.train_samples + cfg.val_samples + cfg.test_samples} records"
        )
    source_metadata = schema.raw.get("source_metadata")
    if not isinstance(source_metadata, Mapping):
        raise CrossFrequencyTrainerContractError("schema source_metadata must be an object")
    datasets: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        try:
            datasets[split] = CrossFrequencyRadiomapDataset(
                dataset_root=dataset_root,
                manifest_path=manifest_path,
                split=split,
                height_max=height_max,
                expected_frequency_hz=(
                    cfg.test_frequency_hz
                    if split == "test"
                    else cfg.train_frequency_hz
                ),
                expected_counts={
                    "train": cfg.train_samples,
                    "val": cfg.val_samples,
                    "test": cfg.test_samples,
                },
                source_metadata=source_metadata,
            )
        except Exception as error:
            raise CrossFrequencyTrainerContractError(
                f"cannot construct {split} cross-frequency dataset: {error}"
            ) from error
        if len(datasets[split]) != {
            "train": cfg.train_samples,
            "val": cfg.val_samples,
            "test": cfg.test_samples,
        }[split]:
            raise CrossFrequencyTrainerContractError(
                f"{split} dataset count does not match the locked protocol"
            )
        sample = datasets[split][0]
        if tuple(sample["condition"].shape) != (3, 256, 256):
            raise CrossFrequencyTrainerContractError(
                f"{split} condition shape must be (3,256,256)"
            )
        if tuple(sample["target"].shape) != (1, 256, 256):
            raise CrossFrequencyTrainerContractError(
                f"{split} target shape must be (1,256,256)"
            )
        if not bool(sample["valid_mask"].any()):
            raise CrossFrequencyTrainerContractError(
                f"{split} sample has no valid pixels"
            )
    archive_sha256 = schema.identities.get("archive_sha256")
    dataset_revision = schema.identities.get("dataset_revision")
    if not isinstance(archive_sha256, str) or not isinstance(dataset_revision, str):
        raise CrossFrequencyTrainerContractError(
            "schema lacks archive or dataset identity"
        )
    _validate_identity_hex(archive_sha256, 64, "archive_sha256")
    _validate_identity_hex(dataset_revision, 40, "dataset_revision")
    if dataset_revision != DATASET_REVISION:
        raise CrossFrequencyTrainerContractError(
            "dataset revision differs from pinned source"
        )
    return CrossFrequencyContext(
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        test_dataset=datasets["test"],
        manifest_path=manifest_path,
        split_path=split_path,
        schema_path=SCHEMA_PATH,
        manifest_sha256=sha256_file(manifest_path),
        split_sha256=sha256_file(split_path),
        schema_sha256=sha256_file(SCHEMA_PATH),
        archive_sha256=archive_sha256,
        dataset_revision=dataset_revision,
        git_commit=checkout.head_commit,
        height_max=height_max,
        schema=schema,
    )


def build_cross_frequency_loaders(
    cfg: CrossFrequencyTrainConfig,
    context: CrossFrequencyContext,
) -> tuple[DataLoader, DataLoader, torch.Generator]:
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    train_loader = DataLoader(
        context.train_dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=cfg.num_workers,
        collate_fn=multiconfig_collate,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        context.val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=multiconfig_collate,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )
    expected_micro_batches = math.ceil(cfg.train_samples / cfg.micro_batch_size)
    if len(train_loader) != expected_micro_batches:
        raise CrossFrequencyTrainerContractError(
            "cross-frequency train DataLoader micro-batch count mismatch"
        )
    if math.ceil(len(train_loader) / cfg.accumulation_steps) != cfg.optimizer_steps_per_epoch:
        raise CrossFrequencyTrainerContractError(
            "cross-frequency optimizer-step count mismatch"
        )
    if len(val_loader) != cfg.val_samples:
        raise CrossFrequencyTrainerContractError(
            "cross-frequency validation DataLoader count mismatch"
        )
    return train_loader, val_loader, generator


def build_cross_frequency_checkpoint_identity(
    cfg: CrossFrequencyTrainConfig,
    context: CrossFrequencyContext,
    model: torch.nn.Module,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        array_size=cfg.array_size,
        model_size=cfg.model_size,
        condition_channels=cfg.condition_channels,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        manifest_sha256=context.manifest_sha256,
        split_sha256=context.split_sha256,
        schema_sha256=context.schema_sha256,
        config_sha256=cfg.config_sha256,
        archive_sha256=context.archive_sha256,
        dataset_revision=context.dataset_revision,
        radioflow_upstream_base=RADIOFLOW_UPSTREAM_BASE,
        git_commit=context.git_commit,
        seed=cfg.seed,
    )


def write_or_validate_cross_frequency_run_config(
    cfg: CrossFrequencyTrainConfig,
    controls: InvocationControls,
) -> Path:
    path = cfg.run_dir / "config.json"
    if path.exists():
        try:
            existing = CrossFrequencyTrainConfig.from_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise CrossFrequencyTrainConfigError(
                f"invalid existing cross-frequency run config {path}: {error}"
            ) from error
        if existing.config_sha256 != cfg.config_sha256:
            raise CrossFrequencyTrainConfigError(
                "existing cross-frequency run config scientific hash mismatch"
            )
        return path
    _atomic_json(path, cfg.to_record(controls))
    return path


def _resolve_resume_path(
    cfg: CrossFrequencyTrainConfig,
    resume: str,
) -> Path | None:
    if resume == "none":
        return None
    if resume == "auto":
        candidate = cfg.run_dir / "last.pt"
        return candidate if candidate.is_file() else None
    return Path(resume).resolve()


def run_cross_frequency_training(
    cfg: CrossFrequencyTrainConfig,
    controls: InvocationControls,
    device: torch.device,
    *,
    preflight_only: bool = False,
) -> dict[str, Any]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise CrossFrequencyTrainerContractError(
            "CUBLAS_WORKSPACE_CONFIG must be ':4096:8' before training"
        )
    context = preflight_cross_frequency(cfg)
    if preflight_only:
        return {
            "status": "preflight_complete",
            "manifest": str(context.manifest_path),
            "manifest_sha256": context.manifest_sha256,
            "train_samples": len(context.train_dataset),
            "val_samples": len(context.val_dataset),
            "test_samples": len(context.test_dataset),
            "height_max": context.height_max,
        }
    smoke = controls.smoke_optimizer_steps is not None
    effective_cfg = replace(cfg, run_root=cfg.run_root / "_smoke") if smoke else cfg
    if not smoke and controls.resume == "none" and (cfg.run_dir / "last.pt").exists():
        raise CrossFrequencyTrainerContractError(
            "resume=none refuses an existing cross-frequency last.pt"
        )
    write_or_validate_cross_frequency_run_config(effective_cfg, controls)
    seed_everything(cfg.seed)
    model = build_locked_radioflow(cfg.model_size).to(device)
    train_loader, val_loader, train_generator = build_cross_frequency_loaders(
        cfg,
        context,
    )
    identity = build_cross_frequency_checkpoint_identity(cfg, context, model)
    trainer = MultiConfigSRMTrainer(
        effective_cfg,
        model,
        train_loader,
        val_loader,
        device,
        train_generator,
        identity,
    )
    if smoke:
        result = trainer.run_smoke(int(controls.smoke_optimizer_steps))
        _validate_smoke_checkpoint_fresh(
            effective_cfg,
            identity,
            Path(result["checkpoint"]),
            scaler_enabled=device.type == "cuda" and cfg.use_amp,
        )
        return result
    resume_path = _resolve_resume_path(effective_cfg, controls.resume)
    if resume_path is not None:
        trainer.resume(resume_path)
    return trainer.fit(stop_after_epoch=controls.stop_after_epoch)


__all__ = [
    "CrossFrequencyContext",
    "CrossFrequencyTrainerContractError",
    "build_cross_frequency_checkpoint_identity",
    "build_cross_frequency_loaders",
    "preflight_cross_frequency",
    "run_cross_frequency_training",
    "write_or_validate_cross_frequency_run_config",
]
