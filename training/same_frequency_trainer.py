from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from data_loaders.multiconfig import multiconfig_collate
from data_loaders.same_frequency import (
    CONDITION_VARIANTS,
    SameFrequencyRadiomapDataset,
    load_same_frequency_height_max,
)
from experiments.cross_frequency import (
    TEST_FREQUENCY_HZ,
    select_zero_degree_configurations_for_array_sizes,
)
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
from training.model_factory import build_locked_radioflow
from training.multiconfig_trainer import (
    REPO_ROOT,
    MultiConfigSRMTrainer,
    _validate_smoke_checkpoint_fresh,
    resolve_device,
    seed_everything,
)
from training.same_frequency_config import (
    SameFrequencyTrainConfig,
    SameFrequencyTrainConfigError,
)


SCHEMA_PATH = REPO_ROOT / "experiments" / "multiconfig_schema.json"


class SameFrequencyTrainerContractError(RuntimeError):
    """Same-frequency preflight or orchestration state is invalid."""


@dataclass(frozen=True)
class SameFrequencyContext:
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
    beam_id: int
    config_id: str
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
        raise SameFrequencyTrainerContractError(
            f"{label} must be lowercase hexadecimal with {length} characters"
        )


def infer_manifest_selection(manifest_path: Path, array_size: str) -> tuple[int, str]:
    try:
        records = load_manifest_jsonl(Path(manifest_path).resolve())
    except Exception as error:
        raise SameFrequencyTrainerContractError(
            f"cannot load same-frequency manifest {manifest_path}: {error}"
        ) from error
    selected = tuple(record for record in records if record.array_name == array_size)
    if not selected:
        raise SameFrequencyTrainerContractError(
            f"manifest has no records for array size {array_size}"
        )
    beam_ids = {record.beam_id for record in selected}
    config_ids = {record.config_id for record in selected}
    frequencies = {record.frequency_hz for record in selected}
    angles = {record.steering_deg for record in selected}
    if len(beam_ids) != 1 or len(config_ids) != 1:
        raise SameFrequencyTrainerContractError(
            "manifest must contain one beam ID and one configuration per array size"
        )
    if frequencies != {TEST_FREQUENCY_HZ} or angles != {0.0}:
        raise SameFrequencyTrainerContractError(
            "same-frequency manifest must contain only 6.7 GHz and 0 degree records"
        )
    try:
        schema = load_schema_lock(SCHEMA_PATH)
        expected = select_zero_degree_configurations_for_array_sizes(
            schema,
            frequency_hz=TEST_FREQUENCY_HZ,
            array_sizes=(array_size,),
        )[array_size]
    except Exception as error:
        raise SameFrequencyTrainerContractError(
            f"cannot resolve schema-selected zero-degree configuration: {error}"
        ) from error
    beam_id = next(iter(beam_ids))
    config_id = next(iter(config_ids))
    if beam_id != expected.beam_id or config_id != expected.config_id:
        raise SameFrequencyTrainerContractError(
            "manifest selection does not match the schema-selected zero-degree "
            f"configuration: expected {expected.config_id}/beam{expected.beam_id:02d}, "
            f"got {config_id}/beam{beam_id:02d}"
        )
    return beam_id, config_id


def preflight_same_frequency(cfg: SameFrequencyTrainConfig) -> SameFrequencyContext:
    try:
        checkout = assert_radioflow_checkout(REPO_ROOT)
    except Exception as error:
        raise SameFrequencyTrainerContractError(
            f"RadioFlow checkout provenance failed: {error}"
        ) from error
    dataset_root = cfg.dataset_root.resolve()
    manifest_path = cfg.manifest_path.resolve()
    stats_path = cfg.height_stats_path.resolve()
    condition_variant = getattr(cfg, "condition_variant", "full")
    if condition_variant not in CONDITION_VARIANTS:
        raise SameFrequencyTrainerContractError(
            f"unsupported condition variant: {condition_variant!r}"
        )
    if not manifest_path.is_file():
        raise SameFrequencyTrainerContractError(
            f"same-frequency manifest is missing: {manifest_path}"
        )
    if not stats_path.is_file():
        raise SameFrequencyTrainerContractError(
            f"height statistics are missing: {stats_path}"
        )
    split_path = manifest_path.parent / "scene_split_seed42.json"
    if not split_path.is_file():
        raise SameFrequencyTrainerContractError(
            f"fixed scene split is missing: {split_path}"
        )
    try:
        schema = load_schema_lock(SCHEMA_PATH)
        height_max = load_same_frequency_height_max(stats_path, split_path=split_path)
        all_records = load_manifest_jsonl(manifest_path)
    except Exception as error:
        raise SameFrequencyTrainerContractError(
            f"same-frequency data contract could not be loaded: {error}"
        ) from error
    expected_total = cfg.train_samples + cfg.val_samples + cfg.test_samples
    if len(all_records) != expected_total:
        raise SameFrequencyTrainerContractError(
            f"manifest must contain {expected_total} records, got {len(all_records)}"
        )
    try:
        beam_id, config_id = infer_manifest_selection(manifest_path, cfg.array_size)
    except SameFrequencyTrainerContractError:
        raise
    if beam_id != cfg.beam_id:
        raise SameFrequencyTrainerContractError(
            f"config beam ID {cfg.beam_id} does not match manifest beam ID {beam_id}"
        )
    source_metadata = schema.raw.get("source_metadata")
    if not isinstance(source_metadata, Mapping):
        raise SameFrequencyTrainerContractError("schema source_metadata must be an object")
    datasets: dict[str, Any] = {}
    counts = {
        "train": cfg.train_samples,
        "val": cfg.val_samples,
        "test": cfg.test_samples,
    }
    for split in ("train", "val", "test"):
        try:
            datasets[split] = SameFrequencyRadiomapDataset(
                dataset_root=dataset_root,
                manifest_path=manifest_path,
                split=split,
                array_size=cfg.array_size,
                height_max=height_max,
                expected_frequency_hz=TEST_FREQUENCY_HZ,
                expected_beam_id=cfg.beam_id,
                expected_counts=counts,
                source_metadata=source_metadata,
                condition_variant=condition_variant,
            )
        except Exception as error:
            raise SameFrequencyTrainerContractError(
                f"cannot construct {split} same-frequency dataset: {error}"
            ) from error
        if len(datasets[split]) != counts[split]:
            raise SameFrequencyTrainerContractError(
                f"{split} dataset count does not match the locked protocol"
            )
        sample = datasets[split][0]
        if tuple(sample["condition"].shape) != (3, 256, 256):
            raise SameFrequencyTrainerContractError(
                f"{split} condition shape must be (3,256,256)"
            )
        beam_nonzero = bool(torch.count_nonzero(sample["condition"][2]))
        if condition_variant == "beam_zero" and beam_nonzero:
            raise SameFrequencyTrainerContractError(
                f"{split} beam-zero condition contains a nonzero beam pixel"
            )
        if tuple(sample["target"].shape) != (1, 256, 256):
            raise SameFrequencyTrainerContractError(
                f"{split} target shape must be (1,256,256)"
            )
        if not bool(sample["valid_mask"].any()):
            raise SameFrequencyTrainerContractError(
                f"{split} sample has no valid pixels"
            )
    archive_sha256 = schema.identities.get("archive_sha256")
    dataset_revision = schema.identities.get("dataset_revision")
    if not isinstance(archive_sha256, str) or not isinstance(dataset_revision, str):
        raise SameFrequencyTrainerContractError(
            "schema lacks archive or dataset identity"
        )
    _validate_identity_hex(archive_sha256, 64, "archive_sha256")
    _validate_identity_hex(dataset_revision, 40, "dataset_revision")
    if dataset_revision != DATASET_REVISION:
        raise SameFrequencyTrainerContractError(
            "dataset revision differs from pinned source"
        )
    return SameFrequencyContext(
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
        beam_id=beam_id,
        config_id=config_id,
        schema=schema,
    )


def build_same_frequency_loaders(
    cfg: SameFrequencyTrainConfig,
    context: SameFrequencyContext,
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
        raise SameFrequencyTrainerContractError(
            "same-frequency train DataLoader micro-batch count mismatch"
        )
    if math.ceil(len(train_loader) / cfg.accumulation_steps) != cfg.optimizer_steps_per_epoch:
        raise SameFrequencyTrainerContractError(
            "same-frequency optimizer-step count mismatch"
        )
    if len(val_loader) != cfg.val_samples:
        raise SameFrequencyTrainerContractError(
            "same-frequency validation DataLoader count mismatch"
        )
    return train_loader, val_loader, generator


def build_same_frequency_checkpoint_identity(
    cfg: SameFrequencyTrainConfig,
    context: SameFrequencyContext,
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


def write_or_validate_same_frequency_run_config(
    cfg: SameFrequencyTrainConfig,
    controls: InvocationControls,
) -> Path:
    path = cfg.run_dir / "config.json"
    if path.exists():
        try:
            existing = SameFrequencyTrainConfig.from_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise SameFrequencyTrainConfigError(
                f"invalid existing same-frequency run config {path}: {error}"
            ) from error
        if existing.config_sha256 != cfg.config_sha256:
            raise SameFrequencyTrainConfigError(
                "existing same-frequency run config scientific hash mismatch"
            )
        return path
    _atomic_json(path, cfg.to_record(controls))
    return path


def _resolve_resume_path(cfg: SameFrequencyTrainConfig, resume: str) -> Path | None:
    if resume == "none":
        return None
    if resume == "auto":
        candidate = cfg.run_dir / "last.pt"
        return candidate if candidate.is_file() else None
    return Path(resume).resolve()


def run_same_frequency_training(
    cfg: SameFrequencyTrainConfig,
    controls: InvocationControls,
    device: torch.device,
    *,
    preflight_only: bool = False,
) -> dict[str, Any]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise SameFrequencyTrainerContractError(
            "CUBLAS_WORKSPACE_CONFIG must be ':4096:8' before training"
        )
    context = preflight_same_frequency(cfg)
    if preflight_only:
        return {
            "status": "preflight_complete",
            "manifest": str(context.manifest_path),
            "manifest_sha256": context.manifest_sha256,
            "array_size": cfg.array_size,
            "beam_id": context.beam_id,
            "config_id": context.config_id,
            "train_samples": len(context.train_dataset),
            "val_samples": len(context.val_dataset),
            "test_samples": len(context.test_dataset),
            "height_max": context.height_max,
        }
    smoke = controls.smoke_optimizer_steps is not None
    effective_cfg = replace(cfg, run_root=cfg.run_root / "_smoke") if smoke else cfg
    if not smoke and controls.resume == "none" and (cfg.run_dir / "last.pt").exists():
        raise SameFrequencyTrainerContractError(
            "resume=none refuses an existing same-frequency last.pt"
        )
    write_or_validate_same_frequency_run_config(effective_cfg, controls)
    seed_everything(cfg.seed)
    model = build_locked_radioflow(cfg.model_size).to(device)
    train_loader, val_loader, train_generator = build_same_frequency_loaders(
        cfg,
        context,
    )
    identity = build_same_frequency_checkpoint_identity(cfg, context, model)
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
    "SameFrequencyContext",
    "SameFrequencyTrainerContractError",
    "build_same_frequency_checkpoint_identity",
    "build_same_frequency_loaders",
    "infer_manifest_selection",
    "preflight_same_frequency",
    "run_same_frequency_training",
    "write_or_validate_same_frequency_run_config",
]
