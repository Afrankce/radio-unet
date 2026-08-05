from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from data_loaders.multiconfig import (
    HEIGHT_STATS_NAME,
    MultiConfigRadiomapDataset,
    multiconfig_collate,
    load_height_stats,
)
from evaluation.radioflow_sampling import euler_cfg_sample, make_sample_noise
from evaluation.radiomap_metrics import MetricAccumulator
from experiments.multiconfig_manifest import (
    DatasetSchemaLock,
    canonical_json_bytes,
    load_schema_lock,
)
from experiments.provenance import (
    DATASET_REVISION,
    RADIOFLOW_UPSTREAM_BASE,
    assert_radioflow_checkout,
    sha256_file,
)
from train import ModelEMA
from training.checkpointing import (
    CheckpointIdentity,
    TrainerState,
    load_checkpoint_strict,
    rebuild_metrics_csv,
    save_checkpoint_atomic,
)
from training.config import (
    ARRAY_SIZES,
    InvocationControls,
    MultiConfigTrainConfig,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
    TEST_SAMPLES,
)
from training.hardware_evidence import (
    LargeHardwareGateContext,
    collect_hardware_snapshot,
    execute_with_large_oom_gate,
)
from training.model_factory import build_locked_radioflow
from training.optimization import (
    build_optimizer_step_scheduler,
    run_accumulation_window,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "experiments" / "multiconfig_schema.json"


class TrainerContractError(RuntimeError):
    """The production training state violates a locked benchmark contract."""


@dataclass(frozen=True)
class BenchmarkContext:
    schema: DatasetSchemaLock
    height_stats: Any
    train_dataset: MultiConfigRadiomapDataset
    val_dataset: MultiConfigRadiomapDataset
    test_dataset: MultiConfigRadiomapDataset
    manifest_path: Path
    split_path: Path
    schema_path: Path
    manifest_sha256: str
    split_sha256: str
    schema_sha256: str
    archive_sha256: str
    dataset_revision: str
    git_commit: str


def seed_everything(seed: int = 42) -> None:
    if seed != 42:
        raise TrainerContractError(f"benchmark seed is locked to 42, got {seed}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise TrainerContractError(f"invalid device {value!r}: {error}") from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise TrainerContractError("CUDA device requested but CUDA is unavailable")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if not 0 <= index < torch.cuda.device_count():
            raise TrainerContractError(f"CUDA device index is unavailable: {index}")
        return torch.device("cuda", index)
    if device.type != "cpu":
        raise TrainerContractError("benchmark device must be CPU or CUDA")
    return device


def _atomic_json(path: Path, payload: Mapping[str, Any], *, replace_existing: bool) -> None:
    path = Path(path)
    expected = canonical_json_bytes(payload)
    if path.exists() and not replace_existing:
        if path.read_bytes() != expected:
            raise TrainerContractError(f"immutable JSON differs: {path}")
        return
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


def write_or_validate_run_config(
    cfg: MultiConfigTrainConfig,
    controls: InvocationControls,
) -> Path:
    path = cfg.run_dir / "config.json"
    record = cfg.to_record(controls)
    if path.exists():
        try:
            existing = MultiConfigTrainConfig.from_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise TrainerContractError(f"invalid existing run config {path}: {error}") from error
        if existing.config_sha256 != cfg.config_sha256:
            raise TrainerContractError("existing run config scientific hash mismatch")
        if existing.run_dir.resolve() != cfg.run_dir.resolve():
            raise TrainerContractError("existing run config path mismatch")
        return path
    _atomic_json(path, record, replace_existing=False)
    return path


def preflight_benchmark(cfg: MultiConfigTrainConfig) -> BenchmarkContext:
    checkout = assert_radioflow_checkout(REPO_ROOT)
    dataset_root = cfg.dataset_root.resolve()
    manifest_dir = cfg.manifest_dir.resolve()
    if manifest_dir != dataset_root / "manifests":
        raise TrainerContractError(
            f"manifest_dir must equal {dataset_root / 'manifests'}, got {manifest_dir}"
        )
    schema = load_schema_lock(SCHEMA_PATH)
    stats = load_height_stats(manifest_dir / HEIGHT_STATS_NAME)
    manifest_path = manifest_dir / f"manifest_{cfg.array_size}.jsonl"
    datasets = {
        split: MultiConfigRadiomapDataset(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            split=split,
            schema=schema,
            height_stats=stats,
            train_scale=cfg.train_scale,
        )
        for split in ("train", "val", "test")
    }
    expected = {"train": cfg.train_samples, "val": VAL_SAMPLES, "test": TEST_SAMPLES}
    for split, dataset in datasets.items():
        if len(dataset) != expected[split]:
            raise TrainerContractError(
                f"{split} sample count mismatch: expected {expected[split]}, got {len(dataset)}"
            )
        sample = dataset[0]
        if sample["condition"].shape != (3, 256, 256):
            raise TrainerContractError(f"{split} decoded condition shape mismatch")
        if sample["target"].shape != (1, 256, 256):
            raise TrainerContractError(f"{split} decoded target shape mismatch")
        if not bool(sample["valid_mask"].any()):
            raise TrainerContractError(f"{split} decoded sample has no valid pixels")
    split_path = manifest_dir / "scene_split_seed42.json"
    archive_sha256 = schema.identities.get("archive_sha256")
    dataset_revision = schema.identities.get("dataset_revision")
    if not isinstance(archive_sha256, str) or not isinstance(dataset_revision, str):
        raise TrainerContractError("schema lacks archive or dataset identity")
    if dataset_revision != DATASET_REVISION:
        raise TrainerContractError("dataset revision differs from pinned source")
    return BenchmarkContext(
        schema=schema,
        height_stats=stats,
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
    )


def build_loaders(
    cfg: MultiConfigTrainConfig,
    context: BenchmarkContext,
) -> tuple[DataLoader, DataLoader, torch.Generator]:
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    train_loader = DataLoader(
        context.train_dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=cfg.num_workers,
        collate_fn=multiconfig_collate,
        worker_init_fn=seed_worker,
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
        worker_init_fn=seed_worker,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )
    expected_micro_batches = math.ceil(cfg.train_samples / cfg.micro_batch_size)
    if len(train_loader) != expected_micro_batches:
        raise TrainerContractError("train DataLoader micro-batch count mismatch")
    if math.ceil(len(train_loader) / cfg.accumulation_steps) != cfg.optimizer_steps_per_epoch:
        raise TrainerContractError("train DataLoader optimizer-step count mismatch")
    if len(val_loader) != VAL_SAMPLES:
        raise TrainerContractError("validation DataLoader count mismatch")
    return train_loader, val_loader, generator


def build_checkpoint_identity(
    cfg: MultiConfigTrainConfig,
    context: BenchmarkContext,
    model: torch.nn.Module,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        array_size=cfg.array_size,
        model_size=cfg.model_size,
        condition_channels=3,
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


class MultiConfigSRMTrainer:
    def __init__(
        self,
        cfg: MultiConfigTrainConfig,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        train_generator: torch.Generator,
        identity: CheckpointIdentity,
    ) -> None:
        parameters = tuple(model.parameters())
        if not parameters:
            raise TrainerContractError("model has no trainable parameters")
        wrong_devices = {
            str(parameter.device)
            for parameter in parameters
            if parameter.device != device
        }
        if wrong_devices:
            raise TrainerContractError(
                f"model parameters must already be on {device}, got {sorted(wrong_devices)}"
            )
        identity.validate()
        actual_parameters = sum(parameter.numel() for parameter in parameters)
        if actual_parameters != identity.parameter_count:
            raise TrainerContractError("checkpoint identity parameter count mismatch")
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.train_generator = train_generator
        self.identity = identity
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.ema = ModelEMA(model, decay=cfg.ema_decay)
        self.scheduler = build_optimizer_step_scheduler(
            self.optimizer,
            total_steps=cfg.planned_optimizer_steps,
            warmup_steps=cfg.warmup_steps,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=device.type == "cuda" and cfg.use_amp,
        )
        self.completed_epochs = 0
        self.next_epoch_index = 0
        self.optimizer_step = 0
        self.micro_batches_seen = 0
        self.samples_seen = 0
        self.best_val_db_rmse = math.inf
        self.epochs_without_improvement = 0
        self.history: list[dict[str, Any]] = []
        self.started_at = time.time()
        cfg.run_dir.mkdir(parents=True, exist_ok=True)

    def _prediction_triplet(self, model: torch.nn.Module, batch: Mapping[str, Any]):
        condition = batch["condition"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(self.device, non_blocking=True)
        enabled = self.device.type == "cuda" and self.cfg.use_amp
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=enabled,
        ):
            x0 = torch.randn_like(target)
            time_step, interpolated, target_velocity = (
                self.flow_matcher.sample_location_and_conditional_flow(x0, target)
            )
            predicted_velocity = model(
                image=condition,
                x=interpolated,
                pred_type="denoise",
                step=time_step,
            )
        return predicted_velocity, target_velocity, valid_mask

    def _run_window(self, window: tuple[Mapping[str, Any], ...]):
        result = run_accumulation_window(
            window,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            scheduler=self.scheduler,
            ema=self.ema,
            predict=self._prediction_triplet,
        )
        self.micro_batches_seen += result.micro_batches
        self.samples_seen += result.samples
        if result.optimizer_ran:
            self.optimizer_step += 1
        return result

    def train_one_epoch(self) -> dict[str, int | float]:
        self.model.train()
        numerator = 0.0
        valid_pixels = 0
        micro_batches = 0
        samples = 0
        optimizer_steps = 0
        iterator = iter(self.train_loader)
        while True:
            window = tuple(itertools.islice(iterator, self.cfg.accumulation_steps))
            if not window:
                break
            result = self._run_window(window)
            numerator += result.squared_error_sum
            valid_pixels += result.valid_pixels
            micro_batches += result.micro_batches
            samples += result.samples
            optimizer_steps += int(result.optimizer_ran)
        if valid_pixels <= 0 or samples <= 0:
            raise TrainerContractError("training epoch produced no valid samples")
        return {
            "loss": numerator / valid_pixels,
            "valid_pixels": valid_pixels,
            "micro_batches": micro_batches,
            "samples": samples,
            "optimizer_steps": optimizer_steps,
        }

    def validate(self) -> dict[str, int | float]:
        network = self.ema.ema_model
        network.eval()
        accumulator = MetricAccumulator()
        for batch in self.val_loader:
            condition = batch["condition"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(self.device, non_blocking=True)
            noises = [
                make_sample_noise(
                    str(metadata["scene_id"]),
                    float(metadata["steering_deg"]),
                    shape=tuple(target.shape[1:]),
                    base_seed=self.cfg.seed,
                )
                for metadata in batch["metadata"]
            ]
            noise = torch.stack(noises).to(self.device)
            prediction = euler_cfg_sample(
                network,
                condition,
                noise,
                cfg_scale=1.0,
                steps=2,
                use_amp=self.cfg.use_amp,
            )
            accumulator.update(prediction, target, valid_mask)
        metrics = accumulator.compute()
        if len(self.val_loader.dataset) == VAL_SAMPLES:
            if metrics["n_samples"] != VAL_SAMPLES:
                raise TrainerContractError("validation did not evaluate all 640 samples")
        return metrics

    def _state(self, *, smoke: bool = False) -> TrainerState:
        best = 0.0 if smoke and not math.isfinite(self.best_val_db_rmse) else self.best_val_db_rmse
        return TrainerState(
            completed_epochs=self.completed_epochs,
            next_epoch_index=self.next_epoch_index,
            optimizer_step=self.optimizer_step,
            micro_batches_seen=self.micro_batches_seen,
            samples_seen=self.samples_seen,
            best_val_db_rmse=best,
            epochs_without_improvement=self.epochs_without_improvement,
            history=tuple(self.history),
        )

    def _save(self, name: str, *, smoke: bool = False) -> Path:
        path = self.cfg.run_dir / name
        save_checkpoint_atomic(
            path,
            model=self.model,
            ema=self.ema,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            trainer_state=self._state(smoke=smoke),
            identity=self.identity,
            train_generator=self.train_generator,
        )
        return path

    def resume(self, path: Path) -> TrainerState:
        state = load_checkpoint_strict(
            path,
            model=self.model,
            ema=self.ema,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            expected_identity=self.identity,
            train_generator=self.train_generator,
        )
        self.completed_epochs = state.completed_epochs
        self.next_epoch_index = state.next_epoch_index
        self.optimizer_step = state.optimizer_step
        self.micro_batches_seen = state.micro_batches_seen
        self.samples_seen = state.samples_seen
        self.best_val_db_rmse = state.best_val_db_rmse
        self.epochs_without_improvement = state.epochs_without_improvement
        self.history = [dict(row) for row in state.history]
        if self.history:
            rebuild_metrics_csv(self.cfg.run_dir / "metrics.csv", self.history)
        return state

    def _runtime_payload(self, status: str) -> dict[str, Any]:
        peak = None
        if self.device.type == "cuda":
            peak = int(torch.cuda.max_memory_allocated(self.device))
        return {
            "schema_version": 1,
            "status": status,
            "elapsed_seconds": time.time() - self.started_at,
            "completed_epochs": self.completed_epochs,
            "optimizer_step": self.optimizer_step,
            "micro_batches_seen": self.micro_batches_seen,
            "samples_seen": self.samples_seen,
            "peak_training_allocated_bytes": peak,
            **self.cfg.precision_runtime(self.device),
        }

    def fit(self, *, stop_after_epoch: int | None = None) -> dict[str, Any]:
        stop = self.cfg.max_epochs if stop_after_epoch is None else stop_after_epoch
        if not self.next_epoch_index <= stop <= self.cfg.max_epochs:
            raise TrainerContractError(
                f"stop_after_epoch must be in [{self.next_epoch_index}, {self.cfg.max_epochs}]"
            )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        early_stopped = False
        while self.next_epoch_index < stop:
            epoch_index = self.next_epoch_index
            train_metrics = self.train_one_epoch()
            validation = self.validate()
            val_rmse = float(validation["db_rmse"])
            if not math.isfinite(val_rmse):
                raise TrainerContractError("validation dB-RMSE is non-finite")
            improved = val_rmse < self.best_val_db_rmse
            if improved:
                self.best_val_db_rmse = val_rmse
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            self.completed_epochs = epoch_index + 1
            self.next_epoch_index = self.completed_epochs
            row = {
                "epoch": self.completed_epochs,
                "optimizer_step": self.optimizer_step,
                "train_loss": float(train_metrics["loss"]),
                "train_valid_pixels": int(train_metrics["valid_pixels"]),
                "train_samples": int(train_metrics["samples"]),
                "val_n_samples": int(validation["n_samples"]),
                "val_n_valid_pixels": int(validation["n_valid_pixels"]),
                "val_db_rmse": val_rmse,
                "val_db_mae": float(validation["db_mae"]),
                "val_mse": float(validation["mse"]),
                "val_nmse": float(validation["nmse"]),
                "val_psnr": float(validation["psnr"]),
                "val_ssim": float(validation["ssim"]),
                "raw_fraction_below_zero": float(
                    validation["raw_fraction_below_zero"]
                ),
                "raw_fraction_above_one": float(
                    validation["raw_fraction_above_one"]
                ),
            }
            if any(
                isinstance(value, float) and not math.isfinite(value)
                for value in row.values()
            ):
                raise TrainerContractError("epoch history contains a non-finite metric")
            self.history.append(row)
            rebuild_metrics_csv(self.cfg.run_dir / "metrics.csv", self.history)
            if improved:
                self._save("best.pt")
            self._save("last.pt")
            if self.epochs_without_improvement >= self.cfg.early_stopping_patience:
                early_stopped = True
                break
        complete = early_stopped or self.completed_epochs >= self.cfg.max_epochs
        status = "complete" if complete else "paused"
        runtime = self._runtime_payload(status)
        _atomic_json(
            self.cfg.run_dir / "training_runtime.json",
            runtime,
            replace_existing=True,
        )
        return {
            "status": status,
            "run_dir": str(self.cfg.run_dir.resolve()),
            "completed_epochs": self.completed_epochs,
            "optimizer_step": self.optimizer_step,
            "best_val_db_rmse": self.best_val_db_rmse,
            "early_stopped": early_stopped,
        }

    def run_smoke(self, optimizer_steps: int) -> dict[str, Any]:
        if optimizer_steps <= 0:
            raise TrainerContractError("smoke optimizer_steps must be positive")
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        successful = 0
        numerator = 0.0
        denominator = 0
        while successful < optimizer_steps:
            iterator = iter(self.train_loader)
            made_progress = False
            while successful < optimizer_steps:
                window = tuple(itertools.islice(iterator, self.cfg.accumulation_steps))
                if not window:
                    break
                result = self._run_window(window)
                numerator += result.squared_error_sum
                denominator += result.valid_pixels
                if result.optimizer_ran:
                    successful += 1
                    made_progress = True
            if not made_progress:
                raise TrainerContractError("smoke made no successful optimizer progress")
        if denominator <= 0 or not math.isfinite(numerator / denominator):
            raise TrainerContractError("smoke loss is non-finite")
        checkpoint = self._save("smoke.pt", smoke=True)
        runtime = self._runtime_payload("smoke_complete")
        runtime["smoke_optimizer_steps"] = successful
        runtime["smoke_loss"] = numerator / denominator
        _atomic_json(
            self.cfg.run_dir / "training_runtime.json",
            runtime,
            replace_existing=True,
        )
        return {
            "status": "smoke_complete",
            "run_dir": str(self.cfg.run_dir.resolve()),
            "optimizer_steps": successful,
            "micro_batches_seen": self.micro_batches_seen,
            "loss": numerator / denominator,
            "checkpoint": str(checkpoint.resolve()),
            "peak_training_allocated_bytes": runtime["peak_training_allocated_bytes"],
        }


def _validate_smoke_checkpoint_fresh(
    cfg: MultiConfigTrainConfig,
    identity: CheckpointIdentity,
    checkpoint: Path,
    *,
    scaler_enabled: bool,
) -> None:
    model = build_locked_radioflow(cfg.model_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    ema = ModelEMA(model, decay=cfg.ema_decay)
    scheduler = build_optimizer_step_scheduler(
        optimizer,
        total_steps=cfg.planned_optimizer_steps,
        warmup_steps=cfg.warmup_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    state = load_checkpoint_strict(
        checkpoint,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_identity=identity,
        train_generator=generator,
    )
    if state.optimizer_step <= 0:
        raise TrainerContractError("fresh smoke reload has no optimizer progress")


def _safe_replace_smoke_dir(path: Path, run_root: Path) -> None:
    path = path.resolve()
    smoke_root = (run_root.resolve() / "_smoke").resolve()
    try:
        path.relative_to(smoke_root)
    except ValueError as error:
        raise TrainerContractError(f"unsafe smoke path: {path}") from error
    if path.exists():
        shutil.rmtree(path)


def _large_gate_context(
    cfg: MultiConfigTrainConfig,
    context: BenchmarkContext,
) -> LargeHardwareGateContext:
    return LargeHardwareGateContext(
        trigger_array=cfg.array_size,
        config_sha256_by_array={
            array_name: replace(cfg, array_size=array_name).config_sha256
            for array_name in ARRAY_SIZES
        },
        manifest_sha256_by_array={
            array_name: sha256_file(
                cfg.manifest_dir / f"manifest_{array_name}.jsonl"
            )
            for array_name in ARRAY_SIZES
        },
        split_sha256=context.split_sha256,
        schema_sha256=context.schema_sha256,
        archive_sha256=context.archive_sha256,
        dataset_revision=context.dataset_revision,
        radioflow_upstream_base=RADIOFLOW_UPSTREAM_BASE,
        git_commit=context.git_commit,
    )


def run_benchmark_training(
    cfg: MultiConfigTrainConfig,
    controls: InvocationControls,
    device: torch.device,
) -> dict[str, Any]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise TrainerContractError(
            "CUBLAS_WORKSPACE_CONFIG must be ':4096:8' before training"
        )
    context = preflight_benchmark(cfg)
    smoke = controls.smoke_optimizer_steps is not None
    effective_cfg = replace(cfg, run_root=cfg.run_root / "_smoke") if smoke else cfg
    if smoke:
        _safe_replace_smoke_dir(effective_cfg.run_dir, cfg.run_root)
    if not smoke and controls.resume == "none":
        if (effective_cfg.run_dir / "last.pt").exists():
            raise TrainerContractError("resume=none refuses an existing last.pt")
    write_or_validate_run_config(effective_cfg, controls)

    seed_everything(cfg.seed)
    model = build_locked_radioflow(cfg.model_size).to(device)
    train_loader, val_loader, train_generator = build_loaders(cfg, context)
    identity = build_checkpoint_identity(cfg, context, model)
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
        def smoke_action():
            return trainer.run_smoke(int(controls.smoke_optimizer_steps))

        try:
            result = smoke_action()
        except torch.cuda.OutOfMemoryError as error:
            if cfg.model_size != "large":
                raise
            hardware = collect_hardware_snapshot(device)

            def repeat_error():
                raise error

            gate = execute_with_large_oom_gate(
                repeat_error,
                gate_path=cfg.run_root / "_hardware" / "large_hardware_gate.json",
                context=_large_gate_context(cfg, context),
                hardware=hardware,
            )
            return {
                "status": "hardware_blocked",
                "gate_path": str(gate.gate_path),
                "gate_sha256": gate.sha256,
            }
        _validate_smoke_checkpoint_fresh(
            effective_cfg,
            identity,
            Path(result["checkpoint"]),
            scaler_enabled=device.type == "cuda" and cfg.use_amp,
        )
        return result

    resume_path: Path | None
    if controls.resume == "none":
        resume_path = None
    elif controls.resume == "auto":
        candidate = effective_cfg.run_dir / "last.pt"
        resume_path = candidate if candidate.is_file() else None
    else:
        resume_path = Path(controls.resume).resolve()
    if resume_path is not None:
        trainer.resume(resume_path)
    return trainer.fit(stop_after_epoch=controls.stop_after_epoch)

