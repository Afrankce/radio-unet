from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loaders.cross_frequency import load_cross_frequency_height_max
from data_loaders.sparse_task2 import (
    SparseTask2RadiomapDataset,
    sparse_task2_collate,
)
from evaluation.sparse_task2_metrics import sparse_task2_metrics_for_json
from evaluation.sparse_task2_sampling import (
    make_task2_sample_noise,
    sparse_task2_euler_cfg_sample,
)
from experiments.multiconfig_manifest import canonical_json_bytes
from experiments.provenance import sha256_file
from experiments.sparse_task2_manifest import (
    MANDATORY_SINGLEBEAM_SCENE_SPLIT_SHA256,
    MANDATORY_SINGLEBEAM_PROTOCOL,
    validate_singlebeam_task2_manifest,
)
from train import ModelEMA
from training.checkpointing import (
    CheckpointIdentity,
    TrainerState,
    load_checkpoint_strict,
    rebuild_metrics_csv,
    save_checkpoint_atomic,
)
from training.config import InvocationControls
from training.model_factory import build_task2_sparse_radioflow
from training.multiconfig_trainer import resolve_device, seed_everything, seed_worker
from training.optimization import build_optimizer_step_scheduler, run_accumulation_window
from training.sparse_task2_config import (
    SINGLEBEAM_TASK2_SAMPLE_COUNT,
    SINGLEBEAM_TASK2_SCENE_COUNTS,
    SparseTask2TrainConfig,
)
from training.sparse_task2_flow import build_task2_flow_pair


class SparseTask2TrainerError(RuntimeError):
    """Task 2 preflight or training state violates the locked protocol."""


@dataclass(frozen=True)
class SparseTask2Context:
    train_dataset: SparseTask2RadiomapDataset
    val_dataset: SparseTask2RadiomapDataset
    test_dataset: SparseTask2RadiomapDataset
    manifest_path: Path
    split_path: Path
    height_stats_path: Path
    manifest_sha256: str
    split_sha256: str
    height_stats_sha256: str
    mask_protocol_sha256: str
    height_max: float
    audit: Mapping[str, object]


def _mask_protocol_sha256(
    cfg: SparseTask2TrainConfig,
    *,
    manifest_sha256: str,
    split_sha256: str,
) -> str:
    payload = {
        "schema_version": 1,
        "protocol": MANDATORY_SINGLEBEAM_PROTOCOL,
        "array_size": cfg.array_size,
        "frequency_hz": 6_700_000_000,
        "steering_deg": 0.0,
        "sample_count": SINGLEBEAM_TASK2_SAMPLE_COUNT,
        "mask_seed": 42,
        "manifest_sha256": manifest_sha256,
        "split_sha256": split_sha256,
        "mask_key": "protocol|seed|scene_id|count",
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _check_dataset_sample(
    sample: Mapping[str, Any],
    *,
    split: str,
    index: int,
) -> None:
    condition = sample.get("condition")
    target = sample.get("target")
    valid_mask = sample.get("valid_mask")
    observation_mask = sample.get("observation_mask")
    sparse_map = sample.get("sparse_map")
    if not isinstance(condition, torch.Tensor) or tuple(condition.shape) != (5, 256, 256):
        raise SparseTask2TrainerError(
            f"{split}[{index}] condition must have shape (5,256,256)"
        )
    if not isinstance(target, torch.Tensor) or tuple(target.shape) != (1, 256, 256):
        raise SparseTask2TrainerError(f"{split}[{index}] target shape mismatch")
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype is not torch.bool:
        raise SparseTask2TrainerError(f"{split}[{index}] valid_mask must be bool")
    if not isinstance(observation_mask, torch.Tensor) or observation_mask.dtype is not torch.bool:
        raise SparseTask2TrainerError(f"{split}[{index}] observation_mask must be bool")
    if observation_mask.shape != valid_mask.shape or sparse_map.shape != target.shape:
        raise SparseTask2TrainerError(f"{split}[{index}] sparse tensor shapes mismatch")
    if int(observation_mask.sum().item()) != SINGLEBEAM_TASK2_SAMPLE_COUNT:
        raise SparseTask2TrainerError(
            f"{split}[{index}] observation count is not "
            f"{SINGLEBEAM_TASK2_SAMPLE_COUNT}"
        )
    if bool((observation_mask & ~valid_mask).any()):
        raise SparseTask2TrainerError(f"{split}[{index}] observation is outside valid mask")
    if not bool(torch.isfinite(condition).all()) or not bool(torch.isfinite(target).all()):
        raise SparseTask2TrainerError(f"{split}[{index}] has non-finite tensor values")


def preflight_sparse_task2(
    cfg: SparseTask2TrainConfig,
    *,
    audit_all_samples: bool = False,
) -> SparseTask2Context:
    manifest_path = cfg.manifest_path.resolve()
    height_stats_path = cfg.height_stats_path.resolve()
    split_path = manifest_path.parent / "scene_split_seed42.json"
    for path, label in (
        (manifest_path, "manifest"),
        (height_stats_path, "height statistics"),
        (split_path, "scene split"),
    ):
        if not path.is_file():
            raise SparseTask2TrainerError(f"{label} is missing: {path}")
    try:
        audit = validate_singlebeam_task2_manifest(
            manifest_path=manifest_path,
            split_path=split_path,
            array_size=cfg.array_size,
        )
        height_max = load_cross_frequency_height_max(
            height_stats_path,
            split_path=split_path,
        )
    except Exception as error:
        raise SparseTask2TrainerError(f"Task 2 data contract failed: {error}") from error
    if sha256_file(split_path) != MANDATORY_SINGLEBEAM_SCENE_SPLIT_SHA256:
        raise SparseTask2TrainerError("scene split hash is not the locked seed-42 split")

    datasets: dict[str, SparseTask2RadiomapDataset] = {}
    for split in ("train", "val", "test"):
        try:
            dataset = SparseTask2RadiomapDataset(
                dataset_root=cfg.dataset_root.resolve(),
                manifest_path=manifest_path,
                split=split,
                array_size=cfg.array_size,
                height_max=height_max,
                expected_counts=SINGLEBEAM_TASK2_SCENE_COUNTS,
                mask_seed=42,
                sample_count=SINGLEBEAM_TASK2_SAMPLE_COUNT,
            )
            if len(dataset) != SINGLEBEAM_TASK2_SCENE_COUNTS[split]:
                raise SparseTask2TrainerError(
                    f"{split} count mismatch: expected "
                    f"{SINGLEBEAM_TASK2_SCENE_COUNTS[split]}, got {len(dataset)}"
                )
            indices = range(len(dataset)) if audit_all_samples else (0, len(dataset) - 1)
            for index in sorted(set(indices)):
                _check_dataset_sample(dataset[index], split=split, index=index)
            datasets[split] = dataset
        except SparseTask2TrainerError:
            raise
        except Exception as error:
            raise SparseTask2TrainerError(
                f"cannot construct or audit {split} Task 2 dataset: {error}"
            ) from error

    manifest_sha256 = sha256_file(manifest_path)
    split_sha256 = sha256_file(split_path)
    return SparseTask2Context(
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        test_dataset=datasets["test"],
        manifest_path=manifest_path,
        split_path=split_path,
        height_stats_path=height_stats_path,
        manifest_sha256=manifest_sha256,
        split_sha256=split_sha256,
        height_stats_sha256=sha256_file(height_stats_path),
        mask_protocol_sha256=_mask_protocol_sha256(
            cfg,
            manifest_sha256=manifest_sha256,
            split_sha256=split_sha256,
        ),
        height_max=height_max,
        audit=audit,
    )


def build_sparse_task2_loaders(
    cfg: SparseTask2TrainConfig,
    context: SparseTask2Context,
) -> tuple[DataLoader, DataLoader, torch.Generator]:
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    train_loader = DataLoader(
        context.train_dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=cfg.num_workers,
        collate_fn=sparse_task2_collate,
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
        collate_fn=sparse_task2_collate,
        worker_init_fn=seed_worker,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )
    if len(train_loader) != math.ceil(cfg.train_samples / cfg.micro_batch_size):
        raise SparseTask2TrainerError("Task 2 train loader count mismatch")
    if math.ceil(len(train_loader) / cfg.accumulation_steps) != cfg.optimizer_steps_per_epoch:
        raise SparseTask2TrainerError("Task 2 optimizer-step count mismatch")
    if len(val_loader) != cfg.val_samples:
        raise SparseTask2TrainerError("Task 2 validation loader count mismatch")
    return train_loader, val_loader, generator


def build_sparse_task2_checkpoint_identity(
    cfg: SparseTask2TrainConfig,
    context: SparseTask2Context,
    model: torch.nn.Module,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        protocol=MANDATORY_SINGLEBEAM_PROTOCOL,
        array_size=cfg.array_size,
        variant=cfg.condition_variant,
        model_size=cfg.model_size,
        condition_channels=cfg.condition_channels,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        config_sha256=cfg.config_sha256,
        manifest_sha256=context.manifest_sha256,
        split_sha256=context.split_sha256,
        mask_protocol_sha256=context.mask_protocol_sha256,
        observation_count=SINGLEBEAM_TASK2_SAMPLE_COUNT,
        split_type="scene_disjoint_single_beam",
    )


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        output.write(canonical_json_bytes(payload))
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _validate_smoke_checkpoint_fresh(
    cfg: SparseTask2TrainConfig,
    identity: CheckpointIdentity,
    checkpoint: Path,
) -> None:
    """Reload a smoke checkpoint through the same strict identity path."""

    model = build_task2_sparse_radioflow(
        condition_variant=cfg.condition_variant,
        model_size=cfg.model_size,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    ema = ModelEMA(model, decay=cfg.ema_decay)
    scheduler = build_optimizer_step_scheduler(
        optimizer,
        total_steps=cfg.planned_optimizer_steps,
        warmup_steps=cfg.warmup_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
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
        raise SparseTask2TrainerError("fresh smoke reload has no optimizer progress")


class SparseTask2Trainer:
    def __init__(
        self,
        cfg: SparseTask2TrainConfig,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        train_generator: torch.Generator,
        identity: CheckpointIdentity,
    ) -> None:
        parameters = tuple(model.parameters())
        if not parameters:
            raise SparseTask2TrainerError("Task 2 model has no parameters")
        if any(parameter.device != device for parameter in parameters):
            raise SparseTask2TrainerError("Task 2 model must already be on the requested device")
        identity.validate()
        actual = sum(parameter.numel() for parameter in parameters)
        if actual != identity.parameter_count:
            raise SparseTask2TrainerError("Task 2 checkpoint parameter count mismatch")
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.train_generator = train_generator
        self.identity = identity
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        self.ema = ModelEMA(model, decay=cfg.ema_decay)
        self.scheduler = build_optimizer_step_scheduler(
            self.optimizer,
            total_steps=cfg.planned_optimizer_steps,
            warmup_steps=cfg.warmup_steps,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and cfg.use_amp
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
            time_step = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
            xt, ut, loss_mask = build_task2_flow_pair(
                x0, target, valid_mask, time_step
            )
            embedding = model.embed_model(condition)
            predicted = model(
                image=condition,
                x=xt,
                pred_type="denoise",
                step=time_step,
                embedding=embedding,
            )
        return predicted, ut, loss_mask

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
        samples = 0
        iterator = iter(self.train_loader)
        while True:
            window = tuple(itertools.islice(iterator, self.cfg.accumulation_steps))
            if not window:
                break
            result = self._run_window(window)
            numerator += result.squared_error_sum
            valid_pixels += result.valid_pixels
            samples += result.samples
        if valid_pixels <= 0 or samples <= 0:
            raise SparseTask2TrainerError("Task 2 epoch produced no valid pixels")
        return {
            "train_loss": numerator / valid_pixels,
            "train_valid_pixels": valid_pixels,
            "train_samples": samples,
        }

    def validate(self) -> dict[str, Any]:
        from evaluation.sparse_task2_metrics import SparseTask2MetricAccumulator

        network = self.ema.ema_model
        network.eval()
        accumulator = SparseTask2MetricAccumulator()
        for batch in self.val_loader:
            condition = batch["condition"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(self.device, non_blocking=True)
            observation_mask = batch["observation_mask"].to(self.device, non_blocking=True)
            sparse_map = batch["sparse_map"].to(self.device, non_blocking=True)
            noises = [
                make_task2_sample_noise(
                    protocol=MANDATORY_SINGLEBEAM_PROTOCOL,
                    array_size=self.cfg.array_size,
                    split="val",
                    sample_key=str(metadata["sample_key"]),
                    shape=tuple(target.shape[1:]),
                    base_seed=self.cfg.seed,
                )
                for metadata in batch["metadata"]
            ]
            noise = torch.stack(noises).to(self.device)
            prediction = sparse_task2_euler_cfg_sample(
                network,
                condition,
                noise,
                cfg_scale=1.0,
                steps=2,
                use_amp=self.cfg.use_amp,
            )
            accumulator.update(prediction, target, valid_mask, observation_mask, batch["metadata"])
        return accumulator.compute()

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
        peak = int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else None
        return {
            "schema_version": 1,
            "status": status,
            "protocol": MANDATORY_SINGLEBEAM_PROTOCOL,
            "array_size": self.cfg.array_size,
            "condition_variant": self.cfg.condition_variant,
            "completed_epochs": self.completed_epochs,
            "optimizer_step": self.optimizer_step,
            "micro_batches_seen": self.micro_batches_seen,
            "samples_seen": self.samples_seen,
            "best_val_db_rmse": (
                self.best_val_db_rmse
                if math.isfinite(self.best_val_db_rmse)
                else None
            ),
            "early_stopped": self.epochs_without_improvement >= self.cfg.early_stopping_patience,
            "elapsed_seconds": time.time() - self.started_at,
            "peak_training_allocated_bytes": peak,
            **self.cfg.precision_runtime(self.device),
        }

    def fit(self, *, stop_after_epoch: int | None = None) -> dict[str, Any]:
        stop = self.cfg.max_epochs if stop_after_epoch is None else stop_after_epoch
        if not self.next_epoch_index <= stop <= self.cfg.max_epochs:
            raise SparseTask2TrainerError("stop_after_epoch is outside the valid range")
        while self.completed_epochs < stop:
            train = self.train_one_epoch()
            validation = self.validate()
            overall = validation["overall"]
            val_rmse = float(overall["db_rmse"])
            if not math.isfinite(val_rmse):
                raise SparseTask2TrainerError("validation dB-RMSE is non-finite")
            improved = val_rmse < self.best_val_db_rmse
            if improved:
                self.best_val_db_rmse = val_rmse
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            self.completed_epochs += 1
            self.next_epoch_index = self.completed_epochs
            row = {
                "epoch": self.completed_epochs,
                "optimizer_step": self.optimizer_step,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                **train,
                "val_overall_db_rmse": val_rmse,
                "val_overall_db_mae": float(overall["db_mae"]),
                "val_overall_nmse": float(overall["nmse"]),
                "val_overall_psnr": float(overall["psnr"]),
                "val_overall_ssim": float(overall["ssim"]),
                "val_missing_db_rmse": float(validation["missing"]["db_rmse"]),
                "val_missing_psnr": float(validation["missing"]["psnr"]),
                "val_missing_ssim": float(validation["missing"]["ssim"]),
                "ema": True,
            }
            if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
                raise SparseTask2TrainerError("epoch history contains a non-finite metric")
            self.history.append(row)
            _write_metrics_csv(self.cfg.run_dir / "metrics.csv", self.history)
            if improved:
                self._save("best.pt")
            self._save("last.pt")
            if self.epochs_without_improvement >= self.cfg.early_stopping_patience:
                break
        complete = self.completed_epochs >= self.cfg.max_epochs or (
            self.epochs_without_improvement >= self.cfg.early_stopping_patience
        )
        status = "complete" if complete else "paused"
        runtime = self._runtime_payload(status)
        _write_json(self.cfg.run_dir / "training_runtime.json", runtime)
        return {
            "status": status,
            "run_dir": str(self.cfg.run_dir.resolve()),
            "completed_epochs": self.completed_epochs,
            "optimizer_step": self.optimizer_step,
            "best_val_db_rmse": self.best_val_db_rmse,
            "early_stopped": runtime["early_stopped"],
        }

    def run_smoke(self, optimizer_steps: int) -> dict[str, Any]:
        if optimizer_steps <= 0:
            raise SparseTask2TrainerError("smoke optimizer_steps must be positive")
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
                raise SparseTask2TrainerError("smoke made no optimizer progress")
        loss = numerator / denominator
        if denominator <= 0 or not math.isfinite(loss):
            raise SparseTask2TrainerError("smoke loss is non-finite")
        self.completed_epochs = 1
        self.next_epoch_index = 1
        checkpoint = self._save("smoke.pt", smoke=True)
        runtime = self._runtime_payload("smoke_complete")
        runtime["smoke_optimizer_steps"] = successful
        runtime["smoke_loss"] = loss
        _write_json(self.cfg.run_dir / "training_runtime.json", runtime)
        return {
            "status": "smoke_complete",
            "run_dir": str(self.cfg.run_dir.resolve()),
            "optimizer_steps": successful,
            "micro_batches_seen": self.micro_batches_seen,
            "loss": loss,
            "checkpoint": str(checkpoint.resolve()),
            "peak_training_allocated_bytes": runtime["peak_training_allocated_bytes"],
        }


def _write_config(cfg: SparseTask2TrainConfig, controls: InvocationControls, context: SparseTask2Context) -> None:
    payload = {
        **cfg.to_record(
            manifest_sha256=context.manifest_sha256,
            split_sha256=context.split_sha256,
            mask_protocol_sha256=context.mask_protocol_sha256,
        ),
        "invocation": controls.to_dict(),
        "height_stats_sha256": context.height_stats_sha256,
    }
    path = cfg.run_dir / "config.json"
    expected = canonical_json_bytes(payload)
    if path.exists() and path.read_bytes() != expected:
        raise SparseTask2TrainerError("existing Task 2 config differs from requested run")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(expected)
        os.replace(temporary, path)


def run_sparse_task2_training(
    cfg: SparseTask2TrainConfig,
    controls: InvocationControls,
    device: torch.device,
    *,
    preflight_only: bool = False,
    audit_all_samples: bool = False,
) -> dict[str, Any]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise SparseTask2TrainerError(
            "CUBLAS_WORKSPACE_CONFIG must be ':4096:8' before Task 2 training"
        )
    context = preflight_sparse_task2(cfg, audit_all_samples=audit_all_samples)
    seed_everything(cfg.seed)
    model = build_task2_sparse_radioflow(
        condition_variant=cfg.condition_variant,
        model_size=cfg.model_size,
    ).to(device)
    identity = build_sparse_task2_checkpoint_identity(cfg, context, model)
    if preflight_only:
        return {
            "status": "preflight_complete",
            "protocol": MANDATORY_SINGLEBEAM_PROTOCOL,
            "array_size": cfg.array_size,
            "manifest": str(context.manifest_path),
            "manifest_sha256": context.manifest_sha256,
            "split_sha256": context.split_sha256,
            "height_stats_sha256": context.height_stats_sha256,
            "mask_protocol_sha256": context.mask_protocol_sha256,
            "train_samples": len(context.train_dataset),
            "val_samples": len(context.val_dataset),
            "test_samples": len(context.test_dataset),
            "condition_channels": cfg.condition_channels,
            "observation_count": SINGLEBEAM_TASK2_SAMPLE_COUNT,
            "parameter_count": identity.parameter_count,
        }
    smoke = controls.smoke_optimizer_steps is not None
    effective_cfg = replace(cfg, run_root=cfg.run_root / "_smoke") if smoke else cfg
    _write_config(effective_cfg, controls, context)
    if not smoke and controls.resume == "none" and (effective_cfg.run_dir / "last.pt").exists():
        raise SparseTask2TrainerError("resume=none refuses an existing Task 2 last.pt")
    train_loader, val_loader, train_generator = build_sparse_task2_loaders(cfg, context)
    trainer = SparseTask2Trainer(
        effective_cfg, model, train_loader, val_loader, device, train_generator, identity
    )
    if smoke:
        result = trainer.run_smoke(int(controls.smoke_optimizer_steps))
        _validate_smoke_checkpoint_fresh(
            effective_cfg,
            identity,
            Path(result["checkpoint"]),
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


__all__ = [
    "SparseTask2Context",
    "SparseTask2Trainer",
    "SparseTask2TrainerError",
    "build_sparse_task2_checkpoint_identity",
    "build_sparse_task2_loaders",
    "preflight_sparse_task2",
    "run_sparse_task2_training",
    "resolve_device",
]
