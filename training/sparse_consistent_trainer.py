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

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from data_loaders.cross_frequency import load_cross_frequency_height_max
from data_loaders.sparse_consistent import (
    SparseConsistentRadiomapDataset,
    sparse_consistent_collate,
)
from evaluation.sparse_consistent_sampling import (
    make_sparse_consistent_sample_noise,
    sparse_consistent_euler_cfg_sample,
)
from evaluation.sparse_task2_metrics import SparseTask2MetricAccumulator
from experiments.multiconfig_manifest import canonical_json_bytes
from experiments.provenance import sha256_file
from experiments.sparse_task2_manifest import (
    MANDATORY_SINGLEBEAM_SCENE_SPLIT_SHA256,
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
from training.masked_flow_loss import masked_velocity_mse
from training.multiconfig_trainer import resolve_device, seed_everything, seed_worker
from training.optimization import build_optimizer_step_scheduler
from training.sparse_consistent_config import (
    SPARSE_CONSISTENT_ARMS,
    SPARSE_CONSISTENT_PROTOCOL,
    SPARSE_CONSISTENT_SAMPLE_COUNT,
    SPARSE_CONSISTENT_SCENE_COUNTS,
    SparseConsistentTrainConfig,
)
from training.sparse_consistent_flow import build_sparse_consistent_flow_pair
from training.sparse_consistent_model import (
    build_sparse_consistent_model,
    embed_sparse_consistent_model,
)


class SparseConsistentTrainerError(RuntimeError):
    """The registered sparse-consistent training contract is invalid."""


@dataclass(frozen=True)
class SparseConsistentContext:
    train_dataset: SparseConsistentRadiomapDataset
    val_dataset: SparseConsistentRadiomapDataset
    test_dataset: SparseConsistentRadiomapDataset
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
    cfg: SparseConsistentTrainConfig,
    *,
    manifest_sha256: str,
    split_sha256: str,
) -> str:
    payload = {
        "schema_version": 1,
        "protocol": SPARSE_CONSISTENT_PROTOCOL,
        "arm": cfg.arm,
        "array_size": cfg.array_size,
        "frequency_hz": 6_700_000_000,
        "steering_deg": 0.0,
        "sample_count": SPARSE_CONSISTENT_SAMPLE_COUNT,
        "mask_seed": 42,
        "manifest_sha256": manifest_sha256,
        "split_sha256": split_sha256,
        "mask_key": "singlebeam_feature5_samples819|seed|scene_id|count",
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _check_sample(
    sample: Mapping[str, Any],
    *,
    cfg: SparseConsistentTrainConfig,
    split: str,
    index: int,
) -> None:
    expected_channels = cfg.condition_channels
    condition = sample.get("condition")
    environment = sample.get("environment_condition")
    target = sample.get("target")
    valid_mask = sample.get("valid_mask")
    observation_mask = sample.get("observation_mask")
    sparse_map = sample.get("sparse_map")
    if not isinstance(condition, Tensor) or tuple(condition.shape) != (
        expected_channels,
        256,
        256,
    ):
        raise SparseConsistentTrainerError(
            f"{split}[{index}] condition must have shape "
            f"({expected_channels},256,256), got {getattr(condition, 'shape', None)}"
        )
    if not isinstance(environment, Tensor) or tuple(environment.shape) != (3, 256, 256):
        raise SparseConsistentTrainerError(f"{split}[{index}] environment shape mismatch")
    if not isinstance(target, Tensor) or tuple(target.shape) != (1, 256, 256):
        raise SparseConsistentTrainerError(f"{split}[{index}] target shape mismatch")
    for name, value in (
        ("valid_mask", valid_mask),
        ("observation_mask", observation_mask),
    ):
        if not isinstance(value, Tensor) or value.shape != target.shape or value.dtype is not torch.bool:
            raise SparseConsistentTrainerError(
                f"{split}[{index}] {name} must be boolean with target shape"
            )
    if not isinstance(sparse_map, Tensor) or sparse_map.shape != target.shape:
        raise SparseConsistentTrainerError(f"{split}[{index}] sparse_map shape mismatch")
    if int(observation_mask.sum().item()) != SPARSE_CONSISTENT_SAMPLE_COUNT:
        raise SparseConsistentTrainerError(
            f"{split}[{index}] observation count is not {SPARSE_CONSISTENT_SAMPLE_COUNT}"
        )
    if bool((observation_mask & ~valid_mask).any()):
        raise SparseConsistentTrainerError(f"{split}[{index}] observation is outside valid mask")
    if bool((sparse_map.masked_select(~observation_mask)).abs().max().item() > 1e-6):
        raise SparseConsistentTrainerError(f"{split}[{index}] sparse_map is nonzero outside observations")
    for name, value in (
        ("condition", condition),
        ("environment_condition", environment),
        ("target", target),
        ("sparse_map", sparse_map),
    ):
        if not bool(torch.isfinite(value).all()):
            raise SparseConsistentTrainerError(f"{split}[{index}] {name} has non-finite values")


def preflight_sparse_consistent(
    cfg: SparseConsistentTrainConfig,
    *,
    audit_all_samples: bool = False,
) -> SparseConsistentContext:
    manifest_path = cfg.manifest_path.resolve()
    height_stats_path = cfg.height_stats_path.resolve()
    split_path = manifest_path.parent / "scene_split_seed42.json"
    for path, label in (
        (manifest_path, "manifest"),
        (height_stats_path, "height statistics"),
        (split_path, "scene split"),
    ):
        if not path.is_file():
            raise SparseConsistentTrainerError(f"{label} is missing: {path}")
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
        raise SparseConsistentTrainerError(f"data contract failed: {error}") from error
    if sha256_file(split_path) != MANDATORY_SINGLEBEAM_SCENE_SPLIT_SHA256:
        raise SparseConsistentTrainerError("scene split hash is not the locked seed-42 split")

    datasets: dict[str, SparseConsistentRadiomapDataset] = {}
    for split in ("train", "val", "test"):
        try:
            dataset = SparseConsistentRadiomapDataset(
                dataset_root=cfg.dataset_root.resolve(),
                manifest_path=manifest_path,
                split=split,
                array_size=cfg.array_size,
                height_max=height_max,
                arm=cfg.arm,
            )
            if len(dataset) != SPARSE_CONSISTENT_SCENE_COUNTS[split]:
                raise SparseConsistentTrainerError(
                    f"{split} count mismatch: expected "
                    f"{SPARSE_CONSISTENT_SCENE_COUNTS[split]}, got {len(dataset)}"
                )
            indices = range(len(dataset)) if audit_all_samples else (0, len(dataset) - 1)
            for index in sorted(set(indices)):
                _check_sample(dataset[index], cfg=cfg, split=split, index=index)
            datasets[split] = dataset
        except SparseConsistentTrainerError:
            raise
        except Exception as error:
            raise SparseConsistentTrainerError(
                f"cannot construct or audit {split} dataset: {error}"
            ) from error

    manifest_sha256 = sha256_file(manifest_path)
    split_sha256 = sha256_file(split_path)
    return SparseConsistentContext(
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


def build_sparse_consistent_loaders(
    cfg: SparseConsistentTrainConfig,
    context: SparseConsistentContext,
) -> tuple[DataLoader, DataLoader, torch.Generator]:
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    train_loader = DataLoader(
        context.train_dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=cfg.num_workers,
        collate_fn=sparse_consistent_collate,
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
        collate_fn=sparse_consistent_collate,
        worker_init_fn=seed_worker,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )
    if len(train_loader) != math.ceil(cfg.train_samples / cfg.micro_batch_size):
        raise SparseConsistentTrainerError("train loader count mismatch")
    if math.ceil(len(train_loader) / cfg.accumulation_steps) != cfg.optimizer_steps_per_epoch:
        raise SparseConsistentTrainerError("optimizer-step count mismatch")
    if len(val_loader) != cfg.val_samples:
        raise SparseConsistentTrainerError("validation loader count mismatch")
    return train_loader, val_loader, generator


def build_sparse_consistent_checkpoint_identity(
    cfg: SparseConsistentTrainConfig,
    context: SparseConsistentContext,
    model: torch.nn.Module,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        experiment=SPARSE_CONSISTENT_PROTOCOL,
        array_size=cfg.array_size,
        variant=cfg.arm,
        model_size=cfg.model_size,
        condition_channels=cfg.condition_channels,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        config_sha256=cfg.config_sha256,
        mask_protocol_sha256=context.mask_protocol_sha256,
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


class SparseConsistentTrainer:
    def __init__(
        self,
        cfg: SparseConsistentTrainConfig,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        train_generator: torch.Generator,
        identity: CheckpointIdentity,
    ) -> None:
        parameters = tuple(model.parameters())
        if not parameters or any(parameter.device != device for parameter in parameters):
            raise SparseConsistentTrainerError("model must have parameters on the requested device")
        identity.validate()
        actual = sum(parameter.numel() for parameter in parameters)
        if actual != identity.parameter_count:
            raise SparseConsistentTrainerError("checkpoint parameter count mismatch")
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

    def _prediction_triplet(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor]:
        condition = batch["condition"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(self.device, non_blocking=True)
        observation_mask = batch["observation_mask"].to(self.device, non_blocking=True)
        sparse_map = batch["sparse_map"].to(self.device, non_blocking=True)
        loss_mask = valid_mask & ~observation_mask if self.cfg.uses_consistent_flow else valid_mask
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda" and self.cfg.use_amp,
        ):
            x0 = torch.randn_like(target)
            time_step = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
            xt, ut, flow_mask = build_sparse_consistent_flow_pair(
                arm=self.cfg.arm,
                x0=x0,
                target=target,
                valid_mask=valid_mask,
                observation_mask=observation_mask,
                sparse_map=sparse_map,
                time=time_step,
            )
            embedding = embed_sparse_consistent_model(
                model,
                arm=self.cfg.arm,
                condition=condition,
                sparse_map=sparse_map,
                observation_mask=observation_mask,
            )
            predicted = model(
                image=condition,
                x=xt,
                pred_type="denoise",
                step=time_step,
                embedding=embedding,
            )
        if flow_mask.shape != loss_mask.shape or not bool(torch.equal(flow_mask, loss_mask)):
            raise SparseConsistentTrainerError("flow loss mask does not match registered objective")
        return predicted, ut, flow_mask

    def _run_window(self, window: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
        if not window:
            raise SparseConsistentTrainerError("empty accumulation window")
        masks = []
        for batch in window:
            valid = batch["valid_mask"]
            if self.cfg.uses_consistent_flow:
                valid = valid & ~batch["observation_mask"]
            masks.append(int(valid.sum().item()))
        total_valid = sum(masks)
        if total_valid <= 0:
            raise SparseConsistentTrainerError("accumulation window has no objective pixels")
        self.optimizer.zero_grad(set_to_none=True)
        squared_error_sum = 0.0
        samples = 0
        for batch, micro_valid in zip(window, masks):
            predicted, target_velocity, loss_mask = self._prediction_triplet(self.model, batch)
            loss = masked_velocity_mse(predicted, target_velocity, loss_mask)
            weight = torch.tensor(
                micro_valid / total_valid, device=loss.device, dtype=torch.float32
            )
            self.scaler.scale(loss * weight).backward()
            squared_error_sum += float(loss.detach().float().item()) * micro_valid
            samples += int(predicted.shape[0])
        before = float(self.scaler.get_scale())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        after = float(self.scaler.get_scale())
        if after >= before:
            self.ema.update(self.model)
            self.scheduler.step()
            optimizer_ran = True
            self.optimizer_step += 1
        else:
            optimizer_ran = False
        self.optimizer.zero_grad(set_to_none=True)
        self.micro_batches_seen += len(window)
        self.samples_seen += samples
        return {
            "optimizer_ran": optimizer_ran,
            "micro_batches": len(window),
            "samples": samples,
            "objective_pixels": total_valid,
            "squared_error_sum": squared_error_sum,
        }

    def train_one_epoch(self) -> dict[str, int | float]:
        self.model.train()
        numerator = 0.0
        denominator = 0
        samples = 0
        iterator = iter(self.train_loader)
        while True:
            window = tuple(itertools.islice(iterator, self.cfg.accumulation_steps))
            if not window:
                break
            result = self._run_window(window)
            numerator += result["squared_error_sum"]
            denominator += result["objective_pixels"]
            samples += result["samples"]
        if denominator <= 0 or samples <= 0:
            raise SparseConsistentTrainerError("epoch produced no objective pixels")
        return {
            "train_loss": numerator / denominator,
            "train_objective_pixels": denominator,
            "train_samples": samples,
        }

    @torch.inference_mode()
    def validate(self) -> dict[str, Any]:
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
                make_sparse_consistent_sample_noise(
                    array_size=self.cfg.array_size,
                    split="val",
                    sample_key=str(metadata["sample_key"]),
                    shape=tuple(target.shape[1:]),
                    base_seed=self.cfg.seed,
                )
                for metadata in batch["metadata"]
            ]
            noise = torch.stack(noises).to(self.device)
            prediction = sparse_consistent_euler_cfg_sample(
                network,
                arm=self.cfg.arm,
                condition=condition,
                x0=noise,
                sparse_map=sparse_map,
                observation_mask=observation_mask,
                cfg_scale=self.cfg.cfg_scale,
                steps=self.cfg.euler_steps,
                use_amp=self.cfg.use_amp,
            )
            accumulator.update(
                prediction,
                target,
                valid_mask,
                observation_mask,
                batch["metadata"],
            )
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
            "protocol": SPARSE_CONSISTENT_PROTOCOL,
            "array_size": self.cfg.array_size,
            "arm": self.cfg.arm,
            "completed_epochs": self.completed_epochs,
            "optimizer_step": self.optimizer_step,
            "micro_batches_seen": self.micro_batches_seen,
            "samples_seen": self.samples_seen,
            "best_val_missing_db_rmse": (
                self.best_val_db_rmse if math.isfinite(self.best_val_db_rmse) else None
            ),
            "early_stopped": (
                self.optimizer_step >= self.cfg.min_optimizer_steps
                and self.epochs_without_improvement >= self.cfg.early_stopping_patience
            ),
            "elapsed_seconds": time.time() - self.started_at,
            "peak_training_allocated_bytes": peak,
            **self.cfg.precision_runtime(self.device),
        }

    def fit(self) -> dict[str, Any]:
        while self.completed_epochs < self.cfg.max_epochs:
            train = self.train_one_epoch()
            validation = self.validate()
            missing = validation["missing"]
            overall = validation["overall"]
            val_rmse = float(missing["db_rmse"])
            if not math.isfinite(val_rmse):
                raise SparseConsistentTrainerError("validation missing dB-RMSE is non-finite")
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
                "val_missing_db_rmse": val_rmse,
                "val_missing_db_mae": float(missing["db_mae"]),
                "val_missing_nmse": float(missing["nmse"]),
                "val_missing_psnr": float(missing["psnr"]),
                "val_missing_ssim": float(missing["ssim"]),
                "val_overall_db_rmse": float(overall["db_rmse"]),
                "val_overall_psnr": float(overall["psnr"]),
                "val_overall_ssim": float(overall["ssim"]),
                "ema": True,
            }
            if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
                raise SparseConsistentTrainerError("epoch history contains a non-finite value")
            self.history.append(row)
            _write_metrics_csv(self.cfg.run_dir / "metrics.csv", self.history)
            if improved:
                self._save("best.pt")
            self._save("last.pt")
            if (
                self.optimizer_step >= self.cfg.min_optimizer_steps
                and self.epochs_without_improvement >= self.cfg.early_stopping_patience
            ):
                break
        status = "complete"
        runtime = self._runtime_payload(status)
        _write_json(self.cfg.run_dir / "training_runtime.json", runtime)
        return {
            "status": status,
            "run_dir": str(self.cfg.run_dir.resolve()),
            "arm": self.cfg.arm,
            "array_size": self.cfg.array_size,
            "completed_epochs": self.completed_epochs,
            "optimizer_step": self.optimizer_step,
            "best_val_missing_db_rmse": self.best_val_db_rmse,
            "early_stopped": runtime["early_stopped"],
        }

    def run_smoke(self, optimizer_steps: int) -> dict[str, Any]:
        if optimizer_steps <= 0:
            raise SparseConsistentTrainerError("smoke optimizer_steps must be positive")
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
                numerator += result["squared_error_sum"]
                denominator += result["objective_pixels"]
                if result["optimizer_ran"]:
                    successful += 1
                    made_progress = True
            if not made_progress:
                raise SparseConsistentTrainerError("smoke made no optimizer progress")
        loss = numerator / denominator
        if denominator <= 0 or not math.isfinite(loss):
            raise SparseConsistentTrainerError("smoke loss is non-finite")
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
            "arm": self.cfg.arm,
            "array_size": self.cfg.array_size,
            "optimizer_steps": successful,
            "loss": loss,
            "checkpoint": str(checkpoint.resolve()),
            "peak_training_allocated_bytes": runtime["peak_training_allocated_bytes"],
        }


def _write_config(
    cfg: SparseConsistentTrainConfig,
    context: SparseConsistentContext,
    *,
    parameter_count: int,
) -> None:
    payload = cfg.to_record(
        manifest_sha256=context.manifest_sha256,
        split_sha256=context.split_sha256,
        mask_protocol_sha256=context.mask_protocol_sha256,
        height_stats_sha256=context.height_stats_sha256,
    )
    payload["parameter_count"] = int(parameter_count)
    path = cfg.run_dir / "config.json"
    expected = canonical_json_bytes(payload)
    if path.exists() and path.read_bytes() != expected:
        raise SparseConsistentTrainerError("existing config differs from requested run")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(expected)
        os.replace(temporary, path)


def _validate_fresh_checkpoint(
    cfg: SparseConsistentTrainConfig,
    identity: CheckpointIdentity,
    checkpoint: Path,
    device: torch.device,
) -> None:
    model = build_sparse_consistent_model(cfg.arm).to(device)
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
        raise SparseConsistentTrainerError("fresh smoke reload has no optimizer progress")


def run_sparse_consistent_training(
    cfg: SparseConsistentTrainConfig,
    device: torch.device,
    *,
    preflight_only: bool = False,
    audit_all_samples: bool = False,
    resume: str = "none",
    smoke_optimizer_steps: int | None = None,
) -> dict[str, Any]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise SparseConsistentTrainerError(
            "CUBLAS_WORKSPACE_CONFIG must be ':4096:8' before training"
        )
    if resume not in {"none", "auto"} and not Path(resume).is_file():
        raise SparseConsistentTrainerError(f"resume checkpoint does not exist: {resume}")
    context = preflight_sparse_consistent(cfg, audit_all_samples=audit_all_samples)
    seed_everything(cfg.seed)
    model = build_sparse_consistent_model(cfg.arm).to(device)
    identity = build_sparse_consistent_checkpoint_identity(cfg, context, model)
    if preflight_only:
        return {
            "status": "preflight_complete",
            "protocol": SPARSE_CONSISTENT_PROTOCOL,
            "array_size": cfg.array_size,
            "arm": cfg.arm,
            "train_samples": len(context.train_dataset),
            "val_samples": len(context.val_dataset),
            "test_samples": len(context.test_dataset),
            "manifest_sha256": context.manifest_sha256,
            "split_sha256": context.split_sha256,
            "mask_protocol_sha256": context.mask_protocol_sha256,
            "condition_channels": cfg.condition_channels,
            "parameter_count": identity.parameter_count,
        }

    smoke = smoke_optimizer_steps is not None
    effective_cfg = replace(cfg, run_root=cfg.run_root / "_smoke") if smoke else cfg
    if not smoke and resume == "none" and (effective_cfg.run_dir / "last.pt").exists():
        raise SparseConsistentTrainerError("resume=none refuses an existing last.pt")
    _write_config(effective_cfg, context, parameter_count=identity.parameter_count)
    train_loader, val_loader, train_generator = build_sparse_consistent_loaders(
        effective_cfg, context
    )
    trainer = SparseConsistentTrainer(
        effective_cfg,
        model,
        train_loader,
        val_loader,
        device,
        train_generator,
        identity,
    )
    if smoke:
        result = trainer.run_smoke(int(smoke_optimizer_steps))
        _validate_fresh_checkpoint(effective_cfg, identity, Path(result["checkpoint"]), device)
        return result
    resume_path: Path | None = None
    if resume == "auto":
        candidate = effective_cfg.run_dir / "last.pt"
        resume_path = candidate if candidate.is_file() else None
    elif resume != "none":
        resume_path = Path(resume).resolve()
    if resume_path is not None:
        trainer.resume(resume_path)
    return trainer.fit()


__all__ = [
    "SparseConsistentContext",
    "SparseConsistentTrainer",
    "SparseConsistentTrainerError",
    "build_sparse_consistent_checkpoint_identity",
    "build_sparse_consistent_loaders",
    "preflight_sparse_consistent",
    "resolve_device",
    "run_sparse_consistent_training",
]
