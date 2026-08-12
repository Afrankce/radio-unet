from __future__ import annotations

import csv
import itertools
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from data_loaders.same_frequency import load_same_frequency_height_max
from data_loaders.sparse_same_frequency import (
    SparseSameFrequencyRadiomapDataset,
    sparse_collate,
)
from evaluation.radioflow_sampling import make_sample_noise
from evaluation.sparse_metrics import SparseMetricAccumulator
from evaluation.sparse_sampling import sparse_euler_cfg_sample
from experiments.multiconfig_manifest import canonical_json_bytes, load_manifest_jsonl
from experiments.provenance import sha256_file
from train import ModelEMA
from training.checkpointing import (
    CheckpointIdentity,
    TrainerState,
    load_checkpoint_strict,
    save_checkpoint_atomic,
)
from training.config import InvocationControls
from training.model_factory import SPARSE_PARAMETER_COUNTS, build_sparse_radioflow
from training.multiconfig_trainer import resolve_device, seed_everything, seed_worker
from training.optimization import build_optimizer_step_scheduler, run_accumulation_window
from training.sparse_config import (
    FORMAL_RUN_VARIANT,
    SCENE_COUNTS,
    SPARSE_EXPERIMENT,
    SparseSameFrequencyTrainConfig,
)
from training.sparse_flow import build_masked_flow_pair


class SparseTrainerContractError(RuntimeError):
    """Sparse training or preflight state violates the locked protocol."""


@dataclass(frozen=True)
class SparseContext:
    train_dataset: Any
    val_dataset: Any
    test_dataset: Any
    manifest_path: Path
    manifest_sha256: str
    height_stats_sha256: str
    mask_protocol_sha256: str
    height_max: float


def _mask_protocol_sha256(cfg: SparseSameFrequencyTrainConfig) -> str:
    import hashlib

    payload = {
        "schema_version": 1,
        "experiment": SPARSE_EXPERIMENT,
        "variant": cfg.variant,
        "observation_ratio": cfg.observation_ratio,
        "mask_seed": cfg.mask_seed,
        "condition_noise_seed": cfg.condition_noise_seed,
        "train_mask_mode": cfg.train_mask_mode,
        "flow_pair": "observed_fixed_missing_valid_velocity_v1",
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def preflight_sparse_same_frequency(cfg: SparseSameFrequencyTrainConfig) -> SparseContext:
    if cfg.variant != FORMAL_RUN_VARIANT:
        raise SparseTrainerContractError("formal sparse CLI only accepts 'beam_masked'")
    manifest_path = cfg.manifest_path.resolve()
    stats_path = cfg.height_stats_path.resolve()
    if not manifest_path.is_file():
        raise SparseTrainerContractError(f"sparse manifest is missing: {manifest_path}")
    if not stats_path.is_file():
        raise SparseTrainerContractError(f"height statistics are missing: {stats_path}")
    split_path = manifest_path.parent / "scene_split_seed42.json"
    if not split_path.is_file():
        raise SparseTrainerContractError(f"fixed scene split is missing: {split_path}")
    try:
        records = load_manifest_jsonl(manifest_path)
        height_max = load_same_frequency_height_max(stats_path, split_path=split_path)
    except Exception as error:
        raise SparseTrainerContractError(f"sparse data contract could not be loaded: {error}") from error
    if len(records) != sum(SCENE_COUNTS.values()):
        raise SparseTrainerContractError(
            f"manifest must contain {sum(SCENE_COUNTS.values())} records, got {len(records)}"
        )
    datasets: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        try:
            datasets[split] = SparseSameFrequencyRadiomapDataset(
                dataset_root=cfg.dataset_root.resolve(),
                manifest_path=manifest_path,
                split=split,  # type: ignore[arg-type]
                array_size=cfg.array_size,
                variant=cfg.variant,
                height_max=height_max,
                observation_ratio=cfg.observation_ratio,
                mask_seed=cfg.mask_seed,
                condition_noise_seed=cfg.condition_noise_seed,
                expected_counts=SCENE_COUNTS,
            )
        except Exception as error:
            raise SparseTrainerContractError(f"cannot construct {split} sparse dataset: {error}") from error
        sample = datasets[split][0]
        if tuple(sample["condition"].shape) != (5, 256, 256):
            raise SparseTrainerContractError(f"{split} condition shape must be (5,256,256)")
        if int(sample["missing_mask"].sum().item()) <= 0:
            raise SparseTrainerContractError(f"{split} sample has no missing pixels")
    return SparseContext(
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        test_dataset=datasets["test"],
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        height_stats_sha256=sha256_file(stats_path),
        mask_protocol_sha256=_mask_protocol_sha256(cfg),
        height_max=height_max,
    )


def build_sparse_loaders(
    cfg: SparseSameFrequencyTrainConfig,
    context: SparseContext,
) -> tuple[DataLoader, DataLoader, torch.Generator]:
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    train_loader = DataLoader(
        context.train_dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=cfg.num_workers,
        collate_fn=sparse_collate,
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
        collate_fn=sparse_collate,
        worker_init_fn=seed_worker,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )
    return train_loader, val_loader, generator


def build_sparse_checkpoint_identity(
    cfg: SparseSameFrequencyTrainConfig,
    context: SparseContext,
    model: torch.nn.Module,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        experiment=SPARSE_EXPERIMENT,
        array_size=cfg.array_size,
        variant=cfg.variant,
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
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


class SparseSameFrequencyTrainer:
    def __init__(
        self,
        cfg: SparseSameFrequencyTrainConfig,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        train_generator: torch.Generator,
        identity: CheckpointIdentity,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.train_generator = train_generator
        self.identity = identity
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
        observed_map = batch["observed_map"].to(self.device, non_blocking=True)
        observation_mask = batch["observation_mask"].to(self.device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(self.device, non_blocking=True)
        enabled = self.device.type == "cuda" and self.cfg.use_amp
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=enabled,
        ):
            z0 = torch.randn_like(target)
            t = torch.rand(target.shape[0], device=target.device)
            xt, ut, missing_mask = build_masked_flow_pair(
                initial_noise=z0,
                target=target,
                observed_map=observed_map,
                observation_mask=observation_mask,
                valid_mask=valid_mask,
                time=t,
            )
            embedding = model.embed_model(condition)
            predicted = model(
                image=condition,
                x=xt,
                pred_type="denoise",
                step=t,
                embedding=embedding,
            )
        return predicted, ut, missing_mask

    def _run_window(self, window: tuple[Mapping[str, Any], ...]):
        window_for_loss = []
        for batch in window:
            copied = dict(batch)
            copied["valid_mask"] = batch["missing_mask"]
            window_for_loss.append(copied)
        result = run_accumulation_window(
            tuple(window_for_loss),
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

    def train_one_epoch(self, epoch: int) -> dict[str, int | float]:
        dataset = getattr(self.train_loader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        self.model.train()
        numerator = 0.0
        missing_pixels = 0
        observed_pixels = 0
        valid_pixels = 0
        samples = 0
        iterator = iter(self.train_loader)
        while True:
            window = tuple(itertools.islice(iterator, self.cfg.accumulation_steps))
            if not window:
                break
            result = self._run_window(window)
            numerator += result.squared_error_sum
            missing_pixels += result.valid_pixels
            observed_pixels += sum(int(batch["observation_mask"].sum().item()) for batch in window)
            valid_pixels += sum(int(batch["valid_mask"].sum().item()) for batch in window)
            samples += result.samples
        if missing_pixels <= 0:
            raise SparseTrainerContractError("training epoch produced no missing pixels")
        return {
            "loss_missing": numerator / missing_pixels,
            "missing_pixels": missing_pixels,
            "observed_pixels": observed_pixels,
            "valid_pixels": valid_pixels,
            "samples": samples,
        }

    def validate(self) -> dict[str, float]:
        network = self.ema.ema_model
        network.eval()
        accumulator = SparseMetricAccumulator()
        for batch in self.val_loader:
            condition = batch["condition"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            observed_map = batch["observed_map"].to(self.device, non_blocking=True)
            observation_mask = batch["observation_mask"].to(self.device, non_blocking=True)
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
            prediction = sparse_euler_cfg_sample(
                network,
                condition,
                observed_map,
                observation_mask,
                noise,
                cfg_scale=1.0,
                steps=2,
                use_amp=self.cfg.use_amp,
            )
            accumulator.update(prediction, target, valid_mask, observation_mask)
        metrics = accumulator.compute()
        return {
            "missing_db_rmse": float(metrics["missing"]["db_rmse"]),
            "missing_db_mae": float(metrics["missing"]["db_mae"]),
            "observed_max_abs_error": float(metrics["observed"]["max_abs_error"]),
            "overall_psnr": float(metrics["overall_valid"]["psnr"]),
            "overall_ssim": float(metrics["overall_valid"]["ssim"]),
        }

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
        return state

    def fit(self) -> dict[str, Any]:
        while self.completed_epochs < self.cfg.max_epochs:
            train = self.train_one_epoch(self.completed_epochs)
            validation = self.validate()
            rmse = float(validation["missing_db_rmse"])
            improved = rmse < self.best_val_db_rmse
            if improved:
                self.best_val_db_rmse = rmse
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            self.completed_epochs += 1
            self.next_epoch_index = self.completed_epochs
            row = {
                "epoch": self.completed_epochs,
                "optimizer_step": self.optimizer_step,
                "loss_missing": float(train["loss_missing"]),
                "missing_pixels": int(train["missing_pixels"]),
                "observed_pixels": int(train["observed_pixels"]),
                "valid_pixels": int(train["valid_pixels"]),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "ema": True,
                **validation,
            }
            self.history.append(row)
            _write_metrics_csv(self.cfg.run_dir / "metrics.csv", self.history)
            if improved:
                self._save("best.pt")
            self._save("last.pt")
            if self.epochs_without_improvement >= self.cfg.early_stopping_patience:
                break
        return {
            "status": "complete" if self.completed_epochs >= self.cfg.max_epochs else "paused",
            "run_dir": str(self.cfg.run_dir.resolve()),
            "completed_epochs": self.completed_epochs,
            "optimizer_step": self.optimizer_step,
            "best_val_db_rmse": self.best_val_db_rmse,
        }

    def run_smoke(self, optimizer_steps: int) -> dict[str, Any]:
        if optimizer_steps <= 0:
            raise SparseTrainerContractError("smoke optimizer_steps must be positive")
        successful = 0
        while successful < optimizer_steps:
            for batch in self.train_loader:
                result = self._run_window((batch,))
                if result.optimizer_ran:
                    successful += 1
                if successful >= optimizer_steps:
                    break
        self.completed_epochs = max(self.completed_epochs, 1)
        self.next_epoch_index = self.completed_epochs
        checkpoint = self._save("smoke.pt", smoke=True)
        return {
            "status": "smoke_complete",
            "run_dir": str(self.cfg.run_dir.resolve()),
            "optimizer_steps": successful,
            "checkpoint": str(checkpoint.resolve()),
        }


def _write_sparse_config(
    cfg: SparseSameFrequencyTrainConfig,
    controls: InvocationControls,
    context: SparseContext,
) -> None:
    payload = {
        **cfg.to_record(
            manifest_sha256=context.manifest_sha256,
            height_stats_sha256=context.height_stats_sha256,
        ),
        "invocation": controls.to_dict(),
        "mask_protocol_sha256": context.mask_protocol_sha256,
    }
    path = cfg.run_dir / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != expected:
            raise SparseTrainerContractError("existing sparse config differs from requested run")
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        output.write(expected)
    os.replace(temporary, path)


def run_sparse_same_frequency_training(
    cfg: SparseSameFrequencyTrainConfig,
    controls: InvocationControls,
    device: torch.device,
    *,
    preflight_only: bool = False,
) -> dict[str, Any]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise SparseTrainerContractError("CUBLAS_WORKSPACE_CONFIG must be ':4096:8' before training")
    context = preflight_sparse_same_frequency(cfg)
    if preflight_only:
        return {
            "status": "preflight_complete",
            "manifest": str(context.manifest_path),
            "manifest_sha256": context.manifest_sha256,
            "array_size": cfg.array_size,
            "variant": cfg.variant,
            "train_samples": len(context.train_dataset),
            "val_samples": len(context.val_dataset),
            "test_samples": len(context.test_dataset),
            "height_max": context.height_max,
            "condition_channels": cfg.condition_channels,
            "parameter_count": SPARSE_PARAMETER_COUNTS[(cfg.model_size, cfg.condition_channels)],
        }
    smoke = controls.smoke_optimizer_steps is not None
    effective_cfg = replace(cfg, run_root=cfg.run_root / "_smoke") if smoke else cfg
    if not smoke and controls.resume == "none" and (effective_cfg.run_dir / "last.pt").exists():
        raise SparseTrainerContractError("resume=none refuses an existing sparse last.pt")
    _write_sparse_config(effective_cfg, controls, context)
    seed_everything(cfg.seed)
    model = build_sparse_radioflow(variant=cfg.variant, model_size=cfg.model_size).to(device)
    train_loader, val_loader, train_generator = build_sparse_loaders(cfg, context)
    identity = build_sparse_checkpoint_identity(cfg, context, model)
    trainer = SparseSameFrequencyTrainer(
        effective_cfg,
        model,
        train_loader,
        val_loader,
        device,
        train_generator,
        identity,
    )
    if smoke:
        return trainer.run_smoke(int(controls.smoke_optimizer_steps))
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
    return trainer.fit()


__all__ = [
    "SparseContext",
    "SparseSameFrequencyTrainer",
    "SparseTrainerContractError",
    "build_sparse_checkpoint_identity",
    "build_sparse_loaders",
    "preflight_sparse_same_frequency",
    "resolve_device",
    "run_sparse_same_frequency_training",
]
