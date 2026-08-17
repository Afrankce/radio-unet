from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from data_loaders.cross_frequency import load_cross_frequency_height_max
from data_loaders.random_task2 import RandomTask2RadiomapDataset, random_task2_collate
from evaluation.random_task2_sampling import random_task2_euler_cfg_sample
from evaluation.sparse_task2_sampling import make_task2_sample_noise
from evaluation.sparse_task2_metrics import (
    SparseTask2MetricAccumulator,
    sparse_task2_metrics_for_json,
)
from experiments.multiconfig_manifest import ARRAY_SPECS
from experiments.provenance import sha256_file
from model.model import DiffUNet
from train import ModelEMA
from training.multiconfig_trainer import resolve_device, seed_everything, seed_worker
from training.optimization import build_optimizer_step_scheduler
from training.random_task2_config import (
    RANDOM_TASK2_COMMON_ANGLES,
    RANDOM_TASK2_OUTPUT_SIZE,
    RANDOM_TASK2_PROTOCOL,
    RANDOM_TASK2_RECORD_COUNTS,
    RANDOM_TASK2_SAMPLE_COUNT,
    RandomTask2ConfigError,
    RandomTask2TrainConfig,
)
from training.random_task2_flow import build_random_task2_pinned_flow_pair
from training.random_task2_model import build_random_task2_pinned_model
from training.sparse_task2_flow import masked_task2_velocity_mse


class RandomTask2TrainerError(RuntimeError):
    """The random-instance sparse Task 2 training contract is invalid."""


_PINNED_FM_CFG_SCALE = 1.0
_PINNED_FM_EULER_STEPS = 2


class _RegressionModelWrapper(torch.nn.Module):
    """Deterministic regression wrapper around the Lite DiffUNet."""

    def __init__(self, condition_channels: int) -> None:
        super().__init__()
        self.network = DiffUNet(
            con_channels=condition_channels,
            model_size="lite",
            activation_checkpointing=False,
        )

    def forward(self, condition: Tensor) -> Tensor:
        embedding = self.network.embed_model(condition)
        step = torch.zeros(condition.shape[0], device=condition.device, dtype=condition.dtype)
        sparse_image = condition[:, 0:1]
        return self.network(
            image=condition,
            x=sparse_image,
            pred_type="denoise",
            step=step,
            embedding=embedding,
        )


def build_random_task2_model(cfg: RandomTask2TrainConfig) -> torch.nn.Module:
    if cfg.mode == "regression":
        return _RegressionModelWrapper(cfg.condition_channels)
    return build_random_task2_pinned_model(condition_variant=cfg.variant)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class RandomTask2Trainer:
    """Deterministic sparse-reconstruction trainer for the random Task 2 split."""

    def __init__(self, cfg: RandomTask2TrainConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        seed_everything(cfg.seed)
        self.height_max = load_cross_frequency_height_max(cfg.height_stats_path)
        self.train_dataset = RandomTask2RadiomapDataset(
            dataset_root=cfg.dataset_root,
            manifest_path=cfg.manifest_path,
            split="train",
            array_size=cfg.array_size,
            height_max=self.height_max,
            variant=cfg.variant,
        )
        self.val_dataset = RandomTask2RadiomapDataset(
            dataset_root=cfg.dataset_root,
            manifest_path=cfg.manifest_path,
            split="val",
            array_size=cfg.array_size,
            height_max=self.height_max,
            variant=cfg.variant,
        )
        self.test_dataset = RandomTask2RadiomapDataset(
            dataset_root=cfg.dataset_root,
            manifest_path=cfg.manifest_path,
            split="test",
            array_size=cfg.array_size,
            height_max=self.height_max,
            variant=cfg.variant,
        )
        self.model = build_random_task2_model(cfg).to(device)
        self.ema = ModelEMA(self.model, decay=cfg.ema_decay)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = build_optimizer_step_scheduler(
            self.optimizer,
            total_steps=cfg.planned_optimizer_steps,
            warmup_steps=cfg.warmup_steps,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and cfg.use_amp
        )
        self.train_generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
        self.manifest_sha256 = sha256_file(cfg.manifest_path)
        self.parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        self.completed_epochs = 0
        self.optimizer_step = 0
        self.best_val_db_rmse = math.inf
        self.epochs_without_improvement = 0
        self.history: list[dict[str, Any]] = []
        cfg.run_dir.mkdir(parents=True, exist_ok=True)

    def _payload(self) -> dict[str, Any]:
        return {
            **self.cfg.canonical_payload(),
            "dataset_root": str(self.cfg.dataset_root.resolve()),
            "manifest_path": str(self.cfg.manifest_path.resolve()),
            "height_stats_path": str(self.cfg.height_stats_path.resolve()),
            "run_root": str(self.cfg.run_root.resolve()),
            "manifest_sha256": self.manifest_sha256,
            "config_sha256": self.cfg.config_sha256,
            "parameter_count": self.parameter_count,
        }

    def preflight(self) -> dict[str, Any]:
        if not self.cfg.manifest_path.is_file():
            raise RandomTask2TrainerError(f"manifest does not exist: {self.cfg.manifest_path}")
        if not self.cfg.height_stats_path.is_file():
            raise RandomTask2TrainerError(
                f"height stats do not exist: {self.cfg.height_stats_path}"
            )
        for split, dataset, expected in (
            ("train", self.train_dataset, RANDOM_TASK2_RECORD_COUNTS["train"]),
            ("val", self.val_dataset, RANDOM_TASK2_RECORD_COUNTS["val"]),
            ("test", self.test_dataset, RANDOM_TASK2_RECORD_COUNTS["test"]),
        ):
            if len(dataset) != expected:
                raise RandomTask2TrainerError(
                    f"{split} sample count mismatch: expected {expected}, got {len(dataset)}"
                )
        sample = self.train_dataset[0]
        if tuple(sample["condition"].shape[-2:]) != RANDOM_TASK2_OUTPUT_SIZE:
            raise RandomTask2TrainerError("condition output size is not 256x256")
        if sample["condition"].shape[0] != self.cfg.condition_channels:
            raise RandomTask2TrainerError(
                f"condition channel mismatch: expected {self.cfg.condition_channels}"
            )
        if int(sample["observation_mask"].sum().item()) != RANDOM_TASK2_SAMPLE_COUNT:
            raise RandomTask2TrainerError("observation count is not 819")
        if not bool(
            (sample["observation_mask"] & ~sample["valid_mask"]).sum().item() == 0
        ):
            raise RandomTask2TrainerError("observation mask is not a subset of valid mask")
        if bool((sample["sparse_map"].masked_select(~sample["observation_mask"])).abs().max().item() > 1e-6):
            raise RandomTask2TrainerError("sparse_map must be zero outside observations")
        _write_json(self.cfg.run_dir / "config.json", self._payload())
        return {
            "status": "preflight_complete",
            "array_size": self.cfg.array_size,
            "variant": self.cfg.variant,
            "mode": self.cfg.mode,
            "condition_channels": self.cfg.condition_channels,
            "manifest_sha256": self.manifest_sha256,
            "parameter_count": self.parameter_count,
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "test_samples": len(self.test_dataset),
        }

    def _loader(
        self,
        dataset: Dataset[Mapping[str, Any]],
        *,
        shuffle: bool,
        generator: torch.Generator | None = None,
    ) -> DataLoader[Mapping[str, Any]]:
        return DataLoader(
            dataset,
            batch_size=self.cfg.micro_batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=random_task2_collate,
            worker_init_fn=seed_worker,
            generator=generator,
            pin_memory=False,
            persistent_workers=False,
        )

    @torch.inference_mode()
    def _predict(self, batch: Mapping[str, Any], *, use_ema: bool = True) -> Tensor:
        if self.cfg.mode != "regression":
            raise RandomTask2TrainerError("_predict is only valid for regression mode")
        condition = batch["condition"].to(self.device, non_blocking=True)
        model = self.ema.ema_model if use_ema else self.model
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda" and self.cfg.use_amp,
        ):
            prediction = model(condition)
        return prediction.float().clamp(0.0, 1.0)

    @torch.inference_mode()
    def _sample_pinned_prediction(self, batch: Mapping[str, Any], *, use_ema: bool = True) -> Tensor:
        if self.cfg.mode != "pinned_fm":
            raise RandomTask2TrainerError("_sample_pinned_prediction is only valid for pinned_fm")
        condition = batch["condition"].to(self.device, non_blocking=True)
        observation_mask = batch["observation_mask"].to(self.device, non_blocking=True)
        sparse_map = batch["sparse_map"].to(self.device, non_blocking=True)
        model = self.ema.ema_model if use_ema else self.model
        noises = [
            make_task2_sample_noise(
                protocol=RANDOM_TASK2_PROTOCOL,
                array_size=self.cfg.array_size,
                split="val",
                sample_key=str(metadata["sample_key"]),
                shape=tuple(batch["target"].shape[1:]),
                base_seed=self.cfg.seed,
                dtype=batch["target"].dtype,
            )
            for metadata in batch["metadata"]
        ]
        x0 = torch.stack(noises).to(self.device, dtype=batch["target"].dtype)
        return random_task2_euler_cfg_sample(
            model,
            condition=condition,
            x0=x0,
            sparse_map=sparse_map,
            observation_mask=observation_mask,
            cfg_scale=_PINNED_FM_CFG_SCALE,
            steps=_PINNED_FM_EULER_STEPS,
            use_amp=self.cfg.use_amp,
        )

    def _run_batch(self, batch: Mapping[str, Any]) -> tuple[Tensor, float]:
        condition = batch["condition"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(self.device, non_blocking=True)
        observation_mask = batch["observation_mask"].to(self.device, non_blocking=True)
        if self.cfg.mode == "regression":
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda" and self.cfg.use_amp,
            ):
                prediction = self.model(condition)
            prediction = prediction.float()
            weight = torch.ones_like(prediction)
            observed = valid_mask & observation_mask
            missing = valid_mask & ~observation_mask
            weight = torch.where(missing, weight, torch.zeros_like(weight))
            weight = torch.where(
                observed,
                torch.full_like(weight, self.cfg.observed_loss_weight),
                weight,
            )
            error = (prediction - target).square() * weight
            loss = error.sum() / weight.sum().clamp_min(1.0)
            return loss, float(loss.detach().item())
        sparse_map = batch["sparse_map"].to(self.device, non_blocking=True)
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda" and self.cfg.use_amp,
        ):
            x0 = torch.randn_like(target)
            time_step = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
            xt, ut, loss_mask = build_random_task2_pinned_flow_pair(
                x0=x0,
                target=target,
                sparse_map=sparse_map,
                observation_mask=observation_mask,
                valid_mask=valid_mask,
                time=time_step,
            )
            embedding = self.model.embed_model(condition, sparse_map, observation_mask)
            prediction = self.model(
                image=condition,
                x=xt,
                pred_type="denoise",
                step=time_step,
                embedding=embedding,
            )
        loss = masked_task2_velocity_mse(
            prediction.float(),
            ut.float(),
            loss_mask,
        )
        return loss, float(loss.detach().item())

    def _train_epoch(self, *, max_optimizer_steps: int | None = None) -> float:
        self.model.train()
        loader = self._loader(
            self.train_dataset, shuffle=True, generator=self.train_generator
        )
        total_loss = 0.0
        micro_seen = 0
        samples_seen = 0
        self.optimizer.zero_grad(set_to_none=True)
        for batch in loader:
            loss, loss_value = self._run_batch(batch)
            self.scaler.scale(loss).backward()
            micro_seen += 1
            total_loss += loss_value * int(batch["target"].shape[0])
            samples_seen += int(batch["target"].shape[0])
            if micro_seen % self.cfg.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_step += 1
                self.ema.update(self.model)
                if (
                    max_optimizer_steps is not None
                    and self.optimizer_step >= max_optimizer_steps
                ):
                    break
        if micro_seen % self.cfg.accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_step += 1
            self.ema.update(self.model)
        return total_loss / max(samples_seen, 1)

    @torch.inference_mode()
    def _validate(self) -> dict[str, Any]:
        self.ema.ema_model.eval()
        accumulator = SparseTask2MetricAccumulator()
        loader = self._loader(self.val_dataset, shuffle=False)
        for batch in loader:
            observation_mask = batch["observation_mask"].to(self.device, non_blocking=True)
            sparse_map = batch["sparse_map"].to(self.device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            if self.cfg.mode == "regression":
                prediction = self._predict(batch, use_ema=True)
                prediction = torch.where(observation_mask, sparse_map, prediction)
            else:
                prediction = self._sample_pinned_prediction(batch, use_ema=True)
            accumulator.update(
                prediction,
                target,
                valid_mask,
                observation_mask,
                batch["metadata"],
            )
        return accumulator.compute()

    def _record(self, train_loss: float, validation: Mapping[str, Any]) -> dict[str, Any]:
        missing = validation["missing"]
        overall = validation["overall"]
        return {
            "epoch": self.completed_epochs + 1,
            "optimizer_step": self.optimizer_step,
            "train_loss": train_loss,
            "val_missing_db_rmse": missing["db_rmse"],
            "val_missing_db_mae": missing["db_mae"],
            "val_missing_nmse": missing["nmse"],
            "val_missing_psnr": missing["psnr"],
            "val_missing_ssim": missing["ssim"],
            "val_overall_db_rmse": overall["db_rmse"],
            "val_overall_psnr": overall["psnr"],
            "val_overall_ssim": overall["ssim"],
        }

    def _save(self, name: str) -> None:
        payload = {
            "schema_version": 1,
            "config": self._payload(),
            "model": self.model.state_dict(),
            "ema": self.ema.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "state": {
                "completed_epochs": self.completed_epochs,
                "optimizer_step": self.optimizer_step,
                "best_val_db_rmse": self.best_val_db_rmse,
                "epochs_without_improvement": self.epochs_without_improvement,
            },
            "rng": {
                "torch": torch.random.get_rng_state(),
                "train_generator": self.train_generator.get_state(),
            },
        }
        _atomic_checkpoint(self.cfg.run_dir / name, payload)

    def resume(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu")
        if payload.get("schema_version") != 1:
            raise RandomTask2TrainerError("unsupported checkpoint schema")
        if payload["config"]["config_sha256"] != self.cfg.config_sha256:
            raise RandomTask2TrainerError("checkpoint config hash does not match current config")
        self.model.load_state_dict(payload["model"])
        self.ema.ema_model.load_state_dict(payload["ema"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        state = payload["state"]
        self.completed_epochs = int(state["completed_epochs"])
        self.optimizer_step = int(state["optimizer_step"])
        self.best_val_db_rmse = float(state["best_val_db_rmse"])
        self.epochs_without_improvement = int(state["epochs_without_improvement"])
        torch.random.set_rng_state(payload["rng"]["torch"])
        self.train_generator.set_state(payload["rng"]["train_generator"])

    def _rebuild_csv(self) -> None:
        if not self.history:
            return
        path = self.cfg.run_dir / "metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)

    def run(
        self,
        *,
        preflight_only: bool = False,
        resume: str = "none",
        stop_after_epoch: int | None = None,
        smoke_optimizer_steps: int | None = None,
    ) -> dict[str, Any]:
        if resume not in {"none", "auto"} and not Path(resume).is_file():
            raise RandomTask2TrainerError(f"resume checkpoint does not exist: {resume}")
        preflight_result = self.preflight()
        if preflight_only:
            return preflight_result
        resume_path: Path | None = None
        if resume == "auto":
            candidate = self.cfg.run_dir / "last.pt"
            if candidate.is_file():
                resume_path = candidate
        elif resume != "none":
            resume_path = Path(resume)
        if resume_path is not None:
            self.resume(resume_path)
        max_epochs = self.cfg.max_epochs
        if smoke_optimizer_steps is not None:
            max_epochs = 1
        started = time.time()
        for _ in range(self.completed_epochs, max_epochs):
            train_loss = self._train_epoch(
                max_optimizer_steps=smoke_optimizer_steps,
            )
            self.completed_epochs += 1
            if smoke_optimizer_steps is not None:
                self._save("smoke.pt")
                return {
                    "status": "smoke_complete",
                    "run_dir": str(self.cfg.run_dir.resolve()),
                    "checkpoint": str((self.cfg.run_dir / "smoke.pt").resolve()),
                    "smoke_loss": train_loss,
                }
            validation = self._validate()
            row = self._record(train_loss, validation)
            self.history.append(row)
            self._rebuild_csv()
            missing_db_rmse = float(validation["missing"]["db_rmse"])
            if missing_db_rmse < self.best_val_db_rmse:
                self.best_val_db_rmse = missing_db_rmse
                self.epochs_without_improvement = 0
                self._save("best.pt")
            else:
                self.epochs_without_improvement += 1
            self._save("last.pt")
            if (
                stop_after_epoch is not None
                and self.completed_epochs >= stop_after_epoch
            ):
                break
            if self.epochs_without_improvement >= self.cfg.early_stopping_patience:
                break
        result = {
            "status": "complete",
            "run_dir": str(self.cfg.run_dir.resolve()),
            "completed_epochs": self.completed_epochs,
            "optimizer_step": self.optimizer_step,
            "best_val_missing_db_rmse": (
                self.best_val_db_rmse if math.isfinite(self.best_val_db_rmse) else None
            ),
            "early_stopped": (
                self.epochs_without_improvement >= self.cfg.early_stopping_patience
            ),
            "elapsed_seconds": time.time() - started,
        }
        _write_json(self.cfg.run_dir / "training_runtime.json", result)
        return result


def run_random_task2_training(
    cfg: RandomTask2TrainConfig,
    device: torch.device,
    *,
    preflight_only: bool = False,
    resume: str = "none",
    stop_after_epoch: int | None = None,
    smoke_optimizer_steps: int | None = None,
) -> dict[str, Any]:
    trainer = RandomTask2Trainer(cfg, device)
    return trainer.run(
        preflight_only=preflight_only,
        resume=resume,
        stop_after_epoch=stop_after_epoch,
        smoke_optimizer_steps=smoke_optimizer_steps,
    )


__all__ = [
    "RandomTask2Trainer",
    "RandomTask2TrainerError",
    "_RegressionModelWrapper",
    "build_random_task2_model",
    "run_random_task2_training",
]
