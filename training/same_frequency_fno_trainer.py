from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from train import ModelEMA
from training.checkpointing import load_checkpoint_strict
from training.complex_grad_scaler import ComplexGradScaler
from training.config import InvocationControls
from training.model_factory import build_same_frequency_backbone
from training.multiconfig_trainer import (
    MultiConfigSRMTrainer,
    seed_everything,
)
from training.optimization import build_optimizer_step_scheduler
from training.same_frequency_fno_config import (
    PAPER_FNO_MODEL_SIZE,
    PaperFNOConfigError,
    PaperFNOTrainConfig,
)
from training.same_frequency_trainer import (
    SameFrequencyTrainerContractError,
    _atomic_json,
    build_same_frequency_checkpoint_identity,
    build_same_frequency_loaders,
    preflight_same_frequency,
)


def write_or_validate_fno_run_config(
    cfg: PaperFNOTrainConfig,
    controls: InvocationControls,
) -> Path:
    """Publish the immutable FNO run record, or validate an existing record."""

    path = cfg.run_dir / "config.json"
    if path.exists():
        try:
            existing = PaperFNOTrainConfig.from_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise PaperFNOConfigError(
                f"invalid existing paper-FNO run config {path}: {error}"
            ) from error
        if existing.config_sha256 != cfg.config_sha256:
            raise PaperFNOConfigError(
                "existing paper-FNO run config scientific hash mismatch"
            )
        return path
    _atomic_json(path, cfg.to_record(controls))
    return path


def _resolve_fno_resume_path(
    cfg: PaperFNOTrainConfig,
    resume: str,
) -> Path | None:
    if resume == "none":
        return None
    if resume == "auto":
        candidate = cfg.run_dir / "last.pt"
        return candidate if candidate.is_file() else None
    return Path(resume).resolve()


def _validate_fno_smoke_checkpoint_fresh(
    cfg: PaperFNOTrainConfig,
    identity,
    checkpoint: Path,
    *,
    scaler_enabled: bool,
) -> None:
    """Reload every mutable training component into a fresh FNO instance."""

    model = build_same_frequency_backbone(PAPER_FNO_MODEL_SIZE)
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
    scaler = ComplexGradScaler("cuda", enabled=scaler_enabled)
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
        raise SameFrequencyTrainerContractError(
            "fresh paper-FNO smoke reload has no optimizer progress"
        )


def run_same_frequency_fno_training(
    cfg: PaperFNOTrainConfig,
    controls: InvocationControls,
    device: torch.device,
    *,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Run the locked same-frequency protocol with only the backbone replaced."""

    if not isinstance(cfg, PaperFNOTrainConfig):
        raise PaperFNOConfigError("cfg must be a PaperFNOTrainConfig")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise SameFrequencyTrainerContractError(
            "CUBLAS_WORKSPACE_CONFIG must be ':4096:8' before training"
        )
    context = preflight_same_frequency(cfg)
    if preflight_only:
        return {
            "status": "preflight_complete",
            "backbone": cfg.backbone,
            "model_size": cfg.model_size,
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
    effective_cfg = (
        cfg.with_run_root(cfg.run_root / "_smoke") if smoke else cfg
    )
    if (
        not smoke
        and controls.resume == "none"
        and (cfg.run_dir / "last.pt").exists()
    ):
        raise SameFrequencyTrainerContractError(
            "resume=none refuses an existing paper-FNO last.pt"
        )
    write_or_validate_fno_run_config(effective_cfg, controls)

    seed_everything(cfg.seed)
    model = build_same_frequency_backbone(cfg.model_size).to(device)
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
        scaler=ComplexGradScaler(
            "cuda",
            enabled=device.type == "cuda" and cfg.use_amp,
        ),
    )
    if smoke:
        result = trainer.run_smoke(int(controls.smoke_optimizer_steps))
        _validate_fno_smoke_checkpoint_fresh(
            effective_cfg,
            identity,
            Path(result["checkpoint"]),
            scaler_enabled=device.type == "cuda" and cfg.use_amp,
        )
        return result

    resume_path = _resolve_fno_resume_path(effective_cfg, controls.resume)
    if resume_path is not None:
        trainer.resume(resume_path)
    return trainer.fit(stop_after_epoch=controls.stop_after_epoch)


__all__ = [
    "run_same_frequency_fno_training",
    "write_or_validate_fno_run_config",
]
