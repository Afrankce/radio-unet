from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, Mapping

import torch

from experiments.multiconfig_manifest import canonical_json_bytes


ARRAY_SIZES = ("8x8", "16x16", "32x32")
MODEL_SIZES = ("lite", "large")
COMMON_ANGLES_DEG = (-28.0, -21.0, -14.0, -7.0, 0.0, 7.0, 14.0, 21.0)
FREQUENCY_HZ = 6_700_000_000
CONDITION_CHANNELS = 3
TRAIN_SAMPLES = 4_480
TRAIN_SCALES = (1.0, 0.1)
TRAIN_SCENE_COUNT = 560
VAL_SAMPLES = 640
TEST_SAMPLES = 1_280


class TrainConfigError(ValueError):
    """A run attempts to change or misrepresent a locked benchmark control."""


@dataclass(frozen=True)
class InvocationControls:
    resume: str = "none"
    stop_after_epoch: int | None = None
    smoke_optimizer_steps: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.resume, str) or not self.resume:
            raise TrainConfigError("resume must be 'none', 'auto', or a checkpoint path")
        if self.stop_after_epoch is not None:
            if (
                isinstance(self.stop_after_epoch, bool)
                or not isinstance(self.stop_after_epoch, int)
                or not 1 <= self.stop_after_epoch <= 200
            ):
                raise TrainConfigError("stop_after_epoch must be between 1 and 200")
        if self.smoke_optimizer_steps is not None:
            if (
                isinstance(self.smoke_optimizer_steps, bool)
                or not isinstance(self.smoke_optimizer_steps, int)
                or self.smoke_optimizer_steps <= 0
            ):
                raise TrainConfigError("smoke_optimizer_steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "resume": self.resume,
            "stop_after_epoch": self.stop_after_epoch,
            "smoke_optimizer_steps": self.smoke_optimizer_steps,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InvocationControls":
        expected = {"resume", "stop_after_epoch", "smoke_optimizer_steps"}
        if set(payload) != expected:
            raise TrainConfigError(
                "invocation keys mismatch: "
                f"unknown={sorted(set(payload) - expected)}, "
                f"missing={sorted(expected - set(payload))}"
            )
        return cls(
            resume=payload["resume"],
            stop_after_epoch=payload["stop_after_epoch"],
            smoke_optimizer_steps=payload["smoke_optimizer_steps"],
        )


@dataclass(frozen=True)
class MultiConfigTrainConfig:
    array_size: Literal["8x8", "16x16", "32x32"]
    model_size: Literal["lite", "large"]
    dataset_root: Path
    manifest_dir: Path
    run_root: Path
    train_scale: float = 1.0
    seed: int = 42
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.10
    ema_decay: float = 0.999
    max_epochs: int = 1000
    early_stopping_patience: int = 20
    num_workers: int = 2
    resolution: int = 256
    use_amp: bool = True
    amp_dtype: str = "float16"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "manifest_dir", Path(self.manifest_dir))
        object.__setattr__(self, "run_root", Path(self.run_root))
        if self.array_size not in ARRAY_SIZES:
            raise TrainConfigError(f"array_size must be one of {ARRAY_SIZES}")
        if self.model_size not in MODEL_SIZES:
            raise TrainConfigError(f"model_size must be one of {MODEL_SIZES}")
        if self.train_scale not in TRAIN_SCALES or type(self.train_scale) is not float:
            raise TrainConfigError(f"train_scale must be one of {TRAIN_SCALES}")
        locked: dict[str, Any] = {
            "seed": 42,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "warmup_ratio": 0.10,
            "ema_decay": 0.999,
            "max_epochs": 1000,
            "early_stopping_patience": 20,
            "num_workers": 2,
            "resolution": 256,
            "use_amp": True,
            "amp_dtype": "float16",
        }
        for name, expected in locked.items():
            actual = getattr(self, name)
            if actual != expected or type(actual) is not type(expected):
                raise TrainConfigError(
                    f"{name} is locked to {expected!r}, got {actual!r}"
                )

    @property
    def train_samples(self) -> int:
        return int(round(TRAIN_SAMPLES * self.train_scale))

    @property
    def micro_batch_size(self) -> int:
        return 2 if self.model_size == "lite" else 1

    @property
    def accumulation_steps(self) -> int:
        # Reference RadioFlow recipe: ~500 train samples, batch 64 => ~8 optimizer
        # steps per epoch. With 448 train samples (0.1x), effective batch 56 gives
        # exactly 8 steps/epoch: lite 2x28, large 1x56.
        return 28 if self.model_size == "lite" else 56

    @property
    def activation_checkpointing(self) -> bool:
        return self.model_size == "large"

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.accumulation_steps

    @property
    def optimizer_steps_per_epoch(self) -> int:
        return math.ceil(
            math.ceil(self.train_samples / self.micro_batch_size)
            / self.accumulation_steps
        )

    @property
    def planned_optimizer_steps(self) -> int:
        return self.optimizer_steps_per_epoch * self.max_epochs

    @property
    def warmup_steps(self) -> int:
        return int(self.planned_optimizer_steps * self.warmup_ratio)

    @property
    def run_dir(self) -> Path:
        return self.run_root / self.array_size / self.model_size

    def scientific_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "array_size": self.array_size,
            "model_size": self.model_size,
            "condition_channels": CONDITION_CHANNELS,
            "frequency_hz": FREQUENCY_HZ,
            "common_angles_deg": list(COMMON_ANGLES_DEG),
            "train_samples": self.train_samples,
            "full_train_samples": TRAIN_SAMPLES,
            "train_scale": self.train_scale,
            "train_scene_count": TRAIN_SCENE_COUNT,
            "train_subsample_rule": "sorted_first_n_scenes",
            "val_samples": VAL_SAMPLES,
            "test_samples": TEST_SAMPLES,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "ema_decay": self.ema_decay,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "num_workers": self.num_workers,
            "resolution": self.resolution,
            "use_amp": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "micro_batch_size": self.micro_batch_size,
            "accumulation_steps": self.accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
            "planned_optimizer_steps": self.planned_optimizer_steps,
            "warmup_steps": self.warmup_steps,
            "activation_checkpointing": self.activation_checkpointing,
        }

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.scientific_payload())).hexdigest()

    def precision_runtime(self, device: torch.device) -> dict[str, Any]:
        enabled = self.use_amp and device.type == "cuda"
        return {
            "amp_requested": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "autocast_enabled": enabled,
            "scaler_enabled": enabled,
        }

    def to_record(
        self,
        invocation: InvocationControls | None = None,
    ) -> dict[str, Any]:
        controls = invocation or InvocationControls()
        return {
            **self.scientific_payload(),
            "dataset_root": str(self.dataset_root.resolve()),
            "manifest_dir": str(self.manifest_dir.resolve()),
            "run_root": str(self.run_root.resolve()),
            "config_sha256": self.config_sha256,
            "invocation": controls.to_dict(),
        }

    @classmethod
    def from_json(cls, text: str) -> "MultiConfigTrainConfig":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise TrainConfigError(f"run config is not valid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise TrainConfigError("run config root must be an object")
        template = cls(
            array_size=payload.get("array_size", ""),
            model_size=payload.get("model_size", ""),
            dataset_root=payload.get("dataset_root", "."),
            manifest_dir=payload.get("manifest_dir", "."),
            run_root=payload.get("run_root", "."),
            **{
                field.name: payload.get(field.name)
                for field in fields(cls)
                if field.name
                not in {
                    "array_size",
                    "model_size",
                    "dataset_root",
                    "manifest_dir",
                    "run_root",
                }
            },
        )
        expected_record = template.to_record(
            InvocationControls.from_dict(payload.get("invocation", {}))
        )
        if set(payload) != set(expected_record):
            raise TrainConfigError(
                "run config keys mismatch: "
                f"unknown={sorted(set(payload) - set(expected_record))}, "
                f"missing={sorted(set(expected_record) - set(payload))}"
            )
        for key, expected in expected_record.items():
            if payload[key] != expected:
                raise TrainConfigError(
                    f"run config {key} mismatch: expected {expected!r}, "
                    f"got {payload[key]!r}"
                )
        return template

