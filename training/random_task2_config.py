from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from experiments.multiconfig_manifest import ARRAY_SPECS, canonical_json_bytes


RANDOM_TASK2_PROTOCOL = "random_sparse_feature_samples819"
RANDOM_TASK2_FREQUENCY_HZ = 6_700_000_000
RANDOM_TASK2_COMMON_ANGLES = (-28.0, -21.0, -14.0, -7.0, 0.0, 7.0, 14.0, 21.0)
RANDOM_TASK2_SAMPLE_COUNT = 819
RANDOM_TASK2_OUTPUT_SIZE = (256, 256)
RANDOM_TASK2_RECORD_COUNTS = {"train": 4480, "val": 640, "test": 1280}
RANDOM_TASK2_SPLIT_TYPE = "random_instance"

ArraySize = Literal["8x8", "16x16", "32x32"]
ConditionVariant = Literal["feature4", "feature5_mask"]
TrainMode = Literal["regression", "pinned_fm"]


class RandomTask2ConfigError(ValueError):
    """A random-split sparse Task 2 configuration is invalid."""


@dataclass(frozen=True)
class RandomTask2TrainConfig:
    """Configuration for one random-instance sparse Task 2 run."""

    dataset_root: Path
    manifest_path: Path
    height_stats_path: Path
    run_root: Path
    array_size: ArraySize
    variant: ConditionVariant = "feature4"
    mode: TrainMode = "regression"
    model_size: Literal["lite"] = "lite"
    max_epochs: int = 100
    early_stopping_patience: int = 25
    seed: int = 42
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.10
    ema_decay: float = 0.995
    num_workers: int = 2
    use_amp: bool = True
    amp_dtype: str = "float16"
    micro_batch_size: int = 2
    accumulation_steps: int = 28
    observed_loss_weight: float = 100.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "height_stats_path", Path(self.height_stats_path))
        object.__setattr__(self, "run_root", Path(self.run_root))
        if self.array_size not in ARRAY_SPECS:
            raise RandomTask2ConfigError(f"unsupported array size: {self.array_size}")
        if self.variant not in ("feature4", "feature5_mask"):
            raise RandomTask2ConfigError(f"unsupported condition variant: {self.variant}")
        if self.mode != "regression":
            raise RandomTask2ConfigError("v3 pilot currently supports mode='regression' only")
        if self.model_size != "lite":
            raise RandomTask2ConfigError("random Task 2 is Lite only for the pilot")
        for name in ("max_epochs", "early_stopping_patience", "num_workers"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise RandomTask2ConfigError(f"{name} must be a positive integer")
        for name in ("micro_batch_size", "accumulation_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise RandomTask2ConfigError(f"{name} must be a positive integer")
        if not math.isfinite(float(self.learning_rate)) or float(self.learning_rate) <= 0:
            raise RandomTask2ConfigError("learning_rate must be finite and positive")
        if not math.isfinite(float(self.weight_decay)) or float(self.weight_decay) < 0:
            raise RandomTask2ConfigError("weight_decay must be finite and non-negative")
        if not math.isfinite(float(self.warmup_ratio)) or not 0.0 <= float(self.warmup_ratio) <= 1.0:
            raise RandomTask2ConfigError("warmup_ratio must be in [0,1]")
        if not math.isfinite(float(self.ema_decay)) or not 0.0 < float(self.ema_decay) <= 1.0:
            raise RandomTask2ConfigError("ema_decay must be in (0,1]")
        if not math.isfinite(float(self.observed_loss_weight)) or float(self.observed_loss_weight) <= 0:
            raise RandomTask2ConfigError("observed_loss_weight must be finite and positive")

    @property
    def condition_channels(self) -> int:
        return 4 if self.variant == "feature4" else 5

    @property
    def counts(self) -> dict[str, int]:
        return dict(RANDOM_TASK2_RECORD_COUNTS)

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.accumulation_steps

    @property
    def optimizer_steps_per_epoch(self) -> int:
        return math.ceil(
            math.ceil(self.counts["train"] / self.micro_batch_size)
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
        return self.run_root / self.array_size / self.variant / self.mode

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "protocol": RANDOM_TASK2_PROTOCOL,
            "split_type": RANDOM_TASK2_SPLIT_TYPE,
            "array_size": self.array_size,
            "variant": self.variant,
            "mode": self.mode,
            "model_size": self.model_size,
            "condition_channels": self.condition_channels,
            "frequency_hz": RANDOM_TASK2_FREQUENCY_HZ,
            "beam_angles": list(RANDOM_TASK2_COMMON_ANGLES),
            "sample_count": RANDOM_TASK2_SAMPLE_COUNT,
            "record_counts": self.counts,
            "output_size": list(RANDOM_TASK2_OUTPUT_SIZE),
            "mask_seed": 42,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "ema_decay": self.ema_decay,
            "num_workers": self.num_workers,
            "use_amp": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "micro_batch_size": self.micro_batch_size,
            "accumulation_steps": self.accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
            "planned_optimizer_steps": self.planned_optimizer_steps,
            "warmup_steps": self.warmup_steps,
            "observed_loss_weight": self.observed_loss_weight,
        }

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.canonical_payload())).hexdigest()


__all__ = [
    "ArraySize",
    "ConditionVariant",
    "RANDOM_TASK2_COMMON_ANGLES",
    "RANDOM_TASK2_FREQUENCY_HZ",
    "RANDOM_TASK2_OUTPUT_SIZE",
    "RANDOM_TASK2_PROTOCOL",
    "RANDOM_TASK2_RECORD_COUNTS",
    "RANDOM_TASK2_SAMPLE_COUNT",
    "RANDOM_TASK2_SPLIT_TYPE",
    "RandomTask2ConfigError",
    "RandomTask2TrainConfig",
    "TrainMode",
]
