from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from experiments.multiconfig_manifest import ARRAY_SPECS, canonical_json_bytes


SINGLEBEAM_TASK2_PROTOCOL = "singlebeam_feature5_samples819"
SINGLEBEAM_TASK2_FREQUENCY_HZ = 6_700_000_000
SINGLEBEAM_TASK2_STEERING_DEG = 0.0
SINGLEBEAM_TASK2_SAMPLE_COUNT = 819
SINGLEBEAM_TASK2_CONDITION_CHANNELS = 5
SINGLEBEAM_TASK2_OUTPUT_SIZE = (256, 256)
SINGLEBEAM_TASK2_SCENE_COUNTS = {"train": 560, "val": 80, "test": 160}

ArraySize = Literal["8x8", "16x16", "32x32"]
SplitName = Literal["train", "val", "test"]


class SparseTask2ConfigError(ValueError):
    """A sparse Task 2 configuration violates the locked experiment contract."""


@dataclass(frozen=True)
class SparseTask2DatasetConfig:
    """Immutable loader configuration for the mandatory single-beam protocol."""

    dataset_root: Path
    manifest_path: Path
    split: SplitName
    array_size: ArraySize
    height_max: float
    expected_counts: Mapping[str, int] | None = None
    mask_seed: int = 42
    sample_count: int = SINGLEBEAM_TASK2_SAMPLE_COUNT
    output_size: tuple[int, int] = SINGLEBEAM_TASK2_OUTPUT_SIZE

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        if self.split not in ("train", "val", "test"):
            raise SparseTask2ConfigError(f"invalid split: {self.split!r}")
        if self.array_size not in ARRAY_SPECS:
            raise SparseTask2ConfigError(
                f"array_size must be one of {tuple(ARRAY_SPECS)}, got {self.array_size!r}"
            )
        if not math.isfinite(float(self.height_max)) or float(self.height_max) <= 0.0:
            raise SparseTask2ConfigError("height_max must be finite and positive")
        if type(self.mask_seed) is not int or self.mask_seed != 42:
            raise SparseTask2ConfigError("mask_seed is locked to 42")
        if type(self.sample_count) is not int or self.sample_count != SINGLEBEAM_TASK2_SAMPLE_COUNT:
            raise SparseTask2ConfigError(
                f"sample_count is locked to {SINGLEBEAM_TASK2_SAMPLE_COUNT}"
            )
        if tuple(self.output_size) != SINGLEBEAM_TASK2_OUTPUT_SIZE:
            raise SparseTask2ConfigError(
                f"output_size is locked to {SINGLEBEAM_TASK2_OUTPUT_SIZE}"
            )
        if self.expected_counts is not None:
            expected = dict(self.expected_counts)
            if set(expected) != {"train", "val", "test"} or any(
                type(value) is not int or value <= 0 for value in expected.values()
            ):
                raise SparseTask2ConfigError(
                    "expected_counts must contain positive train/val/test integers"
                )

    @property
    def condition_channels(self) -> int:
        return SINGLEBEAM_TASK2_CONDITION_CHANNELS

    @property
    def counts(self) -> dict[str, int]:
        return dict(self.expected_counts or SINGLEBEAM_TASK2_SCENE_COUNTS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": SINGLEBEAM_TASK2_PROTOCOL,
            "frequency_hz": SINGLEBEAM_TASK2_FREQUENCY_HZ,
            "steering_deg": SINGLEBEAM_TASK2_STEERING_DEG,
            "condition_channels": self.condition_channels,
            "dataset_root": str(self.dataset_root.resolve()),
            "manifest_path": str(self.manifest_path.resolve()),
            "split": self.split,
            "array_size": self.array_size,
            "height_max": float(self.height_max),
            "expected_counts": self.counts,
            "mask_seed": self.mask_seed,
            "sample_count": self.sample_count,
            "output_size": list(self.output_size),
        }


@dataclass(frozen=True)
class SparseTask2TrainConfig:
    """Locked training configuration for one independent single-beam run."""

    dataset_root: Path
    manifest_path: Path
    height_stats_path: Path
    run_root: Path
    array_size: ArraySize
    model_size: Literal["lite"] = "lite"
    condition_variant: Literal["feature5_mask"] = "feature5_mask"
    max_epochs: int = 1000
    early_stopping_patience: int = 20
    seed: int = 42
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.10
    ema_decay: float = 0.999
    num_workers: int = 2
    use_amp: bool = True
    amp_dtype: str = "float16"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "height_stats_path", Path(self.height_stats_path))
        object.__setattr__(self, "run_root", Path(self.run_root))
        if self.array_size not in ARRAY_SPECS:
            raise SparseTask2ConfigError(
                f"array_size must be one of {tuple(ARRAY_SPECS)}, got {self.array_size!r}"
            )
        if self.model_size != "lite" or type(self.model_size) is not str:
            raise SparseTask2ConfigError("Task 2 model_size is locked to 'lite'")
        if self.condition_variant != "feature5_mask":
            raise SparseTask2ConfigError(
                "single-beam control condition_variant is locked to 'feature5_mask'"
            )
        locked_ints = {
            "max_epochs": 1000,
            "early_stopping_patience": 20,
            "seed": 42,
            "num_workers": 2,
        }
        for name, expected in locked_ints.items():
            actual = getattr(self, name)
            if type(actual) is not int or actual != expected:
                raise SparseTask2ConfigError(f"{name} is locked to {expected}")
        locked_scalars: dict[str, Any] = {
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "warmup_ratio": 0.10,
            "ema_decay": 0.999,
            "use_amp": True,
            "amp_dtype": "float16",
        }
        for name, expected in locked_scalars.items():
            actual = getattr(self, name)
            if actual != expected or type(actual) is not type(expected):
                raise SparseTask2ConfigError(f"{name} is locked to {expected!r}")

    @property
    def condition_channels(self) -> int:
        return SINGLEBEAM_TASK2_CONDITION_CHANNELS

    @property
    def train_samples(self) -> int:
        return SINGLEBEAM_TASK2_SCENE_COUNTS["train"]

    @property
    def val_samples(self) -> int:
        return SINGLEBEAM_TASK2_SCENE_COUNTS["val"]

    @property
    def test_samples(self) -> int:
        return SINGLEBEAM_TASK2_SCENE_COUNTS["test"]

    @property
    def micro_batch_size(self) -> int:
        return 2

    @property
    def accumulation_steps(self) -> int:
        return 28

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
        return self.run_root / self.array_size

    def precision_runtime(self, device: Any) -> dict[str, Any]:
        enabled = self.use_amp and getattr(device, "type", None) == "cuda"
        return {
            "amp_requested": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "autocast_enabled": enabled,
            "scaler_enabled": enabled,
        }

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "protocol": SINGLEBEAM_TASK2_PROTOCOL,
            "array_size": self.array_size,
            "condition_variant": self.condition_variant,
            "model_size": self.model_size,
            "condition_channels": self.condition_channels,
            "frequency_hz": SINGLEBEAM_TASK2_FREQUENCY_HZ,
            "steering_deg": SINGLEBEAM_TASK2_STEERING_DEG,
            "sample_count": SINGLEBEAM_TASK2_SAMPLE_COUNT,
            "split_type": "scene_disjoint_single_beam",
            "scene_counts": dict(SINGLEBEAM_TASK2_SCENE_COUNTS),
            "output_size": list(SINGLEBEAM_TASK2_OUTPUT_SIZE),
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
        }

    @property
    def config_sha256(self) -> str:
        import hashlib

        return hashlib.sha256(canonical_json_bytes(self.canonical_payload())).hexdigest()

    def to_record(
        self,
        *,
        manifest_sha256: str,
        split_sha256: str,
        mask_protocol_sha256: str,
    ) -> dict[str, Any]:
        return {
            **self.canonical_payload(),
            "dataset_root": str(self.dataset_root.resolve()),
            "manifest_path": str(self.manifest_path.resolve()),
            "height_stats_path": str(self.height_stats_path.resolve()),
            "run_root": str(self.run_root.resolve()),
            "manifest_sha256": manifest_sha256,
            "split_sha256": split_sha256,
            "mask_protocol_sha256": mask_protocol_sha256,
            "config_sha256": self.config_sha256,
        }


__all__ = [
    "ArraySize",
    "SINGLEBEAM_TASK2_CONDITION_CHANNELS",
    "SINGLEBEAM_TASK2_FREQUENCY_HZ",
    "SINGLEBEAM_TASK2_OUTPUT_SIZE",
    "SINGLEBEAM_TASK2_PROTOCOL",
    "SINGLEBEAM_TASK2_SAMPLE_COUNT",
    "SINGLEBEAM_TASK2_SCENE_COUNTS",
    "SINGLEBEAM_TASK2_STEERING_DEG",
    "SparseTask2ConfigError",
    "SparseTask2DatasetConfig",
    "SparseTask2TrainConfig",
    "SplitName",
]
