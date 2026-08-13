from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from experiments.multiconfig_manifest import ARRAY_SPECS


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
    "SplitName",
]
