from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from experiments.multiconfig_manifest import ARRAY_SPECS, canonical_json_bytes


SPARSE_CONSISTENT_PROTOCOL = "sparse_consistent_abcd_v1"
SPARSE_CONSISTENT_FREQUENCY_HZ = 6_700_000_000
SPARSE_CONSISTENT_STEERING_DEG = 0.0
SPARSE_CONSISTENT_SAMPLE_COUNT = 819
SPARSE_CONSISTENT_OUTPUT_SIZE = (256, 256)
SPARSE_CONSISTENT_SCENE_COUNTS = {"train": 560, "val": 80, "test": 160}
SPARSE_CONSISTENT_ARMS = (
    "environment_only",
    "concat_fullfm",
    "multiscale_fullfm",
    "multiscale_consistent",
)

ArraySize = Literal["8x8", "16x16", "32x32"]
ArmName = Literal[
    "environment_only",
    "concat_fullfm",
    "multiscale_fullfm",
    "multiscale_consistent",
]


class SparseConsistentConfigError(ValueError):
    """A sparse-consistent A/B/C/D configuration is invalid."""


@dataclass(frozen=True)
class SparseConsistentTrainConfig:
    """Frozen configuration for one arm and one array size."""

    dataset_root: Path
    manifest_path: Path
    height_stats_path: Path
    run_root: Path
    array_size: ArraySize
    arm: ArmName
    model_size: Literal["lite"] = "lite"
    max_epochs: int = 120
    early_stopping_patience: int = 20
    min_optimizer_steps: int = 1000
    seed: int = 42
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.10
    ema_decay: float = 0.995
    num_workers: int = 2
    use_amp: bool = True
    amp_dtype: str = "float16"
    euler_steps: int = 2
    cfg_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "height_stats_path", Path(self.height_stats_path))
        object.__setattr__(self, "run_root", Path(self.run_root))
        if self.array_size not in ARRAY_SPECS:
            raise SparseConsistentConfigError(f"unsupported array size: {self.array_size}")
        if self.arm not in SPARSE_CONSISTENT_ARMS:
            raise SparseConsistentConfigError(f"unsupported A/B/C/D arm: {self.arm}")
        if self.model_size != "lite":
            raise SparseConsistentConfigError("the registered pilot is Lite only")
        locked_ints = {
            "max_epochs": 120,
            "early_stopping_patience": 20,
            "min_optimizer_steps": 1000,
            "seed": 42,
            "num_workers": 2,
            "euler_steps": 2,
        }
        for name, expected in locked_ints.items():
            actual = getattr(self, name)
            if type(actual) is not int or actual != expected:
                raise SparseConsistentConfigError(f"{name} is locked to {expected}")
        locked_scalars: dict[str, Any] = {
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "warmup_ratio": 0.10,
            "ema_decay": 0.995,
            "use_amp": True,
            "amp_dtype": "float16",
            "cfg_scale": 1.0,
        }
        for name, expected in locked_scalars.items():
            actual = getattr(self, name)
            if actual != expected or type(actual) is not type(expected):
                raise SparseConsistentConfigError(f"{name} is locked to {expected!r}")

    @property
    def condition_channels(self) -> int:
        return 5 if self.arm == "concat_fullfm" else 3

    @property
    def uses_sparse_encoder(self) -> bool:
        return self.arm in {"multiscale_fullfm", "multiscale_consistent"}

    @property
    def uses_consistent_flow(self) -> bool:
        return self.arm == "multiscale_consistent"

    @property
    def train_samples(self) -> int:
        return SPARSE_CONSISTENT_SCENE_COUNTS["train"]

    @property
    def val_samples(self) -> int:
        return SPARSE_CONSISTENT_SCENE_COUNTS["val"]

    @property
    def test_samples(self) -> int:
        return SPARSE_CONSISTENT_SCENE_COUNTS["test"]

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
        return self.run_root / self.array_size / self.arm

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
            "protocol": SPARSE_CONSISTENT_PROTOCOL,
            "array_size": self.array_size,
            "arm": self.arm,
            "model_size": self.model_size,
            "condition_channels": self.condition_channels,
            "frequency_hz": SPARSE_CONSISTENT_FREQUENCY_HZ,
            "steering_deg": SPARSE_CONSISTENT_STEERING_DEG,
            "sample_count": SPARSE_CONSISTENT_SAMPLE_COUNT,
            "split_type": "scene_disjoint_single_beam",
            "scene_counts": dict(SPARSE_CONSISTENT_SCENE_COUNTS),
            "output_size": list(SPARSE_CONSISTENT_OUTPUT_SIZE),
            "mask_seed": 42,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "min_optimizer_steps": self.min_optimizer_steps,
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
            "euler_steps": self.euler_steps,
            "cfg_scale": self.cfg_scale,
        }

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.canonical_payload())).hexdigest()

    def to_record(
        self,
        *,
        manifest_sha256: str,
        split_sha256: str,
        mask_protocol_sha256: str,
        height_stats_sha256: str,
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
            "height_stats_sha256": height_stats_sha256,
            "config_sha256": self.config_sha256,
        }


__all__ = [
    "ArmName",
    "ArraySize",
    "SPARSE_CONSISTENT_ARMS",
    "SPARSE_CONSISTENT_FREQUENCY_HZ",
    "SPARSE_CONSISTENT_OUTPUT_SIZE",
    "SPARSE_CONSISTENT_PROTOCOL",
    "SPARSE_CONSISTENT_SAMPLE_COUNT",
    "SPARSE_CONSISTENT_SCENE_COUNTS",
    "SPARSE_CONSISTENT_STEERING_DEG",
    "SparseConsistentConfigError",
    "SparseConsistentTrainConfig",
]
