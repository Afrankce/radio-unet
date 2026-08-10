from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import torch

from experiments.cross_frequency import TEST_FREQUENCY_HZ
from experiments.multiconfig_manifest import ARRAY_SPECS, canonical_json_bytes
from training.config import InvocationControls


CONDITION_CHANNELS = 3
MODEL_SIZES = ("lite", "large")
TRAIN_SAMPLES = 560
VAL_SAMPLES = 80
TEST_SAMPLES = 160
ArraySize = Literal["8x8", "16x16", "32x32"]


class SameFrequencyTrainConfigError(ValueError):
    """A same-frequency control attempts to change a locked experiment value."""


@dataclass(frozen=True)
class SameFrequencyTrainConfig:
    dataset_root: Path
    manifest_path: Path
    height_stats_path: Path
    run_root: Path
    array_size: ArraySize = "8x8"
    beam_id: int = 0
    model_size: Literal["lite", "large"] = "lite"
    train_scale: float = 1.0
    train_frequency_hz: int = TEST_FREQUENCY_HZ
    val_frequency_hz: int = TEST_FREQUENCY_HZ
    test_frequency_hz: int = TEST_FREQUENCY_HZ
    steering_deg: float = 0.0
    train_samples: int = TRAIN_SAMPLES
    val_samples: int = VAL_SAMPLES
    test_samples: int = TEST_SAMPLES
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
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "height_stats_path", Path(self.height_stats_path))
        object.__setattr__(self, "run_root", Path(self.run_root))
        if self.array_size not in ARRAY_SPECS:
            raise SameFrequencyTrainConfigError(
                f"array_size must be one of {tuple(ARRAY_SPECS)}, got {self.array_size!r}"
            )
        if self.model_size not in MODEL_SIZES:
            raise SameFrequencyTrainConfigError(
                f"model_size must be one of {MODEL_SIZES}"
            )
        if type(self.beam_id) is not int or self.beam_id < 0:
            raise SameFrequencyTrainConfigError("beam_id must be a non-negative integer")
        if self.train_scale != 1.0 or type(self.train_scale) is not float:
            raise SameFrequencyTrainConfigError("train_scale is locked to 1.0")
        locked: dict[str, Any] = {
            "train_frequency_hz": TEST_FREQUENCY_HZ,
            "val_frequency_hz": TEST_FREQUENCY_HZ,
            "test_frequency_hz": TEST_FREQUENCY_HZ,
            "steering_deg": 0.0,
            "train_samples": TRAIN_SAMPLES,
            "val_samples": VAL_SAMPLES,
            "test_samples": TEST_SAMPLES,
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
                raise SameFrequencyTrainConfigError(
                    f"{name} is locked to {expected!r}, got {actual!r}"
                )
        if not math.isfinite(self.steering_deg):
            raise SameFrequencyTrainConfigError("steering_deg must be finite")

    @property
    def condition_channels(self) -> int:
        return CONDITION_CHANNELS

    @property
    def array_rows(self) -> int:
        return ARRAY_SPECS[self.array_size].rows

    @property
    def array_cols(self) -> int:
        return ARRAY_SPECS[self.array_size].cols

    @property
    def tx_elements(self) -> int:
        return ARRAY_SPECS[self.array_size].tx_elements

    @property
    def train_scene_count(self) -> int:
        return TRAIN_SAMPLES

    @property
    def micro_batch_size(self) -> int:
        return 2 if self.model_size == "lite" else 1

    @property
    def accumulation_steps(self) -> int:
        return 28 if self.model_size == "lite" else 56

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
    def activation_checkpointing(self) -> bool:
        return self.model_size == "large"

    @property
    def run_dir(self) -> Path:
        return self.run_root

    def scientific_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment": "same_frequency_6.7_single_beam",
            "array_size": self.array_size,
            "model_size": self.model_size,
            "condition_channels": CONDITION_CHANNELS,
            "array_rows": self.array_rows,
            "array_cols": self.array_cols,
            "tx_elements": self.tx_elements,
            "frequency_hz": self.test_frequency_hz,
            "train_frequency_hz": self.train_frequency_hz,
            "val_frequency_hz": self.val_frequency_hz,
            "test_frequency_hz": self.test_frequency_hz,
            "beam_id": self.beam_id,
            "steering_deg": self.steering_deg,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "test_samples": self.test_samples,
            "train_scene_count": self.train_scene_count,
            "train_scale": self.train_scale,
            "scene_split": "scene_split_seed42",
            "scene_split_claim": "same_frequency_scene_disjoint_control",
            "normalization": {
                "target_db_floor": -300.0,
                "target_db_ceiling": 0.0,
                "height": "train_only_height_stats",
                "beam_map": "global_db_interval",
            },
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

    def to_record(self, invocation: InvocationControls | None = None) -> dict[str, Any]:
        controls = invocation or InvocationControls()
        return {
            **self.scientific_payload(),
            "dataset_root": str(self.dataset_root.resolve()),
            "manifest_path": str(self.manifest_path.resolve()),
            "height_stats_path": str(self.height_stats_path.resolve()),
            "run_root": str(self.run_root.resolve()),
            "config_sha256": self.config_sha256,
            "invocation": controls.to_dict(),
        }

    @classmethod
    def from_json(cls, text: str) -> "SameFrequencyTrainConfig":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise SameFrequencyTrainConfigError(
                f"run config is not valid JSON: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise SameFrequencyTrainConfigError("run config root must be an object")
        try:
            invocation = InvocationControls.from_dict(payload["invocation"])
            path_fields = {"dataset_root", "manifest_path", "height_stats_path", "run_root"}
            constructor = {
                field.name: payload.get(field.name)
                for field in fields(cls)
                if field.name not in path_fields
            }
            template = cls(
                dataset_root=payload["dataset_root"],
                manifest_path=payload["manifest_path"],
                height_stats_path=payload["height_stats_path"],
                run_root=payload["run_root"],
                **constructor,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SameFrequencyTrainConfigError(
                f"invalid run config fields: {error}"
            ) from error
        expected = template.to_record(invocation)
        if set(payload) != set(expected):
            raise SameFrequencyTrainConfigError(
                "run config keys mismatch: "
                f"unknown={sorted(set(payload) - set(expected))}, "
                f"missing={sorted(set(expected) - set(payload))}"
            )
        for key, value in expected.items():
            if payload[key] != value:
                raise SameFrequencyTrainConfigError(
                    f"run config {key} mismatch: expected {value!r}, got {payload[key]!r}"
                )
        return template


__all__ = [
    "ArraySize",
    "SameFrequencyTrainConfig",
    "SameFrequencyTrainConfigError",
]
