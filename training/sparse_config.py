from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

from experiments.multiconfig_manifest import canonical_json_bytes


SPARSE_EXPERIMENT = "sparse_same_frequency_6.7_single_beam"
ARRAY_SIZES = ("8x8", "16x16", "32x32")
VARIANTS = ("no_beam_masked", "beam_masked")
MODEL_SIZE = "lite"
FREQUENCY_HZ = 6_700_000_000
STEERING_DEG = 0.0
SCENE_COUNTS = {"train": 560, "val": 80, "test": 160}
OUTPUT_SIZE = (256, 256)
OBSERVATION_RATIO = 0.05
MAX_EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 20
CONDITION_CHANNELS = {
    "no_beam_masked": 4,
    "beam_masked": 5,
}
FIXED_SPLIT_ID = "scene_split_seed42"
FORMAL_RUN_VARIANT = "beam_masked"

ArraySize = Literal["8x8", "16x16", "32x32"]
SparseVariant = Literal["no_beam_masked", "beam_masked"]


class SparseConfigError(ValueError):
    """A sparse same-frequency run attempts to violate the locked contract."""


def variant_to_condition_channels(variant: str) -> int:
    try:
        return CONDITION_CHANNELS[variant]
    except KeyError as error:
        raise SparseConfigError(
            f"variant must be one of {VARIANTS}, got {variant!r}"
        ) from error


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise SparseConfigError(f"{name} must be a 64-character SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as error:
        raise SparseConfigError(f"{name} must be hex") from error
    return value


@dataclass(frozen=True)
class SparseSameFrequencyTrainConfig:
    dataset_root: Path
    manifest_path: Path
    height_stats_path: Path
    run_root: Path
    array_size: ArraySize
    variant: SparseVariant
    model_size: Literal["lite"] = MODEL_SIZE
    observation_ratio: float = OBSERVATION_RATIO
    mask_seed: int = 42
    condition_noise_seed: int = 4242
    train_mask_mode: str = "epoch_deterministic"
    max_epochs: int = MAX_EPOCHS
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "height_stats_path", Path(self.height_stats_path))
        object.__setattr__(self, "run_root", Path(self.run_root))
        if self.array_size not in ARRAY_SIZES:
            raise SparseConfigError(
                f"array_size must be one of {ARRAY_SIZES}, got {self.array_size!r}"
            )
        if self.variant not in VARIANTS:
            raise SparseConfigError(
                f"variant must be one of {VARIANTS}, got {self.variant!r}"
            )
        if self.model_size != MODEL_SIZE or type(self.model_size) is not str:
            raise SparseConfigError(f"model_size is locked to {MODEL_SIZE!r}")
        if (
            not isinstance(self.observation_ratio, float)
            or not math.isfinite(self.observation_ratio)
            or self.observation_ratio != OBSERVATION_RATIO
        ):
            raise SparseConfigError(
                f"observation_ratio is locked to {OBSERVATION_RATIO!r}"
            )
        if self.train_mask_mode != "epoch_deterministic":
            raise SparseConfigError(
                "train_mask_mode is locked to 'epoch_deterministic'"
            )
        locked_ints = {
            "mask_seed": 42,
            "condition_noise_seed": 4242,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        }
        for name, expected in locked_ints.items():
            actual = getattr(self, name)
            if type(actual) is not int or actual != expected:
                raise SparseConfigError(f"{name} is locked to {expected!r}")

    @property
    def condition_channels(self) -> int:
        return variant_to_condition_channels(self.variant)

    @property
    def formal_run_variant(self) -> str:
        return FORMAL_RUN_VARIANT

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment": SPARSE_EXPERIMENT,
            "array_size": self.array_size,
            "variant": self.variant,
            "formal_run_variant": self.formal_run_variant,
            "model_size": self.model_size,
            "condition_channels": self.condition_channels,
            "frequency_hz": FREQUENCY_HZ,
            "steering_deg": STEERING_DEG,
            "scene_counts": dict(SCENE_COUNTS),
            "output_size": list(OUTPUT_SIZE),
            "observation_ratio": self.observation_ratio,
            "mask_seed": self.mask_seed,
            "condition_noise_seed": self.condition_noise_seed,
            "train_mask_mode": self.train_mask_mode,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "split_id": FIXED_SPLIT_ID,
        }

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.canonical_payload())).hexdigest()

    def to_record(
        self,
        *,
        manifest_sha256: str,
        height_stats_sha256: str,
    ) -> dict[str, Any]:
        return {
            **self.canonical_payload(),
            "dataset_root": str(self.dataset_root.resolve()),
            "manifest_path": str(self.manifest_path.resolve()),
            "height_stats_path": str(self.height_stats_path.resolve()),
            "run_root": str(self.run_root.resolve()),
            "manifest_sha256": _require_sha256("manifest_sha256", manifest_sha256),
            "height_stats_sha256": _require_sha256(
                "height_stats_sha256", height_stats_sha256
            ),
            "config_sha256": self.config_sha256,
        }

    def checkpoint_identity_payload(
        self,
        *,
        manifest_sha256: str,
        height_stats_sha256: str,
        checkpoint_sha256: str,
        archive_sha256: str,
        dataset_revision: str,
        radioflow_upstream_base: str,
        git_commit: str,
        parameter_count: int,
    ) -> dict[str, Any]:
        if type(parameter_count) is not int or parameter_count <= 0:
            raise SparseConfigError("parameter_count must be a positive integer")
        return {
            **self.canonical_payload(),
            "manifest_sha256": _require_sha256("manifest_sha256", manifest_sha256),
            "height_stats_sha256": _require_sha256(
                "height_stats_sha256", height_stats_sha256
            ),
            "checkpoint_sha256": _require_sha256(
                "checkpoint_sha256", checkpoint_sha256
            ),
            "archive_sha256": _require_sha256("archive_sha256", archive_sha256),
            "dataset_revision": str(dataset_revision),
            "radioflow_upstream_base": str(radioflow_upstream_base),
            "git_commit": str(git_commit),
            "parameter_count": parameter_count,
            "config_sha256": self.config_sha256,
        }

    @classmethod
    def from_json(cls, text: str) -> "SparseSameFrequencyTrainConfig":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise SparseConfigError(f"run config is not valid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise SparseConfigError("run config root must be an object")
        try:
            path_fields = {
                "dataset_root",
                "manifest_path",
                "height_stats_path",
                "run_root",
            }
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
            expected = template.to_record(
                manifest_sha256=payload["manifest_sha256"],
                height_stats_sha256=payload["height_stats_sha256"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SparseConfigError(f"invalid run config fields: {error}") from error
        if set(payload) != set(expected):
            raise SparseConfigError(
                "run config keys mismatch: "
                f"unknown={sorted(set(payload) - set(expected))}, "
                f"missing={sorted(set(expected) - set(payload))}"
            )
        for key, value in expected.items():
            if payload[key] != value:
                if key == "condition_channels":
                    raise SparseConfigError(
                        "run config variant mismatch: "
                        f"variant={payload.get('variant')!r} implies "
                        f"condition_channels={expected['condition_channels']!r}, "
                        f"got {payload[key]!r}"
                    )
                raise SparseConfigError(
                    f"run config {key} mismatch: expected {value!r}, got {payload[key]!r}"
                )
        return template


__all__ = [
    "ARRAY_SIZES",
    "CONDITION_CHANNELS",
    "EARLY_STOPPING_PATIENCE",
    "FIXED_SPLIT_ID",
    "FORMAL_RUN_VARIANT",
    "FREQUENCY_HZ",
    "MAX_EPOCHS",
    "MODEL_SIZE",
    "OBSERVATION_RATIO",
    "OUTPUT_SIZE",
    "SCENE_COUNTS",
    "SPARSE_EXPERIMENT",
    "STEERING_DEG",
    "VARIANTS",
    "ArraySize",
    "SparseConfigError",
    "SparseSameFrequencyTrainConfig",
    "SparseVariant",
    "variant_to_condition_channels",
]
