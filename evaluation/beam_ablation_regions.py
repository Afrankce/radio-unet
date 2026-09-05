from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


ARRAY_SIZES = ("8x8", "16x16", "32x32")
BUILDING_RADIUS = 5
MINIMUM_PRACTICAL_DB = 0.25


def building_region_masks(
    height: Tensor,
    valid_mask: Tensor,
    *,
    radius: int = BUILDING_RADIUS,
) -> tuple[Tensor, Tensor]:
    """Partition valid pixels into a fixed-radius building band and open region."""

    if (
        not isinstance(height, Tensor)
        or not isinstance(valid_mask, Tensor)
        or height.ndim != 4
        or valid_mask.ndim != 4
        or height.shape != valid_mask.shape
        or height.shape[1] != 1
    ):
        raise ValueError("height and valid_mask must share shape [B,1,H,W]")
    if valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be boolean")
    if height.device != valid_mask.device:
        raise ValueError("height and valid_mask must share a device")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer")

    kernel = 2 * radius + 1
    building = height > 0
    expanded = F.max_pool2d(
        building.to(dtype=torch.float32),
        kernel_size=kernel,
        stride=1,
        padding=radius,
    ).to(dtype=torch.bool)
    near_building = valid_mask & expanded
    open_region = valid_mask & ~expanded
    return near_building, open_region


@dataclass
class RegionErrorAccumulator:
    """Pixel-weighted normalized error accumulator with dB-domain reporting."""

    sum_squared_normalized_error: float = 0.0
    sum_absolute_normalized_error: float = 0.0
    n_pixels: int = 0
    n_samples: int = 0

    def update(self, prediction: Tensor, target: Tensor, mask: Tensor) -> None:
        if (
            not isinstance(prediction, Tensor)
            or not isinstance(target, Tensor)
            or not isinstance(mask, Tensor)
            or prediction.ndim != 4
            or prediction.shape != target.shape
            or prediction.shape != mask.shape
            or prediction.shape[1] != 1
        ):
            raise ValueError("prediction, target, and mask must share shape [B,1,H,W]")
        if mask.dtype is not torch.bool:
            raise ValueError("mask must be boolean")
        if prediction.device != target.device or prediction.device != mask.device:
            raise ValueError("prediction, target, and mask must share a device")
        selected = (prediction.clamp(0.0, 1.0).double() - target.double())[mask]
        if selected.numel():
            self.sum_squared_normalized_error += selected.square().sum().item()
            self.sum_absolute_normalized_error += selected.abs().sum().item()
            self.n_pixels += int(selected.numel())
        self.n_samples += int(prediction.shape[0])

    def compute(self) -> dict[str, int | float]:
        if self.n_samples <= 0 or self.n_pixels <= 0:
            raise ValueError("region accumulator has no evaluated pixels")
        mse = self.sum_squared_normalized_error / self.n_pixels
        return {
            "n_samples": self.n_samples,
            "n_pixels": self.n_pixels,
            "db_rmse": 300.0 * math.sqrt(mse),
            "db_mae": (
                300.0 * self.sum_absolute_normalized_error / self.n_pixels
            ),
        }


def classify_ablation(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold_db: float = MINIMUM_PRACTICAL_DB,
) -> str:
    """Apply the frozen majority-of-three decision rule."""

    if (
        len(rows) != len(ARRAY_SIZES)
        or {str(row.get("array_size")) for row in rows} != set(ARRAY_SIZES)
        or not math.isfinite(float(threshold_db))
        or float(threshold_db) <= 0.0
    ):
        raise ValueError("rows must contain the three arrays and a positive threshold")
    normalized: list[tuple[float, float]] = []
    for row in rows:
        try:
            delta_near = float(row["delta_near_db_rmse"])
            regional_effect = float(row["regional_effect_db"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("ablation row lacks finite decision values") from error
        if not math.isfinite(delta_near) or not math.isfinite(regional_effect):
            raise ValueError("ablation decision values must be finite")
        normalized.append((delta_near, regional_effect))

    threshold = float(threshold_db)
    supporting = [
        (delta_near, effect)
        for delta_near, effect in normalized
        if effect <= -threshold
    ]
    if len(supporting) >= 2 and any(
        delta_near <= -threshold for delta_near, _effect in supporting
    ):
        return "supports_beam_shortcut_hypothesis"
    if sum(delta_near >= threshold for delta_near, _effect in normalized) >= 2:
        return "refutes_directional_beam_shortcut_hypothesis"
    if all(
        abs(delta_near) < threshold and abs(effect) < threshold
        for delta_near, effect in normalized
    ):
        return "approximately_redundant_in_fixed_single_beam_protocol"
    return "heterogeneous_or_inconclusive"


def _load_prediction_artifacts(directory: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"prediction directory does not exist: {root}")
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                required = {"prediction", "target", "valid_mask", "metadata_json"}
                if not required.issubset(archive.files):
                    raise ValueError(f"prediction artifact lacks fields: {path}")
                metadata = json.loads(str(archive["metadata_json"].item()))
                prediction = torch.from_numpy(np.asarray(archive["prediction"]).copy())
                target = torch.from_numpy(np.asarray(archive["target"]).copy())
                valid_mask = torch.from_numpy(
                    np.asarray(archive["valid_mask"]).astype(bool, copy=True)
                )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read prediction artifact {path}: {error}") from error
        if not isinstance(metadata, dict) or not isinstance(metadata.get("scene_id"), str):
            raise ValueError(f"prediction artifact has invalid metadata: {path}")
        scene_id = metadata["scene_id"]
        if scene_id in artifacts:
            raise ValueError(f"duplicate prediction scene: {scene_id}")
        artifacts[scene_id] = {
            "prediction": prediction,
            "target": target,
            "valid_mask": valid_mask,
            "metadata": metadata,
        }
    if not artifacts:
        raise ValueError(f"prediction directory is empty: {root}")
    return artifacts


def compare_array_predictions(
    array_size: str,
    dataset: Sequence[Mapping[str, Any]],
    full_prediction_dir: str | Path,
    beam_zero_prediction_dir: str | Path,
    *,
    radius: int = BUILDING_RADIUS,
) -> dict[str, Any]:
    """Compare Full and Beam-zero artifacts on frozen building/open regions."""

    if array_size not in ARRAY_SIZES or not dataset:
        raise ValueError("array_size must be registered and dataset must be non-empty")
    full_artifacts = _load_prediction_artifacts(full_prediction_dir)
    zero_artifacts = _load_prediction_artifacts(beam_zero_prediction_dir)
    samples = [dataset[index] for index in range(len(dataset))]
    scene_ids = [str(sample["metadata"]["scene_id"]) for sample in samples]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("dataset contains duplicate scene IDs")
    expected_scenes = set(scene_ids)
    if set(full_artifacts) != expected_scenes or set(zero_artifacts) != expected_scenes:
        raise ValueError("prediction scene IDs do not exactly match the dataset")

    accumulators = {
        variant: {
            region: RegionErrorAccumulator()
            for region in ("overall", "near_building", "open")
        }
        for variant in ("full", "beam_zero")
    }
    for sample in samples:
        condition = sample["condition"]
        target = sample["target"]
        valid_mask = sample["valid_mask"]
        metadata = sample["metadata"]
        if (
            not isinstance(condition, Tensor)
            or condition.ndim != 3
            or condition.shape[0] != 3
            or not isinstance(target, Tensor)
            or not isinstance(valid_mask, Tensor)
            or target.ndim != 3
            or valid_mask.shape != target.shape
        ):
            raise ValueError("dataset sample violates the RadioFlow tensor contract")
        scene_id = str(metadata["scene_id"])
        target_batch = target.unsqueeze(0)
        valid_batch = valid_mask.to(dtype=torch.bool).unsqueeze(0)
        height_batch = condition[1:2].unsqueeze(0)
        near, open_region = building_region_masks(
            height_batch,
            valid_batch,
            radius=radius,
        )
        for variant, artifacts in (
            ("full", full_artifacts),
            ("beam_zero", zero_artifacts),
        ):
            artifact = artifacts[scene_id]
            prediction = artifact["prediction"].unsqueeze(0)
            artifact_target = artifact["target"].unsqueeze(0)
            artifact_mask = artifact["valid_mask"].unsqueeze(0)
            if (
                not torch.equal(artifact_target, target_batch)
                or not torch.equal(artifact_mask, valid_batch)
            ):
                raise ValueError(f"{variant} artifact target/mask mismatch for {scene_id}")
            declared_variant = artifact["metadata"].get("condition_variant")
            if variant == "beam_zero" and declared_variant != "beam_zero":
                raise ValueError(f"beam-zero artifact identity mismatch for {scene_id}")
            if variant == "full" and declared_variant not in (None, "full"):
                raise ValueError(f"full artifact identity mismatch for {scene_id}")
            accumulators[variant]["overall"].update(
                prediction,
                target_batch,
                valid_batch,
            )
            accumulators[variant]["near_building"].update(
                prediction,
                target_batch,
                near,
            )
            accumulators[variant]["open"].update(
                prediction,
                target_batch,
                open_region,
            )

    metrics = {
        variant: {
            region: accumulator.compute()
            for region, accumulator in regions.items()
        }
        for variant, regions in accumulators.items()
    }
    delta_near = (
        metrics["beam_zero"]["near_building"]["db_rmse"]
        - metrics["full"]["near_building"]["db_rmse"]
    )
    delta_open = (
        metrics["beam_zero"]["open"]["db_rmse"]
        - metrics["full"]["open"]["db_rmse"]
    )
    return {
        "array_size": array_size,
        "n_samples": len(scene_ids),
        "building_radius_pixels": radius,
        "full": metrics["full"],
        "beam_zero": metrics["beam_zero"],
        "delta_near_db_rmse": delta_near,
        "delta_open_db_rmse": delta_open,
        "regional_effect_db": delta_near - delta_open,
    }


__all__ = [
    "ARRAY_SIZES",
    "BUILDING_RADIUS",
    "MINIMUM_PRACTICAL_DB",
    "RegionErrorAccumulator",
    "building_region_masks",
    "classify_ablation",
    "compare_array_predictions",
]
