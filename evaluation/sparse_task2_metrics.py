from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


class SparseTask2MetricError(ValueError):
    """Task 2 metric inputs or aggregation state are invalid."""


@dataclass
class _RegionAccumulator:
    samples: int = 0
    pixel_count: int = 0
    sum_sq: float = 0.0
    sum_abs: float = 0.0
    sum_target_sq: float = 0.0
    sum_prediction: float = 0.0
    sum_target: float = 0.0
    sum_prediction_sq: float = 0.0
    sum_target_sq: float = 0.0
    sum_product: float = 0.0
    max_abs: float = 0.0
    _sample_metrics: list[dict[str, float]] = field(default_factory=list)

    def update(self, prediction: Tensor, target: Tensor, mask: Tensor) -> None:
        for index in range(prediction.shape[0]):
            selected = mask[index]
            self.samples += 1
            count = int(selected.sum().item())
            if count <= 0:
                continue
            pred = prediction[index][selected].float()
            truth = target[index][selected].float()
            if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(truth).all()):
                raise SparseTask2MetricError("metric region contains non-finite values")
            if bool(((truth < 0.0) | (truth > 1.0)).any()):
                raise SparseTask2MetricError("normalized target is outside [0,1]")
            pred = pred.clamp(0.0, 1.0)
            error = pred - truth
            sq = float(error.square().sum().item())
            absolute = float(error.abs().sum().item())
            target_sq = float(truth.square().sum().item())
            self.pixel_count += count
            self.sum_sq += sq
            self.sum_abs += absolute
            self.sum_target_sq += target_sq
            self.sum_prediction += float(pred.sum().item())
            self.sum_target += float(truth.sum().item())
            self.sum_prediction_sq += float(pred.square().sum().item())
            self.sum_target_sq += target_sq
            self.sum_product += float((pred * truth).sum().item())
            self.max_abs = max(self.max_abs, float(error.abs().max().item()))
            self._sample_metrics.append({
                "db_rmse": math.sqrt(90_000.0 * sq / count),
                "db_mae": 300.0 * absolute / count,
                "mse": sq / count,
                "nmse": sq / max(target_sq, 1e-12),
                "psnr": math.inf if sq == 0.0 else 10.0 * math.log10(count / sq),
            })

    def compute(self, *, include_audit: bool = False) -> dict[str, int | float]:
        if self.pixel_count <= 0:
            result: dict[str, int | float] = {
                "samples": self.samples,
                "pixel_count": 0,
                "db_rmse": 0.0,
                "db_mae": 0.0,
                "mse": 0.0,
                "nmse": 0.0,
                "psnr": 0.0,
                "ssim": 0.0,
                "sample_macro_db_rmse": 0.0,
                "sample_macro_db_mae": 0.0,
                "sample_macro_psnr": 0.0,
            }
        else:
            mse = self.sum_sq / self.pixel_count
            sample_count = max(len(self._sample_metrics), 1)
            result = {
                "samples": self.samples,
                "pixel_count": self.pixel_count,
                "db_rmse": math.sqrt(90_000.0 * mse),
                "db_mae": 300.0 * self.sum_abs / self.pixel_count,
                "mse": mse,
                "nmse": self.sum_sq / max(self.sum_target_sq, 1e-12),
                "psnr": math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse),
                "ssim": self._global_ssim(),
                "sample_macro_db_rmse": sum(item["db_rmse"] for item in self._sample_metrics) / sample_count,
                "sample_macro_db_mae": sum(item["db_mae"] for item in self._sample_metrics) / sample_count,
                "sample_macro_psnr": sum(item["psnr"] for item in self._sample_metrics) / sample_count,
            }
        if include_audit:
            result["max_abs_error"] = self.max_abs
            result["mean_abs_error"] = (
                self.sum_abs / self.pixel_count if self.pixel_count else 0.0
            )
        return result

    def _global_ssim(self) -> float:
        n = float(self.pixel_count)
        mean_prediction = self.sum_prediction / n
        mean_target = self.sum_target / n
        var_prediction = max(self.sum_prediction_sq / n - mean_prediction**2, 0.0)
        var_target = max(self.sum_target_sq / n - mean_target**2, 0.0)
        covariance = self.sum_product / n - mean_prediction * mean_target
        c1 = 0.01**2
        c2 = 0.03**2
        denominator = (mean_prediction**2 + mean_target**2 + c1) * (
            var_prediction + var_target + c2
        )
        if denominator == 0.0:
            return 1.0
        return ((2.0 * mean_prediction * mean_target + c1) * (2.0 * covariance + c2)) / denominator


class SparseTask2MetricAccumulator:
    """Pixel-weighted Task 2 metrics plus scene/angle audit groupings."""

    def __init__(self) -> None:
        self.overall = _RegionAccumulator()
        self.missing = _RegionAccumulator()
        self.observed = _RegionAccumulator()
        self.per_scene: dict[str, SparseTask2MetricAccumulator] = {}
        self.per_array: dict[str, SparseTask2MetricAccumulator] = {}
        self.per_angle: dict[float, SparseTask2MetricAccumulator] = {}

    @staticmethod
    def _validate(
        prediction: Tensor,
        target: Tensor,
        valid_mask: Tensor,
        observation_mask: Tensor,
    ) -> None:
        if not (
            prediction.shape == target.shape
            and prediction.shape == valid_mask.shape
            and prediction.shape == observation_mask.shape
        ):
            raise SparseTask2MetricError("prediction, target, and masks must share shape")
        if prediction.ndim != 4 or prediction.shape[1] != 1:
            raise SparseTask2MetricError("metric tensors must have shape [B,1,H,W]")
        if valid_mask.dtype is not torch.bool or observation_mask.dtype is not torch.bool:
            raise SparseTask2MetricError("metric masks must be boolean")
        if len({prediction.device, target.device, valid_mask.device, observation_mask.device}) != 1:
            raise SparseTask2MetricError("metric tensors must share a device")
        if bool((observation_mask & ~valid_mask).any()):
            raise SparseTask2MetricError("observation mask must be a subset of valid mask")
        if bool(((valid_mask & ~observation_mask).sum(dim=(-1, -2, -3)) == 0).any()):
            raise SparseTask2MetricError("each sample must have missing valid pixels")

    def update(
        self,
        prediction: Tensor,
        target: Tensor,
        valid_mask: Tensor,
        observation_mask: Tensor,
        metadata: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self._validate(prediction, target, valid_mask, observation_mask)
        if metadata is not None and len(metadata) != prediction.shape[0]:
            raise SparseTask2MetricError("metadata count must match batch size")
        self.overall.update(prediction, target, valid_mask)
        self.missing.update(prediction, target, valid_mask & ~observation_mask)
        self.observed.update(prediction, target, observation_mask)
        if metadata is None:
            return
        for index, item in enumerate(metadata):
            scene_id = str(item.get("scene_id", ""))
            array_size = str(item.get("array_size", item.get("array_name", "")))
            angle = float(item.get("steering_deg", 0.0))
            if not scene_id or not array_size or not math.isfinite(angle):
                raise SparseTask2MetricError("metadata lacks scene, array, or angle")
            slices = (
                prediction[index:index + 1],
                target[index:index + 1],
                valid_mask[index:index + 1],
                observation_mask[index:index + 1],
            )
            self.per_scene.setdefault(scene_id, SparseTask2MetricAccumulator()).update(*slices)
            self.per_array.setdefault(array_size, SparseTask2MetricAccumulator()).update(*slices)
            self.per_angle.setdefault(angle, SparseTask2MetricAccumulator()).update(*slices)

    def compute(self) -> dict[str, Any]:
        return {
            "overall": self.overall.compute(),
            "missing": self.missing.compute(),
            "observed": self.observed.compute(include_audit=True),
            "per_scene": {
                key: value.compute() for key, value in sorted(self.per_scene.items())
            },
            "per_array": {
                key: value.compute() for key, value in sorted(self.per_array.items())
            },
            "per_angle": {
                str(key): value.compute() for key, value in sorted(self.per_angle.items())
            },
        }


def sparse_task2_metrics_for_json(value: Any) -> Any:
    """Convert nested metric output to strict JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): sparse_task2_metrics_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sparse_task2_metrics_for_json(item) for item in value]
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isinf(value) and value > 0:
            return None
        if not math.isfinite(value):
            raise SparseTask2MetricError("metric value is non-finite")
        return value
    return value


__all__ = [
    "SparseTask2MetricAccumulator",
    "SparseTask2MetricError",
    "sparse_task2_metrics_for_json",
]
