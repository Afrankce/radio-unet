from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from evaluation.radiomap_metrics import metrics_for_json


class SparseMetricError(ValueError):
    """Sparse reconstruction metrics cannot be computed safely."""


@dataclass
class _RegionTotals:
    samples: int = 0
    pixel_count: int = 0
    sum_sq_norm: float = 0.0
    sum_abs_norm: float = 0.0
    sum_target_sq_norm: float = 0.0
    sum_prediction: float = 0.0
    sum_target: float = 0.0
    sum_prediction_sq: float = 0.0
    sum_target_sq: float = 0.0
    sum_product: float = 0.0
    max_abs_norm: float = 0.0

    def update(self, prediction: Tensor, target: Tensor, mask: Tensor) -> None:
        count = int(mask.sum().item())
        if count <= 0:
            self.samples += int(prediction.shape[0])
            return
        pred = prediction[mask].double().clamp(0.0, 1.0)
        tgt = target[mask].double()
        if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(tgt).all()):
            raise SparseMetricError("metric region contains non-finite values")
        if bool(((tgt < 0.0) | (tgt > 1.0)).any()):
            raise SparseMetricError("normalized target is outside [0,1]")
        error = pred - tgt
        self.samples += int(prediction.shape[0])
        self.pixel_count += count
        self.sum_sq_norm += float(error.square().sum().item())
        self.sum_abs_norm += float(error.abs().sum().item())
        self.sum_target_sq_norm += float(tgt.square().sum().item())
        self.sum_prediction += float(pred.sum().item())
        self.sum_target += float(tgt.sum().item())
        self.sum_prediction_sq += float(pred.square().sum().item())
        self.sum_target_sq += float(tgt.square().sum().item())
        self.sum_product += float((pred * tgt).sum().item())
        self.max_abs_norm = max(self.max_abs_norm, float(error.abs().max().item()))

    def compute(self, *, include_observed_audit: bool = False) -> dict[str, int | float]:
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
            }
            if include_observed_audit:
                result["max_abs_error"] = 0.0
                result["mean_abs_error"] = 0.0
            return result
        mse = self.sum_sq_norm / self.pixel_count
        psnr = math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
        result: dict[str, int | float] = {
            "samples": self.samples,
            "pixel_count": self.pixel_count,
            "db_rmse": math.sqrt(90_000.0 * mse),
            "db_mae": 300.0 * self.sum_abs_norm / self.pixel_count,
            "mse": mse,
            "nmse": self.sum_sq_norm / max(self.sum_target_sq_norm, 1e-12),
            "psnr": psnr,
            "ssim": self._global_ssim(),
        }
        if include_observed_audit:
            result["max_abs_error"] = self.max_abs_norm
            result["mean_abs_error"] = self.sum_abs_norm / self.pixel_count
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
        return ((2 * mean_prediction * mean_target + c1) * (2 * covariance + c2)) / denominator


class SparseMetricAccumulator:
    """Aggregate sparse reconstruction metrics over missing/observed/valid regions."""

    def __init__(self) -> None:
        self._regions = {
            "missing": _RegionTotals(),
            "observed": _RegionTotals(),
            "overall_valid": _RegionTotals(),
        }

    @staticmethod
    def _validate(
        prediction: Tensor,
        target: Tensor,
        valid_mask: Tensor,
        observation_mask: Tensor,
    ) -> None:
        if (
            prediction.shape != target.shape
            or prediction.shape != valid_mask.shape
            or prediction.shape != observation_mask.shape
        ):
            raise SparseMetricError("prediction, target, valid_mask, and observation_mask shapes must match")
        if prediction.ndim != 4 or prediction.shape[1] != 1:
            raise SparseMetricError("metric tensors must have shape [B,1,H,W]")
        if valid_mask.dtype != torch.bool or observation_mask.dtype != torch.bool:
            raise SparseMetricError("valid_mask and observation_mask must be boolean")
        if len({prediction.device, target.device, valid_mask.device, observation_mask.device}) != 1:
            raise SparseMetricError("metric tensors must share a device")
        if int((observation_mask & ~valid_mask).sum().item()) != 0:
            raise SparseMetricError("observation_mask must be a subset of valid_mask")
        if int((valid_mask & ~observation_mask).sum().item()) == 0:
            raise SparseMetricError("missing region contains zero pixels")

    def update(
        self,
        prediction: Tensor,
        target: Tensor,
        valid_mask: Tensor,
        observation_mask: Tensor,
    ) -> None:
        self._validate(prediction, target, valid_mask, observation_mask)
        self._regions["missing"].update(prediction, target, valid_mask & ~observation_mask)
        self._regions["observed"].update(prediction, target, observation_mask)
        self._regions["overall_valid"].update(prediction, target, valid_mask)

    def compute(self) -> dict[str, dict[str, int | float]]:
        return {
            "missing": self._regions["missing"].compute(),
            "observed": self._regions["observed"].compute(include_observed_audit=True),
            "overall_valid": self._regions["overall_valid"].compute(),
        }


class SparseGroupedMetricAccumulators:
    def __init__(self) -> None:
        self.overall = SparseMetricAccumulator()
        self.per_scene: dict[str, SparseMetricAccumulator] = {}
        self.per_array: dict[str, SparseMetricAccumulator] = {}
        self.per_variant: dict[str, SparseMetricAccumulator] = {}

    def update(
        self,
        prediction: Tensor,
        target: Tensor,
        valid_mask: Tensor,
        observation_mask: Tensor,
        metadata: Sequence[Mapping[str, Any]],
    ) -> None:
        if len(metadata) != prediction.shape[0]:
            raise SparseMetricError("metadata count must match batch size")
        self.overall.update(prediction, target, valid_mask, observation_mask)
        for index, item in enumerate(metadata):
            scene = str(item.get("scene_id", ""))
            array = str(item.get("array_size", item.get("array_name", "")))
            variant = str(item.get("variant", "beam_masked"))
            if not scene or not array or not variant:
                raise SparseMetricError(f"incomplete sparse metadata at index {index}")
            slices = (
                prediction[index : index + 1],
                target[index : index + 1],
                valid_mask[index : index + 1],
                observation_mask[index : index + 1],
            )
            self.per_scene.setdefault(scene, SparseMetricAccumulator()).update(*slices)
            self.per_array.setdefault(array, SparseMetricAccumulator()).update(*slices)
            self.per_variant.setdefault(variant, SparseMetricAccumulator()).update(*slices)

    def compute(self) -> dict[str, Any]:
        return {
            "overall": self.overall.compute(),
            "per_scene": {key: value.compute() for key, value in sorted(self.per_scene.items())},
            "per_array": {key: value.compute() for key, value in sorted(self.per_array.items())},
            "per_variant": {key: value.compute() for key, value in sorted(self.per_variant.items())},
        }


def sparse_metrics_for_json(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (
            sparse_metrics_for_json(value)
            if isinstance(value, Mapping)
            else value
        )
        for key, value in metrics_for_json(metrics).items()
    } if all(not isinstance(v, Mapping) for v in metrics.values()) else {
        key: sparse_metrics_for_json(value) if isinstance(value, Mapping) else value
        for key, value in metrics.items()
    }


__all__ = [
    "SparseGroupedMetricAccumulators",
    "SparseMetricAccumulator",
    "SparseMetricError",
    "sparse_metrics_for_json",
]
