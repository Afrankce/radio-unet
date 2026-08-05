from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from data_loaders.multiconfig import denormalize_db, normalize_db


SSIM_WINDOW_SIZE = 11


class MetricInputError(ValueError):
    """A prediction, target, mask, or metadata batch cannot be evaluated safely."""


def normalized_to_db(values: Tensor) -> Tensor:
    return denormalize_db(values)


def db_to_normalized(values: Tensor) -> Tensor:
    return normalize_db(values)


def complete_window_mask(
    valid_mask: Tensor,
    *,
    window_size: int = SSIM_WINDOW_SIZE,
) -> Tensor:
    """Return locations whose entire square SSIM support is valid."""

    if valid_mask.ndim != 4 or valid_mask.shape[1] != 1:
        raise MetricInputError("valid mask must have shape [N,1,H,W]")
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
        raise MetricInputError("window_size must be a positive integer")
    if valid_mask.shape[-2] < window_size or valid_mask.shape[-1] < window_size:
        raise MetricInputError(
            f"valid mask is smaller than the {window_size}x{window_size} SSIM window"
        )
    kernel = torch.ones(
        (1, 1, window_size, window_size),
        device=valid_mask.device,
        dtype=torch.float64,
    )
    counts = F.conv2d(valid_mask.to(dtype=torch.float64), kernel)
    return counts == float(window_size * window_size)


def _gaussian_window(
    *,
    device: torch.device,
    dtype: torch.dtype,
    window_size: int = SSIM_WINDOW_SIZE,
    sigma: float = 1.5,
) -> Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2
    one_dimensional = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    one_dimensional = one_dimensional / one_dimensional.sum()
    return (one_dimensional[:, None] @ one_dimensional[None, :]).reshape(
        1, 1, window_size, window_size
    )


def _ssim_map(prediction: Tensor, target: Tensor) -> Tensor:
    window = _gaussian_window(device=prediction.device, dtype=prediction.dtype)
    mean_prediction = F.conv2d(prediction, window)
    mean_target = F.conv2d(target, window)
    mean_prediction_sq = mean_prediction.square()
    mean_target_sq = mean_target.square()
    mean_product = mean_prediction * mean_target
    variance_prediction = (
        F.conv2d(prediction.square(), window) - mean_prediction_sq
    ).clamp_min(0.0)
    variance_target = F.conv2d(target.square(), window) - mean_target_sq
    variance_target = variance_target.clamp_min(0.0)
    covariance = F.conv2d(prediction * target, window) - mean_product
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mean_product + c1) * (2 * covariance + c2)
    denominator = (
        (mean_prediction_sq + mean_target_sq + c1)
        * (variance_prediction + variance_target + c2)
    )
    return numerator / denominator


class MetricAccumulator:
    """Globally aggregate normalized and dB metrics over valid pixels/windows."""

    def __init__(self) -> None:
        self.n_samples = 0
        self.n_valid_pixels = 0
        self.n_ssim_windows = 0
        self.sum_sq_norm = 0.0
        self.sum_abs_db = 0.0
        self.sum_target_sq_norm = 0.0
        self.sum_ssim = 0.0
        self.raw_below_zero = 0
        self.raw_above_one = 0

    @staticmethod
    def _validate_shapes(prediction: Tensor, target: Tensor, valid_mask: Tensor) -> None:
        if prediction.shape != target.shape or prediction.shape != valid_mask.shape:
            raise MetricInputError(
                "prediction, target, and valid mask shapes must match: "
                f"{tuple(prediction.shape)}, {tuple(target.shape)}, "
                f"{tuple(valid_mask.shape)}"
            )
        if prediction.ndim != 4 or prediction.shape[1] != 1:
            raise MetricInputError("metric tensors must have shape [N,1,H,W]")
        if prediction.shape[0] <= 0:
            raise MetricInputError("metric batch must be non-empty")
        if valid_mask.dtype != torch.bool:
            raise MetricInputError("valid mask must have boolean dtype")
        if prediction.device != target.device or prediction.device != valid_mask.device:
            raise MetricInputError("metric tensors must be on the same device")

    def update(self, prediction: Tensor, target: Tensor, valid_mask: Tensor) -> None:
        self._validate_shapes(prediction, target, valid_mask)
        valid_counts = valid_mask.flatten(1).sum(dim=1)
        empty = torch.nonzero(valid_counts == 0, as_tuple=False).flatten().tolist()
        if empty:
            raise MetricInputError(f"empty valid mask for batch indices {empty}")
        if not bool(torch.isfinite(prediction[valid_mask]).all()):
            raise MetricInputError("non-finite prediction in a valid cell")
        if not bool(torch.isfinite(target[valid_mask]).all()):
            raise MetricInputError("non-finite target in a valid cell")
        valid_target = target[valid_mask]
        if bool(((valid_target < 0.0) | (valid_target > 1.0)).any()):
            raise MetricInputError("normalized target is outside [0,1] in a valid cell")

        complete = complete_window_mask(valid_mask)
        complete_counts = complete.flatten(1).sum(dim=1)
        missing_windows = torch.nonzero(
            complete_counts == 0, as_tuple=False
        ).flatten().tolist()
        if missing_windows:
            raise MetricInputError(
                "sample has no valid 11x11 SSIM window at batch indices "
                f"{missing_windows}"
            )

        valid_prediction_raw = prediction[valid_mask]
        self.raw_below_zero += int((valid_prediction_raw < 0.0).sum().item())
        self.raw_above_one += int((valid_prediction_raw > 1.0).sum().item())
        prediction_eval = prediction.clamp(0.0, 1.0)
        error = prediction_eval.double() - target.double()
        valid_error = error[valid_mask]
        self.sum_sq_norm += valid_error.square().sum().item()
        self.sum_abs_db += (valid_error.abs() * 300.0).sum().item()
        self.sum_target_sq_norm += target.double()[valid_mask].square().sum().item()

        # Invalid values are deliberately neutralized before convolution.  They
        # cannot contribute because only complete-mask windows are accumulated.
        prediction_ssim = torch.where(
            valid_mask,
            prediction_eval,
            torch.zeros_like(prediction_eval),
        ).double()
        target_ssim = torch.where(
            valid_mask,
            target,
            torch.zeros_like(target),
        ).double()
        ssim_map = _ssim_map(prediction_ssim, target_ssim)
        selected_ssim = ssim_map[complete]
        if not bool(torch.isfinite(selected_ssim).all()):
            raise MetricInputError("SSIM produced a non-finite valid-window value")
        self.sum_ssim += selected_ssim.sum().item()
        windows = int(complete.sum().item())
        self.n_ssim_windows += windows
        self.n_valid_pixels += int(valid_mask.sum().item())
        self.n_samples += int(prediction.shape[0])

    def compute(self) -> dict[str, int | float]:
        if self.n_samples <= 0 or self.n_valid_pixels <= 0 or self.n_ssim_windows <= 0:
            raise MetricInputError("metric accumulator has no evaluated samples")
        mse = self.sum_sq_norm / self.n_valid_pixels
        psnr = math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
        return {
            "n_samples": self.n_samples,
            "n_valid_pixels": self.n_valid_pixels,
            "n_ssim_windows": self.n_ssim_windows,
            "db_rmse": math.sqrt(90_000.0 * mse),
            "db_mae": self.sum_abs_db / self.n_valid_pixels,
            "mse": mse,
            "nmse": self.sum_sq_norm / max(self.sum_target_sq_norm, 1e-12),
            "psnr": psnr,
            "ssim": self.sum_ssim / self.n_ssim_windows,
            "raw_fraction_below_zero": self.raw_below_zero / self.n_valid_pixels,
            "raw_fraction_above_one": self.raw_above_one / self.n_valid_pixels,
        }


class PerBeamMetricAccumulators:
    """Keep an independent global accumulator and one accumulator per beam."""

    def __init__(self, beam_angles: Mapping[int, float]) -> None:
        if len(beam_angles) != 8:
            raise MetricInputError(
                f"exactly eight selected beams are required, got {len(beam_angles)}"
            )
        self.beam_angles = {
            int(beam_id): float(angle) for beam_id, angle in beam_angles.items()
        }
        if len(self.beam_angles) != 8 or any(
            not math.isfinite(angle) for angle in self.beam_angles.values()
        ):
            raise MetricInputError("beam IDs must be unique and angles finite")
        self.overall = MetricAccumulator()
        self.per_beam = {
            beam_id: MetricAccumulator() for beam_id in self.beam_angles
        }

    def update(
        self,
        prediction: Tensor,
        target: Tensor,
        valid_mask: Tensor,
        metadata: Sequence[Mapping[str, Any]],
    ) -> None:
        if len(metadata) != prediction.shape[0]:
            raise MetricInputError("metadata count must match metric batch size")
        beam_indices: dict[int, list[int]] = {beam_id: [] for beam_id in self.beam_angles}
        for index, item in enumerate(metadata):
            try:
                beam_id = int(item["beam_id"])
                angle = float(item["steering_deg"])
            except (KeyError, TypeError, ValueError) as error:
                raise MetricInputError(f"invalid beam metadata at index {index}") from error
            if beam_id not in self.beam_angles:
                raise MetricInputError(f"unexpected selected beam ID: {beam_id}")
            if not math.isclose(angle, self.beam_angles[beam_id], abs_tol=1e-9):
                raise MetricInputError(
                    f"beam {beam_id} angle mismatch: expected "
                    f"{self.beam_angles[beam_id]}, got {angle}"
                )
            beam_indices[beam_id].append(index)

        self.overall.update(prediction, target, valid_mask)
        for beam_id, indices in beam_indices.items():
            if not indices:
                continue
            self.per_beam[beam_id].update(
                prediction[indices],
                target[indices],
                valid_mask[indices],
            )

    def compute_overall(self) -> dict[str, int | float]:
        return self.overall.compute()

    def compute_rows(self) -> list[dict[str, int | float]]:
        rows: list[dict[str, int | float]] = []
        for beam_id, angle in self.beam_angles.items():
            metrics = self.per_beam[beam_id].compute()
            rows.append({"angle_deg": angle, "beam_id": beam_id, **metrics})
        return rows


def metrics_for_json(
    metrics: Mapping[str, int | float],
) -> dict[str, int | float | bool | None]:
    """Encode exact perfect PSNR without emitting non-standard JSON Infinity."""

    payload: dict[str, int | float | bool | None] = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or isinstance(value, int):
            payload[key] = value
            continue
        numeric = float(value)
        if key == "psnr" and math.isinf(numeric) and numeric > 0:
            payload[key] = None
            payload["psnr_infinite"] = True
            continue
        if not math.isfinite(numeric):
            raise MetricInputError(f"metric {key!r} is non-finite: {numeric}")
        payload[key] = numeric
    payload.setdefault("psnr_infinite", False)
    return payload

