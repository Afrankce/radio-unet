from __future__ import annotations

import math

import torch
from torch import Tensor


class SparseTask2FlowError(ValueError):
    """A full-target sparse Task 2 FM pair or loss is invalid."""


def _check_image(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or value.ndim != 4 or value.shape[1] != 1:
        raise SparseTask2FlowError(f"{name} must have shape [B,1,H,W]")
    if not value.is_floating_point():
        raise SparseTask2FlowError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise SparseTask2FlowError(f"{name} must be finite")


def _check_valid_mask(valid_mask: Tensor, shape: torch.Size, device: torch.device) -> None:
    if not isinstance(valid_mask, Tensor) or valid_mask.shape != shape:
        raise SparseTask2FlowError(
            f"valid_mask must have shape {tuple(shape)}, got {tuple(valid_mask.shape)}"
        )
    if valid_mask.dtype is not torch.bool:
        raise SparseTask2FlowError("valid_mask must have boolean dtype")
    if valid_mask.device != device:
        raise SparseTask2FlowError("valid_mask must share the image device")


def _expand_time(time: Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if not isinstance(time, Tensor) or time.ndim not in (1, 2):
        raise SparseTask2FlowError("time must have shape [B] or [B,1]")
    if time.ndim == 2 and time.shape[1] != 1:
        raise SparseTask2FlowError("time must have shape [B] or [B,1]")
    if time.shape[0] != batch_size:
        raise SparseTask2FlowError("time batch size must match x0")
    if not time.is_floating_point() or time.device != device:
        raise SparseTask2FlowError("time must be floating point on the image device")
    if not bool(torch.isfinite(time).all()):
        raise SparseTask2FlowError("time must be finite")
    if bool(((time < 0.0) | (time > 1.0)).any()):
        raise SparseTask2FlowError("time must lie in [0,1]")
    return time.to(dtype=dtype).reshape(batch_size, 1, 1, 1)


def build_task2_flow_pair(
    x0: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    time: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build the full-target conditional flow-matching pair for sparse Task 2."""

    _check_image("x0", x0)
    _check_image("target", target)
    if x0.shape != target.shape:
        raise SparseTask2FlowError("x0 and target must have identical shapes")
    if x0.device != target.device or x0.dtype != target.dtype:
        raise SparseTask2FlowError("x0 and target must share device and dtype")
    _check_valid_mask(valid_mask, x0.shape, x0.device)
    time_value = _expand_time(time, x0.shape[0], x0.device, x0.dtype)

    xt = (1.0 - time_value) * x0 + time_value * target
    ut = target - x0
    xt = xt.masked_fill(~valid_mask, 0.0)
    ut = ut.masked_fill(~valid_mask, 0.0)
    return xt, ut, valid_mask


def masked_task2_velocity_mse(
    predicted: Tensor,
    target_velocity: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Compute one global MSE over all valid radiomap pixels."""

    _check_image("predicted", predicted)
    _check_image("target_velocity", target_velocity)
    if predicted.shape != target_velocity.shape:
        raise SparseTask2FlowError("predicted and target_velocity must have identical shapes")
    if predicted.device != target_velocity.device or predicted.dtype != target_velocity.dtype:
        raise SparseTask2FlowError(
            "predicted and target_velocity must share device and dtype"
        )
    _check_valid_mask(valid_mask, predicted.shape, predicted.device)
    count = int(valid_mask.sum().item())
    if count <= 0:
        raise SparseTask2FlowError("valid_mask contains no valid pixels")
    squared = (predicted - target_velocity).square()
    return squared.masked_select(valid_mask).mean()


__all__ = [
    "SparseTask2FlowError",
    "build_task2_flow_pair",
    "masked_task2_velocity_mse",
]
