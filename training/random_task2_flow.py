from __future__ import annotations

import torch
from torch import Tensor


class RandomTask2FlowError(ValueError):
    """A pinned random Task 2 flow pair violates its contract."""


def _check_image(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or value.ndim != 4 or value.shape[1] != 1:
        raise RandomTask2FlowError(f"{name} must have shape [B,1,H,W]")
    if not value.is_floating_point():
        raise RandomTask2FlowError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise RandomTask2FlowError(f"{name} must be finite")


def _check_bool_mask(name: str, value: Tensor, shape: torch.Size, device: torch.device) -> None:
    if not isinstance(value, Tensor) or value.shape != shape:
        raise RandomTask2FlowError(f"{name} must have shape {tuple(shape)}")
    if value.dtype is not torch.bool:
        raise RandomTask2FlowError(f"{name} must be boolean")
    if value.device != device:
        raise RandomTask2FlowError(f"{name} must share the image device")


def _expand_time(
    time: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if not isinstance(time, Tensor) or time.ndim not in (1, 2):
        raise RandomTask2FlowError("time must have shape [B] or [B,1]")
    if time.ndim == 2 and time.shape[1] != 1:
        raise RandomTask2FlowError("time must have shape [B] or [B,1]")
    if time.shape[0] != batch_size:
        raise RandomTask2FlowError("time batch size must match x0")
    if not time.is_floating_point() or time.device != device:
        raise RandomTask2FlowError("time must be floating point on the image device")
    if not bool(torch.isfinite(time).all()):
        raise RandomTask2FlowError("time must be finite")
    if bool(((time < 0.0) | (time > 1.0)).any()):
        raise RandomTask2FlowError("time must lie in [0,1]")
    return time.to(dtype=dtype).reshape(batch_size, 1, 1, 1)


def build_random_task2_pinned_flow_pair(
    *,
    x0: Tensor,
    target: Tensor,
    sparse_map: Tensor,
    observation_mask: Tensor,
    valid_mask: Tensor,
    time: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build the pinned conditional flow-matching pair for random Task 2."""

    _check_image("x0", x0)
    _check_image("target", target)
    _check_image("sparse_map", sparse_map)
    if x0.shape != target.shape or x0.shape != sparse_map.shape:
        raise RandomTask2FlowError("x0, target, and sparse_map must have identical shapes")
    if len({x0.device, target.device, sparse_map.device}) != 1:
        raise RandomTask2FlowError("x0, target, and sparse_map must share a device")
    if len({x0.dtype, target.dtype, sparse_map.dtype}) != 1:
        raise RandomTask2FlowError("x0, target, and sparse_map must share a dtype")
    _check_bool_mask("observation_mask", observation_mask, x0.shape, x0.device)
    _check_bool_mask("valid_mask", valid_mask, x0.shape, x0.device)
    if bool((observation_mask & ~valid_mask).any()):
        raise RandomTask2FlowError("observation_mask must be a subset of valid_mask")
    if bool((sparse_map.masked_select(~observation_mask)).abs().max().item() > 1e-6):
        raise RandomTask2FlowError("sparse_map must be zero outside observations")

    time_value = _expand_time(
        time,
        batch_size=x0.shape[0],
        device=x0.device,
        dtype=x0.dtype,
    )
    missing_mask = valid_mask & ~observation_mask
    xt = torch.where(observation_mask, sparse_map, (1.0 - time_value) * x0 + time_value * target)
    ut = torch.where(missing_mask, target - x0, torch.zeros_like(x0))
    xt = xt.masked_fill(~valid_mask, 0.0)
    ut = ut.masked_fill(~valid_mask, 0.0)
    return xt, ut, missing_mask


__all__ = [
    "RandomTask2FlowError",
    "build_random_task2_pinned_flow_pair",
]
