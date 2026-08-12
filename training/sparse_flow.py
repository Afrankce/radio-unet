from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def _validate_tensor_shape(name: str, value: Tensor) -> None:
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ValueError(f"{name} must have shape [1,H,W] or [B,1,H,W]")
        return
    if value.ndim == 4 and value.shape[1] == 1:
        return
    raise ValueError(f"{name} must have shape [1,H,W] or [B,1,H,W]")


def _validate_bool_mask(name: str, value: Tensor) -> None:
    _validate_tensor_shape(name, value)
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must have boolean dtype")


def build_masked_flow_pair(
    initial_noise: Tensor,
    target: Tensor,
    observed_map: Tensor,
    observation_mask: Tensor,
    valid_mask: Tensor,
    *,
    time: float | Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_tensor_shape("initial_noise", initial_noise)
    _validate_tensor_shape("target", target)
    _validate_tensor_shape("observed_map", observed_map)
    _validate_bool_mask("observation_mask", observation_mask)
    _validate_bool_mask("valid_mask", valid_mask)

    expected_shape = initial_noise.shape
    for name, value in (
        ("target", target),
        ("observed_map", observed_map),
        ("observation_mask", observation_mask),
        ("valid_mask", valid_mask),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"shape mismatch: {name} does not match initial_noise")
        if value.device != initial_noise.device:
            raise ValueError(f"device mismatch: {name} does not match initial_noise")
    if initial_noise.dtype != target.dtype or initial_noise.dtype != observed_map.dtype:
        raise ValueError("dtype mismatch across initial_noise, target, and observed_map")
    if not (
        initial_noise.is_floating_point()
        and target.is_floating_point()
        and observed_map.is_floating_point()
    ):
        raise ValueError("initial_noise, target, and observed_map must be floating point")
    for value_name, value in (
        ("initial_noise", initial_noise),
        ("target", target),
        ("observed_map", observed_map),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{value_name} must be finite")
    time_tensor = _time_tensor(time, reference=target)
    if not bool((observation_mask & ~valid_mask).sum().item() == 0):
        raise ValueError("observation_mask must be a subset of valid_mask")
    if not bool(torch.count_nonzero(observed_map.masked_select(~observation_mask)).item() == 0):
        raise ValueError("observed_map must be zero outside observation_mask")

    missing_valid = valid_mask & ~observation_mask
    if int(missing_valid.sum().item()) == 0:
        raise ValueError("missing valid region must be non-empty")

    missing_float = missing_valid.to(dtype=target.dtype)
    xt = observed_map + missing_float * (
        (1.0 - time_tensor) * initial_noise + time_tensor * target
    )
    ut = missing_float * (target - initial_noise)
    xt = xt.masked_fill(~valid_mask, 0.0)
    ut = ut.masked_fill(~missing_valid, 0.0)
    return xt, ut, missing_valid


def _time_tensor(time: float | Tensor, *, reference: Tensor) -> Tensor:
    if isinstance(time, Tensor):
        if time.device != reference.device:
            raise ValueError("time must be on the same device as target")
        if not time.is_floating_point():
            raise ValueError("time must be floating point")
        if not bool(torch.isfinite(time).all()):
            raise ValueError("time must be finite and satisfy 0 <= time <= 1")
        if bool(((time < 0.0) | (time > 1.0)).any()):
            raise ValueError("time must be finite and satisfy 0 <= time <= 1")
        if time.ndim == 0:
            return time.to(dtype=reference.dtype).reshape(
                *((1,) * reference.ndim)
            )
        if reference.ndim != 4 or time.ndim != 1 or time.shape[0] != reference.shape[0]:
            raise ValueError("time tensor must have shape [B] for batched inputs")
        return time.to(dtype=reference.dtype).reshape(reference.shape[0], 1, 1, 1)
    if isinstance(time, bool) or not isinstance(time, float):
        raise ValueError("time must be a float or floating tensor")
    if not math.isfinite(time) or not 0.0 <= time <= 1.0:
        raise ValueError("time must be finite and satisfy 0 <= time <= 1")
    return torch.tensor(time, device=reference.device, dtype=reference.dtype).reshape(
        *((1,) * reference.ndim)
    )


__all__ = ["build_masked_flow_pair"]
