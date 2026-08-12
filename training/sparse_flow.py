from __future__ import annotations

import math

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
    time: float,
) -> tuple[Tensor, Tensor]:
    if not isinstance(time, float) or not math.isfinite(time) or not 0.0 <= time <= 1.0:
        raise ValueError("time must be finite and satisfy 0 <= time <= 1")

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
    if not bool((observation_mask & ~valid_mask).sum().item() == 0):
        raise ValueError("observation_mask must be a subset of valid_mask")

    missing_valid = valid_mask & ~observation_mask
    if int(missing_valid.sum().item()) == 0:
        raise ValueError("missing valid region must be non-empty")

    missing_float = missing_valid.to(dtype=target.dtype)
    xt = observed_map + missing_float * ((1.0 - time) * initial_noise + time * target)
    ut = missing_float * (target - initial_noise)
    return xt, ut
