from __future__ import annotations

import hashlib
import math

import torch
from torch import Tensor


def _validate_bool_mask(name: str, value: Tensor) -> None:
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must have boolean dtype")
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ValueError(f"{name} must have shape [1,H,W] or [B,1,H,W]")
        return
    if value.ndim == 4 and value.shape[1] == 1:
        return
    raise ValueError(f"{name} must have shape [1,H,W] or [B,1,H,W]")


def _validate_tensor_shape(name: str, value: Tensor) -> None:
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ValueError(f"{name} must have shape [1,H,W] or [B,1,H,W]")
        return
    if value.ndim == 4 and value.shape[1] == 1:
        return
    raise ValueError(f"{name} must have shape [1,H,W] or [B,1,H,W]")


def _hash_seed(*parts: object) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def make_observation_mask(
    valid_mask: Tensor,
    *,
    scene_id: str,
    steering_deg: float,
    ratio: float,
    base_seed: int,
) -> Tensor:
    """Select ceil(ratio * valid_count) valid pixels deterministically."""

    _validate_bool_mask("valid_mask", valid_mask)
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("scene_id must be a non-empty string")
    if not math.isfinite(steering_deg):
        raise ValueError("steering_deg must be finite")
    if not isinstance(ratio, float) or not math.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be finite and satisfy 0 < ratio < 1")
    if type(base_seed) is not int:
        raise ValueError("base_seed must be an integer")

    cpu_mask = valid_mask.detach().to(device="cpu", dtype=torch.bool)
    valid_indices = torch.nonzero(cpu_mask.reshape(-1), as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError("valid_mask contains zero valid pixels")
    observed_count = max(1, math.ceil(ratio * int(valid_indices.numel())))
    seed = _hash_seed(
        base_seed,
        scene_id,
        f"{steering_deg:.6f}",
        f"{ratio:.8f}",
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(valid_indices.numel(), generator=generator)
    selected = valid_indices[order[:observed_count]]
    mask = torch.zeros(cpu_mask.numel(), dtype=torch.bool)
    mask[selected] = True
    observation_mask = mask.reshape(cpu_mask.shape) & cpu_mask
    return observation_mask.to(device=valid_mask.device)


def make_condition_noise(
    shape: tuple[int, int, int],
    *,
    scene_id: str,
    steering_deg: float,
    split: str,
    epoch: int | None,
    base_seed: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return independent Gaussian noise with a stable hash-derived seed."""

    if len(shape) != 3 or any(type(dim) is not int or dim <= 0 for dim in shape):
        raise ValueError("shape must be a tuple of three positive integers")
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("scene_id must be a non-empty string")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a non-empty string")
    if not math.isfinite(steering_deg):
        raise ValueError("steering_deg must be finite")
    if epoch is not None and (type(epoch) is not int or epoch < 0):
        raise ValueError("epoch must be None or a non-negative integer")
    if type(base_seed) is not int:
        raise ValueError("base_seed must be an integer")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("dtype must be floating point")

    epoch_token = "none" if epoch is None else str(epoch)
    seed = _hash_seed(
        base_seed,
        scene_id,
        f"{steering_deg:.6f}",
        split,
        epoch_token,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(shape, generator=generator, dtype=dtype, device="cpu")
    if not bool(torch.isfinite(noise).all()):
        raise ValueError("condition noise must be finite")
    return noise


def build_masked_condition_map(
    target: Tensor,
    valid_mask: Tensor,
    observation_mask: Tensor,
    condition_noise: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return masked map, observed map, and missing-valid mask."""

    _validate_tensor_shape("target", target)
    _validate_tensor_shape("condition_noise", condition_noise)
    _validate_bool_mask("valid_mask", valid_mask)
    _validate_bool_mask("observation_mask", observation_mask)
    if target.shape != condition_noise.shape:
        raise ValueError("shape mismatch between target and condition_noise")
    if target.shape != valid_mask.shape or target.shape != observation_mask.shape:
        raise ValueError("shape mismatch across target and masks")
    if target.device != condition_noise.device:
        raise ValueError("target and condition_noise must share a device")
    if target.device != valid_mask.device or target.device != observation_mask.device:
        raise ValueError("target, valid_mask, and observation_mask must share a device")
    if target.dtype != condition_noise.dtype:
        raise ValueError("target and condition_noise must share a dtype")
    if not target.is_floating_point() or not condition_noise.is_floating_point():
        raise ValueError("target and condition_noise must be floating point")
    if not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(condition_noise).all()):
        raise ValueError("target and condition_noise must be finite")
    if not bool((observation_mask & ~valid_mask).sum().item() == 0):
        raise ValueError("observation_mask must be a subset of valid_mask")

    visible = observation_mask.to(dtype=target.dtype)
    missing_valid = valid_mask & ~observation_mask
    masked_map = (
        target * visible + condition_noise.to(dtype=target.dtype) * missing_valid.to(dtype=target.dtype)
    ).masked_fill(~valid_mask, 0.0)
    observed_map = (target * visible).masked_fill(~valid_mask, 0.0)
    return masked_map, observed_map, missing_valid
