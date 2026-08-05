from __future__ import annotations

import torch
from torch import Tensor


class MaskedLossError(ValueError):
    """Velocity tensors cannot form the locked valid-pixel objective."""


def _validate_velocity_tensors(
    predicted_velocity: Tensor,
    target_velocity: Tensor,
    valid_mask: Tensor,
) -> None:
    if (
        predicted_velocity.shape != target_velocity.shape
        or predicted_velocity.shape != valid_mask.shape
    ):
        raise MaskedLossError(
            "velocity and mask shapes must match: "
            f"{tuple(predicted_velocity.shape)}, "
            f"{tuple(target_velocity.shape)}, {tuple(valid_mask.shape)}"
        )
    if predicted_velocity.ndim != 4 or predicted_velocity.shape[1] != 1:
        raise MaskedLossError("velocity tensors must have shape [N,1,H,W]")
    if valid_mask.dtype != torch.bool:
        raise MaskedLossError("valid_mask must have boolean dtype")
    if (
        predicted_velocity.device != target_velocity.device
        or predicted_velocity.device != valid_mask.device
    ):
        raise MaskedLossError("velocity tensors and mask must share a device")
    if not predicted_velocity.is_floating_point() or not target_velocity.is_floating_point():
        raise MaskedLossError("velocity tensors must use floating-point dtype")


def masked_velocity_mse(
    predicted_velocity: Tensor,
    target_velocity: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Mean squared velocity error over valid propagation pixels only."""

    _validate_velocity_tensors(predicted_velocity, target_velocity, valid_mask)
    valid_count = valid_mask.sum()
    if int(valid_count.item()) == 0:
        raise MaskedLossError("valid_mask contains zero valid pixels")
    predicted_valid = predicted_velocity.float().masked_select(valid_mask)
    target_valid = target_velocity.float().masked_select(valid_mask)
    difference = predicted_valid - target_valid
    if not bool(torch.isfinite(difference).all()):
        raise MaskedLossError("non-finite velocity in valid region")
    return difference.square().mean()

