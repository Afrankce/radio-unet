from __future__ import annotations

import math

import torch
from torch import Tensor

from evaluation.radioflow_sampling import RadioFlowCFGModel


def _validate_sparse_tensors(
    condition: Tensor,
    observed_map: Tensor,
    observation_mask: Tensor,
    x0: Tensor,
) -> None:
    if condition.ndim != 4 or condition.shape[1] != 5:
        raise ValueError("sparse sampling requires a 5-channel NCHW condition")
    for name, value in (("observed_map", observed_map), ("x0", x0)):
        if value.ndim != 4 or value.shape[1] != 1:
            raise ValueError(f"{name} must have shape [B,1,H,W]")
        if not value.is_floating_point():
            raise ValueError(f"{name} must be floating point")
    if observation_mask.ndim != 4 or observation_mask.shape[1] != 1:
        raise ValueError("observation_mask must have shape [B,1,H,W]")
    if observation_mask.dtype != torch.bool:
        raise ValueError("observation_mask must be boolean")
    expected = (condition.shape[0], 1, condition.shape[-2], condition.shape[-1])
    if observed_map.shape != expected or observation_mask.shape != expected or x0.shape != expected:
        raise ValueError("condition, observed_map, observation_mask, and x0 shapes must match")
    devices = {condition.device, observed_map.device, observation_mask.device, x0.device}
    if len(devices) != 1:
        raise ValueError("condition, observed_map, observation_mask, and x0 must share a device")
    if not bool(torch.isfinite(condition).all()):
        raise ValueError("condition contains non-finite values")
    if not bool(torch.isfinite(observed_map).all()) or not bool(torch.isfinite(x0).all()):
        raise ValueError("observed_map and x0 must be finite")
    if int(torch.count_nonzero(observed_map.masked_select(~observation_mask)).item()) != 0:
        raise ValueError("observed_map must be zero in the non-observed region")


@torch.inference_mode()
def sparse_euler_cfg_sample(
    model: RadioFlowCFGModel,
    condition: Tensor,
    observed_map: Tensor,
    observation_mask: Tensor,
    x0: Tensor,
    *,
    cfg_scale: float,
    steps: int = 2,
    use_amp: bool = True,
) -> Tensor:
    """Two-step Euler CFG sampler with hard projection onto observed pixels."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be positive")
    if not math.isfinite(float(cfg_scale)):
        raise ValueError("cfg_scale must be finite")
    _validate_sparse_tensors(condition, observed_map, observation_mask, x0)

    not_observed = ~observation_mask
    x = observed_map + x0 * not_observed.to(dtype=x0.dtype)
    device_type = condition.device.type
    with torch.amp.autocast(
        device_type=device_type,
        dtype=torch.float16,
        enabled=bool(use_amp) and device_type == "cuda",
    ):
        embedding = model.embed_model(condition)
        dt = 1.0 / steps
        for index in range(steps):
            step = torch.full(
                (x.shape[0],),
                index / steps,
                device=x.device,
                dtype=x.dtype,
            )
            velocity = model.forward_with_cfg(
                image=condition,
                x=x,
                step=step,
                embedding=embedding,
                cfg_scale=float(cfg_scale),
            )
            if velocity.shape != x.shape:
                raise ValueError(
                    "RadioFlow velocity shape mismatch: "
                    f"expected {tuple(x.shape)}, got {tuple(velocity.shape)}"
                )
            x = x + dt * velocity
            x = observed_map + x * not_observed.to(dtype=x.dtype)
    return x.float()


__all__ = ["sparse_euler_cfg_sample"]
