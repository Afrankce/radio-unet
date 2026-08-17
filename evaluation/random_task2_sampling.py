from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import Tensor


class RandomTask2CFGModel(Protocol):
    def embed_model(
        self,
        condition: Tensor,
        sparse_map: Tensor,
        observation_mask: Tensor,
    ): ...

    def forward_with_cfg(
        self,
        *,
        image: Tensor,
        x: Tensor,
        step: Tensor,
        embedding: object,
        cfg_scale: float,
    ) -> Tensor: ...


@torch.inference_mode()
def random_task2_euler_cfg_sample(
    model: RandomTask2CFGModel,
    *,
    condition: Tensor,
    x0: Tensor,
    sparse_map: Tensor,
    observation_mask: Tensor,
    cfg_scale: float,
    steps: int = 2,
    use_amp: bool = True,
) -> Tensor:
    """Run projected Euler CFG sampling for the pinned random Task 2 path."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be positive")
    if not math.isfinite(float(cfg_scale)):
        raise ValueError("cfg_scale must be finite")
    if condition.ndim != 4 or x0.ndim != 4:
        raise ValueError("condition and x0 must be NCHW tensors")
    if condition.shape[0] != x0.shape[0] or condition.shape[-2:] != x0.shape[-2:]:
        raise ValueError("condition and x0 batch/spatial dimensions must match")
    if observation_mask.shape != x0.shape or observation_mask.dtype is not torch.bool:
        raise ValueError("observation_mask must be boolean and match x0")
    if sparse_map.shape != x0.shape:
        raise ValueError("sparse_map must match x0")
    if len({condition.device, x0.device, sparse_map.device, observation_mask.device}) != 1:
        raise ValueError("condition, x0, sparse_map, and observation_mask must share a device")
    if bool((observation_mask & ~torch.isfinite(sparse_map)).any()):
        raise ValueError("sparse_map must be finite at observed locations")

    device_type = condition.device.type
    with torch.amp.autocast(
        device_type=device_type,
        dtype=torch.float16,
        enabled=bool(use_amp) and device_type == "cuda",
    ):
        embedding = model.embed_model(condition, sparse_map, observation_mask)
        x = torch.where(observation_mask, sparse_map, x0)
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
                    f"velocity shape mismatch: expected {tuple(x.shape)}, got {tuple(velocity.shape)}"
                )
            x = torch.where(observation_mask, sparse_map, x + dt * velocity)
    return x.float()


__all__ = ["random_task2_euler_cfg_sample"]
