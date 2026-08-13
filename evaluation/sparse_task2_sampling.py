from __future__ import annotations

import hashlib
import math
from typing import Protocol

import torch
from torch import Tensor


class SparseTask2CFGModel(Protocol):
    def embed_model(self, condition: Tensor): ...

    def forward_with_cfg(
        self,
        *,
        image: Tensor,
        x: Tensor,
        step: Tensor,
        embedding: object,
        cfg_scale: float,
    ) -> Tensor: ...


def make_task2_sample_noise(
    *,
    protocol: str,
    array_size: str,
    split: str,
    sample_key: str,
    shape: tuple[int, int, int] = (1, 256, 256),
    base_seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create replayable noise with an explicit Task 2 namespace."""

    if not all(isinstance(value, str) and value for value in (
        protocol, array_size, split, sample_key,
    )):
        raise ValueError("protocol, array_size, split, and sample_key must be non-empty")
    if len(shape) != 3 or any(
        type(value) is not int or value <= 0 for value in shape
    ):
        raise ValueError("noise shape must contain three positive integers")
    material = "|".join(
        (str(int(base_seed)), protocol, array_size, split, sample_key)
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype, device="cpu")


@torch.inference_mode()
def sparse_task2_euler_cfg_sample(
    model: SparseTask2CFGModel,
    condition: Tensor,
    x0: Tensor,
    *,
    cfg_scale: float,
    steps: int = 2,
    observation_mask: Tensor | None = None,
    sparse_map: Tensor | None = None,
    projected_consistency: bool = False,
    use_amp: bool = True,
) -> Tensor:
    """Sample the full target with RadioFlow's Euler-CFG integrator.

    ``projected_consistency`` is deliberately opt-in.  The source-equivalent
    result leaves the observed locations to the learned flow, while the
    projected result is an explicitly separate data-consistency ablation.
    """

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be positive")
    if not math.isfinite(float(cfg_scale)):
        raise ValueError("cfg_scale must be finite")
    if condition.ndim != 4 or x0.ndim != 4:
        raise ValueError("condition and x0 must be NCHW tensors")
    if condition.shape[0] != x0.shape[0] or condition.shape[-2:] != x0.shape[-2:]:
        raise ValueError("condition and x0 batch/spatial dimensions must match")
    if condition.device != x0.device:
        raise ValueError("condition and x0 must be on the same device")
    if projected_consistency:
        if observation_mask is None or sparse_map is None:
            raise ValueError("projection requires observation_mask and sparse_map")
        if observation_mask.shape != x0.shape or observation_mask.dtype is not torch.bool:
            raise ValueError("observation_mask must be boolean and match x0")
        if sparse_map.shape != x0.shape:
            raise ValueError("sparse_map must match x0")
        if observation_mask.device != x0.device or sparse_map.device != x0.device:
            raise ValueError("projection tensors must share x0 device")
        if bool((observation_mask & ~torch.isfinite(sparse_map)).any()):
            raise ValueError("sparse_map must be finite at observed locations")

    device_type = condition.device.type
    with torch.amp.autocast(
        device_type=device_type,
        dtype=torch.float16,
        enabled=bool(use_amp) and device_type == "cuda",
    ):
        embedding = model.embed_model(condition)
        x = x0
        dt = 1.0 / steps
        for index in range(steps):
            step = torch.full(
                (x.shape[0],), index / steps,
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
                    f"velocity shape mismatch: expected {tuple(x.shape)}, "
                    f"got {tuple(velocity.shape)}"
                )
            x = x + dt * velocity
            if projected_consistency:
                x = torch.where(observation_mask, sparse_map, x)
    return x.float()


__all__ = [
    "make_task2_sample_noise",
    "sparse_task2_euler_cfg_sample",
]
