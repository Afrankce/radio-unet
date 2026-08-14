from __future__ import annotations

import hashlib
import math
from typing import Protocol

import torch
from torch import Tensor

from training.sparse_consistent_config import SPARSE_CONSISTENT_PROTOCOL


class SparseConsistentCFGModel(Protocol):
    def embed_model(self, condition: Tensor, *args: Tensor): ...

    def forward_with_cfg(
        self,
        *,
        image: Tensor,
        x: Tensor,
        step: Tensor,
        embedding: object,
        cfg_scale: float,
    ) -> Tensor: ...


def make_sparse_consistent_sample_noise(
    *,
    array_size: str,
    split: str,
    sample_key: str,
    shape: tuple[int, int, int] = (1, 256, 256),
    base_seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    material = "|".join(
        (str(int(base_seed)), SPARSE_CONSISTENT_PROTOCOL, array_size, split, sample_key)
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype, device="cpu")


@torch.inference_mode()
def sparse_consistent_euler_cfg_sample(
    model: SparseConsistentCFGModel,
    *,
    arm: str,
    condition: Tensor,
    x0: Tensor,
    sparse_map: Tensor,
    observation_mask: Tensor,
    cfg_scale: float = 1.0,
    steps: int = 2,
    use_amp: bool = True,
) -> Tensor:
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if condition.ndim != 4 or x0.ndim != 4:
        raise ValueError("condition and x0 must be NCHW tensors")
    if condition.shape[0] != x0.shape[0] or condition.shape[-2:] != x0.shape[-2:]:
        raise ValueError("condition and x0 batch/spatial shapes must match")
    if sparse_map.shape != x0.shape or observation_mask.shape != x0.shape:
        raise ValueError("sparse_map and observation_mask must match x0")
    if observation_mask.dtype is not torch.bool:
        raise ValueError("observation_mask must be boolean")
    if arm == "multiscale_consistent":
        x = torch.where(observation_mask, sparse_map, x0)
    else:
        x = x0
    device_type = condition.device.type
    with torch.amp.autocast(
        device_type=device_type,
        dtype=torch.float16,
        enabled=bool(use_amp) and device_type == "cuda",
    ):
        if arm in {"multiscale_fullfm", "multiscale_consistent"}:
            embedding = model.embed_model(condition, sparse_map, observation_mask)
        else:
            embedding = model.embed_model(condition)
        dt = 1.0 / steps
        for index in range(steps):
            step = torch.full(
                (x.shape[0],), index / steps, device=x.device, dtype=x.dtype
            )
            velocity = model.forward_with_cfg(
                image=condition,
                x=x,
                step=step,
                embedding=embedding,
                cfg_scale=float(cfg_scale),
            )
            if velocity.shape != x.shape:
                raise ValueError("model velocity shape does not match x0")
            x = x + dt * velocity
            if arm == "multiscale_consistent":
                x = torch.where(observation_mask, sparse_map, x)
    return x.float()


__all__ = [
    "make_sparse_consistent_sample_noise",
    "sparse_consistent_euler_cfg_sample",
]
