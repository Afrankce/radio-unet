from __future__ import annotations

import hashlib
import math
from typing import Protocol

import torch
from torch import Tensor


class RadioFlowCFGModel(Protocol):
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


def make_sample_noise(
    scene_id: str,
    steering_deg: float,
    shape: tuple[int, int, int] = (1, 256, 256),
    *,
    base_seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create canonical CPU noise shared by array/model/CFG comparisons."""

    if not scene_id:
        raise ValueError("scene_id must be non-empty")
    if not math.isfinite(float(steering_deg)):
        raise ValueError("steering_deg must be finite")
    if len(shape) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in shape
    ):
        raise ValueError(f"noise shape must contain three positive integers: {shape}")
    material = f"{int(base_seed)}|{scene_id}|{float(steering_deg):.6f}".encode(
        "utf-8"
    )
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        shape,
        generator=generator,
        dtype=dtype,
        device="cpu",
    )


@torch.inference_mode()
def euler_cfg_sample(
    model: RadioFlowCFGModel,
    condition: Tensor,
    x0: Tensor,
    *,
    cfg_scale: float,
    steps: int = 2,
    use_amp: bool = True,
) -> Tensor:
    """Integrate RadioFlow's own CFG velocity field with fixed-step Euler."""

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
    return x.float()

