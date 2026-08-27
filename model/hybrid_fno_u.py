from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.fno import SpectralConv2d


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class FNOOperatorBlock(nn.Module):
    """One lifted spectral-plus-local operator with Flow-time conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        width: int,
        modes: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.in_channels = _require_positive_int(in_channels, "in_channels")
        self.out_channels = _require_positive_int(out_channels, "out_channels")
        self.width = _require_positive_int(width, "width")
        self.modes = _require_positive_int(modes, "modes")
        if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
            raise ValueError("padding must be a non-negative integer")
        self.padding = padding

        self.lifting = nn.Conv2d(self.in_channels, self.width, kernel_size=1)
        self.spectral = SpectralConv2d(
            self.width,
            self.width,
            modes1=self.modes,
            modes2=self.modes,
        )
        self.local = nn.Conv2d(self.width, self.width, kernel_size=1)
        self.time_projection = nn.Linear(512, self.width)
        self.projection = nn.Conv2d(self.width, self.out_channels, kernel_size=1)

    def _validate_inputs(self, value: Tensor, temb: Tensor) -> None:
        if not isinstance(value, Tensor) or value.ndim != 4:
            raise ValueError("operator input must be an NCHW tensor")
        if value.shape[1] != self.in_channels:
            raise ValueError(
                "operator input channel mismatch: "
                f"expected {self.in_channels}, got {value.shape[1]}"
            )
        if not isinstance(temb, Tensor) or temb.ndim != 2 or temb.shape[1] != 512:
            raise ValueError("time embedding must have shape [B,512]")
        if temb.shape[0] != value.shape[0]:
            raise ValueError("operator input and time embedding batch mismatch")
        if temb.device != value.device:
            raise ValueError("operator input and time embedding devices must match")

    def forward(self, value: Tensor, temb: Tensor) -> Tensor:
        self._validate_inputs(value, temb)
        hidden = self.lifting(value)
        if self.padding:
            hidden = F.pad(hidden, (0, self.padding, 0, self.padding))
        time_bias = self.time_projection(F.silu(temb))[:, :, None, None]
        hidden = F.gelu(self.spectral(hidden) + self.local(hidden) + time_bias)
        if self.padding:
            hidden = hidden[..., : -self.padding, : -self.padding]
        return self.projection(hidden)


__all__ = ["FNOOperatorBlock"]
