from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from model.fno import SpectralConv2d
from model.unet.basic_unet_denose import CrossAttention, nonlinearity


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class AttentionConditionedFNOStage(nn.Module):
    """One residual CA/SA-conditioned spectral operator update."""

    def __init__(
        self,
        *,
        channels: int,
        embedding_channels: int,
        operator_width: int,
        modes: int,
        padding: int,
        time_channels: int = 512,
        attention_reduction: int = 16,
    ) -> None:
        super().__init__()
        self._channels = _positive_int(channels, "channels")
        self._embedding_channels = _positive_int(
            embedding_channels, "embedding_channels"
        )
        self._operator_width = _positive_int(operator_width, "operator_width")
        self._modes = _positive_int(modes, "modes")
        self._time_channels = _positive_int(time_channels, "time_channels")
        if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
            raise ValueError("padding must be a non-negative integer")
        self._padding = padding
        if (
            isinstance(attention_reduction, bool)
            or not isinstance(attention_reduction, int)
            or attention_reduction <= 0
            or self._channels // attention_reduction <= 0
        ):
            raise ValueError("attention_reduction must leave at least one channel")

        self.attention = CrossAttention(
            self._channels,
            self._embedding_channels,
            reduction=attention_reduction,
            kernel_size=7,
        )
        self.lifting = nn.Conv2d(self._channels, self._operator_width, kernel_size=1)
        self.spectral = SpectralConv2d(
            self._operator_width,
            self._operator_width,
            self._modes,
            self._modes,
        )
        self.local = nn.Conv2d(
            self._operator_width, self._operator_width, kernel_size=1
        )
        self.time_projection = nn.Linear(self._time_channels, self._operator_width)
        self.projection = nn.Conv2d(
            self._operator_width, self._channels, kernel_size=1
        )

    def _validate_inputs(
        self,
        value: Tensor,
        condition: Tensor,
        time_embedding: Tensor,
    ) -> None:
        if (
            not isinstance(value, Tensor)
            or value.ndim != 4
            or value.shape[1] != self._channels
        ):
            raise ValueError(f"value must have shape [B,{self._channels},H,W]")
        if (
            not isinstance(condition, Tensor)
            or condition.ndim != 4
            or condition.shape[0] != value.shape[0]
            or condition.shape[1] != self._embedding_channels
            or condition.shape[-2:] != value.shape[-2:]
        ):
            raise ValueError(
                "condition must have shape "
                f"[B,{self._embedding_channels},H,W] matching value"
            )
        if (
            not isinstance(time_embedding, Tensor)
            or time_embedding.ndim != 2
            or time_embedding.shape[0] != value.shape[0]
            or time_embedding.shape[1] != self._time_channels
        ):
            raise ValueError(
                "time embedding must have shape " f"[B,{self._time_channels}]"
            )
        if value.device != condition.device or value.device != time_embedding.device:
            raise ValueError("value, condition, and time embedding must share a device")

    def forward(
        self,
        value: Tensor,
        condition: Tensor,
        time_embedding: Tensor,
    ) -> Tensor:
        self._validate_inputs(value, condition, time_embedding)
        attended = self.attention(value, condition)
        hidden = self.lifting(attended)
        height, width = hidden.shape[-2:]
        if self._padding:
            hidden = F.pad(hidden, (0, self._padding, 0, self._padding))
        temporal = self.time_projection(nonlinearity(time_embedding))[:, :, None, None]
        delta = F.gelu(self.spectral(hidden) + self.local(hidden) + temporal)
        delta = delta[..., :height, :width]
        return attended + self.projection(delta)


__all__ = ["AttentionConditionedFNOStage"]
