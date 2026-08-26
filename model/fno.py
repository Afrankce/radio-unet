from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def count_tensor_parameters(module: nn.Module) -> int:
    """Count PyTorch parameter elements, where one complex value counts once."""

    return sum(parameter.numel() for parameter in module.parameters())


def count_real_scalar_parameters(module: nn.Module) -> int:
    """Count independent real scalars, counting every complex value twice."""

    return sum(
        parameter.numel() * (2 if parameter.is_complex() else 1)
        for parameter in module.parameters()
    )


class SpectralConv2d(nn.Module):
    """Dense two-corner spectral convolution from the original FNO2d code."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.out_channels = _positive_int(out_channels, "out_channels")
        self.modes1 = _positive_int(modes1, "modes1")
        self.modes2 = _positive_int(modes2, "modes2")
        scale = 1.0 / (self.in_channels * self.out_channels)
        shape = (
            self.in_channels,
            self.out_channels,
            self.modes1,
            self.modes2,
        )
        self.weights1 = nn.Parameter(
            scale * torch.rand(*shape, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(*shape, dtype=torch.cfloat)
        )

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 4:
            raise ValueError("spectral input must be an NCHW tensor")
        if value.shape[1] != self.in_channels:
            raise ValueError(
                "spectral input channel mismatch: "
                f"expected {self.in_channels}, got {value.shape[1]}"
            )
        height, width = value.shape[-2:]
        if 2 * self.modes1 > height or self.modes2 > width // 2 + 1:
            raise ValueError(
                "retained modes do not fit the spatial grid: "
                f"grid=({height},{width}), modes=({self.modes1},{self.modes2})"
            )

        output_dtype = value.dtype
        with torch.autocast(device_type=value.device.type, enabled=False):
            full_precision = value.float()
            value_ft = torch.fft.rfft2(full_precision)
            output_ft = torch.zeros(
                full_precision.shape[0],
                self.out_channels,
                height,
                width // 2 + 1,
                dtype=torch.cfloat,
                device=full_precision.device,
            )
            output_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
                "bixy,ioxy->boxy",
                value_ft[:, :, : self.modes1, : self.modes2],
                self.weights1,
            )
            output_ft[:, :, -self.modes1 :, : self.modes2] = torch.einsum(
                "bixy,ioxy->boxy",
                value_ft[:, :, -self.modes1 :, : self.modes2],
                self.weights2,
            )
            result = torch.fft.irfft2(output_ft, s=(height, width))
        return result.to(dtype=output_dtype)


class ConditionalFNO2d(nn.Module):
    """Paper-faithful FNO2d adapted to a conditional FM velocity field."""

    def __init__(
        self,
        *,
        condition_channels: int = 3,
        width: int = 40,
        modes1: int = 12,
        modes2: int = 12,
        padding: int = 9,
        cfg_drop_prob: float = 0.25,
    ) -> None:
        super().__init__()
        self.condition_channels = _positive_int(
            condition_channels, "condition_channels"
        )
        self.width = _positive_int(width, "width")
        self.modes1 = _positive_int(modes1, "modes1")
        self.modes2 = _positive_int(modes2, "modes2")
        if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
            raise ValueError("padding must be a non-negative integer")
        self.padding = padding
        if not math.isfinite(float(cfg_drop_prob)) or not 0.0 <= float(
            cfg_drop_prob
        ) <= 1.0:
            raise ValueError("cfg_drop_prob must lie in [0,1]")
        self.cfg_drop_prob = float(cfg_drop_prob)

        input_channels = 1 + self.condition_channels + 1 + 2
        self.lifting = nn.Linear(input_channels, self.width)
        self.spectral_layers = nn.ModuleList(
            SpectralConv2d(
                self.width,
                self.width,
                self.modes1,
                self.modes2,
            )
            for _ in range(4)
        )
        self.local_layers = nn.ModuleList(
            nn.Conv2d(self.width, self.width, kernel_size=1) for _ in range(4)
        )
        self.projection_hidden = nn.Linear(self.width, 128)
        self.projection_output = nn.Linear(128, 1)

    def embed_model(self, condition: Tensor) -> Tensor:
        self._validate_condition(condition)
        return condition

    def _validate_condition(self, condition: Tensor) -> None:
        if not isinstance(condition, Tensor) or condition.ndim != 4:
            raise ValueError("condition must be an NCHW tensor")
        if condition.shape[1] != self.condition_channels:
            raise ValueError(
                "condition channel mismatch: "
                f"expected {self.condition_channels}, got {condition.shape[1]}"
            )

    def _normalize_step(self, step: Tensor | float | int, state: Tensor) -> Tensor:
        if isinstance(step, Tensor):
            normalized = step.to(device=state.device, dtype=state.dtype)
            if normalized.ndim == 0:
                normalized = normalized.repeat(state.shape[0])
            elif normalized.ndim == 1 and normalized.shape[0] == 1:
                normalized = normalized.repeat(state.shape[0])
            elif normalized.ndim != 1 or normalized.shape[0] != state.shape[0]:
                raise ValueError("step must be scalar or have one value per sample")
        elif isinstance(step, (int, float)) and not isinstance(step, bool):
            normalized = torch.full(
                (state.shape[0],),
                float(step),
                device=state.device,
                dtype=state.dtype,
            )
        else:
            raise ValueError("step must be a tensor or finite scalar")
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError("step values must be finite")
        return normalized

    def _validate_state_and_condition(
        self,
        condition: Tensor,
        state: Tensor,
    ) -> None:
        self._validate_condition(condition)
        if not isinstance(state, Tensor) or state.ndim != 4:
            raise ValueError("state must be an NCHW tensor")
        if state.shape[1] != 1:
            raise ValueError("state must contain exactly one channel")
        if state.shape[0] != condition.shape[0] or state.shape[-2:] != condition.shape[-2:]:
            raise ValueError("state and condition batch/spatial dimensions must match")
        if state.device != condition.device:
            raise ValueError("state and condition must be on the same device")

    @staticmethod
    def _coordinate_grid(state: Tensor) -> tuple[Tensor, Tensor]:
        batch, _, height, width = state.shape
        grid_x = torch.linspace(
            0.0, 1.0, height, device=state.device, dtype=state.dtype
        ).view(1, 1, height, 1)
        grid_y = torch.linspace(
            0.0, 1.0, width, device=state.device, dtype=state.dtype
        ).view(1, 1, 1, width)
        return (
            grid_x.expand(batch, 1, height, width),
            grid_y.expand(batch, 1, height, width),
        )

    def _velocity(
        self,
        condition: Tensor,
        state: Tensor,
        step: Tensor | float | int,
    ) -> Tensor:
        self._validate_state_and_condition(condition, state)
        normalized_step = self._normalize_step(step, state)
        time_map = normalized_step.view(-1, 1, 1, 1).expand(
            -1, 1, state.shape[-2], state.shape[-1]
        )
        grid_x, grid_y = self._coordinate_grid(state)
        value = torch.cat(
            (state, condition, time_map, grid_x, grid_y),
            dim=1,
        )
        value = self.lifting(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.padding:
            value = F.pad(value, (0, self.padding, 0, self.padding))
        for index, (spectral, local) in enumerate(
            zip(self.spectral_layers, self.local_layers)
        ):
            value = spectral(value) + local(value)
            if index < len(self.spectral_layers) - 1:
                value = F.gelu(value)
        if self.padding:
            value = value[..., : -self.padding, : -self.padding]
        value = value.permute(0, 2, 3, 1)
        value = F.gelu(self.projection_hidden(value))
        value = self.projection_output(value)
        return value.permute(0, 3, 1, 2)

    def forward(
        self,
        image: Tensor | None = None,
        x: Tensor | None = None,
        pred_type: str = "denoise",
        step: Tensor | float | int | None = None,
        embedding: Tensor | None = None,
    ) -> Tensor:
        if pred_type != "denoise":
            raise ValueError("ConditionalFNO2d only supports pred_type='denoise'")
        if x is None or step is None:
            raise ValueError("x and step are required")
        condition = embedding if embedding is not None else image
        if condition is None:
            raise ValueError("image or embedding condition is required")
        self._validate_state_and_condition(condition, x)
        if self.training and self.cfg_drop_prob > 0.0:
            drop = (
                torch.rand(condition.shape[0], device=condition.device)
                < self.cfg_drop_prob
            )
            if bool(drop.any()):
                condition = condition.masked_fill(drop.view(-1, 1, 1, 1), 0.0)
        return self._velocity(condition, x, step)

    def forward_with_cfg(
        self,
        *,
        image: Tensor,
        x: Tensor,
        step: Tensor,
        embedding: Tensor | None = None,
        cfg_scale: float = 1.0,
    ) -> Tensor:
        if not math.isfinite(float(cfg_scale)):
            raise ValueError("cfg_scale must be finite")
        condition = image if embedding is None else embedding
        self._validate_state_and_condition(condition, x)
        unconditional = self._velocity(torch.zeros_like(condition), x, step)
        conditional = self._velocity(condition, x, step)
        return unconditional + float(cfg_scale) * (conditional - unconditional)


__all__ = [
    "ConditionalFNO2d",
    "SpectralConv2d",
    "count_real_scalar_parameters",
    "count_tensor_parameters",
]

