from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.fno import SpectralConv2d
from model.unet.basic_unet import BasicUNetEncoder
from model.unet.basic_unet_denose import (
    CrossAttention,
    get_timestep_embedding,
    nonlinearity,
)


DEFAULT_ENCODER_FEATURES = (32, 32, 64, 128, 256, 32)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class FullResolutionFNOBlock(nn.Module):
    """One CA/SA-conditioned spectral, local, and time operator update."""

    def __init__(
        self,
        *,
        width: int,
        modes1: int,
        modes2: int,
        padding: int,
        time_channels: int = 512,
        attention_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.width = _positive_int(width, "width")
        self.modes1 = _positive_int(modes1, "modes1")
        self.modes2 = _positive_int(modes2, "modes2")
        self.time_channels = _positive_int(time_channels, "time_channels")
        if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
            raise ValueError("padding must be a non-negative integer")
        self.padding = padding
        if (
            isinstance(attention_reduction, bool)
            or not isinstance(attention_reduction, int)
            or attention_reduction <= 0
            or self.width // attention_reduction <= 0
        ):
            raise ValueError("attention_reduction must leave at least one channel")

        self.attention = CrossAttention(
            self.width,
            self.width,
            reduction=attention_reduction,
            kernel_size=7,
        )
        self.spectral = SpectralConv2d(
            self.width,
            self.width,
            self.modes1,
            self.modes2,
        )
        self.local = nn.Conv2d(self.width, self.width, kernel_size=1)
        self.time_projection = nn.Linear(self.time_channels, self.width)

    def forward(
        self,
        value: Tensor,
        condition: Tensor,
        time_embedding: Tensor,
    ) -> Tensor:
        if value.ndim != 4 or value.shape[1] != self.width:
            raise ValueError(
                f"value must have shape [B,{self.width},H,W]"
            )
        if condition.shape != value.shape:
            raise ValueError("condition feature must match the FNO value shape")
        if (
            time_embedding.ndim != 2
            or time_embedding.shape[0] != value.shape[0]
            or time_embedding.shape[1] != self.time_channels
        ):
            raise ValueError(
                "time embedding must have shape "
                f"[B,{self.time_channels}]"
            )

        attended = self.attention(value, condition)
        if self.padding:
            padded = F.pad(attended, (0, self.padding, 0, self.padding))
            spectral = self.spectral(padded)
            spectral = spectral[..., : attended.shape[-2], : attended.shape[-1]]
        else:
            spectral = self.spectral(attended)
        local = self.local(attended)
        temporal = self.time_projection(nonlinearity(time_embedding))
        temporal = temporal[:, :, None, None]
        return F.gelu(spectral + local + temporal)


class AttentionConditionedFNO2d(nn.Module):
    """RadioFlow condition encoder with a full-resolution FNO velocity field."""

    def __init__(
        self,
        *,
        condition_channels: int = 3,
        width: int = 40,
        modes1: int = 12,
        modes2: int = 12,
        padding: int = 9,
        layers: int = 4,
        encoder_features: Sequence[int] = DEFAULT_ENCODER_FEATURES,
        cfg_drop_prob: float = 0.25,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if condition_channels != 3 or type(condition_channels) is not int:
            raise ValueError("condition_channels is locked to three channels")
        self.condition_channels = condition_channels
        self.width = _positive_int(width, "width")
        self.modes1 = _positive_int(modes1, "modes1")
        self.modes2 = _positive_int(modes2, "modes2")
        if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
            raise ValueError("padding must be a non-negative integer")
        self.padding = padding
        if layers != 4 or type(layers) is not int:
            raise ValueError("layers is locked to four")
        self.layers = layers
        features = tuple(encoder_features)
        if len(features) != 6 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in features
        ):
            raise ValueError("encoder_features must contain six positive integers")
        self.encoder_features = features
        if not math.isfinite(float(cfg_drop_prob)) or not 0.0 <= float(
            cfg_drop_prob
        ) <= 1.0:
            raise ValueError("cfg_drop_prob must lie in [0,1]")
        self.cfg_drop_prob = float(cfg_drop_prob)
        if type(activation_checkpointing) is not bool:
            raise ValueError("activation_checkpointing must be boolean")
        self.activation_checkpointing = activation_checkpointing

        self.condition_encoder = BasicUNetEncoder(
            spatial_dims=2,
            in_channels=self.condition_channels,
            out_channels=features[0],
            features=features,
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm=("instance", {"affine": True}),
            bias=True,
            activation_checkpointing=activation_checkpointing,
        )
        self.condition_projections = nn.ModuleList(
            nn.Conv2d(channels, self.width, kernel_size=1)
            for channels in features[:5]
        )
        self.lifting = nn.Conv2d(
            1 + self.condition_channels + 2,
            self.width,
            kernel_size=1,
        )
        self.time_dense = nn.ModuleList(
            (
                nn.Linear(128, 512),
                nn.Linear(512, 512),
            )
        )
        self.blocks = nn.ModuleList(
            FullResolutionFNOBlock(
                width=self.width,
                modes1=self.modes1,
                modes2=self.modes2,
                padding=self.padding,
            )
            for _ in range(self.layers)
        )
        self.projection_hidden = nn.Conv2d(self.width, 128, kernel_size=1)
        self.projection_output = nn.Conv2d(128, 1, kernel_size=1)

    @staticmethod
    def coordinate_grid(state: Tensor) -> tuple[Tensor, Tensor]:
        if not isinstance(state, Tensor) or state.ndim != 4:
            raise ValueError("state must be an NCHW tensor")
        batch, _, height, width = state.shape
        grid_x = torch.linspace(
            0.0,
            1.0,
            width,
            device=state.device,
            dtype=state.dtype,
        ).view(1, 1, 1, width)
        grid_y = torch.linspace(
            0.0,
            1.0,
            height,
            device=state.device,
            dtype=state.dtype,
        ).view(1, 1, height, 1)
        return (
            grid_x.expand(batch, 1, height, width),
            grid_y.expand(batch, 1, height, width),
        )

    def _validate_condition(self, condition: Tensor) -> None:
        if not isinstance(condition, Tensor) or condition.ndim != 4:
            raise ValueError("condition must be an NCHW tensor")
        if condition.shape[1] != self.condition_channels:
            raise ValueError("condition must contain exactly three channels")

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

    def embed_model(self, condition: Tensor) -> list[Tensor]:
        self._validate_condition(condition)
        return self.condition_encoder(condition)

    def aggregate_condition(
        self,
        embeddings: Sequence[Tensor],
        *,
        output_size: tuple[int, int],
    ) -> Tensor:
        if len(embeddings) != len(self.condition_projections):
            raise ValueError("condition encoder must provide five feature scales")
        height, width = output_size
        if height <= 0 or width <= 0:
            raise ValueError("output_size must be positive")
        result: Tensor | None = None
        for index, (embedding, projection) in enumerate(
            zip(embeddings, self.condition_projections)
        ):
            if not isinstance(embedding, Tensor) or embedding.ndim != 4:
                raise ValueError(f"condition embedding {index} must be NCHW")
            projected = projection(embedding)
            if projected.shape[-2:] != output_size:
                projected = F.interpolate(
                    projected,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            result = projected if result is None else result + projected
        if result is None:
            raise ValueError("condition embeddings cannot be empty")
        return result

    def _time_embedding(self, step: Tensor) -> Tensor:
        embedding = get_timestep_embedding(step, 128)
        embedding = self.time_dense[0](embedding)
        embedding = nonlinearity(embedding)
        return self.time_dense[1](embedding)

    def _velocity(
        self,
        condition: Tensor,
        state: Tensor,
        step: Tensor | float | int,
        embeddings: Sequence[Tensor] | None = None,
    ) -> Tensor:
        self._validate_state_and_condition(condition, state)
        normalized_step = self._normalize_step(step, state)
        condition_embeddings = (
            self.embed_model(condition) if embeddings is None else embeddings
        )
        condition_feature = self.aggregate_condition(
            condition_embeddings,
            output_size=state.shape[-2:],
        )
        grid_x, grid_y = self.coordinate_grid(state)
        value = self.lifting(
            torch.cat((state, condition, grid_x, grid_y), dim=1)
        )
        time_embedding = self._time_embedding(normalized_step)
        for block in self.blocks:
            value = block(value, condition_feature, time_embedding)
        value = F.gelu(self.projection_hidden(value))
        return self.projection_output(value)

    def forward(
        self,
        image: Tensor | None = None,
        x: Tensor | None = None,
        pred_type: str = "denoise",
        step: Tensor | float | int | None = None,
        embedding: Sequence[Tensor] | None = None,
    ) -> Tensor:
        if pred_type != "denoise":
            raise ValueError(
                "AttentionConditionedFNO2d only supports pred_type='denoise'"
            )
        if image is None or x is None or step is None:
            raise ValueError("image, x, and step are required")
        self._validate_state_and_condition(image, x)

        condition = image
        embeddings = embedding
        if self.training and self.cfg_drop_prob > 0.0:
            drop = (
                torch.rand(condition.shape[0], device=condition.device)
                < self.cfg_drop_prob
            )
            if bool(drop.any()):
                condition = condition.masked_fill(drop.view(-1, 1, 1, 1), 0.0)
                embeddings = None
        return self._velocity(condition, x, step, embeddings)

    def forward_with_cfg(
        self,
        *,
        image: Tensor,
        x: Tensor,
        step: Tensor | float | int,
        embedding: Sequence[Tensor] | None = None,
        cfg_scale: float = 1.0,
    ) -> Tensor:
        if not math.isfinite(float(cfg_scale)):
            raise ValueError("cfg_scale must be finite")
        self._validate_state_and_condition(image, x)
        conditional_embeddings = (
            self.embed_model(image) if embedding is None else embedding
        )
        conditional = self._velocity(
            image,
            x,
            step,
            conditional_embeddings,
        )
        if float(cfg_scale) == 1.0:
            return conditional
        unconditional_image = torch.zeros_like(image)
        unconditional = self._velocity(
            unconditional_image,
            x,
            step,
            self.embed_model(unconditional_image),
        )
        return unconditional + float(cfg_scale) * (conditional - unconditional)


__all__ = [
    "AttentionConditionedFNO2d",
    "DEFAULT_ENCODER_FEATURES",
    "FullResolutionFNOBlock",
]

