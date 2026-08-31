from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from model.fno import SpectralConv2d
from model.unet.basic_unet import BasicUNetEncoder
from model.unet.basic_unet_denose import (
    CrossAttention,
    get_timestep_embedding,
    nonlinearity,
)


DEFAULT_STATE_CHANNELS = (32, 64, 128, 256, 256)
DEFAULT_OPERATOR_MODES = (12, 12, 8, 4, 4)
DEFAULT_OPERATOR_PADDING = (9, 5, 3, 2, 1)
DEFAULT_ENCODER_FEATURES = (32, 32, 64, 128, 256, 32)


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


class _Downsample2d(nn.Module):
    """Deterministic half-resolution state projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self._in_channels = _positive_int(in_channels, "in_channels")
        self._out_channels = _positive_int(out_channels, "out_channels")
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.projection = nn.Conv2d(
            self._in_channels,
            self._out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, value: Tensor) -> Tensor:
        if (
            not isinstance(value, Tensor)
            or value.ndim != 4
            or value.shape[1] != self._in_channels
        ):
            raise ValueError(
                f"value must have shape [B,{self._in_channels},H,W]"
            )
        if value.shape[-2] % 2 or value.shape[-1] % 2:
            raise ValueError("downsample input must have even spatial dimensions")
        return self.projection(self.pool(value))


class _UpsampleFuse2d(nn.Module):
    """Resize to an encoder skip, concatenate, and compress with a 1x1 map."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self._in_channels = _positive_int(in_channels, "in_channels")
        self._skip_channels = _positive_int(skip_channels, "skip_channels")
        self._out_channels = _positive_int(out_channels, "out_channels")
        self.projection = nn.Conv2d(
            self._in_channels + self._skip_channels,
            self._out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        if (
            not isinstance(value, Tensor)
            or value.ndim != 4
            or value.shape[1] != self._in_channels
        ):
            raise ValueError(
                f"value must have shape [B,{self._in_channels},H,W]"
            )
        if (
            not isinstance(skip, Tensor)
            or skip.ndim != 4
            or skip.shape[1] != self._skip_channels
        ):
            raise ValueError(
                f"skip must have shape [B,{self._skip_channels},H,W]"
            )
        if value.shape[0] != skip.shape[0]:
            raise ValueError("value and skip batch dimensions must match")
        if value.device != skip.device or value.dtype != skip.dtype:
            raise ValueError("value and skip must share device and dtype")
        resized = F.interpolate(
            value,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if resized.shape[0] != skip.shape[0] or resized.shape[-2:] != skip.shape[-2:]:
            raise ValueError("resized value must match skip batch/spatial dimensions")
        return self.projection(torch.cat((resized, skip), dim=1))


def _positive_int_sequence(
    values: Sequence[int],
    *,
    length: int,
    name: str,
) -> tuple[int, ...]:
    result = tuple(values)
    if len(result) != length or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in result
    ):
        raise ValueError(f"{name} must contain {length} positive integers")
    return result


class AttentionMultiscaleUNO2d(nn.Module):
    """RadioFlow condition encoder with a U-shaped multiscale FNO velocity field."""

    def __init__(
        self,
        *,
        condition_channels: int = 3,
        state_channels: Sequence[int] = DEFAULT_STATE_CHANNELS,
        operator_width: int = 24,
        operator_modes: Sequence[int] = DEFAULT_OPERATOR_MODES,
        operator_padding: Sequence[int] = DEFAULT_OPERATOR_PADDING,
        encoder_features: Sequence[int] = DEFAULT_ENCODER_FEATURES,
        cfg_drop_prob: float = 0.25,
        attention_reduction: int = 16,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if condition_channels != 3 or type(condition_channels) is not int:
            raise ValueError("condition_channels is locked to three channels")
        self.condition_channels = condition_channels
        self.state_channels = _positive_int_sequence(
            state_channels,
            length=5,
            name="state_channels",
        )
        self.operator_width = _positive_int(operator_width, "operator_width")
        self.operator_modes = _positive_int_sequence(
            operator_modes,
            length=5,
            name="operator_modes",
        )
        padding = tuple(operator_padding)
        if len(padding) != 5 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in padding
        ):
            raise ValueError(
                "operator_padding must contain five non-negative integers"
            )
        self.operator_padding = padding
        self.encoder_features = _positive_int_sequence(
            encoder_features,
            length=6,
            name="encoder_features",
        )
        try:
            finite_drop_probability = math.isfinite(float(cfg_drop_prob))
        except (TypeError, ValueError):
            finite_drop_probability = False
        if not finite_drop_probability or not 0.0 <= float(cfg_drop_prob) <= 1.0:
            raise ValueError("cfg_drop_prob must lie in [0,1]")
        self.cfg_drop_prob = float(cfg_drop_prob)
        self.attention_reduction = _positive_int(
            attention_reduction,
            "attention_reduction",
        )
        if any(
            channels // self.attention_reduction <= 0
            for channels in self.state_channels
        ):
            raise ValueError("attention_reduction must leave at least one channel")
        if type(activation_checkpointing) is not bool:
            raise ValueError("activation_checkpointing must be boolean")
        self.activation_checkpointing = activation_checkpointing

        self.condition_encoder = BasicUNetEncoder(
            spatial_dims=2,
            in_channels=self.condition_channels,
            out_channels=self.encoder_features[0],
            features=self.encoder_features,
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm=("instance", {"affine": True}),
            bias=True,
            activation_checkpointing=activation_checkpointing,
        )
        self.lifting = nn.Conv2d(
            1 + self.condition_channels + 2,
            self.state_channels[0],
            kernel_size=1,
        )
        self.time_dense = nn.ModuleList(
            (
                nn.Linear(128, 512),
                nn.Linear(512, 512),
            )
        )
        self.encoder_stages = nn.ModuleList(
            AttentionConditionedFNOStage(
                channels=self.state_channels[index],
                embedding_channels=self.encoder_features[index],
                operator_width=self.operator_width,
                modes=self.operator_modes[index],
                padding=self.operator_padding[index],
                attention_reduction=self.attention_reduction,
            )
            for index in range(4)
        )
        self.downsamples = nn.ModuleList(
            _Downsample2d(
                self.state_channels[index],
                self.state_channels[index + 1],
            )
            for index in range(4)
        )
        self.bottleneck = AttentionConditionedFNOStage(
            channels=self.state_channels[4],
            embedding_channels=self.encoder_features[4],
            operator_width=self.operator_width,
            modes=self.operator_modes[4],
            padding=self.operator_padding[4],
            attention_reduction=self.attention_reduction,
        )
        decoder_levels = tuple(reversed(range(4)))
        self.upsample_fusions = nn.ModuleList(
            _UpsampleFuse2d(
                self.state_channels[index + 1],
                self.state_channels[index],
                self.state_channels[index],
            )
            for index in decoder_levels
        )
        self.decoder_stages = nn.ModuleList(
            AttentionConditionedFNOStage(
                channels=self.state_channels[index],
                embedding_channels=self.encoder_features[index],
                operator_width=self.operator_width,
                modes=self.operator_modes[index],
                padding=self.operator_padding[index],
                attention_reduction=self.attention_reduction,
            )
            for index in decoder_levels
        )
        self.projection_hidden = nn.Conv2d(
            self.state_channels[0],
            128,
            kernel_size=1,
        )
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
        if (
            state.shape[0] != condition.shape[0]
            or state.shape[-2:] != condition.shape[-2:]
        ):
            raise ValueError("state and condition batch/spatial dimensions must match")
        if state.device != condition.device or state.dtype != condition.dtype:
            raise ValueError("state and condition must share device and dtype")
        if state.shape[-2] % 16 or state.shape[-1] % 16:
            raise ValueError("state spatial dimensions must be divisible by 16")

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

    def _validate_embeddings(
        self,
        embeddings: Sequence[Tensor],
        state: Tensor,
    ) -> tuple[Tensor, ...]:
        values = tuple(embeddings)
        if len(values) != 5:
            raise ValueError("condition encoder must provide five feature scales")
        for index, (embedding, channels) in enumerate(
            zip(values, self.encoder_features[:5])
        ):
            expected_spatial = (
                state.shape[-2] // (2**index),
                state.shape[-1] // (2**index),
            )
            if (
                not isinstance(embedding, Tensor)
                or embedding.ndim != 4
                or embedding.shape[0] != state.shape[0]
                or embedding.shape[1] != channels
                or embedding.shape[-2:] != expected_spatial
            ):
                raise ValueError(
                    f"condition embedding {index} must have shape "
                    f"[B,{channels},{expected_spatial[0]},{expected_spatial[1]}]"
                )
            if embedding.device != state.device:
                raise ValueError(
                    f"condition embedding {index} must share the state device"
                )
        return values

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
        condition_embeddings = self._validate_embeddings(
            self.embed_model(condition) if embeddings is None else embeddings,
            state,
        )
        grid_x, grid_y = self.coordinate_grid(state)
        value = self.lifting(
            torch.cat((state, condition, grid_x, grid_y), dim=1)
        )
        time_embedding = self._time_embedding(normalized_step)

        skips: list[Tensor] = []
        for index, (stage, downsample) in enumerate(
            zip(self.encoder_stages, self.downsamples)
        ):
            value = stage(value, condition_embeddings[index], time_embedding)
            skips.append(value)
            value = downsample(value)

        value = self.bottleneck(value, condition_embeddings[4], time_embedding)
        for decoder_index, level in enumerate(reversed(range(4))):
            value = self.upsample_fusions[decoder_index](value, skips[level])
            value = self.decoder_stages[decoder_index](
                value,
                condition_embeddings[level],
                time_embedding,
            )

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
                "AttentionMultiscaleUNO2d only supports pred_type='denoise'"
            )
        if image is None or x is None or step is None:
            raise ValueError("image, x, and step are required")
        self._validate_state_and_condition(image, x)

        condition = image
        embeddings: Sequence[Tensor] = (
            self.embed_model(image) if embedding is None else embedding
        )
        embeddings = self._validate_embeddings(embeddings, x)
        if self.training and self.cfg_drop_prob > 0.0:
            drop = (
                torch.rand(condition.shape[0], device=condition.device)
                < self.cfg_drop_prob
            )
            if bool(drop.any()):
                drop_mask = drop.view(-1, 1, 1, 1)
                condition = condition.masked_fill(drop_mask, 0.0)
                embeddings = tuple(
                    value.masked_fill(drop_mask, 0.0) for value in embeddings
                )
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
        try:
            finite_cfg_scale = math.isfinite(float(cfg_scale))
        except (TypeError, ValueError):
            finite_cfg_scale = False
        if not finite_cfg_scale:
            raise ValueError("cfg_scale must be finite")
        self._validate_state_and_condition(image, x)
        conditional_embeddings = self._validate_embeddings(
            self.embed_model(image) if embedding is None else embedding,
            x,
        )
        conditional = self._velocity(
            image,
            x,
            step,
            conditional_embeddings,
        )
        if float(cfg_scale) == 1.0:
            return conditional
        unconditional = self._velocity(
            torch.zeros_like(image),
            x,
            step,
            tuple(torch.zeros_like(value) for value in conditional_embeddings),
        )
        return unconditional + float(cfg_scale) * (conditional - unconditional)


__all__ = [
    "AttentionConditionedFNOStage",
    "AttentionMultiscaleUNO2d",
    "DEFAULT_ENCODER_FEATURES",
    "DEFAULT_OPERATOR_MODES",
    "DEFAULT_OPERATOR_PADDING",
    "DEFAULT_STATE_CHANNELS",
]
