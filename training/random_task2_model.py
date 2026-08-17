from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import MODEL_FEATURES
from model.model import DiffUNet


class RandomTask2ModelError(ValueError):
    """A random Task 2 sparse-aware model input is invalid."""


class RandomTask2PinnedFMModel(nn.Module):
    """Lite RadioFlow with multi-scale sparse-map injection for pinned FM."""

    _FEATURE_CHANNELS = (32, 32, 64, 128, 256)

    def __init__(self, *, condition_channels: int) -> None:
        super().__init__()
        if condition_channels not in {4, 5}:
            raise RandomTask2ModelError("condition_channels must be 4 or 5")
        if tuple(MODEL_FEATURES.get("lite", ())) != (32, 32, 64, 128, 256, 32):
            raise RandomTask2ModelError("RadioFlow Lite feature tuple changed")
        self.condition_channels = condition_channels
        self.base = DiffUNet(
            con_channels=condition_channels,
            model_size="lite",
            activation_checkpointing=False,
        )
        self.cfg_drop_prob = self.base.cfg_drop_prob
        self.sparse_projections = nn.ModuleList(
            nn.Conv2d(2, channels, kernel_size=1)
            for channels in self._FEATURE_CHANNELS
        )
        self.sparse_gates = nn.Parameter(torch.full((len(self._FEATURE_CHANNELS),), 0.05))

    def _validate_condition(self, condition: Tensor) -> None:
        if condition.ndim != 4 or condition.shape[1] != self.condition_channels:
            raise RandomTask2ModelError(
                f"condition must have shape [B,{self.condition_channels},H,W]"
            )
        if not condition.is_floating_point():
            raise RandomTask2ModelError("condition must be floating point")
        if not bool(torch.isfinite(condition).all()):
            raise RandomTask2ModelError("condition must be finite")

    @staticmethod
    def _validate_sparse_inputs(
        sparse_map: Tensor,
        observation_mask: Tensor,
        reference: Tensor,
    ) -> None:
        if sparse_map.ndim != 4 or sparse_map.shape[1] != 1:
            raise RandomTask2ModelError("sparse_map must have shape [B,1,H,W]")
        if observation_mask.shape != sparse_map.shape:
            raise RandomTask2ModelError("observation_mask must match sparse_map")
        if observation_mask.dtype is not torch.bool:
            raise RandomTask2ModelError("observation_mask must be boolean")
        if sparse_map.device != reference.device or observation_mask.device != reference.device:
            raise RandomTask2ModelError("sparse inputs must share the condition device")
        if not bool(torch.isfinite(sparse_map).all()):
            raise RandomTask2ModelError("sparse_map must be finite")
        if bool((sparse_map.masked_select(~observation_mask)).abs().max().item() > 1e-6):
            raise RandomTask2ModelError("sparse_map must be zero outside observations")

    def embed_model(
        self,
        condition: Tensor,
        sparse_map: Tensor,
        observation_mask: Tensor,
    ) -> list[Tensor]:
        self._validate_condition(condition)
        self._validate_sparse_inputs(sparse_map, observation_mask, condition)
        embeddings = self.base.embed_model(condition)
        coverage = observation_mask.to(dtype=sparse_map.dtype)
        for index, (feature, projection) in enumerate(zip(embeddings, self.sparse_projections)):
            scale = 2**index
            pooled_value = F.avg_pool2d(
                sparse_map * coverage,
                kernel_size=scale,
                stride=scale,
            )
            pooled_coverage = F.avg_pool2d(
                coverage,
                kernel_size=scale,
                stride=scale,
            )
            pooled_value = torch.where(
                pooled_coverage > 1e-6,
                pooled_value / pooled_coverage.clamp_min(1e-6),
                torch.zeros_like(pooled_value),
            )
            sparse_feature = projection(torch.cat((pooled_value, pooled_coverage), dim=1))
            if sparse_feature.shape[-2:] != feature.shape[-2:]:
                raise RandomTask2ModelError("sparse encoder scale does not match RadioFlow encoder scale")
            embeddings[index] = feature + torch.tanh(self.sparse_gates[index]) * sparse_feature
        return embeddings

    @staticmethod
    def _step_tensor(step: Tensor | int | float, x: Tensor) -> Tensor:
        if isinstance(step, int) or isinstance(step, float):
            step = torch.tensor([step], device=x.device, dtype=x.dtype)
        if not isinstance(step, Tensor):
            raise RandomTask2ModelError("step must be a scalar or tensor")
        if step.ndim == 0:
            step = step.repeat(x.shape[0])
        if step.ndim != 1 or step.shape[0] not in {1, x.shape[0]}:
            raise RandomTask2ModelError("step must have shape [B]")
        if step.shape[0] == 1 and x.shape[0] != 1:
            step = step.repeat(x.shape[0])
        return step.to(device=x.device, dtype=x.dtype)

    def forward(
        self,
        *,
        image: Tensor,
        x: Tensor,
        pred_type: str = "denoise",
        step: Tensor | int | float,
        embedding: list[Tensor] | None = None,
    ) -> Tensor:
        if pred_type != "denoise":
            raise RandomTask2ModelError("only denoise prediction is supported")
        self._validate_condition(image)
        if x.ndim != 4 or x.shape[1] != 1:
            raise RandomTask2ModelError("x must have shape [B,1,H,W]")
        if x.shape[0] != image.shape[0] or x.shape[-2:] != image.shape[-2:]:
            raise RandomTask2ModelError("x must match image batch and spatial dimensions")
        if embedding is None:
            raise RandomTask2ModelError("embedding is required for pinned FM forward")
        embeddings = list(embedding)
        step_tensor = self._step_tensor(step, x)
        if self.training:
            drop = torch.rand(embeddings[0].shape[0], device=embeddings[0].device) < self.cfg_drop_prob
            if bool(drop.any()):
                view = drop.view(-1, 1, 1, 1)
                embeddings = [value.masked_fill(view, 0.0) for value in embeddings]
        return self.base.model(x, t=step_tensor, image=image, embeddings=embeddings)

    def forward_with_cfg(
        self,
        *,
        image: Tensor,
        x: Tensor,
        step: Tensor | int | float,
        embedding: list[Tensor] | None = None,
        cfg_scale: float = 1.0,
    ) -> Tensor:
        self._validate_condition(image)
        if embedding is None:
            raise RandomTask2ModelError("embedding is required for pinned FM CFG")
        step_tensor = self._step_tensor(step, x)
        uncond = [torch.zeros_like(value) for value in embedding]
        pred_uncond = self.base.model(x, t=step_tensor, image=image, embeddings=uncond)
        pred_cond = self.base.model(x, t=step_tensor, image=image, embeddings=embedding)
        return pred_uncond + float(cfg_scale) * (pred_cond - pred_uncond)


def build_random_task2_pinned_model(
    *,
    condition_variant: Literal["feature4", "feature5_mask"],
    model_size: Literal["lite"] = "lite",
) -> RandomTask2PinnedFMModel:
    if condition_variant not in {"feature4", "feature5_mask"}:
        raise RandomTask2ModelError(f"unsupported condition_variant: {condition_variant}")
    if model_size != "lite":
        raise RandomTask2ModelError("random Task 2 pinned model is locked to model_size='lite'")
    return RandomTask2PinnedFMModel(
        condition_channels=4 if condition_variant == "feature4" else 5
    )


__all__ = [
    "RandomTask2ModelError",
    "RandomTask2PinnedFMModel",
    "build_random_task2_pinned_model",
]
