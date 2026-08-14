from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.model import DiffUNet
from training.model_factory import build_locked_radioflow, build_task2_sparse_radioflow


class SparseConsistentModelError(ValueError):
    """A sparse-aware model input or embedding is invalid."""


class MaskAwareSparseDiffUNet(nn.Module):
    """RadioFlow Lite with mask-aware multi-scale sparse feature injection.

    The environment encoder remains the locked three-channel RadioFlow
    encoder.  At each encoder resolution, sparse values are average pooled
    only over observed pixels and concatenated with pooled coverage.  A
    zero-initialized scalar gate makes the additional branch a controlled
    residual ablation at initialization.
    """

    _FEATURE_CHANNELS = (32, 32, 64, 128, 256)

    def __init__(self) -> None:
        super().__init__()
        self.base = build_locked_radioflow("lite")
        self.cfg_drop_prob = self.base.cfg_drop_prob
        self.sparse_projections = nn.ModuleList(
            nn.Conv2d(2, channels, kernel_size=1)
            for channels in self._FEATURE_CHANNELS
        )
        self.sparse_gates = nn.Parameter(torch.zeros(len(self._FEATURE_CHANNELS)))

    @staticmethod
    def _validate_sparse_inputs(
        sparse_map: Tensor,
        observation_mask: Tensor,
        reference: Tensor,
    ) -> None:
        if sparse_map.ndim != 4 or sparse_map.shape[1] != 1:
            raise SparseConsistentModelError("sparse_map must have shape [B,1,H,W]")
        if observation_mask.shape != sparse_map.shape:
            raise SparseConsistentModelError("observation_mask must match sparse_map")
        if observation_mask.dtype is not torch.bool:
            raise SparseConsistentModelError("observation_mask must be boolean")
        if sparse_map.device != reference.device or observation_mask.device != reference.device:
            raise SparseConsistentModelError("sparse inputs must share the condition device")
        if not bool(torch.isfinite(sparse_map).all()):
            raise SparseConsistentModelError("sparse_map must be finite")
        if bool((sparse_map.masked_select(~observation_mask)).abs().max().item() > 1e-6):
            raise SparseConsistentModelError("sparse_map must be zero outside observations")

    def embed_model(
        self,
        condition: Tensor,
        sparse_map: Tensor | None = None,
        observation_mask: Tensor | None = None,
    ) -> list[Tensor]:
        if condition.ndim != 4 or condition.shape[1] != 3:
            raise SparseConsistentModelError(
                "mask-aware RadioFlow environment condition must have shape [B,3,H,W]"
            )
        embeddings = self.base.embed_model(condition)
        if sparse_map is None or observation_mask is None:
            if sparse_map is not None or observation_mask is not None:
                raise SparseConsistentModelError(
                    "sparse_map and observation_mask must be supplied together"
                )
            return embeddings
        self._validate_sparse_inputs(sparse_map, observation_mask, condition)
        coverage = observation_mask.to(dtype=sparse_map.dtype)
        for index, (feature, projection) in enumerate(
            zip(embeddings, self.sparse_projections)
        ):
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
            sparse_feature = projection(
                torch.cat((pooled_value, pooled_coverage), dim=1)
            )
            if sparse_feature.shape[-2:] != feature.shape[-2:]:
                raise SparseConsistentModelError(
                    "sparse encoder scale does not match RadioFlow encoder scale"
                )
            embeddings[index] = feature + torch.tanh(self.sparse_gates[index]) * sparse_feature
        return embeddings

    @staticmethod
    def _step_tensor(step: Tensor | int | float, x: Tensor) -> Tensor:
        if isinstance(step, int) or isinstance(step, float):
            step = torch.tensor([step], device=x.device, dtype=x.dtype)
        if not isinstance(step, Tensor):
            raise SparseConsistentModelError("step must be a scalar or tensor")
        if step.ndim == 0:
            step = step.repeat(x.shape[0])
        if step.ndim != 1 or step.shape[0] not in {1, x.shape[0]}:
            raise SparseConsistentModelError("step must have shape [B]")
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
            raise SparseConsistentModelError("only denoise prediction is supported")
        if embedding is None:
            embedding = self.embed_model(image)
        embeddings = list(embedding)
        step_tensor = self._step_tensor(step, x)
        if self.training:
            drop = torch.rand(
                embeddings[0].shape[0], device=embeddings[0].device
            ) < self.cfg_drop_prob
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
        if embedding is None:
            embedding = self.embed_model(image)
        step_tensor = self._step_tensor(step, x)
        uncond = [torch.zeros_like(value) for value in embedding]
        pred_uncond = self.base.model(
            x, t=step_tensor, image=image, embeddings=uncond
        )
        pred_cond = self.base.model(
            x, t=step_tensor, image=image, embeddings=embedding
        )
        return pred_uncond + float(cfg_scale) * (pred_cond - pred_uncond)


def build_sparse_consistent_model(
    arm: Literal[
        "environment_only",
        "concat_fullfm",
        "multiscale_fullfm",
        "multiscale_consistent",
    ],
) -> nn.Module:
    if arm == "environment_only":
        return build_locked_radioflow("lite")
    if arm == "concat_fullfm":
        return build_task2_sparse_radioflow(condition_variant="feature5_mask")
    if arm in {"multiscale_fullfm", "multiscale_consistent"}:
        return MaskAwareSparseDiffUNet()
    raise SparseConsistentModelError(f"unsupported arm: {arm}")


def embed_sparse_consistent_model(
    model: nn.Module,
    *,
    arm: str,
    condition: Tensor,
    sparse_map: Tensor,
    observation_mask: Tensor,
) -> list[Tensor]:
    if arm in {"multiscale_fullfm", "multiscale_consistent"}:
        return model.embed_model(condition, sparse_map, observation_mask)
    return model.embed_model(condition)


__all__ = [
    "MaskAwareSparseDiffUNet",
    "SparseConsistentModelError",
    "build_sparse_consistent_model",
    "embed_sparse_consistent_model",
]
