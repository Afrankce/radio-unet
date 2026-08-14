from __future__ import annotations

import torch
from torch import Tensor

from training.sparse_flow import build_masked_flow_pair
from training.sparse_task2_flow import build_task2_flow_pair


def build_sparse_consistent_flow_pair(
    *,
    arm: str,
    x0: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    observation_mask: Tensor,
    sparse_map: Tensor,
    time: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the registered full-target or pinned-observation FM pair."""

    if arm == "multiscale_consistent":
        return build_masked_flow_pair(
            x0,
            target,
            sparse_map,
            observation_mask,
            valid_mask,
            time=time,
        )
    return build_task2_flow_pair(x0, target, valid_mask, time)


__all__ = ["build_sparse_consistent_flow_pair"]
