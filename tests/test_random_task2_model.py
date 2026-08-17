from __future__ import annotations

from pathlib import Path

import pytest
import torch

from training.random_task2_config import RandomTask2TrainConfig
from training.random_task2_trainer import build_random_task2_model


def _cfg(tmp_path: Path, *, variant: str) -> RandomTask2TrainConfig:
    return RandomTask2TrainConfig(
        dataset_root=tmp_path / "dataset",
        manifest_path=tmp_path / "manifest.jsonl",
        height_stats_path=tmp_path / "height.json",
        run_root=tmp_path / "runs",
        array_size="8x8",
        variant=variant,
        mode="pinned_fm",
    )


@pytest.mark.parametrize(("variant", "condition_channels"), (("feature4", 4), ("feature5_mask", 5)))
def test_pinned_model_supports_both_condition_widths_and_keeps_shape(
    tmp_path: Path,
    variant: str,
    condition_channels: int,
) -> None:
    cfg = _cfg(tmp_path, variant=variant)
    model = build_random_task2_model(cfg)
    condition = torch.randn(2, condition_channels, 32, 32)
    xt = torch.randn(2, 1, 32, 32)
    sparse_map = torch.zeros_like(xt)
    observation_mask = torch.zeros_like(xt, dtype=torch.bool)
    observation_mask[..., 0, 0] = True
    sparse_map[..., 0, 0] = 2.5
    step = torch.tensor([0.0, 0.5], dtype=xt.dtype)

    embedding = model.embed_model(condition, sparse_map, observation_mask)
    prediction = model(
        image=condition,
        x=xt,
        pred_type="denoise",
        step=step,
        embedding=embedding,
    )
    guided = model.forward_with_cfg(
        image=condition,
        x=xt,
        step=step,
        embedding=embedding,
        cfg_scale=1.5,
    )

    assert prediction.shape == xt.shape
    assert guided.shape == xt.shape
    assert hasattr(model, "sparse_gates")
    assert model.sparse_gates.shape[0] == 5
    assert torch.count_nonzero(model.sparse_gates).item() == 5
