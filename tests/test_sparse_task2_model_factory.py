from __future__ import annotations

import pytest

from config import MODEL_FEATURES
from model.model import DiffUNet


def _count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_task2_factory_builds_locked_five_channel_lite() -> None:
    from training.model_factory import build_task2_sparse_radioflow

    model = build_task2_sparse_radioflow()
    assert type(model) is DiffUNet
    assert MODEL_FEATURES["lite"] == (32, 32, 64, 128, 256, 32)
    assert model.embed_model.conv_0.conv_0.conv.in_channels == 5
    assert model.cfg_drop_prob == 0.25
    assert model.activation_checkpointing is False
    assert _count(model) == 3_996_011


@pytest.mark.parametrize(
    "kwargs",
    ({"condition_variant": "feature4"}, {"model_size": "large"}),
)
def test_task2_factory_rejects_other_variants(kwargs) -> None:
    from training.model_factory import FrameworkLockError, build_task2_sparse_radioflow

    with pytest.raises(FrameworkLockError):
        build_task2_sparse_radioflow(**kwargs)
