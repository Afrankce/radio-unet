from __future__ import annotations

import pytest

from config import MODEL_FEATURES
from model.model import DiffUNet
from model.unet.basic_unet import BasicUNetEncoder
from model.unet.basic_unet_denose import BasicUNetDe


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_sparse_factory_builds_locked_beam_masked_lite_model() -> None:
    from training.model_factory import build_sparse_radioflow

    model = build_sparse_radioflow(variant="beam_masked", model_size="lite")

    assert type(model) is DiffUNet
    assert type(model.embed_model) is BasicUNetEncoder
    assert type(model.model) is BasicUNetDe
    assert MODEL_FEATURES["lite"] == (32, 32, 64, 128, 256, 32)
    assert model.embed_model.conv_0.conv_0.conv.in_channels == 5
    assert model.cfg_drop_prob == 0.25
    assert model.activation_checkpointing is False
    assert model.embed_model.activation_checkpointing is False
    assert model.model.activation_checkpointing is False
    assert _parameter_count(model) == 3_996_011


@pytest.mark.parametrize(
    ("variant", "model_size"),
    [
        ("no_beam_masked", "lite"),
        ("no_beam", "lite"),
        ("beam_masked", "large"),
    ],
)
def test_sparse_factory_rejects_nonformal_variant_or_model_size(
    variant: str,
    model_size: str,
) -> None:
    from training.model_factory import FrameworkLockError, build_sparse_radioflow

    with pytest.raises(FrameworkLockError):
        build_sparse_radioflow(variant=variant, model_size=model_size)
