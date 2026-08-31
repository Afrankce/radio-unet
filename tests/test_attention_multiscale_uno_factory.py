from __future__ import annotations

from model.attention_fno import AttentionConditionedFNO2d
from model.attention_multiscale_uno import AttentionMultiscaleUNO2d
from model.fno import count_real_scalar_parameters, count_tensor_parameters
from model.model import DiffUNet
from model.unet.basic_unet_denose import CrossAttention
from training.model_factory import (
    MULTISCALE_UNO_MODEL_SIZE,
    build_attention_multiscale_uno,
    build_same_frequency_backbone,
)


def test_factory_registers_multiscale_uno_without_mutating_existing_models() -> None:
    """A wrong registered architecture or factory branch must fail here."""

    model = build_attention_multiscale_uno()

    assert MULTISCALE_UNO_MODEL_SIZE == "attention_multiscale_uno_lite"
    assert type(model) is AttentionMultiscaleUNO2d
    assert tuple(model.state_channels) == (32, 64, 128, 256, 256)
    assert model.operator_width == 24
    assert tuple(model.operator_modes) == (12, 12, 8, 4, 4)
    assert tuple(model.operator_padding) == (9, 5, 3, 2, 1)
    assert tuple(model.encoder_features) == (32, 32, 64, 128, 256, 32)
    assert model.condition_channels == 3
    assert model.cfg_drop_prob == 0.25
    assert sum(isinstance(module, CrossAttention) for module in model.modules()) == 9
    assert count_tensor_parameters(model) == 3_059_355
    assert count_real_scalar_parameters(model) == 3_925_659
    assert type(build_same_frequency_backbone(MULTISCALE_UNO_MODEL_SIZE)) is AttentionMultiscaleUNO2d
    assert type(build_same_frequency_backbone("attention_fno_lite")) is AttentionConditionedFNO2d
    assert type(build_same_frequency_backbone("lite")) is DiffUNet
