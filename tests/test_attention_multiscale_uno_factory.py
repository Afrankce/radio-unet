from __future__ import annotations

import pytest
import torch

import training.model_factory as model_factory
from model.attention_fno import AttentionConditionedFNO2d
from model.attention_multiscale_uno import AttentionMultiscaleUNO2d
from model.fno import count_real_scalar_parameters, count_tensor_parameters
from model.model import DiffUNet
from model.unet.basic_unet_denose import CrossAttention
from training.model_factory import (
    FrameworkLockError,
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


def test_factory_guard_rejects_real_model_with_wrong_operator_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drifted = AttentionMultiscaleUNO2d(operator_width=25)
    monkeypatch.setattr(model_factory, "AttentionMultiscaleUNO2d", lambda: drifted)

    with pytest.raises(FrameworkLockError, match="operator width"):
        build_attention_multiscale_uno()


def test_factory_guard_rejects_real_model_with_extra_trainable_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drifted = AttentionMultiscaleUNO2d()
    drifted.register_parameter(
        "unexpected_trainable_parameter",
        torch.nn.Parameter(torch.zeros(1)),
    )
    monkeypatch.setattr(model_factory, "AttentionMultiscaleUNO2d", lambda: drifted)

    with pytest.raises(FrameworkLockError, match="tensor parameter count"):
        build_attention_multiscale_uno()
