from __future__ import annotations

from model.attention_fno import AttentionConditionedFNO2d
from model.fno import ConditionalFNO2d
from model.model import DiffUNet
from training.model_factory import (
    ATTENTION_FNO_MODEL_SIZE,
    build_attention_fno,
    build_same_frequency_backbone,
)


def test_factory_registers_attention_fno_without_changing_existing_models() -> None:
    attention = build_attention_fno()
    paper = build_same_frequency_backbone("paper_fno_lite")
    unet = build_same_frequency_backbone("lite")

    assert ATTENTION_FNO_MODEL_SIZE == "attention_fno_lite"
    assert type(attention) is AttentionConditionedFNO2d
    assert type(build_same_frequency_backbone(ATTENTION_FNO_MODEL_SIZE)) is AttentionConditionedFNO2d
    assert type(paper) is ConditionalFNO2d
    assert type(unet) is DiffUNet
    assert attention.width == 40
    assert (attention.modes1, attention.modes2) == (12, 12)
    assert attention.padding == 9
    assert len(attention.blocks) == 4

