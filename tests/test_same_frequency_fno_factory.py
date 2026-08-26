from __future__ import annotations

import pytest

from model.fno import (
    ConditionalFNO2d,
    count_real_scalar_parameters,
    count_tensor_parameters,
)
from model.model import DiffUNet
from training.model_factory import (
    EXPECTED_PARAMETER_COUNTS,
    FrameworkLockError,
    build_same_frequency_backbone,
)
from training.same_frequency_fno_config import PAPER_FNO_MODEL_SIZE


def test_same_frequency_factory_preserves_unet_and_builds_locked_fno() -> None:
    unet = build_same_frequency_backbone("lite")
    fno = build_same_frequency_backbone(PAPER_FNO_MODEL_SIZE)

    assert type(unet) is DiffUNet
    assert sum(parameter.numel() for parameter in unet.parameters()) == 3_994_859
    assert isinstance(fno, ConditionalFNO2d)
    assert count_tensor_parameters(fno) == 1_855_457
    assert count_real_scalar_parameters(fno) == 3_698_657
    assert EXPECTED_PARAMETER_COUNTS == {
        "lite": 3_994_859,
        "large": 54_126_059,
    }


def test_same_frequency_factory_rejects_unknown_model_size() -> None:
    with pytest.raises(FrameworkLockError, match="unsupported"):
        build_same_frequency_backbone("not-a-model")

