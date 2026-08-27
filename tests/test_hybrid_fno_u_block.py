from __future__ import annotations

import pytest
import torch

from model.fno import count_real_scalar_parameters, count_tensor_parameters
from model.hybrid_fno_u import FNOOperatorBlock


def test_operator_block_preserves_shape_and_has_finite_gradients() -> None:
    block = FNOOperatorBlock(4, 32, width=24, modes=12, padding=9)
    value = torch.randn(1, 4, 32, 32, requires_grad=True)
    temb = torch.randn(1, 512)

    output = block(value, temb)
    output.square().mean().backward()

    assert output.shape == (1, 32, 32, 32)
    assert torch.isfinite(output).all()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in block.parameters()
        if parameter.requires_grad
    )


def test_operator_block_parameter_counts_include_complex_real_and_imaginary_parts() -> None:
    block = FNOOperatorBlock(4, 32, width=24, modes=12, padding=9)
    complex_elements = 2 * 24 * 24 * 12 * 12
    noncomplex_elements = (
        (4 * 24 + 24)
        + (24 * 24 + 24)
        + (512 * 24 + 24)
        + (24 * 32 + 32)
    )

    assert count_tensor_parameters(block) == complex_elements + noncomplex_elements
    assert count_real_scalar_parameters(block) == 2 * complex_elements + noncomplex_elements


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("in_channels", 0, "in_channels"),
        ("out_channels", 0, "out_channels"),
        ("width", 0, "width"),
        ("modes", 0, "modes"),
        ("padding", -1, "padding"),
    ),
)
def test_operator_block_rejects_invalid_controls(
    field: str,
    value: int,
    message: str,
) -> None:
    controls = {
        "in_channels": 4,
        "out_channels": 32,
        "width": 24,
        "modes": 12,
        "padding": 9,
    }
    controls[field] = value

    with pytest.raises(ValueError, match=message):
        FNOOperatorBlock(**controls)


def test_operator_block_rejects_modes_that_do_not_fit_the_padded_grid() -> None:
    block = FNOOperatorBlock(4, 32, width=24, modes=12, padding=0)

    with pytest.raises(ValueError, match="retained modes"):
        block(torch.zeros(1, 4, 16, 16), torch.zeros(1, 512))


def test_operator_block_rejects_invalid_input_and_time_shapes() -> None:
    block = FNOOperatorBlock(4, 32, width=24, modes=4, padding=1)

    with pytest.raises(ValueError, match="NCHW"):
        block(torch.zeros(1, 4, 16), torch.zeros(1, 512))
    with pytest.raises(ValueError, match="input channel"):
        block(torch.zeros(1, 3, 16, 16), torch.zeros(1, 512))
    with pytest.raises(ValueError, match="time embedding"):
        block(torch.zeros(1, 4, 16, 16), torch.zeros(1, 511))
    with pytest.raises(ValueError, match="batch"):
        block(torch.zeros(2, 4, 16, 16), torch.zeros(1, 512))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_operator_block_runs_under_cuda_float16_autocast() -> None:
    block = FNOOperatorBlock(4, 32, width=24, modes=12, padding=9).cuda()
    value = torch.randn(1, 4, 32, 32, device="cuda", requires_grad=True)
    temb = torch.randn(1, 512, device="cuda")

    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
        output = block(value, temb)
        loss = output.float().square().mean()
    loss.backward()

    assert output.shape == (1, 32, 32, 32)
    assert torch.isfinite(output).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in block.parameters()
        if parameter.requires_grad
    )
