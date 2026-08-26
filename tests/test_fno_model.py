from __future__ import annotations

import pytest
import torch

from model.fno import (
    ConditionalFNO2d,
    SpectralConv2d,
    count_real_scalar_parameters,
    count_tensor_parameters,
)


def test_spectral_conv_matches_hand_built_two_corner_fft() -> None:
    layer = SpectralConv2d(1, 1, modes1=2, modes2=2)
    with torch.no_grad():
        layer.weights1.fill_(1.0 + 0.0j)
        layer.weights2.fill_(2.0 + 0.0j)
    value = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) / 64.0

    value_ft = torch.fft.rfft2(value)
    expected_ft = torch.zeros(1, 1, 8, 5, dtype=torch.cfloat)
    expected_ft[:, :, :2, :2] = value_ft[:, :, :2, :2]
    expected_ft[:, :, -2:, :2] = 2.0 * value_ft[:, :, -2:, :2]
    expected = torch.fft.irfft2(expected_ft, s=(8, 8))

    assert torch.allclose(layer(value), expected, atol=1e-6, rtol=1e-6)


def test_spectral_conv_rejects_more_modes_than_the_grid() -> None:
    layer = SpectralConv2d(1, 1, modes1=5, modes2=5)

    with pytest.raises(ValueError, match="retained modes"):
        layer(torch.zeros(1, 1, 8, 8))


def test_locked_fno_parameter_counts_use_real_complex_degrees_of_freedom() -> None:
    model = ConditionalFNO2d()

    assert count_tensor_parameters(model) == 1_855_457
    assert count_real_scalar_parameters(model) == 3_698_657


def test_fno_forward_preserves_shape_and_has_finite_gradients() -> None:
    model = ConditionalFNO2d(width=4, modes1=2, modes2=2, padding=1)
    condition = torch.randn(2, 3, 16, 16)
    state = torch.randn(2, 1, 16, 16, requires_grad=True)
    time = torch.tensor([0.0, 0.75])

    output = model(image=condition, x=state, step=time)
    output.square().mean().backward()

    assert output.shape == state.shape
    assert torch.isfinite(output).all()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_cfg_one_equals_conditional_velocity() -> None:
    torch.manual_seed(7)
    model = ConditionalFNO2d(width=4, modes1=2, modes2=2, padding=1).eval()
    condition = torch.randn(1, 3, 16, 16)
    state = torch.randn(1, 1, 16, 16)
    time = torch.tensor([0.5])

    conditional = model(image=condition, x=state, step=time)
    guided = model.forward_with_cfg(
        image=condition,
        x=state,
        step=time,
        embedding=model.embed_model(condition),
        cfg_scale=1.0,
    )

    assert torch.allclose(guided, conditional, atol=1e-6, rtol=1e-5)


def test_training_condition_dropout_zeros_only_condition_without_mutating_input() -> None:
    torch.manual_seed(11)
    model = ConditionalFNO2d(
        width=4,
        modes1=2,
        modes2=2,
        padding=1,
        cfg_drop_prob=1.0,
    )
    condition = torch.randn(2, 3, 16, 16)
    original = condition.clone()
    state = torch.randn(2, 1, 16, 16)
    time = torch.tensor([0.25, 0.75])

    model.train()
    dropped = model(image=condition, x=state, step=time)
    model.eval()
    expected = model(image=torch.zeros_like(condition), x=state, step=time)

    assert torch.equal(condition, original)
    assert torch.allclose(dropped, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_locked_fno_runs_padded_fft_under_cuda_float16_autocast() -> None:
    model = ConditionalFNO2d().cuda().eval()
    condition = torch.randn(1, 3, 256, 256, device="cuda")
    state = torch.randn(1, 1, 256, 256, device="cuda")
    time = torch.tensor([0.5], device="cuda")

    with torch.inference_mode(), torch.amp.autocast(
        "cuda", dtype=torch.float16, enabled=True
    ):
        output = model(image=condition, x=state, step=time)

    assert output.shape == state.shape
    assert torch.isfinite(output).all()

