from __future__ import annotations

import pytest
import torch

from model.attention_multiscale_uno import AttentionConditionedFNOStage


def _stage() -> AttentionConditionedFNOStage:
    return AttentionConditionedFNOStage(
        channels=8,
        embedding_channels=4,
        operator_width=4,
        modes=2,
        padding=1,
        attention_reduction=4,
    )


def test_stage_preserves_external_shape_and_backpropagates_every_branch() -> None:
    stage = _stage()
    value = torch.randn(2, 8, 16, 16, requires_grad=True)
    condition = torch.randn(2, 4, 16, 16)
    time = torch.randn(2, 512)
    output = stage(value, condition, time)
    output.square().mean().backward()
    assert output.shape == value.shape
    assert torch.isfinite(output).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert stage.spectral.weights1.grad is not None
    assert stage.local.weight.grad is not None
    assert stage.time_projection.weight.grad is not None
    assert stage.attention.embedding_proj.weight.grad is not None


def test_zero_operator_update_leaves_the_attended_residual() -> None:
    stage = _stage().eval()
    value = torch.randn(1, 8, 16, 16)
    condition = torch.randn(1, 4, 16, 16)
    time = torch.randn(1, 512)
    with torch.no_grad():
        stage.projection.weight.zero_()
        stage.projection.bias.zero_()
        expected = stage.attention(value, condition)
        actual = stage(value, condition, time)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("channels", 0, "channels"),
        ("embedding_channels", 0, "embedding_channels"),
        ("operator_width", 0, "operator_width"),
        ("modes", 0, "modes"),
        ("padding", -1, "padding"),
        ("time_channels", 0, "time_channels"),
        ("attention_reduction", 0, "attention_reduction"),
        ("attention_reduction", 9, "attention_reduction"),
    ),
)
def test_stage_rejects_invalid_constructor_controls(
    field: str,
    value: int,
    message: str,
) -> None:
    controls = {
        "channels": 8,
        "embedding_channels": 4,
        "operator_width": 4,
        "modes": 2,
        "padding": 1,
        "time_channels": 512,
        "attention_reduction": 4,
    }
    controls[field] = value

    with pytest.raises(ValueError, match=message):
        AttentionConditionedFNOStage(**controls)


def test_stage_rejects_mismatched_value_condition_and_time_inputs() -> None:
    stage = _stage()
    value = torch.zeros(2, 8, 16, 16)
    condition = torch.zeros(2, 4, 16, 16)
    time = torch.zeros(2, 512)

    with pytest.raises(ValueError, match="value must have shape"):
        stage(value[:, :, :, 0], condition, time)
    with pytest.raises(ValueError, match="value must have shape"):
        stage(value[:, :7], condition, time)
    with pytest.raises(ValueError, match="condition must have shape"):
        stage(value, condition[:, :3], time)
    with pytest.raises(ValueError, match="condition must have shape"):
        stage(value, condition[:, :, :-1], time)
    with pytest.raises(ValueError, match="time embedding must have shape"):
        stage(value, condition, time[:, :511])
    with pytest.raises(ValueError, match="time embedding must have shape"):
        stage(value, condition, time[:1])


def test_stage_rejects_modes_that_do_not_fit_the_padded_grid() -> None:
    stage = AttentionConditionedFNOStage(
        channels=8,
        embedding_channels=4,
        operator_width=4,
        modes=5,
        padding=0,
        attention_reduction=4,
    )

    with pytest.raises(ValueError, match="retained modes"):
        stage(torch.zeros(1, 8, 8, 8), torch.zeros(1, 4, 8, 8), torch.zeros(1, 512))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_stage_runs_under_cuda_float16_autocast_with_finite_gradients() -> None:
    stage = _stage().cuda()
    value = torch.randn(2, 8, 16, 16, device="cuda", requires_grad=True)
    condition = torch.randn(2, 4, 16, 16, device="cuda")
    time = torch.randn(2, 512, device="cuda")

    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
        output = stage(value, condition, time)
        loss = output.float().square().mean()
    loss.backward()

    assert output.shape == value.shape
    assert torch.isfinite(output).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in stage.parameters()
        if parameter.requires_grad
    )
