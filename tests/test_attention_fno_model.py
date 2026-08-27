from __future__ import annotations

import pytest
import torch
from torch import nn

from model.attention_fno import (
    AttentionConditionedFNO2d,
    FullResolutionFNOBlock,
)
from model.fno import SpectralConv2d
from model.unet.basic_unet import BasicUNetEncoder
from model.unet.basic_unet_denose import CrossAttention


SMALL_FEATURES = (4, 4, 8, 16, 32, 4)


def _small_model(*, cfg_drop_prob: float = 0.0) -> AttentionConditionedFNO2d:
    return AttentionConditionedFNO2d(
        width=16,
        modes1=2,
        modes2=2,
        padding=1,
        encoder_features=SMALL_FEATURES,
        cfg_drop_prob=cfg_drop_prob,
    )


def test_attention_fno_has_approved_full_resolution_components() -> None:
    model = _small_model()

    assert type(model.condition_encoder) is BasicUNetEncoder
    assert model.lifting.in_channels == 6
    assert model.lifting.out_channels == 16
    assert model.lifting.kernel_size == (1, 1)
    assert len(model.condition_projections) == 5
    assert len(model.blocks) == 4
    assert all(type(block) is FullResolutionFNOBlock for block in model.blocks)
    assert all(type(block.attention) is CrossAttention for block in model.blocks)
    assert all(type(block.spectral) is SpectralConv2d for block in model.blocks)


def test_coordinate_grid_encodes_horizontal_x_and_vertical_y() -> None:
    state = torch.zeros(2, 1, 3, 4)

    grid_x, grid_y = AttentionConditionedFNO2d.coordinate_grid(state)

    assert grid_x.shape == grid_y.shape == (2, 1, 3, 4)
    assert torch.allclose(
        grid_x[0, 0, 0],
        torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]),
        atol=1e-7,
        rtol=1e-7,
    )
    assert torch.equal(grid_x[0, 0, 0], grid_x[0, 0, 2])
    assert torch.allclose(
        grid_y[0, 0, :, 0],
        torch.tensor([0.0, 0.5, 1.0]),
        atol=1e-7,
        rtol=1e-7,
    )
    assert torch.equal(grid_y[0, 0, :, 0], grid_y[0, 0, :, 3])


def test_five_scale_condition_features_are_projected_resized_and_summed() -> None:
    model = _small_model()
    channels = (4, 4, 8, 16, 32)
    spatial = (32, 16, 8, 4, 2)
    embeddings = []
    with torch.no_grad():
        for index, (projection, in_channels, size) in enumerate(
            zip(model.condition_projections, channels, spatial)
        ):
            projection.weight.zero_()
            projection.bias.zero_()
            projection.weight[:, 0, 0, 0] = 1.0
            embedding = torch.zeros(1, in_channels, size, size)
            embedding[:, 0] = float(index + 1)
            embeddings.append(embedding)

    aggregated = model.aggregate_condition(embeddings, output_size=(32, 32))

    assert aggregated.shape == (1, 16, 32, 32)
    assert torch.equal(aggregated, torch.full_like(aggregated, 15.0))


def test_each_fno_block_uses_its_own_time_projection() -> None:
    torch.manual_seed(7)
    model = _small_model().eval()
    value = torch.randn(1, 16, 16, 16)
    condition = torch.randn_like(value)
    time_zero = torch.zeros(1, 512)
    time_one = torch.ones(1, 512)

    for block in model.blocks:
        with torch.no_grad():
            block.time_projection.weight.fill_(0.01)
            block.time_projection.bias.zero_()
            at_zero = block(value, condition, time_zero)
            at_one = block(value, condition, time_one)
        assert at_zero.shape == at_one.shape == value.shape
        assert not torch.allclose(at_zero, at_one)

    assert len({id(block.time_projection) for block in model.blocks}) == 4


def test_attention_fno_forward_is_finite_and_backpropagates_all_branches() -> None:
    torch.manual_seed(11)
    model = _small_model()
    condition = torch.randn(1, 3, 32, 32)
    state = torch.randn(1, 1, 32, 32, requires_grad=True)
    time = torch.tensor([0.35])

    output = model(image=condition, x=state, step=time)
    output.square().mean().backward()

    assert output.shape == state.shape
    assert torch.isfinite(output).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert model.lifting.weight.grad is not None
    assert model.condition_encoder.conv_0.conv_0.conv.weight.grad is not None
    for block in model.blocks:
        assert block.spectral.weights1.grad is not None
        assert block.local.weight.grad is not None
        assert block.time_projection.weight.grad is not None
        assert block.attention.embedding_proj.weight.grad is not None


def test_cfg_one_is_exactly_the_conditional_velocity() -> None:
    torch.manual_seed(13)
    model = _small_model().eval()
    condition = torch.randn(1, 3, 32, 32)
    state = torch.randn(1, 1, 32, 32)
    time = torch.tensor([0.5])
    embedding = model.embed_model(condition)

    expected = model(
        image=condition,
        x=state,
        step=time,
        embedding=embedding,
    )
    actual = model.forward_with_cfg(
        image=condition,
        x=state,
        step=time,
        embedding=embedding,
        cfg_scale=1.0,
    )

    assert torch.equal(actual, expected)


def test_full_condition_dropout_matches_explicit_zero_condition() -> None:
    torch.manual_seed(17)
    model = _small_model(cfg_drop_prob=1.0)
    condition = torch.randn(2, 3, 32, 32)
    original = condition.clone()
    state = torch.randn(2, 1, 32, 32)
    time = torch.tensor([0.25, 0.75])

    model.train()
    dropped = model(image=condition, x=state, step=time)
    model.eval()
    expected = model.forward_with_cfg(
        image=condition,
        x=state,
        step=time,
        cfg_scale=0.0,
    )

    assert torch.equal(condition, original)
    assert torch.allclose(dropped, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP is required")
def test_full_condition_dropout_has_finite_amp_backward() -> None:
    torch.manual_seed(19)
    model = _small_model(cfg_drop_prob=1.0).cuda().train()
    condition = torch.randn(2, 3, 32, 32, device="cuda")
    state = torch.randn(2, 1, 32, 32, device="cuda")
    target = torch.randn_like(state)
    time = torch.tensor([0.2, 0.8], device="cuda")

    with torch.amp.autocast("cuda", dtype=torch.float16):
        output = model(image=condition, x=state, step=time)
        loss = (output - target).square().mean()
    loss.backward()

    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and (
            not bool(torch.isfinite(parameter.grad.real).all())
            or (
                parameter.grad.is_complex()
                and not bool(torch.isfinite(parameter.grad.imag).all())
            )
        )
    ]
    assert nonfinite == []


def test_attention_fno_rejects_invalid_state_condition_and_time() -> None:
    model = _small_model()
    condition = torch.randn(1, 3, 32, 32)
    state = torch.randn(1, 1, 32, 32)

    with pytest.raises(ValueError, match="three channels"):
        model(image=condition[:, :2], x=state, step=torch.tensor([0.5]))
    with pytest.raises(ValueError, match="one channel"):
        model(image=condition, x=state.repeat(1, 2, 1, 1), step=torch.tensor([0.5]))
    with pytest.raises(ValueError, match="one value per sample"):
        model(image=condition, x=state, step=torch.tensor([0.2, 0.8]))
