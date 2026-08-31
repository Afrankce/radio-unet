from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from model.attention_multiscale_uno import (
    AttentionConditionedFNOStage,
    AttentionMultiscaleUNO2d,
    _Downsample2d,
    _UpsampleFuse2d,
)
from model.unet.basic_unet import BasicUNetEncoder
from model.unet.basic_unet_denose import CrossAttention, CrossAttention_old


def _tiny_model(cfg_drop_prob: float = 0.0) -> AttentionMultiscaleUNO2d:
    return AttentionMultiscaleUNO2d(
        state_channels=(8, 8, 16, 16, 16),
        operator_width=4,
        operator_modes=(2, 2, 2, 1, 1),
        operator_padding=(1, 1, 1, 1, 1),
        encoder_features=(4, 4, 8, 16, 32, 4),
        attention_reduction=4,
        cfg_drop_prob=cfg_drop_prob,
    )


def _all_stages(model: AttentionMultiscaleUNO2d) -> list[AttentionConditionedFNOStage]:
    return [*model.encoder_stages, model.bottleneck, *model.decoder_stages]


def test_topology_keeps_five_native_scales_and_nine_current_attention_modules() -> None:
    model = _tiny_model()

    assert type(model.condition_encoder) is BasicUNetEncoder
    assert model.lifting.in_channels == 6
    assert model.lifting.out_channels == 8
    assert model.lifting.kernel_size == (1, 1)
    assert len(model.encoder_stages) == 4
    assert len(model.decoder_stages) == 4
    assert type(model.bottleneck) is AttentionConditionedFNOStage
    assert sum(isinstance(module, CrossAttention) for module in model.modules()) == 9
    assert not any(isinstance(module, CrossAttention_old) for module in model.modules())
    assert all(type(stage) is AttentionConditionedFNOStage for stage in _all_stages(model))
    state_convolutions = [
        model.lifting,
        model.projection_hidden,
        model.projection_output,
        *(module.projection for module in model.downsamples),
        *(module.projection for module in model.upsample_fusions),
        *(
            convolution
            for stage in _all_stages(model)
            for convolution in (stage.lifting, stage.local, stage.projection)
        ),
    ]
    assert all(convolution.kernel_size == (1, 1) for convolution in state_convolutions)
    assert all(type(module.pool) is nn.AvgPool2d for module in model.downsamples)
    assert not any(isinstance(module, nn.ConvTranspose2d) for module in model.modules())


def test_stage_hooks_catch_accidental_full_resolution_execution() -> None:
    model = _tiny_model().eval()
    condition = torch.randn(1, 3, 32, 32)
    state = torch.randn(1, 1, 32, 32)
    observed: list[tuple[int, int]] = []
    handles = [
        stage.register_forward_hook(
            lambda _module, _inputs, output: observed.append(output.shape[-2:])
        )
        for stage in _all_stages(model)
    ]
    try:
        with torch.no_grad():
            model(image=condition, x=state, step=0.5)
    finally:
        for handle in handles:
            handle.remove()

    assert observed == [
        (32, 32),
        (16, 16),
        (8, 8),
        (4, 4),
        (2, 2),
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
    ]


def test_coordinate_grid_catches_swapped_horizontal_and_vertical_axes() -> None:
    state = torch.zeros(2, 1, 3, 4)

    grid_x, grid_y = AttentionMultiscaleUNO2d.coordinate_grid(state)

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


def test_lifting_input_catches_reordered_raw_condition_and_coordinate_channels() -> None:
    model = _tiny_model().eval()
    state = torch.full((1, 1, 32, 32), 11.0)
    condition = torch.cat(
        (
            torch.full_like(state, 22.0),
            torch.full_like(state, 33.0),
            torch.full_like(state, 44.0),
        ),
        dim=1,
    )
    captured: list[torch.Tensor] = []
    handle = model.lifting.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        with torch.no_grad():
            model(image=condition, x=state, step=0.25)
    finally:
        handle.remove()

    grid_x, grid_y = model.coordinate_grid(state)
    expected = torch.cat((state, condition, grid_x, grid_y), dim=1)
    assert len(captured) == 1
    assert torch.equal(captured[0], expected)


def test_velocity_output_is_finite_and_backpropagates_every_spectral_stage() -> None:
    torch.manual_seed(11)
    model = _tiny_model()
    condition = torch.randn(1, 3, 32, 32)
    state = torch.randn(1, 1, 32, 32, requires_grad=True)

    output = model(image=condition, x=state, step=torch.tensor([0.35]))
    output.square().mean().backward()

    assert output.shape == (1, 1, 32, 32)
    assert torch.isfinite(output).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    for index, stage in enumerate(_all_stages(model)):
        assert stage.spectral.weights1.grad is not None, index
        assert stage.spectral.weights2.grad is not None, index
        assert torch.isfinite(stage.spectral.weights1.grad).all(), index
        assert torch.isfinite(stage.spectral.weights2.grad).all(), index


def test_downsample_rejects_odd_encoder_shapes_instead_of_silently_rounding() -> None:
    downsample = _Downsample2d(8, 16)

    with pytest.raises(ValueError, match="even spatial dimensions"):
        downsample(torch.randn(1, 8, 7, 8))
    with pytest.raises(ValueError, match="shape.*8"):
        downsample(torch.randn(1, 7, 8, 8))


def test_resize_modules_use_average_pool_and_exact_skip_target_geometry() -> None:
    downsample = _Downsample2d(2, 3)
    value = torch.arange(64, dtype=torch.float32).reshape(1, 2, 4, 8)
    with torch.no_grad():
        downsample.projection.weight.fill_(0.25)
        downsample.projection.bias.fill_(0.5)
        actual_down = downsample(value)
        expected_down = downsample.projection(nn.functional.avg_pool2d(value, 2))
    assert torch.equal(actual_down, expected_down)

    fuse = _UpsampleFuse2d(3, 2, 4)
    coarse = torch.randn(2, 3, 3, 4)
    skip = torch.randn(2, 2, 7, 9)
    captured: list[torch.Tensor] = []
    handle = fuse.projection.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        output = fuse(coarse, skip)
    finally:
        handle.remove()
    expected_resized = nn.functional.interpolate(
        coarse,
        size=(7, 9),
        mode="bilinear",
        align_corners=False,
    )
    assert output.shape == (2, 4, 7, 9)
    assert torch.equal(captured[0], torch.cat((expected_resized, skip), dim=1))


def test_upsample_fuse_rejects_wrong_skip_batch_and_channels() -> None:
    fuse = _UpsampleFuse2d(8, 4, 4)
    coarse = torch.randn(2, 8, 4, 4)

    with pytest.raises(ValueError, match="batch"):
        fuse(coarse, torch.randn(1, 4, 8, 8))
    with pytest.raises(ValueError, match="skip.*4"):
        fuse(coarse, torch.randn(2, 3, 8, 8))


def test_cfg_one_is_bit_exact_with_the_conditional_velocity() -> None:
    torch.manual_seed(13)
    model = _tiny_model().eval()
    condition = torch.randn(1, 3, 32, 32)
    state = torch.randn(1, 1, 32, 32)
    step = torch.tensor([0.5])
    embedding = model.embed_model(condition)

    expected = model(image=condition, x=state, step=step, embedding=embedding)
    actual = model.forward_with_cfg(
        image=condition,
        x=state,
        step=step,
        embedding=embedding,
        cfg_scale=1.0,
    )

    assert torch.equal(actual, expected)


def test_full_cfg_dropout_matches_the_explicit_unconditional_branch() -> None:
    torch.manual_seed(17)
    model = _tiny_model(cfg_drop_prob=1.0)
    condition = torch.randn(2, 3, 32, 32)
    original = condition.clone()
    state = torch.randn(2, 1, 32, 32)
    step = torch.tensor([0.25, 0.75])

    model.train()
    dropped = model(image=condition, x=state, step=step)
    model.eval()
    expected = model.forward_with_cfg(
        image=condition,
        x=state,
        step=step,
        cfg_scale=0.0,
    )

    assert torch.equal(condition, original)
    assert torch.allclose(dropped, expected, atol=1e-6, rtol=1e-5)


def test_sample_dropout_jointly_zeros_raw_condition_and_all_five_embeddings() -> None:
    model = _tiny_model(cfg_drop_prob=0.5).train()
    condition = torch.randn(8, 3, 32, 32)
    state = torch.randn(8, 1, 32, 32)
    baseline_embeddings = model.embed_model(condition)
    captured_raw: list[torch.Tensor] = []
    captured_embeddings: list[torch.Tensor] = []
    handles = [
        model.lifting.register_forward_pre_hook(
            lambda _module, inputs: captured_raw.append(inputs[0][:, 1:4].detach().clone())
        )
    ]
    for stage in [*model.encoder_stages, model.bottleneck]:
        handles.append(
            stage.register_forward_pre_hook(
                lambda _module, inputs: captured_embeddings.append(
                    inputs[1].detach().clone()
                )
            )
        )
    try:
        torch.manual_seed(2026)
        model(image=condition, x=state, step=0.5)
    finally:
        for handle in handles:
            handle.remove()

    dropped = captured_raw[0].flatten(1).eq(0).all(dim=1)
    assert bool(dropped.any()) and bool((~dropped).any())
    assert torch.equal(captured_raw[0][~dropped], condition[~dropped])
    for actual, expected in zip(captured_embeddings, baseline_embeddings):
        assert torch.count_nonzero(actual[dropped]) == 0
        assert torch.equal(actual[~dropped], expected[~dropped])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP is required")
def test_full_condition_dropout_keeps_amp_backward_finite() -> None:
    torch.manual_seed(19)
    model = _tiny_model(cfg_drop_prob=1.0).cuda().train()
    condition = torch.randn(2, 3, 32, 32, device="cuda")
    state = torch.randn(2, 1, 32, 32, device="cuda")
    target = torch.randn_like(state)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        output = model(image=condition, x=state, step=torch.tensor([0.2, 0.8], device="cuda"))
        loss = (output - target).square().mean()
    loss.backward()

    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
    ]
    assert nonfinite == []


def test_public_forward_rejects_invalid_trainer_inputs_and_embeddings() -> None:
    model = _tiny_model()
    condition = torch.randn(1, 3, 32, 32)
    state = torch.randn(1, 1, 32, 32)

    with pytest.raises(ValueError, match="only supports pred_type='denoise'"):
        model(image=condition, x=state, step=0.5, pred_type="noise")
    with pytest.raises(ValueError, match="image, x, and step are required"):
        model(image=condition, x=state)
    with pytest.raises(ValueError, match="three channels"):
        model(image=condition[:, :2], x=state, step=0.5)
    with pytest.raises(ValueError, match="one channel"):
        model(image=condition, x=state.repeat(1, 2, 1, 1), step=0.5)
    with pytest.raises(ValueError, match="batch/spatial"):
        model(image=condition, x=state[..., :-1], step=0.5)
    with pytest.raises(ValueError, match="one value per sample"):
        model(image=condition, x=state, step=torch.tensor([0.2, 0.8]))
    with pytest.raises(ValueError, match="five feature scales"):
        model(image=condition, x=state, step=0.5, embedding=[])


def test_existing_model_api_embeds_five_scales_and_propagates_checkpointing() -> None:
    model = AttentionMultiscaleUNO2d(
        state_channels=(8, 8, 16, 16, 16),
        operator_width=4,
        operator_modes=(2, 2, 2, 1, 1),
        operator_padding=(1, 1, 1, 1, 1),
        encoder_features=(4, 4, 8, 16, 32, 4),
        attention_reduction=4,
        activation_checkpointing=True,
    )
    condition = torch.randn(1, 3, 32, 32)

    embeddings = model.embed_model(condition)

    assert model.condition_channels == 3
    assert model.cfg_drop_prob == 0.25
    assert model.activation_checkpointing is True
    assert model.condition_encoder.activation_checkpointing is True
    assert [tuple(value.shape) for value in embeddings] == [
        (1, 4, 32, 32),
        (1, 4, 16, 16),
        (1, 8, 8, 8),
        (1, 16, 4, 4),
        (1, 32, 2, 2),
    ]
    assert math.isfinite(float(model(image=condition, x=torch.randn(1, 1, 32, 32), step=0.5).mean()))
