from __future__ import annotations

import gc

import pytest
import torch
import torch.utils.checkpoint as checkpoint_utils

from model.model import DiffUNet
from model.unet.basic_unet import BasicUNetEncoder


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_checkpoint_switch_preserves_lite_outputs_state_and_gradients() -> None:
    torch.manual_seed(2026)
    reference = DiffUNet(
        con_channels=3,
        model_size="lite",
        activation_checkpointing=False,
    )
    checkpointed = DiffUNet(
        con_channels=3,
        model_size="lite",
        activation_checkpointing=True,
    )
    checkpointed.load_state_dict(reference.state_dict(), strict=True)
    reference.cfg_drop_prob = 0.0
    checkpointed.cfg_drop_prob = 0.0
    reference.train()
    checkpointed.train()
    condition = torch.randn(1, 3, 32, 32)
    noise = torch.randn(1, 1, 32, 32)
    step = torch.tensor([0.25])

    reference_output = reference(
        image=condition,
        x=noise.clone().requires_grad_(True),
        step=step,
    )
    reference_output.square().mean().backward()
    reference_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in reference.named_parameters()
    }
    expected_output = reference_output.detach().clone()

    checkpointed_output = checkpointed(
        image=condition,
        x=noise.clone().requires_grad_(True),
        step=step,
    )
    checkpointed_output.square().mean().backward()

    assert tuple(reference.state_dict()) == tuple(checkpointed.state_dict())
    assert _parameter_count(reference) == _parameter_count(checkpointed) == 3_994_859
    assert torch.allclose(
        checkpointed_output,
        expected_output,
        atol=1e-6,
        rtol=1e-5,
    )
    for name, parameter in checkpointed.named_parameters():
        expected = reference_gradients[name]
        assert (parameter.grad is None) == (expected is None), name
        if expected is not None:
            assert torch.allclose(
                parameter.grad,
                expected,
                atol=2e-6,
                rtol=2e-5,
            ), name


def test_checkpoint_calls_only_during_training_with_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    original = checkpoint_utils.checkpoint

    def recording_checkpoint(function, *args, **kwargs):
        calls.append(dict(kwargs))
        return original(function, *args, **kwargs)

    monkeypatch.setattr(checkpoint_utils, "checkpoint", recording_checkpoint)
    encoder = BasicUNetEncoder(
        spatial_dims=2,
        in_channels=3,
        features=(4, 4, 8, 16, 32, 4),
        activation_checkpointing=True,
    )
    values = torch.randn(1, 3, 32, 32, requires_grad=True)

    encoder.train()
    encoder(values)
    assert len(calls) == 5
    assert all(call["use_reentrant"] is False for call in calls)
    assert all(call["preserve_rng_state"] is True for call in calls)

    calls.clear()
    encoder.eval()
    encoder(values)
    assert calls == []

    encoder.train()
    with torch.no_grad():
        encoder(values)
    assert calls == []


def test_large_checkpoint_switch_preserves_parameter_and_state_identity() -> None:
    baseline = DiffUNet(
        con_channels=3,
        model_size="large",
        activation_checkpointing=False,
    )
    expected_keys = tuple(baseline.state_dict())
    assert _parameter_count(baseline) == 54_126_059
    del baseline
    gc.collect()

    checkpointed = DiffUNet(
        con_channels=3,
        model_size="large",
        activation_checkpointing=True,
    )

    assert tuple(checkpointed.state_dict()) == expected_keys
    assert _parameter_count(checkpointed) == 54_126_059
    assert checkpointed.activation_checkpointing is True
    assert checkpointed.embed_model.activation_checkpointing is True
    assert checkpointed.model.activation_checkpointing is True


@pytest.mark.parametrize(
    ("model_size", "expected_count", "checkpointing"),
    [("lite", 3_994_859, False), ("large", 54_126_059, True)],
)
def test_locked_factory_is_the_only_benchmark_model_policy(
    model_size: str,
    expected_count: int,
    checkpointing: bool,
) -> None:
    from training.model_factory import build_locked_radioflow

    model = build_locked_radioflow(model_size)

    assert type(model) is DiffUNet
    assert _parameter_count(model) == expected_count
    assert model.cfg_drop_prob == 0.25
    assert model.activation_checkpointing is checkpointing

