from __future__ import annotations

import pytest
import torch


def _loss_module():
    from training import masked_flow_loss

    return masked_flow_loss


def test_masked_velocity_loss_matches_hand_example_and_zeroes_invalid_gradients() -> None:
    module = _loss_module()
    prediction = torch.tensor(
        [[[[1.0, 3.0], [100.0, -99.0]]]],
        requires_grad=True,
    )
    target = torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]])
    mask = torch.tensor([[[[True, True], [False, False]]]])

    loss = module.masked_velocity_mse(prediction, target, mask)
    loss.backward()

    assert loss.item() == 2.5
    assert torch.equal(
        prediction.grad,
        torch.tensor([[[[1.0, 2.0], [0.0, 0.0]]]]),
    )


def test_masked_velocity_loss_rejects_shape_mismatch() -> None:
    module = _loss_module()

    with pytest.raises(module.MaskedLossError, match="shapes must match"):
        module.masked_velocity_mse(
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 1),
            torch.ones(1, 1, 2, 2, dtype=torch.bool),
        )


def test_masked_velocity_loss_requires_boolean_mask() -> None:
    module = _loss_module()

    with pytest.raises(module.MaskedLossError, match="boolean"):
        module.masked_velocity_mse(
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
        )


@pytest.mark.parametrize("which", ["prediction", "target"])
def test_masked_velocity_loss_rejects_nonfinite_valid_values(which: str) -> None:
    module = _loss_module()
    prediction = torch.zeros(1, 1, 2, 2)
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    if which == "prediction":
        prediction[0, 0, 0, 0] = float("inf")
    else:
        target[0, 0, 0, 0] = float("nan")

    with pytest.raises(module.MaskedLossError, match="non-finite velocity"):
        module.masked_velocity_mse(prediction, target, mask)


def test_masked_velocity_loss_ignores_nonfinite_invalid_values() -> None:
    module = _loss_module()
    prediction = torch.tensor([[[[1.0, float("inf")]]]], requires_grad=True)
    target = torch.tensor([[[[0.0, float("nan")]]]])
    mask = torch.tensor([[[[True, False]]]])

    loss = module.masked_velocity_mse(prediction, target, mask)
    loss.backward()

    assert loss.item() == 1.0
    assert torch.equal(prediction.grad, torch.tensor([[[[2.0, 0.0]]]]))


def test_masked_velocity_loss_rejects_zero_valid_pixels() -> None:
    module = _loss_module()

    with pytest.raises(module.MaskedLossError, match="zero valid pixels"):
        module.masked_velocity_mse(
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2, dtype=torch.bool),
        )

