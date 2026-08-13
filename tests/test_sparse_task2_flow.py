from __future__ import annotations

import pytest
import torch

from training.sparse_task2_flow import (
    SparseTask2FlowError,
    build_task2_flow_pair,
    masked_task2_velocity_mse,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x0 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]])
    valid = torch.tensor([[[[True, True], [False, True]]]])
    return x0, target, valid


@pytest.mark.parametrize("time_value", (0.0, 0.5, 1.0))
def test_full_target_interpolation_and_velocity(time_value: float) -> None:
    x0, target, valid = _inputs()
    xt, ut, loss_mask = build_task2_flow_pair(
        x0, target, valid, torch.tensor([time_value])
    )
    expected_xt = (1.0 - time_value) * x0 + time_value * target
    expected_ut = target - x0
    assert torch.equal(xt[valid], expected_xt[valid])
    assert torch.equal(ut[valid], expected_ut[valid])
    assert torch.equal(xt[~valid], torch.zeros_like(xt[~valid]))
    assert torch.equal(ut[~valid], torch.zeros_like(ut[~valid]))
    assert torch.equal(loss_mask, valid)


def test_pair_has_no_observation_path_and_is_full_valid() -> None:
    x0, target, valid = _inputs()
    first = build_task2_flow_pair(x0, target, valid, torch.tensor([0.25]))
    changed_target = target.clone()
    changed_target[0, 0, 0, 0] = 99.0
    second = build_task2_flow_pair(x0, changed_target, valid, torch.tensor([0.25]))
    assert not torch.equal(first[0], second[0])
    assert torch.equal(first[2], valid)
    assert int(first[2].sum().item()) == 3


def test_time_batch_broadcast_accepts_b_one() -> None:
    x0, target, valid = _inputs()
    xt, _, _ = build_task2_flow_pair(x0, target, valid, torch.tensor([[0.25]]))
    assert xt.shape == x0.shape


@pytest.mark.parametrize(
    "mutator",
    (
        lambda x0, target, valid, time: (x0[:, :, :, :1], target, valid, time),
        lambda x0, target, valid, time: (x0, target, valid, torch.tensor([1.1])),
        lambda x0, target, valid, time: (x0, target, valid, torch.tensor([float("nan")])) ,
        lambda x0, target, valid, time: (x0, target, valid.to(torch.float32), time),
    ),
)
def test_pair_rejects_contract_violations(mutator) -> None:
    x0, target, valid = _inputs()
    with pytest.raises(SparseTask2FlowError):
        build_task2_flow_pair(*mutator(x0, target, valid, torch.tensor([0.5])))


def test_velocity_mse_is_global_over_valid_pixels() -> None:
    predicted = torch.zeros((2, 1, 2, 2))
    target = torch.ones_like(predicted)
    valid = torch.tensor(
        [[[[True, True], [True, True]]], [[[True, False], [False, False]]]]
    )
    assert masked_task2_velocity_mse(predicted, target, valid).item() == pytest.approx(1.0)


def test_velocity_mse_rejects_empty_mask_and_shape_error() -> None:
    predicted = torch.zeros((1, 1, 2, 2))
    empty = torch.zeros_like(predicted, dtype=torch.bool)
    with pytest.raises(SparseTask2FlowError, match="no valid"):
        masked_task2_velocity_mse(predicted, predicted, empty)
    with pytest.raises(SparseTask2FlowError):
        masked_task2_velocity_mse(predicted, torch.zeros((1, 1, 1, 1)), ~empty)
