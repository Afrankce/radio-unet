from __future__ import annotations

import pytest
import torch

from training.random_task2_flow import (
    RandomTask2FlowError,
    build_random_task2_pinned_flow_pair,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x0 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[11.0, 12.0], [13.0, 14.0]]]])
    sparse_map = torch.tensor([[[[101.0, 0.0], [0.0, 0.0]]]])
    observation_mask = torch.tensor([[[[True, False], [False, False]]]])
    valid_mask = torch.tensor([[[[True, True], [False, True]]]])
    return x0, target, sparse_map, observation_mask, valid_mask


@pytest.mark.parametrize("time_value", (0.0, 0.25, 1.0))
def test_pinned_pair_pins_observations_and_interpolates_only_missing_pixels(
    time_value: float,
) -> None:
    x0, target, sparse_map, observation_mask, valid_mask = _inputs()

    xt, ut, loss_mask = build_random_task2_pinned_flow_pair(
        x0=x0,
        target=target,
        sparse_map=sparse_map,
        observation_mask=observation_mask,
        valid_mask=valid_mask,
        time=torch.tensor([time_value]),
    )

    missing_mask = valid_mask & ~observation_mask
    expected_xt = (1.0 - time_value) * x0 + time_value * target
    expected_velocity = target - x0

    assert torch.equal(loss_mask, missing_mask)
    assert torch.equal(xt[observation_mask], sparse_map[observation_mask])
    assert torch.equal(xt[missing_mask], expected_xt[missing_mask])
    assert torch.equal(xt[~valid_mask], torch.zeros_like(xt[~valid_mask]))
    assert torch.equal(ut[missing_mask], expected_velocity[missing_mask])
    assert torch.equal(ut[observation_mask], torch.zeros_like(ut[observation_mask]))
    assert torch.equal(ut[~valid_mask], torch.zeros_like(ut[~valid_mask]))


def test_pinned_pair_rejects_observation_masks_outside_valid_pixels() -> None:
    x0, target, sparse_map, observation_mask, valid_mask = _inputs()
    bad_mask = observation_mask.clone()
    bad_mask[0, 0, 1, 0] = True

    with pytest.raises(RandomTask2FlowError, match="subset"):
        build_random_task2_pinned_flow_pair(
            x0=x0,
            target=target,
            sparse_map=sparse_map,
            observation_mask=bad_mask,
            valid_mask=valid_mask,
            time=torch.tensor([0.5]),
        )
