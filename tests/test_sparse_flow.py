from __future__ import annotations

import pytest
import torch


def _module():
    from training import sparse_flow

    return sparse_flow


def test_build_masked_flow_pair_respects_t_zero_and_one() -> None:
    module = _module()
    initial_noise = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    target = torch.tensor([[[5.0, 7.0], [11.0, 13.0]]])
    valid_mask = torch.tensor([[[True, True], [False, True]]])
    observation_mask = torch.tensor([[[True, False], [False, False]]])
    observed_map = torch.tensor([[[5.0, 0.0], [0.0, 0.0]]])

    x0, velocity0, loss_mask0 = module.build_masked_flow_pair(
        initial_noise,
        target,
        observed_map,
        observation_mask,
        valid_mask,
        time=0.0,
    )
    x1, velocity1, loss_mask1 = module.build_masked_flow_pair(
        initial_noise,
        target,
        observed_map,
        observation_mask,
        valid_mask,
        time=1.0,
    )

    assert torch.equal(x0, torch.tensor([[[5.0, 2.0], [0.0, 4.0]]]))
    assert torch.equal(x1, torch.tensor([[[5.0, 7.0], [0.0, 13.0]]]))
    assert torch.equal(velocity0, torch.tensor([[[0.0, 5.0], [0.0, 9.0]]]))
    assert torch.equal(velocity1, torch.tensor([[[0.0, 5.0], [0.0, 9.0]]]))
    assert torch.equal(loss_mask0, valid_mask & ~observation_mask)
    assert torch.equal(loss_mask1, valid_mask & ~observation_mask)


def test_build_masked_flow_pair_uses_ut_only_on_missing_valid_region() -> None:
    module = _module()
    initial_noise = torch.tensor([[[0.0, 10.0], [20.0, 30.0]]])
    target = torch.tensor([[[1.0, 14.0], [40.0, 50.0]]])
    valid_mask = torch.tensor([[[True, True], [True, False]]])
    observation_mask = torch.tensor([[[True, False], [False, False]]])
    observed_map = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])

    xt, ut, missing_mask = module.build_masked_flow_pair(
        initial_noise,
        target,
        observed_map,
        observation_mask,
        valid_mask,
        time=0.25,
    )

    assert torch.equal(xt, torch.tensor([[[1.0, 11.0], [25.0, 0.0]]]))
    assert torch.equal(ut, torch.tensor([[[0.0, 4.0], [20.0, 0.0]]]))
    assert not bool((missing_mask & observation_mask).any())
    assert not bool((missing_mask & ~valid_mask).any())


def test_build_masked_flow_pair_accepts_batched_time_tensor() -> None:
    module = _module()
    initial_noise = torch.zeros(2, 1, 2, 2)
    target = torch.ones(2, 1, 2, 2)
    valid_mask = torch.ones(2, 1, 2, 2, dtype=torch.bool)
    observation_mask = torch.zeros(2, 1, 2, 2, dtype=torch.bool)
    observed_map = torch.zeros_like(target)

    xt, ut, missing_mask = module.build_masked_flow_pair(
        initial_noise=initial_noise,
        target=target,
        observed_map=observed_map,
        observation_mask=observation_mask,
        valid_mask=valid_mask,
        time=torch.tensor([0.25, 0.75]),
    )

    assert torch.allclose(xt[0], torch.full((1, 2, 2), 0.25))
    assert torch.allclose(xt[1], torch.full((1, 2, 2), 0.75))
    assert torch.equal(ut, torch.ones_like(target))
    assert torch.equal(missing_mask, valid_mask)


@pytest.mark.parametrize("bad_time", [-0.1, 1.1, float("nan")])
def test_build_masked_flow_pair_rejects_invalid_time(bad_time: float) -> None:
    module = _module()
    tensor = torch.zeros(1, 2, 2)
    mask = torch.ones(1, 2, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="time"):
        module.build_masked_flow_pair(
            tensor,
            tensor,
            tensor,
            mask,
            mask,
            time=bad_time,
        )


def test_build_masked_flow_pair_rejects_bad_inputs() -> None:
    module = _module()
    initial_noise = torch.zeros(1, 2, 2)
    target = torch.ones(1, 2, 2)
    observed_map = torch.zeros(1, 2, 2)
    valid_mask = torch.ones(1, 2, 2, dtype=torch.bool)
    observation_mask = torch.tensor([[[True, False], [False, True]]])

    with pytest.raises(ValueError, match="missing"):
        module.build_masked_flow_pair(
            initial_noise,
            target,
            observed_map,
            valid_mask,
            valid_mask,
            time=0.5,
        )
    with pytest.raises(ValueError, match="boolean"):
        module.build_masked_flow_pair(
            initial_noise,
            target,
            observed_map,
            observation_mask.to(torch.float32),  # type: ignore[arg-type]
            valid_mask,
            time=0.5,
        )
    with pytest.raises(ValueError, match="shape"):
        module.build_masked_flow_pair(
            initial_noise[:, :1],
            target,
            observed_map,
            observation_mask,
            valid_mask,
            time=0.5,
        )
    with pytest.raises(ValueError, match="finite"):
        module.build_masked_flow_pair(
            torch.tensor([[[0.0, float("nan")], [0.0, 0.0]]]),
            target,
            observed_map,
            observation_mask,
            valid_mask,
            time=0.5,
        )
