from __future__ import annotations

import pytest
import torch

from evaluation.random_task2_sampling import random_task2_euler_cfg_sample


class _RecordingVelocityModel:
    def __init__(self) -> None:
        self.seen_states: list[torch.Tensor] = []
        self.seen_steps: list[torch.Tensor] = []

    def embed_model(self, condition: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
        del args
        return condition

    def forward_with_cfg(
        self,
        *,
        image: torch.Tensor,
        x: torch.Tensor,
        step: torch.Tensor,
        embedding: object,
        cfg_scale: float,
    ) -> torch.Tensor:
        del image, embedding, cfg_scale
        self.seen_states.append(x.clone())
        self.seen_steps.append(step.clone())
        return torch.full_like(x, 3.0)


def test_projected_sampler_keeps_observed_pixels_fixed_and_uses_two_step_schedule() -> None:
    model = _RecordingVelocityModel()
    condition = torch.zeros(1, 4, 4, 4)
    x0 = torch.full((1, 1, 4, 4), 4.0)
    observation_mask = torch.zeros_like(x0, dtype=torch.bool)
    observation_mask[..., 0, 0] = True
    sparse_map = torch.zeros_like(x0)
    sparse_map[..., 0, 0] = 7.0

    sample = random_task2_euler_cfg_sample(
        model,
        condition=condition,
        x0=x0,
        sparse_map=sparse_map,
        observation_mask=observation_mask,
        cfg_scale=1.0,
        steps=2,
        use_amp=False,
    )

    assert len(model.seen_states) == 2
    assert len(model.seen_steps) == 2
    for state in model.seen_states:
        assert torch.equal(state[observation_mask], sparse_map[observation_mask])
    assert model.seen_steps[0].tolist() == pytest.approx([0.0])
    assert model.seen_steps[1].tolist() == pytest.approx([0.5])
    assert torch.equal(sample[observation_mask], sparse_map[observation_mask])
    assert torch.equal(sample[~observation_mask], torch.full_like(sample[~observation_mask], 7.0))
