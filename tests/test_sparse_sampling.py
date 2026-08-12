from __future__ import annotations

import pytest
import torch


class SparseRecordingModel:
    def __init__(self) -> None:
        self.embed_calls = 0
        self.steps: list[torch.Tensor] = []

    def embed_model(self, condition: torch.Tensor):
        self.embed_calls += 1
        return {"shape": tuple(condition.shape)}

    def forward_with_cfg(
        self,
        *,
        image: torch.Tensor,
        x: torch.Tensor,
        step: torch.Tensor,
        embedding: object,
        cfg_scale: float,
    ) -> torch.Tensor:
        self.steps.append(step.detach().clone())
        return torch.ones_like(x) * float(cfg_scale)


def test_sparse_euler_cfg_sample_projects_observed_pixels_at_start_and_each_step() -> None:
    from evaluation.sparse_sampling import sparse_euler_cfg_sample

    model = SparseRecordingModel()
    condition = torch.zeros(1, 5, 3, 3)
    observed_map = torch.zeros(1, 1, 3, 3)
    observed_map[:, :, 1, 1] = 0.25
    observation_mask = observed_map > 0
    x0 = torch.zeros(1, 1, 3, 3)

    prediction = sparse_euler_cfg_sample(
        model,
        condition,
        observed_map,
        observation_mask,
        x0,
        cfg_scale=2.0,
        steps=2,
        use_amp=False,
    )

    assert model.embed_calls == 1
    assert torch.equal(model.steps[0], torch.tensor([0.0]))
    assert torch.equal(model.steps[1], torch.tensor([0.5]))
    assert prediction[0, 0, 1, 1].item() == pytest.approx(0.25)
    assert torch.allclose(prediction[~observation_mask], torch.full((8,), 2.0))


def test_sparse_euler_cfg_sample_rejects_bad_sparse_contract() -> None:
    from evaluation.sparse_sampling import sparse_euler_cfg_sample

    condition = torch.zeros(1, 4, 3, 3)
    observed_map = torch.zeros(1, 1, 3, 3)
    observation_mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    x0 = torch.zeros(1, 1, 3, 3)

    with pytest.raises(ValueError, match="5-channel"):
        sparse_euler_cfg_sample(
            SparseRecordingModel(),
            condition,
            observed_map,
            observation_mask,
            x0,
            cfg_scale=1.0,
            use_amp=False,
        )
    with pytest.raises(ValueError, match="non-observed"):
        sparse_euler_cfg_sample(
            SparseRecordingModel(),
            torch.zeros(1, 5, 3, 3),
            observed_map + 0.1,
            observation_mask,
            x0,
            cfg_scale=1.0,
            use_amp=False,
        )
