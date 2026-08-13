from __future__ import annotations

import torch

from evaluation.sparse_task2_sampling import (
    make_task2_sample_noise,
    sparse_task2_euler_cfg_sample,
)


class _DummyModel:
    def embed_model(self, condition):
        return condition

    def forward_with_cfg(self, *, image, x, step, embedding, cfg_scale):
        return torch.ones_like(x) * float(cfg_scale)


def test_task2_noise_is_namespaced_and_deterministic() -> None:
    first = make_task2_sample_noise(
        protocol="singlebeam_feature5_samples819",
        array_size="8x8",
        split="val",
        sample_key="8x8|u1|beam04",
    )
    second = make_task2_sample_noise(
        protocol="singlebeam_feature5_samples819",
        array_size="8x8",
        split="val",
        sample_key="8x8|u1|beam04",
    )
    other_split = make_task2_sample_noise(
        protocol="singlebeam_feature5_samples819",
        array_size="8x8",
        split="test",
        sample_key="8x8|u1|beam04",
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, other_split)


def test_source_equivalent_does_not_project_and_projection_is_opt_in() -> None:
    model = _DummyModel()
    condition = torch.zeros(1, 5, 4, 4)
    noise = torch.zeros(1, 1, 4, 4)
    observation_mask = torch.zeros_like(noise, dtype=torch.bool)
    observation_mask[..., 0, 0] = True
    sparse_map = torch.full_like(noise, 7.0)

    source = sparse_task2_euler_cfg_sample(
        model, condition, noise, cfg_scale=1.0, steps=2,
        observation_mask=observation_mask, sparse_map=sparse_map,
    )
    projected = sparse_task2_euler_cfg_sample(
        model, condition, noise, cfg_scale=1.0, steps=2,
        observation_mask=observation_mask, sparse_map=sparse_map,
        projected_consistency=True,
    )
    assert source[0, 0, 0, 0].item() == 1.0
    assert projected[0, 0, 0, 0].item() == 7.0
    assert projected[0, 0, 0, 1].item() == 1.0
