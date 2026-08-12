from __future__ import annotations

import hashlib
import math

import pytest
import torch


def _module():
    from training import sparse_masks

    return sparse_masks


def _expected_mask(
    valid_mask: torch.Tensor,
    *,
    scene_id: str,
    steering_deg: float,
    ratio: float,
    base_seed: int,
) -> torch.Tensor:
    valid_indices = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).flatten()
    observed_count = max(1, math.ceil(ratio * int(valid_indices.numel())))
    seed_material = f"{base_seed}|{scene_id}|{steering_deg:.6f}|{ratio:.8f}".encode(
        "utf-8"
    )
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(valid_indices.numel(), generator=generator)
    selected = valid_indices[order[:observed_count]]
    mask = torch.zeros(valid_mask.numel(), dtype=torch.bool)
    mask[selected] = True
    return mask.reshape(valid_mask.shape) & valid_mask


def test_make_observation_mask_is_deterministic_uses_ceil_count_and_never_observes_invalid() -> None:
    module = _module()
    valid_mask = torch.tensor([[[True, False, True], [True, True, False]]])

    mask = module.make_observation_mask(
        valid_mask,
        scene_id="scene-1",
        steering_deg=0.0,
        ratio=0.26,
        base_seed=42,
    )

    assert torch.equal(
        mask,
        _expected_mask(
            valid_mask,
            scene_id="scene-1",
            steering_deg=0.0,
            ratio=0.26,
            base_seed=42,
        ),
    )
    assert mask.dtype == torch.bool
    assert torch.equal(mask & ~valid_mask, torch.zeros_like(mask))
    assert int(mask.sum().item()) == 2
    assert torch.equal(
        mask,
        module.make_observation_mask(
            valid_mask,
            scene_id="scene-1",
            steering_deg=0.0,
            ratio=0.26,
            base_seed=42,
        ),
    )


def test_make_observation_mask_supports_batched_bool_shape() -> None:
    module = _module()
    valid_mask = torch.tensor(
        [
            [[[True, False], [True, True]]],
            [[[False, True], [True, False]]],
        ]
    )

    mask = module.make_observation_mask(
        valid_mask,
        scene_id="scene-batch",
        steering_deg=0.0,
        ratio=0.4,
        base_seed=7,
    )

    assert mask.shape == valid_mask.shape
    assert mask.dtype == torch.bool
    assert torch.equal(mask & ~valid_mask, torch.zeros_like(mask))
    assert int(mask.sum().item()) == math.ceil(0.4 * int(valid_mask.sum().item()))


@pytest.mark.parametrize("bad_ratio", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_make_observation_mask_rejects_invalid_ratio(bad_ratio: float) -> None:
    module = _module()

    with pytest.raises(ValueError, match="ratio"):
        module.make_observation_mask(
            torch.ones(1, 2, 2, dtype=torch.bool),
            scene_id="scene-1",
            steering_deg=0.0,
            ratio=bad_ratio,
            base_seed=42,
        )


def test_make_observation_mask_rejects_bad_mask_inputs() -> None:
    module = _module()

    with pytest.raises(ValueError, match="boolean"):
        module.make_observation_mask(
            torch.ones(1, 2, 2),
            scene_id="scene-1",
            steering_deg=0.0,
            ratio=0.5,
            base_seed=42,
        )
    with pytest.raises(ValueError, match="shape"):
        module.make_observation_mask(
            torch.ones(2, 2, dtype=torch.bool),
            scene_id="scene-1",
            steering_deg=0.0,
            ratio=0.5,
            base_seed=42,
        )
    with pytest.raises(ValueError, match="valid"):
        module.make_observation_mask(
            torch.zeros(1, 2, 2, dtype=torch.bool),
            scene_id="scene-1",
            steering_deg=0.0,
            ratio=0.5,
            base_seed=42,
        )


def test_make_condition_noise_is_deterministic_split_sensitive_and_cpu_float() -> None:
    module = _module()

    noise = module.make_condition_noise(
        (1, 2, 3),
        scene_id="scene-1",
        steering_deg=0.0,
        split="val",
        epoch=5,
        base_seed=4242,
    )
    repeat = module.make_condition_noise(
        (1, 2, 3),
        scene_id="scene-1",
        steering_deg=0.0,
        split="val",
        epoch=5,
        base_seed=4242,
    )
    different_epoch = module.make_condition_noise(
        (1, 2, 3),
        scene_id="scene-1",
        steering_deg=0.0,
        split="val",
        epoch=6,
        base_seed=4242,
    )

    assert torch.equal(noise, repeat)
    assert not torch.equal(noise, different_epoch)
    assert noise.shape == (1, 2, 3)
    assert noise.dtype == torch.float32
    assert noise.device.type == "cpu"


@pytest.mark.parametrize("shape", [(1, 2), (1, -1, 3), (1, 2, 3, 4)])
def test_make_condition_noise_rejects_invalid_shape(shape: tuple[int, ...]) -> None:
    module = _module()

    with pytest.raises(ValueError, match="shape"):
        module.make_condition_noise(
            shape,  # type: ignore[arg-type]
            scene_id="scene-1",
            steering_deg=0.0,
            split="test",
            epoch=None,
            base_seed=4242,
        )


def test_build_masked_condition_map_uses_noise_only_for_missing_valid_pixels() -> None:
    module = _module()
    target = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
    valid_mask = torch.tensor([[[True, True], [False, True]]])
    observation_mask = torch.tensor([[[True, False], [False, True]]])
    condition_noise = torch.tensor([[[9.0, 8.0], [7.0, 6.0]]])

    masked, observed, missing = module.build_masked_condition_map(
        target,
        valid_mask,
        observation_mask,
        condition_noise,
    )

    assert torch.equal(missing, torch.tensor([[[False, True], [False, False]]]))
    assert torch.equal(observed, torch.tensor([[[0.1, 0.0], [0.0, 0.4]]]))
    assert torch.equal(masked, torch.tensor([[[0.1, 8.0], [0.0, 0.4]]]))


def test_build_masked_condition_map_rejects_bad_inputs() -> None:
    module = _module()
    target = torch.zeros(1, 2, 2)
    valid_mask = torch.ones(1, 2, 2, dtype=torch.bool)
    observation_mask = torch.ones(1, 2, 2, dtype=torch.bool)
    condition_noise = torch.zeros(1, 2, 2)

    with pytest.raises(ValueError, match="shape"):
        module.build_masked_condition_map(
            target[:, :1],
            valid_mask,
            observation_mask,
            condition_noise,
        )
    with pytest.raises(ValueError, match="boolean"):
        module.build_masked_condition_map(
            target,
            valid_mask.to(torch.float32),  # type: ignore[arg-type]
            observation_mask,
            condition_noise,
        )
    with pytest.raises(ValueError, match="subset"):
        module.build_masked_condition_map(
            target,
            torch.tensor([[[True, False], [True, True]]]),
            torch.tensor([[[True, True], [True, True]]]),
            condition_noise,
        )
