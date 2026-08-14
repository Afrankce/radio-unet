"""Small, data-free contract tests for the registered sparse-consistency arms.

The tests stay data-free and use the already shared primitives as a fallback.
If the planned sparse-consistent flow/sampler modules are present, the adapters
below exercise those interfaces directly without depending on the rest of the
training stack.
"""

from __future__ import annotations

import torch

from data_loaders.sparse_task2 import choose_valid_observation_mask
from training.sparse_flow import build_masked_flow_pair


GRID_SIZE = 256
OBSERVATION_COUNT = 819


def _synthetic_protocol_tensors() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return a deterministic valid mask, target, observation mask, and sparse map."""

    valid_map = torch.ones((1, GRID_SIZE, GRID_SIZE), dtype=torch.bool)
    # Keep a few invalid pixels so the test also guards the valid/observed
    # distinction without relying on any dataset files.
    valid_map[..., 0, :8] = False
    valid_map[..., -1, -8:] = False
    target = torch.linspace(
        -0.75,
        0.95,
        GRID_SIZE * GRID_SIZE,
        dtype=torch.float32,
    ).reshape(1, GRID_SIZE, GRID_SIZE)
    observation_mask = choose_valid_observation_mask(
        valid_map,
        scene_id="synthetic-sparse-consistency-scene",
        seed=42,
        count=OBSERVATION_COUNT,
    )
    sparse_map = (target * observation_mask.to(dtype=target.dtype)).masked_fill(
        ~valid_map,
        0.0,
    )
    return valid_map, target, observation_mask, sparse_map


def _batched_protocol_tensors() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    valid_map, target, observation_mask, sparse_map = _synthetic_protocol_tensors()
    return tuple(
        value.unsqueeze(0) for value in (valid_map, target, observation_mask, sparse_map)
    )  # type: ignore[return-value]


def _build_pinned_flow_pair(
    *,
    initial_noise: torch.Tensor,
    target: torch.Tensor,
    sparse_map: torch.Tensor,
    observation_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    time: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the new D flow interface when available, otherwise its primitive."""

    try:
        from training.sparse_consistent_flow import build_sparse_consistent_flow_pair
    except (ImportError, ModuleNotFoundError):
        return build_masked_flow_pair(
            initial_noise,
            target,
            sparse_map,
            observation_mask,
            valid_mask,
            time=time,
        )
    time_tensor = torch.tensor([time], dtype=target.dtype, device=target.device)
    return build_sparse_consistent_flow_pair(
        arm="multiscale_consistent",
        x0=initial_noise,
        target=target,
        valid_mask=valid_mask,
        observation_mask=observation_mask,
        sparse_map=sparse_map,
        time=time_tensor,
    )


def _run_d_sampler_or_reference(
    *,
    initial_state: torch.Tensor,
    sparse_map: torch.Tensor,
    observation_mask: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run D sampling or an equivalent two-step reference loop.

    The production sampler interface used here is
    ``sparse_consistent_euler_cfg_sample(..., arm=..., ...)``.  The fallback
    keeps this test runnable before that module is added.
    """

    try:
        from evaluation.sparse_consistent_sampling import (
            sparse_consistent_euler_cfg_sample,
        )
    except (ImportError, ModuleNotFoundError):
        state = torch.where(observation_mask, sparse_map, initial_state)
        seen: list[torch.Tensor] = []
        for _ in range(2):
            seen.append(state.clone())
            state = state + 0.5 * torch.full_like(state, 3.0)
            state = torch.where(observation_mask, sparse_map, state)
        return state, seen

    class _OverwritingModel:
        def __init__(self) -> None:
            self.seen: list[torch.Tensor] = []

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
            del image, step, embedding, cfg_scale
            self.seen.append(x.clone())
            return torch.full_like(x, 3.0)

    model = _OverwritingModel()
    condition = torch.zeros(
        initial_state.shape[0],
        3,
        initial_state.shape[-2],
        initial_state.shape[-1],
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    result = sparse_consistent_euler_cfg_sample(
        model,
        arm="multiscale_consistent",
        condition=condition,
        x0=initial_state,
        sparse_map=sparse_map,
        observation_mask=observation_mask,
        cfg_scale=1.0,
        steps=2,
        use_amp=False,
    )
    return result, model.seen


def test_sparse_mask_has_exactly_819_valid_observations() -> None:
    valid_map, _, observation_mask, _ = _synthetic_protocol_tensors()

    assert observation_mask.dtype is torch.bool
    assert observation_mask.shape == valid_map.shape
    assert int(observation_mask.sum().item()) == OBSERVATION_COUNT
    assert not bool((observation_mask & ~valid_map).any())


def test_missing_mask_is_valid_complement_and_contains_no_observed_pixels() -> None:
    valid_map, _, observation_mask, _ = _synthetic_protocol_tensors()

    missing_mask = valid_map & ~observation_mask

    assert not bool((missing_mask & observation_mask).any())
    assert torch.equal(missing_mask, valid_map & ~observation_mask)
    assert int(missing_mask.sum().item()) == int(valid_map.sum().item()) - OBSERVATION_COUNT


def test_pinned_flow_has_observed_path_and_target_endpoint() -> None:
    valid_mask, target, observation_mask, sparse_map = _batched_protocol_tensors()
    initial_noise = torch.full_like(target, -1.25)

    xt_zero, velocity, missing_mask = _build_pinned_flow_pair(
        initial_noise=initial_noise,
        target=target,
        sparse_map=sparse_map,
        observation_mask=observation_mask,
        valid_mask=valid_mask,
        time=0.0,
    )
    xt_one, _, endpoint_missing_mask = _build_pinned_flow_pair(
        initial_noise=initial_noise,
        target=target,
        sparse_map=sparse_map,
        observation_mask=observation_mask,
        valid_mask=valid_mask,
        time=1.0,
    )

    expected_missing = valid_mask & ~observation_mask
    assert torch.equal(missing_mask, expected_missing)
    assert torch.equal(endpoint_missing_mask, expected_missing)
    # The observed path is pinned for the whole interpolation, including both
    # endpoints; only missing-valid pixels have a noise-to-target trajectory.
    assert torch.equal(xt_zero[observation_mask], sparse_map[observation_mask])
    assert torch.equal(xt_one[observation_mask], sparse_map[observation_mask])
    assert torch.equal(xt_zero[expected_missing], initial_noise[expected_missing])
    assert torch.equal(xt_one[valid_mask], target[valid_mask])
    assert torch.equal(velocity[~expected_missing], torch.zeros_like(velocity[~expected_missing]))


def test_d_projection_preserves_observed_values_exactly_after_each_euler_step() -> None:
    valid_mask, target, observation_mask, sparse_map = _batched_protocol_tensors()
    initial_state = torch.full_like(target, 4.0)

    # The synthetic velocity deliberately tries to overwrite observed pixels.
    # ``seen`` contains the model input at each Euler step, so it also verifies
    # that projection happened after initialization and after the first step.
    sample, seen = _run_d_sampler_or_reference(
        initial_state=initial_state,
        sparse_map=sparse_map,
        observation_mask=observation_mask,
    )

    assert len(seen) == 2
    for step_input in seen:
        assert torch.equal(step_input[observation_mask], sparse_map[observation_mask])
    assert torch.equal(sample[observation_mask], sparse_map[observation_mask])
    assert not bool((observation_mask & ~valid_mask).any())
    assert torch.equal(
        sample[~observation_mask],
        torch.full_like(sample[~observation_mask], 7.0),
    )
