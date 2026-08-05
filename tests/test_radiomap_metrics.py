from __future__ import annotations

import math

import pytest
import torch


def _metrics_module():
    from evaluation import radiomap_metrics

    return radiomap_metrics


def _full_case(size: int = 11) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.linspace(0.1, 0.9, size * size).reshape(1, 1, size, size)
    prediction = target.clone()
    mask = torch.ones_like(target, dtype=torch.bool)
    return prediction, target, mask


def test_perfect_prediction_metrics_have_exact_limits() -> None:
    module = _metrics_module()
    prediction, target, mask = _full_case()
    accumulator = module.MetricAccumulator()

    accumulator.update(prediction, target, mask)
    metrics = accumulator.compute()

    assert metrics["n_samples"] == 1
    assert metrics["n_valid_pixels"] == 121
    assert metrics["db_rmse"] == 0.0
    assert metrics["db_mae"] == 0.0
    assert metrics["mse"] == 0.0
    assert metrics["nmse"] == 0.0
    assert math.isinf(metrics["psnr"])
    assert metrics["ssim"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["raw_fraction_below_zero"] == 0.0
    assert metrics["raw_fraction_above_one"] == 0.0


def test_invalid_pixels_affect_no_metric() -> None:
    module = _metrics_module()
    target = torch.full((1, 1, 13, 13), 0.5)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:, :, 1:12, 1:12] = True
    reference = target.clone()
    changed = target.clone()
    changed[~mask] = 999.0
    first = module.MetricAccumulator()
    second = module.MetricAccumulator()

    first.update(reference, target, mask)
    second.update(changed, target, mask)

    assert first.compute() == second.compute()


def test_nonfinite_prediction_in_valid_cell_fails() -> None:
    module = _metrics_module()
    prediction, target, mask = _full_case()
    prediction[0, 0, 5, 5] = float("nan")

    with pytest.raises(module.MetricInputError, match="non-finite prediction"):
        module.MetricAccumulator().update(prediction, target, mask)


def test_empty_valid_mask_fails() -> None:
    module = _metrics_module()
    prediction = torch.zeros(1, 1, 11, 11)
    target = torch.zeros_like(prediction)
    mask = torch.zeros_like(prediction, dtype=torch.bool)

    with pytest.raises(module.MetricInputError, match="empty valid mask"):
        module.MetricAccumulator().update(prediction, target, mask)


def test_metrics_use_global_pixel_weighting() -> None:
    module = _metrics_module()
    accumulator = module.MetricAccumulator()
    target_a = torch.zeros(1, 1, 12, 12)
    pred_a = torch.ones_like(target_a)
    mask_a = torch.ones_like(target_a, dtype=torch.bool)
    target_b = torch.full((1, 1, 12, 12), 0.5)
    pred_b = target_b.clone()
    mask_b = torch.zeros_like(target_b, dtype=torch.bool)
    mask_b[:, :, :11, :11] = True

    accumulator.update(pred_a, target_a, mask_a)
    accumulator.update(pred_b, target_b, mask_b)
    metrics = accumulator.compute()

    expected_mse = 144.0 / (144 + 121)
    assert metrics["n_samples"] == 2
    assert metrics["n_valid_pixels"] == 265
    assert metrics["mse"] == pytest.approx(expected_mse)
    assert metrics["db_rmse"] == pytest.approx(math.sqrt(90_000 * expected_mse))
    assert metrics["db_mae"] == pytest.approx(300 * 144 / 265)


def test_complete_window_mask_erodes_invalid_boundaries() -> None:
    module = _metrics_module()
    mask = torch.ones(1, 1, 13, 13, dtype=torch.bool)

    complete = module.complete_window_mask(mask, window_size=11)
    mask[:, :, 6, 6] = False
    with_hole = module.complete_window_mask(mask, window_size=11)

    assert complete.shape == (1, 1, 3, 3)
    assert complete.sum().item() == 9
    assert with_hole.sum().item() == 0


def test_normalized_and_db_values_round_trip() -> None:
    module = _metrics_module()
    values = torch.tensor([0.0, 1 / 300, 0.5, 1.0])

    db = module.normalized_to_db(values)

    assert torch.allclose(db, torch.tensor([-300.0, -299.0, -150.0, 0.0]))
    assert torch.allclose(module.db_to_normalized(db), values)


def test_per_beam_metrics_have_eight_rows_and_independent_overall() -> None:
    module = _metrics_module()
    beam_angles = {
        0: -28.0,
        1: -21.0,
        2: -14.0,
        3: -7.0,
        4: 0.0,
        5: 7.0,
        6: 14.0,
        7: 21.0,
    }
    collection = module.PerBeamMetricAccumulators(beam_angles)
    target = torch.full((8, 1, 11, 11), 0.5)
    prediction = torch.stack(
        [torch.full((1, 11, 11), 0.5 + beam_id / 20) for beam_id in beam_angles]
    )
    mask = torch.ones_like(target, dtype=torch.bool)
    metadata = [
        {"beam_id": beam_id, "steering_deg": angle}
        for beam_id, angle in beam_angles.items()
    ]

    collection.update(prediction, target, mask, metadata)
    overall = collection.compute_overall()
    rows = collection.compute_rows()

    assert overall["n_samples"] == 8
    assert overall["n_valid_pixels"] == 8 * 121
    assert len(rows) == 8
    assert [row["beam_id"] for row in rows] == list(beam_angles)
    assert [row["angle_deg"] for row in rows] == list(beam_angles.values())
    assert all(row["n_samples"] == 1 for row in rows)


def test_perfect_psnr_has_canonical_json_representation() -> None:
    module = _metrics_module()
    prediction, target, mask = _full_case()
    accumulator = module.MetricAccumulator()
    accumulator.update(prediction, target, mask)

    payload = module.metrics_for_json(accumulator.compute())

    assert payload["psnr"] is None
    assert payload["psnr_infinite"] is True

