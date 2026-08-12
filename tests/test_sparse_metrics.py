from __future__ import annotations

import math

import pytest
import torch


def test_sparse_metrics_report_missing_observed_and_overall_regions() -> None:
    from evaluation.sparse_metrics import SparseMetricAccumulator

    target = torch.full((1, 1, 4, 4), 0.5)
    prediction = target.clone()
    valid_mask = torch.ones_like(target, dtype=torch.bool)
    observation_mask = torch.zeros_like(valid_mask)
    observation_mask[:, :, :, :1] = True
    missing_mask = valid_mask & ~observation_mask
    prediction[missing_mask] += 0.1

    accumulator = SparseMetricAccumulator()
    accumulator.update(prediction, target, valid_mask, observation_mask)
    metrics = accumulator.compute()

    assert set(metrics) == {"missing", "observed", "overall_valid"}
    assert metrics["missing"]["pixel_count"] == 12
    assert metrics["observed"]["pixel_count"] == 4
    assert metrics["overall_valid"]["pixel_count"] == 16
    assert metrics["missing"]["db_rmse"] == pytest.approx(30.0)
    assert metrics["missing"]["db_mae"] == pytest.approx(30.0)
    assert metrics["observed"]["max_abs_error"] == 0.0
    assert metrics["observed"]["mean_abs_error"] == 0.0
    assert math.isfinite(float(metrics["missing"]["ssim"]))


def test_sparse_metrics_use_global_pixel_weighting_not_per_image_average() -> None:
    from evaluation.sparse_metrics import SparseMetricAccumulator

    accumulator = SparseMetricAccumulator()
    target_a = torch.zeros(1, 1, 2, 2)
    pred_a = torch.ones_like(target_a)
    valid_a = torch.ones_like(target_a, dtype=torch.bool)
    obs_a = torch.zeros_like(valid_a)
    target_b = torch.zeros(1, 1, 2, 2)
    pred_b = target_b.clone()
    valid_b = torch.zeros_like(target_b, dtype=torch.bool)
    valid_b[:, :, 0, 0] = True
    obs_b = torch.zeros_like(valid_b)

    accumulator.update(pred_a, target_a, valid_a, obs_a)
    accumulator.update(pred_b, target_b, valid_b, obs_b)

    missing = accumulator.compute()["missing"]
    expected_mse = 4.0 / 5.0
    assert missing["pixel_count"] == 5
    assert missing["db_rmse"] == pytest.approx(math.sqrt(90_000.0 * expected_mse))


def test_sparse_metrics_reject_observation_outside_valid_mask() -> None:
    from evaluation.sparse_metrics import SparseMetricAccumulator, SparseMetricError

    target = torch.zeros(1, 1, 2, 2)
    valid = torch.zeros_like(target, dtype=torch.bool)
    observation = torch.ones_like(valid)

    with pytest.raises(SparseMetricError, match="subset"):
        SparseMetricAccumulator().update(target, target, valid, observation)
