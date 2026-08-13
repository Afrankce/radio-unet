from __future__ import annotations

import torch

from evaluation.sparse_task2_metrics import SparseTask2MetricAccumulator


def test_task2_metrics_report_overall_missing_observed_and_groups() -> None:
    target = torch.full((2, 1, 4, 4), 0.5)
    prediction = target + 0.1
    valid = torch.ones_like(target, dtype=torch.bool)
    observed = torch.zeros_like(valid)
    observed[..., :2, :2] = True
    metadata = [
        {"scene_id": "u1", "array_size": "8x8", "steering_deg": 0.0},
        {"scene_id": "u2", "array_size": "8x8", "steering_deg": 0.0},
    ]
    accumulator = SparseTask2MetricAccumulator()
    accumulator.update(prediction, target, valid, observed, metadata)
    metrics = accumulator.compute()
    assert metrics["overall"]["pixel_count"] == 32
    assert metrics["missing"]["pixel_count"] == 24
    assert metrics["observed"]["pixel_count"] == 8
    assert set(metrics["per_scene"]) == {"u1", "u2"}
    assert set(metrics["per_array"]) == {"8x8"}
    assert set(metrics["per_angle"]) == {"0.0"}
    assert metrics["overall"]["psnr"] > 0.0
