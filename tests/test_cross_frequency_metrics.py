from __future__ import annotations

import pytest
import torch


def _case(count: int = 2):
    target = torch.full((count, 1, 11, 11), 0.5)
    prediction = target.clone()
    prediction[0] += 0.1
    mask = torch.ones_like(target, dtype=torch.bool)
    return prediction, target, mask


def test_frequency_metric_accumulator_groups_and_orders_rows() -> None:
    from evaluation.radiomap_metrics import PerFrequencyMetricAccumulators

    prediction, target, mask = _case()
    collection = PerFrequencyMetricAccumulators(
        expected_groups=((6_700_000_000, 0.0), (4_900_000_000, 0.0))
    )
    collection.update(
        prediction,
        target,
        mask,
        [
            {"frequency_hz": 6_700_000_000, "steering_deg": 0.0},
            {"frequency_hz": 4_900_000_000, "steering_deg": 0.0},
        ],
    )

    overall = collection.compute_overall()
    rows = collection.compute_rows()

    assert overall["n_samples"] == 2
    assert [(row["frequency_hz"], row["angle_deg"]) for row in rows] == [
        (4_900_000_000, 0.0),
        (6_700_000_000, 0.0),
    ]
    assert [row["n_samples"] for row in rows] == [1, 1]


def test_frequency_metric_accumulator_rejects_unknown_or_inconsistent_groups() -> None:
    from evaluation.radiomap_metrics import (
        MetricInputError,
        PerFrequencyMetricAccumulators,
    )

    prediction, target, mask = _case(1)
    collection = PerFrequencyMetricAccumulators(
        expected_groups=((6_700_000_000, 0.0),)
    )
    with pytest.raises(MetricInputError, match="unexpected frequency/angle group"):
        collection.update(
            prediction,
            target,
            mask,
            [{"frequency_hz": 4_900_000_000, "steering_deg": 0.0}],
        )
    with pytest.raises(MetricInputError, match="metadata count"):
        collection.update(prediction, target, mask, [])


def test_frequency_metric_accumulator_requires_nonempty_expected_groups() -> None:
    from evaluation.radiomap_metrics import MetricInputError, PerFrequencyMetricAccumulators

    prediction, target, mask = _case(1)
    collection = PerFrequencyMetricAccumulators(
        expected_groups=((4_900_000_000, 0.0), (6_700_000_000, 0.0))
    )
    collection.update(
        prediction,
        target,
        mask,
        [{"frequency_hz": 4_900_000_000, "steering_deg": 0.0}],
    )
    with pytest.raises(MetricInputError, match="no evaluated samples"):
        collection.compute_rows()
