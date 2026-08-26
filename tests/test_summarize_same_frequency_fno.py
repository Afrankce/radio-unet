from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from summarize_same_frequency_fno import (
    FNOExperimentSummaryError,
    apply_registered_decision,
    collect_registered_results,
    write_registered_summary,
)


def test_registered_decision_uses_all_three_guards() -> None:
    result = apply_registered_decision(
        {"8x8": 11.20, "16x16": 11.30, "32x32": 11.35}
    )

    assert result["mean_delta_db"] >= 0.3
    assert result["n_improved"] == 3
    assert result["worst_delta_db"] > -0.5
    assert result["h1_confirmed"] is True


def test_large_single_array_regression_disconfirms_h1() -> None:
    result = apply_registered_decision(
        {"8x8": 10.5, "16x16": 10.5, "32x32": 12.3}
    )

    assert result["mean_delta_db"] >= 0.3
    assert result["n_improved"] == 2
    assert result["worst_delta_db"] < -0.5
    assert result["h1_confirmed"] is False


@pytest.mark.parametrize(
    "values",
    [
        {"8x8": 11.0, "16x16": 11.0},
        {"8x8": 11.0, "16x16": 11.0, "32x32": math.inf},
        {"8x8": 11.0, "16x16": 11.0, "32x32": 11.0, "extra": 1.0},
    ],
)
def test_registered_decision_rejects_incomplete_or_nonfinite_inputs(values) -> None:
    with pytest.raises(FNOExperimentSummaryError):
        apply_registered_decision(values)


def _write_result(root: Path, array_size: str, rmse: float) -> None:
    directory = root / "test" / array_size
    directory.mkdir(parents=True)
    metrics = {
        "schema_version": 1,
        "experiment": "same_frequency_6.7_single_beam_paper_fno",
        "array_size": array_size,
        "model_size": "paper_fno_lite",
        "split": "test",
        "n_samples": 160,
        "db_rmse": rmse,
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "same_frequency_6.7_single_beam_paper_fno",
        "array_size": array_size,
        "model_size": "paper_fno_lite",
        "split": "test",
        "n_samples": 160,
    }
    (directory / "metrics_test.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_collect_and_write_registered_summary_uses_exact_three_paths(
    tmp_path: Path,
) -> None:
    for array_size, rmse in {"8x8": 11.2, "16x16": 11.3, "32x32": 11.4}.items():
        _write_result(tmp_path, array_size, rmse)

    collected = collect_registered_results(tmp_path)
    outputs = write_registered_summary(tmp_path / "summary", collected)

    assert collected == {"8x8": 11.2, "16x16": 11.3, "32x32": 11.4}
    assert outputs["json"].is_file()
    assert outputs["csv"].is_file()
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["fno_db_rmse"] == collected
    assert payload["decision"]["h1_confirmed"] is True
    assert len(outputs["csv"].read_text(encoding="utf-8").splitlines()) == 4
