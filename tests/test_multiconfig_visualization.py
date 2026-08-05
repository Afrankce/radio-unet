from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _visualization_module():
    from evaluation import visualization

    return visualization


def _case():
    condition = torch.zeros(3, 16, 16)
    condition[0, 7, 7] = 1.0
    condition[1] = 0.25
    condition[2] = 0.5
    target = torch.full((1, 16, 16), 0.5)
    prediction = target.clone()
    prediction[0, 0, 0] = -1.0
    prediction[0, 0, 1] = 2.0
    mask = torch.ones(1, 16, 16, dtype=torch.bool)
    mask[0, 1, 1] = False
    metadata = {
        "sample_key": "u731|8x8|beam04",
        "scene_id": "u731",
        "array_name": "8x8",
        "beam_id": 4,
        "steering_deg": 0.0,
    }
    return condition, target, prediction, mask, metadata


def test_visualization_arrays_use_fixed_db_ranges_and_gray_invalid_mask() -> None:
    module = _visualization_module()
    condition, target, prediction, mask, _metadata = _case()

    arrays = module.prepare_visualization_arrays(
        condition,
        target,
        prediction,
        mask,
    )

    assert module.POWER_DB_RANGE == (-300.0, 0.0)
    assert module.ERROR_DB_RANGE == (0.0, 300.0)
    assert arrays["height"].shape == (16, 16)
    assert arrays["beam_db"][0, 0] == -150.0
    assert arrays["ground_truth_db"][0, 0] == -150.0
    assert arrays["prediction_db"][0, 0] == -300.0
    assert arrays["prediction_db"][0, 1] == 0.0
    assert arrays["absolute_error_db"][0, 0] == 150.0
    assert arrays["ground_truth_db"].mask[1, 1]
    assert arrays["prediction_db"].mask[1, 1]
    assert arrays["absolute_error_db"].mask[1, 1]


def test_prediction_npz_is_compressed_pickle_free_and_contains_metadata(
    tmp_path: Path,
) -> None:
    module = _visualization_module()
    _condition, target, prediction, mask, metadata = _case()
    path = tmp_path / "prediction.npz"

    module.save_prediction_npz(
        path,
        prediction=prediction,
        target=target,
        valid_mask=mask,
        metadata=metadata,
    )

    with np.load(path, allow_pickle=False) as payload:
        assert set(payload.files) == {"prediction", "target", "valid_mask", "metadata_json"}
        assert payload["prediction"].shape == (1, 16, 16)
        assert payload["target"].shape == (1, 16, 16)
        assert payload["valid_mask"].dtype == np.bool_
        assert json.loads(payload["metadata_json"].item()) == metadata


def test_stable_case_filename_and_rendered_outputs(tmp_path: Path) -> None:
    module = _visualization_module()
    condition, target, prediction, mask, metadata = _case()

    stem = module.stable_case_stem(metadata, model_size="lite", cfg_scale=1.5)
    comparison = tmp_path / "comparisons" / f"{stem}.png"
    error_map = tmp_path / "error_maps" / f"{stem}.png"
    module.render_comparison(
        comparison,
        condition=condition,
        target=target,
        prediction=prediction,
        valid_mask=mask,
        metadata=metadata,
        model_size="lite",
        cfg_scale=1.5,
    )
    module.render_error_map(
        error_map,
        target=target,
        prediction=prediction,
        valid_mask=mask,
        metadata=metadata,
        model_size="lite",
        cfg_scale=1.5,
    )

    assert stem == "u731__8x8__beam04__angle_p00.0__lite__cfg1.5"
    assert comparison.is_file() and comparison.stat().st_size > 1000
    assert error_map.is_file() and error_map.stat().st_size > 1000

