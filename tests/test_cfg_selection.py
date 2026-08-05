from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.multiconfig_manifest import canonical_json_bytes


def _evaluator_module():
    from evaluation import multiconfig_evaluator

    return multiconfig_evaluator


def _metrics(values: dict[float, float]) -> dict[float, dict[str, float | int]]:
    return {
        scale: {
            "n_samples": 640,
            "n_valid_pixels": 1000,
            "db_rmse": rmse,
            "db_mae": rmse / 2,
            "mse": (rmse / 300) ** 2,
            "nmse": 0.1,
            "psnr": 20.0,
            "ssim": 0.8,
            "raw_fraction_below_zero": 0.0,
            "raw_fraction_above_one": 0.0,
        }
        for scale, rmse in values.items()
    }


def _identity() -> dict[str, str]:
    return {
        "checkpoint_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "manifest_sha256": "3" * 64,
        "split_sha256": "4" * 64,
        "schema_sha256": "5" * 64,
        "archive_sha256": "6" * 64,
        "dataset_revision": "7" * 40,
        "radioflow_upstream_base": "8" * 40,
        "git_commit": "9" * 40,
    }


def test_cfg_candidates_and_smaller_scale_tie_break_are_fixed() -> None:
    module = _evaluator_module()
    candidate_metrics = _metrics({1.0: 10.0, 1.5: 8.0, 2.0: 8.0, 2.5: 9.0})

    selected = module.select_cfg_candidate(candidate_metrics)

    assert module.CFG_CANDIDATES == (1.0, 1.5, 2.0, 2.5)
    assert selected == 1.5


def test_cfg_selection_payload_freezes_epoch_metrics_and_hashes(tmp_path: Path) -> None:
    module = _evaluator_module()
    metrics = _metrics({1.0: 10.0, 1.5: 8.0, 2.0: 9.0, 2.5: 11.0})
    payload = module.build_cfg_selection_payload(
        array_size="8x8",
        model_size="lite",
        selected_epoch=7,
        candidate_metrics=metrics,
        identity=_identity(),
    )
    path = tmp_path / "cfg_selection.json"

    digest = module.freeze_cfg_selection(path, payload)

    assert len(digest) == 64
    assert path.read_bytes() == canonical_json_bytes(payload)
    assert payload["candidates"] == [1.0, 1.5, 2.0, 2.5]
    assert payload["selected_scale"] == 1.5
    assert payload["selected_epoch"] == 7
    assert payload["best_validation_db_rmse_cfg1"] == 10.0
    assert payload["selected_validation_db_rmse"] == 8.0
    assert payload["tie_break_rule"] == "minimum_db_rmse_then_smaller_cfg"
    assert payload["solver"] == "euler"
    assert payload["euler_steps"] == 2
    assert payload["ema"] is True
    assert payload["identity"] == _identity()
    assert module.load_cfg_selection(path, _identity()) == payload


def test_existing_cfg_selection_is_validation_only_and_never_overwritten(
    tmp_path: Path,
) -> None:
    module = _evaluator_module()
    metrics = _metrics({1.0: 10.0, 1.5: 8.0, 2.0: 9.0, 2.5: 11.0})
    payload = module.build_cfg_selection_payload(
        array_size="8x8",
        model_size="lite",
        selected_epoch=7,
        candidate_metrics=metrics,
        identity=_identity(),
    )
    path = tmp_path / "cfg_selection.json"
    module.freeze_cfg_selection(path, payload)
    before = path.read_bytes()

    assert module.freeze_cfg_selection(path, payload) == module.sha256_file(path)
    assert path.read_bytes() == before
    changed = dict(payload, selected_scale=2.0)
    with pytest.raises(module.EvaluationContractError, match="already exists"):
        module.freeze_cfg_selection(path, changed)
    assert path.read_bytes() == before


def test_completed_test_receipt_forbids_new_selection(tmp_path: Path) -> None:
    module = _evaluator_module()
    metrics = _metrics({1.0: 10.0, 1.5: 8.0, 2.0: 9.0, 2.5: 11.0})
    payload = module.build_cfg_selection_payload(
        array_size="8x8",
        model_size="lite",
        selected_epoch=7,
        candidate_metrics=metrics,
        identity=_identity(),
    )
    receipt = tmp_path / "results" / "run_manifest.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"status": "complete"}), encoding="utf-8")

    with pytest.raises(module.EvaluationContractError, match="completed test"):
        module.freeze_cfg_selection(
            tmp_path / "cfg_selection.json",
            payload,
            completed_manifest_path=receipt,
        )


def test_cfg_selection_requires_exact_candidate_grid() -> None:
    module = _evaluator_module()

    with pytest.raises(module.EvaluationContractError, match="candidate grid"):
        module.select_cfg_candidate(_metrics({1.0: 10.0, 1.5: 8.0, 2.0: 9.0}))


@pytest.mark.parametrize("changed_key", sorted(_identity()))
def test_cfg_selection_rejects_every_changed_source_identity(
    tmp_path: Path,
    changed_key: str,
) -> None:
    module = _evaluator_module()
    identity = _identity()
    payload = module.build_cfg_selection_payload(
        array_size="8x8",
        model_size="lite",
        selected_epoch=7,
        candidate_metrics=_metrics({1.0: 10.0, 1.5: 8.0, 2.0: 9.0, 2.5: 11.0}),
        identity=identity,
    )
    path = tmp_path / "cfg_selection.json"
    module.freeze_cfg_selection(path, payload)
    changed = dict(identity)
    changed[changed_key] = ("a" if identity[changed_key][0] != "a" else "b") + identity[
        changed_key
    ][1:]

    with pytest.raises(module.EvaluationContractError, match="identity mismatch"):
        module.load_cfg_selection(path, changed)


def test_cfg_selection_is_required_and_selected_value_is_recomputed(
    tmp_path: Path,
) -> None:
    module = _evaluator_module()
    missing = tmp_path / "cfg_selection.json"
    with pytest.raises(module.EvaluationContractError, match="cannot read CFG selection"):
        module.load_cfg_selection(missing, _identity())

    payload = module.build_cfg_selection_payload(
        array_size="8x8",
        model_size="lite",
        selected_epoch=7,
        candidate_metrics=_metrics({1.0: 10.0, 1.5: 8.0, 2.0: 9.0, 2.5: 11.0}),
        identity=_identity(),
    )
    payload["selected_scale"] = 2.0
    missing.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(module.EvaluationContractError, match="selected scale"):
        module.load_cfg_selection(missing, _identity())
