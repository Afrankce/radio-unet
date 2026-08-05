from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.multiconfig_manifest import canonical_json_bytes


def _module():
    from evaluation import multiconfig_evaluator

    return multiconfig_evaluator


def _metric_row(beam_id: int, angle: float, n_samples: int = 160):
    return {
        "angle_deg": angle,
        "beam_id": beam_id,
        "n_samples": n_samples,
        "n_valid_pixels": n_samples * 121,
        "n_ssim_windows": n_samples,
        "db_rmse": 10.0 + beam_id,
        "db_mae": 5.0,
        "mse": 0.01,
        "nmse": 0.02,
        "psnr": 20.0,
        "ssim": 0.8,
        "raw_fraction_below_zero": 0.0,
        "raw_fraction_above_one": 0.0,
    }


def _overall():
    return {
        "n_samples": 1280,
        "n_valid_pixels": 1280 * 121,
        "n_ssim_windows": 1280,
        "db_rmse": 12.0,
        "db_mae": 5.0,
        "mse": 0.01,
        "nmse": 0.02,
        "psnr": 20.0,
        "ssim": 0.8,
        "raw_fraction_below_zero": 0.0,
        "raw_fraction_above_one": 0.0,
    }


def test_test_metric_contract_and_per_beam_csv_are_exact(tmp_path: Path) -> None:
    module = _module()
    angles = (-28.0, -21.0, -14.0, -7.0, 0.0, 7.0, 14.0, 21.0)
    rows = [_metric_row(index, angle) for index, angle in enumerate(angles)]

    module.validate_test_metric_counts(_overall(), rows)
    path = tmp_path / "metrics_per_beam.csv"
    module.write_metrics_per_beam_csv(path, rows)

    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        serialized = list(reader)
        assert reader.fieldnames == list(module.PER_BEAM_COLUMNS)
    assert len(serialized) == 8
    assert [float(row["angle_deg"]) for row in serialized] == list(angles)

    bad_rows = [dict(row) for row in rows]
    bad_rows[3]["n_samples"] = 159
    with pytest.raises(module.EvaluationContractError, match="160"):
        module.validate_test_metric_counts(_overall(), bad_rows)


@pytest.mark.parametrize(
    "failure_stage",
    ["prediction", "metrics", "runtime", "visualization"],
)
def test_failure_at_any_evaluation_stage_publishes_nothing(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    module = _module()
    final_dir = tmp_path / "results" / "8x8" / "lite"
    stages = ["prediction", "metrics", "runtime", "visualization"]

    def builder(staging: Path) -> None:
        for stage in stages:
            (staging / f"{stage}.tmp").write_text(stage, encoding="utf-8")
            if stage == failure_stage:
                raise RuntimeError(f"injected {stage} failure")

    with pytest.raises(RuntimeError, match=failure_stage):
        module.atomic_result_transaction(final_dir, builder)

    assert not final_dir.exists()
    assert not (final_dir / "metrics_test.json").exists()
    assert not (final_dir / "run_manifest.json").exists()
    assert list(final_dir.parent.glob("lite.staging-*")) == []


def test_atomic_result_transaction_validates_hashes_and_refuses_rerun(
    tmp_path: Path,
) -> None:
    module = _module()
    final_dir = tmp_path / "results" / "8x8" / "lite"

    def builder(staging: Path) -> None:
        (staging / "metrics_test.json").write_bytes(
            canonical_json_bytes({"n_samples": 1280})
        )
        module.write_run_manifest(
            staging,
            {
                "schema_version": 1,
                "status": "complete",
                "array_size": "8x8",
                "model_size": "lite",
                "cfg_selection_sha256": "a" * 64,
            },
        )

    published = module.atomic_result_transaction(final_dir, builder)

    assert published == final_dir
    manifest = module.validate_run_manifest_artifacts(final_dir)
    assert manifest["status"] == "complete"
    assert "metrics_test.json" in manifest["artifacts"]
    with pytest.raises(module.EvaluationContractError, match="already exists"):
        module.atomic_result_transaction(final_dir, builder)

    (final_dir / "metrics_test.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(module.EvaluationContractError, match="hash mismatch"):
        module.validate_run_manifest_artifacts(final_dir)


def test_transaction_rejects_builder_without_completion_receipt(tmp_path: Path) -> None:
    module = _module()
    final_dir = tmp_path / "results" / "8x8" / "lite"

    with pytest.raises(module.EvaluationContractError, match="run_manifest"):
        module.atomic_result_transaction(
            final_dir,
            lambda staging: (staging / "metrics_test.json").write_text(
                "partial", encoding="utf-8"
            ),
        )

    assert not final_dir.exists()


def test_fixed_visualization_cases_are_cross_array_scene_angle_keys() -> None:
    module = _module()
    angles = (-28.0, -21.0, -14.0, -7.0, 0.0, 7.0, 14.0, 21.0)
    metadata = [
        {
            "sample_key": f"u{scene}|8x8|beam{beam:02d}",
            "scene_id": f"u{scene}",
            "beam_id": beam,
            "steering_deg": angle,
        }
        for scene in range(160)
        for beam, angle in enumerate(angles)
    ]

    selected = module.fixed_visualization_sample_keys(metadata)

    assert len(selected) == 12
    selected_pairs = {
        (item.split("|")[0], float(item.rsplit("|", 1)[1])) for item in selected
    }
    assert {scene for scene, _angle in selected_pairs} == {"u0", "u80", "u159"}
    assert {angle for _scene, angle in selected_pairs} == {-28.0, -7.0, 7.0, 21.0}


def test_manifest_must_be_canonical_and_complete(tmp_path: Path) -> None:
    module = _module()
    directory = tmp_path / "result"
    directory.mkdir()
    (directory / "artifact.txt").write_text("x", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "partial",
        "artifacts": {"artifact.txt": module.sha256_file(directory / "artifact.txt")},
    }
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    with pytest.raises(module.EvaluationContractError, match="canonical|complete"):
        module.validate_run_manifest_artifacts(directory)


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


def _selection_payload(module, *, selected_epoch: int = 7):
    values = {1.0: 10.0, 1.5: 8.0, 2.0: 9.0, 2.5: 11.0}
    metrics = {
        scale: {
            "n_samples": 640,
            "n_valid_pixels": 640 * 121,
            "n_ssim_windows": 640,
            "db_rmse": rmse,
            "db_mae": 5.0,
            "mse": 0.01,
            "nmse": 0.02,
            "psnr": 20.0,
            "ssim": 0.8,
            "raw_fraction_below_zero": 0.0,
            "raw_fraction_above_one": 0.0,
        }
        for scale, rmse in values.items()
    }
    return module.build_cfg_selection_payload(
        array_size="8x8",
        model_size="lite",
        selected_epoch=selected_epoch,
        candidate_metrics=metrics,
        identity=_identity(),
    )


def _fake_prepared(tmp_path: Path, module):
    from training.config import MultiConfigTrainConfig

    cfg = MultiConfigTrainConfig(
        array_size="8x8",
        model_size="lite",
        dataset_root=tmp_path / "dataset",
        manifest_dir=tmp_path / "dataset" / "manifests",
        run_root=tmp_path / "runs",
    )
    cfg.run_dir.mkdir(parents=True)
    state = SimpleNamespace(
        completed_epochs=7,
        best_val_db_rmse=10.0,
        history=({"epoch": 7, "val_db_rmse": 10.0},),
    )
    return cfg, SimpleNamespace(
        cfg=cfg,
        selection_identity=_identity(),
        trainer_state=state,
    )


def test_existing_cfg_command_performs_validation_but_no_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    cfg, prepared = _fake_prepared(tmp_path, module)
    selection = _selection_payload(module)
    module.freeze_cfg_selection(cfg.run_dir / "cfg_selection.json", selection)
    monkeypatch.setattr(module, "_prepare_evaluation", lambda _cfg, _device: prepared)
    monkeypatch.setattr(
        module,
        "_evaluate_validation_candidate",
        lambda *_args, **_kwargs: pytest.fail("existing selection must not generate"),
    )

    result = module.run_cfg_selection(cfg, torch.device("cpu"), tmp_path / "results")

    assert result["status"] == "validated_existing"
    assert result["selected_scale"] == 1.5


def test_new_cfg_command_scans_fixed_grid_once_and_freezes_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    cfg, prepared = _fake_prepared(tmp_path, module)
    calls = []
    rmse = {1.0: 10.0, 1.5: 8.0, 2.0: 9.0, 2.5: 11.0}

    def fake_candidate(_prepared, scale):
        calls.append(scale)
        return {
            "n_samples": 640,
            "n_valid_pixels": 640 * 121,
            "n_ssim_windows": 640,
            "db_rmse": rmse[scale],
            "db_mae": 5.0,
            "mse": 0.01,
            "nmse": 0.02,
            "psnr": 20.0,
            "ssim": 0.8,
            "raw_fraction_below_zero": 0.0,
            "raw_fraction_above_one": 0.0,
        }

    monkeypatch.setattr(module, "_prepare_evaluation", lambda _cfg, _device: prepared)
    monkeypatch.setattr(module, "_evaluate_validation_candidate", fake_candidate)

    result = module.run_cfg_selection(cfg, torch.device("cpu"), tmp_path / "results")

    assert calls == [1.0, 1.5, 2.0, 2.5]
    assert result["status"] == "selected"
    assert result["selected_scale"] == 1.5
    assert (cfg.run_dir / "cfg_selection.json").is_file()


def test_test_command_refuses_existing_final_directory_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    cfg, _prepared = _fake_prepared(tmp_path, module)
    final = tmp_path / "results" / "8x8" / "lite"
    final.mkdir(parents=True)
    monkeypatch.setattr(
        module,
        "_prepare_evaluation",
        lambda *_args: pytest.fail("completed/existing result must be checked first"),
    )

    with pytest.raises(module.EvaluationContractError, match="cannot be rerun"):
        module.run_test_evaluation(cfg, torch.device("cpu"), tmp_path / "results")


def test_full_test_artifact_builder_writes_1280_predictions_then_runtime_and_visuals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    cfg, prepared_base = _fake_prepared(tmp_path, module)
    angles = (-28.0, -21.0, -14.0, -7.0, 0.0, 7.0, 14.0, 21.0)

    class Record:
        def __init__(self, scene: int, beam: int, angle: float) -> None:
            self.scene_id = f"u{scene}"
            self.array_name = "8x8"
            self.beam_id = beam
            self.steering_deg = angle
            self.sample_key = f"u{scene}|8x8|beam{beam:02d}"

        def to_dict(self):
            return {
                "sample_key": self.sample_key,
                "scene_id": self.scene_id,
                "array_name": self.array_name,
                "beam_id": self.beam_id,
                "steering_deg": self.steering_deg,
            }

    records = [
        Record(scene, beam, angle)
        for scene in range(160)
        for beam, angle in enumerate(angles)
    ]
    dataset = SimpleNamespace(records=records)
    model = torch.nn.Conv2d(1, 1, 1)
    checkpoint = cfg.run_dir / "best.pt"
    checkpoint.write_bytes(b"best")
    prepared = SimpleNamespace(
        **prepared_base.__dict__,
        context=SimpleNamespace(test_dataset=dataset),
        model=model,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint,
    )
    selection_path = cfg.run_dir / "cfg_selection.json"
    selection = _selection_payload(module)
    module.freeze_cfg_selection(selection_path, selection)

    class Loader:
        def __len__(self):
            return 1280

        def __iter__(self):
            for record in records:
                yield {"metadata": [record.to_dict()]}

    class FakeBeamMetrics:
        def __init__(self, _beam_angles):
            self.updates = 0

        def update(self, _prediction, _target, _mask, _metadata):
            self.updates += 1

        def compute_overall(self):
            return _overall()

        def compute_rows(self):
            return [_metric_row(index, angle) for index, angle in enumerate(angles)]

    stage = {"predictions": 0, "runtime": False, "visuals": 0}
    tensor = torch.full((1, 1, 11, 11), 0.5)
    mask = torch.ones_like(tensor, dtype=torch.bool)
    condition = torch.zeros(1, 3, 11, 11)
    monkeypatch.setattr(module, "_evaluation_loader", lambda *_args: Loader())
    monkeypatch.setattr(
        module,
        "_batch_prediction",
        lambda *_args, **_kwargs: (condition, tensor, mask, tensor),
    )
    monkeypatch.setattr(module, "PerBeamMetricAccumulators", FakeBeamMetrics)

    def fake_save(path, **_kwargs):
        assert stage["runtime"] is False
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"npz")
        stage["predictions"] += 1

    def fake_runtime(**_kwargs):
        assert stage["predictions"] == 1280
        stage["runtime"] = True
        return {"schema_version": 1, "measured_calls": 100}

    def fake_render(path, **_kwargs):
        assert stage["runtime"] is True
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"png")
        stage["visuals"] += 1

    monkeypatch.setattr(module, "save_prediction_npz", fake_save)
    monkeypatch.setattr(module, "benchmark_generation", fake_runtime)
    monkeypatch.setattr(module, "render_comparison", fake_render)
    monkeypatch.setattr(module, "render_error_map", fake_render)
    final = tmp_path / "results" / "8x8" / "lite"

    module.atomic_result_transaction(
        final,
        lambda staging: module._write_test_transaction(
            staging,
            prepared,
            selection,
            selection_path,
            module.sha256_file(selection_path),
        ),
    )

    assert stage == {"predictions": 1280, "runtime": True, "visuals": 24}
    assert len(list((final / "predictions").glob("*.npz"))) == 1280
    assert module.validate_run_manifest_artifacts(final)["n_samples"] == 1280


def _summary_record(module, array_size: str, model_size: str):
    summary = {column: "" for column in module.SUMMARY_COLUMNS}
    summary.update(
        {
            "array_size": array_size,
            "model_size": model_size,
            "status": "complete",
            "selected_epoch": 7,
            "selected_cfg_scale": 1.5,
            "db_rmse": 10.0,
            "db_mae": 5.0,
            "ssim": 0.8,
        }
    )
    identity = _identity()
    identity["manifest_sha256"] = {
        "8x8": "a" * 64,
        "16x16": "b" * 64,
        "32x32": "c" * 64,
    }[array_size]
    angle_rows = []
    for beam, angle in enumerate(module.COMMON_ANGLES_DEG):
        row = {column: "" for column in module.ANGLE_SUMMARY_COLUMNS}
        row.update(
            {
                "array_size": array_size,
                "model_size": model_size,
                "status": "complete",
                **_metric_row(beam, angle),
                "hardware_gate_sha256": "",
            }
        )
        angle_rows.append(row)
    return {
        "summary_row": summary,
        "angle_rows": angle_rows,
        "identity": identity,
        "cfg": SimpleNamespace(),
    }


def test_summary_accepts_exactly_six_complete_pairs_and_joins_by_angle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    run_root = tmp_path / "runs"
    results_root = tmp_path / "results"
    records = {}
    for array in module.ARRAY_NAMES:
        for size in module.MODEL_SIZES:
            (results_root / array / size).mkdir(parents=True)
            records[(array, size)] = _summary_record(module, array, size)
    monkeypatch.setattr(
        module,
        "_load_completed_pair",
        lambda _run, _results, array, size: records[(array, size)],
    )

    result = module.summarize_benchmark(run_root, results_root)

    assert result["status"] == "complete"
    with (results_root / "benchmark_summary.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        assert len(list(csv.DictReader(source))) == 6
    with (results_root / "metrics_per_angle_comparison.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        angle_rows = list(csv.DictReader(source))
    assert len(angle_rows) == 48
    assert [float(row["angle_deg"]) for row in angle_rows[:8]] == list(
        module.COMMON_ANGLES_DEG
    )


def test_summary_accepts_three_lite_plus_one_global_gate_and_labels_blocked_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    run_root = tmp_path / "runs"
    results_root = tmp_path / "results"
    records = {}
    for array in module.ARRAY_NAMES:
        (results_root / array / "lite").mkdir(parents=True)
        records[(array, "lite")] = _summary_record(module, array, "lite")
    gate = run_root / "_hardware" / "large_hardware_gate.json"
    gate.parent.mkdir(parents=True)
    gate.write_bytes(b"gate")
    monkeypatch.setattr(
        module,
        "_load_completed_pair",
        lambda _run, _results, array, size: records[(array, size)],
    )
    monkeypatch.setattr(
        module,
        "_validate_global_large_gate",
        lambda *_args: "f" * 64,
    )

    result = module.summarize_benchmark(run_root, results_root)

    assert result["status"] == "large_hardware_blocked"
    assert result["completed_pairs"] == 3
    with (results_root / "benchmark_summary.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        rows = list(csv.DictReader(source))
    blocked = [row for row in rows if row["status"] == "hardware_blocked"]
    assert len(blocked) == 3
    assert {row["hardware_gate_sha256"] for row in blocked} == {"f" * 64}
    assert all(row["db_rmse"] == "" for row in blocked)


@pytest.mark.parametrize(
    ("pairs", "gate_exists"),
    [
        ({("8x8", "lite")}, False),
        (
            {
                ("8x8", "lite"),
                ("16x16", "lite"),
                ("32x32", "lite"),
                ("8x8", "large"),
            },
            True,
        ),
        (
            {
                (array, size)
                for array in ("8x8", "16x16", "32x32")
                for size in ("lite", "large")
            },
            True,
        ),
    ],
)
def test_summary_rejects_partial_or_contradictory_terminal_states(
    pairs: set[tuple[str, str]],
    gate_exists: bool,
) -> None:
    module = _module()

    with pytest.raises(module.EvaluationContractError, match="terminal state"):
        module._terminal_state(pairs, gate_exists=gate_exists)


def test_summary_rejects_shared_source_identity_mismatch() -> None:
    module = _module()
    first = _summary_record(module, "8x8", "lite")
    second = _summary_record(module, "16x16", "lite")
    second["identity"]["split_sha256"] = "0" * 64

    with pytest.raises(module.EvaluationContractError, match="split_sha256"):
        module._validate_common_completed_contract([first, second])
