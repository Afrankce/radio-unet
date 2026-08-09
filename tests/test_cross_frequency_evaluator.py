from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from experiments.multiconfig_manifest import ManifestRecord
from training.cross_frequency_config import CrossFrequencyTrainConfig


TRAIN_FREQUENCY = 4_900_000_000
TEST_FREQUENCY = 6_700_000_000


class ZeroVelocityModel(torch.nn.Module):
    def embed_model(self, condition):
        return condition

    def forward_with_cfg(self, *, image, x, step, embedding, cfg_scale):
        return torch.zeros_like(x)


class TinyEvaluationDataset(Dataset):
    def __init__(self, count: int, split: str, frequency_hz: int, beam_id: int) -> None:
        self.records = tuple(
            ManifestRecord(
                sample_key=f"u{index + 1}|{split}",
                split=split,
                scene_id=f"u{index + 1}",
                array_name="8x8",
                array_rows=8,
                array_cols=8,
                frequency_hz=frequency_hz,
                config_id=f"freq_{frequency_hz}",
                beam_id=beam_id,
                steering_deg=0.0,
                height_path="height.npy",
                beam_map_path="beam.npy",
                radiomap_path="radiomap.npy",
            )
            for index in range(count)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        target = torch.full((1, 11, 11), 0.5)
        return {
            "condition": torch.zeros(3, 11, 11),
            "target": target,
            "valid_mask": torch.ones(1, 11, 11, dtype=torch.bool),
            "metadata": record.to_dict(),
        }


def _prepared(tmp_path: Path):
    cfg = CrossFrequencyTrainConfig(
        dataset_root=tmp_path / "dataset",
        manifest_path=tmp_path / "manifest.jsonl",
        height_stats_path=tmp_path / "height.json",
        run_root=tmp_path / "run",
    )
    cfg.run_dir.mkdir(parents=True)
    checkpoint = cfg.run_dir / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    context = SimpleNamespace(
        val_dataset=TinyEvaluationDataset(80, "val", TRAIN_FREQUENCY, 0),
        test_dataset=TinyEvaluationDataset(160, "test", TEST_FREQUENCY, 4),
    )
    trainer_state = SimpleNamespace(
        completed_epochs=3,
        best_val_db_rmse=1.0,
        history=[{"epoch": 3, "val_db_rmse": 1.0}],
    )
    identity = SimpleNamespace(
        config_sha256="1" * 64,
        manifest_sha256="2" * 64,
        split_sha256="3" * 64,
        schema_sha256="4" * 64,
        archive_sha256="5" * 64,
        dataset_revision="6" * 40,
        radioflow_upstream_base="7" * 40,
        git_commit="8" * 40,
    )
    return SimpleNamespace(
        cfg=cfg,
        context=context,
        model=ZeroVelocityModel(),
        device=torch.device("cpu"),
        checkpoint_path=checkpoint,
        checkpoint_identity=identity,
        selection_identity={
            "checkpoint_sha256": "9" * 64,
            "config_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "split_sha256": "3" * 64,
            "schema_sha256": "4" * 64,
            "archive_sha256": "5" * 64,
            "dataset_revision": "6" * 40,
            "radioflow_upstream_base": "7" * 40,
            "git_commit": "8" * 40,
        },
        trainer_state=trainer_state,
    )


def test_cfg_selection_uses_exactly_80_validation_samples(tmp_path: Path, monkeypatch) -> None:
    import evaluation.cross_frequency_evaluator as evaluator

    prepared = _prepared(tmp_path)
    monkeypatch.setattr(evaluator, "_prepare_evaluation", lambda cfg, device: prepared)
    calls: list[float] = []

    def fake_candidate(prepared, scale):
        assert len(prepared.context.val_dataset) == 80
        calls.append(scale)
        return {
            "n_samples": 80,
            "n_valid_pixels": 80 * 121,
            "db_rmse": 1.0,
        }

    monkeypatch.setattr(evaluator, "_evaluate_validation_candidate", fake_candidate)

    result = evaluator.run_cfg_selection(
        prepared.cfg,
        torch.device("cpu"),
        tmp_path / "results",
    )

    assert result["status"] == "selected"
    assert result["selected_epoch"] == 3
    assert calls == [1.0, 1.5, 2.0, 2.5]
    payload = json.loads((prepared.cfg.run_dir / "cfg_selection.json").read_text())
    assert payload["selected_scale"] == 1.0
    assert set(payload["candidate_metrics"]) == {"1.0", "1.5", "2.0", "2.5"}
    assert all(value["n_samples"] == 80 for value in payload["candidate_metrics"].values())


def test_test_evaluation_writes_grouped_metrics_and_is_one_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import evaluation.cross_frequency_evaluator as evaluator

    prepared = _prepared(tmp_path)
    monkeypatch.setattr(evaluator, "_prepare_evaluation", lambda cfg, device: prepared)
    monkeypatch.setattr(
        evaluator,
        "_evaluate_validation_candidate",
        lambda prepared, scale: {
            "n_samples": 80,
            "n_valid_pixels": 80 * 121,
            "db_rmse": 1.0,
        },
    )
    evaluator.run_cfg_selection(prepared.cfg, torch.device("cpu"), tmp_path / "results")
    monkeypatch.setattr(
        evaluator,
        "benchmark_generation",
        lambda **kwargs: {"latency_ms_p50": 1.0, "latency_ms_p95": 2.0},
    )
    monkeypatch.setattr(
        evaluator,
        "render_comparison",
        lambda path, **kwargs: Path(path).parent.mkdir(parents=True, exist_ok=True)
        or Path(path).write_bytes(b"png"),
    )
    monkeypatch.setattr(
        evaluator,
        "render_error_map",
        lambda path, **kwargs: Path(path).parent.mkdir(parents=True, exist_ok=True)
        or Path(path).write_bytes(b"png"),
    )

    result = evaluator.run_test_evaluation(
        prepared.cfg,
        torch.device("cpu"),
        tmp_path / "results",
    )

    result_dir = Path(result["result_dir"])
    assert result["n_samples"] == 160
    assert (result_dir / "metrics_test.json").is_file()
    assert (result_dir / "metrics_per_frequency.csv").is_file()
    assert (result_dir / "run_manifest.json").is_file()
    metrics = json.loads((result_dir / "metrics_test.json").read_text())
    assert metrics["test_frequency_hz"] == TEST_FREQUENCY
    assert metrics["n_samples"] == 160
    assert (result_dir / "predictions").exists()

    with pytest.raises(evaluator.CrossFrequencyEvaluationError):
        evaluator.run_test_evaluation(
            prepared.cfg,
            torch.device("cpu"),
            tmp_path / "results",
        )
