from __future__ import annotations

from pathlib import Path

import pytest

from training.same_frequency_config import (
    SameFrequencyTrainConfig,
    SameFrequencyTrainConfigError,
)


def _config(tmp_path: Path, array_size: str = "8x8") -> SameFrequencyTrainConfig:
    return SameFrequencyTrainConfig(
        dataset_root=tmp_path / "dataset",
        manifest_path=tmp_path / "manifest.jsonl",
        height_stats_path=tmp_path / "height_stats.json",
        run_root=tmp_path / "run",
        array_size=array_size,
        beam_id=4 if array_size == "8x8" else 8,
    )


@pytest.mark.parametrize(
    "array_size,rows,cols,tx_elements",
    [("8x8", 8, 8, 64), ("16x16", 16, 16, 256), ("32x32", 32, 32, 1024)],
)
def test_same_frequency_config_has_shared_training_controls(
    tmp_path: Path,
    array_size: str,
    rows: int,
    cols: int,
    tx_elements: int,
) -> None:
    cfg = _config(tmp_path, array_size)

    assert cfg.array_rows == rows
    assert cfg.array_cols == cols
    assert cfg.tx_elements == tx_elements
    assert cfg.train_frequency_hz == 6_700_000_000
    assert cfg.val_frequency_hz == 6_700_000_000
    assert cfg.test_frequency_hz == 6_700_000_000
    assert cfg.train_samples == 560
    assert cfg.val_samples == 80
    assert cfg.test_samples == 160
    assert cfg.micro_batch_size == 2
    assert cfg.accumulation_steps == 28
    assert cfg.optimizer_steps_per_epoch == 10
    assert cfg.scientific_payload()["beam_id"] == cfg.beam_id


def test_same_frequency_config_round_trips_exactly(tmp_path: Path) -> None:
    cfg = _config(tmp_path, "16x16")
    restored = SameFrequencyTrainConfig.from_json(
        __import__("json").dumps(cfg.to_record())
    )
    assert restored == cfg
    assert restored.config_sha256 == cfg.config_sha256


def test_same_frequency_config_rejects_non_67ghz(tmp_path: Path) -> None:
    with pytest.raises(SameFrequencyTrainConfigError):
        SameFrequencyTrainConfig(
            dataset_root=tmp_path / "dataset",
            manifest_path=tmp_path / "manifest.jsonl",
            height_stats_path=tmp_path / "height_stats.json",
            run_root=tmp_path / "run",
            train_frequency_hz=4_900_000_000,
        )
