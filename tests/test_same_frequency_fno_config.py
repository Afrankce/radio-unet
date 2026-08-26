from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.same_frequency_config import SameFrequencyTrainConfig
from training.same_frequency_fno_config import (
    PAPER_FNO_MODEL_SIZE,
    PaperFNOConfigError,
    PaperFNOTrainConfig,
)


def _base_config(
    tmp_path: Path,
    *,
    array_size: str = "8x8",
    beam_id: int = 4,
    model_size: str = "lite",
) -> SameFrequencyTrainConfig:
    return SameFrequencyTrainConfig(
        dataset_root=tmp_path / "dataset",
        manifest_path=tmp_path / "manifest.jsonl",
        height_stats_path=tmp_path / "height_stats.json",
        run_root=tmp_path / "run",
        array_size=array_size,
        beam_id=beam_id,
        model_size=model_size,
    )


def test_fno_config_keeps_protocol_and_adds_architecture_identity(
    tmp_path: Path,
) -> None:
    cfg = PaperFNOTrainConfig(
        _base_config(tmp_path, array_size="16x16", beam_id=8)
    )
    payload = cfg.scientific_payload()

    assert cfg.model_size == PAPER_FNO_MODEL_SIZE == "paper_fno_lite"
    assert cfg.condition_channels == 3
    assert cfg.train_samples == 560
    assert cfg.val_samples == 80
    assert cfg.test_samples == 160
    assert cfg.micro_batch_size == 2
    assert cfg.accumulation_steps == 28
    assert cfg.effective_batch_size == 56
    assert cfg.optimizer_steps_per_epoch == 10
    assert cfg.cfg_candidates == (1.0,)
    assert payload["experiment"] == "same_frequency_6.7_single_beam_paper_fno"
    assert payload["backbone"] == "paper_fno2d"
    assert payload["model_size"] == PAPER_FNO_MODEL_SIZE
    assert payload["fno_width"] == 40
    assert payload["fno_modes"] == [12, 12]
    assert payload["fno_padding"] == 9
    assert payload["fno_layers"] == 4
    assert payload["cfg_candidates"] == [1.0]
    assert payload["tensor_parameter_count"] == 1_855_457
    assert payload["real_scalar_parameter_count"] == 3_698_657


def test_fno_config_round_trips_exactly_and_changes_scientific_hash(
    tmp_path: Path,
) -> None:
    base = _base_config(tmp_path, array_size="32x32", beam_id=32)
    cfg = PaperFNOTrainConfig(base)

    restored = PaperFNOTrainConfig.from_json(json.dumps(cfg.to_record()))

    assert restored == cfg
    assert restored.config_sha256 == cfg.config_sha256
    assert cfg.config_sha256 != base.config_sha256


def test_fno_config_with_run_root_preserves_scientific_identity(tmp_path: Path) -> None:
    cfg = PaperFNOTrainConfig(_base_config(tmp_path))

    moved = cfg.with_run_root(tmp_path / "other-run")

    assert moved.run_root == (tmp_path / "other-run")
    assert moved.run_dir == (tmp_path / "other-run")
    assert moved.config_sha256 == cfg.config_sha256


def test_fno_config_rejects_non_lite_base_and_architecture_changes(
    tmp_path: Path,
) -> None:
    with pytest.raises(PaperFNOConfigError, match="base model_size"):
        PaperFNOTrainConfig(_base_config(tmp_path, model_size="large"))
    with pytest.raises(PaperFNOConfigError, match="width"):
        PaperFNOTrainConfig(_base_config(tmp_path), fno_width=41)
    with pytest.raises(PaperFNOConfigError, match="modes"):
        PaperFNOTrainConfig(_base_config(tmp_path), fno_modes=(16, 16))
    with pytest.raises(PaperFNOConfigError, match="CFG candidates"):
        PaperFNOTrainConfig(_base_config(tmp_path), cfg_candidates=(1.0, 1.5))

