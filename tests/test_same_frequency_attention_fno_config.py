from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.same_frequency_attention_fno_config import (
    ATTENTION_FNO_MODEL_SIZE,
    AttentionFNOConfigError,
    AttentionFNOTrainConfig,
)
from training.same_frequency_config import SameFrequencyTrainConfig


def _base(
    tmp_path: Path,
    *,
    array_size: str = "8x8",
    beam_id: int = 4,
    model_size: str = "lite",
) -> SameFrequencyTrainConfig:
    return SameFrequencyTrainConfig(
        dataset_root=tmp_path / "dataset",
        manifest_path=tmp_path / "manifest.jsonl",
        height_stats_path=tmp_path / "height.json",
        run_root=tmp_path / "run",
        array_size=array_size,
        beam_id=beam_id,
        model_size=model_size,
    )


def test_attention_fno_config_locks_protocol_and_approved_architecture(
    tmp_path: Path,
) -> None:
    cfg = AttentionFNOTrainConfig(_base(tmp_path, array_size="16x16", beam_id=8))
    payload = cfg.scientific_payload()

    assert cfg.model_size == ATTENTION_FNO_MODEL_SIZE == "attention_fno_lite"
    assert cfg.train_samples == 560
    assert cfg.val_samples == 80
    assert cfg.test_samples == 160
    assert cfg.max_epochs == 1000
    assert cfg.early_stopping_patience == 20
    assert cfg.micro_batch_size == 2
    assert cfg.accumulation_steps == 28
    assert cfg.cfg_candidates == (1.0,)
    assert payload["experiment"] == "same_frequency_6.7_single_beam_attention_fno"
    assert payload["backbone"] == "attention_conditioned_full_resolution_fno2d"
    assert payload["condition_encoder"] == "BasicUNetEncoder_lite"
    assert payload["condition_aggregation"] == "project_resize_sum"
    assert payload["attention"] == "RadioFlow_CA_SA_each_block"
    assert payload["time_conditioning"] == "sinusoidal_mlp_each_block"
    assert payload["fno_width"] == 40
    assert payload["fno_modes"] == [12, 12]
    assert payload["fno_padding"] == 9
    assert payload["fno_layers"] == 4
    assert payload["fno_input_order"] == [
        "x_t",
        "tx_mask",
        "height",
        "beam_map",
        "grid_x",
        "grid_y",
    ]
    assert payload["tensor_parameter_count"] == 3_487_273
    assert payload["real_scalar_parameter_count"] == 5_330_473


def test_attention_fno_config_round_trips_and_keeps_path_out_of_hash(
    tmp_path: Path,
) -> None:
    cfg = AttentionFNOTrainConfig(_base(tmp_path, array_size="32x32", beam_id=32))

    restored = AttentionFNOTrainConfig.from_json(json.dumps(cfg.to_record()))
    moved = cfg.with_run_root(tmp_path / "moved")

    assert restored == cfg
    assert restored.config_sha256 == cfg.config_sha256
    assert moved.run_root == tmp_path / "moved"
    assert moved.config_sha256 == cfg.config_sha256
    assert cfg.config_sha256 != cfg.base.config_sha256


def test_attention_fno_config_rejects_architecture_drift(tmp_path: Path) -> None:
    with pytest.raises(AttentionFNOConfigError, match="base model_size"):
        AttentionFNOTrainConfig(_base(tmp_path, model_size="large"))
    with pytest.raises(AttentionFNOConfigError, match="width"):
        AttentionFNOTrainConfig(_base(tmp_path), fno_width=48)
    with pytest.raises(AttentionFNOConfigError, match="modes"):
        AttentionFNOTrainConfig(_base(tmp_path), fno_modes=(16, 16))
    with pytest.raises(AttentionFNOConfigError, match="CFG candidates"):
        AttentionFNOTrainConfig(_base(tmp_path), cfg_candidates=(1.0, 1.5))

