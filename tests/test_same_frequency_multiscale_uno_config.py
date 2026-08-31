from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.same_frequency_config import SameFrequencyTrainConfig
from training.same_frequency_multiscale_uno_config import (
    MULTISCALE_UNO_MODEL_SIZE,
    MultiscaleUNOConfigError,
    MultiscaleUNOTrainConfig,
)


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


def test_multiscale_uno_config_locks_protocol_and_scientific_payload(
    tmp_path: Path,
) -> None:
    """Any architecture or protocol identity drift must be observable here."""

    cfg = MultiscaleUNOTrainConfig(_base(tmp_path, array_size="16x16", beam_id=8))
    payload = cfg.scientific_payload()

    expected = {
        "experiment": "same_frequency_6.7_single_beam_attention_multiscale_uno",
        "model_size": "attention_multiscale_uno_lite",
        "backbone": "attention_conditioned_multiscale_uno2d",
        "state_channels": [32, 64, 128, 256, 256],
        "operator_width": 24,
        "operator_modes": [12, 12, 8, 4, 4],
        "operator_padding": [9, 5, 3, 2, 1],
        "operator_stages": 9,
        "condition_injection": "native_scale_CA_SA_encoder_decoder",
        "downsample": "avgpool2_plus_1x1",
        "upsample": "bilinear_concat_1x1",
        "state_skip_connections": True,
        "tensor_parameter_count": 3_059_355,
        "real_scalar_parameter_count": 3_925_659,
    }

    assert cfg.base.model_size == "lite"
    assert cfg.model_size == MULTISCALE_UNO_MODEL_SIZE == "attention_multiscale_uno_lite"
    assert {key: payload[key] for key in expected} == expected
    assert payload["condition_encoder"] == "BasicUNetEncoder_lite"
    assert payload["condition_encoder_features"] == [32, 32, 64, 128, 256, 32]
    assert payload["attention_modules"] == 9
    assert payload["cfg_drop_prob"] == 0.25
    assert payload["cfg_dropout_scope"] == "raw_and_encoded_condition"
    assert payload["cfg_candidates"] == [1.0]
    assert cfg.train_samples == 560
    assert cfg.val_samples == 80
    assert cfg.test_samples == 160
    assert cfg.max_epochs == 1000
    assert cfg.early_stopping_patience == 20


def test_multiscale_uno_config_round_trips_and_keeps_paths_out_of_hash(
    tmp_path: Path,
) -> None:
    cfg = MultiscaleUNOTrainConfig(_base(tmp_path, array_size="32x32", beam_id=32))

    restored = MultiscaleUNOTrainConfig.from_json(json.dumps(cfg.to_record()))
    moved = cfg.with_run_root(tmp_path / "moved")

    assert restored == cfg
    assert restored.config_sha256 == cfg.config_sha256
    assert moved.run_root == tmp_path / "moved"
    assert moved.config_sha256 == cfg.config_sha256
    assert cfg.config_sha256 != cfg.base.config_sha256


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("state_channels", (32, 64, 128, 256, 128), "state_channels"),
        ("operator_width", 32, "operator_width"),
        ("operator_modes", (12, 12, 8, 4, 2), "operator_modes"),
        ("operator_padding", (9, 5, 3, 2, 0), "operator_padding"),
        ("operator_stages", 8, "operator_stages"),
        ("condition_encoder_features", (32, 32, 64, 128, 128, 32), "condition_encoder_features"),
        ("attention_modules", 8, "attention_modules"),
        ("condition_injection", "resized_sum", "condition_injection"),
        ("downsample", "strided_conv", "downsample"),
        ("upsample", "transposed_conv", "upsample"),
        ("state_skip_connections", False, "state_skip_connections"),
        ("cfg_drop_prob", 0.0, "cfg_drop_prob"),
        ("cfg_candidates", (1.0, 1.5), "CFG candidates"),
        ("tensor_parameter_count", 3_059_354, "tensor_parameter_count"),
        ("real_scalar_parameter_count", 3_925_658, "real_scalar_parameter_count"),
    ],
)
def test_multiscale_uno_config_rejects_architecture_and_cfg_drift(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    with pytest.raises(MultiscaleUNOConfigError, match=match):
        MultiscaleUNOTrainConfig(_base(tmp_path), **{field: value})

    with pytest.raises(MultiscaleUNOConfigError, match="base model_size"):
        MultiscaleUNOTrainConfig(_base(tmp_path, model_size="large"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operator_width", 32),
        ("cfg_drop_prob", 0.0),
    ],
)
def test_multiscale_uno_config_rejects_serialized_architecture_and_cfg_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    record = MultiscaleUNOTrainConfig(_base(tmp_path)).to_record()
    record[field] = value

    with pytest.raises(MultiscaleUNOConfigError, match=f"{field} mismatch"):
        MultiscaleUNOTrainConfig.from_json(json.dumps(record))
