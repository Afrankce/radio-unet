from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


def _module():
    from training import sparse_config

    return sparse_config


def test_sparse_config_constants_and_variant_channel_mapping() -> None:
    module = _module()

    assert module.SPARSE_EXPERIMENT == "sparse_same_frequency_6.7_single_beam"
    assert module.ARRAY_SIZES == ("8x8", "16x16", "32x32")
    assert module.VARIANTS == ("no_beam_masked", "beam_masked")
    assert module.MODEL_SIZE == "lite"
    assert module.FREQUENCY_HZ == 6_700_000_000
    assert module.STEERING_DEG == 0.0
    assert module.SCENE_COUNTS == {"train": 560, "val": 80, "test": 160}
    assert module.OUTPUT_SIZE == (256, 256)
    assert module.OBSERVATION_RATIO == 0.05
    assert module.MAX_EPOCHS == 1000
    assert module.EARLY_STOPPING_PATIENCE == 20
    assert module.CONDITION_CHANNELS == {
        "no_beam_masked": 4,
        "beam_masked": 5,
    }
    assert module.variant_to_condition_channels("no_beam_masked") == 4
    assert module.variant_to_condition_channels("beam_masked") == 5
    with pytest.raises(module.SparseConfigError, match="variant"):
        module.variant_to_condition_channels("other")


def test_sparse_config_is_frozen_and_round_trips_with_canonical_hashes(
    tmp_path: Path,
) -> None:
    module = _module()
    cfg = module.SparseSameFrequencyTrainConfig(
        dataset_root=tmp_path / "dataset",
        manifest_path=tmp_path / "manifests" / "manifest.jsonl",
        height_stats_path=tmp_path / "manifests" / "height_stats.json",
        run_root=tmp_path / "runs",
        array_size="16x16",
        variant="beam_masked",
    )

    with pytest.raises(FrozenInstanceError):
        cfg.variant = "no_beam_masked"  # type: ignore[misc]

    record = cfg.to_record(
        manifest_sha256="1" * 64,
        height_stats_sha256="2" * 64,
    )
    restored = module.SparseSameFrequencyTrainConfig.from_json(json.dumps(record))

    assert restored == cfg
    assert cfg.condition_channels == 5
    assert record["config_sha256"] == cfg.config_sha256
    assert record["manifest_sha256"] == "1" * 64
    assert record["height_stats_sha256"] == "2" * 64
    assert record["split_id"] == module.FIXED_SPLIT_ID
    assert record["formal_run_variant"] == "beam_masked"
    assert restored.canonical_payload() == cfg.canonical_payload()


def test_sparse_config_rejects_unknown_or_mutated_fields(tmp_path: Path) -> None:
    module = _module()
    cfg = module.SparseSameFrequencyTrainConfig(
        dataset_root=tmp_path,
        manifest_path=tmp_path / "manifest.jsonl",
        height_stats_path=tmp_path / "height_stats.json",
        run_root=tmp_path / "runs",
        array_size="8x8",
        variant="no_beam_masked",
    )
    record = cfg.to_record(
        manifest_sha256="3" * 64,
        height_stats_sha256="4" * 64,
    )

    with pytest.raises(module.SparseConfigError, match="unknown"):
        module.SparseSameFrequencyTrainConfig.from_json(
            json.dumps(dict(record, surprise=True))
        )
    with pytest.raises(module.SparseConfigError, match="variant"):
        module.SparseSameFrequencyTrainConfig.from_json(
            json.dumps(dict(record, variant="beam_masked", config_sha256="0" * 64))
        )


def test_sparse_config_checkpoint_identity_payload_is_locked(tmp_path: Path) -> None:
    module = _module()
    cfg = module.SparseSameFrequencyTrainConfig(
        dataset_root=tmp_path,
        manifest_path=tmp_path / "manifest.jsonl",
        height_stats_path=tmp_path / "height_stats.json",
        run_root=tmp_path / "runs",
        array_size="32x32",
        variant="beam_masked",
    )

    payload = cfg.checkpoint_identity_payload(
        manifest_sha256="5" * 64,
        height_stats_sha256="6" * 64,
        checkpoint_sha256="7" * 64,
        archive_sha256="8" * 64,
        dataset_revision="9" * 40,
        radioflow_upstream_base="a" * 40,
        git_commit="b" * 40,
        parameter_count=123,
    )

    assert payload["experiment"] == module.SPARSE_EXPERIMENT
    assert payload["array_size"] == "32x32"
    assert payload["variant"] == "beam_masked"
    assert payload["condition_channels"] == 5
    assert payload["frequency_hz"] == module.FREQUENCY_HZ
    assert payload["steering_deg"] == module.STEERING_DEG
    assert payload["manifest_sha256"] == "5" * 64
    assert payload["height_stats_sha256"] == "6" * 64
    assert payload["config_sha256"] == cfg.config_sha256
    assert payload["split_id"] == module.FIXED_SPLIT_ID
