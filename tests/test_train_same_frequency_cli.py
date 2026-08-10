from __future__ import annotations

from pathlib import Path

import pytest

from experiments.multiconfig_manifest import ManifestRecord, write_manifest_jsonl
from train_same_frequency import build_parser
from training.same_frequency_trainer import (
    SameFrequencyTrainerContractError,
    infer_manifest_selection,
)


def _record(*, beam_id: int = 4, frequency_hz: int = 6_700_000_000) -> ManifestRecord:
    return ManifestRecord(
        sample_key="u1|samefreq",
        split="train",
        scene_id="u1",
        array_name="8x8",
        array_rows=8,
        array_cols=8,
        frequency_hz=frequency_hz,
        config_id="freq_6.7GHz_64TR_8beams_pattern_tr38901",
        beam_id=beam_id,
        steering_deg=0.0,
        height_path="raw/Dataset/height_maps/u1/u1_height_matrix.npy",
        beam_map_path="raw/Dataset/beam_maps/config/u0/beam_04_angle_0.0_matrix.npy",
        radiomap_path="raw/Dataset/radiomaps/config_beam04/u1_labeled_radiomap.npy",
    )


def test_infer_manifest_selection_returns_real_beam_and_config(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_manifest_jsonl(manifest, (_record(),))

    assert infer_manifest_selection(manifest, "8x8") == (
        4,
        "freq_6.7GHz_64TR_8beams_pattern_tr38901",
    )


@pytest.mark.parametrize(
    "record",
    [_record(beam_id=0), _record(frequency_hz=4_900_000_000)],
)
def test_infer_manifest_selection_rejects_wrong_protocol(
    tmp_path: Path,
    record: ManifestRecord,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_manifest_jsonl(manifest, (record,))

    with pytest.raises(SameFrequencyTrainerContractError):
        infer_manifest_selection(manifest, "8x8")


def test_cli_exposes_array_size_and_preflight_controls() -> None:
    args = build_parser().parse_args(
        [
            "--dataset-root", "dataset",
            "--manifest-path", "manifest.jsonl",
            "--height-stats-path", "height.json",
            "--run-root", "run",
            "--array-size", "32x32",
            "--device", "cpu",
            "--resume", "none",
            "--preflight-only",
        ]
    )

    assert args.array_size == "32x32"
    assert args.preflight_only is True
