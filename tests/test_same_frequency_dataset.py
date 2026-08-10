from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from data_loaders.same_frequency import (
    SameFrequencyDatasetError,
    SameFrequencyRadiomapDataset,
)
from experiments.multiconfig_manifest import ManifestRecord, write_manifest_jsonl


FREQUENCY = 6_700_000_000
EXPECTED_COUNTS = {"train": 1, "val": 1, "test": 1}


def _write_sample(root: Path, scene_id: str, config_id: str, beam_id: int) -> tuple[str, str, str]:
    height_rel = f"raw/Dataset/height_maps/{scene_id}/{scene_id}_height_matrix.npy"
    beam_rel = (
        f"raw/Dataset/beam_maps/{config_id}/u0/"
        f"beam_{beam_id:02d}_angle_0.0_matrix.npy"
    )
    target_rel = (
        f"raw/Dataset/radiomaps/{config_id}_beam{beam_id:02d}/"
        f"{scene_id}_labeled_radiomap.npy"
    )
    for relative in (height_rel, beam_rel, target_rel):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    np.save(root / height_rel, np.full((256, 256), 20.0, dtype=np.float32))
    np.save(root / beam_rel, np.full((128, 128), -100.0, dtype=np.float64))
    target = np.full((128, 128), -50.0, dtype=np.float32)
    target[:2, :2] = -300.0
    target[-2:, -2:] = 1000.0
    np.save(root / target_rel, target)
    return height_rel, beam_rel, target_rel


def _records(root: Path) -> tuple[ManifestRecord, ...]:
    records: list[ManifestRecord] = []
    for split, scene_id in (("train", "u1"), ("val", "u2"), ("test", "u3")):
        height, beam, target = _write_sample(
            root,
            scene_id,
            "freq_6.7GHz_64TR_8beams_pattern_tr38901",
            4,
        )
        records.append(
            ManifestRecord(
                sample_key=f"{scene_id}|samefreq",
                split=split,
                scene_id=scene_id,
                array_name="8x8",
                array_rows=8,
                array_cols=8,
                frequency_hz=FREQUENCY,
                config_id="freq_6.7GHz_64TR_8beams_pattern_tr38901",
                beam_id=4,
                steering_deg=0.0,
                height_path=height,
                beam_map_path=beam,
                radiomap_path=target,
            )
        )
    return tuple(records)


def _dataset(
    root: Path,
    records: tuple[ManifestRecord, ...],
    split: str,
    *,
    expected_beam_id: int | None = 4,
) -> SameFrequencyRadiomapDataset:
    manifest = root / "manifest.jsonl"
    write_manifest_jsonl(manifest, records)
    return SameFrequencyRadiomapDataset(
        dataset_root=root,
        manifest_path=manifest,
        split=split,
        array_size="8x8",
        height_max=40.0,
        expected_beam_id=expected_beam_id,
        expected_counts=EXPECTED_COUNTS,
    )


def test_same_frequency_dataset_decodes_three_channel_sample(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, _records(tmp_path), "train")
    sample = dataset[0]

    assert len(dataset) == 1
    assert tuple(sample["condition"].shape) == (3, 256, 256)
    assert tuple(sample["target"].shape) == (1, 256, 256)
    assert sample["condition"].dtype == torch.float32
    assert sample["target"].dtype == torch.float32
    assert sample["valid_mask"].dtype == torch.bool
    assert bool(sample["valid_mask"].any())
    assert sample["metadata"]["frequency_hz"] == FREQUENCY
    assert sample["metadata"]["beam_id"] == 4
    assert sample["metadata"]["steering_deg"] == 0.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("array_rows", 4),
        ("array_cols", 4),
        ("steering_deg", 7.0),
        ("frequency_hz", 4_900_000_000),
        ("beam_id", 0),
    ],
)
def test_same_frequency_dataset_rejects_protocol_mismatches(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    records = list(_records(tmp_path))
    records[0] = replace(records[0], **{field: value})

    with pytest.raises(SameFrequencyDatasetError):
        _dataset(tmp_path, tuple(records), "train")


def test_same_frequency_dataset_rejects_mixed_beams(tmp_path: Path) -> None:
    records = list(_records(tmp_path))
    records[1] = replace(records[1], beam_id=5)
    with pytest.raises(SameFrequencyDatasetError):
        _dataset(tmp_path, tuple(records), "val")
