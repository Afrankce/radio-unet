from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.multiconfig_manifest import ManifestRecord, write_manifest_jsonl
from data_loaders.cross_frequency import (
    CrossFrequencyDatasetError,
    CrossFrequencyRadiomapDataset,
)


TRAIN_FREQUENCY = 4_900_000_000
TEST_FREQUENCY = 6_700_000_000
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
    for split, scene_id, frequency_hz, config_id, beam_id in (
        ("train", "u1", TRAIN_FREQUENCY, "freq_4.9", 0),
        ("val", "u2", TRAIN_FREQUENCY, "freq_4.9", 0),
        ("test", "u3", TEST_FREQUENCY, "freq_6.7", 4),
    ):
        height, beam, target = _write_sample(root, scene_id, config_id, beam_id)
        records.append(
            ManifestRecord(
                sample_key=f"{scene_id}|crossfreq",
                split=split,
                scene_id=scene_id,
                array_name="8x8",
                array_rows=8,
                array_cols=8,
                frequency_hz=frequency_hz,
                config_id=config_id,
                beam_id=beam_id,
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
    height_max: float = 40.0,
    expected_frequency_hz: int | None = None,
) -> CrossFrequencyRadiomapDataset:
    manifest = root / "manifest.jsonl"
    write_manifest_jsonl(manifest, records)
    return CrossFrequencyRadiomapDataset(
        dataset_root=root,
        manifest_path=manifest,
        split=split,
        height_max=height_max,
        expected_frequency_hz=expected_frequency_hz,
        expected_counts=EXPECTED_COUNTS,
    )


def test_cross_frequency_dataset_decodes_three_channel_sample(tmp_path: Path) -> None:
    records = _records(tmp_path)
    dataset = _dataset(tmp_path, records, "train")

    sample = dataset[0]

    assert len(dataset) == 1
    assert tuple(sample["condition"].shape) == (3, 256, 256)
    assert tuple(sample["target"].shape) == (1, 256, 256)
    assert sample["condition"].dtype == torch.float32
    assert sample["target"].dtype == torch.float32
    assert sample["valid_mask"].dtype == torch.bool
    assert bool(sample["valid_mask"].any())
    assert sample["metadata"]["frequency_hz"] == TRAIN_FREQUENCY
    assert sample["metadata"]["beam_id"] == 0
    assert sample["metadata"]["steering_deg"] == 0.0
    assert float(sample["condition"][0, 127, 127]) == 1.0
    assert float(sample["condition"][1].max()) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "field,value",
    [
        ("array_rows", 4),
        ("array_cols", 4),
        ("steering_deg", 7.0),
        ("frequency_hz", TEST_FREQUENCY),
    ],
)
def test_dataset_rejects_protocol_mismatches(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    records = list(_records(tmp_path))
    records[0] = replace(records[0], **{field: value})

    with pytest.raises(CrossFrequencyDatasetError):
        _dataset(tmp_path, tuple(records), "train")


def test_dataset_rejects_unsafe_paths_and_bad_height_max(tmp_path: Path) -> None:
    records = list(_records(tmp_path))
    records[0] = replace(records[0], height_path="../outside.npy")
    with pytest.raises(CrossFrequencyDatasetError):
        _dataset(tmp_path, tuple(records), "train")

    height_root = tmp_path / "height_max"
    height_root.mkdir()
    records = _records(height_root)
    with pytest.raises(CrossFrequencyDatasetError):
        _dataset(height_root, records, "train", height_max=0.0)


def test_dataset_rejects_unknown_target_value(tmp_path: Path) -> None:
    records = _records(tmp_path)
    np.save(tmp_path / records[0].radiomap_path, np.full((128, 128), 1.0, dtype=np.float32))

    dataset = _dataset(tmp_path, records, "train")
    with pytest.raises(CrossFrequencyDatasetError):
        _ = dataset[0]


def test_dataset_rejects_empty_valid_mask(tmp_path: Path) -> None:
    records = _records(tmp_path)
    np.save(
        tmp_path / records[0].radiomap_path,
        np.full((128, 128), 1000.0, dtype=np.float32),
    )

    dataset = _dataset(tmp_path, records, "train")
    with pytest.raises(CrossFrequencyDatasetError):
        _ = dataset[0]
