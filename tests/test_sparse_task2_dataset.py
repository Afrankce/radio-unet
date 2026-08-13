from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from data_loaders.cross_frequency import DEFAULT_SOURCE_METADATA
from data_loaders.sparse_task2 import (
    SparseTask2DatasetError,
    SparseTask2RadiomapDataset,
    choose_valid_observation_mask,
    sparse_task2_collate,
)
from experiments.multiconfig_manifest import ARRAY_SPECS, ManifestRecord, write_manifest_jsonl
from training.sparse_task2_config import (
    SINGLEBEAM_TASK2_CONDITION_CHANNELS,
    SINGLEBEAM_TASK2_FREQUENCY_HZ,
    SINGLEBEAM_TASK2_SAMPLE_COUNT,
    SINGLEBEAM_TASK2_STEERING_DEG,
    SparseTask2DatasetConfig,
)


HEIGHT_MAX = 40.0
MINI_COUNTS = {"train": 2, "val": 1, "test": 1}


def _zero_degree_beam_id(array_size: str) -> int:
    matches = [
        beam.beam_id
        for beam in ARRAY_SPECS[array_size].beams
        if beam.steering_deg == 0.0
    ]
    assert len(matches) == 1
    return matches[0]


def _make_sample_files(root: Path, array_size: str, scene_id: str) -> tuple[str, str, str]:
    spec = ARRAY_SPECS[array_size]
    config_id = f"freq_6.7GHz_{spec.tx_elements}TR_8beams_pattern_tr38901"
    beam_id = _zero_degree_beam_id(array_size)
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
    np.save(root / beam_rel, np.full((128, 128), -150.0, dtype=np.float64))
    target = np.full((128, 128), -50.0, dtype=np.float32)
    target[0, 0] = -300.0
    target[0, 1] = 1000.0
    np.save(root / target_rel, target)
    return height_rel, beam_rel, target_rel


def _make_records(root: Path, *, array_size: str) -> tuple[ManifestRecord, ...]:
    spec = ARRAY_SPECS[array_size]
    beam_id = _zero_degree_beam_id(array_size)
    config_id = f"freq_6.7GHz_{spec.tx_elements}TR_8beams_pattern_tr38901"
    records: list[ManifestRecord] = []
    for split, count in MINI_COUNTS.items():
        for index in range(count):
            scene_id = f"{array_size.lower()}_{split}_{index:02d}"
            height_rel, beam_rel, target_rel = _make_sample_files(root, array_size, scene_id)
            records.append(
                ManifestRecord(
                    sample_key=f"{scene_id}|{array_size}|singlebeam_feature5_samples819",
                    split=split,
                    scene_id=scene_id,
                    array_name=array_size,
                    array_rows=spec.rows,
                    array_cols=spec.cols,
                    frequency_hz=SINGLEBEAM_TASK2_FREQUENCY_HZ,
                    config_id=config_id,
                    beam_id=beam_id,
                    steering_deg=SINGLEBEAM_TASK2_STEERING_DEG,
                    height_path=height_rel,
                    beam_map_path=beam_rel,
                    radiomap_path=target_rel,
                )
            )
    return tuple(records)


def _write_manifest(root: Path, array_size: str) -> Path:
    manifest = root / f"manifest_{array_size}.jsonl"
    write_manifest_jsonl(manifest, _make_records(root, array_size=array_size))
    return manifest


def _dataset(root: Path, array_size: str = "8x8") -> SparseTask2RadiomapDataset:
    return SparseTask2RadiomapDataset(
        dataset_root=root,
        manifest_path=_write_manifest(root, array_size),
        split="val",
        array_size=array_size,
        height_max=HEIGHT_MAX,
        expected_counts=MINI_COUNTS,
        source_metadata=DEFAULT_SOURCE_METADATA,
    )


def test_five_channel_order_and_exact_819_valid_observations(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    sample = dataset[0]

    assert tuple(sample["condition"].shape) == (SINGLEBEAM_TASK2_CONDITION_CHANNELS, 256, 256)
    assert tuple(sample["target"].shape) == (1, 256, 256)
    assert tuple(sample["valid_mask"].shape) == (1, 256, 256)
    assert tuple(sample["observation_mask"].shape) == (1, 256, 256)
    assert sample["condition"].dtype == torch.float32
    assert sample["target"].dtype == torch.float32
    assert sample["valid_mask"].dtype == torch.bool
    assert sample["observation_mask"].dtype == torch.bool
    assert sample["observation_mask"].sum().item() == SINGLEBEAM_TASK2_SAMPLE_COUNT
    assert torch.all(~sample["observation_mask"] | sample["valid_mask"])
    assert torch.equal(sample["condition"][0:1], sample["sparse_map"])
    assert torch.equal(sample["condition"][1:2], sample["observation_mask"].float())
    assert sample["condition"][2, 127, 127].item() == 1.0
    assert torch.count_nonzero(sample["sparse_map"][~sample["observation_mask"]]) == 0
    assert sample["metadata"]["observed_pixels"] == SINGLEBEAM_TASK2_SAMPLE_COUNT
    assert sample["metadata"]["protocol"] == "singlebeam_feature5_samples819"


def test_mask_is_deterministic_and_scene_keyed(tmp_path: Path) -> None:
    first = _dataset(tmp_path / "first")[0]
    second = _dataset(tmp_path / "second")[0]
    assert torch.equal(first["observation_mask"], second["observation_mask"])
    assert torch.equal(first["condition"], second["condition"])

    valid = torch.ones((1, 256, 256), dtype=torch.bool)
    mask_a = choose_valid_observation_mask(valid, scene_id="scene-a")
    mask_b = choose_valid_observation_mask(valid, scene_id="scene-b")
    assert not torch.equal(mask_a, mask_b)


@pytest.mark.parametrize("array_size", ("8x8", "16x16", "32x32"))
def test_all_array_sizes_use_their_zero_degree_configuration(
    tmp_path: Path,
    array_size: str,
) -> None:
    sample = _dataset(tmp_path / array_size, array_size)[0]
    assert sample["metadata"]["array_size"] == array_size
    assert sample["metadata"]["steering_deg"] == 0.0
    assert sample["metadata"]["beam_id"] == _zero_degree_beam_id(array_size)


@pytest.mark.parametrize(
    ("field", "value"),
    (("frequency_hz", 4_900_000_000), ("steering_deg", 7.0), ("array_name", "16x16")),
)
def test_rejects_record_protocol_mismatches(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root = tmp_path / field
    records = list(_make_records(root, array_size="8x8"))
    records[0] = replace(records[0], **{field: value})
    manifest = root / "bad.jsonl"
    write_manifest_jsonl(manifest, records)
    with pytest.raises(SparseTask2DatasetError):
        SparseTask2RadiomapDataset(
            dataset_root=root,
            manifest_path=manifest,
            split="train",
            array_size="8x8",
            height_max=HEIGHT_MAX,
            expected_counts=MINI_COUNTS,
        )


def test_rejects_wrong_count_and_too_few_valid_pixels(tmp_path: Path) -> None:
    root = tmp_path / "bad_count"
    manifest = _write_manifest(root, "8x8")
    with pytest.raises(SparseTask2DatasetError, match="locked to 819"):
        SparseTask2RadiomapDataset(
            dataset_root=root,
            manifest_path=manifest,
            split="val",
            array_size="8x8",
            height_max=HEIGHT_MAX,
            expected_counts=MINI_COUNTS,
            sample_count=818,
        )
    with pytest.raises(ValueError, match="exceeds valid pixel count"):
        choose_valid_observation_mask(torch.ones((1, 10, 10), dtype=torch.bool), scene_id="x")


def test_rejects_negative_or_nonfinite_height(tmp_path: Path) -> None:
    root = tmp_path / "bad_height"
    manifest = _write_manifest(root, "8x8")
    record = _make_records(root, array_size="8x8")[0]
    height_path = root / record.height_path
    np.save(height_path, np.full((256, 256), -1.0, dtype=np.float32))
    with pytest.raises(SparseTask2DatasetError, match="negative"):
        _dataset_from_manifest(root, manifest, "train")[0]

    np.save(height_path, np.full((256, 256), np.nan, dtype=np.float32))
    with pytest.raises(SparseTask2DatasetError, match="non-finite"):
        _dataset_from_manifest(root, manifest, "train")[0]


def _dataset_from_manifest(
    root: Path,
    manifest: Path,
    split: str,
) -> SparseTask2RadiomapDataset:
    return SparseTask2RadiomapDataset(
        dataset_root=root,
        manifest_path=manifest,
        split=split,
        array_size="8x8",
        height_max=HEIGHT_MAX,
        expected_counts=MINI_COUNTS,
    )


def test_collate_stacks_tensors_and_preserves_metadata(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    sample = dataset[0]
    batch = sparse_task2_collate([sample, sample])
    assert batch["condition"].shape == (2, 5, 256, 256)
    assert batch["target"].shape == (2, 1, 256, 256)
    assert batch["valid_mask"].shape == (2, 1, 256, 256)
    assert len(batch["metadata"]) == 2


def test_config_locks_the_mandatory_protocol(tmp_path: Path) -> None:
    config = SparseTask2DatasetConfig(
        dataset_root=tmp_path,
        manifest_path=tmp_path / "manifest.jsonl",
        split="val",
        array_size="8x8",
        height_max=HEIGHT_MAX,
        expected_counts=MINI_COUNTS,
    )
    payload = config.to_dict()
    assert payload["protocol"] == "singlebeam_feature5_samples819"
    assert payload["condition_channels"] == 5
    assert payload["sample_count"] == 819
    assert payload["frequency_hz"] == 6_700_000_000
