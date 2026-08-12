from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from data_loaders.same_frequency import SameFrequencyDatasetError, SameFrequencyRadiomapDataset
from experiments.multiconfig_manifest import ARRAY_SPECS, ManifestRecord, write_manifest_jsonl
from training.sparse_masks import build_masked_condition_map, make_condition_noise, make_observation_mask


FREQUENCY_HZ = 6_700_000_000
HEIGHT_MAX = 40.0
FORMAL_VARIANT = "beam_masked"
FULL_COUNTS = {"train": 560, "val": 80, "test": 160}
MINI_COUNTS = {"train": 2, "val": 1, "test": 1}


def _zero_degree_beam_id(array_size: str) -> int:
    for beam in ARRAY_SPECS[array_size].beams:
        if math.isclose(beam.steering_deg, 0.0, abs_tol=1e-9):
            return beam.beam_id
    raise AssertionError(f"{array_size} is missing a zero-degree beam")


def _write_shared_sample(root: Path, array_size: str) -> tuple[str, str, str]:
    beam_id = _zero_degree_beam_id(array_size)
    spec = ARRAY_SPECS[array_size]
    config_id = f"freq_6.7GHz_{spec.tx_elements}TR_8beams_pattern_tr38901"
    scene_id = f"{array_size.lower()}_template"
    height_rel = f"raw/Dataset/height_maps/{scene_id}/{scene_id}_height_matrix.npy"
    beam_rel = f"raw/Dataset/beam_maps/{config_id}/u0/beam_{beam_id:02d}_angle_0.0_matrix.npy"
    target_rel = f"raw/Dataset/radiomaps/{config_id}_beam{beam_id:02d}/{scene_id}_labeled_radiomap.npy"
    for relative in (height_rel, beam_rel, target_rel):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    np.save(root / height_rel, np.full((256, 256), 20.0, dtype=np.float32))
    np.save(root / beam_rel, np.full((128, 128), -150.0, dtype=np.float64))
    target = np.full((128, 128), -50.0, dtype=np.float32)
    target[0, 0] = -300.0
    target[0, 1] = 1000.0
    np.save(root / target_rel, target)
    return height_rel, beam_rel, target_rel


def _make_records(
    root: Path,
    *,
    array_size: str,
    counts: dict[str, int],
) -> tuple[ManifestRecord, ...]:
    height_rel, beam_rel, target_rel = _write_shared_sample(root, array_size)
    beam_id = _zero_degree_beam_id(array_size)
    spec = ARRAY_SPECS[array_size]
    config_id = f"freq_6.7GHz_{spec.tx_elements}TR_8beams_pattern_tr38901"
    records: list[ManifestRecord] = []
    for split in ("train", "val", "test"):
        for index in range(counts[split]):
            scene_id = f"{array_size.lower()}_{split}_{index:04d}"
            records.append(
                ManifestRecord(
                    sample_key=f"{scene_id}|{array_size}|beam_masked",
                    split=split,
                    scene_id=scene_id,
                    array_name=array_size,
                    array_rows=spec.rows,
                    array_cols=spec.cols,
                    frequency_hz=FREQUENCY_HZ,
                    config_id=config_id,
                    beam_id=beam_id,
                    steering_deg=0.0,
                    height_path=height_rel,
                    beam_map_path=beam_rel,
                    radiomap_path=target_rel,
                )
            )
    return tuple(records)


def _write_manifest(root: Path, array_size: str, records: tuple[ManifestRecord, ...]) -> Path:
    manifest = root / f"{array_size}_manifest.jsonl"
    write_manifest_jsonl(manifest, records)
    return manifest


def test_sparse_dataset_returns_five_channel_condition_for_all_arrays(tmp_path: Path) -> None:
    from data_loaders.sparse_same_frequency import SparseSameFrequencyRadiomapDataset

    for array_size in ARRAY_SPECS:
        root = tmp_path / array_size
        records = _make_records(root, array_size=array_size, counts=MINI_COUNTS)
        manifest = _write_manifest(root, array_size, records)
        dataset = SparseSameFrequencyRadiomapDataset(
            dataset_root=root,
            manifest_path=manifest,
            split="val",
            array_size=array_size,
            variant=FORMAL_VARIANT,
            height_max=HEIGHT_MAX,
            expected_counts=MINI_COUNTS,
        )
        sample = dataset[0]

        assert tuple(sample["condition"].shape) == (5, 256, 256)
        assert tuple(sample["target"].shape) == (1, 256, 256)
        assert tuple(sample["masked_map"].shape) == (1, 256, 256)
        assert tuple(sample["observed_map"].shape) == (1, 256, 256)
        assert tuple(sample["valid_mask"].shape) == (1, 256, 256)
        assert tuple(sample["observation_mask"].shape) == (1, 256, 256)
        assert tuple(sample["missing_mask"].shape) == (1, 256, 256)
        assert sample["condition"].dtype == torch.float32
        assert sample["target"].dtype == torch.float32
        assert sample["valid_mask"].dtype == torch.bool
        assert torch.equal(sample["condition"][3], sample["masked_map"][0])
        assert torch.equal(
            sample["condition"][4],
            sample["observation_mask"].to(dtype=torch.float32)[0],
        )
        assert sample["metadata"]["array_size"] == array_size
        assert sample["metadata"]["split"] == "val"
        assert sample["metadata"]["observation_ratio"] == 0.05
        assert sample["metadata"]["steering_deg"] == 0.0


def test_sparse_dataset_preserves_legacy_three_channel_dataset_default(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    records = _make_records(root, array_size="8x8", counts={"train": 1, "val": 1, "test": 1})
    manifest = _write_manifest(root, "8x8", records)

    dataset = SameFrequencyRadiomapDataset(
        dataset_root=root,
        manifest_path=manifest,
        split="train",
        array_size="8x8",
        height_max=HEIGHT_MAX,
        expected_counts={"train": 1, "val": 1, "test": 1},
    )

    assert tuple(dataset[0]["condition"].shape) == (3, 256, 256)


def test_sparse_dataset_mask_and_noise_contract_for_validation_split(tmp_path: Path) -> None:
    from data_loaders.sparse_same_frequency import SparseSameFrequencyRadiomapDataset

    root = tmp_path / "contract"
    records = _make_records(root, array_size="8x8", counts=MINI_COUNTS)
    manifest = _write_manifest(root, "8x8", records)
    dataset = SparseSameFrequencyRadiomapDataset(
        dataset_root=root,
        manifest_path=manifest,
        split="val",
        array_size="8x8",
        variant=FORMAL_VARIANT,
        height_max=HEIGHT_MAX,
        expected_counts=MINI_COUNTS,
    )
    sample = dataset[0]
    expected_observation_mask = make_observation_mask(
        sample["valid_mask"],
        scene_id=sample["metadata"]["scene_id"],
        steering_deg=sample["metadata"]["steering_deg"],
        ratio=0.05,
        base_seed=42,
    )
    expected_noise = make_condition_noise(
        tuple(sample["target"].shape),
        scene_id=sample["metadata"]["scene_id"],
        steering_deg=sample["metadata"]["steering_deg"],
        split="val",
        epoch=None,
        base_seed=4242,
    )
    expected_masked_map, expected_observed_map, expected_missing_mask = build_masked_condition_map(
        sample["target"],
        sample["valid_mask"],
        expected_observation_mask,
        expected_noise,
    )

    assert not bool((sample["observation_mask"] & ~sample["valid_mask"]).any())
    assert torch.equal(sample["observation_mask"], expected_observation_mask)
    assert torch.equal(sample["missing_mask"], sample["valid_mask"] & ~sample["observation_mask"])
    assert torch.equal(sample["missing_mask"], expected_missing_mask)
    assert torch.equal(sample["masked_map"], expected_masked_map)
    assert torch.equal(sample["observed_map"], expected_observed_map)
    assert torch.equal(
        sample["masked_map"][sample["observation_mask"]],
        sample["target"][sample["observation_mask"]],
    )
    assert torch.equal(sample["observed_map"][sample["observation_mask"]], sample["target"][sample["observation_mask"]])
    assert torch.count_nonzero(sample["masked_map"][~sample["valid_mask"]]) == 0
    assert torch.count_nonzero(sample["observed_map"][~sample["valid_mask"]]) == 0
    valid_pixels = int(sample["valid_mask"].sum().item())
    observed_pixels = int(sample["observation_mask"].sum().item())
    assert valid_pixels == 65_528
    assert observed_pixels == math.ceil(valid_pixels * 0.05)
    assert sample["metadata"]["valid_pixels"] == valid_pixels
    assert sample["metadata"]["observed_pixels"] == observed_pixels


def test_sparse_dataset_train_noise_changes_with_epoch_but_mask_stays_fixed(tmp_path: Path) -> None:
    from data_loaders.sparse_same_frequency import SparseSameFrequencyRadiomapDataset

    root = tmp_path / "epoch"
    records = _make_records(root, array_size="16x16", counts=MINI_COUNTS)
    manifest = _write_manifest(root, "16x16", records)
    dataset = SparseSameFrequencyRadiomapDataset(
        dataset_root=root,
        manifest_path=manifest,
        split="train",
        array_size="16x16",
        variant=FORMAL_VARIANT,
        height_max=HEIGHT_MAX,
        expected_counts=MINI_COUNTS,
    )

    sample_epoch0 = dataset[0]
    dataset.set_epoch(1)
    sample_epoch1 = dataset[0]

    expected_noise0 = make_condition_noise(
        tuple(sample_epoch0["target"].shape),
        scene_id=sample_epoch0["metadata"]["scene_id"],
        steering_deg=sample_epoch0["metadata"]["steering_deg"],
        split="train",
        epoch=0,
        base_seed=4242,
    )
    expected_noise1 = make_condition_noise(
        tuple(sample_epoch1["target"].shape),
        scene_id=sample_epoch1["metadata"]["scene_id"],
        steering_deg=sample_epoch1["metadata"]["steering_deg"],
        split="train",
        epoch=1,
        base_seed=4242,
    )

    assert torch.equal(sample_epoch0["observation_mask"], sample_epoch1["observation_mask"])
    assert torch.equal(sample_epoch0["missing_mask"], sample_epoch1["missing_mask"])
    assert torch.equal(sample_epoch0["target"], sample_epoch1["target"])
    assert torch.equal(
        sample_epoch0["masked_map"][sample_epoch0["missing_mask"]],
        expected_noise0[sample_epoch0["missing_mask"]],
    )
    assert torch.equal(
        sample_epoch1["masked_map"][sample_epoch1["missing_mask"]],
        expected_noise1[sample_epoch1["missing_mask"]],
    )
    assert not torch.allclose(
        sample_epoch0["masked_map"][sample_epoch0["missing_mask"]],
        sample_epoch1["masked_map"][sample_epoch1["missing_mask"]],
    )


def test_sparse_dataset_enforces_full_count_contract(tmp_path: Path) -> None:
    from data_loaders.sparse_same_frequency import SparseSameFrequencyRadiomapDataset

    root = tmp_path / "counts"
    records = _make_records(root, array_size="8x8", counts=FULL_COUNTS)
    manifest = _write_manifest(root, "8x8", records)

    train = SparseSameFrequencyRadiomapDataset(
        dataset_root=root,
        manifest_path=manifest,
        split="train",
        array_size="8x8",
        variant=FORMAL_VARIANT,
        height_max=HEIGHT_MAX,
    )
    assert len(train) == 560

    bad_manifest = _write_manifest(root, "8x8_bad", records[:-1])
    with pytest.raises(SameFrequencyDatasetError, match="count mismatch"):
        SparseSameFrequencyRadiomapDataset(
            dataset_root=root,
            manifest_path=bad_manifest,
            split="test",
            array_size="8x8",
            variant=FORMAL_VARIANT,
            height_max=HEIGHT_MAX,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_key", "shared-key"),
        ("scene_id", "duplicate-scene"),
        ("frequency_hz", 4_900_000_000),
        ("steering_deg", 7.0),
    ],
)
def test_sparse_dataset_rejects_duplicate_or_protocol_mismatch_records(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    from data_loaders.sparse_same_frequency import SparseSameFrequencyRadiomapDataset

    root = tmp_path / f"invalid_{field}"
    records = list(_make_records(root, array_size="32x32", counts=MINI_COUNTS))
    records[1] = replace(records[0], **{field: value})
    manifest = _write_manifest(root, "32x32", tuple(records))

    with pytest.raises(SameFrequencyDatasetError):
        SparseSameFrequencyRadiomapDataset(
            dataset_root=root,
            manifest_path=manifest,
            split="train",
            array_size="32x32",
            variant=FORMAL_VARIANT,
            height_max=HEIGHT_MAX,
            expected_counts=MINI_COUNTS,
        )


def test_sparse_dataset_rejects_nonformal_variant(tmp_path: Path) -> None:
    from data_loaders.sparse_same_frequency import SparseSameFrequencyRadiomapDataset

    root = tmp_path / "variant"
    records = _make_records(root, array_size="8x8", counts=MINI_COUNTS)
    manifest = _write_manifest(root, "8x8", records)

    with pytest.raises(SameFrequencyDatasetError, match="beam_masked"):
        SparseSameFrequencyRadiomapDataset(
            dataset_root=root,
            manifest_path=manifest,
            split="train",
            array_size="8x8",
            variant="no_beam_masked",
            height_max=HEIGHT_MAX,
            expected_counts=MINI_COUNTS,
        )
