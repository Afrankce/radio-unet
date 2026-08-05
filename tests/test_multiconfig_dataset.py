from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.multiconfig_manifest import (
    DatasetSchemaLock,
    ManifestRecord,
    canonical_json_bytes,
    write_manifest_jsonl,
)
from experiments.provenance import DATASET_REVISION, REFERENCE_CODE_REVISION, sha256_file


def _dataset_module():
    from data_loaders import multiconfig

    return multiconfig


def _record(
    *,
    split: str,
    height_path: str,
    beam_map_path: str,
    radiomap_path: str,
    scene_id: str = "u1",
) -> ManifestRecord:
    return ManifestRecord(
        sample_key=f"{scene_id}|8x8|beam00",
        split=split,
        scene_id=scene_id,
        array_name="8x8",
        array_rows=8,
        array_cols=8,
        frequency_hz=6_700_000_000,
        config_id="config-a",
        beam_id=0,
        steering_deg=-28.0,
        height_path=height_path,
        beam_map_path=beam_map_path,
        radiomap_path=radiomap_path,
    )


def _synthetic_dataset_context(
    tmp_path: Path,
    *,
    split: str = "train",
    height_value: float = 10.0,
    height_max: float = 20.0,
    object_radiomap: bool = False,
):
    module = _dataset_module()
    root = tmp_path
    data_root = root / "raw" / "Dataset"
    height_path = data_root / "height_maps" / "u1" / "u1_height_matrix.npy"
    beam_path = (
        data_root
        / "beam_maps"
        / "config-a"
        / "u0"
        / "beam_00_angle_-28.0_matrix.npy"
    )
    radio_path = (
        data_root
        / "radiomaps"
        / "config-a_beam00"
        / "u1_labeled_radiomap.npy"
    )
    for path in (height_path, beam_path, radio_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(height_path, np.full((256, 256), height_value, dtype=np.float32))
    np.save(
        beam_path,
        np.array([[-300.0, 0.0], [-150.0, -75.0]], dtype=np.float64),
    )
    radiomap = np.array([[-300.0, -299.0], [1000.0, -150.0]], dtype=np.float32)
    if object_radiomap:
        radiomap = radiomap.astype(object)
    np.save(radio_path, radiomap)

    relative = lambda path: path.relative_to(root).as_posix()
    record = _record(
        split=split,
        height_path=relative(height_path),
        beam_map_path=relative(beam_path),
        radiomap_path=relative(radio_path),
    )
    manifest_path = root / "manifests" / "manifest_8x8.jsonl"
    write_manifest_jsonl(manifest_path, (record,))
    split_path = root / "manifests" / "scene_split_seed42.json"
    split_path.write_bytes(b"{}\n")

    schema_payload = {
        "schema_version": 1,
        "data_root": "raw/Dataset",
        "identities": {
            "dataset_revision": DATASET_REVISION,
            "reference_code_revision": REFERENCE_CODE_REVISION,
        },
        "configurations": [{"configuration_id": "config-a"}],
        "arrays": [
            {
                "name": "8x8",
                "configuration_id": "config-a",
                "rows": 8,
                "cols": 8,
                "tx_elements": 64,
                "frequency_hz": 6_700_000_000,
                "selected_beams": [{"beam_id": 0, "steering_deg": -28.0}],
            }
        ],
        "source_metadata": {
            "height": {"shape": [256, 256], "dtype": "float32"},
            "beam_map": {"shape": [2, 2], "dtype": "float64"},
            "radiomap": {"shape": [2, 2], "dtype": "float32"},
        },
        "transmitter": {"output_pixel_rc": [127, 127]},
        "target_domain": {
            "floor_db": -300.0,
            "building_sentinel": 1000.0,
            "valid_lower_exclusive_db": -300.0,
            "valid_upper_exclusive_db": 0.0,
        },
        "configuration_text_hashes": [],
        "reference_scripts": [],
    }
    schema = DatasetSchemaLock.from_json(json.dumps(schema_payload))
    schema.validate_source_revisions()
    schema_sha256 = hashlib.sha256(canonical_json_bytes(schema.raw)).hexdigest()
    evidence = tuple(
        module.HeightFileEvidence(
            scene_id=f"u{index}",
            relative_path=f"train/u{index}.npy",
            sha256="a" * 64,
        )
        for index in range(1, 561)
    )
    stats = module.HeightStats(
        schema_version=1,
        height_max=height_max,
        derived_from="train",
        scene_count=560,
        height_files=evidence,
        split_sha256=sha256_file(split_path),
        manifest_sha256={manifest_path.name: sha256_file(manifest_path)},
        schema_sha256=schema_sha256,
        schema_identity={
            "dataset_revision": DATASET_REVISION,
            "archive_sha256": "synthetic",
        },
    )
    dataset = module.MultiConfigRadiomapDataset(
        dataset_root=root,
        manifest_path=manifest_path,
        split=split,
        schema=schema,
        height_stats=stats,
    )
    return module, dataset, record, stats, radio_path


def test_db_normalization_round_trip() -> None:
    module = _dataset_module()
    values = torch.tensor([-300.0, -299.0, -150.0, 0.0])

    normalized = module.normalize_db(values)

    assert torch.allclose(normalized, torch.tensor([0.0, 1 / 300, 0.5, 1.0]))
    assert torch.allclose(module.denormalize_db(normalized), values)


def test_db_normalization_supports_an_explicit_locked_interval() -> None:
    module = _dataset_module()
    values = torch.tensor([-120.0, -60.0, 0.0])

    normalized = module.normalize_db(values, floor_db=-120.0, ceiling_db=0.0)

    assert torch.allclose(normalized, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(
        module.denormalize_db(
            normalized,
            floor_db=-120.0,
            ceiling_db=0.0,
        ),
        values,
    )


def test_target_mask_excludes_floor_and_buildings() -> None:
    module = _dataset_module()
    values = torch.tensor([[-300.0, -299.0], [1000.0, -150.0]])

    target, mask = module.prepare_target(values)

    assert torch.equal(mask, torch.tensor([[False, True], [False, True]]))
    assert torch.allclose(
        target,
        torch.tensor([[0.0, 1 / 300], [0.0, 0.5]]),
    )


def test_unexpected_target_value_fails() -> None:
    module = _dataset_module()

    with pytest.raises(module.TargetValueError, match="unknown"):
        module.prepare_target(torch.tensor([[-299.0, 5.0]]))
    with pytest.raises(module.TargetValueError, match="non-finite"):
        module.prepare_target(torch.tensor([[-299.0, float("nan")]]))


def test_empty_valid_mask_fails() -> None:
    module = _dataset_module()

    with pytest.raises(module.EmptyValidMaskError, match="no valid"):
        module.prepare_target(torch.tensor([[-300.0, 1000.0]]))


def test_height_max_uses_train_scenes_only(tmp_path: Path) -> None:
    module = _dataset_module()
    records = []
    for index in range(1, 561):
        path = tmp_path / "height" / f"u{index}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.array([[float((index % 10) + 1)]], dtype=np.float32))
        records.append(
            _record(
                split="train",
                scene_id=f"u{index}",
                height_path=path.relative_to(tmp_path).as_posix(),
                beam_map_path="unused-beam.npy",
                radiomap_path="unused-radio.npy",
            )
        )
    validation_path = tmp_path / "height" / "u999.npy"
    np.save(validation_path, np.array([[999.0]], dtype=np.float32))

    summary = module.compute_train_height_max(tmp_path, records)

    assert summary.height_max == 10.0
    assert summary.scene_count == 560
    assert len(summary.height_files) == 560
    assert all(item.scene_id != "u999" for item in summary.height_files)


def test_continuous_maps_use_bilinear() -> None:
    module = _dataset_module()
    values = torch.tensor([[[[0.0, 1.0], [0.5, 0.75]]]])

    resized = module.resize_continuous(values, (4, 4))

    expected = F.interpolate(values, size=(4, 4), mode="bilinear", align_corners=False)
    assert torch.allclose(resized, expected)
    assert not torch.equal(resized, F.interpolate(values, size=(4, 4), mode="nearest"))


def test_valid_mask_uses_nearest() -> None:
    module = _dataset_module()
    mask = torch.tensor([[[[False, True], [True, False]]]])

    resized = module.resize_valid_mask(mask, (4, 4))

    expected = F.interpolate(mask.float(), size=(4, 4), mode="nearest").bool()
    assert resized.dtype == torch.bool
    assert torch.equal(resized, expected)


def test_tx_mask_has_one_pixel_at_127_127() -> None:
    module = _dataset_module()

    tx = module.build_tx_mask((256, 256), (127, 127))

    assert tx.shape == (1, 256, 256)
    assert tx.dtype == torch.float32
    assert tx.sum().item() == 1.0
    assert tx[0, 127, 127].item() == 1.0


def test_dataset_returns_fixed_shapes_and_channel_order(tmp_path: Path) -> None:
    module, dataset, record, _stats, _radio = _synthetic_dataset_context(tmp_path)

    sample = dataset[0]

    assert sample["condition"].shape == (3, 256, 256)
    assert sample["condition"].dtype == torch.float32
    assert sample["target"].shape == (1, 256, 256)
    assert sample["target"].dtype == torch.float32
    assert sample["valid_mask"].shape == (1, 256, 256)
    assert sample["valid_mask"].dtype == torch.bool
    assert sample["condition"][0].sum().item() == 1.0
    assert sample["condition"][0, 127, 127].item() == 1.0
    assert torch.allclose(sample["condition"][1], torch.full((256, 256), 0.5))
    source_beam = torch.tensor([[[[0.0, 1.0], [0.5, 0.75]]]])
    expected_beam = module.resize_continuous(source_beam, (256, 256))[0, 0]
    assert torch.allclose(sample["condition"][2], expected_beam)
    assert sample["metadata"]["sample_key"] == record.sample_key
    assert set(sample["metadata"]) == {
        "sample_key",
        "split",
        "scene_id",
        "array_name",
        "array_rows",
        "array_cols",
        "frequency_hz",
        "config_id",
        "beam_id",
        "steering_deg",
        "height_path",
        "beam_map_path",
        "radiomap_path",
        "tx_rc",
    }


def test_dataset_rejects_nonfixed_output_size(tmp_path: Path) -> None:
    module, dataset, _record_value, stats, _radio = _synthetic_dataset_context(tmp_path)

    with pytest.raises(module.DatasetContractError, match="fixed output size"):
        module.MultiConfigRadiomapDataset(
            dataset_root=dataset.dataset_root,
            manifest_path=dataset.manifest_path,
            split=dataset.split,
            schema=dataset.schema,
            height_stats=stats,
            output_size=(128, 128),
        )


@pytest.mark.parametrize("split", ["val", "test"])
def test_val_and_test_reuse_train_height_max(
    tmp_path: Path,
    split: str,
) -> None:
    _module, dataset, _record_value, _stats, _radio = _synthetic_dataset_context(
        tmp_path,
        split=split,
        height_value=20.0,
        height_max=10.0,
    )

    sample = dataset[0]

    assert sample["condition"][1].max().item() == 2.0


def test_np_load_rejects_object_arrays(tmp_path: Path) -> None:
    module, dataset, _record_value, _stats, radio_path = _synthetic_dataset_context(
        tmp_path,
        object_radiomap=True,
    )

    with pytest.raises(module.DataFormatError, match=str(radio_path).replace("\\", "\\\\")):
        _ = dataset[0]


def test_collate_preserves_metadata_as_records(tmp_path: Path) -> None:
    module, dataset, _record_value, _stats, _radio = _synthetic_dataset_context(tmp_path)
    sample = dataset[0]

    batch = module.multiconfig_collate([sample, sample])

    assert batch["condition"].shape == (2, 3, 256, 256)
    assert batch["target"].shape == (2, 1, 256, 256)
    assert batch["valid_mask"].shape == (2, 1, 256, 256)
    assert isinstance(batch["metadata"], list)
    assert batch["metadata"][0] == sample["metadata"]


def test_compute_height_stats_cli_uses_fixed_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare = __import__("prepare_multiconfig")
    output = tmp_path / "manifests" / "height_stats_train.json"
    calls: list[tuple[Path, Path, Path]] = []

    def fake_compute(dataset_root: Path, manifest_dir: Path, schema_path: Path):
        calls.append((dataset_root, manifest_dir, schema_path))
        return output, type(
            "Stats",
            (),
            {"height_max": 42.0, "scene_count": 560, "derived_from": "train"},
        )()

    monkeypatch.setattr(prepare, "compute_height_stats_artifact", fake_compute)

    result = prepare.main(
        [
            "compute-height-stats",
            "--dataset-root",
            str(tmp_path),
            "--manifest-dir",
            str(tmp_path / "manifests"),
        ]
    )

    assert result == 0
    assert calls == [
        (
            tmp_path.resolve(),
            (tmp_path / "manifests").resolve(),
            (Path(prepare.__file__).parent / "experiments" / "multiconfig_schema.json").resolve(),
        )
    ]
