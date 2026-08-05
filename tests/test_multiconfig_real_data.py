from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pytest


SCHEMA_PATH = Path(__file__).parents[1] / "experiments" / "multiconfig_schema.json"


def _dataset_root() -> Path:
    value = os.environ.get("MULTICONFIG_ROOT")
    if not value:
        pytest.skip("MULTICONFIG_ROOT is not set")
    root = Path(value)
    if not root.is_dir():
        pytest.skip(f"MULTICONFIG_ROOT does not exist: {root}")
    return root


@pytest.mark.dataset
def test_real_schema_split_and_manifests_are_strictly_bound() -> None:
    from experiments import multiconfig_manifest as manifest

    dataset_root = _dataset_root()
    manifest_dir = dataset_root / "manifests"
    summary = manifest.verify_schema_lock(
        dataset_root,
        SCHEMA_PATH,
        progress=False,
    )
    assert summary["scene_count"] == 800
    assert summary["beam_map_files"] == 24
    assert summary["radiomap_files"] == 19200

    schema = manifest.load_schema_lock(SCHEMA_PATH)
    inventory = manifest.inventory_samples(dataset_root, schema)
    split_payload = json.loads(
        (manifest_dir / "scene_split_seed42.json").read_text(encoding="utf-8")
    )
    split = manifest.SceneSplit.from_dict(split_payload)
    manifest.validate_scene_split(split, inventory)
    assert (len(split.train), len(split.val), len(split.test)) == (560, 80, 160)

    expected_ids = {
        "8x8": (0, 1, 2, 3, 4, 5, 6, 7),
        "16x16": (0, 2, 4, 6, 8, 10, 12, 14),
        "32x32": (4, 11, 18, 25, 32, 39, 46, 53),
    }
    expected_angles = (-28.0, -21.0, -14.0, -7.0, 0.0, 7.0, 14.0, 21.0)
    scene_sets: dict[str, set[str]] = {}
    for array_name in manifest.ARRAY_SPECS:
        path = manifest_dir / f"manifest_{array_name}.jsonl"
        records = manifest.load_manifest_jsonl(path)
        manifest.validate_manifest(
            records,
            inventory,
            schema,
            split,
            array_name,
        )
        assert len(records) == 6400
        assert Counter(record.split for record in records) == {
            "train": 4480,
            "val": 640,
            "test": 1280,
        }
        assert tuple(sorted({record.beam_id for record in records})) == (
            expected_ids[array_name]
        )
        angle_by_beam = {
            record.beam_id: record.steering_deg for record in records
        }
        assert tuple(
            angle_by_beam[beam_id] for beam_id in expected_ids[array_name]
        ) == expected_angles
        per_beam_split = Counter(
            (record.beam_id, record.split) for record in records
        )
        for beam_id in expected_ids[array_name]:
            assert per_beam_split[(beam_id, "train")] == 560
            assert per_beam_split[(beam_id, "val")] == 80
            assert per_beam_split[(beam_id, "test")] == 160
        assert len({record.sample_key for record in records}) == 6400
        assert len({record.radiomap_path for record in records}) == 6400
        assert len(
            {
                (
                    record.height_path,
                    record.beam_map_path,
                    record.radiomap_path,
                )
                for record in records
            }
        ) == 6400
        assert all(
            (dataset_root / record.height_path).is_file()
            and (dataset_root / record.beam_map_path).is_file()
            and (dataset_root / record.radiomap_path).is_file()
            for record in records
        )
        scene_sets[array_name] = {record.scene_id for record in records}
        assert len(scene_sets[array_name]) == 800

    assert scene_sets["8x8"] == scene_sets["16x16"] == scene_sets["32x32"]
    visualization = json.loads(
        (manifest_dir / "visualization_cases_seed42.json").read_text(
            encoding="utf-8"
        )
    )
    assert visualization == manifest.build_visualization_cases(split)
