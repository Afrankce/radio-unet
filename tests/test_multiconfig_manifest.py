from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest


SCHEMA_PATH = Path(__file__).parents[1] / "experiments" / "multiconfig_schema.json"
ARRAY_CONFIGS = {
    "8x8": "freq_6.7GHz_64TR_8beams_pattern_tr38901",
    "16x16": "freq_6.7GHz_256TR_16beams_pattern_tr38901",
    "32x32": "freq_6.7GHz_1024TR_64beams_pattern_tr38901",
}


def _manifest_module():
    from experiments import multiconfig_manifest

    return multiconfig_manifest


class SyntheticInventory:
    def __init__(
        self,
        workspace_root: Path,
        scene_sets: dict[str, set[str]] | None = None,
    ) -> None:
        self.workspace_root = workspace_root
        common = {f"u{index}" for index in range(1, 801)}
        self._scene_sets = scene_sets or {
            array_name: set(common) for array_name in ARRAY_CONFIGS
        }
        self.missing: set[tuple[str, int, str]] = set()

    def scene_ids_by_array(self) -> dict[str, set[str]]:
        return {
            array_name: set(scene_ids)
            for array_name, scene_ids in self._scene_sets.items()
        }

    def require_unique_triplet(
        self,
        config_id: str,
        beam_id: int,
        scene_id: str,
    ) -> tuple[Path, Path, Path]:
        manifest = _manifest_module()
        key = (config_id, beam_id, scene_id)
        if key in self.missing:
            raise manifest.MissingSamplePathError(f"missing synthetic {key!r}")
        root = self.workspace_root / "raw" / "Dataset"
        height = root / "height_maps" / scene_id / f"{scene_id}_height_matrix.npy"
        beam_map = (
            root
            / "beam_maps"
            / config_id
            / "u0"
            / f"beam_{beam_id:02d}_angle_synthetic_matrix.npy"
        )
        radiomap = (
            root
            / "radiomaps"
            / f"{config_id}_beam{beam_id:02d}"
            / f"{scene_id}_labeled_radiomap.npy"
        )
        return height, beam_map, radiomap


def _schema():
    return _manifest_module().load_schema_lock(SCHEMA_PATH)


def test_scene_split_is_fixed_natural_sorted_and_disjoint(tmp_path: Path) -> None:
    manifest = _manifest_module()
    inventory = SyntheticInventory(tmp_path)

    split = manifest.build_scene_split(inventory)

    assert split.seed == 42
    assert split.algorithm == "python_random_v1"
    assert (len(split.train), len(split.val), len(split.test)) == (560, 80, 160)
    assert split.train[:10] == (
        "u733",
        "u375",
        "u574",
        "u491",
        "u648",
        "u532",
        "u359",
        "u650",
        "u451",
        "u774",
    )
    assert not (set(split.train) & set(split.val))
    assert not (set(split.train) & set(split.test))
    assert not (set(split.val) & set(split.test))
    assert set(split.train) | set(split.val) | set(split.test) == {
        f"u{index}" for index in range(1, 801)
    }


def test_scene_split_checks_each_array_before_comparing_sets(tmp_path: Path) -> None:
    manifest = _manifest_module()
    common = {f"u{index}" for index in range(1, 801)}
    missing_one = {
        "8x8": set(common),
        "16x16": set(common) - {"u800"},
        "32x32": set(common),
    }

    with pytest.raises(
        manifest.SplitContractError,
        match="16x16: expected 800 unique scenes, got 799",
    ):
        manifest.build_scene_split(SyntheticInventory(tmp_path, missing_one))

    different = {
        "8x8": set(common),
        "16x16": set(common),
        "32x32": (set(common) - {"u800"}) | {"u801"},
    }
    with pytest.raises(
        manifest.SplitContractError,
        match="scene sets differ across arrays",
    ):
        manifest.build_scene_split(SyntheticInventory(tmp_path, different))


def test_scene_split_bytes_are_stable_and_existing_file_is_immutable(
    tmp_path: Path,
) -> None:
    manifest = _manifest_module()
    inventory = SyntheticInventory(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    split_a = manifest.load_or_create_scene_split(first, inventory)
    split_b = manifest.load_or_create_scene_split(second, inventory)

    assert split_a == split_b
    assert first.read_bytes() == second.read_bytes()

    original = first.read_bytes()
    first.write_text(
        first.read_text(encoding="utf-8").replace('"seed":42', '"seed":7'),
        encoding="utf-8",
    )
    changed = first.read_bytes()
    assert changed != original
    with pytest.raises(manifest.SplitContractError, match="seed"):
        manifest.load_or_create_scene_split(first, inventory)
    assert first.read_bytes() == changed


@pytest.mark.parametrize("array_name", ["8x8", "16x16", "32x32"])
def test_manifest_has_fixed_counts_beams_paths_and_order(
    tmp_path: Path,
    array_name: str,
) -> None:
    manifest = _manifest_module()
    inventory = SyntheticInventory(tmp_path)
    split = manifest.build_scene_split(inventory)

    records = manifest.build_manifest(
        inventory,
        _schema(),
        split,
        array_name,
    )
    manifest.validate_manifest(
        records,
        inventory,
        _schema(),
        split,
        array_name,
    )

    assert len(records) == 6400
    assert Counter(record.split for record in records) == {
        "train": 4480,
        "val": 640,
        "test": 1280,
    }
    expected_beams = {
        beam.beam_id
        for beam in manifest.ARRAY_SPECS[array_name].beams
    }
    assert {record.beam_id for record in records} == expected_beams
    per_beam_split = Counter(
        (record.beam_id, record.split) for record in records
    )
    for beam_id in expected_beams:
        assert per_beam_split[(beam_id, "train")] == 560
        assert per_beam_split[(beam_id, "val")] == 80
        assert per_beam_split[(beam_id, "test")] == 160
    assert records[0].scene_id == "u1"
    assert records[0].beam_id == min(expected_beams)
    assert records[-1].scene_id == "u800"
    assert len({record.sample_key for record in records}) == 6400
    assert all("\\" not in record.height_path for record in records)
    assert all(record.array_name == array_name for record in records)


def test_all_array_manifests_use_identical_scene_sets(tmp_path: Path) -> None:
    manifest = _manifest_module()
    inventory = SyntheticInventory(tmp_path)
    split = manifest.build_scene_split(inventory)

    scene_sets = {}
    for array_name in ARRAY_CONFIGS:
        records = manifest.build_manifest(
            inventory,
            _schema(),
            split,
            array_name,
        )
        scene_sets[array_name] = {record.scene_id for record in records}

    assert scene_sets["8x8"] == scene_sets["16x16"] == scene_sets["32x32"]


def test_manifest_rejects_duplicate_wrong_beam_and_missing_file(
    tmp_path: Path,
) -> None:
    manifest = _manifest_module()
    inventory = SyntheticInventory(tmp_path)
    schema = _schema()
    split = manifest.build_scene_split(inventory)
    records = manifest.build_manifest(inventory, schema, split, "8x8")

    with pytest.raises(manifest.ManifestContractError, match="duplicate"):
        manifest.validate_manifest(
            (*records, records[0]),
            inventory,
            schema,
            split,
            "8x8",
        )

    wrong_beam = (replace(records[0], beam_id=99), *records[1:])
    with pytest.raises(manifest.ManifestContractError, match="beam"):
        manifest.validate_manifest(
            wrong_beam,
            inventory,
            schema,
            split,
            "8x8",
        )

    first = records[0]
    inventory.missing.add((first.config_id, first.beam_id, first.scene_id))
    with pytest.raises(manifest.MissingSamplePathError, match="missing synthetic"):
        manifest.validate_manifest(
            records,
            inventory,
            schema,
            split,
            "8x8",
        )


def test_manifest_jsonl_is_stable_round_trippable_and_immutable(
    tmp_path: Path,
) -> None:
    manifest = _manifest_module()
    inventory = SyntheticInventory(tmp_path)
    schema = _schema()
    split = manifest.build_scene_split(inventory)
    records = manifest.build_manifest(inventory, schema, split, "8x8")
    first = tmp_path / "manifest_a.jsonl"
    second = tmp_path / "manifest_b.jsonl"

    manifest.write_manifest_jsonl(first, records)
    manifest.write_manifest_jsonl(second, records)

    assert first.read_bytes() == second.read_bytes()
    assert manifest.load_manifest_jsonl(first) == records
    with pytest.raises(manifest.ExistingSchemaMismatchError, match="different"):
        manifest.write_manifest_jsonl(first, records[:-1])


def test_visualization_cases_are_fixed_from_sorted_test_scenes(
    tmp_path: Path,
) -> None:
    manifest = _manifest_module()
    split = manifest.build_scene_split(SyntheticInventory(tmp_path))

    payload = manifest.build_visualization_cases(split)

    sorted_test = manifest.natural_sorted(split.test)
    assert payload["scene_ids"] == [
        sorted_test[0],
        sorted_test[53],
        sorted_test[106],
        sorted_test[159],
    ]
    assert payload["steering_deg"] == [-28.0, 0.0, 21.0]
    assert payload["split"] == "test"
