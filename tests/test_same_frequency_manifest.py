from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from experiments.multiconfig_manifest import SceneSplit
from experiments.cross_frequency import (
    build_same_frequency_records,
    select_zero_degree_configurations,
    validate_same_frequency_records,
)


@pytest.fixture
def schema() -> dict[str, object]:
    return {
        "configurations": [
            {
                "configuration_id": "freq_6.7GHz_64TR_8beams_pattern_tr38901",
                "frequency_hz": 6_700_000_000,
                "rows": 8,
                "cols": 8,
                "tx_elements": 64,
                "beam_angles": [-28.0, -21.0, -14.0, -7.0, 0.0, 7.0, 14.0, 21.0],
                "beam_maps": [
                    {"beam_id": index, "steering_deg": -28.0 + 7.0 * index}
                    for index in range(8)
                ],
            },
            {
                "configuration_id": "freq_6.7GHz_256TR_8beams_pattern_tr38901",
                "frequency_hz": 6_700_000_000,
                "rows": 16,
                "cols": 16,
                "tx_elements": 256,
                "beam_angles": tuple(-28.0 + 3.5 * index for index in range(16)),
                "beam_maps": [
                    {"beam_id": index, "steering_deg": -28.0 + 3.5 * index}
                    for index in range(16)
                ],
            },
            {
                "configuration_id": "freq_6.7GHz_1024TR_64beams_pattern_tr38901",
                "frequency_hz": 6_700_000_000,
                "rows": 32,
                "cols": 32,
                "tx_elements": 1024,
                "beam_angles": tuple(-32.0 + index for index in range(64)),
                "beam_maps": [
                    {"beam_id": index, "steering_deg": -32.0 + index}
                    for index in range(64)
                ],
            },
        ]
    }


@pytest.fixture
def split() -> SceneSplit:
    return SceneSplit(
        seed=42,
        algorithm="fixture",
        train=tuple(f"u{index}" for index in range(1, 561)),
        val=tuple(f"u{index}" for index in range(561, 641)),
        test=tuple(f"u{index}" for index in range(641, 801)),
    )


def scene_ids(records, split_name: str) -> set[str]:
    return {record.scene_id for record in records if record.split == split_name}


def _materialize_dataset(
    root: Path,
    split: SceneSplit,
    selected: dict[str, object],
) -> None:
    dataset = root / "raw" / "Dataset"
    for scene_id in (*split.train, *split.val, *split.test):
        height = dataset / "height_maps" / scene_id / f"{scene_id}_height_matrix.npy"
        height.parent.mkdir(parents=True, exist_ok=True)
        height.write_bytes(b"height")
    for selection in selected.values():
        beam_map = (
            dataset
            / "beam_maps"
            / selection.config_id
            / "u0"
            / f"beam_{selection.beam_id:02d}_angle_{selection.steering_deg:.1f}_matrix.npy"
        )
        beam_map.parent.mkdir(parents=True, exist_ok=True)
        beam_map.write_bytes(b"beam")
        radiomap_dir = (
            dataset
            / "radiomaps"
            / f"{selection.config_id}_beam{selection.beam_id:02d}"
        )
        radiomap_dir.mkdir(parents=True, exist_ok=True)
        for scene_id in (*split.train, *split.val, *split.test):
            (radiomap_dir / f"{scene_id}_labeled_radiomap.npy").write_bytes(b"radio")


def test_selects_one_67ghz_zero_degree_config_per_array(schema) -> None:
    selected = select_zero_degree_configurations(
        schema,
        frequency_hz=6_700_000_000,
        array_sizes=("8x8", "16x16", "32x32"),
    )
    assert set(selected) == {"8x8", "16x16", "32x32"}
    assert all(item.frequency_hz == 6_700_000_000 for item in selected.values())
    assert all(item.steering_deg == 0.0 for item in selected.values())
    assert all(item.beam_id >= 0 for item in selected.values())


def test_same_frequency_manifest_is_560_80_160_and_scene_disjoint(
    tmp_path: Path,
    schema,
    split: SceneSplit,
) -> None:
    selected = select_zero_degree_configurations(
        schema,
        frequency_hz=6_700_000_000,
        array_sizes=("8x8",),
    )
    _materialize_dataset(tmp_path, split, selected)
    records = build_same_frequency_records(
        schema=schema,
        split=split,
        selected=selected,
        workspace_root=tmp_path,
        array_size="8x8",
    )
    assert Counter(record.split for record in records) == {
        "train": 560,
        "val": 80,
        "test": 160,
    }
    assert scene_ids(records, "train").isdisjoint(scene_ids(records, "val"))
    assert scene_ids(records, "train").isdisjoint(scene_ids(records, "test"))
    assert scene_ids(records, "val").isdisjoint(scene_ids(records, "test"))
    validate_same_frequency_records(
        records,
        split=split,
        selected=selected,
        array_size="8x8",
        schema=schema,
        workspace_root=tmp_path,
    )
