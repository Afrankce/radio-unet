from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from experiments.multiconfig_manifest import SceneSplit
from experiments.cross_frequency import (
    CrossFrequencyManifestError,
    build_cross_frequency_records,
    cross_frequency_spec,
    select_zero_degree_configurations,
    validate_cross_frequency_records,
)


class FakeInventory:
    def __init__(self, root: Path) -> None:
        self.workspace_root = root

    def require_unique_triplet(
        self,
        config_id: str,
        beam_id: int,
        scene_id: str,
    ) -> tuple[Path, Path, Path]:
        root = self.workspace_root
        height = root / "raw" / "Dataset" / "height_maps" / scene_id / (
            f"{scene_id}_height_matrix.npy"
        )
        beam = (
            root
            / "raw"
            / "Dataset"
            / "beam_maps"
            / config_id
            / "u0"
            / f"beam_{beam_id:02d}_angle_0.0_matrix.npy"
        )
        radiomap = (
            root
            / "raw"
            / "Dataset"
            / "radiomaps"
            / f"{config_id}_beam{beam_id:02d}"
            / f"{scene_id}_labeled_radiomap.npy"
        )
        return height, beam, radiomap


def _schema() -> dict[str, object]:
    return {
        "configurations": [
            {
                "configuration_id": "freq_4.9GHz_64TR_1beams_pattern_tr38901",
                "frequency_hz": 4_900_000_000,
                "rows": 8,
                "cols": 8,
                "tx_elements": 64,
                "beam_angles": [0.0],
                "beam_maps": [
                    {"beam_id": 0, "steering_deg": 0.0},
                ],
            },
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
        ]
    }


def _split() -> SceneSplit:
    return SceneSplit(
        seed=42,
        algorithm="fixture",
        train=tuple(f"u{index}" for index in range(1, 561)),
        val=tuple(f"u{index}" for index in range(561, 641)),
        test=tuple(f"u{index}" for index in range(641, 801)),
    )


def test_selects_zero_degree_by_angle_and_preserves_source_beam_ids() -> None:
    selected = select_zero_degree_configurations(_schema())

    assert selected[4_900_000_000].config_id == (
        "freq_4.9GHz_64TR_1beams_pattern_tr38901"
    )
    assert selected[4_900_000_000].beam_id == 0
    assert selected[6_700_000_000].config_id == (
        "freq_6.7GHz_64TR_8beams_pattern_tr38901"
    )
    assert selected[6_700_000_000].beam_id == 4
    assert selected[6_700_000_000].steering_deg == 0.0


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload["configurations"].pop(),
        lambda payload: payload["configurations"].append(
            {
                "configuration_id": "duplicate",
                "frequency_hz": 4_900_000_000,
                "rows": 8,
                "cols": 8,
                "tx_elements": 64,
                "beam_angles": [0.0],
                "beam_maps": [{"beam_id": 0, "steering_deg": 0.0}],
            }
        ),
        lambda payload: payload["configurations"][0].update({"rows": 4}),
        lambda payload: payload["configurations"][0].update(
            {"beam_angles": [7.0]}
        ),
    ],
)
def test_selector_rejects_missing_ambiguous_or_mismatched_configurations(mutator) -> None:
    payload = _schema()
    mutator(payload)

    with pytest.raises(CrossFrequencyManifestError):
        select_zero_degree_configurations(payload)


def test_manifest_has_fixed_split_counts_and_frequency_pairing() -> None:
    inventory = FakeInventory(Path("C:/cross-frequency-fixture"))
    spec = cross_frequency_spec()
    selected = select_zero_degree_configurations(_schema())
    records = build_cross_frequency_records(inventory, _split(), selected, spec)

    assert len(records) == 800
    assert {record.frequency_hz for record in records if record.split in {"train", "val"}} == {
        4_900_000_000
    }
    assert {record.frequency_hz for record in records if record.split == "test"} == {
        6_700_000_000
    }
    assert {record.beam_id for record in records if record.frequency_hz == 4_900_000_000} == {0}
    assert {record.beam_id for record in records if record.frequency_hz == 6_700_000_000} == {4}
    assert {record.steering_deg for record in records} == {0.0}
    assert len({record.sample_key for record in records}) == 800
    assert {record.scene_id for record in records if record.split == "train"}.isdisjoint(
        {record.scene_id for record in records if record.split == "val"}
    )
    assert {record.scene_id for record in records if record.split == "val"}.isdisjoint(
        {record.scene_id for record in records if record.split == "test"}
    )

    validate_cross_frequency_records(
        records,
        _split(),
        selected,
        spec,
        inventory=inventory,
    )


def test_validator_rejects_frequency_target_pairing_or_duplicate_path() -> None:
    inventory = FakeInventory(Path("C:/cross-frequency-fixture"))
    spec = cross_frequency_spec()
    selected = select_zero_degree_configurations(_schema())
    records = list(build_cross_frequency_records(inventory, _split(), selected, spec))

    records[0] = replace(records[0], frequency_hz=6_700_000_000)
    with pytest.raises(CrossFrequencyManifestError):
        validate_cross_frequency_records(
            records,
            _split(),
            selected,
            spec,
            inventory=inventory,
        )

    records = list(build_cross_frequency_records(inventory, _split(), selected, spec))
    records[1] = replace(records[1], radiomap_path=records[0].radiomap_path)
    with pytest.raises(CrossFrequencyManifestError):
        validate_cross_frequency_records(
            records,
            _split(),
            selected,
            spec,
            inventory=inventory,
        )
