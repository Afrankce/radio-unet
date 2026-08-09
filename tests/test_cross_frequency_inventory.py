from __future__ import annotations

from pathlib import Path

from experiments.cross_frequency import (
    inventory_cross_frequency_samples,
    select_zero_degree_configurations,
)


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
                "beam_maps": [{"beam_id": 0, "steering_deg": 0.0}],
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


def test_cross_frequency_inventory_indexes_nonbaseline_configurations(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    data = root / "raw" / "Dataset"
    scene = "u1"
    (data / "height_maps" / scene).mkdir(parents=True)
    (data / "height_maps" / scene / f"{scene}_height_matrix.npy").write_bytes(b"height")
    selected = select_zero_degree_configurations(_schema())
    for configuration in selected.values():
        beam_dir = data / "beam_maps" / configuration.config_id / "u0"
        beam_dir.mkdir(parents=True)
        (beam_dir / f"beam_{configuration.beam_id:02d}_angle_0.0_matrix.npy").write_bytes(b"beam")
        radio_dir = data / "radiomaps" / f"{configuration.config_id}_beam{configuration.beam_id:02d}"
        radio_dir.mkdir(parents=True)
        (radio_dir / f"{scene}_labeled_radiomap.npy").write_bytes(b"radio")

    inventory = inventory_cross_frequency_samples(root, selected, scene_ids=(scene,))

    for configuration in selected.values():
        height, beam, radiomap = inventory.require_unique_triplet(
            configuration.config_id,
            configuration.beam_id,
            scene,
        )
        assert height.name == "u1_height_matrix.npy"
        assert beam.name.endswith("angle_0.0_matrix.npy")
        assert radiomap.name.endswith("labeled_radiomap.npy")
