from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.provenance import (
    DATASET_REVISION,
    REFERENCE_CODE_REVISION,
    sha256_file,
)


FIXTURES = Path(__file__).parent / "fixtures" / "multiconfig_config"
COMMON_ANGLES = (-28.0, -21.0, -14.0, -7.0, 0.0, 7.0, 14.0, 21.0)


def _manifest_module():
    from experiments import multiconfig_manifest

    return multiconfig_manifest


@pytest.mark.parametrize(
    ("array_name", "expected_beam_ids"),
    [
        ("8x8", (0, 1, 2, 3, 4, 5, 6, 7)),
        ("16x16", (0, 2, 4, 6, 8, 10, 12, 14)),
        ("32x32", (4, 11, 18, 25, 32, 39, 46, 53)),
    ],
)
def test_array_specs_lock_common_angles_and_released_beam_ids(
    array_name: str,
    expected_beam_ids: tuple[int, ...],
) -> None:
    manifest = _manifest_module()

    spec = manifest.ARRAY_SPECS[array_name]

    assert tuple(beam.beam_id for beam in spec.beams) == expected_beam_ids
    assert tuple(beam.steering_deg for beam in spec.beams) == COMMON_ANGLES
    assert spec.frequency_hz == 6_700_000_000


@pytest.mark.parametrize("array_name", ["8x8", "16x16", "32x32"])
def test_byte_faithful_settings_resolve_the_common_beams(
    array_name: str,
) -> None:
    manifest = _manifest_module()
    fixture = FIXTURES / f"beam_settings_{array_name}.txt"
    parsed = manifest.parse_configuration_text(
        fixture.read_text(encoding="utf-8"),
        source_path=fixture.name,
    )

    manifest.validate_config_against_spec(
        parsed,
        manifest.ARRAY_SPECS[array_name],
    )
    selected = manifest.resolve_selected_beams(
        parsed,
        manifest.ARRAY_SPECS[array_name],
    )

    assert selected == manifest.ARRAY_SPECS[array_name].beams
    assert parsed.beam_angles[0] == parsed.start_angle_deg
    assert parsed.beam_angles[-1] == parsed.end_angle_deg


def test_same_element_count_with_wrong_shape_is_rejected() -> None:
    manifest = _manifest_module()
    text = (FIXTURES / "beam_settings_8x8.txt").read_text(encoding="utf-8")
    text = text.replace("  num_rows: 8\n", "  num_rows: 4\n")
    text = text.replace("  num_cols: 8\n", "  num_cols: 16\n")
    parsed = manifest.parse_configuration_text(
        text,
        source_path="synthetic_4x16.txt",
    )

    with pytest.raises(
        manifest.ConfigurationMismatchError,
        match="expected 8x8, got 4x16",
    ):
        manifest.validate_config_against_spec(
            parsed,
            manifest.ARRAY_SPECS["8x8"],
        )


def test_parser_retains_unknown_fields_and_literal_raw_values() -> None:
    manifest = _manifest_module()
    fixture = FIXTURES / "beam_settings_8x8.txt"

    parsed = manifest.parse_configuration_text(
        fixture.read_text(encoding="utf-8"),
        source_path=fixture.name,
    )
    fields = {field.path: field for field in parsed.fields}

    assert fields["transmitter_array.spacing"].raw_value == " 0.5λ"
    assert fields["beam_configuration.coverage_deg"].raw_value == " 49"
    assert fields["transmitter.position"].raw_value == " [0, 0, 40]"


def test_audit_retains_real_relative_paths_hashes_and_fields(
    tmp_path: Path,
) -> None:
    manifest = _manifest_module()
    fixture = FIXTURES / "beam_settings_8x8.txt"
    target = tmp_path / "raw" / "Dataset" / "configs" / "example.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(fixture.read_bytes())

    report = manifest.audit_config_files(tmp_path)

    assert report.data_root == "raw/Dataset"
    assert len(report.text_files) == 1
    audited = report.text_files[0]
    assert audited.relative_path == "raw/Dataset/configs/example.txt"
    assert audited.sha256 == sha256_file(target)
    assert any(
        field.path == "transmitter_array.num_rows"
        and field.raw_value == " 8"
        for field in audited.fields
    )


def _write_released_configuration_fixture(
    workspace_root: Path,
    *,
    array_name: str,
    configuration_id: str,
) -> None:
    manifest = _manifest_module()
    fixture = FIXTURES / f"beam_settings_{array_name}.txt"
    text = fixture.read_text(encoding="utf-8")
    parsed = manifest.parse_configuration_text(
        text,
        source_path=fixture.name,
    )
    data_root = workspace_root / "raw" / "Dataset"
    aggregate = (
        data_root
        / "beam_maps"
        / configuration_id
        / "u0"
        / "beam_settings.txt"
    )
    aggregate.parent.mkdir(parents=True)
    aggregate.write_text(text, encoding="utf-8")
    for beam_id, angle in enumerate(parsed.beam_angles):
        config = (
            data_root
            / "configs"
            / (
                f"{configuration_id}_beam{beam_id:02d}"
                "_beam_settings.txt"
            )
        )
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(text, encoding="utf-8")
        radiomap_dir = (
            data_root
            / "radiomaps"
            / f"{configuration_id}_beam{beam_id:02d}"
        )
        radiomap_dir.mkdir(parents=True)
        (radiomap_dir / "beam_settings.txt").write_text(
            text,
            encoding="utf-8",
        )
        beam_map = aggregate.parent / (
            f"beam_{beam_id:02d}_angle_{angle:.1f}_matrix.npy"
        )
        beam_map.touch()


def test_audit_builds_complete_released_beam_inventory(tmp_path: Path) -> None:
    manifest = _manifest_module()
    configuration_id = "freq_6.7GHz_64TR_8beams_pattern_tr38901"
    _write_released_configuration_fixture(
        tmp_path,
        array_name="8x8",
        configuration_id=configuration_id,
    )

    report = manifest.audit_config_files(tmp_path)

    assert len(report.configurations) == 1
    configuration = report.configurations[0]
    assert configuration.configuration_id == configuration_id
    assert (configuration.rows, configuration.cols) == (8, 8)
    assert len(configuration.config_files) == 8
    assert len(configuration.radiomap_beam_settings) == 8
    assert tuple(item.beam_id for item in configuration.beam_maps) == tuple(
        range(8)
    )
    assert tuple(item.steering_deg for item in configuration.beam_maps) == (
        COMMON_ANGLES
    )


def test_audit_rejects_incomplete_released_beam_inventory(
    tmp_path: Path,
) -> None:
    manifest = _manifest_module()
    configuration_id = "freq_6.7GHz_64TR_8beams_pattern_tr38901"
    _write_released_configuration_fixture(
        tmp_path,
        array_name="8x8",
        configuration_id=configuration_id,
    )
    missing = (
        tmp_path
        / "raw"
        / "Dataset"
        / "configs"
        / f"{configuration_id}_beam07_beam_settings.txt"
    )
    missing.unlink()

    with pytest.raises(manifest.BeamInventoryMismatchError, match="config files"):
        manifest.audit_config_files(tmp_path)


def test_benchmark_configuration_ids_are_derived_from_real_fields(
    tmp_path: Path,
) -> None:
    manifest = _manifest_module()
    expected = {
        "8x8": "released-square-64",
        "16x16": "released-square-256",
        "32x32": "released-square-1024",
    }
    for array_name, configuration_id in expected.items():
        _write_released_configuration_fixture(
            tmp_path,
            array_name=array_name,
            configuration_id=configuration_id,
        )

    report = manifest.audit_config_files(tmp_path)
    selected = manifest.select_benchmark_configurations(report)

    assert {
        name: configuration.configuration_id
        for name, configuration in selected.items()
    } == expected


def test_schema_payload_freezes_paths_shapes_tx_and_sentinels(
    tmp_path: Path,
) -> None:
    manifest = _manifest_module()
    for array_name, configuration_id in {
        "8x8": "released-square-64",
        "16x16": "released-square-256",
        "32x32": "released-square-1024",
    }.items():
        _write_released_configuration_fixture(
            tmp_path,
            array_name=array_name,
            configuration_id=configuration_id,
        )
    report = manifest.audit_config_files(tmp_path)
    report = replace(
        report,
        identities={
            "dataset_revision": DATASET_REVISION,
            "reference_code_revision": REFERENCE_CODE_REVISION,
            "archive_sha256": "a" * 64,
        },
    )
    source_metadata = {
        "height": {"shape": [256, 256], "dtype": "float32", "files": 800},
        "beam_map": {"shape": [128, 128], "dtype": "float64", "files": 24},
        "radiomap": {"shape": [128, 128], "dtype": "float32", "files": 19200},
    }

    payload = manifest.assemble_schema_payload(
        report,
        audit_report_relative_path="config_audit.json",
        audit_report_sha256="b" * 64,
        source_metadata=source_metadata,
        scene_count=800,
    )

    assert payload["data_root"] == "raw/Dataset"
    assert payload["path_rules"] == {
        "height": (
            "raw/Dataset/height_maps/{scene_id}/"
            "{scene_id}_height_matrix.npy"
        ),
        "beam_map": (
            "raw/Dataset/beam_maps/{configuration_id}/u0/"
            "beam_{beam_id:02d}_angle_{steering_deg:.1f}_matrix.npy"
        ),
        "radiomap": (
            "raw/Dataset/radiomaps/{configuration_id}_beam{beam_id:02d}/"
            "{scene_id}_labeled_radiomap.npy"
        ),
    }
    assert payload["source_metadata"] == source_metadata
    assert payload["transmitter"]["output_pixel_rc"] == [127, 127]
    assert payload["target_domain"] == {
        "floor_db": -300.0,
        "building_sentinel": 1000.0,
        "valid_lower_exclusive_db": -300.0,
        "valid_upper_exclusive_db": 0.0,
    }
    assert [item["configuration_id"] for item in payload["arrays"]] == [
        "released-square-64",
        "released-square-256",
        "released-square-1024",
    ]
    assert payload["identities"]["audit_report_sha256"] == "b" * 64


def test_canonical_payload_equality_survives_json_tuple_round_trip(
    tmp_path: Path,
) -> None:
    manifest = _manifest_module()
    _write_released_configuration_fixture(
        tmp_path,
        array_name="8x8",
        configuration_id="released-square-64",
    )
    payload = manifest.audit_config_files(tmp_path).to_dict()
    round_tripped = json.loads(
        manifest.canonical_json_bytes(payload).decode("utf-8")
    )

    assert manifest.canonical_payloads_equal(payload, round_tripped)


def _write_minimal_schema(path: Path, configuration_ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data_root": "raw/Dataset",
                "identities": {
                    "dataset_revision": DATASET_REVISION,
                    "reference_code_revision": REFERENCE_CODE_REVISION,
                },
                "configurations": [
                    {"configuration_id": value}
                    for value in configuration_ids
                ],
                "arrays": [],
            }
        ),
        encoding="utf-8",
    )


def test_schema_lock_rejects_wrong_source_revision(tmp_path: Path) -> None:
    manifest = _manifest_module()
    schema_path = tmp_path / "schema.json"
    _write_minimal_schema(schema_path, ["config-a"])
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    payload["identities"]["dataset_revision"] = "moving-main"
    schema_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(manifest.SchemaIdentityError, match="dataset revision"):
        manifest.load_schema_lock(schema_path)


def test_schema_lock_rejects_duplicate_configuration_ids(
    tmp_path: Path,
) -> None:
    manifest = _manifest_module()
    schema_path = tmp_path / "schema.json"
    _write_minimal_schema(schema_path, ["duplicate", "duplicate"])

    with pytest.raises(
        manifest.DuplicateConfigurationError,
        match="duplicate",
    ):
        manifest.load_schema_lock(schema_path)


def test_sample_path_resolution_requires_exactly_one_file_per_key(
    tmp_path: Path,
) -> None:
    manifest = _manifest_module()
    relative_paths = [
        "raw/Dataset/height_maps/u1/u1_height_matrix.npy",
        (
            "raw/Dataset/beam_maps/config-a/u0/"
            "beam_00_angle_-28.0_matrix.npy"
        ),
        (
            "raw/Dataset/radiomaps/config-a_beam00/"
            "u1_labeled_radiomap.npy"
        ),
        (
            "raw/Dataset/radiomaps/config-a_beam00/"
            "duplicate_u1_labeled_radiomap.npy"
        ),
    ]
    for relative_path in relative_paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    inventory = manifest.SampleInventory(
        workspace_root=tmp_path,
        height_paths={"u1": (relative_paths[0],)},
        beam_map_paths={("config-a", 0): (relative_paths[1],)},
        radiomap_paths={("config-a", 0, "u1"): (relative_paths[2],)},
    )
    height, beam_map, radiomap = manifest.resolve_sample_paths(
        inventory,
        "config-a",
        0,
        "u1",
    )
    assert (height, beam_map, radiomap) == tuple(
        tmp_path / path for path in relative_paths[:3]
    )

    ambiguous = manifest.SampleInventory(
        workspace_root=tmp_path,
        height_paths=inventory.height_paths,
        beam_map_paths=inventory.beam_map_paths,
        radiomap_paths={
            ("config-a", 0, "u1"): (
                relative_paths[2],
                relative_paths[3],
            )
        },
    )
    with pytest.raises(manifest.AmbiguousSamplePathError, match="radiomap"):
        manifest.resolve_sample_paths(ambiguous, "config-a", 0, "u1")


def test_inventory_samples_indexes_released_layout_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    manifest = _manifest_module()
    data_root = tmp_path / "raw" / "Dataset"
    height = data_root / "height_maps" / "u1" / "u1_height_matrix.npy"
    beam_map = (
        data_root
        / "beam_maps"
        / "config-a"
        / "u0"
        / "beam_00_angle_-28.0_matrix.npy"
    )
    radiomap = (
        data_root
        / "radiomaps"
        / "config-a_beam00"
        / "u1_labeled_radiomap.npy"
    )
    for path in (height, beam_map, radiomap):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    schema_path = tmp_path / "schema.json"
    _write_minimal_schema(schema_path, ["config-a"])
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    payload["arrays"] = [
        {
            "name": "8x8",
            "configuration_id": "config-a",
            "selected_beams": [
                {"beam_id": 0, "steering_deg": -28.0},
            ],
        }
    ]
    schema_path.write_text(json.dumps(payload), encoding="utf-8")
    schema = manifest.load_schema_lock(schema_path)

    inventory = manifest.inventory_samples(tmp_path, schema)

    assert manifest.resolve_sample_paths(
        inventory,
        "config-a",
        0,
        "u1",
    ) == (height, beam_map, radiomap)

    duplicate = beam_map.with_name("beam_00_angle_-27.9_matrix.npy")
    duplicate.touch()
    with pytest.raises(manifest.AmbiguousSamplePathError, match="beam map"):
        manifest.inventory_samples(tmp_path, schema)
