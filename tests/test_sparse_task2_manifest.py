from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.cross_frequency import select_zero_degree_configurations_same_frequency
from experiments.multiconfig_manifest import (
    ManifestRecord,
    SceneSplit,
    canonical_json_bytes,
    load_manifest_jsonl,
    load_schema_lock,
)
from experiments.provenance import sha256_file
from experiments.sparse_task2_manifest import (
    MANDATORY_SINGLEBEAM_ARRAY_SIZES,
    MANDATORY_SINGLEBEAM_FREQUENCY_HZ,
    MANDATORY_SINGLEBEAM_PROTOCOL,
    MANDATORY_SINGLEBEAM_SAMPLE_COUNT,
    MANDATORY_SINGLEBEAM_SCENE_COUNTS,
    MANDATORY_SINGLEBEAM_STEERING_DEG,
    SingleBeamTask2ManifestError,
    build_singlebeam_task2_manifest,
    validate_singlebeam_task2_manifest,
)


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "experiments" / "multiconfig_schema.json"


def _split_payload() -> dict[str, object]:
    return {
        "seed": 42,
        "algorithm": "scene_disjoint_seed42_fixture",
        "train": [f"u{index}" for index in range(1, 561)],
        "val": [f"u{index}" for index in range(561, 641)],
        "test": [f"u{index}" for index in range(641, 801)],
    }


def _write_split(path: Path) -> SceneSplit:
    path.write_bytes(canonical_json_bytes(_split_payload()))
    return SceneSplit.from_dict(_split_payload())


def _materialize_dataset(root: Path, split: SceneSplit) -> None:
    schema = load_schema_lock(_schema_path())
    selected = select_zero_degree_configurations_same_frequency(
        schema,
        MANDATORY_SINGLEBEAM_FREQUENCY_HZ,
        MANDATORY_SINGLEBEAM_ARRAY_SIZES,
    )
    dataset = root / "raw" / "Dataset"
    scene_ids = (*split.train, *split.val, *split.test)
    for scene_id in scene_ids:
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
        for scene_id in scene_ids:
            (radiomap_dir / f"{scene_id}_labeled_radiomap.npy").write_bytes(b"radio")


@pytest.fixture
def manifest_fixture(tmp_path: Path) -> dict[str, object]:
    split_path = tmp_path / "scene_split_seed42.json"
    split = _write_split(split_path)
    _materialize_dataset(tmp_path, split)
    audits: dict[str, object] = {}
    manifests: dict[str, Path] = {}
    for array_size in MANDATORY_SINGLEBEAM_ARRAY_SIZES:
        manifest_path = tmp_path / f"{array_size}.jsonl"
        audits[array_size] = build_singlebeam_task2_manifest(
            dataset_root=tmp_path,
            split_path=split_path,
            array_size=array_size,
            output_path=manifest_path,
        )
        manifests[array_size] = manifest_path
    return {
        "root": tmp_path,
        "split_path": split_path,
        "split": split,
        "audits": audits,
        "manifests": manifests,
    }


def _rewrite_manifest_and_sidecar(
    manifest_path: Path,
    records: tuple[ManifestRecord, ...],
    split_path: Path,
) -> None:
    manifest_path.write_bytes(
        b"".join(canonical_json_bytes(record.to_dict()) for record in records)
    )
    sidecar_path = manifest_path.with_suffix(".audit.json")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["manifest_sha256"] = sha256_file(manifest_path)
    payload["scene_split_sha256"] = sha256_file(split_path)
    sidecar_path.write_bytes(canonical_json_bytes(payload))


@pytest.mark.parametrize("array_size", MANDATORY_SINGLEBEAM_ARRAY_SIZES)
def test_builds_800_records_with_locked_scene_disjoint_counts(
    manifest_fixture: dict[str, object],
    array_size: str,
) -> None:
    audit = manifest_fixture["audits"][array_size]
    manifest_path = manifest_fixture["manifests"][array_size]
    records = load_manifest_jsonl(manifest_path)

    assert len(records) == 800
    assert audit["records"] == 800
    assert audit["split_counts"] == MANDATORY_SINGLEBEAM_SCENE_COUNTS
    assert {record.scene_id for record in records if record.split == "train"} == set(
        audit["scene_ids_by_split"]["train"]
    )
    assert {record.scene_id for record in records if record.split == "val"} == set(
        audit["scene_ids_by_split"]["val"]
    )
    assert {record.scene_id for record in records if record.split == "test"} == set(
        audit["scene_ids_by_split"]["test"]
    )
    assert validate_singlebeam_task2_manifest(
        manifest_path=manifest_path,
        split_path=manifest_fixture["split_path"],
        array_size=array_size,
    )["array_size"] == array_size


def test_all_arrays_share_the_exact_same_scene_split_and_locked_metadata(
    manifest_fixture: dict[str, object],
) -> None:
    audits = manifest_fixture["audits"]
    first = audits["8x8"]
    for array_size in MANDATORY_SINGLEBEAM_ARRAY_SIZES:
        audit = audits[array_size]
        assert audit["protocol"] == MANDATORY_SINGLEBEAM_PROTOCOL
        assert audit["frequency_hz"] == MANDATORY_SINGLEBEAM_FREQUENCY_HZ
        assert audit["steering_deg"] == MANDATORY_SINGLEBEAM_STEERING_DEG
        assert audit["sample_count"] == MANDATORY_SINGLEBEAM_SAMPLE_COUNT
        assert audit["scene_split_sha256"] == first["scene_split_sha256"]
        assert audit["scene_ids_by_split"] == first["scene_ids_by_split"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("steering_deg", 7.0),
        ("frequency_hz", 4_900_000_000),
        ("split", "test"),
    ),
)
def test_validate_rejects_mutated_record_values(
    manifest_fixture: dict[str, object],
    field: str,
    value: object,
) -> None:
    manifest_path = manifest_fixture["manifests"]["8x8"]
    split_path = manifest_fixture["split_path"]
    original = load_manifest_jsonl(manifest_path)
    mutated = list(original)
    mutated[0] = ManifestRecord(**{**mutated[0].to_dict(), field: value})
    _rewrite_manifest_and_sidecar(manifest_path, tuple(mutated), split_path)

    with pytest.raises(SingleBeamTask2ManifestError):
        validate_singlebeam_task2_manifest(
            manifest_path=manifest_path,
            split_path=split_path,
            array_size="8x8",
        )


def test_validate_rejects_changed_split_bytes_even_if_scene_lists_are_identical(
    manifest_fixture: dict[str, object],
) -> None:
    manifest_path = manifest_fixture["manifests"]["8x8"]
    split_path = manifest_fixture["split_path"]
    payload = _split_payload()
    payload["algorithm"] = "same-scenes-different-bytes"
    split_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(SingleBeamTask2ManifestError):
        validate_singlebeam_task2_manifest(
            manifest_path=manifest_path,
            split_path=split_path,
            array_size="8x8",
        )
