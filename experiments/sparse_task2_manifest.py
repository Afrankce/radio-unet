from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from experiments.cross_frequency import (
    CrossFrequencyManifestError,
    build_same_frequency_records,
    inventory_selected_configurations,
    select_zero_degree_configurations_same_frequency,
    validate_same_frequency_records,
)
from experiments.multiconfig_manifest import (
    ManifestRecord,
    SceneSplit,
    _write_immutable_json,
    canonical_json_bytes,
    load_manifest_jsonl,
    load_schema_lock,
    sha256_file,
    write_manifest_jsonl,
)

MANDATORY_SINGLEBEAM_PROTOCOL = "singlebeam_feature5_samples819"
MANDATORY_SINGLEBEAM_FREQUENCY_HZ = 6_700_000_000
MANDATORY_SINGLEBEAM_STEERING_DEG = 0.0
MANDATORY_SINGLEBEAM_SAMPLE_COUNT = 819
MANDATORY_SINGLEBEAM_SCENE_COUNTS = {"train": 560, "val": 80, "test": 160}
MANDATORY_SINGLEBEAM_ARRAY_SIZES = ("8x8", "16x16", "32x32")
_MANDATORY_RECORD_COUNT = sum(MANDATORY_SINGLEBEAM_SCENE_COUNTS.values())

_SCENE_SPLIT_NAME = "scene_split_seed42.json"
_SIDE_CAR_SUFFIX = ".audit.json"
_SCHEMA_PATH = Path(__file__).resolve().parent / "multiconfig_schema.json"


class SingleBeamTask2ManifestError(RuntimeError):
    """Raised when the locked single-beam task-2 manifest contract is violated."""


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise SingleBeamTask2ManifestError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SingleBeamTask2ManifestError(f"{label} must be a JSON object: {path}")
    return payload


def _scene_split_from_path(path: Path) -> SceneSplit:
    return SceneSplit.from_dict(_read_json_object(path, label="scene split"))


def _scene_split_path_expected(manifest_path: Path) -> Path:
    return Path(manifest_path).resolve().with_name(_SCENE_SPLIT_NAME)


def _sidecar_path(manifest_path: Path) -> Path:
    return Path(manifest_path).resolve().with_suffix(_SIDE_CAR_SUFFIX)


def _scene_ids_by_split(records: tuple[ManifestRecord, ...]) -> dict[str, list[str]]:
    return {
        split: [record.scene_id for record in records if record.split == split]
        for split in ("train", "val", "test")
    }


def _split_counts(records: tuple[ManifestRecord, ...]) -> dict[str, int]:
    counts = Counter(record.split for record in records)
    return {split: counts.get(split, 0) for split in ("train", "val", "test")}


def _selection_info(selected: Mapping[str, object], array_size: str) -> dict[str, object]:
    selection = selected[array_size]
    return {
        "config_id": getattr(selection, "config_id"),
        "beam_id": getattr(selection, "beam_id"),
        "frequency_hz": getattr(selection, "frequency_hz"),
        "steering_deg": getattr(selection, "steering_deg"),
    }


def _base_audit(
    *,
    manifest_path: Path,
    split_path: Path,
    dataset_root: Path,
    schema_path: Path,
    array_size: str,
    records: tuple[ManifestRecord, ...],
    selection: Mapping[str, object],
) -> dict[str, object]:
    return {
        "protocol": MANDATORY_SINGLEBEAM_PROTOCOL,
        "array_size": array_size,
        "frequency_hz": MANDATORY_SINGLEBEAM_FREQUENCY_HZ,
        "steering_deg": MANDATORY_SINGLEBEAM_STEERING_DEG,
        "sample_count": MANDATORY_SINGLEBEAM_SAMPLE_COUNT,
        "split_type": "scene_disjoint_single_beam",
        "scene_split_path": str(split_path.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "schema_path": str(schema_path.resolve()),
        "selected_configuration": dict(selection),
        "records": len(records),
        "split_counts": _split_counts(records),
        "scene_ids_by_split": _scene_ids_by_split(records),
        "manifest": str(manifest_path.resolve()),
    }


def build_singlebeam_task2_manifest(
    *,
    dataset_root: Path,
    split_path: Path,
    array_size: str,
    output_path: Path | None = None,
) -> dict[str, object]:
    dataset_root = Path(dataset_root).resolve()
    split_path = Path(split_path).resolve()
    if array_size not in MANDATORY_SINGLEBEAM_ARRAY_SIZES:
        raise SingleBeamTask2ManifestError(f"unsupported array size: {array_size}")
    if not split_path.is_file():
        raise SingleBeamTask2ManifestError(f"scene split is missing: {split_path}")

    schema = load_schema_lock(_SCHEMA_PATH)
    split = _scene_split_from_path(split_path)
    selected = select_zero_degree_configurations_same_frequency(
        schema,
        MANDATORY_SINGLEBEAM_FREQUENCY_HZ,
        (array_size,),
    )
    inventory = inventory_selected_configurations(
        dataset_root,
        selected,
        scene_ids=(*split.train, *split.val, *split.test),
        array_key=f"singlebeam_task2_{array_size}",
    )
    records = build_same_frequency_records(
        schema=schema,
        split=split,
        selected=selected,
        workspace_root=dataset_root,
        array_size=array_size,
        frequency_hz=MANDATORY_SINGLEBEAM_FREQUENCY_HZ,
        steering_deg=MANDATORY_SINGLEBEAM_STEERING_DEG,
    )
    validate_same_frequency_records(
        records,
        split=split,
        selected=selected,
        array_size=array_size,
        frequency_hz=MANDATORY_SINGLEBEAM_FREQUENCY_HZ,
        steering_deg=MANDATORY_SINGLEBEAM_STEERING_DEG,
        schema=schema,
        workspace_root=dataset_root,
    )

    manifest_path = (
        Path(output_path).resolve()
        if output_path is not None
        else split_path.parent
        / f"manifest_{MANDATORY_SINGLEBEAM_PROTOCOL}_{array_size}.jsonl"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest_jsonl(manifest_path, records)
    written = load_manifest_jsonl(manifest_path)

    selection_info = _selection_info(selected, array_size)
    audit = _base_audit(
        manifest_path=manifest_path,
        split_path=split_path,
        dataset_root=inventory.workspace_root,
        schema_path=_SCHEMA_PATH,
        array_size=array_size,
        records=written,
        selection=selection_info,
    )
    audit["manifest_sha256"] = sha256_file(manifest_path)
    audit["scene_split_sha256"] = sha256_file(split_path)

    _write_immutable_json(_sidecar_path(manifest_path), audit)
    return audit


def validate_singlebeam_task2_manifest(
    *,
    manifest_path: Path,
    split_path: Path,
    array_size: str,
) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve()
    split_path = Path(split_path).resolve()
    if array_size not in MANDATORY_SINGLEBEAM_ARRAY_SIZES:
        raise SingleBeamTask2ManifestError(f"unsupported array size: {array_size}")
    sidecar_path = _sidecar_path(manifest_path)
    if not sidecar_path.is_file():
        raise SingleBeamTask2ManifestError(
            f"missing single-beam manifest audit sidecar: {sidecar_path}"
        )

    audit = _read_json_object(sidecar_path, label="single-beam manifest audit")
    expected_manifest_path = Path(str(audit.get("manifest", ""))).resolve()
    expected_split_path = Path(str(audit.get("scene_split_path", ""))).resolve()
    if expected_manifest_path != manifest_path:
        raise SingleBeamTask2ManifestError(
            f"manifest path mismatch: expected {expected_manifest_path}, got {manifest_path}"
        )
    if expected_split_path != split_path:
        raise SingleBeamTask2ManifestError(
            f"scene split path mismatch: expected {expected_split_path}, got {split_path}"
        )
    if audit.get("protocol") != MANDATORY_SINGLEBEAM_PROTOCOL:
        raise SingleBeamTask2ManifestError("protocol mismatch")
    if audit.get("array_size") != array_size:
        raise SingleBeamTask2ManifestError("array size mismatch")
    if audit.get("frequency_hz") != MANDATORY_SINGLEBEAM_FREQUENCY_HZ:
        raise SingleBeamTask2ManifestError("frequency mismatch")
    if audit.get("steering_deg") != MANDATORY_SINGLEBEAM_STEERING_DEG:
        raise SingleBeamTask2ManifestError("steering mismatch")
    if audit.get("sample_count") != MANDATORY_SINGLEBEAM_SAMPLE_COUNT:
        raise SingleBeamTask2ManifestError("sample-count mismatch")
    if audit.get("split_type") != "scene_disjoint_single_beam":
        raise SingleBeamTask2ManifestError("split type mismatch")

    if sha256_file(manifest_path) != audit.get("manifest_sha256"):
        raise SingleBeamTask2ManifestError("manifest hash mismatch")
    if sha256_file(split_path) != audit.get("scene_split_sha256"):
        raise SingleBeamTask2ManifestError("scene split hash mismatch")

    schema_path = Path(str(audit.get("schema_path", "")))
    dataset_root = Path(str(audit.get("dataset_root", "")))
    if not schema_path.is_file():
        raise SingleBeamTask2ManifestError(f"schema lock is missing: {schema_path}")
    schema = load_schema_lock(schema_path)
    split = _scene_split_from_path(split_path)
    selected = select_zero_degree_configurations_same_frequency(
        schema,
        MANDATORY_SINGLEBEAM_FREQUENCY_HZ,
        (array_size,),
    )
    records = load_manifest_jsonl(manifest_path)
    if len(records) != _MANDATORY_RECORD_COUNT:
        raise SingleBeamTask2ManifestError(
            f"record count mismatch: expected {_MANDATORY_RECORD_COUNT}, got {len(records)}"
        )
    if _split_counts(records) != dict(audit.get("split_counts", {})):
        raise SingleBeamTask2ManifestError("split counts mismatch")
    if _scene_ids_by_split(records) != dict(audit.get("scene_ids_by_split", {})):
        raise SingleBeamTask2ManifestError("scene IDs by split mismatch")

    try:
        validate_same_frequency_records(
            records,
            split=split,
            selected=selected,
            array_size=array_size,
            frequency_hz=MANDATORY_SINGLEBEAM_FREQUENCY_HZ,
            steering_deg=MANDATORY_SINGLEBEAM_STEERING_DEG,
            schema=schema,
            workspace_root=dataset_root,
        )
    except CrossFrequencyManifestError as error:
        raise SingleBeamTask2ManifestError(str(error)) from error
    return audit
