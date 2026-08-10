from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from experiments.multiconfig_manifest import (
    ARRAY_SPECS,
    ManifestRecord,
    MissingSamplePathError,
    SampleInventory,
    SceneSplit,
    load_manifest_jsonl,
    load_schema_lock,
    write_manifest_jsonl,
)
from experiments.provenance import sha256_file


TRAIN_FREQUENCY_HZ = 4_900_000_000
VAL_FREQUENCY_HZ = TRAIN_FREQUENCY_HZ
TEST_FREQUENCY_HZ = 6_700_000_000
ZERO_DEGREE = 0.0


class CrossFrequencyManifestError(RuntimeError):
    """The cross-frequency manifest violates its locked protocol."""


@dataclass(frozen=True)
class CrossFrequencySpec:
    array_name: str = "8x8"
    array_rows: int = 8
    array_cols: int = 8
    tx_elements: int = 64
    train_frequency_hz: int = TRAIN_FREQUENCY_HZ
    val_frequency_hz: int = VAL_FREQUENCY_HZ
    test_frequency_hz: int = TEST_FREQUENCY_HZ
    steering_deg: float = ZERO_DEGREE
    train_samples: int = 560
    val_samples: int = 80
    test_samples: int = 160
    seed: int = 42

    @property
    def total_samples(self) -> int:
        return self.train_samples + self.val_samples + self.test_samples


@dataclass(frozen=True)
class SelectedZeroDegreeConfiguration:
    frequency_hz: int
    config_id: str
    rows: int
    cols: int
    tx_elements: int
    beam_id: int
    steering_deg: float


def cross_frequency_spec() -> CrossFrequencySpec:
    return CrossFrequencySpec()


def _configurations(schema: Mapping[str, Any] | Any) -> tuple[Mapping[str, Any], ...]:
    if hasattr(schema, "configurations"):
        values = getattr(schema, "configurations")
    elif isinstance(schema, Mapping):
        values = schema.get("configurations")
    else:
        values = None
    if not isinstance(values, (list, tuple)):
        raise CrossFrequencyManifestError("schema configurations must be a list")
    result: list[Mapping[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise CrossFrequencyManifestError("schema configuration must be an object")
        result.append(value)
    return tuple(result)


def _beam_entries(configuration: Mapping[str, Any]) -> tuple[tuple[int, float], ...]:
    raw_maps = configuration.get("beam_maps")
    entries: list[tuple[int, float]] = []
    if isinstance(raw_maps, (list, tuple)):
        for item in raw_maps:
            if not isinstance(item, Mapping):
                raise CrossFrequencyManifestError("beam_maps entries must be objects")
            try:
                beam_id = int(item["beam_id"])
                angle = float(item["steering_deg"])
            except (KeyError, TypeError, ValueError) as error:
                raise CrossFrequencyManifestError(
                    "beam_maps entry lacks beam_id or steering_deg"
                ) from error
            entries.append((beam_id, angle))
    else:
        angles = configuration.get("beam_angles")
        if not isinstance(angles, (list, tuple)):
            raise CrossFrequencyManifestError(
                "configuration must contain beam_maps or beam_angles"
            )
        for beam_id, raw_angle in enumerate(angles):
            try:
                entries.append((beam_id, float(raw_angle)))
            except (TypeError, ValueError) as error:
                raise CrossFrequencyManifestError("beam angle is not numeric") from error
    if not entries or any(not math.isfinite(angle) for _beam_id, angle in entries):
        raise CrossFrequencyManifestError("configuration has no finite beam angles")
    if len({beam_id for beam_id, _angle in entries}) != len(entries):
        raise CrossFrequencyManifestError("configuration contains duplicate beam IDs")
    declared_angles = configuration.get("beam_angles")
    if isinstance(declared_angles, (list, tuple)):
        try:
            normalized_declared = tuple(float(value) for value in declared_angles)
        except (TypeError, ValueError) as error:
            raise CrossFrequencyManifestError("declared beam angles are not numeric") from error
        actual_angles = tuple(angle for _beam_id, angle in entries)
        if len(normalized_declared) != len(actual_angles) or any(
            not math.isclose(actual, declared, abs_tol=1e-9)
            for actual, declared in zip(actual_angles, normalized_declared)
        ):
            raise CrossFrequencyManifestError(
                "beam_maps and declared beam_angles disagree"
            )
    return tuple(entries)


def _select_one(
    configurations: tuple[Mapping[str, Any], ...],
    *,
    frequency_hz: int,
    spec: CrossFrequencySpec,
) -> SelectedZeroDegreeConfiguration:
    matches: list[SelectedZeroDegreeConfiguration] = []
    for configuration in configurations:
        try:
            config_id = str(configuration["configuration_id"])
            actual_frequency = int(configuration["frequency_hz"])
            rows = int(configuration["rows"])
            cols = int(configuration["cols"])
            tx_elements = int(configuration["tx_elements"])
        except (KeyError, TypeError, ValueError) as error:
            raise CrossFrequencyManifestError(
                "configuration lacks required identity fields"
            ) from error
        if actual_frequency != frequency_hz:
            continue
        if (rows, cols, tx_elements) != (
            spec.array_rows,
            spec.array_cols,
            spec.tx_elements,
        ):
            continue
        for beam_id, angle in _beam_entries(configuration):
            if math.isclose(angle, spec.steering_deg, abs_tol=1e-9):
                matches.append(
                    SelectedZeroDegreeConfiguration(
                        frequency_hz=actual_frequency,
                        config_id=config_id,
                        rows=rows,
                        cols=cols,
                        tx_elements=tx_elements,
                        beam_id=beam_id,
                        steering_deg=angle,
                    )
                )
    if len(matches) != 1:
        raise CrossFrequencyManifestError(
            f"expected exactly one {frequency_hz} Hz 8x8/64TR zero-degree "
            f"configuration, found {len(matches)}"
        )
    return matches[0]


def _select_one_for_array(
    configurations: tuple[Mapping[str, Any], ...],
    *,
    frequency_hz: int,
    array_size: str,
    steering_deg: float,
) -> SelectedZeroDegreeConfiguration:
    try:
        array_spec = ARRAY_SPECS[array_size]
    except KeyError as error:
        raise CrossFrequencyManifestError(f"unsupported array size: {array_size}") from error
    matches: list[SelectedZeroDegreeConfiguration] = []
    candidates: list[str] = []
    for configuration in configurations:
        try:
            config_id = str(configuration["configuration_id"])
            actual_frequency = int(configuration["frequency_hz"])
            rows = int(configuration["rows"])
            cols = int(configuration["cols"])
            tx_elements = int(configuration["tx_elements"])
        except (KeyError, TypeError, ValueError) as error:
            raise CrossFrequencyManifestError(
                "configuration lacks required identity fields"
            ) from error
        if actual_frequency != frequency_hz:
            continue
        if (rows, cols, tx_elements) != (
            array_spec.rows,
            array_spec.cols,
            array_spec.tx_elements,
        ):
            continue
        for beam_id, angle in _beam_entries(configuration):
            candidates.append(f"{config_id}:beam{beam_id:02d}@{angle:.6f}")
            if math.isclose(angle, steering_deg, abs_tol=1e-9):
                matches.append(
                    SelectedZeroDegreeConfiguration(
                        frequency_hz=actual_frequency,
                        config_id=config_id,
                        rows=rows,
                        cols=cols,
                        tx_elements=tx_elements,
                        beam_id=beam_id,
                        steering_deg=angle,
                    )
                )
    if len(matches) != 1:
        raise CrossFrequencyManifestError(
            "expected exactly one "
            f"{frequency_hz} Hz {array_size}/{array_spec.tx_elements}TR "
            f"{steering_deg:.1f}° configuration, found {len(matches)}; "
            f"candidates={candidates}"
        )
    return matches[0]


def _select_cross_frequency_configurations(
    schema: Mapping[str, Any] | Any,
    spec: CrossFrequencySpec | None = None,
) -> dict[int, SelectedZeroDegreeConfiguration]:
    spec = spec or cross_frequency_spec()
    configurations = _configurations(schema)
    selected = {
        frequency_hz: _select_one(
            configurations,
            frequency_hz=frequency_hz,
            spec=spec,
        )
        for frequency_hz in (spec.train_frequency_hz, spec.test_frequency_hz)
    }
    if selected[spec.train_frequency_hz].config_id == selected[spec.test_frequency_hz].config_id:
        raise CrossFrequencyManifestError(
            "train and test frequencies must use distinct released configurations"
        )
    return selected


def select_zero_degree_configurations(
    schema: Mapping[str, Any] | Any,
    spec: CrossFrequencySpec | None = None,
    *,
    frequency_hz: int | None = None,
    array_sizes: Sequence[str] | None = None,
) -> dict[Any, SelectedZeroDegreeConfiguration]:
    if frequency_hz is not None or array_sizes is not None:
        if frequency_hz is None or array_sizes is None:
            raise CrossFrequencyManifestError(
                "same-frequency selection requires both frequency_hz and array_sizes"
            )
        return select_zero_degree_configurations_for_array_sizes(
            schema,
            frequency_hz=frequency_hz,
            array_sizes=array_sizes,
            steering_deg=ZERO_DEGREE,
        )
    return _select_cross_frequency_configurations(schema, spec)


def inventory_cross_frequency_samples(
    workspace_root: Path,
    selected: Mapping[int, SelectedZeroDegreeConfiguration],
    *,
    scene_ids: Sequence[str] | None = None,
) -> SampleInventory:
    """Index only the two released configurations used by this experiment."""

    return inventory_selected_configurations(
        workspace_root,
        selected,
        scene_ids=scene_ids,
        array_key="cross_frequency",
    )


def inventory_selected_configurations(
    workspace_root: Path,
    selected: Mapping[object, SelectedZeroDegreeConfiguration],
    *,
    scene_ids: Sequence[str] | None = None,
    array_key: str = "selected_configurations",
) -> SampleInventory:
    workspace_root = Path(workspace_root).resolve()
    data_root = workspace_root / "raw" / "Dataset"
    if not data_root.is_dir():
        raise CrossFrequencyManifestError(f"cross-frequency data root is missing: {data_root}")
    if scene_ids is None:
        discovered = sorted(
            path.parent.name
            for path in data_root.glob("height_maps/u*/u*_height_matrix.npy")
            if path.is_file()
        )
        scene_ids = tuple(dict.fromkeys(discovered))
    scenes = tuple(scene_ids)
    if not scenes:
        raise CrossFrequencyManifestError("cross-frequency inventory has no scenes")

    def relative(path: Path) -> str:
        return _relative_path(path, workspace_root)

    height_paths: dict[str, tuple[str, ...]] = {}
    for scene_id in scenes:
        candidates = tuple(
            path
            for path in (
                data_root / "height_maps" / scene_id / f"{scene_id}_height_matrix.npy",
            )
            if path.is_file()
        )
        if len(candidates) != 1:
            raise MissingSamplePathError(
                f"expected one height map for {scene_id}, got {len(candidates)}"
            )
        height_paths[scene_id] = (relative(candidates[0]),)

    beam_map_paths: dict[tuple[str, int], tuple[str, ...]] = {}
    radiomap_paths: dict[tuple[str, int, str], tuple[str, ...]] = {}
    array_keys: list[tuple[str, int]] = []
    for configuration in selected.values():
        key = (configuration.config_id, configuration.beam_id)
        if key in array_keys:
            continue
        array_keys.append(key)
        beam_directory = data_root / "beam_maps" / configuration.config_id / "u0"
        beam_candidates = tuple(
            sorted(
                path
                for path in beam_directory.glob(
                    f"beam_{configuration.beam_id:02d}_angle_*_matrix.npy"
                )
                if path.is_file()
            )
        )
        if len(beam_candidates) != 1:
            raise MissingSamplePathError(
                f"expected one beam map for {key}, got {len(beam_candidates)}"
            )
        beam_map_paths[key] = (relative(beam_candidates[0]),)
        radiomap_directory = data_root / "radiomaps" / (
            f"{configuration.config_id}_beam{configuration.beam_id:02d}"
        )
        for scene_id in scenes:
            candidates = tuple(
                path
                for path in (
                    radiomap_directory / f"{scene_id}_labeled_radiomap.npy",
                )
                if path.is_file()
            )
            if len(candidates) != 1:
                raise MissingSamplePathError(
                    f"expected one radiomap for {configuration.config_id}/"
                    f"beam{configuration.beam_id:02d}/{scene_id}, got {len(candidates)}"
                )
            radiomap_paths[(configuration.config_id, configuration.beam_id, scene_id)] = (
                relative(candidates[0]),
            )
    return SampleInventory(
        workspace_root=workspace_root,
        height_paths=height_paths,
        beam_map_paths=beam_map_paths,
        radiomap_paths=radiomap_paths,
        array_keys={array_key: tuple(array_keys)},
    )


def _split_lookup(split: SceneSplit) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name, scenes in (("train", split.train), ("val", split.val), ("test", split.test)):
        for scene_id in scenes:
            if scene_id in lookup:
                raise CrossFrequencyManifestError(
                    f"scene appears in multiple splits: {scene_id}"
                )
            lookup[scene_id] = name
    if (len(split.train), len(split.val), len(split.test)) != (560, 80, 160):
        raise CrossFrequencyManifestError(
            "scene split must contain exactly 560/80/160 scenes"
        )
    return lookup


def _relative_path(path: Path, workspace_root: Path) -> str:
    try:
        relative = Path(path).resolve().relative_to(workspace_root)
    except ValueError as error:
        raise CrossFrequencyManifestError(
            f"sample path escapes workspace root: {path}"
        ) from error
    value = relative.as_posix()
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise CrossFrequencyManifestError(f"unsafe sample path: {value}")
    return value


def build_cross_frequency_records(
    inventory: Any,
    split: SceneSplit,
    selected: Mapping[int, SelectedZeroDegreeConfiguration],
    spec: CrossFrequencySpec | None = None,
) -> tuple[ManifestRecord, ...]:
    spec = spec or cross_frequency_spec()
    lookup = _split_lookup(split)
    for frequency_hz in (spec.train_frequency_hz, spec.test_frequency_hz):
        if frequency_hz not in selected:
            raise CrossFrequencyManifestError(
                f"selected configuration missing frequency {frequency_hz}"
            )
    workspace_root = Path(inventory.workspace_root).resolve()
    records: list[ManifestRecord] = []
    for scene_id in sorted(lookup, key=lambda value: (value.rstrip("0123456789"), int(value.lstrip("u")))):
        split_name = lookup[scene_id]
        frequency_hz = (
            spec.test_frequency_hz
            if split_name == "test"
            else spec.train_frequency_hz
        )
        configuration = selected[frequency_hz]
        height, beam_map, radiomap = inventory.require_unique_triplet(
            configuration.config_id,
            configuration.beam_id,
            scene_id,
        )
        records.append(
            ManifestRecord(
                sample_key=(
                    f"{scene_id}|{spec.array_name}|freq{frequency_hz}"
                    f"|angle{spec.steering_deg:.1f}|beam{configuration.beam_id:02d}"
                ),
                split=split_name,
                scene_id=scene_id,
                array_name=spec.array_name,
                array_rows=configuration.rows,
                array_cols=configuration.cols,
                frequency_hz=frequency_hz,
                config_id=configuration.config_id,
                beam_id=configuration.beam_id,
                steering_deg=configuration.steering_deg,
                height_path=_relative_path(height, workspace_root),
                beam_map_path=_relative_path(beam_map, workspace_root),
                radiomap_path=_relative_path(radiomap, workspace_root),
            )
        )
    return tuple(records)


def validate_cross_frequency_records(
    records: Any,
    split: SceneSplit,
    selected: Mapping[int, SelectedZeroDegreeConfiguration],
    spec: CrossFrequencySpec | None = None,
    *,
    inventory: Any | None = None,
) -> None:
    spec = spec or cross_frequency_spec()
    values = tuple(records)
    if len(values) != spec.total_samples:
        raise CrossFrequencyManifestError(
            f"expected {spec.total_samples} records, got {len(values)}"
        )
    if len({record.sample_key for record in values}) != len(values):
        raise CrossFrequencyManifestError("sample keys are not unique")
    logical_keys = [(record.scene_id, record.frequency_hz) for record in values]
    if len(set(logical_keys)) != len(logical_keys):
        raise CrossFrequencyManifestError("logical scene/frequency keys are not unique")
    radiomap_paths = [record.radiomap_path for record in values]
    if len(set(radiomap_paths)) != len(radiomap_paths):
        raise CrossFrequencyManifestError("radiomap paths are not unique")
    lookup = _split_lookup(split)
    for record in values:
        if not isinstance(record, ManifestRecord):
            raise CrossFrequencyManifestError("all records must be ManifestRecord values")
        expected_split = lookup.get(record.scene_id)
        if expected_split != record.split:
            raise CrossFrequencyManifestError(
                f"record split mismatch for {record.scene_id}: {record.split}"
            )
        expected_frequency = (
            spec.test_frequency_hz
            if record.split == "test"
            else spec.train_frequency_hz
        )
        configuration = selected.get(expected_frequency)
        if configuration is None:
            raise CrossFrequencyManifestError(
                f"no selected configuration for {expected_frequency}"
            )
        expected_fields = {
            "array_name": spec.array_name,
            "array_rows": spec.array_rows,
            "array_cols": spec.array_cols,
            "frequency_hz": expected_frequency,
            "config_id": configuration.config_id,
            "beam_id": configuration.beam_id,
        }
        for field, expected in expected_fields.items():
            if getattr(record, field) != expected:
                raise CrossFrequencyManifestError(
                    f"record {record.sample_key} {field} mismatch"
                )
        if not math.isclose(record.steering_deg, spec.steering_deg, abs_tol=1e-9):
            raise CrossFrequencyManifestError("record steering angle is not zero degrees")
    split_counts = {
        name: sum(record.split == name for record in values)
        for name in ("train", "val", "test")
    }
    expected_counts = {
        "train": spec.train_samples,
        "val": spec.val_samples,
        "test": spec.test_samples,
    }
    if split_counts != expected_counts:
        raise CrossFrequencyManifestError(
            f"split counts mismatch: {split_counts}, expected {expected_counts}"
        )
    if inventory is not None:
        expected = build_cross_frequency_records(inventory, split, selected, spec)
        if values != expected:
            raise CrossFrequencyManifestError(
                "records differ from the deterministic inventory resolution"
            )


def select_zero_degree_configurations_for_array_sizes(
    schema: Mapping[str, Any] | Any,
    *,
    frequency_hz: int,
    array_sizes: Sequence[str],
    steering_deg: float = ZERO_DEGREE,
) -> dict[str, SelectedZeroDegreeConfiguration]:
    configurations = _configurations(schema)
    return {
        array_size: _select_one_for_array(
            configurations,
            frequency_hz=frequency_hz,
            array_size=array_size,
            steering_deg=steering_deg,
        )
        for array_size in array_sizes
    }


def select_zero_degree_configurations_same_frequency(
    schema: Mapping[str, Any] | Any,
    frequency_hz: int,
    array_sizes: Sequence[str],
) -> dict[str, SelectedZeroDegreeConfiguration]:
    return select_zero_degree_configurations_for_array_sizes(
        schema,
        frequency_hz=frequency_hz,
        array_sizes=array_sizes,
        steering_deg=ZERO_DEGREE,
    )


def _scene_sort_key(scene_id: str) -> tuple[str, int]:
    if not scene_id.startswith("u"):
        return (scene_id, -1)
    suffix = scene_id[1:]
    if suffix.isdigit():
        return ("u", int(suffix))
    return (scene_id, -1)


def build_same_frequency_records(
    *,
    schema: Mapping[str, Any] | Any,
    split: SceneSplit,
    selected: Mapping[str, SelectedZeroDegreeConfiguration],
    workspace_root: Path,
    array_size: str,
    frequency_hz: int = TEST_FREQUENCY_HZ,
    steering_deg: float = ZERO_DEGREE,
) -> tuple[ManifestRecord, ...]:
    if array_size not in selected:
        raise CrossFrequencyManifestError(
            f"selected configuration missing array size {array_size}"
        )
    selection = selected[array_size]
    if selection.frequency_hz != frequency_hz:
        raise CrossFrequencyManifestError(
            f"selected frequency mismatch for {array_size}: {selection.frequency_hz}"
        )
    if not math.isclose(selection.steering_deg, steering_deg, abs_tol=1e-9):
        raise CrossFrequencyManifestError(
            f"selected steering mismatch for {array_size}: {selection.steering_deg}"
        )
    inventory = inventory_selected_configurations(
        workspace_root,
        {array_size: selection},
        scene_ids=(*split.train, *split.val, *split.test),
        array_key=array_size,
    )
    lookup = _split_lookup(split)
    workspace_root = Path(workspace_root).resolve()
    records: list[ManifestRecord] = []
    for scene_id in sorted(lookup, key=_scene_sort_key):
        height, beam_map, radiomap = inventory.require_unique_triplet(
            selection.config_id,
            selection.beam_id,
            scene_id,
        )
        if height.name != f"{scene_id}_height_matrix.npy":
            raise CrossFrequencyManifestError(
                f"height filename does not match scene {scene_id}: {height.name}"
            )
        if radiomap.name != f"{scene_id}_labeled_radiomap.npy":
            raise CrossFrequencyManifestError(
                f"radiomap filename does not match scene {scene_id}: {radiomap.name}"
            )
        records.append(
            ManifestRecord(
                sample_key=(
                    f"{scene_id}|{array_size}|freq{frequency_hz}"
                    f"|angle{steering_deg:.1f}|beam{selection.beam_id:02d}"
                ),
                split=lookup[scene_id],
                scene_id=scene_id,
                array_name=array_size,
                array_rows=selection.rows,
                array_cols=selection.cols,
                frequency_hz=frequency_hz,
                config_id=selection.config_id,
                beam_id=selection.beam_id,
                steering_deg=selection.steering_deg,
                height_path=_relative_path(height, workspace_root),
                beam_map_path=_relative_path(beam_map, workspace_root),
                radiomap_path=_relative_path(radiomap, workspace_root),
            )
        )
    return tuple(records)


def validate_same_frequency_records(
    records: Sequence[ManifestRecord],
    *,
    split: SceneSplit,
    selected: Mapping[str, SelectedZeroDegreeConfiguration],
    array_size: str,
    frequency_hz: int = TEST_FREQUENCY_HZ,
    steering_deg: float = ZERO_DEGREE,
    schema: Mapping[str, Any] | Any | None = None,
    workspace_root: Path | None = None,
) -> None:
    values = tuple(records)
    expected_counts = {"train": 560, "val": 80, "test": 160}
    if len(values) != sum(expected_counts.values()):
        raise CrossFrequencyManifestError(
            f"expected 800 records, got {len(values)}"
        )
    if len({record.sample_key for record in values}) != len(values):
        raise CrossFrequencyManifestError("sample keys are not unique")
    if len({record.scene_id for record in values}) != len(values):
        raise CrossFrequencyManifestError("scene IDs are not unique")
    if len({record.radiomap_path for record in values}) != len(values):
        raise CrossFrequencyManifestError("radiomap paths are not unique")
    lookup = _split_lookup(split)
    split_counts = {
        name: sum(record.split == name for record in values)
        for name in ("train", "val", "test")
    }
    if split_counts != expected_counts:
        raise CrossFrequencyManifestError(
            f"split counts mismatch: {split_counts}, expected {expected_counts}"
        )
    selection = selected.get(array_size)
    if selection is None:
        raise CrossFrequencyManifestError(
            f"selected configuration missing array size {array_size}"
        )
    for record in values:
        if not isinstance(record, ManifestRecord):
            raise CrossFrequencyManifestError("all records must be ManifestRecord values")
        if lookup.get(record.scene_id) != record.split:
            raise CrossFrequencyManifestError(
                f"record split mismatch for {record.scene_id}: {record.split}"
            )
        if record.array_name != array_size:
            raise CrossFrequencyManifestError(
                f"record array size mismatch: {record.array_name}"
            )
        if record.frequency_hz != frequency_hz:
            raise CrossFrequencyManifestError(
                f"record frequency mismatch for {record.scene_id}"
            )
        if record.config_id != selection.config_id or record.beam_id != selection.beam_id:
            raise CrossFrequencyManifestError(
                f"record selection mismatch for {record.scene_id}"
            )
        if (record.array_rows, record.array_cols) != (selection.rows, selection.cols):
            raise CrossFrequencyManifestError(
                f"record array geometry mismatch for {record.scene_id}"
            )
        if not math.isclose(record.steering_deg, steering_deg, abs_tol=1e-9):
            raise CrossFrequencyManifestError(
                f"record steering angle is not {steering_deg}"
            )
        if Path(record.height_path).name != f"{record.scene_id}_height_matrix.npy":
            raise CrossFrequencyManifestError(
                f"height path mismatch for {record.scene_id}"
            )
        if Path(record.radiomap_path).name != f"{record.scene_id}_labeled_radiomap.npy":
            raise CrossFrequencyManifestError(
                f"radiomap path mismatch for {record.scene_id}"
            )
    if schema is not None and workspace_root is not None:
        expected = build_same_frequency_records(
            schema=schema,
            split=split,
            selected=selected,
            workspace_root=workspace_root,
            array_size=array_size,
            frequency_hz=frequency_hz,
            steering_deg=steering_deg,
        )
        if values != expected:
            raise CrossFrequencyManifestError(
                "records differ from the deterministic inventory resolution"
            )


def build_same_frequency_manifest_artifact(
    *,
    dataset_root: Path,
    split_path: Path,
    array_size: str,
    frequency_hz: int = TEST_FREQUENCY_HZ,
    steering_deg: float = ZERO_DEGREE,
    output_path: Path | None = None,
    schema_path: Path | None = None,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    split_path = Path(split_path).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    resolved_schema = (
        Path(schema_path).resolve()
        if schema_path is not None
        else repo_root / "experiments" / "multiconfig_schema.json"
    )
    output = (
        Path(output_path).resolve()
        if output_path is not None
        else split_path.parent / f"manifest_samefreq_{frequency_hz / 1_000_000_000:.1f}ghz_{array_size}_{steering_deg:.0f}deg.jsonl"
    )
    schema = load_schema_lock(resolved_schema)
    try:
        payload = __import__("json").loads(split_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError(f"cannot read fixed scene split {split_path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("fixed scene split must be a JSON object")
    split = SceneSplit.from_dict(payload)
    selected = select_zero_degree_configurations_same_frequency(
        schema,
        frequency_hz,
        (array_size,),
    )
    records = build_same_frequency_records(
        schema=schema,
        split=split,
        selected=selected,
        workspace_root=dataset_root,
        array_size=array_size,
        frequency_hz=frequency_hz,
        steering_deg=steering_deg,
    )
    validate_same_frequency_records(
        records,
        split=split,
        selected=selected,
        array_size=array_size,
        frequency_hz=frequency_hz,
        steering_deg=steering_deg,
        schema=schema,
        workspace_root=dataset_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest_jsonl(output, records)
    written = load_manifest_jsonl(output)
    validate_same_frequency_records(
        written,
        split=split,
        selected=selected,
        array_size=array_size,
        frequency_hz=frequency_hz,
        steering_deg=steering_deg,
        schema=schema,
        workspace_root=dataset_root,
    )
    return {
        "schema_version": 1,
        "manifest": str(output),
        "manifest_sha256": sha256_file(output),
        "scene_split": str(split_path),
        "scene_split_sha256": sha256_file(split_path),
        "dataset_revision": schema.identities.get("dataset_revision"),
        "records": len(written),
        "split_counts": {
            name: sum(record.split == name for record in written)
            for name in ("train", "val", "test")
        },
        "selected": {
            array_size: {
                "config_id": selected[array_size].config_id,
                "beam_id": selected[array_size].beam_id,
                "steering_deg": selected[array_size].steering_deg,
                "frequency_hz": selected[array_size].frequency_hz,
            }
        },
        "schema_path": str(resolved_schema),
    }
