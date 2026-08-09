from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from experiments.multiconfig_manifest import ManifestRecord, SceneSplit


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


def select_zero_degree_configurations(
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
