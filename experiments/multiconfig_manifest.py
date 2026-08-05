from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

import numpy as np

from experiments.provenance import (
    DATASET_FILENAME,
    DATASET_REPO_ID,
    DATASET_REVISION,
    REFERENCE_CODE_URL,
    REFERENCE_CODE_REVISION,
    git_output,
    sha256_file,
)


DATA_ROOT_RELATIVE = "raw/Dataset"
COMMON_ANGLES = (-28.0, -21.0, -14.0, -7.0, 0.0, 7.0, 14.0, 21.0)
REFERENCE_SCRIPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "DatasetGeneration_Step6_BeammapGenerator.py",
        ("beam_geometry", "array_geometry"),
    ),
    (
        "DatasetGeneration_Step5_RadiomapValidation.py",
        ("radiomap_shape", "target_sentinels"),
    ),
    (
        "multiconfig_dataset_prepcocess_Unet.py",
        ("transmitter_pixel", "target_sentinels"),
    ),
)


class ConfigurationFormatError(RuntimeError):
    """A released configuration cannot be parsed without guessing."""


class ConfigurationMismatchError(RuntimeError):
    """A parsed configuration disagrees with an immutable benchmark spec."""


class SchemaIdentityError(RuntimeError):
    """A schema lock names a source other than the approved immutable source."""


class DuplicateConfigurationError(RuntimeError):
    """A schema lock contains a repeated logical configuration identifier."""


class MissingSamplePathError(RuntimeError):
    """A logical sample component has no released file."""


class AmbiguousSamplePathError(RuntimeError):
    """A logical sample component resolves to more than one released file."""


class BeamInventoryMismatchError(RuntimeError):
    """Configuration, beam-map, and radiomap releases are not one-to-one."""


class ExistingSchemaMismatchError(RuntimeError):
    """An immutable audit or schema file already exists with other bytes."""


@dataclass(frozen=True)
class BeamSpec:
    beam_id: int
    steering_deg: float


@dataclass(frozen=True)
class ArraySpec:
    name: Literal["8x8", "16x16", "32x32"]
    rows: int
    cols: int
    tx_elements: int
    frequency_hz: int
    beams: tuple[BeamSpec, ...]


ARRAY_SPECS: dict[str, ArraySpec] = {
    "8x8": ArraySpec(
        "8x8",
        8,
        8,
        64,
        6_700_000_000,
        tuple(BeamSpec(index, -28.0 + 7.0 * index) for index in range(8)),
    ),
    "16x16": ArraySpec(
        "16x16",
        16,
        16,
        256,
        6_700_000_000,
        tuple(
            BeamSpec(index, -28.0 + 3.5 * index)
            for index in (0, 2, 4, 6, 8, 10, 12, 14)
        ),
    ),
    "32x32": ArraySpec(
        "32x32",
        32,
        32,
        1024,
        6_700_000_000,
        tuple(
            BeamSpec(index, -32.0 + index)
            for index in (4, 11, 18, 25, 32, 39, 46, 53)
        ),
    ),
}


@dataclass(frozen=True)
class ParsedField:
    path: str
    name: str
    raw_value: str
    line_number: int
    indentation: int


@dataclass(frozen=True)
class ParsedConfiguration:
    source_path: str
    rows: int
    cols: int
    tx_elements: int
    frequency_hz: int
    num_beams: int
    start_angle_deg: float
    beam_spacing_deg: float
    end_angle_deg: float
    beam_angles: tuple[float, ...]
    fields: tuple[ParsedField, ...]


@dataclass(frozen=True)
class AuditedTextFile:
    relative_path: str
    sha256: str
    fields: tuple[ParsedField, ...]
    parsed: ParsedConfiguration

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "fields": [asdict(field) for field in self.fields],
            "parsed": {
                key: value
                for key, value in asdict(self.parsed).items()
                if key != "fields"
            },
        }


@dataclass(frozen=True)
class FileEvidence:
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class BeamFileEvidence:
    beam_id: int
    relative_path: str
    sha256: str
    steering_deg: float | None = None


@dataclass(frozen=True)
class ReferenceScriptEvidence:
    relative_path: str
    sha256: str
    purposes: tuple[str, ...]


@dataclass(frozen=True)
class ReleasedConfiguration:
    configuration_id: str
    rows: int
    cols: int
    tx_elements: int
    frequency_hz: int
    num_beams: int
    start_angle_deg: float
    beam_spacing_deg: float
    end_angle_deg: float
    beam_angles: tuple[float, ...]
    aggregate_beam_setting: FileEvidence
    config_files: tuple[BeamFileEvidence, ...]
    radiomap_beam_settings: tuple[BeamFileEvidence, ...]
    beam_maps: tuple[BeamFileEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConfigAuditReport:
    data_root: str
    text_files: tuple[AuditedTextFile, ...]
    configurations: tuple[ReleasedConfiguration, ...] = ()
    identities: Mapping[str, Any] = field(default_factory=dict)
    reference_scripts: tuple[ReferenceScriptEvidence, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "data_root": self.data_root,
            "text_files": [item.to_dict() for item in self.text_files],
            "configurations": [
                configuration.to_dict()
                for configuration in self.configurations
            ],
            "identities": dict(self.identities),
            "reference_scripts": [
                asdict(script) for script in self.reference_scripts
            ],
        }


def _parse_fields(text: str, source_path: str) -> tuple[ParsedField, ...]:
    stack: list[tuple[int, str]] = []
    fields: list[ParsedField] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.strip().startswith("==="):
            continue
        indentation = len(line) - len(line.lstrip(" "))
        if "\t" in line[:indentation] or line[indentation :].startswith("\t"):
            raise ConfigurationFormatError(
                f"tab indentation is unsupported in {source_path}:{line_number}"
            )
        content = line[indentation:]
        if ":" not in content:
            raise ConfigurationFormatError(
                f"unparsed configuration line {source_path}:{line_number}: {line!r}"
            )
        name, raw_value = content.split(":", 1)
        name = name.strip()
        if not name:
            raise ConfigurationFormatError(
                f"empty field name in {source_path}:{line_number}"
            )
        while stack and stack[-1][0] >= indentation:
            stack.pop()
        path = ".".join([*(part for _level, part in stack), name])
        fields.append(
            ParsedField(
                path=path,
                name=name,
                raw_value=raw_value,
                line_number=line_number,
                indentation=indentation,
            )
        )
        if not raw_value.strip():
            stack.append((indentation, name))
    return tuple(fields)


def _unique_field(
    fields: tuple[ParsedField, ...],
    path: str,
    source_path: str,
) -> ParsedField:
    matches = [field for field in fields if field.path == path]
    if len(matches) != 1:
        raise ConfigurationFormatError(
            f"expected exactly one {path!r} in {source_path}, got {len(matches)}"
        )
    return matches[0]


def _integer_value(field: ParsedField, source_path: str) -> int:
    raw = field.raw_value.strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ConfigurationFormatError(
            f"{field.path} is not numeric in {source_path}: {raw!r}"
        ) from error
    if not math.isfinite(value) or not value.is_integer():
        raise ConfigurationFormatError(
            f"{field.path} is not an integer in {source_path}: {raw!r}"
        )
    return int(value)


def _float_value(field: ParsedField, source_path: str) -> float:
    raw = field.raw_value.strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ConfigurationFormatError(
            f"{field.path} is not numeric in {source_path}: {raw!r}"
        ) from error
    if not math.isfinite(value):
        raise ConfigurationFormatError(
            f"{field.path} is not finite in {source_path}: {raw!r}"
        )
    return value


def _beam_angles(
    fields: tuple[ParsedField, ...],
    source_path: str,
    num_beams: int,
    start_angle_deg: float,
    beam_spacing_deg: float,
) -> tuple[float, ...]:
    field = _unique_field(fields, "beam_configuration.beam_angles", source_path)
    raw = field.raw_value.strip()
    generated = tuple(
        start_angle_deg + beam_spacing_deg * index
        for index in range(num_beams)
    )
    if raw.startswith("[列表长度=") and raw.endswith("]"):
        declared = raw[len("[列表长度=") : -1]
        try:
            declared_count = int(declared)
        except ValueError as error:
            raise ConfigurationFormatError(
                f"invalid summarized beam list in {source_path}: {raw!r}"
            ) from error
        if declared_count != num_beams:
            raise ConfigurationFormatError(
                f"beam list length mismatch in {source_path}"
            )
        return generated
    try:
        literal = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as error:
        raise ConfigurationFormatError(
            f"invalid beam angle list in {source_path}: {raw!r}"
        ) from error
    if not isinstance(literal, (list, tuple)) or len(literal) != num_beams:
        raise ConfigurationFormatError(
            f"beam angle count mismatch in {source_path}"
        )
    try:
        parsed = tuple(float(value) for value in literal)
    except (TypeError, ValueError) as error:
        raise ConfigurationFormatError(
            f"beam angle list is not numeric in {source_path}"
        ) from error
    if any(not math.isfinite(value) for value in parsed):
        raise ConfigurationFormatError(
            f"beam angle list contains a non-finite value in {source_path}"
        )
    for actual, expected in zip(parsed, generated):
        if not math.isclose(actual, expected, abs_tol=1e-9):
            raise ConfigurationFormatError(
                f"beam angle sequence mismatch in {source_path}"
            )
    return parsed


def parse_configuration_text(
    text: str,
    *,
    source_path: str,
) -> ParsedConfiguration:
    fields = _parse_fields(text, source_path)
    rows = _integer_value(
        _unique_field(fields, "transmitter_array.num_rows", source_path),
        source_path,
    )
    cols = _integer_value(
        _unique_field(fields, "transmitter_array.num_cols", source_path),
        source_path,
    )
    tx_elements = _integer_value(
        _unique_field(fields, "transmitter_array.total_elements", source_path),
        source_path,
    )
    frequency_hz = _integer_value(
        _unique_field(fields, "scene_basic.frequency_hz", source_path),
        source_path,
    )
    num_beams = _integer_value(
        _unique_field(fields, "beam_configuration.num_beams", source_path),
        source_path,
    )
    start_angle_deg = _float_value(
        _unique_field(
            fields,
            "beam_configuration.start_angle_deg",
            source_path,
        ),
        source_path,
    )
    beam_spacing_deg = _float_value(
        _unique_field(
            fields,
            "beam_configuration.beam_spacing_deg",
            source_path,
        ),
        source_path,
    )
    end_angle_deg = _float_value(
        _unique_field(
            fields,
            "beam_configuration.end_angle_deg",
            source_path,
        ),
        source_path,
    )
    if rows <= 0 or cols <= 0 or tx_elements <= 0 or num_beams <= 0:
        raise ConfigurationFormatError(
            f"array and beam counts must be positive in {source_path}"
        )
    expected_end = start_angle_deg + beam_spacing_deg * (num_beams - 1)
    if not math.isclose(end_angle_deg, expected_end, abs_tol=1e-9):
        raise ConfigurationFormatError(
            f"beam end angle mismatch in {source_path}: "
            f"expected {expected_end}, got {end_angle_deg}"
        )
    angles = _beam_angles(
        fields,
        source_path,
        num_beams,
        start_angle_deg,
        beam_spacing_deg,
    )
    return ParsedConfiguration(
        source_path=source_path,
        rows=rows,
        cols=cols,
        tx_elements=tx_elements,
        frequency_hz=frequency_hz,
        num_beams=num_beams,
        start_angle_deg=start_angle_deg,
        beam_spacing_deg=beam_spacing_deg,
        end_angle_deg=end_angle_deg,
        beam_angles=angles,
        fields=fields,
    )


def validate_config_against_spec(
    config: ParsedConfiguration,
    spec: ArraySpec,
) -> None:
    if (config.rows, config.cols) != (spec.rows, spec.cols):
        raise ConfigurationMismatchError(
            f"expected {spec.rows}x{spec.cols}, got {config.rows}x{config.cols}"
        )
    if config.tx_elements != spec.tx_elements:
        raise ConfigurationMismatchError("transmitter element count mismatch")
    if config.frequency_hz != spec.frequency_hz:
        raise ConfigurationMismatchError("carrier frequency mismatch")


def resolve_selected_beams(
    config: ParsedConfiguration,
    spec: ArraySpec,
) -> tuple[BeamSpec, ...]:
    validate_config_against_spec(config, spec)
    for beam in spec.beams:
        if beam.beam_id < 0 or beam.beam_id >= len(config.beam_angles):
            raise ConfigurationMismatchError(
                f"selected beam {beam.beam_id} is not released"
            )
        actual = config.beam_angles[beam.beam_id]
        if not math.isclose(actual, beam.steering_deg, abs_tol=1e-9):
            raise ConfigurationMismatchError(
                f"beam {beam.beam_id} angle mismatch: "
                f"expected {beam.steering_deg}, got {actual}"
            )
    return spec.beams


_CONFIG_FILE_PATTERN = re.compile(
    r"^raw/Dataset/configs/"
    r"(?P<configuration_id>.+)_beam(?P<beam_id>\d{2})_beam_settings\.txt$"
)
_AGGREGATE_SETTING_PATTERN = re.compile(
    r"^raw/Dataset/beam_maps/(?P<configuration_id>[^/]+)/u0/"
    r"beam_settings\.txt$"
)
_RADIOMAP_SETTING_PATTERN = re.compile(
    r"^raw/Dataset/radiomaps/"
    r"(?P<configuration_id>.+)_beam(?P<beam_id>\d{2})/"
    r"beam_settings\.txt$"
)
_BEAM_MAP_PATTERN = re.compile(
    r"^raw/Dataset/beam_maps/(?P<configuration_id>[^/]+)/u0/"
    r"beam_(?P<beam_id>\d{2})_angle_"
    r"(?P<angle>-?\d+(?:\.\d+)?)_matrix\.npy$"
)
_HEIGHT_PATTERN = re.compile(
    r"^raw/Dataset/height_maps/(?P<scene_id>u[1-9]\d*)/"
    r"(?P=scene_id)_height_matrix\.npy$"
)
_RADIOMAP_PATTERN = re.compile(
    r"^raw/Dataset/radiomaps/"
    r"(?P<configuration_id>.+)_beam(?P<beam_id>\d{2})/"
    r"(?P<scene_id>u[1-9]\d*)_labeled_radiomap\.npy$"
)


def _configuration_signature(
    parsed: ParsedConfiguration,
) -> tuple[object, ...]:
    return (
        parsed.rows,
        parsed.cols,
        parsed.tx_elements,
        parsed.frequency_hz,
        parsed.num_beams,
        parsed.start_angle_deg,
        parsed.beam_spacing_deg,
        parsed.end_angle_deg,
        parsed.beam_angles,
    )


def _add_unique_beam_text(
    index: dict[str, dict[int, AuditedTextFile]],
    configuration_id: str,
    beam_id: int,
    item: AuditedTextFile,
    *,
    label: str,
) -> None:
    beams = index.setdefault(configuration_id, {})
    if beam_id in beams:
        raise BeamInventoryMismatchError(
            f"duplicate {label} for {configuration_id} beam {beam_id:02d}"
        )
    beams[beam_id] = item


def _build_released_configurations(
    workspace_root: Path,
    text_files: tuple[AuditedTextFile, ...],
) -> tuple[ReleasedConfiguration, ...]:
    config_files: dict[str, dict[int, AuditedTextFile]] = {}
    radiomap_settings: dict[str, dict[int, AuditedTextFile]] = {}
    aggregate_settings: dict[str, AuditedTextFile] = {}
    for item in text_files:
        match = _CONFIG_FILE_PATTERN.fullmatch(item.relative_path)
        if match is not None:
            _add_unique_beam_text(
                config_files,
                match.group("configuration_id"),
                int(match.group("beam_id")),
                item,
                label="config file",
            )
            continue
        match = _AGGREGATE_SETTING_PATTERN.fullmatch(item.relative_path)
        if match is not None:
            configuration_id = match.group("configuration_id")
            if configuration_id in aggregate_settings:
                raise BeamInventoryMismatchError(
                    f"duplicate aggregate beam setting for {configuration_id}"
                )
            aggregate_settings[configuration_id] = item
            continue
        match = _RADIOMAP_SETTING_PATTERN.fullmatch(item.relative_path)
        if match is not None:
            _add_unique_beam_text(
                radiomap_settings,
                match.group("configuration_id"),
                int(match.group("beam_id")),
                item,
                label="radiomap beam setting",
            )

    data_root = workspace_root.joinpath(*PurePosixPath(DATA_ROOT_RELATIVE).parts)
    beam_maps: dict[str, dict[int, BeamFileEvidence]] = {}
    for path in sorted(
        data_root.glob("beam_maps/*/u0/beam_*_angle_*_matrix.npy"),
        key=lambda value: value.relative_to(workspace_root).as_posix(),
    ):
        if not path.is_file():
            continue
        relative_path = path.relative_to(workspace_root).as_posix()
        match = _BEAM_MAP_PATTERN.fullmatch(relative_path)
        if match is None:
            raise BeamInventoryMismatchError(
                f"unparsed beam-map filename: {relative_path}"
            )
        configuration_id = match.group("configuration_id")
        beam_id = int(match.group("beam_id"))
        beams = beam_maps.setdefault(configuration_id, {})
        if beam_id in beams:
            raise BeamInventoryMismatchError(
                f"duplicate beam map for {configuration_id} beam {beam_id:02d}"
            )
        beams[beam_id] = BeamFileEvidence(
            beam_id=beam_id,
            steering_deg=float(match.group("angle")),
            relative_path=relative_path,
            sha256=sha256_file(path),
        )

    configuration_ids = sorted(
        set(config_files)
        | set(radiomap_settings)
        | set(aggregate_settings)
        | set(beam_maps)
    )
    released: list[ReleasedConfiguration] = []
    for configuration_id in configuration_ids:
        aggregate = aggregate_settings.get(configuration_id)
        if aggregate is None:
            raise BeamInventoryMismatchError(
                f"missing aggregate beam setting for {configuration_id}"
            )
        parsed = aggregate.parsed
        if parsed.rows * parsed.cols != parsed.tx_elements:
            raise BeamInventoryMismatchError(
                f"direct rows/cols disagree with total elements for {configuration_id}"
            )
        expected_beams = set(range(parsed.num_beams))
        categories: tuple[
            tuple[str, dict[int, object]], ...
        ] = (
            ("config files", config_files.get(configuration_id, {})),
            (
                "radiomap beam settings",
                radiomap_settings.get(configuration_id, {}),
            ),
            ("beam maps", beam_maps.get(configuration_id, {})),
        )
        for label, values in categories:
            actual_beams = set(values)
            if actual_beams != expected_beams:
                missing = sorted(expected_beams - actual_beams)
                extra = sorted(actual_beams - expected_beams)
                raise BeamInventoryMismatchError(
                    f"{label} mismatch for {configuration_id}: "
                    f"missing={missing}, extra={extra}"
                )
        expected_signature = _configuration_signature(parsed)
        for collection in (
            config_files[configuration_id],
            radiomap_settings[configuration_id],
        ):
            for item in collection.values():
                if _configuration_signature(item.parsed) != expected_signature:
                    raise BeamInventoryMismatchError(
                        "configuration fields disagree for "
                        f"{configuration_id}: {item.relative_path}"
                    )
        for beam_id, evidence in beam_maps[configuration_id].items():
            expected_angle = parsed.beam_angles[beam_id]
            assert evidence.steering_deg is not None
            if not math.isclose(
                evidence.steering_deg,
                expected_angle,
                abs_tol=1e-9,
            ):
                raise BeamInventoryMismatchError(
                    f"beam-map angle mismatch for {configuration_id} "
                    f"beam {beam_id:02d}: expected {expected_angle}, "
                    f"got {evidence.steering_deg}"
                )
        released.append(
            ReleasedConfiguration(
                configuration_id=configuration_id,
                rows=parsed.rows,
                cols=parsed.cols,
                tx_elements=parsed.tx_elements,
                frequency_hz=parsed.frequency_hz,
                num_beams=parsed.num_beams,
                start_angle_deg=parsed.start_angle_deg,
                beam_spacing_deg=parsed.beam_spacing_deg,
                end_angle_deg=parsed.end_angle_deg,
                beam_angles=parsed.beam_angles,
                aggregate_beam_setting=FileEvidence(
                    relative_path=aggregate.relative_path,
                    sha256=aggregate.sha256,
                ),
                config_files=tuple(
                    BeamFileEvidence(
                        beam_id=beam_id,
                        relative_path=item.relative_path,
                        sha256=item.sha256,
                    )
                    for beam_id, item in sorted(
                        config_files[configuration_id].items()
                    )
                ),
                radiomap_beam_settings=tuple(
                    BeamFileEvidence(
                        beam_id=beam_id,
                        relative_path=item.relative_path,
                        sha256=item.sha256,
                    )
                    for beam_id, item in sorted(
                        radiomap_settings[configuration_id].items()
                    )
                ),
                beam_maps=tuple(
                    item
                    for _beam_id, item in sorted(
                        beam_maps[configuration_id].items()
                    )
                ),
            )
        )
    return tuple(released)


def select_benchmark_configurations(
    report: ConfigAuditReport,
) -> dict[str, ReleasedConfiguration]:
    selected: dict[str, ReleasedConfiguration] = {}
    for array_name, spec in ARRAY_SPECS.items():
        candidates = [
            configuration
            for configuration in report.configurations
            if configuration.rows == spec.rows
            and configuration.cols == spec.cols
            and configuration.tx_elements == spec.tx_elements
            and configuration.frequency_hz == spec.frequency_hz
        ]
        if len(candidates) != 1:
            raise ConfigurationMismatchError(
                f"expected exactly one released configuration for {array_name}, "
                f"got {[item.configuration_id for item in candidates]}"
            )
        configuration = candidates[0]
        for beam in spec.beams:
            if beam.beam_id >= len(configuration.beam_angles):
                raise ConfigurationMismatchError(
                    f"selected beam {beam.beam_id} is absent from "
                    f"{configuration.configuration_id}"
                )
            actual = configuration.beam_angles[beam.beam_id]
            if not math.isclose(actual, beam.steering_deg, abs_tol=1e-9):
                raise ConfigurationMismatchError(
                    f"selected beam {beam.beam_id} in "
                    f"{configuration.configuration_id} has angle {actual}, "
                    f"expected {beam.steering_deg}"
                )
        selected[array_name] = configuration
    return selected


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SchemaIdentityError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SchemaIdentityError(f"{label} must contain a JSON object: {path}")
    return payload


def _normalized_url(value: str) -> str:
    return value.strip().replace("\\", "/").rstrip("/")


def _audit_source_identities(
    workspace_root: Path,
) -> tuple[dict[str, Any], tuple[ReferenceScriptEvidence, ...]]:
    download_receipt_path = workspace_root / "download_receipt.json"
    extraction_receipt_path = workspace_root / "extraction_receipt.json"
    reference_root = workspace_root / "reference_code"
    expected_paths = (
        download_receipt_path,
        extraction_receipt_path,
        reference_root,
    )
    present = [path.exists() for path in expected_paths]
    if not any(present):
        return {}, ()
    if not all(present):
        missing = [
            str(path) for path, exists in zip(expected_paths, present) if not exists
        ]
        raise SchemaIdentityError(
            f"source identity evidence is incomplete; missing {missing}"
        )

    download = _read_json_object(
        download_receipt_path,
        label="download receipt",
    )
    extraction = _read_json_object(
        extraction_receipt_path,
        label="extraction receipt",
    )
    required_download = {
        "repo_id": DATASET_REPO_ID,
        "revision": DATASET_REVISION,
        "filename": DATASET_FILENAME,
    }
    for key, expected in required_download.items():
        if download.get(key) != expected:
            raise SchemaIdentityError(
                f"download receipt {key} mismatch: "
                f"expected {expected}, got {download.get(key)}"
            )
    archive_path = workspace_root / "downloads" / DATASET_FILENAME
    if not archive_path.is_file():
        raise SchemaIdentityError(f"dataset archive is missing: {archive_path}")
    archive_sha256 = sha256_file(archive_path)
    if download.get("sha256") != archive_sha256:
        raise SchemaIdentityError(
            "dataset archive SHA-256 disagrees with the download receipt"
        )
    if download.get("size_bytes") != archive_path.stat().st_size:
        raise SchemaIdentityError(
            "dataset archive size disagrees with the download receipt"
        )
    if extraction.get("archive_sha256") != archive_sha256:
        raise SchemaIdentityError(
            "extraction receipt archive SHA-256 mismatch"
        )
    if extraction.get("destination") != "raw":
        raise SchemaIdentityError(
            f"extraction destination mismatch: {extraction.get('destination')}"
        )

    actual_origin = git_output(reference_root, "remote", "get-url", "origin")
    if _normalized_url(actual_origin) != _normalized_url(REFERENCE_CODE_URL):
        raise SchemaIdentityError(
            "reference-code origin mismatch: "
            f"expected {REFERENCE_CODE_URL}, got {actual_origin}"
        )
    actual_revision = git_output(reference_root, "rev-parse", "HEAD")
    if actual_revision != REFERENCE_CODE_REVISION:
        raise SchemaIdentityError(
            "reference-code revision mismatch: "
            f"expected {REFERENCE_CODE_REVISION}, got {actual_revision}"
        )
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=reference_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if symbolic.returncode == 0:
        raise SchemaIdentityError(
            "reference-code checkout must be detached at the locked revision"
        )
    if symbolic.returncode != 1:
        raise SchemaIdentityError(
            f"cannot determine detached reference-code state: {symbolic.stderr}"
        )
    reference_status = git_output(
        reference_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if reference_status:
        raise SchemaIdentityError(
            f"reference-code checkout is dirty: {reference_status}"
        )

    scripts: list[ReferenceScriptEvidence] = []
    for script_name, purposes in REFERENCE_SCRIPTS:
        path = reference_root / script_name
        if not path.is_file():
            raise SchemaIdentityError(f"reference script is missing: {path}")
        scripts.append(
            ReferenceScriptEvidence(
                relative_path=path.relative_to(workspace_root).as_posix(),
                sha256=sha256_file(path),
                purposes=purposes,
            )
        )

    identities = {
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "archive_filename": DATASET_FILENAME,
        "archive_relative_path": (
            Path("downloads") / DATASET_FILENAME
        ).as_posix(),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_path.stat().st_size,
        "download_receipt_relative_path": "download_receipt.json",
        "download_receipt_sha256": sha256_file(download_receipt_path),
        "extraction_receipt_relative_path": "extraction_receipt.json",
        "extraction_receipt_sha256": sha256_file(extraction_receipt_path),
        "extraction_inventory_sha256": extraction.get("inventory_sha256"),
        "extraction_inventory_files": extraction.get("inventory_files"),
        "extraction_inventory_bytes": extraction.get("inventory_bytes"),
        "reference_code_origin": actual_origin,
        "reference_code_revision": actual_revision,
    }
    return identities, tuple(scripts)


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_payloads_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _transmitter_position_from_report(
    report: ConfigAuditReport,
    selected: Mapping[str, ReleasedConfiguration],
) -> list[float]:
    text_by_path = {
        item.relative_path: item for item in report.text_files
    }
    positions: list[tuple[float, float, float]] = []
    for configuration in selected.values():
        audited = text_by_path.get(
            configuration.aggregate_beam_setting.relative_path
        )
        if audited is None:
            raise ConfigurationFormatError(
                "aggregate beam setting is absent from the text audit: "
                f"{configuration.aggregate_beam_setting.relative_path}"
            )
        field_value = _unique_field(
            audited.fields,
            "transmitter.position",
            audited.relative_path,
        ).raw_value.strip()
        try:
            literal = ast.literal_eval(field_value)
        except (SyntaxError, ValueError) as error:
            raise ConfigurationFormatError(
                f"invalid transmitter.position in {audited.relative_path}: "
                f"{field_value!r}"
            ) from error
        if not isinstance(literal, (list, tuple)) or len(literal) != 3:
            raise ConfigurationFormatError(
                f"transmitter.position must contain three values in "
                f"{audited.relative_path}"
            )
        try:
            position = tuple(float(value) for value in literal)
        except (TypeError, ValueError) as error:
            raise ConfigurationFormatError(
                f"transmitter.position is not numeric in {audited.relative_path}"
            ) from error
        if any(not math.isfinite(value) for value in position):
            raise ConfigurationFormatError(
                f"transmitter.position is non-finite in {audited.relative_path}"
            )
        positions.append(position)
    if not positions or any(position != positions[0] for position in positions[1:]):
        raise ConfigurationMismatchError(
            f"benchmark transmitter positions disagree: {positions}"
        )
    return list(positions[0])


def assemble_schema_payload(
    report: ConfigAuditReport,
    *,
    audit_report_relative_path: str,
    audit_report_sha256: str,
    source_metadata: Mapping[str, Any],
    scene_count: int,
) -> dict[str, Any]:
    selected = select_benchmark_configurations(report)
    arrays: list[dict[str, Any]] = []
    for array_name, spec in ARRAY_SPECS.items():
        configuration = selected[array_name]
        config_files = {
            item.beam_id: item for item in configuration.config_files
        }
        radiomap_settings = {
            item.beam_id: item
            for item in configuration.radiomap_beam_settings
        }
        beam_maps = {
            item.beam_id: item for item in configuration.beam_maps
        }
        arrays.append(
            {
                "name": array_name,
                "configuration_id": configuration.configuration_id,
                "rows": spec.rows,
                "cols": spec.cols,
                "tx_elements": spec.tx_elements,
                "frequency_hz": spec.frequency_hz,
                "released_beam_count": configuration.num_beams,
                "selected_beams": [asdict(beam) for beam in spec.beams],
                "shape_evidence": {
                    "kind": "released_configuration_fields",
                    "relative_path": (
                        configuration.aggregate_beam_setting.relative_path
                    ),
                    "sha256": configuration.aggregate_beam_setting.sha256,
                    "row_field": "transmitter_array.num_rows",
                    "column_field": "transmitter_array.num_cols",
                    "element_count_field": (
                        "transmitter_array.total_elements"
                    ),
                },
                "selected_source_evidence": [
                    {
                        "beam_id": beam.beam_id,
                        "steering_deg": beam.steering_deg,
                        "config_file": asdict(config_files[beam.beam_id]),
                        "radiomap_beam_setting": asdict(
                            radiomap_settings[beam.beam_id]
                        ),
                        "beam_map": asdict(beam_maps[beam.beam_id]),
                    }
                    for beam in spec.beams
                ],
            }
        )
    identities = dict(report.identities)
    identities.update(
        {
            "audit_report_relative_path": audit_report_relative_path,
            "audit_report_sha256": audit_report_sha256,
        }
    )
    return {
        "schema_version": 1,
        "data_root": report.data_root,
        "identities": identities,
        "path_rules": {
            "height": (
                "raw/Dataset/height_maps/{scene_id}/"
                "{scene_id}_height_matrix.npy"
            ),
            "beam_map": (
                "raw/Dataset/beam_maps/{configuration_id}/u0/"
                "beam_{beam_id:02d}_angle_"
                "{steering_deg:.1f}_matrix.npy"
            ),
            "radiomap": (
                "raw/Dataset/radiomaps/"
                "{configuration_id}_beam{beam_id:02d}/"
                "{scene_id}_labeled_radiomap.npy"
            ),
        },
        "scene_domain": {
            "pattern": "u[1-9][0-9]*",
            "count": scene_count,
        },
        "source_metadata": dict(source_metadata),
        "transmitter": {
            "position_xyz_m": _transmitter_position_from_report(
                report,
                selected,
            ),
            "output_pixel_rc": [127, 127],
        },
        "target_domain": {
            "floor_db": -300.0,
            "building_sentinel": 1000.0,
            "valid_lower_exclusive_db": -300.0,
            "valid_upper_exclusive_db": 0.0,
        },
        "configurations": [
            configuration.to_dict()
            for configuration in report.configurations
        ],
        "arrays": arrays,
        "configuration_text_hashes": [
            {
                "relative_path": item.relative_path,
                "sha256": item.sha256,
            }
            for item in report.text_files
        ],
        "reference_scripts": [
            asdict(script) for script in report.reference_scripts
        ],
    }


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    expected = canonical_json_bytes(payload)
    if path.exists():
        try:
            actual = path.read_bytes()
        except OSError as error:
            raise ExistingSchemaMismatchError(
                f"cannot read existing immutable JSON {path}: {error}"
            ) from error
        if actual != expected:
            raise ExistingSchemaMismatchError(
                f"immutable JSON already exists with different bytes: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as output:
            output.write(expected)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def audit_config_files(dataset_root: Path) -> ConfigAuditReport:
    workspace_root = Path(dataset_root).resolve()
    data_root = workspace_root.joinpath(*PurePosixPath(DATA_ROOT_RELATIVE).parts)
    if not data_root.is_dir():
        raise FileNotFoundError(f"released data root is missing: {data_root}")
    text_paths = sorted(
        (path for path in data_root.rglob("*.txt") if path.is_file()),
        key=lambda path: path.relative_to(workspace_root).as_posix(),
    )
    audited: list[AuditedTextFile] = []
    for path in text_paths:
        relative_path = path.relative_to(workspace_root).as_posix()
        try:
            text = path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise ConfigurationFormatError(
                f"cannot read UTF-8 configuration {relative_path}: {error}"
            ) from error
        parsed = parse_configuration_text(text, source_path=relative_path)
        audited.append(
            AuditedTextFile(
                relative_path=relative_path,
                sha256=sha256_file(path),
                fields=parsed.fields,
                parsed=parsed,
            )
        )
    audited_tuple = tuple(audited)
    configurations = _build_released_configurations(
        workspace_root,
        audited_tuple,
    )
    identities, reference_scripts = _audit_source_identities(workspace_root)
    return ConfigAuditReport(
        data_root=DATA_ROOT_RELATIVE,
        text_files=audited_tuple,
        configurations=configurations,
        identities=identities,
        reference_scripts=reference_scripts,
    )


@dataclass(frozen=True)
class DatasetSchemaLock:
    schema_version: int
    data_root: str
    identities: Mapping[str, Any]
    configurations: tuple[Mapping[str, Any], ...]
    arrays: tuple[Mapping[str, Any], ...]
    raw: Mapping[str, Any]

    @classmethod
    def from_json(cls, text: str) -> "DatasetSchemaLock":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise SchemaIdentityError(f"schema is not valid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise SchemaIdentityError("schema root must be a JSON object")
        try:
            schema_version = int(payload["schema_version"])
            data_root = str(payload["data_root"])
            identities = payload["identities"]
            configurations = payload["configurations"]
            arrays = payload["arrays"]
        except (KeyError, TypeError, ValueError) as error:
            raise SchemaIdentityError(
                f"schema is missing a required field: {error}"
            ) from error
        if schema_version != 1:
            raise SchemaIdentityError(
                f"unsupported schema version: {schema_version}"
            )
        if data_root != DATA_ROOT_RELATIVE:
            raise SchemaIdentityError(
                f"data root mismatch: expected {DATA_ROOT_RELATIVE}, got {data_root}"
            )
        if not isinstance(identities, dict):
            raise SchemaIdentityError("schema identities must be an object")
        if not isinstance(configurations, list) or not isinstance(arrays, list):
            raise SchemaIdentityError("configurations and arrays must be lists")
        return cls(
            schema_version=schema_version,
            data_root=data_root,
            identities=identities,
            configurations=tuple(configurations),
            arrays=tuple(arrays),
            raw=payload,
        )

    def validate_source_revisions(self) -> None:
        actual_dataset = self.identities.get("dataset_revision")
        if actual_dataset != DATASET_REVISION:
            raise SchemaIdentityError(
                "dataset revision mismatch: "
                f"expected {DATASET_REVISION}, got {actual_dataset}"
            )
        actual_reference = self.identities.get("reference_code_revision")
        if actual_reference != REFERENCE_CODE_REVISION:
            raise SchemaIdentityError(
                "reference-code revision mismatch: "
                f"expected {REFERENCE_CODE_REVISION}, got {actual_reference}"
            )

    def validate_unique_configuration_ids(self) -> None:
        seen: set[str] = set()
        for configuration in self.configurations:
            if not isinstance(configuration, Mapping):
                raise SchemaIdentityError("configuration record must be an object")
            value = configuration.get("configuration_id")
            if not isinstance(value, str) or not value:
                raise SchemaIdentityError(
                    "configuration_id must be a non-empty string"
                )
            if value in seen:
                raise DuplicateConfigurationError(
                    f"duplicate configuration_id: {value}"
                )
            seen.add(value)


def load_schema_lock(path: Path) -> DatasetSchemaLock:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise SchemaIdentityError(f"cannot read schema lock {path}: {error}") from error
    lock = DatasetSchemaLock.from_json(text)
    lock.validate_source_revisions()
    lock.validate_unique_configuration_ids()
    return lock


@dataclass(frozen=True)
class SampleInventory:
    workspace_root: Path
    height_paths: Mapping[str, tuple[str, ...]]
    beam_map_paths: Mapping[tuple[str, int], tuple[str, ...]]
    radiomap_paths: Mapping[tuple[str, int, str], tuple[str, ...]]

    def _require_one(
        self,
        candidates: tuple[str, ...] | None,
        *,
        label: str,
        key: object,
    ) -> Path:
        count = 0 if candidates is None else len(candidates)
        if count == 0:
            raise MissingSamplePathError(f"missing {label} for {key!r}")
        if count != 1:
            raise AmbiguousSamplePathError(
                f"ambiguous {label} for {key!r}: {count} files"
            )
        assert candidates is not None
        relative = PurePosixPath(candidates[0])
        if relative.is_absolute() or ".." in relative.parts:
            raise AmbiguousSamplePathError(
                f"unsafe {label} path for {key!r}: {candidates[0]}"
            )
        resolved = Path(self.workspace_root).joinpath(*relative.parts)
        if not resolved.is_file():
            raise MissingSamplePathError(
                f"missing {label} file for {key!r}: {resolved}"
            )
        return resolved

    def require_unique_triplet(
        self,
        config_id: str,
        beam_id: int,
        scene_id: str,
    ) -> tuple[Path, Path, Path]:
        height = self._require_one(
            self.height_paths.get(scene_id),
            label="height map",
            key=scene_id,
        )
        beam_map = self._require_one(
            self.beam_map_paths.get((config_id, beam_id)),
            label="beam map",
            key=(config_id, beam_id),
        )
        radiomap = self._require_one(
            self.radiomap_paths.get((config_id, beam_id, scene_id)),
            label="radiomap",
            key=(config_id, beam_id, scene_id),
        )
        return height, beam_map, radiomap


def resolve_sample_paths(
    inventory: SampleInventory,
    config_id: str,
    beam_id: int,
    scene_id: str,
) -> tuple[Path, Path, Path]:
    return inventory.require_unique_triplet(config_id, beam_id, scene_id)


def inventory_samples(
    workspace_root: Path,
    schema: DatasetSchemaLock,
) -> SampleInventory:
    workspace_root = Path(workspace_root).resolve()
    data_root = workspace_root.joinpath(*PurePosixPath(schema.data_root).parts)
    if not data_root.is_dir():
        raise MissingSamplePathError(f"data root is missing: {data_root}")

    height_paths: dict[str, list[str]] = {}
    for path in sorted(
        data_root.glob("height_maps/u*/u*_height_matrix.npy"),
        key=lambda value: value.relative_to(workspace_root).as_posix(),
    ):
        if not path.is_file():
            continue
        relative_path = path.relative_to(workspace_root).as_posix()
        match = _HEIGHT_PATTERN.fullmatch(relative_path)
        if match is None:
            continue
        height_paths.setdefault(match.group("scene_id"), []).append(
            relative_path
        )

    beam_map_paths: dict[tuple[str, int], list[str]] = {}
    beam_map_angles: dict[tuple[str, int], list[float]] = {}
    radiomap_paths: dict[tuple[str, int, str], list[str]] = {}
    expected_angle_by_key: dict[tuple[str, int], float] = {}
    selected_keys: list[tuple[str, int]] = []
    for array in schema.arrays:
        if not isinstance(array, Mapping):
            raise SchemaIdentityError("array record must be an object")
        configuration_id = array.get("configuration_id")
        selected_beams = array.get("selected_beams")
        if not isinstance(configuration_id, str) or not configuration_id:
            raise SchemaIdentityError(
                "array configuration_id must be a non-empty string"
            )
        if not isinstance(selected_beams, list):
            raise SchemaIdentityError("selected_beams must be a list")
        for beam in selected_beams:
            if not isinstance(beam, Mapping):
                raise SchemaIdentityError("selected beam must be an object")
            try:
                beam_id = int(beam["beam_id"])
                steering_deg = float(beam["steering_deg"])
            except (KeyError, TypeError, ValueError) as error:
                raise SchemaIdentityError(
                    f"invalid selected beam for {configuration_id}: {error}"
                ) from error
            key = (configuration_id, beam_id)
            if key in expected_angle_by_key:
                raise DuplicateConfigurationError(
                    f"duplicate selected beam key: {key!r}"
                )
            expected_angle_by_key[key] = steering_deg
            selected_keys.append(key)
            beam_directory = data_root / "beam_maps" / configuration_id / "u0"
            for path in sorted(
                beam_directory.glob(f"beam_{beam_id:02d}_angle_*_matrix.npy"),
                key=lambda value: value.name,
            ):
                if not path.is_file():
                    continue
                relative_path = path.relative_to(workspace_root).as_posix()
                match = _BEAM_MAP_PATTERN.fullmatch(relative_path)
                if match is None or int(match.group("beam_id")) != beam_id:
                    continue
                beam_map_paths.setdefault(key, []).append(relative_path)
                beam_map_angles.setdefault(key, []).append(
                    float(match.group("angle"))
                )
            radiomap_directory = (
                data_root
                / "radiomaps"
                / f"{configuration_id}_beam{beam_id:02d}"
            )
            for path in sorted(
                radiomap_directory.glob("u*_labeled_radiomap.npy"),
                key=lambda value: value.name,
            ):
                if not path.is_file():
                    continue
                relative_path = path.relative_to(workspace_root).as_posix()
                match = _RADIOMAP_PATTERN.fullmatch(relative_path)
                if match is None:
                    continue
                radiomap_key = (
                    configuration_id,
                    beam_id,
                    match.group("scene_id"),
                )
                radiomap_paths.setdefault(radiomap_key, []).append(
                    relative_path
                )

    inventory = SampleInventory(
        workspace_root=workspace_root,
        height_paths={
            key: tuple(values) for key, values in height_paths.items()
        },
        beam_map_paths={
            key: tuple(values) for key, values in beam_map_paths.items()
        },
        radiomap_paths={
            key: tuple(values) for key, values in radiomap_paths.items()
        },
    )
    if not height_paths:
        raise MissingSamplePathError("no released height maps were found")
    scene_ids = set(height_paths)
    for key in selected_keys:
        candidates = inventory.beam_map_paths.get(key)
        beam_path = inventory._require_one(
            candidates,
            label="beam map",
            key=key,
        )
        angles = beam_map_angles.get(key, [])
        if len(angles) != 1 or not math.isclose(
            angles[0],
            expected_angle_by_key[key],
            abs_tol=1e-9,
        ):
            raise ConfigurationMismatchError(
                f"locked angle mismatch for {key!r}: "
                f"expected {expected_angle_by_key[key]}, got {angles} "
                f"at {beam_path}"
            )
        radiomap_scene_ids = {
            scene_id
            for config_id, beam_id, scene_id in radiomap_paths
            if (config_id, beam_id) == key
        }
        if radiomap_scene_ids != scene_ids:
            missing = sorted(scene_ids - radiomap_scene_ids)
            extra = sorted(radiomap_scene_ids - scene_ids)
            raise MissingSamplePathError(
                f"radiomap scene mismatch for {key!r}: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        for scene_id in scene_ids:
            inventory.require_unique_triplet(key[0], key[1], scene_id)
    return inventory


def _inventory_schema(report: ConfigAuditReport) -> DatasetSchemaLock:
    selected = select_benchmark_configurations(report)
    arrays = tuple(
        {
            "name": array_name,
            "configuration_id": selected[array_name].configuration_id,
            "selected_beams": [asdict(beam) for beam in spec.beams],
        }
        for array_name, spec in ARRAY_SPECS.items()
    )
    raw = {
        "schema_version": 1,
        "data_root": report.data_root,
        "identities": dict(report.identities),
        "configurations": [
            configuration.to_dict()
            for configuration in report.configurations
        ],
        "arrays": list(arrays),
    }
    return DatasetSchemaLock(
        schema_version=1,
        data_root=report.data_root,
        identities=report.identities,
        configurations=tuple(raw["configurations"]),
        arrays=arrays,
        raw=raw,
    )


def _flatten_unique_paths(
    inventory: SampleInventory,
    values: Mapping[object, tuple[str, ...]],
    *,
    label: str,
) -> list[Path]:
    relative_paths: set[str] = set()
    for key, candidates in values.items():
        if len(candidates) != 1:
            if len(candidates) == 0:
                raise MissingSamplePathError(f"missing {label} for {key!r}")
            raise AmbiguousSamplePathError(
                f"ambiguous {label} for {key!r}: {len(candidates)} files"
            )
        relative_paths.add(candidates[0])
    return [
        Path(inventory.workspace_root).joinpath(
            *PurePosixPath(relative_path).parts
        )
        for relative_path in sorted(relative_paths)
    ]


def _inspect_array_group(
    paths: list[Path],
    *,
    label: str,
    progress: bool,
) -> dict[str, Any]:
    if not paths:
        raise MissingSamplePathError(f"no {label} files were indexed")
    shapes: set[tuple[int, ...]] = set()
    dtypes: set[str] = set()
    floor_count = 0
    building_count = 0
    valid_count = 0
    for index, path in enumerate(paths, start=1):
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError, TypeError) as error:
            raise ConfigurationFormatError(
                f"cannot load {label} NPY {path}: {error}"
            ) from error
        try:
            shapes.add(tuple(int(value) for value in array.shape))
            dtypes.add(str(array.dtype))
            if not bool(np.isfinite(array).all()):
                raise ConfigurationFormatError(
                    f"{label} contains a non-finite value: {path}"
                )
            if label == "height":
                if bool((array < 0).any()):
                    raise ConfigurationFormatError(
                        f"height contains a negative value: {path}"
                    )
            elif label == "radiomap":
                floor = array == -300.0
                building = array == 1000.0
                valid = (array > -300.0) & (array < 0.0)
                known = floor | building | valid
                if not bool(known.all()):
                    raise ConfigurationFormatError(
                        f"radiomap contains an unknown target value: {path}"
                    )
                if not bool(valid.any()):
                    raise ConfigurationFormatError(
                        f"radiomap has no valid propagation cells: {path}"
                    )
                floor_count += int(np.count_nonzero(floor))
                building_count += int(np.count_nonzero(building))
                valid_count += int(np.count_nonzero(valid))
        finally:
            del array
        if progress and (index % 1000 == 0 or index == len(paths)):
            print(
                json.dumps(
                    {
                        "phase": "inspect-npy",
                        "kind": label,
                        "completed": index,
                        "total": len(paths),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if len(shapes) != 1 or len(dtypes) != 1:
        raise ConfigurationFormatError(
            f"{label} metadata is not uniform: shapes={sorted(shapes)}, "
            f"dtypes={sorted(dtypes)}"
        )
    metadata: dict[str, Any] = {
        "shape": list(next(iter(shapes))),
        "dtype": next(iter(dtypes)),
        "files": len(paths),
    }
    if label == "radiomap":
        if floor_count == 0 or building_count == 0 or valid_count == 0:
            raise ConfigurationFormatError(
                "radiomap inventory does not contain every locked target class"
            )
        metadata["value_counts"] = {
            "floor": floor_count,
            "building": building_count,
            "valid": valid_count,
        }
    return metadata


def inspect_source_inventory(
    inventory: SampleInventory,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    groups = {
        "height": _flatten_unique_paths(
            inventory,
            inventory.height_paths,
            label="height",
        ),
        "beam_map": _flatten_unique_paths(
            inventory,
            inventory.beam_map_paths,
            label="beam map",
        ),
        "radiomap": _flatten_unique_paths(
            inventory,
            inventory.radiomap_paths,
            label="radiomap",
        ),
    }
    metadata = {
        label: _inspect_array_group(
            paths,
            label=label,
            progress=progress,
        )
        for label, paths in groups.items()
    }
    expected = {
        "height": ([256, 256], "float32"),
        "beam_map": ([128, 128], "float64"),
        "radiomap": ([128, 128], "float32"),
    }
    for label, (shape, dtype) in expected.items():
        if metadata[label]["shape"] != shape or metadata[label]["dtype"] != dtype:
            raise ConfigurationMismatchError(
                f"unexpected {label} source metadata: {metadata[label]}"
            )
    return metadata


def write_audit_report(
    dataset_root: Path,
    report_path: Path,
) -> ConfigAuditReport:
    report = audit_config_files(dataset_root)
    _write_immutable_json(Path(report_path), report.to_dict())
    return report


def freeze_schema_lock(
    dataset_root: Path,
    audit_report_path: Path,
    output_path: Path,
    *,
    progress: bool = False,
) -> DatasetSchemaLock:
    workspace_root = Path(dataset_root).resolve()
    audit_report_path = Path(audit_report_path).resolve()
    try:
        audit_relative_path = audit_report_path.relative_to(
            workspace_root
        ).as_posix()
    except ValueError as error:
        raise SchemaIdentityError(
            "audit report must be stored below the dataset workspace root"
        ) from error
    supplied_payload = _read_json_object(
        audit_report_path,
        label="configuration audit report",
    )
    live_report = audit_config_files(workspace_root)
    if not canonical_payloads_equal(supplied_payload, live_report.to_dict()):
        raise SchemaIdentityError(
            "configuration audit report does not match the current pinned data"
        )
    if not live_report.identities or not live_report.reference_scripts:
        raise SchemaIdentityError(
            "real source identities and reference scripts are required to freeze schema"
        )
    inventory = inventory_samples(
        workspace_root,
        _inventory_schema(live_report),
    )
    expected_scenes = {f"u{index}" for index in range(1, 801)}
    actual_scenes = set(inventory.height_paths)
    if actual_scenes != expected_scenes:
        missing = sorted(expected_scenes - actual_scenes)
        extra = sorted(actual_scenes - expected_scenes)
        raise MissingSamplePathError(
            "released scene domain mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    source_metadata = inspect_source_inventory(
        inventory,
        progress=progress,
    )
    payload = assemble_schema_payload(
        live_report,
        audit_report_relative_path=audit_relative_path,
        audit_report_sha256=sha256_file(audit_report_path),
        source_metadata=source_metadata,
        scene_count=len(actual_scenes),
    )
    _write_immutable_json(Path(output_path), payload)
    return load_schema_lock(output_path)


def verify_schema_lock(
    dataset_root: Path,
    schema_path: Path,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    workspace_root = Path(dataset_root).resolve()
    schema = load_schema_lock(schema_path)
    live_report = audit_config_files(workspace_root)
    audit_relative = schema.identities.get("audit_report_relative_path")
    audit_sha256 = schema.identities.get("audit_report_sha256")
    if not isinstance(audit_relative, str) or not isinstance(audit_sha256, str):
        raise SchemaIdentityError("schema has no locked audit report identity")
    audit_path = workspace_root.joinpath(*PurePosixPath(audit_relative).parts)
    if not audit_path.is_file() or sha256_file(audit_path) != audit_sha256:
        raise SchemaIdentityError("locked audit report is missing or changed")
    if not canonical_payloads_equal(
        _read_json_object(audit_path, label="configuration audit report"),
        live_report.to_dict(),
    ):
        raise SchemaIdentityError(
            "locked audit report no longer matches current source evidence"
        )
    inventory = inventory_samples(workspace_root, schema)
    source_metadata = inspect_source_inventory(
        inventory,
        progress=progress,
    )
    expected_payload = assemble_schema_payload(
        live_report,
        audit_report_relative_path=audit_relative,
        audit_report_sha256=audit_sha256,
        source_metadata=source_metadata,
        scene_count=len(inventory.height_paths),
    )
    if not canonical_payloads_equal(dict(schema.raw), expected_payload):
        raise SchemaIdentityError(
            "schema lock does not match the current audited dataset"
        )
    return {
        "schema": str(Path(schema_path).resolve()),
        "scene_count": len(inventory.height_paths),
        "array_count": len(schema.arrays),
        "height_files": source_metadata["height"]["files"],
        "beam_map_files": source_metadata["beam_map"]["files"],
        "radiomap_files": source_metadata["radiomap"]["files"],
        "archive_sha256": schema.identities.get("archive_sha256"),
    }
