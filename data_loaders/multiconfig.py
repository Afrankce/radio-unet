from __future__ import annotations

import hashlib
import io
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from experiments.multiconfig_manifest import (
    DatasetSchemaLock,
    ManifestRecord,
    canonical_json_bytes,
    load_manifest_jsonl,
    load_schema_lock,
)
from experiments.provenance import sha256_file


OUTPUT_SIZE = (256, 256)
HEIGHT_STATS_NAME = "height_stats_train.json"
SPLIT_NAME = "scene_split_seed42.json"
ARRAY_NAMES = ("8x8", "16x16", "32x32")


class DataFormatError(RuntimeError):
    """A released NPY file does not satisfy its locked representation."""


class TargetValueError(DataFormatError):
    """A target contains a value outside the locked radio-map domain."""


class EmptyValidMaskError(DataFormatError):
    """A target contains no propagation pixel on which to train or evaluate."""


class HeightStatsContractError(RuntimeError):
    """The train-only height-normalization artifact is missing or inconsistent."""


class DatasetContractError(RuntimeError):
    """A manifest record disagrees with the benchmark schema or split."""


@dataclass(frozen=True)
class HeightFileEvidence:
    scene_id: str
    relative_path: str
    sha256: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HeightFileEvidence":
        expected = {"scene_id", "relative_path", "sha256"}
        if set(payload) != expected:
            raise HeightStatsContractError(
                "height evidence keys mismatch: "
                f"missing={sorted(expected - set(payload))}, "
                f"extra={sorted(set(payload) - expected)}"
            )
        return cls(
            scene_id=str(payload["scene_id"]),
            relative_path=str(payload["relative_path"]),
            sha256=str(payload["sha256"]),
        )


@dataclass(frozen=True)
class TrainHeightSummary:
    height_max: float
    scene_count: int
    height_files: tuple[HeightFileEvidence, ...]


@dataclass(frozen=True)
class HeightStats:
    schema_version: int
    height_max: float
    derived_from: str
    scene_count: int
    height_files: tuple[HeightFileEvidence, ...]
    split_sha256: str
    manifest_sha256: Mapping[str, str]
    schema_sha256: str
    schema_identity: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "height_max": self.height_max,
            "derived_from": self.derived_from,
            "scene_count": self.scene_count,
            "height_files": [asdict(item) for item in self.height_files],
            "split_sha256": self.split_sha256,
            "manifest_sha256": dict(self.manifest_sha256),
            "schema_sha256": self.schema_sha256,
            "schema_identity": dict(self.schema_identity),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HeightStats":
        expected = {
            "schema_version",
            "height_max",
            "derived_from",
            "scene_count",
            "height_files",
            "split_sha256",
            "manifest_sha256",
            "schema_sha256",
            "schema_identity",
        }
        if set(payload) != expected:
            raise HeightStatsContractError(
                "height statistics keys mismatch: "
                f"missing={sorted(expected - set(payload))}, "
                f"extra={sorted(set(payload) - expected)}"
            )
        try:
            files_payload = payload["height_files"]
            manifests = payload["manifest_sha256"]
            identity = payload["schema_identity"]
            if not isinstance(files_payload, list):
                raise TypeError("height_files must be a list")
            if not isinstance(manifests, dict):
                raise TypeError("manifest_sha256 must be an object")
            if not isinstance(identity, dict):
                raise TypeError("schema_identity must be an object")
            return cls(
                schema_version=int(payload["schema_version"]),
                height_max=float(payload["height_max"]),
                derived_from=str(payload["derived_from"]),
                scene_count=int(payload["scene_count"]),
                height_files=tuple(
                    HeightFileEvidence.from_dict(item) for item in files_payload
                ),
                split_sha256=str(payload["split_sha256"]),
                manifest_sha256={str(key): str(value) for key, value in manifests.items()},
                schema_sha256=str(payload["schema_sha256"]),
                schema_identity=identity,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise HeightStatsContractError(
                f"invalid height statistics: {error}"
            ) from error


def _require_db_interval(floor_db: float, ceiling_db: float) -> tuple[float, float]:
    floor = float(floor_db)
    ceiling = float(ceiling_db)
    if not math.isfinite(floor) or not math.isfinite(ceiling) or floor >= ceiling:
        raise DataFormatError(
            f"invalid dB interval: floor={floor_db}, ceiling={ceiling_db}"
        )
    return floor, ceiling


def normalize_db(
    values: torch.Tensor,
    floor_db: float = -300.0,
    ceiling_db: float = 0.0,
) -> torch.Tensor:
    """Map a locked dB interval linearly to [0, 1]."""

    floor, ceiling = _require_db_interval(floor_db, ceiling_db)
    return (values.clamp(min=floor, max=ceiling) - floor) / (ceiling - floor)


def denormalize_db(
    values: torch.Tensor,
    floor_db: float = -300.0,
    ceiling_db: float = 0.0,
) -> torch.Tensor:
    """Invert :func:`normalize_db` for values in [0, 1]."""

    floor, ceiling = _require_db_interval(floor_db, ceiling_db)
    return values * (ceiling - floor) + floor


def prepare_target(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a radio map and return its propagation-pixel validity mask."""

    values = values.to(dtype=torch.float32)
    if not bool(torch.isfinite(values).all()):
        raise TargetValueError("target contains a non-finite value")
    floor = values == -300.0
    building = values == 1000.0
    valid = (values > -300.0) & (values < 0.0)
    known = floor | building | valid
    if not bool(known.all()):
        unknown = values[~known]
        preview = unknown.flatten()[:5].tolist()
        raise TargetValueError(f"target contains unknown values: {preview}")
    if not bool(valid.any()):
        raise EmptyValidMaskError("target has no valid propagation pixels")
    target = torch.zeros_like(values, dtype=torch.float32)
    target[valid] = normalize_db(values[valid])
    return target, valid


def resize_continuous(
    values: torch.Tensor,
    size: tuple[int, int] = OUTPUT_SIZE,
) -> torch.Tensor:
    if values.ndim != 4:
        raise DataFormatError(
            f"continuous resize expects NCHW input, got shape {tuple(values.shape)}"
        )
    return F.interpolate(values, size=size, mode="bilinear", align_corners=False)


def resize_valid_mask(
    mask: torch.Tensor,
    size: tuple[int, int] = OUTPUT_SIZE,
) -> torch.Tensor:
    if mask.ndim != 4:
        raise DataFormatError(
            f"mask resize expects NCHW input, got shape {tuple(mask.shape)}"
        )
    return F.interpolate(mask.float(), size=size, mode="nearest").bool()


def build_tx_mask(
    size: tuple[int, int] = OUTPUT_SIZE,
    tx_rc: tuple[int, int] = (127, 127),
) -> torch.Tensor:
    row, column = tx_rc
    height, width = size
    if not (0 <= row < height and 0 <= column < width):
        raise DatasetContractError(
            f"transmitter pixel {tx_rc} is outside output size {size}"
        )
    mask = torch.zeros((1, height, width), dtype=torch.float32)
    mask[0, row, column] = 1.0
    return mask


def _safe_path(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise DatasetContractError(f"unsafe manifest path: {relative_path!r}")
    root = Path(root).resolve()
    path = root.joinpath(*pure.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise DatasetContractError(
            f"manifest path escapes dataset root: {relative_path!r}"
        ) from error
    return path


def _load_npy(
    path: Path,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
    label: str,
) -> np.ndarray:
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError, TypeError) as error:
        raise DataFormatError(f"cannot load {label} NPY {path}: {error}") from error
    if not isinstance(array, np.ndarray):
        raise DataFormatError(f"{label} NPY {path} is not an ndarray")
    if tuple(array.shape) != expected_shape:
        raise DataFormatError(
            f"{label} NPY {path} shape mismatch: "
            f"expected {expected_shape}, got {tuple(array.shape)}"
        )
    if str(array.dtype) != expected_dtype:
        raise DataFormatError(
            f"{label} NPY {path} dtype mismatch: "
            f"expected {expected_dtype}, got {array.dtype}"
        )
    if not bool(np.isfinite(array).all()):
        raise DataFormatError(f"{label} NPY {path} contains a non-finite value")
    return array


def _scene_sort_key(scene_id: str) -> tuple[str, int | str]:
    prefix = scene_id.rstrip("0123456789")
    suffix = scene_id[len(prefix) :]
    return (prefix, int(suffix) if suffix else scene_id)


def compute_train_height_max(
    dataset_root: Path,
    records: Sequence[ManifestRecord],
) -> TrainHeightSummary:
    """Read every unique train height file once and retain hash evidence."""

    if not records:
        raise HeightStatsContractError("no train records were supplied")
    scene_paths: dict[str, str] = {}
    for record in records:
        if record.split != "train":
            raise HeightStatsContractError(
                f"height maximum may only use train records, got {record.split!r}"
            )
        previous = scene_paths.setdefault(record.scene_id, record.height_path)
        if previous != record.height_path:
            raise HeightStatsContractError(
                f"scene {record.scene_id} has more than one height path"
            )
    if len(scene_paths) != 560:
        raise HeightStatsContractError(
            f"expected 560 unique train scenes, got {len(scene_paths)}"
        )

    maximum = 0.0
    evidence: list[HeightFileEvidence] = []
    for scene_id in sorted(scene_paths, key=_scene_sort_key):
        relative_path = scene_paths[scene_id]
        path = _safe_path(dataset_root, relative_path)
        try:
            payload = path.read_bytes()
            array = np.load(io.BytesIO(payload), allow_pickle=False)
        except (OSError, ValueError, TypeError) as error:
            raise DataFormatError(f"cannot load height NPY {path}: {error}") from error
        if not isinstance(array, np.ndarray) or array.ndim != 2:
            raise DataFormatError(
                f"height NPY {path} must be a two-dimensional ndarray"
            )
        if array.dtype.kind not in "fiu":
            raise DataFormatError(f"height NPY {path} is not numeric: {array.dtype}")
        if not bool(np.isfinite(array).all()):
            raise DataFormatError(f"height NPY {path} contains a non-finite value")
        if bool((array < 0).any()):
            raise DataFormatError(f"height NPY {path} contains a negative value")
        maximum = max(maximum, float(array.max(initial=0.0)))
        evidence.append(
            HeightFileEvidence(
                scene_id=scene_id,
                relative_path=relative_path,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise HeightStatsContractError(
            f"train height maximum must be finite and positive, got {maximum}"
        )
    return TrainHeightSummary(
        height_max=maximum,
        scene_count=len(scene_paths),
        height_files=tuple(evidence),
    )


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    expected = canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != expected:
            raise HeightStatsContractError(
                f"immutable height statistics already differ: {path}"
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


def compute_height_stats_artifact(
    dataset_root: Path,
    manifest_dir: Path,
    schema_path: Path,
) -> tuple[Path, HeightStats]:
    dataset_root = Path(dataset_root).resolve()
    manifest_dir = Path(manifest_dir).resolve()
    schema_path = Path(schema_path).resolve()
    schema = load_schema_lock(schema_path)

    manifests: dict[str, Path] = {}
    train_records: dict[str, tuple[ManifestRecord, ...]] = {}
    scene_height_maps: dict[str, dict[str, str]] = {}
    for array_name in ARRAY_NAMES:
        path = manifest_dir / f"manifest_{array_name}.jsonl"
        records = load_manifest_jsonl(path)
        selected = tuple(record for record in records if record.split == "train")
        if len(selected) != 4480:
            raise HeightStatsContractError(
                f"{array_name} expected 4480 train samples, got {len(selected)}"
            )
        scene_map = {record.scene_id: record.height_path for record in selected}
        if len(scene_map) != 560:
            raise HeightStatsContractError(
                f"{array_name} expected 560 train scenes, got {len(scene_map)}"
            )
        counts: dict[str, int] = {}
        for record in selected:
            counts[record.scene_id] = counts.get(record.scene_id, 0) + 1
            if scene_map[record.scene_id] != record.height_path:
                raise HeightStatsContractError(
                    f"{array_name}/{record.scene_id} has inconsistent height paths"
                )
        if set(counts.values()) != {8}:
            raise HeightStatsContractError(
                f"{array_name} does not contain exactly eight train beams per scene"
            )
        manifests[array_name] = path
        train_records[array_name] = selected
        scene_height_maps[array_name] = scene_map
    baseline = scene_height_maps[ARRAY_NAMES[0]]
    for array_name in ARRAY_NAMES[1:]:
        if scene_height_maps[array_name] != baseline:
            raise HeightStatsContractError(
                f"train scene/height mapping differs for {array_name}"
            )

    summary = compute_train_height_max(dataset_root, train_records[ARRAY_NAMES[0]])
    split_path = manifest_dir / SPLIT_NAME
    if not split_path.is_file():
        raise HeightStatsContractError(f"fixed split file is missing: {split_path}")
    stats = HeightStats(
        schema_version=1,
        height_max=summary.height_max,
        derived_from="train",
        scene_count=summary.scene_count,
        height_files=summary.height_files,
        split_sha256=sha256_file(split_path),
        manifest_sha256={
            path.name: sha256_file(path) for path in manifests.values()
        },
        schema_sha256=hashlib.sha256(canonical_json_bytes(schema.raw)).hexdigest(),
        schema_identity=dict(schema.identities),
    )
    output_path = manifest_dir / HEIGHT_STATS_NAME
    _write_immutable_json(output_path, stats.to_dict())
    return output_path, stats


def load_height_stats(path: Path) -> HeightStats:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HeightStatsContractError(
            f"cannot read height statistics {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise HeightStatsContractError("height statistics root must be an object")
    return HeightStats.from_dict(payload)


def _validate_schema_evidence(dataset_root: Path, schema: DatasetSchemaLock) -> None:
    schema.validate_source_revisions()
    data_root = _safe_path(dataset_root, schema.data_root)
    if not data_root.is_dir():
        raise DatasetContractError(f"locked dataset directory is missing: {data_root}")

    # These files are small identity anchors.  The 4.3 GB archive itself is not
    # rehashed in every Dataset constructor; its receipt and the canonical schema
    # are bound into height_stats_train.json instead.
    for prefix in ("download_receipt", "extraction_receipt", "audit_report"):
        relative = schema.identities.get(f"{prefix}_relative_path")
        expected = schema.identities.get(f"{prefix}_sha256")
        if relative is None and expected is None:
            continue
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise DatasetContractError(f"incomplete {prefix} identity in schema")
        path = _safe_path(dataset_root, relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise DatasetContractError(f"locked {prefix} is missing or changed: {path}")
    scripts = schema.raw.get("reference_scripts", [])
    if not isinstance(scripts, list):
        raise DatasetContractError("schema reference_scripts must be a list")
    for item in scripts:
        if not isinstance(item, Mapping):
            raise DatasetContractError("invalid reference script evidence")
        path = _safe_path(dataset_root, str(item.get("relative_path", "")))
        expected = item.get("sha256")
        if not isinstance(expected, str) or not path.is_file() or sha256_file(path) != expected:
            raise DatasetContractError(f"locked reference script is missing or changed: {path}")


def _validate_height_stats(
    stats: HeightStats,
    *,
    manifest_path: Path,
    schema: DatasetSchemaLock,
) -> None:
    if stats.schema_version != 1 or stats.derived_from != "train":
        raise HeightStatsContractError("height statistics are not train-only schema v1")
    if not math.isfinite(stats.height_max) or stats.height_max <= 0.0:
        raise HeightStatsContractError("height maximum must be finite and positive")
    if stats.scene_count != 560 or len(stats.height_files) != 560:
        raise HeightStatsContractError("height statistics must contain 560 train scenes")
    scenes = [item.scene_id for item in stats.height_files]
    paths = [item.relative_path for item in stats.height_files]
    if len(set(scenes)) != 560 or len(set(paths)) != 560:
        raise HeightStatsContractError("height evidence must be unique by scene and path")
    for item in stats.height_files:
        if len(item.sha256) != 64:
            raise HeightStatsContractError(
                f"invalid height evidence digest for {item.scene_id}"
            )
    expected_manifest = stats.manifest_sha256.get(Path(manifest_path).name)
    if expected_manifest != sha256_file(manifest_path):
        raise HeightStatsContractError(
            f"height statistics do not bind current manifest: {manifest_path}"
        )
    split_path = Path(manifest_path).parent / SPLIT_NAME
    if not split_path.is_file() or stats.split_sha256 != sha256_file(split_path):
        raise HeightStatsContractError("height statistics do not bind the fixed split")
    schema_sha256 = hashlib.sha256(canonical_json_bytes(schema.raw)).hexdigest()
    if stats.schema_sha256 != schema_sha256:
        raise HeightStatsContractError("height statistics do not bind the schema lock")
    dataset_revision = schema.identities.get("dataset_revision")
    if stats.schema_identity.get("dataset_revision") != dataset_revision:
        raise HeightStatsContractError("height statistics dataset revision mismatch")
    if "archive_sha256" in schema.identities:
        if stats.schema_identity.get("archive_sha256") != schema.identities["archive_sha256"]:
            raise HeightStatsContractError("height statistics archive identity mismatch")


def _schema_array(schema: DatasetSchemaLock, name: str) -> Mapping[str, Any]:
    matches = [item for item in schema.arrays if item.get("name") == name]
    if len(matches) != 1:
        raise DatasetContractError(
            f"schema must contain exactly one array named {name!r}"
        )
    return matches[0]


def _validate_record(record: ManifestRecord, schema: DatasetSchemaLock) -> None:
    array = _schema_array(schema, record.array_name)
    expected = {
        "array_rows": int(array.get("rows", -1)),
        "array_cols": int(array.get("cols", -1)),
        "frequency_hz": int(array.get("frequency_hz", -1)),
        "config_id": str(array.get("configuration_id", "")),
    }
    for field, wanted in expected.items():
        if getattr(record, field) != wanted:
            raise DatasetContractError(
                f"record {record.sample_key} {field} mismatch: "
                f"expected {wanted!r}, got {getattr(record, field)!r}"
            )
    beams = array.get("selected_beams")
    if not isinstance(beams, list):
        raise DatasetContractError("schema selected_beams must be a list")
    matches = [item for item in beams if int(item.get("beam_id", -1)) == record.beam_id]
    if len(matches) != 1 or not math.isclose(
        float(matches[0].get("steering_deg", math.nan)),
        record.steering_deg,
        abs_tol=1e-9,
    ):
        raise DatasetContractError(
            f"record {record.sample_key} beam is not selected by the schema"
        )


class MultiConfigRadiomapDataset(Dataset):
    """Strict RadioFlow adapter for one fixed array and one fixed scene split."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        manifest_path: Path,
        split: str,
        schema: DatasetSchemaLock,
        height_stats: HeightStats,
        output_size: tuple[int, int] = OUTPUT_SIZE,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise DatasetContractError(f"invalid split: {split!r}")
        self.dataset_root = Path(dataset_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.schema = schema
        self.height_stats = height_stats
        self.split = split
        if tuple(output_size) != OUTPUT_SIZE:
            raise DatasetContractError(
                f"fixed output size must be {OUTPUT_SIZE}, got {tuple(output_size)}"
            )
        self.output_size = OUTPUT_SIZE
        _validate_schema_evidence(self.dataset_root, schema)
        _validate_height_stats(
            height_stats,
            manifest_path=self.manifest_path,
            schema=schema,
        )
        records = tuple(
            record
            for record in load_manifest_jsonl(self.manifest_path)
            if record.split == split
        )
        if not records:
            raise DatasetContractError(
                f"manifest {self.manifest_path} has no {split} samples"
            )
        sample_keys = [record.sample_key for record in records]
        logical = [
            (record.scene_id, record.array_name, record.beam_id)
            for record in records
        ]
        if len(sample_keys) != len(set(sample_keys)) or len(logical) != len(set(logical)):
            raise DatasetContractError("selected manifest split contains duplicates")
        for record in records:
            _validate_record(record, schema)
            for relative in (
                record.height_path,
                record.beam_map_path,
                record.radiomap_path,
            ):
                _safe_path(self.dataset_root, relative)
        self.records = records

        metadata = schema.raw.get("source_metadata")
        if not isinstance(metadata, Mapping):
            raise DatasetContractError("schema source_metadata must be an object")
        self.source_metadata = metadata
        transmitter = schema.raw.get("transmitter")
        if not isinstance(transmitter, Mapping):
            raise DatasetContractError("schema transmitter must be an object")
        tx = transmitter.get("output_pixel_rc")
        if not isinstance(tx, list) or len(tx) != 2:
            raise DatasetContractError("schema transmitter pixel must have two values")
        self.tx_rc = (int(tx[0]), int(tx[1]))
        if self.tx_rc != (127, 127):
            raise DatasetContractError(
                f"benchmark transmitter pixel must be (127, 127), got {self.tx_rc}"
            )
        self.tx_mask = build_tx_mask(self.output_size, self.tx_rc)

    def __len__(self) -> int:
        return len(self.records)

    def _source_contract(self, label: str) -> tuple[tuple[int, ...], str]:
        value = self.source_metadata.get(label)
        if not isinstance(value, Mapping):
            raise DatasetContractError(f"missing {label} source metadata")
        shape = value.get("shape")
        dtype = value.get("dtype")
        if not isinstance(shape, list) or not isinstance(dtype, str):
            raise DatasetContractError(f"invalid {label} source metadata")
        return tuple(int(item) for item in shape), dtype

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        height_shape, height_dtype = self._source_contract("height")
        beam_shape, beam_dtype = self._source_contract("beam_map")
        target_shape, target_dtype = self._source_contract("radiomap")
        height_path = _safe_path(self.dataset_root, record.height_path)
        beam_path = _safe_path(self.dataset_root, record.beam_map_path)
        target_path = _safe_path(self.dataset_root, record.radiomap_path)

        height_array = _load_npy(
            height_path,
            expected_shape=height_shape,
            expected_dtype=height_dtype,
            label="height",
        )
        if bool((height_array < 0).any()):
            raise DataFormatError(f"height NPY {height_path} contains a negative value")
        height = torch.from_numpy(height_array.astype(np.float32, copy=False)).unsqueeze(0)
        height = height / float(self.height_stats.height_max)
        if tuple(height.shape[-2:]) != self.output_size:
            height = resize_continuous(height.unsqueeze(0), self.output_size)[0]

        beam_array = _load_npy(
            beam_path,
            expected_shape=beam_shape,
            expected_dtype=beam_dtype,
            label="beam map",
        )
        beam = torch.from_numpy(beam_array.astype(np.float32, copy=False))
        beam = normalize_db(beam).unsqueeze(0).unsqueeze(0)
        beam = resize_continuous(beam, self.output_size)[0]

        target_array = _load_npy(
            target_path,
            expected_shape=target_shape,
            expected_dtype=target_dtype,
            label="radiomap",
        )
        target_source, valid_source = prepare_target(torch.from_numpy(target_array))
        target = resize_continuous(
            target_source.unsqueeze(0).unsqueeze(0), self.output_size
        )[0]
        valid_mask = resize_valid_mask(
            valid_source.unsqueeze(0).unsqueeze(0), self.output_size
        )[0]
        target = target.masked_fill(~valid_mask, 0.0)

        condition = torch.cat((self.tx_mask, height, beam), dim=0).contiguous()
        metadata = {
            **record.to_dict(),
            "tx_rc": [self.tx_rc[0], self.tx_rc[1]],
        }
        return {
            "condition": condition.to(dtype=torch.float32),
            "target": target.to(dtype=torch.float32).contiguous(),
            "valid_mask": valid_mask.to(dtype=torch.bool).contiguous(),
            "metadata": metadata,
        }


def multiconfig_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise DatasetContractError("cannot collate an empty sample list")
    return {
        "condition": torch.stack([sample["condition"] for sample in samples]),
        "target": torch.stack([sample["target"] for sample in samples]),
        "valid_mask": torch.stack([sample["valid_mask"] for sample in samples]),
        "metadata": [sample["metadata"] for sample in samples],
    }
