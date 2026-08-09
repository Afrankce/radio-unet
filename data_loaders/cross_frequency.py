from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from data_loaders.multiconfig import (
    OUTPUT_SIZE,
    DataFormatError,
    DatasetContractError,
    EmptyValidMaskError,
    _load_npy,
    _safe_path,
    build_tx_mask,
    load_height_stats,
    multiconfig_collate,
    normalize_db,
    prepare_target,
    resize_continuous,
    resize_valid_mask,
)
from experiments.cross_frequency import (
    TEST_FREQUENCY_HZ,
    TRAIN_FREQUENCY_HZ,
    cross_frequency_spec,
)
from experiments.multiconfig_manifest import ManifestRecord, load_manifest_jsonl
from experiments.provenance import sha256_file


class CrossFrequencyDatasetError(RuntimeError):
    """A cross-frequency sample or normalization artifact violates its contract."""


DEFAULT_SOURCE_METADATA: dict[str, dict[str, Any]] = {
    "height": {"shape": [256, 256], "dtype": "float32"},
    "beam_map": {"shape": [128, 128], "dtype": "float64"},
    "radiomap": {"shape": [128, 128], "dtype": "float32"},
}


def load_cross_frequency_height_max(
    path: Path,
    *,
    split_path: Path | None = None,
) -> float:
    """Load the fixed train-only height normalization maximum."""

    try:
        stats = load_height_stats(Path(path))
    except Exception as error:
        raise CrossFrequencyDatasetError(
            f"cannot load cross-frequency height statistics {path}: {error}"
        ) from error
    if stats.schema_version != 1 or stats.derived_from != "train":
        raise CrossFrequencyDatasetError(
            "height statistics must be schema v1 and derived from train"
        )
    if stats.scene_count != 560 or len(stats.height_files) != 560:
        raise CrossFrequencyDatasetError(
            "height statistics must contain 560 train scenes"
        )
    if not math.isfinite(stats.height_max) or stats.height_max <= 0.0:
        raise CrossFrequencyDatasetError("height maximum must be finite and positive")
    scenes = [item.scene_id for item in stats.height_files]
    if len(set(scenes)) != 560:
        raise CrossFrequencyDatasetError("height evidence contains duplicate scenes")
    if split_path is not None:
        split_path = Path(split_path)
        if not split_path.is_file() or stats.split_sha256 != sha256_file(split_path):
            raise CrossFrequencyDatasetError(
                "height statistics are not bound to scene_split_seed42.json"
            )
    return float(stats.height_max)


def _source_contract(
    source_metadata: Mapping[str, Any],
    label: str,
) -> tuple[tuple[int, ...], str]:
    value = source_metadata.get(label)
    if not isinstance(value, Mapping):
        raise CrossFrequencyDatasetError(f"missing {label} source metadata")
    shape = value.get("shape")
    dtype = value.get("dtype")
    if not isinstance(shape, (list, tuple)) or not isinstance(dtype, str):
        raise CrossFrequencyDatasetError(f"invalid {label} source metadata")
    return tuple(int(item) for item in shape), dtype


def _expected_frequency(split: str) -> int:
    return TEST_FREQUENCY_HZ if split == "test" else TRAIN_FREQUENCY_HZ


def _validate_record(
    record: ManifestRecord,
    *,
    split: str,
    expected_frequency_hz: int,
) -> None:
    if record.split != split:
        raise CrossFrequencyDatasetError("record split does not match selected dataset split")
    if record.array_name != "8x8" or (record.array_rows, record.array_cols) != (8, 8):
        raise CrossFrequencyDatasetError(
            f"record {record.sample_key} is not an 8x8 sample"
        )
    if record.frequency_hz != expected_frequency_hz:
        raise CrossFrequencyDatasetError(
            f"record {record.sample_key} frequency mismatch: "
            f"expected {expected_frequency_hz}, got {record.frequency_hz}"
        )
    if not math.isclose(record.steering_deg, 0.0, abs_tol=1e-9):
        raise CrossFrequencyDatasetError(
            f"record {record.sample_key} must use the zero-degree beam"
        )
    expected_beam_id = 0 if expected_frequency_hz == TRAIN_FREQUENCY_HZ else 4
    if record.beam_id != expected_beam_id:
        raise CrossFrequencyDatasetError(
            f"record {record.sample_key} has beam ID {record.beam_id}; "
            f"expected {expected_beam_id} for {expected_frequency_hz} Hz zero degrees"
        )


class CrossFrequencyRadiomapDataset(Dataset):
    """Decode the locked one-beam-per-frequency cross-frequency manifest."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        manifest_path: Path,
        split: str,
        height_max: float,
        expected_frequency_hz: int | None = None,
        expected_counts: Mapping[str, int] | None = None,
        source_metadata: Mapping[str, Any] | None = None,
        output_size: tuple[int, int] = OUTPUT_SIZE,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise CrossFrequencyDatasetError(f"invalid split: {split!r}")
        if tuple(output_size) != OUTPUT_SIZE:
            raise CrossFrequencyDatasetError(
                f"fixed output size must be {OUTPUT_SIZE}, got {tuple(output_size)}"
            )
        if not math.isfinite(float(height_max)) or float(height_max) <= 0.0:
            raise CrossFrequencyDatasetError("height_max must be finite and positive")
        self.dataset_root = Path(dataset_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.split = split
        self.height_max = float(height_max)
        self.output_size = OUTPUT_SIZE
        self.expected_frequency_hz = (
            _expected_frequency(split)
            if expected_frequency_hz is None
            else int(expected_frequency_hz)
        )
        if self.expected_frequency_hz not in (TRAIN_FREQUENCY_HZ, TEST_FREQUENCY_HZ):
            raise CrossFrequencyDatasetError(
                f"unsupported cross-frequency value {self.expected_frequency_hz}"
            )
        counts = dict(expected_counts or {
            "train": cross_frequency_spec().train_samples,
            "val": cross_frequency_spec().val_samples,
            "test": cross_frequency_spec().test_samples,
        })
        if set(counts) != {"train", "val", "test"} or any(
            isinstance(value, bool) or int(value) <= 0 for value in counts.values()
        ):
            raise CrossFrequencyDatasetError("expected_counts must define positive train/val/test counts")
        try:
            all_records = load_manifest_jsonl(self.manifest_path)
        except Exception as error:
            raise CrossFrequencyDatasetError(
                f"cannot load cross-frequency manifest {self.manifest_path}: {error}"
            ) from error
        records = tuple(record for record in all_records if record.split == split)
        if len(records) != int(counts[split]):
            raise CrossFrequencyDatasetError(
                f"{split} sample count mismatch: expected {counts[split]}, got {len(records)}"
            )
        if not records:
            raise CrossFrequencyDatasetError(
                f"manifest {self.manifest_path} has no {split} samples"
            )
        sample_keys = [record.sample_key for record in records]
        logical_keys = [(record.scene_id, record.frequency_hz) for record in records]
        if len(sample_keys) != len(set(sample_keys)) or len(logical_keys) != len(set(logical_keys)):
            raise CrossFrequencyDatasetError("selected split contains duplicate samples")
        for record in records:
            try:
                _validate_record(
                    record,
                    split=split,
                    expected_frequency_hz=self.expected_frequency_hz,
                )
                for relative_path in (
                    record.height_path,
                    record.beam_map_path,
                    record.radiomap_path,
                ):
                    _safe_path(self.dataset_root, relative_path)
            except (DatasetContractError, CrossFrequencyDatasetError) as error:
                if isinstance(error, CrossFrequencyDatasetError):
                    raise
                raise CrossFrequencyDatasetError(str(error)) from error
        self.records = records
        self.source_metadata = dict(source_metadata or DEFAULT_SOURCE_METADATA)
        self.tx_rc = (127, 127)
        self.tx_mask = build_tx_mask(self.output_size, self.tx_rc)
        self._cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        try:
            sample = self._decode(index)
        except CrossFrequencyDatasetError:
            raise
        except (DataFormatError, DatasetContractError, EmptyValidMaskError) as error:
            raise CrossFrequencyDatasetError(str(error)) from error
        self._cache[index] = sample
        return sample

    def _decode(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        height_shape, height_dtype = _source_contract(self.source_metadata, "height")
        beam_shape, beam_dtype = _source_contract(self.source_metadata, "beam_map")
        target_shape, target_dtype = _source_contract(self.source_metadata, "radiomap")
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
            raise CrossFrequencyDatasetError(
                f"height NPY {height_path} contains a negative value"
            )
        height = torch.from_numpy(height_array.astype("float32", copy=False)).unsqueeze(0)
        height = height / self.height_max
        if tuple(height.shape[-2:]) != self.output_size:
            height = resize_continuous(height.unsqueeze(0), self.output_size)[0]

        beam_array = _load_npy(
            beam_path,
            expected_shape=beam_shape,
            expected_dtype=beam_dtype,
            label="beam map",
        )
        beam = torch.from_numpy(beam_array.astype("float32", copy=False))
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
        metadata = {**record.to_dict(), "tx_rc": [self.tx_rc[0], self.tx_rc[1]]}
        return {
            "condition": condition.to(dtype=torch.float32),
            "target": target.to(dtype=torch.float32).contiguous(),
            "valid_mask": valid_mask.to(dtype=torch.bool).contiguous(),
            "metadata": metadata,
        }


__all__ = [
    "CrossFrequencyDatasetError",
    "CrossFrequencyRadiomapDataset",
    "load_cross_frequency_height_max",
    "multiconfig_collate",
]
