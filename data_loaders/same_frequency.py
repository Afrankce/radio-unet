from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from data_loaders.cross_frequency import (
    DEFAULT_SOURCE_METADATA,
    _source_contract,
    load_cross_frequency_height_max,
)
from data_loaders.multiconfig import (
    OUTPUT_SIZE,
    DataFormatError,
    DatasetContractError,
    EmptyValidMaskError,
    _load_npy,
    _safe_path,
    build_tx_mask,
    multiconfig_collate,
    normalize_db,
    prepare_target,
    resize_continuous,
    resize_valid_mask,
)
from experiments.cross_frequency import TEST_FREQUENCY_HZ
from experiments.multiconfig_manifest import ARRAY_SPECS, ManifestRecord, load_manifest_jsonl
from experiments.provenance import sha256_file


class SameFrequencyDatasetError(RuntimeError):
    """A fixed-frequency single-beam sample violates its data contract."""


def load_same_frequency_height_max(
    path: Path,
    *,
    split_path: Path | None = None,
) -> float:
    try:
        return load_cross_frequency_height_max(path, split_path=split_path)
    except Exception as error:
        if isinstance(error, SameFrequencyDatasetError):
            raise
        raise SameFrequencyDatasetError(
            f"cannot load same-frequency height statistics {path}: {error}"
        ) from error


def _validate_record(
    record: ManifestRecord,
    *,
    array_size: str,
    split: str,
    expected_frequency_hz: int,
) -> None:
    if record.split != split:
        raise SameFrequencyDatasetError("record split does not match selected dataset split")
    try:
        array_spec = ARRAY_SPECS[array_size]
    except KeyError as error:
        raise SameFrequencyDatasetError(f"unsupported array size: {array_size}") from error
    if record.array_name != array_size or (
        record.array_rows,
        record.array_cols,
    ) != (array_spec.rows, array_spec.cols):
        raise SameFrequencyDatasetError(
            f"record {record.sample_key} is not a {array_size} sample"
        )
    if record.frequency_hz != expected_frequency_hz:
        raise SameFrequencyDatasetError(
            f"record {record.sample_key} frequency mismatch: "
            f"expected {expected_frequency_hz}, got {record.frequency_hz}"
        )
    if not math.isclose(record.steering_deg, 0.0, abs_tol=1e-9):
        raise SameFrequencyDatasetError(
            f"record {record.sample_key} must use the zero-degree beam"
        )


class SameFrequencyRadiomapDataset(Dataset):
    """Decode a fixed 6.7 GHz, one-beam-per-scene manifest."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        manifest_path: Path,
        split: str,
        array_size: str,
        height_max: float,
        expected_frequency_hz: int = TEST_FREQUENCY_HZ,
        expected_beam_id: int | None = None,
        expected_counts: Mapping[str, int] | None = None,
        source_metadata: Mapping[str, Any] | None = None,
        output_size: tuple[int, int] = OUTPUT_SIZE,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise SameFrequencyDatasetError(f"invalid split: {split!r}")
        if array_size not in ARRAY_SPECS:
            raise SameFrequencyDatasetError(f"unsupported array size: {array_size}")
        if tuple(output_size) != OUTPUT_SIZE:
            raise SameFrequencyDatasetError(
                f"fixed output size must be {OUTPUT_SIZE}, got {tuple(output_size)}"
            )
        if not math.isfinite(float(height_max)) or float(height_max) <= 0.0:
            raise SameFrequencyDatasetError("height_max must be finite and positive")
        if expected_frequency_hz != TEST_FREQUENCY_HZ:
            raise SameFrequencyDatasetError(
                f"same-frequency experiment is locked to {TEST_FREQUENCY_HZ} Hz"
            )
        self.dataset_root = Path(dataset_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.split = split
        self.array_size = array_size
        self.height_max = float(height_max)
        self.output_size = OUTPUT_SIZE
        counts = dict(expected_counts or {"train": 560, "val": 80, "test": 160})
        if set(counts) != {"train", "val", "test"} or any(
            isinstance(value, bool) or int(value) <= 0 for value in counts.values()
        ):
            raise SameFrequencyDatasetError(
                "expected_counts must define positive train/val/test counts"
            )
        try:
            all_records = load_manifest_jsonl(self.manifest_path)
        except Exception as error:
            raise SameFrequencyDatasetError(
                f"cannot load same-frequency manifest {self.manifest_path}: {error}"
            ) from error
        records = tuple(record for record in all_records if record.split == split)
        if len(records) != int(counts[split]):
            raise SameFrequencyDatasetError(
                f"{split} sample count mismatch: expected {counts[split]}, got {len(records)}"
            )
        if not records:
            raise SameFrequencyDatasetError(
                f"manifest {self.manifest_path} has no {split} samples"
            )
        sample_keys = [record.sample_key for record in records]
        logical_keys = [(record.scene_id, record.beam_id) for record in records]
        if len(sample_keys) != len(set(sample_keys)) or len(logical_keys) != len(set(logical_keys)):
            raise SameFrequencyDatasetError("selected split contains duplicate samples")
        beam_ids = {record.beam_id for record in records}
        config_ids = {record.config_id for record in records}
        if len(beam_ids) != 1 or len(config_ids) != 1:
            raise SameFrequencyDatasetError(
                "same-frequency split must contain exactly one beam and one configuration"
            )
        beam_id = next(iter(beam_ids))
        if expected_beam_id is not None and beam_id != int(expected_beam_id):
            raise SameFrequencyDatasetError(
                f"manifest beam ID mismatch: expected {expected_beam_id}, got {beam_id}"
            )
        for record in records:
            try:
                _validate_record(
                    record,
                    array_size=array_size,
                    split=split,
                    expected_frequency_hz=expected_frequency_hz,
                )
                for relative_path in (
                    record.height_path,
                    record.beam_map_path,
                    record.radiomap_path,
                ):
                    _safe_path(self.dataset_root, relative_path)
            except (DatasetContractError, SameFrequencyDatasetError) as error:
                if isinstance(error, SameFrequencyDatasetError):
                    raise
                raise SameFrequencyDatasetError(str(error)) from error
        self.records = records
        self.beam_id = beam_id
        self.config_id = next(iter(config_ids))
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
        except SameFrequencyDatasetError:
            raise
        except (DataFormatError, DatasetContractError, EmptyValidMaskError) as error:
            raise SameFrequencyDatasetError(str(error)) from error
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
            raise SameFrequencyDatasetError(
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
        metadata = {
            **record.to_dict(),
            "tx_rc": [self.tx_rc[0], self.tx_rc[1]],
            "array_size": self.array_size,
            "beam_id": self.beam_id,
        }
        return {
            "condition": condition.to(dtype=torch.float32),
            "target": target.to(dtype=torch.float32).contiguous(),
            "valid_mask": valid_mask.to(dtype=torch.bool).contiguous(),
            "metadata": metadata,
        }


__all__ = [
    "SameFrequencyDatasetError",
    "SameFrequencyRadiomapDataset",
    "load_same_frequency_height_max",
    "multiconfig_collate",
]
