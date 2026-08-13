from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from data_loaders.cross_frequency import DEFAULT_SOURCE_METADATA, _source_contract
from data_loaders.multiconfig import (
    OUTPUT_SIZE,
    DataFormatError,
    DatasetContractError,
    EmptyValidMaskError,
    _load_npy,
    _safe_path,
    build_tx_mask,
    normalize_db,
    prepare_target,
    resize_continuous,
    resize_valid_mask,
)
from experiments.multiconfig_manifest import ARRAY_SPECS, ManifestRecord, load_manifest_jsonl
from training.sparse_task2_config import (
    SINGLEBEAM_TASK2_CONDITION_CHANNELS,
    SINGLEBEAM_TASK2_FREQUENCY_HZ,
    SINGLEBEAM_TASK2_OUTPUT_SIZE,
    SINGLEBEAM_TASK2_PROTOCOL,
    SINGLEBEAM_TASK2_SAMPLE_COUNT,
    SINGLEBEAM_TASK2_SCENE_COUNTS,
    SINGLEBEAM_TASK2_STEERING_DEG,
)


class SparseTask2DatasetError(RuntimeError):
    """A sample violates the mandatory single-beam sparse Task 2 contract."""


def _hash_seed(*parts: object) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def choose_valid_observation_mask(
    valid_mask: Tensor,
    *,
    scene_id: str,
    seed: int = 42,
    count: int = SINGLEBEAM_TASK2_SAMPLE_COUNT,
) -> Tensor:
    """Select exactly ``count`` unique valid final-grid pixels deterministically."""

    if valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must have boolean dtype")
    if valid_mask.ndim == 2:
        mask_shape = (1, *valid_mask.shape)
        flat_mask = valid_mask.reshape(-1)
    elif valid_mask.ndim == 3 and valid_mask.shape[0] == 1:
        mask_shape = tuple(valid_mask.shape)
        flat_mask = valid_mask.reshape(-1)
    else:
        raise ValueError("valid_mask must have shape [H,W] or [1,H,W]")
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("scene_id must be a non-empty string")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    valid_indices = torch.nonzero(flat_mask.detach().to(device="cpu"), as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError("valid_mask contains no valid pixels")
    if count > int(valid_indices.numel()):
        raise ValueError(
            f"observation count {count} exceeds valid pixel count {int(valid_indices.numel())}"
        )
    generator = torch.Generator(device="cpu").manual_seed(
        _hash_seed(SINGLEBEAM_TASK2_PROTOCOL, seed, scene_id, count)
    )
    order = torch.randperm(valid_indices.numel(), generator=generator)
    selected = valid_indices[order[:count]]
    output = torch.zeros(flat_mask.numel(), dtype=torch.bool)
    output[selected] = True
    output = output.reshape(mask_shape).to(device=valid_mask.device)
    output = output & valid_mask.detach().to(dtype=torch.bool).reshape(mask_shape)
    return output


def _zero_degree_beam_id(array_size: str) -> int:
    matches = [
        beam.beam_id
        for beam in ARRAY_SPECS[array_size].beams
        if math.isclose(beam.steering_deg, SINGLEBEAM_TASK2_STEERING_DEG, abs_tol=1e-9)
    ]
    if len(matches) != 1:
        raise SparseTask2DatasetError(
            f"array {array_size} must have exactly one selected zero-degree beam"
        )
    return matches[0]


class SparseTask2RadiomapDataset(Dataset):
    """Decode the mandatory 6.7 GHz, 0-degree, sparse measurement protocol."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        manifest_path: Path,
        split: Literal["train", "val", "test"],
        array_size: str,
        height_max: float,
        expected_counts: Mapping[str, int] | None = None,
        source_metadata: Mapping[str, Any] | None = None,
        output_size: tuple[int, int] = OUTPUT_SIZE,
        mask_seed: int = 42,
        sample_count: int = SINGLEBEAM_TASK2_SAMPLE_COUNT,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise SparseTask2DatasetError(f"invalid split: {split!r}")
        if array_size not in ARRAY_SPECS:
            raise SparseTask2DatasetError(f"unsupported array size: {array_size}")
        if tuple(output_size) != SINGLEBEAM_TASK2_OUTPUT_SIZE:
            raise SparseTask2DatasetError(
                f"fixed output size must be {SINGLEBEAM_TASK2_OUTPUT_SIZE}, got {tuple(output_size)}"
            )
        if not math.isfinite(float(height_max)) or float(height_max) <= 0.0:
            raise SparseTask2DatasetError("height_max must be finite and positive")
        if type(mask_seed) is not int or mask_seed != 42:
            raise SparseTask2DatasetError("mask_seed is locked to 42")
        if type(sample_count) is not int or sample_count != SINGLEBEAM_TASK2_SAMPLE_COUNT:
            raise SparseTask2DatasetError(
                f"sample_count is locked to {SINGLEBEAM_TASK2_SAMPLE_COUNT}"
            )

        self.dataset_root = Path(dataset_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.split = split
        self.array_size = array_size
        self.height_max = float(height_max)
        self.mask_seed = mask_seed
        self.sample_count = sample_count
        self.output_size = SINGLEBEAM_TASK2_OUTPUT_SIZE
        self.source_metadata = dict(source_metadata or DEFAULT_SOURCE_METADATA)
        self.tx_rc = (127, 127)
        self.tx_mask = build_tx_mask(self.output_size, self.tx_rc)
        self._cache: dict[int, dict[str, Any]] = {}

        counts = dict(expected_counts or SINGLEBEAM_TASK2_SCENE_COUNTS)
        if set(counts) != {"train", "val", "test"} or any(
            type(value) is not int or value <= 0 for value in counts.values()
        ):
            raise SparseTask2DatasetError(
                "expected_counts must contain positive train/val/test integers"
            )
        try:
            all_records = load_manifest_jsonl(self.manifest_path)
        except Exception as error:
            raise SparseTask2DatasetError(
                f"cannot load sparse Task 2 manifest {self.manifest_path}: {error}"
            ) from error
        records = tuple(record for record in all_records if record.split == split)
        if len(records) != counts[split]:
            raise SparseTask2DatasetError(
                f"{split} sample count mismatch: expected {counts[split]}, got {len(records)}"
            )
        if not records:
            raise SparseTask2DatasetError(f"manifest has no {split} samples")

        expected_beam_id = _zero_degree_beam_id(array_size)
        sample_keys = [record.sample_key for record in records]
        logical_keys = [(record.scene_id, record.beam_id) for record in records]
        if len(sample_keys) != len(set(sample_keys)) or len(logical_keys) != len(set(logical_keys)):
            raise SparseTask2DatasetError("selected split contains duplicate samples")
        beam_ids = {record.beam_id for record in records}
        config_ids = {record.config_id for record in records}
        if beam_ids != {expected_beam_id} or len(config_ids) != 1:
            raise SparseTask2DatasetError(
                f"sparse Task 2 split must contain one zero-degree beam {expected_beam_id}"
            )
        for record in records:
            self._validate_record(record, expected_beam_id)
        self.records = records

    def _validate_record(self, record: ManifestRecord, expected_beam_id: int) -> None:
        spec = ARRAY_SPECS[self.array_size]
        if record.array_name != self.array_size or (
            record.array_rows,
            record.array_cols,
        ) != (spec.rows, spec.cols):
            raise SparseTask2DatasetError(
                f"record {record.sample_key} is not a {self.array_size} sample"
            )
        if record.frequency_hz != SINGLEBEAM_TASK2_FREQUENCY_HZ:
            raise SparseTask2DatasetError(
                f"record {record.sample_key} frequency is not 6.7 GHz"
            )
        if not math.isclose(
            record.steering_deg, SINGLEBEAM_TASK2_STEERING_DEG, abs_tol=1e-9
        ):
            raise SparseTask2DatasetError(
                f"record {record.sample_key} steering angle is not 0 degrees"
            )
        if record.beam_id != expected_beam_id:
            raise SparseTask2DatasetError(
                f"record {record.sample_key} beam ID mismatch: expected {expected_beam_id}"
            )
        for relative_path in (
            record.height_path,
            record.beam_map_path,
            record.radiomap_path,
        ):
            try:
                _safe_path(self.dataset_root, relative_path)
            except DatasetContractError as error:
                raise SparseTask2DatasetError(str(error)) from error

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        try:
            sample = self._decode(index)
        except SparseTask2DatasetError:
            raise
        except (DataFormatError, DatasetContractError, EmptyValidMaskError) as error:
            raise SparseTask2DatasetError(str(error)) from error
        self._cache[index] = sample
        return sample

    def _decode(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        height_shape, height_dtype = _source_contract(self.source_metadata, "height")
        beam_shape, beam_dtype = _source_contract(self.source_metadata, "beam_map")
        target_shape, target_dtype = _source_contract(self.source_metadata, "radiomap")

        height_array = _load_npy(
            _safe_path(self.dataset_root, record.height_path),
            expected_shape=height_shape,
            expected_dtype=height_dtype,
            label="height",
        )
        if bool((height_array < 0).any()):
            raise SparseTask2DatasetError("height contains a negative value")
        height = torch.from_numpy(height_array.astype("float32", copy=False)).unsqueeze(0)
        height = height / self.height_max
        if tuple(height.shape[-2:]) != self.output_size:
            height = resize_continuous(height.unsqueeze(0), self.output_size)[0]

        beam_array = _load_npy(
            _safe_path(self.dataset_root, record.beam_map_path),
            expected_shape=beam_shape,
            expected_dtype=beam_dtype,
            label="beam map",
        )
        beam = torch.from_numpy(beam_array.astype("float32", copy=False))
        beam = normalize_db(beam).unsqueeze(0).unsqueeze(0)
        beam = resize_continuous(beam, self.output_size)[0]

        target_array = _load_npy(
            _safe_path(self.dataset_root, record.radiomap_path),
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
        observation_mask = choose_valid_observation_mask(
            valid_mask,
            scene_id=record.scene_id,
            seed=self.mask_seed,
            count=self.sample_count,
        )
        sparse_map = (target * observation_mask.to(dtype=target.dtype)).masked_fill(
            ~valid_mask, 0.0
        )
        observation_channel = observation_mask.to(dtype=torch.float32)
        condition = torch.cat(
            (sparse_map, observation_channel, self.tx_mask, height, beam), dim=0
        ).contiguous()
        if condition.shape[0] != SINGLEBEAM_TASK2_CONDITION_CHANNELS:
            raise SparseTask2DatasetError(
                f"condition must have {SINGLEBEAM_TASK2_CONDITION_CHANNELS} channels"
            )
        valid_pixels = int(valid_mask.sum().item())
        observed_pixels = int(observation_mask.sum().item())
        metadata = {
            **record.to_dict(),
            "protocol": SINGLEBEAM_TASK2_PROTOCOL,
            "split": self.split,
            "array_size": self.array_size,
            "tx_rc": [self.tx_rc[0], self.tx_rc[1]],
            "mask_seed": self.mask_seed,
            "mask_key": f"{SINGLEBEAM_TASK2_PROTOCOL}|seed{self.mask_seed}|scene{record.scene_id}",
            "observed_pixels": observed_pixels,
            "valid_pixels": valid_pixels,
        }
        if observed_pixels != self.sample_count:
            raise SparseTask2DatasetError(
                f"observation count mismatch: expected {self.sample_count}, got {observed_pixels}"
            )
        return {
            "condition": condition.to(dtype=torch.float32),
            "target": target.to(dtype=torch.float32).contiguous(),
            "valid_mask": valid_mask.to(dtype=torch.bool).contiguous(),
            "observation_mask": observation_mask.to(dtype=torch.bool).contiguous(),
            "sparse_map": sparse_map.to(dtype=torch.float32).contiguous(),
            "metadata": metadata,
        }


def sparse_task2_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise SparseTask2DatasetError("cannot collate an empty sample list")
    tensor_keys = (
        "condition",
        "target",
        "valid_mask",
        "observation_mask",
        "sparse_map",
    )
    return {
        **{key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys},
        "metadata": [sample["metadata"] for sample in samples],
    }


__all__ = [
    "SINGLEBEAM_TASK2_CONDITION_CHANNELS",
    "SINGLEBEAM_TASK2_FREQUENCY_HZ",
    "SINGLEBEAM_TASK2_PROTOCOL",
    "SINGLEBEAM_TASK2_SAMPLE_COUNT",
    "SparseTask2DatasetError",
    "SparseTask2RadiomapDataset",
    "choose_valid_observation_mask",
    "sparse_task2_collate",
]
