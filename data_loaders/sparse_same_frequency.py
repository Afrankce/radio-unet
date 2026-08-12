from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
from torch.utils.data import Dataset, get_worker_info

from data_loaders.cross_frequency import DEFAULT_SOURCE_METADATA, _source_contract
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
from data_loaders.same_frequency import SameFrequencyDatasetError, _validate_record
from experiments.cross_frequency import TEST_FREQUENCY_HZ
from experiments.multiconfig_manifest import ARRAY_SPECS, ManifestRecord, load_manifest_jsonl
from training.sparse_masks import (
    build_masked_condition_map,
    make_condition_noise,
    make_observation_mask,
)


logger = logging.getLogger(__name__)


class SparseSameFrequencyRadiomapDataset(Dataset):
    """Fixed 6.7 GHz single-beam sparse dataset for the formal beam-masked run."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        manifest_path: Path,
        split: Literal["train", "val", "test"],
        array_size: str,
        variant: Literal["no_beam_masked", "beam_masked"],
        height_max: float,
        observation_ratio: float = 0.05,
        mask_seed: int = 42,
        condition_noise_seed: int = 4242,
        expected_counts: Mapping[str, int] | None = None,
        source_metadata: Mapping[str, Any] | None = None,
        output_size: tuple[int, int] = OUTPUT_SIZE,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise SameFrequencyDatasetError(f"invalid split: {split!r}")
        if array_size not in ARRAY_SPECS:
            raise SameFrequencyDatasetError(f"unsupported array size: {array_size}")
        if variant != "beam_masked":
            raise SameFrequencyDatasetError(
                "sparse same-frequency dataset only supports the formal 'beam_masked' variant"
            )
        if tuple(output_size) != OUTPUT_SIZE:
            raise SameFrequencyDatasetError(
                f"fixed output size must be {OUTPUT_SIZE}, got {tuple(output_size)}"
            )
        if not math.isfinite(float(height_max)) or float(height_max) <= 0.0:
            raise SameFrequencyDatasetError("height_max must be finite and positive")
        if (
            not isinstance(observation_ratio, float)
            or not math.isfinite(observation_ratio)
            or not 0.0 < observation_ratio < 1.0
        ):
            raise SameFrequencyDatasetError("observation_ratio must satisfy 0 < ratio < 1")
        if type(mask_seed) is not int or type(condition_noise_seed) is not int:
            raise SameFrequencyDatasetError("mask_seed and condition_noise_seed must be integers")

        self.dataset_root = Path(dataset_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.split = split
        self.array_size = array_size
        self.variant = variant
        self.height_max = float(height_max)
        self.observation_ratio = observation_ratio
        self.mask_seed = mask_seed
        self.condition_noise_seed = condition_noise_seed
        self.output_size = OUTPUT_SIZE
        self.source_metadata = dict(source_metadata or DEFAULT_SOURCE_METADATA)
        self.tx_rc = (127, 127)
        self.tx_mask = build_tx_mask(self.output_size, self.tx_rc)
        self._epoch = 0
        self._raw_cache: dict[int, dict[str, Any]] = {}

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
        for record in records:
            try:
                _validate_record(
                    record,
                    array_size=array_size,
                    split=split,
                    expected_frequency_hz=TEST_FREQUENCY_HZ,
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
        logger.info(
            "Initialized sparse dataset split=%s array=%s variant=%s samples=%s mask_seed=%s noise_seed=%s worker=%s",
            self.split,
            self.array_size,
            self.variant,
            len(self.records),
            self.mask_seed,
            self.condition_noise_seed,
            None if get_worker_info() is None else get_worker_info().id,
        )

    def set_epoch(self, epoch: int) -> None:
        """Change only the deterministic train condition-noise key."""

        if type(epoch) is not int or epoch < 0:
            raise SameFrequencyDatasetError("epoch must be a non-negative integer")
        self._epoch = epoch
        logger.info("Sparse dataset set_epoch(split=%s, epoch=%s)", self.split, epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self._raw_cache.get(index)
        if raw is None:
            try:
                raw = self._decode(index)
            except SameFrequencyDatasetError:
                raise
            except (DataFormatError, DatasetContractError, EmptyValidMaskError) as error:
                raise SameFrequencyDatasetError(str(error)) from error
            self._raw_cache[index] = raw

        height = raw["height"].clone()
        beam = raw["beam"].clone()
        target = raw["target"].clone()
        valid_mask = raw["valid_mask"].clone()
        metadata = dict(raw["metadata"])

        observation_mask = make_observation_mask(
            valid_mask,
            scene_id=metadata["scene_id"],
            steering_deg=float(metadata["steering_deg"]),
            ratio=self.observation_ratio,
            base_seed=self.mask_seed,
        )
        condition_noise = make_condition_noise(
            tuple(target.shape),
            scene_id=metadata["scene_id"],
            steering_deg=float(metadata["steering_deg"]),
            split=self.split,
            epoch=self._epoch if self.split == "train" else None,
            base_seed=self.condition_noise_seed,
            dtype=target.dtype,
        ).to(device=target.device)
        masked_map, observed_map, missing_mask = build_masked_condition_map(
            target,
            valid_mask,
            observation_mask,
            condition_noise,
        )
        visible_mask = observation_mask.to(dtype=torch.float32)
        condition = torch.cat(
            (self.tx_mask, height, beam, masked_map, visible_mask),
            dim=0,
        ).contiguous()

        valid_pixels = int(valid_mask.sum().item())
        observed_pixels = int(observation_mask.sum().item())
        sample_metadata = {
            **metadata,
            "split": self.split,
            "observation_ratio": self.observation_ratio,
            "observed_pixels": observed_pixels,
            "valid_pixels": valid_pixels,
        }
        return {
            "condition": condition.to(dtype=torch.float32),
            "target": target.to(dtype=torch.float32).contiguous(),
            "valid_mask": valid_mask.to(dtype=torch.bool).contiguous(),
            "observation_mask": observation_mask.to(dtype=torch.bool).contiguous(),
            "missing_mask": missing_mask.to(dtype=torch.bool).contiguous(),
            "observed_map": observed_map.to(dtype=torch.float32).contiguous(),
            "masked_map": masked_map.to(dtype=torch.float32).contiguous(),
            "valid": valid_mask.to(dtype=torch.bool).contiguous(),
            "observation": observation_mask.to(dtype=torch.bool).contiguous(),
            "missing": missing_mask.to(dtype=torch.bool).contiguous(),
            "observed": observed_map.to(dtype=torch.float32).contiguous(),
            "masked": masked_map.to(dtype=torch.float32).contiguous(),
            "metadata": sample_metadata,
        }

    def _decode(self, index: int) -> dict[str, Any]:
        record: ManifestRecord = self.records[index]
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
            target_source.unsqueeze(0).unsqueeze(0),
            self.output_size,
        )[0]
        valid_mask = resize_valid_mask(
            valid_source.unsqueeze(0).unsqueeze(0),
            self.output_size,
        )[0]
        target = target.masked_fill(~valid_mask, 0.0)
        return {
            "height": height.to(dtype=torch.float32).contiguous(),
            "beam": beam.to(dtype=torch.float32).contiguous(),
            "target": target.to(dtype=torch.float32).contiguous(),
            "valid_mask": valid_mask.to(dtype=torch.bool).contiguous(),
            "metadata": {
                **record.to_dict(),
                "tx_rc": [self.tx_rc[0], self.tx_rc[1]],
                "array_size": self.array_size,
                "beam_id": record.beam_id,
            },
        }


__all__ = [
    "SameFrequencyDatasetError",
    "SparseSameFrequencyRadiomapDataset",
    "multiconfig_collate",
]
