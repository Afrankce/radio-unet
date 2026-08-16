from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from data_loaders.sparse_task2 import SparseTask2RadiomapDataset
from training.random_task2_config import (
    RANDOM_TASK2_COMMON_ANGLES,
    RANDOM_TASK2_PROTOCOL,
    RANDOM_TASK2_RECORD_COUNTS,
    RANDOM_TASK2_SAMPLE_COUNT,
)


class RandomTask2DatasetError(RuntimeError):
    """The random-instance sparse Task 2 dataset contract is violated."""


class RandomTask2RadiomapDataset(Dataset):
    """Random-instance sparse Task 2 dataset over the eight common beams."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        manifest_path: Path,
        split: Literal["train", "val", "test"],
        array_size: str,
        height_max: float,
        variant: Literal["feature4", "feature5_mask"] = "feature4",
    ) -> None:
        self.variant = variant
        self.base = SparseTask2RadiomapDataset(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            split=split,
            array_size=array_size,
            height_max=height_max,
            expected_counts=RANDOM_TASK2_RECORD_COUNTS,
            mask_seed=42,
            sample_count=RANDOM_TASK2_SAMPLE_COUNT,
            beam_angles=RANDOM_TASK2_COMMON_ANGLES,
            condition_variant=variant,
            protocol=RANDOM_TASK2_PROTOCOL,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        metadata = dict(sample["metadata"])
        metadata["random_task2_variant"] = self.variant
        sample["metadata"] = metadata
        return sample


def random_task2_collate(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not samples:
        raise RandomTask2DatasetError("cannot collate an empty sample list")
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
    "RandomTask2DatasetError",
    "RandomTask2RadiomapDataset",
    "random_task2_collate",
]
