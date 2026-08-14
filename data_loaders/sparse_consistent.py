from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from data_loaders.sparse_task2 import SparseTask2RadiomapDataset, sparse_task2_collate
from training.sparse_consistent_config import (
    SPARSE_CONSISTENT_ARMS,
    SPARSE_CONSISTENT_OUTPUT_SIZE,
    SPARSE_CONSISTENT_SAMPLE_COUNT,
    SPARSE_CONSISTENT_SCENE_COUNTS,
)


class SparseConsistentDatasetError(RuntimeError):
    """The shared A/B/C/D sparse dataset contract is invalid."""


ArmName = Literal[
    "environment_only",
    "concat_fullfm",
    "multiscale_fullfm",
    "multiscale_consistent",
]


class SparseConsistentRadiomapDataset(Dataset):
    """Expose one immutable sparse sample in the registered arm format.

    The underlying Task 2 dataset owns the split, single-beam selection, target
    normalization, and deterministic 819-point mask.  This wrapper only
    separates the three environment channels from the sparse measurement
    channels so all arms see exactly the same sample and mask.
    """

    def __init__(
        self,
        *,
        dataset_root: Path,
        manifest_path: Path,
        split: Literal["train", "val", "test"],
        array_size: str,
        height_max: float,
        arm: ArmName,
    ) -> None:
        if arm not in SPARSE_CONSISTENT_ARMS:
            raise SparseConsistentDatasetError(f"unsupported arm: {arm}")
        self.arm = arm
        self.base = SparseTask2RadiomapDataset(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            split=split,
            array_size=array_size,
            height_max=height_max,
            expected_counts=SPARSE_CONSISTENT_SCENE_COUNTS,
            mask_seed=42,
            sample_count=SPARSE_CONSISTENT_SAMPLE_COUNT,
            output_size=SPARSE_CONSISTENT_OUTPUT_SIZE,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        full_condition = sample["condition"]
        environment_condition = full_condition[2:5].contiguous()
        condition = (
            full_condition.contiguous()
            if self.arm == "concat_fullfm"
            else environment_condition
        )
        metadata = dict(sample["metadata"])
        metadata["sparse_consistent_arm"] = self.arm
        return {
            "condition": condition,
            "environment_condition": environment_condition,
            "target": sample["target"],
            "valid_mask": sample["valid_mask"],
            "observation_mask": sample["observation_mask"],
            "sparse_map": sample["sparse_map"],
            "metadata": metadata,
        }


def sparse_consistent_collate(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not samples:
        raise SparseConsistentDatasetError("cannot collate an empty sample list")
    tensor_keys = (
        "condition",
        "environment_condition",
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
    "SparseConsistentDatasetError",
    "SparseConsistentRadiomapDataset",
    "sparse_consistent_collate",
]
