from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from data_loaders.cross_frequency import load_cross_frequency_height_max
from data_loaders.random_task2 import RandomTask2RadiomapDataset
from training.random_task2_config import (
    RANDOM_TASK2_COMMON_ANGLES,
    RANDOM_TASK2_PROTOCOL,
    RANDOM_TASK2_RECORD_COUNTS,
    RandomTask2TrainConfig,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="kNN-IDW interpolation baseline for Task 2")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--height-stats-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--array-size", choices=("8x8", "16x16", "32x32"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--power", type=float, default=2.0)
    return parser


def idw_interpolate(
    known_points: np.ndarray,
    known_values: np.ndarray,
    query_points: np.ndarray,
    *,
    neighbors: int,
    power: float,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(known_points)
    distances, indices = tree.query(query_points, k=neighbors)
    if neighbors == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / np.maximum(distances, 1e-6) ** power
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.sum(weights * known_values[indices], axis=1)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    cfg = RandomTask2TrainConfig(
        dataset_root=arguments.dataset_root,
        manifest_path=arguments.manifest_path,
        height_stats_path=arguments.height_stats_path,
        run_root=arguments.run_root,
        array_size=arguments.array_size,
        variant="feature4",
    )
    height_max = load_cross_frequency_height_max(cfg.height_stats_path)
    dataset = RandomTask2RadiomapDataset(
        dataset_root=cfg.dataset_root,
        manifest_path=cfg.manifest_path,
        split=arguments.split,
        array_size=cfg.array_size,
        height_max=height_max,
        variant="feature4",
    )
    sum_sq = 0.0
    sum_abs = 0.0
    count = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        target = sample["target"][0].numpy()
        valid = sample["valid_mask"][0].numpy()
        observed = sample["observation_mask"][0].numpy()
        sparse_map = sample["sparse_map"][0].numpy()
        known_y, known_x = np.nonzero(observed)
        known_points = np.stack((known_y, known_x), axis=1)
        known_values = sparse_map[known_y, known_x]
        missing = valid & ~observed
        query_y, query_x = np.nonzero(missing)
        query_points = np.stack((query_y, query_x), axis=1)
        prediction = idw_interpolate(
            known_points,
            known_values,
            query_points,
            neighbors=arguments.neighbors,
            power=arguments.power,
        )
        truth = target[query_y, query_x]
        error = np.clip(prediction, 0.0, 1.0) - truth
        sum_sq += float(np.square(error).sum())
        sum_abs += float(np.abs(error).sum())
        count += int(len(query_y))
    output = {
        "schema_version": 1,
        "protocol": RANDOM_TASK2_PROTOCOL,
        "array_size": cfg.array_size,
        "split": arguments.split,
        "neighbors": arguments.neighbors,
        "power": arguments.power,
        "pixel_count": count,
        "db_mae": 300.0 * sum_abs / max(count, 1),
        "db_rmse": math.sqrt(90_000.0 * sum_sq / max(count, 1)),
        "mse": sum_sq / max(count, 1),
    }
    path = cfg.run_root / "baselines" / f"idw_{arguments.split}.json"
    _write_json(path, output)
    print(json.dumps(output, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
