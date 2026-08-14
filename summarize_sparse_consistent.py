from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from training.sparse_consistent_config import SPARSE_CONSISTENT_ARMS


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_row(run_root: Path, array_size: str, arm: str) -> dict[str, Any]:
    path = run_root / array_size / arm / "test_metrics.json"
    payload = _load_json(path)
    metrics = payload["metrics"]
    return {
        "array_size": array_size,
        "arm": arm,
        "checkpoint_completed_epochs": payload["checkpoint_completed_epochs"],
        "checkpoint_optimizer_step": payload["checkpoint_optimizer_step"],
        "missing_db_rmse": metrics["missing"]["db_rmse"],
        "missing_db_mae": metrics["missing"]["db_mae"],
        "missing_nmse": metrics["missing"]["nmse"],
        "missing_psnr": metrics["missing"]["psnr"],
        "missing_ssim": metrics["missing"]["ssim"],
        "overall_db_rmse": metrics["overall"]["db_rmse"],
        "overall_psnr": metrics["overall"]["psnr"],
        "overall_ssim": metrics["overall"]["ssim"],
        "observed_mean_abs": metrics["observed"]["mean_abs_error"],
        "observed_max_abs": metrics["observed"]["max_abs_error"],
        "condition_channels": _load_json(run_root / array_size / arm / "config.json").get("condition_channels"),
        "parameter_count": _load_json(run_root / array_size / arm / "config.json").get("parameter_count"),
    }


def _load_scene_rows(run_root: Path, array_size: str, arm: str) -> dict[str, dict[str, Any]]:
    path = run_root / array_size / arm / "per_scene_metrics.json"
    rows = _load_json(path)
    return {str(row["scene_id"]): row for row in rows}


def _pooled_missing_rmse(rows: Sequence[dict[str, Any]]) -> float:
    sq = sum(float(row["missing_sq_sum"]) for row in rows)
    count = sum(int(row["missing_pixel_count"]) for row in rows)
    return math.sqrt(90_000.0 * sq / count)


def _paired_bootstrap(
    run_root: Path,
    *,
    arrays: Sequence[str],
    left_arm: str,
    right_arm: str,
    seed: int = 42,
    replicates: int = 10_000,
) -> dict[str, float | int | list[float]]:
    left = {array: _load_scene_rows(run_root, array, left_arm) for array in arrays}
    right = {array: _load_scene_rows(run_root, array, right_arm) for array in arrays}
    scene_ids = sorted(set.intersection(*[set(left[array]) for array in arrays]))
    scene_ids = sorted(set(scene_ids).intersection(*[set(right[array]) for array in arrays]))
    if len(scene_ids) != 160:
        raise ValueError(f"paired bootstrap expected 160 common scenes, got {len(scene_ids)}")
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(scene_ids), size=len(scene_ids))
        left_rows: list[dict[str, Any]] = []
        right_rows: list[dict[str, Any]] = []
        for selected_index in selected:
            scene_id = scene_ids[int(selected_index)]
            for array in arrays:
                left_rows.append(left[array][scene_id])
                right_rows.append(right[array][scene_id])
        deltas[index] = _pooled_missing_rmse(right_rows) - _pooled_missing_rmse(left_rows)
    left_point = _pooled_missing_rmse(
        [left[array][scene] for array in arrays for scene in scene_ids]
    )
    right_point = _pooled_missing_rmse(
        [right[array][scene] for array in arrays for scene in scene_ids]
    )
    return {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "scene_count": len(scene_ids),
        "replicates": replicates,
        "seed": seed,
        "left_pooled_db_rmse": left_point,
        "right_pooled_db_rmse": right_point,
        "point_delta_db_rmse": right_point - left_point,
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "bootstrap_mean": float(deltas.mean()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize A/B/C/D sparse-consistent results")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arrays", nargs="+", default=["8x8", "16x16", "32x32"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    arrays = tuple(arguments.arrays)
    rows = [_metric_row(arguments.run_root, array, arm) for array in arrays for arm in SPARSE_CONSISTENT_ARMS]
    summary = {
        "schema_version": 1,
        "protocol": "sparse_consistent_abcd_v1",
        "rows": rows,
        "primary_pooled_missing_delta_D_minus_A": _paired_bootstrap(
            arguments.run_root,
            arrays=arrays,
            left_arm="environment_only",
            right_arm="multiscale_consistent",
        ),
        "secondary_pooled_missing_delta_B_minus_A": _paired_bootstrap(
            arguments.run_root,
            arrays=arrays,
            left_arm="environment_only",
            right_arm="concat_fullfm",
        ),
        "secondary_pooled_missing_delta_C_minus_B": _paired_bootstrap(
            arguments.run_root,
            arrays=arrays,
            left_arm="concat_fullfm",
            right_arm="multiscale_fullfm",
        ),
        "secondary_pooled_missing_delta_D_minus_C": _paired_bootstrap(
            arguments.run_root,
            arrays=arrays,
            left_arm="multiscale_fullfm",
            right_arm="multiscale_consistent",
        ),
    }
    output_path = arguments.output or arguments.run_root / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
