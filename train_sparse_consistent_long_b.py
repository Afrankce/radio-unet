from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from training.sparse_consistent_config import SparseConsistentTrainConfig
from training.sparse_consistent_trainer import resolve_device, run_sparse_consistent_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exploratory long-budget 8x8 B sparse-reconstruction training"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--height-stats-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--resume", default="none")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    cfg = SparseConsistentTrainConfig(
        dataset_root=arguments.dataset_root,
        manifest_path=arguments.manifest_path,
        height_stats_path=arguments.height_stats_path,
        run_root=arguments.run_root,
        array_size="8x8",
        arm="concat_fullfm",
        max_epochs=600,
        early_stopping_patience=600,
        min_optimizer_steps=6000,
        exploratory=True,
    )
    result = run_sparse_consistent_training(
        cfg,
        resolve_device(arguments.device),
        preflight_only=arguments.preflight_only,
        resume=arguments.resume,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
