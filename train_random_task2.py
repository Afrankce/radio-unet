from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from training.multiconfig_trainer import resolve_device
from training.random_task2_config import RandomTask2TrainConfig
from training.random_task2_trainer import run_random_task2_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the random-instance sparse Task 2 reconstruction run"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--height-stats-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--array-size", choices=("8x8", "16x16", "32x32"), required=True)
    parser.add_argument("--variant", choices=("feature4", "feature5_mask"), default="feature4")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument("--observed-loss-weight", type=float, default=100.0)
    parser.add_argument("--device", required=True)
    parser.add_argument("--resume", default="none")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--smoke-optimizer-steps", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    cfg = RandomTask2TrainConfig(
        dataset_root=arguments.dataset_root,
        manifest_path=arguments.manifest_path,
        height_stats_path=arguments.height_stats_path,
        run_root=arguments.run_root,
        array_size=arguments.array_size,
        variant=arguments.variant,
        mode="regression",
        max_epochs=arguments.max_epochs,
        early_stopping_patience=arguments.early_stopping_patience,
        observed_loss_weight=arguments.observed_loss_weight,
    )
    result = run_random_task2_training(
        cfg,
        resolve_device(arguments.device),
        preflight_only=arguments.preflight_only,
        resume=arguments.resume,
        stop_after_epoch=arguments.stop_after_epoch,
        smoke_optimizer_steps=arguments.smoke_optimizer_steps,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
