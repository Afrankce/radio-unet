from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from training.config import InvocationControls
from training.sparse_task2_config import SparseTask2TrainConfig
from training.sparse_task2_trainer import (
    resolve_device,
    run_sparse_task2_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the mandatory 6.7 GHz, 0-degree, 819-sample Task 2 Lite run"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--height-stats-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--array-size", choices=("8x8", "16x16", "32x32"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--resume", default="none")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--audit-all-samples", action="store_true")
    parser.add_argument("--smoke-optimizer-steps", type=int)
    parser.add_argument("--stop-after-epoch", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.preflight_only and (
        arguments.smoke_optimizer_steps is not None
        or arguments.stop_after_epoch is not None
    ):
        raise ValueError("preflight-only cannot be combined with training controls")
    cfg = SparseTask2TrainConfig(
        dataset_root=arguments.dataset_root,
        manifest_path=arguments.manifest_path,
        height_stats_path=arguments.height_stats_path,
        run_root=arguments.run_root,
        array_size=arguments.array_size,
    )
    controls = InvocationControls(
        resume=arguments.resume,
        stop_after_epoch=arguments.stop_after_epoch,
        smoke_optimizer_steps=arguments.smoke_optimizer_steps,
    )
    result = run_sparse_task2_training(
        cfg,
        controls,
        resolve_device(arguments.device),
        preflight_only=arguments.preflight_only,
        audit_all_samples=arguments.audit_all_samples,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
