from __future__ import annotations

import os

# This must be fixed before any CUDA context can be created.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import json
from pathlib import Path
from typing import Sequence

from training.config import InvocationControls, MultiConfigTrainConfig
from training.multiconfig_trainer import resolve_device, run_benchmark_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the locked Multi-config SRM benchmark with RadioFlow"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument(
        "--array",
        dest="array_size",
        choices=("8x8", "16x16", "32x32"),
        required=True,
    )
    parser.add_argument(
        "--model-size",
        choices=("lite", "large"),
        required=True,
    )
    parser.add_argument("--train-scale", type=float, choices=(0.1, 1.0), default=1.0)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--resume",
        required=True,
        help="'none', 'auto', or an explicit local full-state checkpoint",
    )
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--stop-after-epoch", type=int)
    controls.add_argument("--smoke-optimizer-steps", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    cfg = MultiConfigTrainConfig(
        array_size=arguments.array_size,
        model_size=arguments.model_size,
        dataset_root=arguments.dataset_root,
        manifest_dir=arguments.manifest_dir,
        run_root=arguments.run_root,
        train_scale=arguments.train_scale,
    )
    controls = InvocationControls(
        resume=arguments.resume,
        stop_after_epoch=arguments.stop_after_epoch,
        smoke_optimizer_steps=arguments.smoke_optimizer_steps,
    )
    result = run_benchmark_training(
        cfg,
        controls,
        resolve_device(arguments.device),
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

