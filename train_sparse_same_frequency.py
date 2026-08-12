from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from training.config import InvocationControls
from training.multiconfig_trainer import resolve_device
from training.sparse_config import FORMAL_RUN_VARIANT, SparseSameFrequencyTrainConfig
from training.sparse_trainer import run_sparse_same_frequency_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the locked sparse 6.7GHz 5% single-beam RadioFlow run"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--height-stats-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--array-size", choices=("8x8", "16x16", "32x32"), required=True)
    parser.add_argument("--variant", choices=(FORMAL_RUN_VARIANT,), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-optimizer-steps", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.preflight_only and arguments.smoke_optimizer_steps is not None:
        raise ValueError("preflight-only cannot be combined with smoke-optimizer-steps")
    cfg = SparseSameFrequencyTrainConfig(
        dataset_root=arguments.dataset_root,
        manifest_path=arguments.manifest_path,
        height_stats_path=arguments.height_stats_path,
        run_root=arguments.run_root,
        array_size=arguments.array_size,
        variant=arguments.variant,
    )
    controls = InvocationControls(
        resume=arguments.resume,
        smoke_optimizer_steps=arguments.smoke_optimizer_steps,
    )
    result = run_sparse_same_frequency_training(
        cfg,
        controls,
        resolve_device(arguments.device),
        preflight_only=arguments.preflight_only,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
