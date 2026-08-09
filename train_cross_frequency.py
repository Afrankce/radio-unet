from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

# Set this before importing torch-backed training modules or creating CUDA state.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from training.config import InvocationControls
from training.cross_frequency_config import CrossFrequencyTrainConfig
from training.cross_frequency_trainer import (
    CrossFrequencyTrainerContractError,
    resolve_device,
    run_cross_frequency_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the locked 8x8 4.9GHz-to-6.7GHz RadioFlow experiment"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--height-stats-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-size", choices=("lite", "large"), default="lite")
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--resume",
        required=True,
        help="'none', 'auto', or an explicit local full-state checkpoint",
    )
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--stop-after-epoch", type=int)
    controls.add_argument("--smoke-optimizer-steps", type=int)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate source data and identities without constructing a model",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.preflight_only and (
        arguments.stop_after_epoch is not None
        or arguments.smoke_optimizer_steps is not None
    ):
        raise CrossFrequencyTrainerContractError(
            "preflight-only cannot be combined with training controls"
        )
    cfg = CrossFrequencyTrainConfig(
        dataset_root=arguments.dataset_root,
        manifest_path=arguments.manifest_path,
        height_stats_path=arguments.height_stats_path,
        run_root=arguments.run_root,
        model_size=arguments.model_size,
    )
    controls = InvocationControls(
        resume=arguments.resume,
        stop_after_epoch=arguments.stop_after_epoch,
        smoke_optimizer_steps=arguments.smoke_optimizer_steps,
    )
    result = run_cross_frequency_training(
        cfg,
        controls,
        resolve_device(arguments.device),
        preflight_only=arguments.preflight_only,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
