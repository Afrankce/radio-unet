from __future__ import annotations

import os

# Deterministic CUDA kernels must be requested before torch creates a context.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import json
from pathlib import Path
from typing import Sequence

from evaluation.multiconfig_evaluator import (
    run_cfg_selection,
    run_test_evaluation,
    summarize_benchmark,
)
from training.config import MultiConfigTrainConfig
from training.multiconfig_trainer import resolve_device


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument(
        "--array",
        dest="array_size",
        choices=("8x8", "16x16", "32x32"),
        required=True,
    )
    parser.add_argument("--model-size", choices=("lite", "large"), required=True)
    parser.add_argument("--train-scale", type=float, choices=(0.1, 1.0), default=1.0)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--device", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the locked Multi-config SRM RadioFlow benchmark"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _add_run_arguments(commands.add_parser("select-cfg"))
    _add_run_arguments(commands.add_parser("test"))
    summary = commands.add_parser("summarize")
    summary.add_argument("--run-root", type=Path, required=True)
    summary.add_argument("--results-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "summarize":
        result = summarize_benchmark(arguments.run_root, arguments.results_root)
    else:
        cfg = MultiConfigTrainConfig(
            array_size=arguments.array_size,
            model_size=arguments.model_size,
            dataset_root=arguments.dataset_root,
            manifest_dir=arguments.manifest_dir,
            run_root=arguments.run_root,
            train_scale=arguments.train_scale,
        )
        device = resolve_device(arguments.device)
        action = run_cfg_selection if arguments.command == "select-cfg" else run_test_evaluation
        result = action(cfg, device, arguments.results_root)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
