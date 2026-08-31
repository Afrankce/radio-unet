from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from evaluation.same_frequency_evaluator import run_cfg_selection, run_test_evaluation
from training.config import InvocationControls
from training.multiconfig_trainer import resolve_device
from training.same_frequency_config import SameFrequencyTrainConfig
from training.same_frequency_multiscale_uno_config import MultiscaleUNOTrainConfig
from training.same_frequency_multiscale_uno_trainer import (
    run_same_frequency_multiscale_uno_training,
)
from training.same_frequency_trainer import (
    SameFrequencyTrainerContractError,
    infer_manifest_selection,
)


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--height-stats-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--array-size",
        choices=("8x8", "16x16", "32x32"),
        required=True,
    )
    parser.add_argument("--device", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the attention-conditioned multiscale UNO lifecycle on the "
            "6.7GHz zero-degree single-beam RadioFlow protocol"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train or resume the frozen run")
    _add_data_arguments(train)
    train.add_argument(
        "--resume",
        required=True,
        help="'none', 'auto', or an explicit local full-state checkpoint",
    )
    controls = train.add_mutually_exclusive_group()
    controls.add_argument("--stop-after-epoch", type=int)
    controls.add_argument("--smoke-optimizer-steps", type=int)
    train.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate source data and identities without constructing a model",
    )

    select = subparsers.add_parser(
        "select-cfg",
        help="validate the fixed CFG=1.0 on the frozen validation split",
    )
    _add_data_arguments(select)
    select.add_argument("--results-root", type=Path, required=True)

    test = subparsers.add_parser(
        "test",
        help="evaluate the frozen 6.7GHz test split exactly once",
    )
    _add_data_arguments(test)
    test.add_argument("--results-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "train" and arguments.preflight_only and (
        arguments.stop_after_epoch is not None
        or arguments.smoke_optimizer_steps is not None
    ):
        raise SameFrequencyTrainerContractError(
            "preflight-only cannot be combined with training controls"
        )

    beam_id, _config_id = infer_manifest_selection(
        arguments.manifest_path,
        arguments.array_size,
    )
    cfg = MultiscaleUNOTrainConfig(
        SameFrequencyTrainConfig(
            dataset_root=arguments.dataset_root,
            manifest_path=arguments.manifest_path,
            height_stats_path=arguments.height_stats_path,
            run_root=arguments.run_root,
            array_size=arguments.array_size,
            beam_id=beam_id,
            model_size="lite",
        )
    )
    device = resolve_device(arguments.device)
    if arguments.command == "train":
        result = run_same_frequency_multiscale_uno_training(
            cfg,
            InvocationControls(
                resume=arguments.resume,
                stop_after_epoch=arguments.stop_after_epoch,
                smoke_optimizer_steps=arguments.smoke_optimizer_steps,
            ),
            device,
            preflight_only=arguments.preflight_only,
        )
    elif arguments.command == "select-cfg":
        result = run_cfg_selection(cfg, device, arguments.results_root)
    elif arguments.command == "test":
        result = run_test_evaluation(cfg, device, arguments.results_root)
    else:
        raise RuntimeError(f"unsupported command: {arguments.command}")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
