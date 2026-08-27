from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from evaluation.same_frequency_evaluator import run_cfg_selection, run_test_evaluation
from training.multiconfig_trainer import resolve_device
from training.same_frequency_attention_fno_config import AttentionFNOTrainConfig
from training.same_frequency_config import SameFrequencyTrainConfig
from training.same_frequency_trainer import infer_manifest_selection


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--height-stats-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--array-size",
        choices=("8x8", "16x16", "32x32"),
        required=True,
    )
    parser.add_argument("--device", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the attention-conditioned full-resolution FNO on the "
            "6.7GHz zero-degree single-beam RadioFlow protocol"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser(
        "select-cfg",
        help="validate the fixed CFG=1.0 on the frozen validation split",
    )
    _add_common_arguments(select)
    test = subparsers.add_parser(
        "test",
        help="evaluate the frozen 6.7GHz test split exactly once",
    )
    _add_common_arguments(test)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    beam_id, _config_id = infer_manifest_selection(
        arguments.manifest_path,
        arguments.array_size,
    )
    cfg = AttentionFNOTrainConfig(
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
    if arguments.command == "select-cfg":
        result = run_cfg_selection(cfg, device, arguments.results_root)
    elif arguments.command == "test":
        result = run_test_evaluation(cfg, device, arguments.results_root)
    else:
        raise RuntimeError(f"unsupported command: {arguments.command}")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

