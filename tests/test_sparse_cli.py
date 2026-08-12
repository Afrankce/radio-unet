from __future__ import annotations

import pytest


def test_sparse_train_parser_only_accepts_formal_beam_masked_variant() -> None:
    from train_sparse_same_frequency import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset-root", "dataset",
            "--manifest-path", "manifest.jsonl",
            "--height-stats-path", "height.json",
            "--run-root", "run",
            "--array-size", "8x8",
            "--variant", "beam_masked",
            "--device", "cpu",
            "--resume", "none",
        ]
    )

    assert args.variant == "beam_masked"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--dataset-root", "dataset",
                "--manifest-path", "manifest.jsonl",
                "--height-stats-path", "height.json",
                "--run-root", "run",
                "--array-size", "8x8",
                "--variant", "no_beam_masked",
                "--device", "cpu",
                "--resume", "none",
            ]
        )


def test_sparse_evaluate_parser_requires_variant_and_array_size() -> None:
    from evaluate_sparse_same_frequency import build_parser

    args = build_parser().parse_args(
        [
            "select-cfg",
            "--dataset-root", "dataset",
            "--manifest-path", "manifest.jsonl",
            "--height-stats-path", "height.json",
            "--run-root", "run",
            "--results-root", "results",
            "--array-size", "32x32",
            "--variant", "beam_masked",
            "--device", "cpu",
        ]
    )

    assert args.command == "select-cfg"
    assert args.array_size == "32x32"
    assert args.variant == "beam_masked"
