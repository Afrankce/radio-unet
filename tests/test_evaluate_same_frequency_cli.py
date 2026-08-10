from __future__ import annotations

from evaluate_same_frequency import build_parser


def test_same_frequency_evaluator_parser_requires_array_size() -> None:
    args = build_parser().parse_args(
        [
            "select-cfg",
            "--dataset-root", "dataset",
            "--manifest-path", "manifest.jsonl",
            "--height-stats-path", "height.json",
            "--run-root", "run",
            "--results-root", "results",
            "--array-size", "16x16",
            "--device", "cpu",
        ]
    )

    assert args.command == "select-cfg"
    assert args.array_size == "16x16"
