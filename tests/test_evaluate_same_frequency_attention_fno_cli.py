from __future__ import annotations

from evaluate_same_frequency_attention_fno import build_parser


def test_attention_fno_evaluator_cli_has_fixed_architecture_and_cfg() -> None:
    args = build_parser().parse_args(
        [
            "select-cfg",
            "--dataset-root",
            "dataset",
            "--manifest-path",
            "manifest.jsonl",
            "--height-stats-path",
            "height.json",
            "--run-root",
            "run",
            "--results-root",
            "results",
            "--array-size",
            "16x16",
            "--device",
            "cuda:0",
        ]
    )

    assert args.command == "select-cfg"
    assert args.array_size == "16x16"
    assert not hasattr(args, "model_size")
    assert not hasattr(args, "cfg_scale")
    assert not hasattr(args, "modes")
    assert not hasattr(args, "width")

