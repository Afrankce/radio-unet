from __future__ import annotations

from train_same_frequency_attention_fno import build_parser


def test_attention_fno_cli_exposes_only_operational_controls() -> None:
    args = build_parser().parse_args(
        [
            "--dataset-root",
            "dataset",
            "--manifest-path",
            "manifest.jsonl",
            "--height-stats-path",
            "height.json",
            "--run-root",
            "run",
            "--array-size",
            "32x32",
            "--device",
            "cuda:0",
            "--resume",
            "auto",
            "--preflight-only",
        ]
    )

    assert args.array_size == "32x32"
    assert args.preflight_only is True
    assert args.resume == "auto"
    assert not hasattr(args, "model_size")
    assert not hasattr(args, "modes")
    assert not hasattr(args, "width")
    assert not hasattr(args, "cfg_scale")


def test_attention_fno_cli_keeps_stop_and_smoke_mutually_exclusive() -> None:
    parser = build_parser()
    common = [
        "--dataset-root",
        "dataset",
        "--manifest-path",
        "manifest.jsonl",
        "--height-stats-path",
        "height.json",
        "--run-root",
        "run",
        "--array-size",
        "8x8",
        "--device",
        "cpu",
        "--resume",
        "none",
    ]

    try:
        parser.parse_args(
            common + ["--stop-after-epoch", "1", "--smoke-optimizer-steps", "1"]
        )
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("mutually exclusive training controls were accepted")

