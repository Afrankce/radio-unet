from __future__ import annotations

from pathlib import Path

from training.random_task2_config import RandomTask2TrainConfig


def _base_kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "dataset_root": tmp_path / "dataset",
        "manifest_path": tmp_path / "manifest.jsonl",
        "height_stats_path": tmp_path / "height.json",
        "run_root": tmp_path / "runs",
        "array_size": "8x8",
    }


def test_config_accepts_both_modes_and_hashes_them_differently(tmp_path: Path) -> None:
    regression = RandomTask2TrainConfig(**_base_kwargs(tmp_path), mode="regression")
    pinned = RandomTask2TrainConfig(**_base_kwargs(tmp_path), mode="pinned_fm")

    assert regression.mode == "regression"
    assert pinned.mode == "pinned_fm"
    assert regression.config_sha256 != pinned.config_sha256
    assert regression.run_dir != pinned.run_dir


def test_train_and_evaluate_parsers_accept_both_modes() -> None:
    from evaluate_random_task2 import build_parser as build_evaluate_parser
    from train_random_task2 import build_parser as build_train_parser

    train_args = build_train_parser().parse_args(
        [
            "--dataset-root",
            "dataset",
            "--manifest-path",
            "manifest.jsonl",
            "--height-stats-path",
            "height.json",
            "--run-root",
            "runs",
            "--array-size",
            "8x8",
            "--variant",
            "feature4",
            "--mode",
            "pinned_fm",
            "--device",
            "cpu",
        ]
    )
    evaluate_args = build_evaluate_parser().parse_args(
        [
            "--dataset-root",
            "dataset",
            "--manifest-path",
            "manifest.jsonl",
            "--height-stats-path",
            "height.json",
            "--run-root",
            "runs",
            "--array-size",
            "8x8",
            "--variant",
            "feature4",
            "--mode",
            "pinned_fm",
            "--device",
            "cpu",
        ]
    )

    assert train_args.mode == "pinned_fm"
    assert evaluate_args.mode == "pinned_fm"
