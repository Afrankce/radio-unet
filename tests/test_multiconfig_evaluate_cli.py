from __future__ import annotations

from pathlib import Path

import pytest
import torch


def _common_arguments(tmp_path: Path) -> list[str]:
    return [
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--manifest-dir",
        str(tmp_path / "dataset" / "manifests"),
        "--array",
        "16x16",
        "--model-size",
        "lite",
        "--run-root",
        str(tmp_path / "runs"),
        "--results-root",
        str(tmp_path / "results"),
        "--device",
        "cpu",
    ]


@pytest.mark.parametrize(
    ("command", "function_name"),
    [
        ("select-cfg", "run_cfg_selection"),
        ("test", "run_test_evaluation"),
    ],
)
def test_evaluation_cli_builds_only_the_locked_training_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    function_name: str,
) -> None:
    cli = __import__("evaluate_multiconfig")
    captured = []

    def fake_action(cfg, device, results_root):
        captured.append((cfg, device, results_root))
        return {"status": "ok"}

    monkeypatch.setattr(cli, function_name, fake_action)

    assert cli.main([command, *_common_arguments(tmp_path)]) == 0

    cfg, device, results_root = captured[0]
    assert cfg.array_size == "16x16"
    assert cfg.model_size == "lite"
    assert cfg.resolution == 256
    assert cfg.use_amp is True
    assert cfg.run_dir == tmp_path / "runs" / "16x16" / "lite"
    assert device == torch.device("cpu")
    assert results_root == tmp_path / "results"


@pytest.mark.parametrize(
    "forbidden",
    [
        "--cfg",
        "--cfg-scale",
        "--solver",
        "--steps",
        "--resolution",
        "--batch-size",
        "--ema",
        "--seed",
        "--beam",
        "--frequency",
    ],
)
def test_evaluation_cli_rejects_scientific_overrides(
    tmp_path: Path,
    forbidden: str,
) -> None:
    cli = __import__("evaluate_multiconfig")

    with pytest.raises(SystemExit):
        cli.main(["test", *_common_arguments(tmp_path), forbidden, "2"])


def test_summarize_cli_accepts_only_run_and_result_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = __import__("evaluate_multiconfig")
    captured = []
    monkeypatch.setattr(
        cli,
        "summarize_benchmark",
        lambda run_root, results_root: captured.append((run_root, results_root))
        or {"status": "complete"},
    )

    assert cli.main(
        [
            "summarize",
            "--run-root",
            str(tmp_path / "runs"),
            "--results-root",
            str(tmp_path / "results"),
        ]
    ) == 0
    assert captured == [(tmp_path / "runs", tmp_path / "results")]

    with pytest.raises(SystemExit):
        cli.main(
            [
                "summarize",
                "--run-root",
                str(tmp_path / "runs"),
                "--results-root",
                str(tmp_path / "results"),
                "--array",
                "8x8",
            ]
        )
