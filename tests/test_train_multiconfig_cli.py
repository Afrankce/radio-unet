from __future__ import annotations

from pathlib import Path

import pytest
import torch


def _base_arguments(tmp_path: Path) -> list[str]:
    return [
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--manifest-dir",
        str(tmp_path / "manifests"),
        "--array",
        "8x8",
        "--model-size",
        "lite",
        "--run-root",
        str(tmp_path / "runs"),
        "--device",
        "cpu",
        "--resume",
        "none",
    ]


def test_cli_builds_locked_config_and_run_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = __import__("train_multiconfig")
    captured = []

    def fake_run(cfg, controls, device):
        captured.append((cfg, controls, device))
        return {"status": "paused", "run_dir": str(cfg.run_dir)}

    monkeypatch.setattr(cli, "run_benchmark_training", fake_run)

    assert cli.main(_base_arguments(tmp_path)) == 0

    cfg, controls, device = captured[0]
    assert cfg.array_size == "8x8"
    assert cfg.model_size == "lite"
    assert cfg.run_dir == tmp_path / "runs" / "8x8" / "lite"
    assert cfg.micro_batch_size == 2
    assert cfg.accumulation_steps == 28
    assert cfg.effective_batch_size == 56
    assert controls.resume == "none"
    assert device == torch.device("cpu")


@pytest.mark.parametrize(
    "forbidden",
    [
        "--resolution",
        "--batch-size",
        "--accumulation-steps",
        "--activation-checkpointing",
        "--cfg-drop-prob",
        "--frequency",
        "--beam",
        "--seed",
        "--learning-rate",
        "--epochs",
    ],
)
def test_cli_rejects_scientific_overrides(tmp_path: Path, forbidden: str) -> None:
    cli = __import__("train_multiconfig")

    with pytest.raises(SystemExit):
        cli.main([*_base_arguments(tmp_path), forbidden, "123"])


@pytest.mark.parametrize("resume", ["none", "auto", "D:/runs/last.pt"])
def test_cli_accepts_locked_resume_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resume: str,
) -> None:
    cli = __import__("train_multiconfig")
    captured = []
    monkeypatch.setattr(
        cli,
        "run_benchmark_training",
        lambda cfg, controls, device: captured.append(controls) or {"status": "ok"},
    )
    arguments = _base_arguments(tmp_path)
    arguments[-1] = resume

    assert cli.main(arguments) == 0
    assert captured[0].resume == resume


def test_cli_keeps_pause_and_smoke_as_invocation_only_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = __import__("train_multiconfig")
    captured = []
    monkeypatch.setattr(
        cli,
        "run_benchmark_training",
        lambda cfg, controls, device: captured.append((cfg, controls))
        or {"status": "ok"},
    )

    assert cli.main([*_base_arguments(tmp_path), "--stop-after-epoch", "5"]) == 0
    pause_cfg, pause = captured.pop()
    assert pause.stop_after_epoch == 5
    assert pause_cfg.max_epochs == 1000
    assert pause_cfg.planned_optimizer_steps == 80_000

    assert cli.main(
        [*_base_arguments(tmp_path), "--smoke-optimizer-steps", "1"]
    ) == 0
    smoke_cfg, smoke = captured.pop()
    assert smoke.smoke_optimizer_steps == 1
    assert smoke_cfg.config_sha256 == pause_cfg.config_sha256


def test_cli_rejects_simultaneous_pause_and_smoke(tmp_path: Path) -> None:
    cli = __import__("train_multiconfig")

    with pytest.raises(SystemExit):
        cli.main(
            [
                *_base_arguments(tmp_path),
                "--stop-after-epoch",
                "5",
                "--smoke-optimizer-steps",
                "1",
            ]
        )

def test_cli_accepts_train_scale_0p1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = __import__("train_multiconfig")
    captured = []
    monkeypatch.setattr(
        cli,
        "run_benchmark_training",
        lambda cfg, controls, device: captured.append(cfg) or {"status": "ok"},
    )

    assert cli.main([*_base_arguments(tmp_path), "--train-scale", "0.1"]) == 0
    assert captured[0].train_scale == 0.1
    assert captured[0].train_samples == 448
