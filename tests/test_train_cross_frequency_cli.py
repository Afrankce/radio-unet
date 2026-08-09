from __future__ import annotations

import json
from pathlib import Path

import train_cross_frequency


def test_cross_frequency_cli_has_no_scientific_overrides() -> None:
    parser = train_cross_frequency.build_parser()
    help_text = parser.format_help()
    assert "--dataset-root" in help_text
    assert "--manifest-path" in help_text
    assert "--height-stats-path" in help_text
    assert "--preflight-only" in help_text
    for forbidden in (
        "--array",
        "--frequency-hz",
        "--train-frequency",
        "--test-frequency",
        "--beam-id",
        "--steering-deg",
        "--train-samples",
        "--val-samples",
        "--test-samples",
    ):
        assert forbidden not in help_text


def test_cross_frequency_cli_dispatches_preflight(monkeypatch, tmp_path: Path, capsys) -> None:
    called: dict[str, object] = {}

    def fake_run(cfg, controls, device, *, preflight_only):
        called.update(
            {
                "cfg": cfg,
                "controls": controls,
                "device": device,
                "preflight_only": preflight_only,
            }
        )
        return {"status": "preflight_complete"}

    monkeypatch.setattr(train_cross_frequency, "run_cross_frequency_training", fake_run)
    monkeypatch.setattr(
        train_cross_frequency,
        "resolve_device",
        lambda value: value,
    )

    result = train_cross_frequency.main(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--manifest-path",
            str(tmp_path / "manifest.jsonl"),
            "--height-stats-path",
            str(tmp_path / "height.json"),
            "--run-root",
            str(tmp_path / "runs"),
            "--device",
            "cpu",
            "--resume",
            "none",
            "--preflight-only",
        ]
    )

    assert result == 0
    assert called["preflight_only"] is True
    assert called["device"] == "cpu"
    assert called["cfg"].model_size == "lite"
    assert json.loads(capsys.readouterr().out)["status"] == "preflight_complete"
