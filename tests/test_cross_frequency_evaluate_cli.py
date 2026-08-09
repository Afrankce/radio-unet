from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_cross_frequency_evaluation_cli_locks_protocol(capsys) -> None:
    import evaluate_cross_frequency

    parser = evaluate_cross_frequency.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["select-cfg", "--help"])
    help_text = capsys.readouterr().out
    assert "--manifest-path" in help_text
    assert "--results-root" in help_text
    assert "select-cfg" in parser.format_help()
    for forbidden in (
        "--frequency-hz",
        "--train-frequency",
        "--test-frequency",
        "--beam-id",
        "--steering-deg",
        "--array-size",
        "--val-samples",
    ):
        assert forbidden not in help_text


def test_cross_frequency_evaluation_cli_dispatches_select(monkeypatch, tmp_path: Path, capsys) -> None:
    import evaluate_cross_frequency

    called: dict[str, object] = {}
    monkeypatch.setattr(
        evaluate_cross_frequency,
        "run_cfg_selection",
        lambda cfg, device, results_root: called.update(
            {"cfg": cfg, "device": device, "results_root": results_root}
        )
        or {"status": "selected"},
    )
    monkeypatch.setattr(evaluate_cross_frequency, "resolve_device", lambda value: value)

    result = evaluate_cross_frequency.main(
        [
            "select-cfg",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--manifest-path",
            str(tmp_path / "manifest.jsonl"),
            "--height-stats-path",
            str(tmp_path / "height.json"),
            "--run-root",
            str(tmp_path / "runs"),
            "--results-root",
            str(tmp_path / "results"),
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    assert called["device"] == "cpu"
    assert called["cfg"].model_size == "lite"
    assert json.loads(capsys.readouterr().out)["status"] == "selected"
