from __future__ import annotations

import json

import pytest

import prepare_cross_frequency


def test_build_manifest_cli_exposes_only_data_and_output_paths(capsys) -> None:
    parser = prepare_cross_frequency.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["build-manifest", "--help"])
    help_text = capsys.readouterr().out

    assert "build-manifest" in help_text
    assert "--dataset-root" in help_text
    assert "--schema" in help_text
    assert "--manifest-dir" in help_text
    for forbidden in (
        "--frequency",
        "--frequency-hz",
        "--beam-id",
        "--steering-deg",
        "--array-size",
        "--train-scenes",
        "--test-scenes",
    ):
        assert forbidden not in help_text


def test_build_manifest_cli_dispatches_to_artifact_builder(monkeypatch, tmp_path, capsys) -> None:
    called: dict[str, object] = {}

    def fake_builder(**kwargs):
        called.update(kwargs)
        return {
            "manifest": str(tmp_path / "manifest.jsonl"),
            "records": 800,
        }

    monkeypatch.setattr(
        prepare_cross_frequency,
        "build_cross_frequency_manifest_artifact",
        fake_builder,
    )

    result = prepare_cross_frequency.main(
        [
            "build-manifest",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--schema",
            str(tmp_path / "schema.json"),
            "--manifest-dir",
            str(tmp_path / "manifests"),
        ]
    )

    assert result == 0
    assert called == {
        "dataset_root": tmp_path / "dataset",
        "schema_path": tmp_path / "schema.json",
        "manifest_dir": tmp_path / "manifests",
        "output_path": None,
    }
    assert json.loads(capsys.readouterr().out)["records"] == 800
