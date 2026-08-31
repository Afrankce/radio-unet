from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import run_same_frequency_multiscale_uno as cli
from training.same_frequency_multiscale_uno_config import (
    MULTISCALE_UNO_MODEL_SIZE,
    MultiscaleUNOTrainConfig,
)


def _common(command: str) -> list[str]:
    arguments = [
        command,
        "--dataset-root",
        "dataset",
        "--manifest-path",
        "manifest.jsonl",
        "--height-stats-path",
        "height.json",
        "--run-root",
        "run",
        "--array-size",
        "16x16",
        "--device",
        "cpu",
    ]
    if command == "train":
        arguments.extend(("--resume", "auto", "--smoke-optimizer-steps", "1"))
    else:
        arguments.extend(("--results-root", "results"))
    return arguments


@pytest.mark.parametrize("command", ["train", "select-cfg", "test"])
def test_lifecycle_parser_accepts_exact_subcommands(command: str) -> None:
    arguments = cli.build_parser().parse_args(_common(command))

    assert arguments.command == command
    assert arguments.array_size == "16x16"


def test_lifecycle_parser_rejects_unknown_subcommand() -> None:
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(["evaluate"])

    assert error.value.code == 2


@pytest.mark.parametrize(
    "flag",
    ["--width", "--modes", "--padding", "--channels", "--model-size", "--cfg-scale"],
)
def test_lifecycle_parser_rejects_architecture_and_cfg_overrides(flag: str) -> None:
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(_common("train") + [flag, "1"])

    assert error.value.code == 2


def test_train_parser_keeps_stop_and_smoke_mutually_exclusive() -> None:
    arguments = _common("train")
    arguments[-2:] = [
        "--stop-after-epoch",
        "1",
        "--smoke-optimizer-steps",
        "1",
    ]

    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(arguments)

    assert error.value.code == 2


def _patch_common_boundaries(monkeypatch) -> None:
    monkeypatch.setattr(cli, "infer_manifest_selection", lambda *_args: (4, "cfg"))
    monkeypatch.setattr(cli, "resolve_device", lambda value: torch.device(value))


def test_train_route_constructs_frozen_config_and_prints_one_json(
    monkeypatch,
    capsys,
) -> None:
    _patch_common_boundaries(monkeypatch)
    observed = {}

    def fake_train(cfg, controls, device, *, preflight_only=False):
        observed.update(
            cfg=cfg,
            controls=controls,
            device=device,
            preflight_only=preflight_only,
        )
        return {"route": "train", "status": "ok"}

    monkeypatch.setattr(cli, "run_same_frequency_multiscale_uno_training", fake_train)

    assert cli.main(_common("train")) == 0
    captured = capsys.readouterr()

    assert captured.err == ""
    assert captured.out == '{"route": "train", "status": "ok"}\n'
    assert isinstance(observed["cfg"], MultiscaleUNOTrainConfig)
    assert observed["cfg"].model_size == MULTISCALE_UNO_MODEL_SIZE
    assert observed["cfg"].base.model_size == "lite"
    assert observed["cfg"].beam_id == 4
    assert observed["controls"].resume == "auto"
    assert observed["controls"].smoke_optimizer_steps == 1
    assert observed["device"] == torch.device("cpu")
    assert observed["preflight_only"] is False


@pytest.mark.parametrize(
    ("command", "target"),
    [("select-cfg", "run_cfg_selection"), ("test", "run_test_evaluation")],
)
def test_evaluation_routes_use_existing_same_frequency_evaluator(
    command: str,
    target: str,
    monkeypatch,
    capsys,
) -> None:
    _patch_common_boundaries(monkeypatch)
    observed = {}

    def fake_route(cfg, device, results_root):
        observed.update(cfg=cfg, device=device, results_root=results_root)
        return {"route": command, "status": "ok"}

    def wrong_route(*_args, **_kwargs):
        raise AssertionError("wrong evaluator route")

    monkeypatch.setattr(cli, target, fake_route)
    other = "run_test_evaluation" if target == "run_cfg_selection" else "run_cfg_selection"
    monkeypatch.setattr(cli, other, wrong_route)

    assert cli.main(_common(command)) == 0
    captured = capsys.readouterr()

    assert captured.err == ""
    assert json.loads(captured.out) == {"route": command, "status": "ok"}
    assert captured.out.count("\n") == 1
    assert isinstance(observed["cfg"], MultiscaleUNOTrainConfig)
    assert observed["cfg"].model_size == MULTISCALE_UNO_MODEL_SIZE
    assert observed["device"] == torch.device("cpu")
    assert observed["results_root"] == Path("results")
