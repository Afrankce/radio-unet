from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import evaluation.same_frequency_evaluator as evaluator
import run_same_frequency_multiscale_uno as cli
from model.attention_multiscale_uno import AttentionMultiscaleUNO2d
from train import ModelEMA
from training.checkpointing import (
    TrainerState,
    load_ema_for_evaluation,
    save_checkpoint_atomic,
)
from training.complex_grad_scaler import ComplexGradScaler
from training.optimization import build_optimizer_step_scheduler
from training.same_frequency_config import SameFrequencyTrainConfig
from training.same_frequency_multiscale_uno_config import (
    MULTISCALE_UNO_MODEL_SIZE,
    MultiscaleUNOTrainConfig,
)
from training.same_frequency_trainer import (
    build_same_frequency_checkpoint_identity,
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


def _tiny_multiscale_uno() -> AttentionMultiscaleUNO2d:
    return AttentionMultiscaleUNO2d(
        state_channels=(4, 4, 8, 16, 16),
        operator_width=4,
        operator_modes=(1, 1, 1, 1, 1),
        operator_padding=(0, 0, 0, 0, 0),
        encoder_features=(4, 4, 8, 16, 16, 4),
        attention_reduction=4,
    )


def _evaluation_context() -> SimpleNamespace:
    return SimpleNamespace(
        train_dataset=range(560),
        val_dataset=range(80),
        test_dataset=range(160),
        manifest_path=Path("manifest.jsonl"),
        manifest_sha256="1" * 64,
        split_sha256="2" * 64,
        schema_sha256="3" * 64,
        archive_sha256="4" * 64,
        dataset_revision="5" * 40,
        git_commit="6" * 40,
        height_max=100.0,
        beam_id=4,
        config_id="synthetic-zero-degree",
    )


def _evaluation_config(tmp_path: Path) -> MultiscaleUNOTrainConfig:
    return MultiscaleUNOTrainConfig(
        SameFrequencyTrainConfig(
            dataset_root=tmp_path / "dataset",
            manifest_path=tmp_path / "manifest.jsonl",
            height_stats_path=tmp_path / "height.json",
            run_root=tmp_path / "run",
            array_size="16x16",
            beam_id=4,
            model_size="lite",
        )
    )


def _write_best_checkpoint(
    cfg: MultiscaleUNOTrainConfig,
    context: SimpleNamespace,
) -> None:
    model = _tiny_multiscale_uno()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    ema = ModelEMA(model, decay=cfg.ema_decay)
    scheduler = build_optimizer_step_scheduler(
        optimizer,
        total_steps=cfg.planned_optimizer_steps,
        warmup_steps=cfg.warmup_steps,
    )
    scaler = ComplexGradScaler("cuda", enabled=False)
    state = TrainerState(
        completed_epochs=1,
        next_epoch_index=1,
        optimizer_step=1,
        micro_batches_seen=1,
        samples_seen=2,
        best_val_db_rmse=1.25,
        epochs_without_improvement=0,
        history=({"epoch": 1, "val_db_rmse": 1.25},),
    )
    save_checkpoint_atomic(
        cfg.run_dir / "best.pt",
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        trainer_state=state,
        identity=build_same_frequency_checkpoint_identity(cfg, context, model),
        train_generator=torch.Generator(device="cpu").manual_seed(cfg.seed),
    )


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


def test_select_cfg_preparation_loads_multiscale_backbone_and_identity(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    cfg = _evaluation_config(tmp_path)
    context = _evaluation_context()
    _write_best_checkpoint(cfg, context)
    capsys.readouterr()
    observed: dict[str, object] = {"factory_model_sizes": []}

    monkeypatch.setattr(cli, "infer_manifest_selection", lambda *_args: (4, "cfg"))
    monkeypatch.setattr(evaluator, "preflight_same_frequency", lambda _cfg: context)

    def tiny_factory(model_size: str) -> AttentionMultiscaleUNO2d:
        observed["factory_model_sizes"].append(model_size)
        return _tiny_multiscale_uno()

    monkeypatch.setattr(evaluator, "build_same_frequency_backbone", tiny_factory)

    def observed_strict_load(path, *, model, expected_identity):
        observed["loaded_model"] = model
        observed["expected_identity"] = expected_identity
        return load_ema_for_evaluation(
            path,
            model=model,
            expected_identity=expected_identity,
        )

    monkeypatch.setattr(evaluator, "load_ema_for_evaluation", observed_strict_load)
    monkeypatch.setattr(
        evaluator,
        "_evaluate_validation_candidate",
        lambda _prepared, scale: {
            "n_samples": 80,
            "db_rmse": 1.25 if scale == 1.0 else 9.0,
        },
    )
    argv = [
        "select-cfg",
        "--dataset-root",
        str(cfg.dataset_root),
        "--manifest-path",
        str(cfg.manifest_path),
        "--height-stats-path",
        str(cfg.height_stats_path),
        "--run-root",
        str(cfg.run_root),
        "--results-root",
        str(tmp_path / "results"),
        "--array-size",
        cfg.array_size,
        "--device",
        "cpu",
    ]

    assert cli.main(argv) == 0
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    identity = observed["expected_identity"]

    assert result["status"] == "selected"
    assert observed["factory_model_sizes"] == [MULTISCALE_UNO_MODEL_SIZE]
    assert isinstance(observed["loaded_model"], AttentionMultiscaleUNO2d)
    assert identity.model_size == MULTISCALE_UNO_MODEL_SIZE
    assert identity.config_sha256 == cfg.config_sha256


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
