from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from evaluation.same_frequency_evaluator import (
    PreparedSameFrequencyEvaluation,
    _prepare_evaluation,
    build_cfg_selection_payload,
    load_cfg_selection,
)
from experiments.multiconfig_manifest import canonical_json_bytes
from training.checkpointing import CheckpointIdentity, TrainerState
from training.same_frequency_config import SameFrequencyTrainConfig
from training.same_frequency_fno_config import (
    PAPER_FNO_MODEL_SIZE,
    PaperFNOTrainConfig,
)


def _base_config(tmp_path: Path) -> SameFrequencyTrainConfig:
    return SameFrequencyTrainConfig(
        dataset_root=tmp_path / "dataset",
        manifest_path=tmp_path / "manifest.jsonl",
        height_stats_path=tmp_path / "height.json",
        run_root=tmp_path / "run",
        array_size="8x8",
        beam_id=4,
        model_size="lite",
    )


def _context():
    return SimpleNamespace(
        beam_id=4,
        config_id="freq_6.7GHz_64TR_8beams_pattern_tr38901",
        manifest_sha256="1" * 64,
        split_sha256="2" * 64,
        schema_sha256="3" * 64,
        archive_sha256="4" * 64,
        dataset_revision="5" * 40,
        git_commit="6" * 40,
    )


def _selection_identity() -> dict[str, str]:
    return {
        "checkpoint_sha256": "0" * 64,
        "config_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "split_sha256": "3" * 64,
        "schema_sha256": "4" * 64,
        "archive_sha256": "5" * 64,
        "dataset_revision": "6" * 40,
        "radioflow_upstream_base": "7" * 40,
        "git_commit": "8" * 40,
    }


def _trainer_state(rmse: float = 10.0) -> TrainerState:
    return TrainerState(
        completed_epochs=1,
        next_epoch_index=1,
        optimizer_step=10,
        micro_batches_seen=280,
        samples_seen=560,
        best_val_db_rmse=rmse,
        epochs_without_improvement=0,
        history=({"epoch": 1, "val_db_rmse": rmse},),
    )


def _metrics(rmse: float, n_samples: int = 80) -> dict[str, int | float]:
    return {
        "n_samples": n_samples,
        "n_valid_pixels": 1000,
        "db_rmse": rmse,
        "db_mae": rmse / 2,
        "mse": (rmse / 300.0) ** 2,
        "nmse": 0.01,
        "psnr": 25.0,
        "ssim": 0.8,
        "raw_fraction_below_zero": 0.0,
        "raw_fraction_above_one": 0.0,
    }


def _prepared(cfg) -> PreparedSameFrequencyEvaluation:
    return PreparedSameFrequencyEvaluation(
        cfg=cfg,
        context=_context(),
        model=torch.nn.Identity(),
        device=torch.device("cpu"),
        checkpoint_path=Path("best.pt"),
        checkpoint_identity=SimpleNamespace(),
        selection_identity=_selection_identity(),
        trainer_state=_trainer_state(),
    )


def test_fno_selection_payload_is_locked_to_cfg_one(tmp_path: Path) -> None:
    cfg = PaperFNOTrainConfig(_base_config(tmp_path))

    payload = build_cfg_selection_payload(
        prepared=_prepared(cfg),
        candidate_metrics={1.0: _metrics(10.0)},
    )

    assert payload["experiment"] == "same_frequency_6.7_single_beam_paper_fno"
    assert payload["model_size"] == PAPER_FNO_MODEL_SIZE
    assert payload["candidates"] == [1.0]
    assert payload["selected_scale"] == 1.0
    assert list(payload["candidate_metrics"]) == ["1.0"]


def test_unet_selection_grid_is_unchanged(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    candidates = (1.0, 1.5, 2.0, 2.5)

    payload = build_cfg_selection_payload(
        prepared=_prepared(cfg),
        candidate_metrics={scale: _metrics(10.0 + scale) for scale in candidates},
    )

    assert payload["experiment"] == "same_frequency_6.7_single_beam"
    assert payload["candidates"] == [1.0, 1.5, 2.0, 2.5]


def test_fno_selection_round_trips_with_exact_candidate_grid(tmp_path: Path) -> None:
    cfg = PaperFNOTrainConfig(_base_config(tmp_path))
    prepared = _prepared(cfg)
    payload = build_cfg_selection_payload(
        prepared=prepared,
        candidate_metrics={1.0: _metrics(10.0)},
    )
    path = tmp_path / "cfg_selection.json"
    path.write_bytes(canonical_json_bytes(payload))

    restored = load_cfg_selection(
        path,
        expected_identity=prepared.selection_identity,
        cfg=cfg,
        context=prepared.context,
    )

    assert restored == payload


def test_prepare_evaluation_routes_fno_model_size_to_shared_factory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import evaluation.same_frequency_evaluator as module

    cfg = PaperFNOTrainConfig(_base_config(tmp_path))
    cfg.run_dir.mkdir(parents=True)
    (cfg.run_dir / "best.pt").write_bytes(b"checkpoint identity source")
    model = torch.nn.Conv2d(1, 1, 1)
    context = _context()
    requested: list[str] = []
    identity = CheckpointIdentity(
        array_size="8x8",
        model_size=PAPER_FNO_MODEL_SIZE,
        condition_channels=3,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        manifest_sha256=context.manifest_sha256,
        split_sha256=context.split_sha256,
        schema_sha256=context.schema_sha256,
        config_sha256=cfg.config_sha256,
        archive_sha256=context.archive_sha256,
        dataset_revision=context.dataset_revision,
        radioflow_upstream_base="7" * 40,
        git_commit=context.git_commit,
        seed=42,
    )

    monkeypatch.setattr(module, "preflight_same_frequency", lambda _cfg: context)
    monkeypatch.setattr(
        module,
        "build_same_frequency_backbone",
        lambda model_size: requested.append(model_size) or model,
    )
    monkeypatch.setattr(
        module,
        "build_same_frequency_checkpoint_identity",
        lambda _cfg, _context, _model: identity,
    )
    monkeypatch.setattr(
        module,
        "load_ema_for_evaluation",
        lambda *args, **kwargs: _trainer_state(),
    )

    prepared = _prepare_evaluation(cfg, torch.device("cpu"))

    assert requested == [PAPER_FNO_MODEL_SIZE]
    assert prepared.model is model
