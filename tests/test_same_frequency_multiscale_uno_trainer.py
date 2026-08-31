from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from data_loaders.multiconfig import multiconfig_collate
from model.attention_multiscale_uno import AttentionMultiscaleUNO2d
from training.checkpointing import CHECKPOINT_KEYS, CheckpointIdentityError
from training.config import InvocationControls
from training.same_frequency_config import SameFrequencyTrainConfig
from training.same_frequency_multiscale_uno_config import (
    MULTISCALE_UNO_MODEL_SIZE,
    MultiscaleUNOConfigError,
    MultiscaleUNOTrainConfig,
)
from training.same_frequency_multiscale_uno_trainer import (
    run_same_frequency_multiscale_uno_training,
    write_or_validate_multiscale_uno_run_config,
)


class _TinyDataset(Dataset):
    def __init__(self, count: int, split: str) -> None:
        self.count = count
        self.split = split

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        target = torch.linspace(0.1, 0.9, 32).repeat(32, 1).unsqueeze(0)
        condition = torch.stack(
            (
                torch.zeros(32, 32),
                torch.full((32, 32), index / max(self.count, 1)),
                torch.full((32, 32), 0.5),
            )
        )
        condition[0, 15, 15] = 1.0
        return {
            "condition": condition.float(),
            "target": target.float(),
            "valid_mask": torch.ones(1, 32, 32, dtype=torch.bool),
            "metadata": {
                "sample_key": f"u{index + 1}|8x8|beam04",
                "split": self.split,
                "scene_id": f"u{index + 1}",
                "array_name": "8x8",
                "array_rows": 8,
                "array_cols": 8,
                "frequency_hz": 6_700_000_000,
                "config_id": "synthetic-zero-degree",
                "beam_id": 4,
                "steering_deg": 0.0,
                "height_path": "height.npy",
                "beam_map_path": "beam.npy",
                "radiomap_path": "radio.npy",
                "tx_rc": [127, 127],
            },
        }


def _config(tmp_path: Path, *, run_name: str = "run") -> MultiscaleUNOTrainConfig:
    return MultiscaleUNOTrainConfig(
        SameFrequencyTrainConfig(
            dataset_root=tmp_path / "dataset",
            manifest_path=tmp_path / "manifest.jsonl",
            height_stats_path=tmp_path / "height.json",
            run_root=tmp_path / run_name,
            array_size="8x8",
            beam_id=4,
            model_size="lite",
        )
    )


def _context() -> SimpleNamespace:
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


def _tiny_loaders(_cfg, _context):
    generator = torch.Generator(device="cpu").manual_seed(42)
    train = DataLoader(
        _TinyDataset(2, "train"),
        batch_size=2,
        shuffle=False,
        collate_fn=multiconfig_collate,
        num_workers=0,
    )
    val = DataLoader(
        _TinyDataset(1, "val"),
        batch_size=1,
        shuffle=False,
        collate_fn=multiconfig_collate,
        num_workers=0,
    )
    return train, val, generator


def _tiny_multiscale_uno(model_size: str) -> AttentionMultiscaleUNO2d:
    assert model_size == MULTISCALE_UNO_MODEL_SIZE
    return AttentionMultiscaleUNO2d(
        state_channels=(4, 4, 8, 16, 16),
        operator_width=4,
        operator_modes=(1, 1, 1, 1, 1),
        operator_padding=(0, 0, 0, 0, 0),
        encoder_features=(4, 4, 8, 16, 16, 4),
        attention_reduction=4,
    )


@pytest.fixture
def tiny_training(monkeypatch, tmp_path: Path):
    import training.same_frequency_multiscale_uno_trainer as module

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(module, "preflight_same_frequency", lambda _cfg: _context())
    monkeypatch.setattr(module, "build_same_frequency_loaders", _tiny_loaders)
    monkeypatch.setattr(module, "build_same_frequency_backbone", _tiny_multiscale_uno)
    return module, tmp_path


def _smoke_checkpoint(tmp_path: Path) -> tuple[MultiscaleUNOTrainConfig, dict]:
    cfg = _config(tmp_path)
    result = run_same_frequency_multiscale_uno_training(
        cfg,
        InvocationControls(resume="none", smoke_optimizer_steps=1),
        torch.device("cpu"),
    )
    return cfg, result


def test_multiscale_uno_run_config_is_written_and_revalidated(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    controls = InvocationControls(resume="auto")

    first = write_or_validate_multiscale_uno_run_config(cfg, controls)
    second = write_or_validate_multiscale_uno_run_config(cfg, controls)
    text = first.read_text(encoding="utf-8")
    record = json.loads(text)
    reloaded = MultiscaleUNOTrainConfig.from_json(text)

    assert first == second == cfg.run_dir / "config.json"
    assert record["model_size"] == MULTISCALE_UNO_MODEL_SIZE
    assert record["config_sha256"] == cfg.config_sha256
    assert reloaded.config_sha256 == cfg.config_sha256

    record["model_size"] = "attention_fno_lite"
    first.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(MultiscaleUNOConfigError, match="model_size"):
        write_or_validate_multiscale_uno_run_config(cfg, controls)


def test_one_step_multiscale_uno_smoke_writes_reloadable_full_state_checkpoint(
    tiny_training,
) -> None:
    _module, tmp_path = tiny_training
    cfg, result = _smoke_checkpoint(tmp_path)

    checkpoint = Path(result["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert result["status"] == "smoke_complete"
    assert result["optimizer_steps"] == 1
    assert set(payload) == CHECKPOINT_KEYS
    assert payload["run_identity"]["model_size"] == MULTISCALE_UNO_MODEL_SIZE
    assert payload["run_identity"]["config_sha256"] == cfg.config_sha256
    assert payload["trainer_state"]["optimizer_step"] == 1
    assert payload["optimizer"]["state"]
    assert payload["scheduler"]
    assert payload["rng_state"]


def test_auto_and_explicit_resume_reload_optimizer_progress(
    tiny_training,
    monkeypatch,
) -> None:
    module, tmp_path = tiny_training
    source_cfg, smoke = _smoke_checkpoint(tmp_path)
    checkpoint = Path(smoke["checkpoint"])

    def finish_without_more_training(self, *, stop_after_epoch=None):
        return {
            "optimizer_step": self.optimizer_step,
            "optimizer_state_nonempty": bool(self.optimizer.state),
            "stop_after_epoch": stop_after_epoch,
        }

    monkeypatch.setattr(module.MultiConfigSRMTrainer, "fit", finish_without_more_training)

    explicit_cfg = _config(tmp_path, run_name="explicit")
    explicit = run_same_frequency_multiscale_uno_training(
        explicit_cfg,
        InvocationControls(resume=str(checkpoint), stop_after_epoch=1),
        torch.device("cpu"),
    )

    auto_cfg = _config(tmp_path, run_name="auto")
    auto_cfg.run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, auto_cfg.run_dir / "last.pt")
    auto = run_same_frequency_multiscale_uno_training(
        auto_cfg,
        InvocationControls(resume="auto", stop_after_epoch=1),
        torch.device("cpu"),
    )

    assert source_cfg.config_sha256 == explicit_cfg.config_sha256 == auto_cfg.config_sha256
    assert explicit == {
        "optimizer_step": 1,
        "optimizer_state_nonempty": True,
        "stop_after_epoch": 1,
    }
    assert auto == explicit


def test_multiscale_uno_rejects_attention_fno_checkpoint_identity(
    tiny_training,
    monkeypatch,
) -> None:
    module, tmp_path = tiny_training
    _cfg, smoke = _smoke_checkpoint(tmp_path)
    checkpoint = Path(smoke["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["run_identity"]["model_size"] = "attention_fno_lite"
    attention_fno_checkpoint = tmp_path / "attention-fno.pt"
    torch.save(payload, attention_fno_checkpoint)
    monkeypatch.setattr(
        module.MultiConfigSRMTrainer,
        "fit",
        lambda self, *, stop_after_epoch=None: {"optimizer_step": self.optimizer_step},
    )

    with pytest.raises(CheckpointIdentityError, match="model_size"):
        run_same_frequency_multiscale_uno_training(
            _config(tmp_path, run_name="reject"),
            InvocationControls(resume=str(attention_fno_checkpoint), stop_after_epoch=1),
            torch.device("cpu"),
        )
