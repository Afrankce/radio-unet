from __future__ import annotations

import copy
import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
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
from training.same_frequency_trainer import SameFrequencyTrainerContractError


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
        _TinyDataset(4, "train"),
        batch_size=2,
        shuffle=True,
        generator=generator,
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


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _nested_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _process_rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "train_generator": generator.get_state(),
    }


def _trainer_state(trainer) -> dict[str, Any]:
    return {
        "completed_epochs": trainer.completed_epochs,
        "next_epoch_index": trainer.next_epoch_index,
        "optimizer_step": trainer.optimizer_step,
        "micro_batches_seen": trainer.micro_batches_seen,
        "samples_seen": trainer.samples_seen,
        "best_val_db_rmse": trainer.best_val_db_rmse,
        "epochs_without_improvement": trainer.epochs_without_improvement,
        "history": copy.deepcopy(trainer.history),
    }


def _trainer_snapshot(trainer) -> dict[str, Any]:
    return {
        "model": copy.deepcopy(trainer.model.state_dict()),
        "ema": copy.deepcopy(trainer.ema.ema_model.state_dict()),
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "scheduler": copy.deepcopy(trainer.scheduler.state_dict()),
        "scaler": copy.deepcopy(trainer.scaler.state_dict()),
        "trainer_state": _trainer_state(trainer),
        "rng_state": copy.deepcopy(_process_rng_state(trainer.train_generator)),
    }


def _assert_full_checkpoint_restored(trainer, payload: dict[str, Any]) -> None:
    assert _nested_equal(trainer.model.state_dict(), payload["model"]), (
        "model state was not fully restored"
    )
    assert _nested_equal(trainer.ema.ema_model.state_dict(), payload["ema"]), (
        "EMA state was not fully restored"
    )
    assert _nested_equal(trainer.optimizer.state_dict(), payload["optimizer"]), (
        "optimizer state was not fully restored"
    )
    assert _nested_equal(trainer.scheduler.state_dict(), payload["scheduler"]), (
        "scheduler state was not fully restored"
    )
    assert _nested_equal(trainer.scaler.state_dict(), payload["scaler"]), (
        "scaler state was not fully restored"
    )
    assert _nested_equal(_trainer_state(trainer), payload["trainer_state"]), (
        "trainer counters or history were not fully restored"
    )
    restored_rng = _process_rng_state(trainer.train_generator)
    for source, expected in payload["rng_state"].items():
        assert _nested_equal(restored_rng[source], expected), (
            f"{source} RNG state was not fully restored"
        )


@pytest.fixture
def tiny_training(monkeypatch, tmp_path: Path):
    import training.same_frequency_multiscale_uno_trainer as module

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(module, "preflight_same_frequency", lambda _cfg: _context())
    monkeypatch.setattr(module, "build_same_frequency_loaders", _tiny_loaders)
    monkeypatch.setattr(module, "build_same_frequency_backbone", _tiny_multiscale_uno)
    real_run_smoke = module.MultiConfigSRMTrainer.run_smoke

    def run_smoke_after_advancing_all_rngs(self, optimizer_steps: int):
        assert self.train_loader.sampler.generator is self.train_generator
        random.random()
        np.random.random()
        torch.rand(())
        if torch.cuda.is_available():
            torch.rand((), device="cuda")
        torch.rand((), generator=self.train_generator)
        generator_before_shuffle = self.train_generator.get_state().clone()

        result = real_run_smoke(self, optimizer_steps)

        assert not torch.equal(
            generator_before_shuffle,
            self.train_generator.get_state(),
        ), "the real shuffled sampler did not advance the loader generator"
        return result

    monkeypatch.setattr(
        module.MultiConfigSRMTrainer,
        "run_smoke",
        run_smoke_after_advancing_all_rngs,
    )
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


def test_auto_and_explicit_resume_restore_every_checkpoint_component(
    tiny_training,
    monkeypatch,
) -> None:
    module, tmp_path = tiny_training
    source_cfg, smoke = _smoke_checkpoint(tmp_path)
    checkpoint = Path(smoke["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    verified_resumes: list[int] = []
    observed: dict[str, Any] = {
        "fresh_rng_states": [],
        "scaler_load_payloads": [],
    }
    real_trainer = module.MultiConfigSRMTrainer

    class ObservedResumeTrainer(real_trainer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            real_scaler_load_state_dict = self.scaler.load_state_dict

            def tracked_scaler_load_state_dict(scaler_payload):
                observed["scaler_load_payloads"].append(
                    copy.deepcopy(scaler_payload)
                )
                return real_scaler_load_state_dict(scaler_payload)

            monkeypatch.setattr(
                self.scaler,
                "load_state_dict",
                tracked_scaler_load_state_dict,
            )

        def resume(self, path: Path):
            fresh_rng = copy.deepcopy(_process_rng_state(self.train_generator))
            observed["fresh_rng_states"].append(fresh_rng)
            for source, saved in payload["rng_state"].items():
                assert not _nested_equal(fresh_rng[source], saved), (
                    f"checkpoint {source} RNG state did not differ from fresh state"
                )
            return super().resume(path)

    def finish_without_more_training(self, *, stop_after_epoch=None):
        _assert_full_checkpoint_restored(self, payload)
        verified_resumes.append(self.optimizer_step)
        return {
            "full_state_verified": True,
            "stop_after_epoch": stop_after_epoch,
        }

    monkeypatch.setattr(real_trainer, "fit", finish_without_more_training)
    monkeypatch.setattr(module, "MultiConfigSRMTrainer", ObservedResumeTrainer)

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

    assert (
        source_cfg.config_sha256
        == explicit_cfg.config_sha256
        == auto_cfg.config_sha256
    )
    assert explicit == {
        "full_state_verified": True,
        "stop_after_epoch": 1,
    }
    assert auto == explicit
    assert verified_resumes == [1, 1]
    assert len(observed["fresh_rng_states"]) == 2
    assert len(observed["scaler_load_payloads"]) == 2
    for scaler_payload in observed["scaler_load_payloads"]:
        assert _nested_equal(scaler_payload, payload["scaler"])


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
    real_trainer = module.MultiConfigSRMTrainer
    observed: dict[str, Any] = {"load_calls": []}

    class ObservedTrainer(real_trainer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            observed["trainer"] = self
            observed["before"] = _trainer_snapshot(self)

            def instrument(target, label: str) -> None:
                original = target.load_state_dict

                def tracked(*load_args, **load_kwargs):
                    observed["load_calls"].append(label)
                    return original(*load_args, **load_kwargs)

                monkeypatch.setattr(target, "load_state_dict", tracked)

            instrument(self.model, "model")
            instrument(self.ema.ema_model, "ema")
            instrument(self.optimizer, "optimizer")
            instrument(self.scheduler, "scheduler")
            instrument(self.scaler, "scaler")

    monkeypatch.setattr(module, "MultiConfigSRMTrainer", ObservedTrainer)

    with pytest.raises(CheckpointIdentityError, match="model_size"):
        run_same_frequency_multiscale_uno_training(
            _config(tmp_path, run_name="reject"),
            InvocationControls(resume=str(attention_fno_checkpoint), stop_after_epoch=1),
            torch.device("cpu"),
        )

    trainer = observed["trainer"]
    assert observed["load_calls"] == []
    assert _nested_equal(_trainer_snapshot(trainer), observed["before"])


def test_resume_none_refuses_existing_production_last_checkpoint(
    tiny_training,
) -> None:
    _module, tmp_path = tiny_training
    cfg = _config(tmp_path, run_name="production")
    cfg.run_dir.mkdir(parents=True)
    (cfg.run_dir / "last.pt").write_bytes(b"existing production checkpoint")

    with pytest.raises(
        SameFrequencyTrainerContractError,
        match="resume=none refuses an existing multiscale-UNO last.pt",
    ):
        run_same_frequency_multiscale_uno_training(
            cfg,
            InvocationControls(resume="none", stop_after_epoch=1),
            torch.device("cpu"),
        )

    assert not (cfg.run_dir / "config.json").exists()


@pytest.mark.parametrize("existing_artifact", ["config.json", "last.pt"])
def test_smoke_artifacts_are_isolated_from_production_run(
    tiny_training,
    existing_artifact: str,
) -> None:
    _module, tmp_path = tiny_training
    cfg = _config(tmp_path, run_name=f"isolation-{existing_artifact.split('.')[0]}")
    cfg.run_dir.mkdir(parents=True)
    sentinel = cfg.run_dir / existing_artifact
    sentinel.write_bytes(b"production sentinel")
    absent_name = "last.pt" if existing_artifact == "config.json" else "config.json"
    absent_production_artifact = (
        cfg.run_dir / absent_name
    )

    result = run_same_frequency_multiscale_uno_training(
        cfg,
        InvocationControls(resume="none", smoke_optimizer_steps=1),
        torch.device("cpu"),
    )

    smoke_dir = (cfg.run_root / "_smoke").resolve()
    assert Path(result["run_dir"]) == smoke_dir
    assert Path(result["checkpoint"]) == smoke_dir / "smoke.pt"
    assert (smoke_dir / "config.json").is_file()
    assert sentinel.read_bytes() == b"production sentinel"
    assert not absent_production_artifact.exists()
