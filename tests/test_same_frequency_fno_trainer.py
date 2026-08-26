from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

from data_loaders.multiconfig import multiconfig_collate
from model.fno import ConditionalFNO2d
from training.checkpointing import CHECKPOINT_KEYS
from training.config import InvocationControls
from training.same_frequency_config import SameFrequencyTrainConfig
from training.same_frequency_fno_config import (
    PAPER_FNO_MODEL_SIZE,
    PaperFNOTrainConfig,
)
from training.same_frequency_fno_trainer import (
    run_same_frequency_fno_training,
    write_or_validate_fno_run_config,
)


class _TinyDataset(Dataset):
    def __init__(self, count: int, split: str) -> None:
        self.count = count
        self.split = split

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        target = torch.linspace(0.1, 0.9, 8).repeat(8, 1).unsqueeze(0)
        condition = torch.stack(
            (
                torch.zeros(8, 8),
                torch.full((8, 8), index / max(self.count, 1)),
                torch.full((8, 8), 0.5),
            )
        )
        condition[0, 3, 3] = 1.0
        return {
            "condition": condition.float(),
            "target": target.float(),
            "valid_mask": torch.ones(1, 8, 8, dtype=torch.bool),
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


def _config(tmp_path: Path) -> PaperFNOTrainConfig:
    return PaperFNOTrainConfig(
        SameFrequencyTrainConfig(
            dataset_root=tmp_path / "dataset",
            manifest_path=tmp_path / "manifest.jsonl",
            height_stats_path=tmp_path / "height.json",
            run_root=tmp_path / "run",
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
        _TinyDataset(56, "train"),
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


def _tiny_fno(model_size: str) -> ConditionalFNO2d:
    assert model_size == PAPER_FNO_MODEL_SIZE
    return ConditionalFNO2d(
        width=4,
        modes1=2,
        modes2=2,
        padding=1,
    )


def test_fno_run_config_is_written_and_revalidated_exactly(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    controls = InvocationControls(resume="auto")

    first = write_or_validate_fno_run_config(cfg, controls)
    second = write_or_validate_fno_run_config(cfg, controls)

    assert first == second == cfg.run_dir / "config.json"
    record = json.loads(first.read_text(encoding="utf-8"))
    assert record["model_size"] == PAPER_FNO_MODEL_SIZE
    assert record["backbone"] == "paper_fno2d"
    assert record["cfg_candidates"] == [1.0]
    assert record["config_sha256"] == cfg.config_sha256


def test_one_step_fno_smoke_writes_strict_fresh_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import training.same_frequency_fno_trainer as module

    monkeypatch.setattr(module, "preflight_same_frequency", lambda _cfg: _context())
    monkeypatch.setattr(module, "build_same_frequency_loaders", _tiny_loaders)
    monkeypatch.setattr(module, "build_same_frequency_backbone", _tiny_fno)
    cfg = _config(tmp_path)

    result = run_same_frequency_fno_training(
        cfg,
        InvocationControls(resume="none", smoke_optimizer_steps=1),
        torch.device("cpu"),
    )

    checkpoint = Path(result["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert result["status"] == "smoke_complete"
    assert result["optimizer_steps"] == 1
    assert set(payload) == CHECKPOINT_KEYS
    assert payload["run_identity"]["model_size"] == PAPER_FNO_MODEL_SIZE
    assert payload["run_identity"]["config_sha256"] == cfg.config_sha256
    assert payload["trainer_state"]["optimizer_step"] == 1
    assert payload["optimizer"]["state"]
    assert payload["scheduler"]
    assert payload["rng_state"]
