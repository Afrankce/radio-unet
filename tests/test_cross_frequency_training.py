from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from training.cross_frequency_config import (
    CrossFrequencyTrainConfig,
    CrossFrequencyTrainConfigError,
)
from training.cross_frequency_trainer import (
    CrossFrequencyContext,
    build_cross_frequency_loaders,
    preflight_cross_frequency,
    write_or_validate_cross_frequency_run_config,
)
from training.config import InvocationControls
from experiments.provenance import DATASET_REVISION


class TinyDataset(Dataset):
    def __init__(self, count: int, split: str) -> None:
        self.count = count
        self.split = split

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        return {
            "condition": torch.zeros(3, 256, 256),
            "target": torch.zeros(1, 256, 256),
            "valid_mask": torch.ones(1, 256, 256, dtype=torch.bool),
            "metadata": {
                "scene_id": f"u{index + 1}",
                "steering_deg": 0.0,
                "frequency_hz": 4_900_000_000,
            },
        }


def _cfg(tmp_path: Path, **overrides) -> CrossFrequencyTrainConfig:
    values = {
        "dataset_root": tmp_path / "dataset",
        "manifest_path": tmp_path / "manifests" / "manifest_cross_frequency_8x8.jsonl",
        "height_stats_path": tmp_path / "manifests" / "height_stats_train.json",
        "run_root": tmp_path / "runs",
    }
    values.update(overrides)
    return CrossFrequencyTrainConfig(**values)


def test_cross_frequency_config_locks_scientific_protocol(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    assert cfg.array_size == "8x8"
    assert cfg.model_size == "lite"
    assert cfg.condition_channels == 3
    assert cfg.train_frequency_hz == 4_900_000_000
    assert cfg.val_frequency_hz == 4_900_000_000
    assert cfg.test_frequency_hz == 6_700_000_000
    assert cfg.train_samples == 560
    assert cfg.val_samples == 80
    assert cfg.test_samples == 160
    assert cfg.max_epochs == 1000
    assert cfg.optimizer_steps_per_epoch == 10
    assert cfg.effective_batch_size == 56
    assert cfg.config_sha256 == cfg.config_sha256

    with pytest.raises(CrossFrequencyTrainConfigError):
        _cfg(tmp_path, array_size="16x16")
    with pytest.raises(CrossFrequencyTrainConfigError):
        _cfg(tmp_path, train_frequency_hz=6_700_000_000)
    with pytest.raises(CrossFrequencyTrainConfigError):
        _cfg(tmp_path, val_samples=640)
    with pytest.raises(CrossFrequencyTrainConfigError):
        _cfg(tmp_path, train_scale=0.1)


def test_cross_frequency_config_round_trips_canonically(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    record = cfg.to_record(InvocationControls(resume="none"))
    restored = CrossFrequencyTrainConfig.from_json(
        json.dumps(record, sort_keys=True, separators=(",", ":"))
    )

    assert restored == cfg
    changed = dict(record)
    changed["test_frequency_hz"] = 4_900_000_000
    with pytest.raises(CrossFrequencyTrainConfigError):
        CrossFrequencyTrainConfig.from_json(json.dumps(changed))


def test_cross_frequency_loader_counts_and_optimizer_steps(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    context = CrossFrequencyContext(
        train_dataset=TinyDataset(560, "train"),
        val_dataset=TinyDataset(80, "val"),
        test_dataset=TinyDataset(160, "test"),
        manifest_path=cfg.manifest_path,
        split_path=tmp_path / "split.json",
        schema_path=tmp_path / "schema.json",
        manifest_sha256="1" * 64,
        split_sha256="2" * 64,
        schema_sha256="3" * 64,
        archive_sha256="4" * 64,
        dataset_revision="5" * 40,
        git_commit="6" * 40,
        height_max=1.0,
    )

    train_loader, val_loader, generator = build_cross_frequency_loaders(cfg, context)

    assert len(train_loader) == 280
    assert len(val_loader) == 80
    assert cfg.optimizer_steps_per_epoch == 10
    assert generator.initial_seed() == 42


def test_cross_frequency_preflight_builds_three_fixed_splits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.manifest_path.parent.mkdir(parents=True)
    cfg.manifest_path.write_text("manifest\n", encoding="utf-8")
    cfg.height_stats_path.write_text("stats\n", encoding="utf-8")
    split_path = cfg.manifest_path.parent / "scene_split_seed42.json"
    split_path.write_text("{}", encoding="utf-8")
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}", encoding="utf-8")

    @dataclass(frozen=True)
    class Checkout:
        head_commit: str = "7" * 40

    class FakeSchema:
        raw = {
            "source_metadata": {
                "height": {"shape": [256, 256], "dtype": "float32"},
                "beam_map": {"shape": [128, 128], "dtype": "float64"},
                "radiomap": {"shape": [128, 128], "dtype": "float32"},
            }
        }
        identities = {
            "archive_sha256": "8" * 64,
            "dataset_revision": DATASET_REVISION,
        }

    class FakeDataset(TinyDataset):
        def __init__(self, **kwargs):
            super().__init__(
                {"train": 560, "val": 80, "test": 160}[kwargs["split"]],
                kwargs["split"],
            )

    monkeypatch.setattr("training.cross_frequency_trainer.SCHEMA_PATH", schema_path)
    monkeypatch.setattr(
        "training.cross_frequency_trainer.assert_radioflow_checkout",
        lambda _root: Checkout(),
    )
    monkeypatch.setattr(
        "training.cross_frequency_trainer.load_schema_lock",
        lambda _path: FakeSchema(),
    )
    monkeypatch.setattr(
        "training.cross_frequency_trainer.load_cross_frequency_height_max",
        lambda _path, split_path=None: 10.0,
    )
    monkeypatch.setattr(
        "training.cross_frequency_trainer.CrossFrequencyRadiomapDataset",
        FakeDataset,
    )
    monkeypatch.setattr(
        "training.cross_frequency_trainer.load_manifest_jsonl",
        lambda _path: tuple(range(800)),
    )

    context = preflight_cross_frequency(cfg)

    assert len(context.train_dataset) == 560
    assert len(context.val_dataset) == 80
    assert len(context.test_dataset) == 160
    assert context.archive_sha256 == "8" * 64
    assert context.dataset_revision == DATASET_REVISION
    assert context.git_commit == "7" * 40


def test_cross_frequency_run_config_is_immutable(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    controls = InvocationControls(resume="none")
    path = write_or_validate_cross_frequency_run_config(cfg, controls)
    assert path.is_file()
    assert write_or_validate_cross_frequency_run_config(cfg, controls) == path

    with pytest.raises(CrossFrequencyTrainConfigError):
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '"test_frequency_hz":6700000000',
                '"test_frequency_hz":6700000001',
            ),
            encoding="utf-8",
        )
        write_or_validate_cross_frequency_run_config(cfg, controls)
