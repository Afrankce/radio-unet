from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from data_loaders.multiconfig import multiconfig_collate
from training.checkpointing import CheckpointIdentity
from training.config import MultiConfigTrainConfig


class TinyDataset(Dataset):
    def __init__(self, count: int, split: str) -> None:
        self.count = count
        self.split = split

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        row = torch.linspace(0.1, 0.9, 16)
        target = row.repeat(16, 1).unsqueeze(0)
        condition = torch.stack(
            (
                torch.zeros(16, 16),
                torch.full((16, 16), index / max(self.count, 1)),
                torch.full((16, 16), 0.5),
            )
        )
        condition[0, 7, 7] = 1.0
        return {
            "condition": condition.float(),
            "target": target.float(),
            "valid_mask": torch.ones(1, 16, 16, dtype=torch.bool),
            "metadata": {
                "sample_key": f"u{index + 1}|8x8|beam00",
                "split": self.split,
                "scene_id": f"u{index + 1}",
                "array_name": "8x8",
                "array_rows": 8,
                "array_cols": 8,
                "frequency_hz": 6_700_000_000,
                "config_id": "synthetic",
                "beam_id": 0,
                "steering_deg": -28.0,
                "height_path": "height.npy",
                "beam_map_path": "beam.npy",
                "radiomap_path": "radio.npy",
                "tx_rc": [127, 127],
            },
        }


class TinyRadioFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.velocity = torch.nn.Conv2d(4, 1, kernel_size=1)
        self.cfg_drop_prob = 0.25

    def embed_model(self, condition: torch.Tensor):
        return [condition]

    def forward(
        self,
        image=None,
        x=None,
        pred_type="denoise",
        step=None,
        embedding=None,
    ):
        return self.velocity(torch.cat((image, x), dim=1))

    def forward_with_cfg(
        self,
        *,
        image,
        x,
        step,
        embedding,
        cfg_scale,
    ):
        return self.forward(image=image, x=x, step=step, embedding=embedding)


def _identity(model: torch.nn.Module, cfg: MultiConfigTrainConfig) -> CheckpointIdentity:
    return CheckpointIdentity(
        array_size="8x8",
        model_size="lite",
        condition_channels=3,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        manifest_sha256="1" * 64,
        split_sha256="2" * 64,
        schema_sha256="3" * 64,
        config_sha256=cfg.config_sha256,
        archive_sha256="4" * 64,
        dataset_revision="5" * 40,
        radioflow_upstream_base="6" * 40,
        git_commit="7" * 40,
        seed=42,
    )


def _trainer(tmp_path: Path):
    from training.multiconfig_trainer import MultiConfigSRMTrainer, seed_everything

    cfg = MultiConfigTrainConfig(
        array_size="8x8",
        model_size="lite",
        dataset_root=tmp_path / "dataset",
        manifest_dir=tmp_path / "manifests",
        run_root=tmp_path / "runs",
    )
    seed_everything(42)
    model = TinyRadioFlow().to("cpu")
    generator = torch.Generator(device="cpu").manual_seed(42)
    train_loader = DataLoader(
        TinyDataset(16, "train"),
        batch_size=2,
        shuffle=True,
        generator=generator,
        collate_fn=multiconfig_collate,
    )
    val_loader = DataLoader(
        TinyDataset(2, "val"),
        batch_size=1,
        shuffle=False,
        collate_fn=multiconfig_collate,
    )
    trainer = MultiConfigSRMTrainer(
        cfg,
        model,
        train_loader,
        val_loader,
        torch.device("cpu"),
        generator,
        _identity(model, cfg),
    )
    return cfg, trainer


def test_synthetic_train_validate_checkpoint_resume_and_continue(tmp_path: Path) -> None:
    cfg, first = _trainer(tmp_path)

    first_result = first.fit(stop_after_epoch=1)

    assert first_result["status"] == "paused"
    assert first.optimizer_step == 1
    assert first.completed_epochs == 1
    assert (cfg.run_dir / "last.pt").is_file()
    assert (cfg.run_dir / "best.pt").is_file()
    assert (cfg.run_dir / "metrics.csv").is_file()
    assert len(first.history) == 1
    assert first.history[0]["val_n_samples"] == 2

    _cfg, resumed = _trainer(tmp_path)
    state = resumed.resume(cfg.run_dir / "last.pt")
    assert state.next_epoch_index == 1
    assert resumed.optimizer_step == 1

    second_result = resumed.fit(stop_after_epoch=2)

    assert second_result["status"] == "paused"
    assert resumed.optimizer_step == 2
    assert resumed.completed_epochs == 2
    assert [row["epoch"] for row in resumed.history] == [1, 2]
    lines = (cfg.run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3


def test_trainer_rejects_model_parameters_on_the_wrong_device(tmp_path: Path) -> None:
    from training.multiconfig_trainer import MultiConfigSRMTrainer, TrainerContractError

    cfg, trainer = _trainer(tmp_path)

    try:
        with torch.device("meta"):
            wrong = TinyRadioFlow()
        identity = _identity(wrong, cfg)
        try:
            MultiConfigSRMTrainer(
                cfg,
                wrong,
                trainer.train_loader,
                trainer.val_loader,
                torch.device("cpu"),
                trainer.train_generator,
                identity,
            )
        except TrainerContractError:
            pass
        else:
            raise AssertionError("wrong-device model was accepted")
    finally:
        del trainer

