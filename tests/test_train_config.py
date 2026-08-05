from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


def _config_module():
    from training import config

    return config


@pytest.mark.parametrize(
    ("model_size", "micro_batch", "accumulation", "checkpointing"),
    [("lite", 2, 8, False), ("large", 1, 16, True)],
)
def test_locked_training_config_derives_effective_batch_and_steps(
    tmp_path: Path,
    model_size: str,
    micro_batch: int,
    accumulation: int,
    checkpointing: bool,
) -> None:
    module = _config_module()
    cfg = module.MultiConfigTrainConfig(
        array_size="8x8",
        model_size=model_size,
        dataset_root=tmp_path / "dataset",
        manifest_dir=tmp_path / "manifests",
        run_root=tmp_path / "runs",
    )

    assert cfg.micro_batch_size == micro_batch
    assert cfg.accumulation_steps == accumulation
    assert cfg.effective_batch_size == 16
    assert cfg.optimizer_steps_per_epoch == 280
    assert cfg.planned_optimizer_steps == 56_000
    assert cfg.warmup_steps == 5_600
    assert cfg.use_amp is True
    assert cfg.activation_checkpointing is checkpointing


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 7),
        ("learning_rate", 2e-3),
        ("max_epochs", 5),
        ("resolution", 128),
        ("use_amp", False),
        ("amp_dtype", "bfloat16"),
    ],
)
def test_scientific_training_controls_are_immutable(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    module = _config_module()
    arguments = {
        "array_size": "8x8",
        "model_size": "lite",
        "dataset_root": tmp_path,
        "manifest_dir": tmp_path,
        "run_root": tmp_path,
        field: value,
    }

    with pytest.raises(module.TrainConfigError, match=field):
        module.MultiConfigTrainConfig(**arguments)


def test_run_config_round_trip_rejects_unknown_or_changed_fields(tmp_path: Path) -> None:
    module = _config_module()
    cfg = module.MultiConfigTrainConfig(
        array_size="16x16",
        model_size="large",
        dataset_root=tmp_path / "dataset",
        manifest_dir=tmp_path / "manifests",
        run_root=tmp_path / "runs",
    )
    controls = module.InvocationControls(
        resume="auto",
        stop_after_epoch=5,
        smoke_optimizer_steps=None,
    )
    record = cfg.to_record(controls)

    restored = module.MultiConfigTrainConfig.from_json(json.dumps(record))

    assert restored == cfg
    assert record["config_sha256"] == cfg.config_sha256
    with_unknown = dict(record, surprise=True)
    with pytest.raises(module.TrainConfigError, match="unknown"):
        module.MultiConfigTrainConfig.from_json(json.dumps(with_unknown))
    changed = dict(record, resolution=128)
    with pytest.raises(module.TrainConfigError, match="resolution"):
        module.MultiConfigTrainConfig.from_json(json.dumps(changed))


def test_invocation_controls_do_not_change_scientific_hash(tmp_path: Path) -> None:
    module = _config_module()
    cfg = module.MultiConfigTrainConfig(
        array_size="32x32",
        model_size="lite",
        dataset_root=tmp_path,
        manifest_dir=tmp_path,
        run_root=tmp_path,
    )

    paused = cfg.to_record(
        module.InvocationControls(
            resume="none",
            stop_after_epoch=5,
            smoke_optimizer_steps=None,
        )
    )
    resumed = cfg.to_record(
        module.InvocationControls(
            resume="auto",
            stop_after_epoch=None,
            smoke_optimizer_steps=None,
        )
    )

    assert paused["config_sha256"] == resumed["config_sha256"]
    assert paused["invocation"] != resumed["invocation"]


def test_cpu_runtime_records_amp_request_but_disables_scaler(tmp_path: Path) -> None:
    module = _config_module()
    cfg = module.MultiConfigTrainConfig(
        array_size="8x8",
        model_size="lite",
        dataset_root=tmp_path,
        manifest_dir=tmp_path,
        run_root=tmp_path,
    )

    runtime = cfg.precision_runtime(torch.device("cpu"))

    assert runtime == {
        "amp_requested": True,
        "amp_dtype": "float16",
        "autocast_enabled": False,
        "scaler_enabled": False,
    }


def test_train_scale_0p1_derives_448_samples_and_28_steps(tmp_path: Path) -> None:
    module = _config_module()
    cfg = module.MultiConfigTrainConfig(
        array_size="8x8",
        model_size="lite",
        dataset_root=tmp_path / "dataset",
        manifest_dir=tmp_path / "manifests",
        run_root=tmp_path / "runs",
        train_scale=0.1,
    )

    assert cfg.train_samples == 448
    assert cfg.optimizer_steps_per_epoch == 28
    assert cfg.planned_optimizer_steps == 5_600
    assert cfg.warmup_steps == 560
    payload = cfg.scientific_payload()
    assert payload["train_scale"] == 0.1
    assert payload["train_samples"] == 448
    assert payload["full_train_samples"] == 4480
    assert payload["train_scene_count"] == 560
    assert payload["train_subsample_rule"] == "sorted_first_n_scenes"

    with pytest.raises(module.TrainConfigError, match="train_scale"):
        module.MultiConfigTrainConfig(
            array_size="8x8",
            model_size="lite",
            dataset_root=tmp_path,
            manifest_dir=tmp_path,
            run_root=tmp_path,
            train_scale=0.2,
        )
