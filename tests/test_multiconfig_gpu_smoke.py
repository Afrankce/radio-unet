from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch

from experiments.multiconfig_manifest import canonical_json_bytes
from training.checkpointing import (
    CHECKPOINT_KEYS,
    CheckpointIdentity,
    TrainerState,
    load_ema_for_evaluation,
)
from training.config import MultiConfigTrainConfig
from training.hardware_evidence import LARGE_SCOPE, validate_large_hardware_gate
from training.model_factory import EXPECTED_PARAMETER_COUNTS, build_locked_radioflow
from training.multiconfig_trainer import (
    _large_gate_context,
    build_checkpoint_identity,
    preflight_benchmark,
)


pytestmark = [pytest.mark.gpu, pytest.mark.dataset]

DATASET_ROOT = Path(
    os.environ.get("MULTICONFIG_ROOT", r"E:\datasets\MultiConfigRadiomap")
)
MANIFEST_DIR = Path(
    os.environ.get("MULTICONFIG_MANIFEST_DIR", str(DATASET_ROOT / "manifests"))
)
RUN_ROOT = Path(
    os.environ.get(
        "RADIOFLOW_RUN_ROOT", r"E:\RadioFlow\runs\srm_6.7ghz_common8"
    )
)


def _cfg(model_size: str) -> MultiConfigTrainConfig:
    return MultiConfigTrainConfig(
        array_size="8x8",
        model_size=model_size,
        dataset_root=DATASET_ROOT,
        manifest_dir=MANIFEST_DIR,
        run_root=RUN_ROOT,
    )


def _canonical_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    assert isinstance(payload, dict)
    assert raw == canonical_json_bytes(payload)
    return payload


def _checkpoint(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(payload, Mapping)
    assert set(payload) == CHECKPOINT_KEYS
    assert payload["schema_version"] == 1
    return payload


def _assert_smoke_receipt(model_size: str, expected_micro_batches: int) -> None:
    cfg = _cfg(model_size)
    smoke_dir = RUN_ROOT / "_smoke" / "8x8" / model_size
    stored_cfg = MultiConfigTrainConfig.from_json(
        (smoke_dir / "config.json").read_text(encoding="utf-8")
    )
    assert stored_cfg.run_root.resolve() == (RUN_ROOT / "_smoke").resolve()
    assert stored_cfg.config_sha256 == cfg.config_sha256
    runtime = _canonical_json(smoke_dir / "training_runtime.json")
    assert runtime["status"] == "smoke_complete"
    assert runtime["smoke_optimizer_steps"] == 1
    assert runtime["optimizer_step"] == 1
    assert runtime["micro_batches_seen"] == expected_micro_batches
    assert runtime["samples_seen"] == 16
    assert runtime["amp_requested"] is True
    assert runtime["amp_dtype"] == "float16"
    assert runtime["autocast_enabled"] is True
    assert runtime["scaler_enabled"] is True
    assert isinstance(runtime["peak_training_allocated_bytes"], int)
    assert runtime["peak_training_allocated_bytes"] > 0
    assert runtime["smoke_loss"] >= 0.0

    checkpoint_path = smoke_dir / "smoke.pt"
    payload = _checkpoint(checkpoint_path)
    identity = CheckpointIdentity.from_dict(payload["run_identity"])
    assert identity.array_size == "8x8"
    assert identity.model_size == model_size
    assert identity.parameter_count == EXPECTED_PARAMETER_COUNTS[model_size]
    state = TrainerState.from_dict(payload["trainer_state"])
    assert state.optimizer_step == 1
    assert state.micro_batches_seen == expected_micro_batches
    assert state.samples_seen == 16

    context = preflight_benchmark(cfg)
    model = build_locked_radioflow(model_size)
    expected_identity = build_checkpoint_identity(cfg, context, model)
    assert identity == expected_identity
    reloaded = load_ema_for_evaluation(
        checkpoint_path,
        model=model,
        expected_identity=expected_identity,
    )
    assert reloaded.optimizer_step == 1


def test_cuda_device_and_lite_complete_optimizer_step_smoke() -> None:
    assert torch.cuda.is_available(), "the approved benchmark requires CUDA"
    properties = torch.cuda.get_device_properties(0)
    assert properties.total_memory > 0

    _assert_smoke_receipt("lite", expected_micro_batches=8)


def test_large_smoke_is_complete_or_has_one_valid_global_oom_gate() -> None:
    gate_path = RUN_ROOT / "_hardware" / "large_hardware_gate.json"
    if gate_path.is_file():
        cfg = _cfg("large")
        context = preflight_benchmark(cfg)
        payload = validate_large_hardware_gate(
            gate_path,
            _large_gate_context(cfg, context),
        )
        assert payload["status"] == "hardware_blocked"
        assert payload["reason"] == "cuda_out_of_memory"
        assert payload["scope"] == list(LARGE_SCOPE)
        assert payload["parameter_count"] == EXPECTED_PARAMETER_COUNTS["large"]
        assert payload["resolution"] == 256
        assert payload["micro_batch_size"] == 1
        assert payload["accumulation_steps"] == 16
        assert payload["activation_checkpointing"] is True
        return

    _assert_smoke_receipt("large", expected_micro_batches=16)
