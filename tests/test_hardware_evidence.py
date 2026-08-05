from __future__ import annotations

from pathlib import Path

import pytest
import torch


def _hardware_module():
    from training import hardware_evidence

    return hardware_evidence


def _context(module):
    return module.LargeHardwareGateContext(
        trigger_array="8x8",
        config_sha256_by_array={
            "8x8": "1" * 64,
            "16x16": "2" * 64,
            "32x32": "3" * 64,
        },
        manifest_sha256_by_array={
            "8x8": "4" * 64,
            "16x16": "5" * 64,
            "32x32": "6" * 64,
        },
        split_sha256="7" * 64,
        schema_sha256="8" * 64,
        archive_sha256="9" * 64,
        dataset_revision="a" * 40,
        radioflow_upstream_base="b" * 40,
        git_commit="c" * 40,
    )


def _hardware(module):
    return module.HardwareSnapshot(
        gpu_name="Synthetic 8GB GPU",
        gpu_total_memory_bytes=8 * 1024**3,
        peak_allocated_bytes=7 * 1024**3,
        torch_version="2.5.1+cu121",
        cuda_version="12.1",
    )


def test_large_cuda_oom_writes_global_hashable_gate_without_production_checkpoint(
    tmp_path: Path,
) -> None:
    module = _hardware_module()
    gate = tmp_path / "runs" / "_hardware" / "large_hardware_gate.json"

    def fail():
        raise torch.cuda.OutOfMemoryError("synthetic allocation failed")

    result = module.execute_with_large_oom_gate(
        fail,
        gate_path=gate,
        context=_context(module),
        hardware=_hardware(module),
    )

    assert result.blocked is True
    assert gate.is_file()
    assert len(result.sha256) == 64
    payload = module.validate_large_hardware_gate(gate, _context(module))
    assert payload["scope"] == ["8x8/large", "16x16/large", "32x32/large"]
    assert payload["parameter_count"] == 54_126_059
    assert payload["condition_shape"] == [1, 3, 256, 256]
    assert payload["target_shape"] == [1, 1, 256, 256]
    assert payload["activation_checkpointing"] is True
    assert not (tmp_path / "runs" / "8x8" / "large" / "last.pt").exists()


def test_non_oom_exception_propagates_without_hardware_gate(tmp_path: Path) -> None:
    module = _hardware_module()
    gate = tmp_path / "large_hardware_gate.json"

    with pytest.raises(RuntimeError, match="real bug"):
        module.execute_with_large_oom_gate(
            lambda: (_ for _ in ()).throw(RuntimeError("real bug")),
            gate_path=gate,
            context=_context(module),
            hardware=_hardware(module),
        )

    assert not gate.exists()


def test_successful_large_smoke_writes_no_gate(tmp_path: Path) -> None:
    module = _hardware_module()
    gate = tmp_path / "large_hardware_gate.json"

    result = module.execute_with_large_oom_gate(
        lambda: {"optimizer_steps": 1},
        gate_path=gate,
        context=_context(module),
        hardware=_hardware(module),
    )

    assert result.blocked is False
    assert result.value == {"optimizer_steps": 1}
    assert result.sha256 is None
    assert not gate.exists()

