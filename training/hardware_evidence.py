from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, TypeVar

import torch

from experiments.multiconfig_manifest import canonical_json_bytes


ARRAY_NAMES = ("8x8", "16x16", "32x32")
LARGE_SCOPE = tuple(f"{array}/large" for array in ARRAY_NAMES)
T = TypeVar("T")


class HardwareGateError(RuntimeError):
    """Global Large hardware evidence is incomplete, inconsistent, or mutable."""


@dataclass(frozen=True)
class HardwareSnapshot:
    gpu_name: str
    gpu_total_memory_bytes: int
    peak_allocated_bytes: int | None
    torch_version: str
    cuda_version: str | None

    def validate(self) -> None:
        if not self.gpu_name or self.gpu_total_memory_bytes <= 0:
            raise HardwareGateError("hardware snapshot requires a named positive-memory GPU")
        if self.peak_allocated_bytes is not None and self.peak_allocated_bytes < 0:
            raise HardwareGateError("peak_allocated_bytes must be non-negative")
        if not self.torch_version:
            raise HardwareGateError("torch_version must be non-empty")


@dataclass(frozen=True)
class LargeHardwareGateContext:
    trigger_array: str
    config_sha256_by_array: Mapping[str, str]
    manifest_sha256_by_array: Mapping[str, str]
    split_sha256: str
    schema_sha256: str
    archive_sha256: str
    dataset_revision: str
    radioflow_upstream_base: str
    git_commit: str

    def validate(self) -> None:
        if self.trigger_array not in ARRAY_NAMES:
            raise HardwareGateError(f"invalid trigger_array: {self.trigger_array!r}")
        for label, values in (
            ("config_sha256_by_array", self.config_sha256_by_array),
            ("manifest_sha256_by_array", self.manifest_sha256_by_array),
        ):
            if set(values) != set(ARRAY_NAMES):
                raise HardwareGateError(f"{label} must cover all three arrays")
            for array_name, digest in values.items():
                _require_digest(f"{label}.{array_name}", digest, 64)
        for field in ("split_sha256", "schema_sha256", "archive_sha256"):
            _require_digest(field, getattr(self, field), 64)
        for field in ("dataset_revision", "radioflow_upstream_base", "git_commit"):
            _require_digest(field, getattr(self, field), 40)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "trigger_array": self.trigger_array,
            "config_sha256_by_array": dict(self.config_sha256_by_array),
            "manifest_sha256_by_array": dict(self.manifest_sha256_by_array),
            "split_sha256": self.split_sha256,
            "schema_sha256": self.schema_sha256,
            "archive_sha256": self.archive_sha256,
            "dataset_revision": self.dataset_revision,
            "radioflow_upstream_base": self.radioflow_upstream_base,
            "git_commit": self.git_commit,
        }


@dataclass(frozen=True)
class LargeGateResult(Generic[T]):
    blocked: bool
    value: T | None
    sha256: str | None
    gate_path: Path | None


def _require_digest(label: str, value: str, length: int) -> None:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HardwareGateError(f"{label} must be a lowercase {length}-hex digest")


def collect_hardware_snapshot(device: torch.device) -> HardwareSnapshot:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise HardwareGateError("Large OOM evidence requires an available CUDA device")
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return HardwareSnapshot(
        gpu_name=properties.name,
        gpu_total_memory_bytes=int(properties.total_memory),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )


def _gate_payload(
    error: torch.cuda.OutOfMemoryError,
    context: LargeHardwareGateContext,
    hardware: HardwareSnapshot,
) -> dict[str, Any]:
    context.validate()
    hardware.validate()
    return {
        "schema_version": 1,
        "status": "hardware_blocked",
        "reason": "cuda_out_of_memory",
        "scope": list(LARGE_SCOPE),
        "trigger_array": context.trigger_array,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "model_size": "large",
        "parameter_count": 54_126_059,
        "condition_shape": [1, 3, 256, 256],
        "target_shape": [1, 1, 256, 256],
        "resolution": 256,
        "amp_requested": True,
        "amp_dtype": "float16",
        "micro_batch_size": 1,
        "accumulation_steps": 16,
        "effective_batch_size": 16,
        "activation_checkpointing": True,
        "shape_scope_rationale": (
            "Array identity changes condition values, not condition/output tensor "
            "shape or the locked Large RadioFlow network."
        ),
        "gpu_name": hardware.gpu_name,
        "gpu_total_memory_bytes": hardware.gpu_total_memory_bytes,
        "peak_allocated_bytes": hardware.peak_allocated_bytes,
        "torch_version": hardware.torch_version,
        "cuda_version": hardware.cuda_version,
        "identities": context.to_dict(),
    }


def _write_immutable_gate(path: Path, payload: Mapping[str, Any]) -> str:
    path = Path(path)
    expected = canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != expected:
            raise HardwareGateError(
                f"existing Large hardware gate differs and cannot be overwritten: {path}"
            )
        return hashlib.sha256(expected).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as output:
            output.write(expected)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(expected).hexdigest()


def validate_large_hardware_gate(
    path: Path,
    expected_context: LargeHardwareGateContext,
) -> dict[str, Any]:
    path = Path(path)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HardwareGateError(f"cannot read Large hardware gate {path}: {error}") from error
    if not isinstance(payload, dict):
        raise HardwareGateError("Large hardware gate root must be an object")
    if raw != canonical_json_bytes(payload):
        raise HardwareGateError("Large hardware gate is not canonical JSON")
    required = {
        "schema_version",
        "status",
        "reason",
        "scope",
        "trigger_array",
        "exception_type",
        "exception_message",
        "model_size",
        "parameter_count",
        "condition_shape",
        "target_shape",
        "resolution",
        "amp_requested",
        "amp_dtype",
        "micro_batch_size",
        "accumulation_steps",
        "effective_batch_size",
        "activation_checkpointing",
        "shape_scope_rationale",
        "gpu_name",
        "gpu_total_memory_bytes",
        "peak_allocated_bytes",
        "torch_version",
        "cuda_version",
        "identities",
    }
    if set(payload) != required:
        raise HardwareGateError("Large hardware gate keys mismatch")
    fixed = {
        "schema_version": 1,
        "status": "hardware_blocked",
        "reason": "cuda_out_of_memory",
        "scope": list(LARGE_SCOPE),
        "model_size": "large",
        "parameter_count": 54_126_059,
        "condition_shape": [1, 3, 256, 256],
        "target_shape": [1, 1, 256, 256],
        "resolution": 256,
        "amp_requested": True,
        "amp_dtype": "float16",
        "micro_batch_size": 1,
        "accumulation_steps": 16,
        "effective_batch_size": 16,
        "activation_checkpointing": True,
    }
    for key, expected in fixed.items():
        if payload[key] != expected:
            raise HardwareGateError(f"Large hardware gate {key} mismatch")
    expected_identity = expected_context.to_dict()
    if payload["identities"] != expected_identity:
        raise HardwareGateError("Large hardware gate identity mismatch")
    if payload["trigger_array"] != expected_context.trigger_array:
        raise HardwareGateError("Large hardware gate trigger_array mismatch")
    if not payload["exception_message"] or payload["exception_type"] != "OutOfMemoryError":
        raise HardwareGateError("Large hardware gate has invalid exception evidence")
    HardwareSnapshot(
        gpu_name=payload["gpu_name"],
        gpu_total_memory_bytes=payload["gpu_total_memory_bytes"],
        peak_allocated_bytes=payload["peak_allocated_bytes"],
        torch_version=payload["torch_version"],
        cuda_version=payload["cuda_version"],
    ).validate()
    return payload


def execute_with_large_oom_gate(
    action: Callable[[], T],
    *,
    gate_path: Path,
    context: LargeHardwareGateContext,
    hardware: HardwareSnapshot,
) -> LargeGateResult[T]:
    try:
        value = action()
    except torch.cuda.OutOfMemoryError as error:
        payload = _gate_payload(error, context, hardware)
        digest = _write_immutable_gate(gate_path, payload)
        return LargeGateResult(
            blocked=True,
            value=None,
            sha256=digest,
            gate_path=Path(gate_path).resolve(),
        )
    return LargeGateResult(blocked=False, value=value, sha256=None, gate_path=None)

