from __future__ import annotations

import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


WARMUP_CALLS = 20
MEASURED_CALLS = 100


class RuntimeBenchmarkError(RuntimeError):
    """The locked generation latency benchmark cannot be measured faithfully."""


def _summary(
    durations_ms: list[float],
    *,
    model: torch.nn.Module,
    device: torch.device,
    checkpoint_path: Path,
    max_memory_allocated: int | None,
) -> dict[str, Any]:
    if len(durations_ms) != MEASURED_CALLS:
        raise RuntimeBenchmarkError(
            f"expected {MEASURED_CALLS} measurements, got {len(durations_ms)}"
        )
    if any(not math.isfinite(value) or value < 0.0 for value in durations_ms):
        raise RuntimeBenchmarkError("latency measurements must be finite and non-negative")
    values = np.asarray(durations_ms, dtype=np.float64)
    gpu_name = None
    gpu_total_memory = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    return {
        "schema_version": 1,
        "device": str(device),
        "warmup_calls": WARMUP_CALLS,
        "measured_calls": MEASURED_CALLS,
        "latency_ms_p50": float(np.percentile(values, 50)),
        "latency_ms_p95": float(np.percentile(values, 95)),
        "latency_ms_mean": float(values.mean()),
        "latency_ms_std": float(values.std(ddof=0)),
        "max_memory_allocated_bytes": max_memory_allocated,
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": gpu_total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
    }


def benchmark_generation(
    *,
    generate: Callable[[], torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Measure exactly 20 warmups and 100 batch-one generation calls."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise RuntimeBenchmarkError(f"checkpoint does not exist: {checkpoint_path}")
    model.eval()
    durations_ms: list[float] = []
    max_memory: int | None = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(WARMUP_CALLS):
            generate()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(MEASURED_CALLS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            generate()
            end.record()
            torch.cuda.synchronize(device)
            durations_ms.append(float(start.elapsed_time(end)))
        max_memory = int(torch.cuda.max_memory_allocated(device))
    elif device.type == "cpu":
        for _ in range(WARMUP_CALLS):
            generate()
        for _ in range(MEASURED_CALLS):
            start_time = time.perf_counter()
            generate()
            durations_ms.append((time.perf_counter() - start_time) * 1000.0)
    else:
        raise RuntimeBenchmarkError(f"unsupported benchmark device: {device}")
    return _summary(
        durations_ms,
        model=model,
        device=device,
        checkpoint_path=checkpoint_path,
        max_memory_allocated=max_memory,
    )

