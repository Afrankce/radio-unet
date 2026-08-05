from __future__ import annotations

from pathlib import Path

import pytest
import torch


class EvalTrackingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3))
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return super().eval()


def test_cuda_runtime_uses_fixed_warmups_measurements_and_sync_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation.runtime_benchmark import benchmark_generation

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    clock = {"value": 0.0, "calls": 0, "sync": 0, "reset": 0, "empty": 0}
    measured_durations = [float(index) for index in range(1, 101)]

    class FakeEvent:
        def __init__(self, enable_timing: bool) -> None:
            assert enable_timing is True
            self.value = None

        def record(self) -> None:
            self.value = clock["value"]

        def elapsed_time(self, other: "FakeEvent") -> float:
            return float(other.value - self.value)

    def generate() -> torch.Tensor:
        call = clock["calls"]
        if call >= 20:
            clock["value"] += measured_durations[call - 20]
        clock["calls"] += 1
        return torch.zeros(1)

    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda _device=None: clock.__setitem__("sync", clock["sync"] + 1),
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda _device=None: clock.__setitem__("reset", clock["reset"] + 1),
    )
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: clock.__setitem__("empty", clock["empty"] + 1),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device=None: 123456)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device=None: "Mock GPU")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device=None: type("Props", (), {"total_memory": 8 * 1024**3})(),
    )
    model = EvalTrackingModel()

    result = benchmark_generation(
        generate=generate,
        model=model,
        device=torch.device("cuda:0"),
        checkpoint_path=checkpoint,
    )

    assert clock["calls"] == 120
    assert clock["empty"] == 1
    assert clock["reset"] == 2
    assert clock["sync"] == 101
    assert model.eval_calls == 1
    assert result["warmup_calls"] == 20
    assert result["measured_calls"] == 100
    assert result["latency_ms_p50"] == pytest.approx(50.5)
    assert result["latency_ms_p95"] == pytest.approx(95.05)
    assert result["latency_ms_mean"] == pytest.approx(50.5)
    assert result["max_memory_allocated_bytes"] == 123456
    assert result["gpu_name"] == "Mock GPU"
    assert result["gpu_total_memory_bytes"] == 8 * 1024**3
    assert result["parameter_count"] == 3
    assert result["checkpoint_size_bytes"] == len(b"checkpoint-bytes")


def test_runtime_rejects_missing_checkpoint(tmp_path: Path) -> None:
    from evaluation.runtime_benchmark import RuntimeBenchmarkError, benchmark_generation

    with pytest.raises(RuntimeBenchmarkError, match="checkpoint does not exist"):
        benchmark_generation(
            generate=lambda: torch.zeros(1),
            model=EvalTrackingModel(),
            device=torch.device("cpu"),
            checkpoint_path=tmp_path / "missing.pt",
        )

