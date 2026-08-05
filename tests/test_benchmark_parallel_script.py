from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_multiconfig_benchmark_parallel.ps1"


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_parallel_script_locks_roots_scale_and_concurrency() -> None:
    text = _text()

    assert "param(" in text
    assert "[ValidateSet(0.1, 1.0)][double]$TrainScale = 0.1" in text
    assert "[ValidateRange(1, 6)][int]$MaxConcurrent = 3" in text
    assert '"E:\\RadioFlow\\runs\\srm_6.7ghz_common8$RunSuffix"' in text
    assert '"E:\\RadioFlow\\results\\srm_6.7ghz_common8$RunSuffix"' in text
    assert '--train-scale", "$TrainScale"' in text
    assert "$env:RADIOFLOW_RUN_ROOT = $runRoot" in text
    assert '$env:MULTICONFIG_TRAIN_SCALE = "$TrainScale"' in text


def test_parallel_script_orders_phases_and_is_fail_closed() -> None:
    text = _text()

    for name in (
        "PHASE_TRAIN_PARALLEL",
        "PHASE_CFG_SELECTION_PARALLEL",
        "PHASE_TEST_ONCE_PARALLEL",
        "PHASE_SUMMARY",
    ):
        assert name in text
    assert '$ErrorActionPreference = "Stop"' in text
    assert 'throw "training job' in text
    assert '"select-cfg"' in text
    assert '"test"' in text
    assert '"summarize"' in text
    assert "$MaxConcurrent" in text


def test_parallel_script_passes_no_scientific_overrides() -> None:
    text = _text()
    forbidden = (
        "--resolution",
        "--batch-size",
        "--accumulation-steps",
        "--activation-checkpointing",
        "--cfg-scale",
        "--solver",
        "--steps",
        "--beam",
        "--frequency",
        "--seed",
        "--learning-rate",
        "--epochs",
    )

    assert all(option not in text for option in forbidden)