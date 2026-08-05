from __future__ import annotations

import re
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_multiconfig_benchmark.ps1"


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_script_locks_interpreter_roots_arrays_and_models() -> None:
    text = _text()

    assert 'D:\\Anaconda3\\envs\\radioflow-win\\python.exe' in text
    assert 'E:\\datasets\\MultiConfigRadiomap' in text
    assert 'E:\\RadioFlow\\runs\\srm_6.7ghz_common8' in text
    assert 'E:\\RadioFlow\\results\\srm_6.7ghz_common8' in text
    assert '$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"' in text
    assert re.search(
        r'\$arrays\s*=\s*@\("8x8",\s*"16x16",\s*"32x32"\)', text
    )
    assert re.search(r'\$modelSizes\s*=\s*@\("lite",\s*"large"\)', text)


def test_script_orders_smokes_pilots_full_runs_and_evaluation() -> None:
    text = _text()
    phase_names = (
        "PHASE_LITE_SMOKE",
        "PHASE_LARGE_SMOKE",
        "PHASE_GPU_EVIDENCE",
        "PHASE_LITE_PILOTS",
        "PHASE_LARGE_PILOT",
        "PHASE_LITE_FULL",
        "PHASE_LARGE_FULL",
        "PHASE_CFG_SELECTION",
        "PHASE_TEST_ONCE",
        "PHASE_SUMMARY",
    )

    positions = [text.index(name) for name in phase_names]

    assert positions == sorted(positions)
    assert '--smoke-optimizer-steps", "1"' in text
    assert '--stop-after-epoch", "5"' in text
    assert '"--resume", "auto"' in text
    assert '"select-cfg"' in text
    assert '"test"' in text
    assert '"summarize"' in text


def test_script_is_fail_closed_and_large_gate_only_skips_large_work() -> None:
    text = _text()

    assert '$ErrorActionPreference = "Stop"' in text
    assert "if ($exitCode -ne 0)" in text
    assert '$largeBlocked = $true' in text
    assert 'large_hardware_gate.json' in text
    assert text.count('if (-not $largeBlocked)') >= 3
    assert 'if ($modelSize -eq "large" -and $largeBlocked)' in text
    assert "continue" in text
    assert "throw" in text


def test_script_passes_no_scientific_overrides() -> None:
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


def test_script_parameterizes_run_suffix_and_train_scale() -> None:
    text = _text()

    assert "param(" in text
    assert '[ValidateSet(0.1, 1.0)][double]$TrainScale = 1.0' in text
    assert '"E:\\RadioFlow\\runs\\srm_6.7ghz_common8$RunSuffix"' in text
    assert '"E:\\RadioFlow\\results\\srm_6.7ghz_common8$RunSuffix"' in text
    assert '--train-scale", "$TrainScale"' in text
    assert '$env:RADIOFLOW_RUN_ROOT = $runRoot' in text
    assert '$env:MULTICONFIG_TRAIN_SCALE = "$TrainScale"' in text
