from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_same_frequency_attention_fno_server.sh"


@pytest.mark.skipif(os.name == "nt", reason="launcher behavior is verified on Linux server")
def test_attention_fno_server_launcher_maps_three_physical_gpus(
    tmp_path: Path,
) -> None:
    assert shutil.which("bash") is not None
    environment = dict(os.environ)
    environment.update(
        {
            "RADIOFLOW_ATTENTION_FNO_CODE_ROOT": str(REPO_ROOT),
            "RADIOFLOW_ATTENTION_FNO_DATASET_ROOT": str(tmp_path / "dataset"),
            "RADIOFLOW_ATTENTION_FNO_RESULT_ROOT": str(tmp_path / "results"),
        }
    )

    completed = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("TRAIN ")]

    assert len(lines) == 3
    expected = {"8x8": "0", "16x16": "1", "32x32": "2"}
    for line, (array_size, gpu) in zip(lines, expected.items()):
        assert f"ARRAY={array_size} " in line
        assert f"CUDA_VISIBLE_DEVICES={gpu} " in line
        assert "--device cuda:0" in line
        assert "--resume auto" in line
        assert f"manifest_samefreq_6.7ghz_{array_size}_0deg.jsonl" in line
        assert f"runs/{array_size}" in line.replace("\\", "/")


@pytest.mark.skipif(os.name == "nt", reason="launcher behavior is verified on Linux server")
def test_attention_fno_server_launcher_requires_explicit_mode() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "explicit mode" in completed.stderr

