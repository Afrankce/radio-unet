from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_same_frequency_multiscale_uno_server.sh"


def _bash() -> str:
    if os.name == "nt":
        candidate = Path(r"C:\Program Files\Git\bin\bash.exe")
        if candidate.is_file():
            return str(candidate)
        pytest.skip("Git for Windows Bash is required to execute the launcher")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required to execute the launcher")
    return bash


def _bash_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    return subprocess.check_output(
        [_bash(), "-lc", 'cygpath -u "$1"', "--", str(path)],
        text=True,
    ).strip()


def _run(*arguments: str, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_bash(), _bash_path(SCRIPT), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _launcher_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    code_root = tmp_path / "code"
    dataset_root = tmp_path / "dataset"
    result_root = tmp_path / "results"
    env_file = tmp_path / "remote_env.sh"
    invocation_log = tmp_path / "python-invocations.log"
    bin_dir = tmp_path / "bin"
    code_root.mkdir()
    dataset_root.mkdir()
    bin_dir.mkdir()
    (code_root / "run_same_frequency_multiscale_uno.py").touch()
    env_file.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (bin_dir / "python").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >> \"${RADIOFLOW_TEST_INVOCATION_LOG:?}\"\n"
        "array_size=''\n"
        "for ((index = 1; index <= $#; index++)); do\n"
        "  if [[ \"${!index}\" == '--array-size' ]]; then\n"
        "    next=$((index + 1))\n"
        "    array_size=\"${!next}\"\n"
        "  fi\n"
        "done\n"
        "if [[ \"${RADIOFLOW_TEST_SMOKE_FAIL_ARRAY:-}\" == \"$array_size\" ]]; then\n"
        "  exit 9\n"
        "fi\n"
        "if [[ \"$*\" == *'--smoke-optimizer-steps 1'* ]]; then\n"
        "  sleep 0.2\n"
        "fi\n"
        "printf 'complete %s\\n' \"$array_size\" >> \"${RADIOFLOW_TEST_INVOCATION_LOG:?}\"\n",
        encoding="utf-8",
    )
    (bin_dir / "python").chmod(0o755)
    environment = dict(os.environ)
    if os.name == "nt":
        path = _bash_path(bin_dir) + ":/usr/bin:/bin"
    else:
        path = str(bin_dir) + os.pathsep + environment.get("PATH", "")
    environment.update(
        {
            "PATH": path,
            "RADIOFLOW_MULTISCALE_UNO_CODE_ROOT": _bash_path(code_root),
            "RADIOFLOW_MULTISCALE_UNO_DATASET_ROOT": _bash_path(dataset_root),
            "RADIOFLOW_MULTISCALE_UNO_RESULT_ROOT": _bash_path(result_root),
            "RADIOFLOW_MULTISCALE_UNO_ENV_FILE": _bash_path(env_file),
            "RADIOFLOW_TEST_INVOCATION_LOG": _bash_path(invocation_log),
        }
    )
    return environment, invocation_log


def test_dry_run_emits_three_isolated_default_training_commands() -> None:
    completed = _run("--dry-run")

    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.startswith("TRAIN ")]
    assert len(lines) == 3
    expected = (("8x8", "0"), ("16x16", "1"), ("32x32", "2"))
    for line, (array_size, gpu) in zip(lines, expected):
        assert f"ARRAY={array_size} CUDA_VISIBLE_DEVICES={gpu} " in line
        assert "run_same_frequency_multiscale_uno.py train" in line
        assert "--device cuda:0" in line
        assert "--resume auto" in line
        assert f"manifest_samefreq_6.7ghz_{array_size}_0deg.jsonl" in line
        assert f"multiscale_uno_samefreq_6.7ghz/runs/{array_size}" in line.replace("\\", "/")
    assert "/home/wys/radioflow_20260823/multiscale-uno-singlebeam" in lines[0]


def test_dry_run_exposes_root_overrides_in_the_emitted_commands(tmp_path: Path) -> None:
    environment, _ = _launcher_environment(tmp_path)

    completed = _run("--dry-run", environment=environment)

    assert completed.returncode == 0, completed.stderr
    output = completed.stdout.replace("\\", "/")
    assert _bash_path(tmp_path / "code") in output
    assert _bash_path(tmp_path / "dataset") in output
    assert _bash_path(tmp_path / "results") in output


@pytest.mark.parametrize("arguments", [(), ("--unknown",)])
def test_launcher_rejects_missing_or_unknown_mode_with_exit_two(
    arguments: tuple[str, ...],
) -> None:
    completed = _run(*arguments)

    assert completed.returncode == 2


@pytest.mark.parametrize(
    ("mode", "expected_command", "expected_extra"),
    [
        ("--preflight", "train", "--preflight-only"),
        ("--select-cfg", "select-cfg", "--results-root"),
        ("--test", "test", "--results-root"),
    ],
)
def test_lifecycle_modes_run_all_three_arrays(
    tmp_path: Path,
    mode: str,
    expected_command: str,
    expected_extra: str,
) -> None:
    environment, invocation_log = _launcher_environment(tmp_path)

    completed = _run(mode, environment=environment)

    assert completed.returncode == 0, completed.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    commands = [line for line in invocations if not line.startswith("complete ")]
    assert len(commands) == 3
    for array_size, command in zip(("8x8", "16x16", "32x32"), commands):
        assert f"run_same_frequency_multiscale_uno.py {expected_command} " in command
        assert f"--array-size {array_size}" in command
        assert expected_extra in command


def test_smoke_waits_for_every_array_and_propagates_a_failure(tmp_path: Path) -> None:
    environment, invocation_log = _launcher_environment(tmp_path)
    environment["RADIOFLOW_TEST_SMOKE_FAIL_ARRAY"] = "16x16"

    started = time.monotonic()
    completed = _run("--smoke", environment=environment)
    elapsed = time.monotonic() - started

    assert completed.returncode == 1
    assert elapsed >= 0.15
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    commands = [line for line in invocations if not line.startswith("complete ")]
    assert len(commands) == 3
    assert all("train" in command and "--smoke-optimizer-steps 1" in command for command in commands)
    assert {line.removeprefix("complete ") for line in invocations if line.startswith("complete ")} == {
        "8x8",
        "32x32",
    }


def test_train_refuses_a_live_pid_without_starting_a_new_process(tmp_path: Path) -> None:
    environment, invocation_log = _launcher_environment(tmp_path)
    result_root = tmp_path / "results"
    pid_dir = result_root / "pids"
    pid_dir.mkdir(parents=True)
    sleeper = subprocess.Popen(
        [_bash(), "-c", 'sleep 30 & child=$!; printf "%s\\n" "$child"; wait "$child"'],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert sleeper.stdout is not None
        sleep_pid = sleeper.stdout.readline().strip()
        assert sleep_pid.isdecimal()
        (pid_dir / "train_8x8.pid").write_text(f"{sleep_pid}\n", encoding="utf-8")

        completed = _run("--train", environment=environment)

        assert completed.returncode == 1
        assert "already live for 8x8" in completed.stderr
        assert not invocation_log.exists()
    finally:
        sleeper.terminate()
        sleeper.wait(timeout=5)
