from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_same_frequency_multiscale_uno_server.sh"
ARRAYS = ("8x8", "16x16", "32x32")


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


def _launcher_command(*arguments: str) -> list[str]:
    return [_bash(), _bash_path(SCRIPT), *arguments]


def _run(
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _launcher_command(*arguments),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )


def _launcher_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    code_root = tmp_path / "code"
    dataset_root = tmp_path / "dataset"
    result_root = tmp_path / "results"
    proc_root = tmp_path / "proc"
    env_file = tmp_path / "remote_env.sh"
    invocation_log = tmp_path / "python-invocations.log"
    bin_dir = tmp_path / "bin"
    code_root.mkdir()
    dataset_root.mkdir()
    proc_root.mkdir()
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
        "  if [[ \"$array_size\" == \"${RADIOFLOW_TEST_SLOW_SMOKE_ARRAY:-}\" ]]; then\n"
        "    sleep \"${RADIOFLOW_TEST_SLOW_SMOKE_SECONDS:-0.4}\"\n"
        "    : > \"${RADIOFLOW_TEST_SMOKE_SENTINEL:?}\"\n"
        "  else\n"
        "    sleep 0.05\n"
        "  fi\n"
        "fi\n"
        "if [[ \"${3:-}\" == 'train' && \"$*\" != *'--preflight-only'* "
        "&& \"$*\" != *'--smoke-optimizer-steps'* ]]; then\n"
        "  sleep \"${RADIOFLOW_TEST_PROC_WRITE_DELAY:-0}\"\n"
        "  process_dir=\"${RADIOFLOW_MULTISCALE_UNO_TEST_PROC_ROOT:?}/$$\"\n"
        "  mkdir -p \"$process_dir\"\n"
        "  fields=(S)\n"
        "  for _field in {4..21}; do fields+=(0); done\n"
        "  fields+=(\"${RADIOFLOW_TEST_PROCESS_START_TICKS:-424242}\")\n"
        "  printf '%s (fake-python) %s\\n' \"$$\" \"${fields[*]}\" > \"$process_dir/stat\"\n"
        "  sleep \"${RADIOFLOW_TEST_TRAIN_HOLD_SECONDS:-0.6}\"\n"
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
            "RADIOFLOW_MULTISCALE_UNO_TEST_PROC_ROOT": _bash_path(proc_root),
            "RADIOFLOW_TEST_INVOCATION_LOG": _bash_path(invocation_log),
        }
    )
    return environment, invocation_log


def _commands(invocation_log: Path) -> list[str]:
    if not invocation_log.exists():
        return []
    return [
        line
        for line in invocation_log.read_text(encoding="utf-8").splitlines()
        if not line.startswith("complete ")
    ]


def _owner_fingerprint(environment: dict[str, str], array_size: str) -> str:
    code_root = environment["RADIOFLOW_MULTISCALE_UNO_CODE_ROOT"]
    run_root = f'{environment["RADIOFLOW_MULTISCALE_UNO_RESULT_ROOT"]}/runs/{array_size}'
    payload = f"{code_root}\0{run_root}\0{array_size}\0".encode()
    return hashlib.sha256(payload).hexdigest()


def _write_proc_stat(proc_root: Path, pid: str, start_ticks: int) -> None:
    process_dir = proc_root / pid
    process_dir.mkdir(parents=True, exist_ok=True)
    fields = ["S", *("0" for _ in range(18)), str(start_ticks), "0"]
    (process_dir / "stat").write_text(
        f"{pid} (test sleeper) {' '.join(fields)}\n",
        encoding="utf-8",
    )


def _write_pid_metadata(
    path: Path,
    *,
    pid: str,
    start_ticks: int,
    owner_fingerprint: str,
) -> None:
    path.write_bytes(
        (
            f"pid={pid}\n"
            f"start_ticks={start_ticks}\n"
            f"owner_fingerprint={owner_fingerprint}\n"
        ).encode()
    )


def _read_pid_metadata(path: Path) -> tuple[str, str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert re.fullmatch(r"pid=[1-9][0-9]*", lines[0])
    assert re.fullmatch(r"start_ticks=[1-9][0-9]*", lines[1])
    assert re.fullmatch(r"owner_fingerprint=[0-9a-f]{64}", lines[2])
    return tuple(line.split("=", 1)[1] for line in lines)  # type: ignore[return-value]


def _start_sleep() -> tuple[subprocess.Popen[str], str]:
    process = subprocess.Popen(
        [_bash(), "-c", 'sleep 30 & child=$!; printf "%s\\n" "$child"; wait "$child"'],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    sleep_pid = process.stdout.readline().strip()
    assert sleep_pid.isdecimal()
    return process, sleep_pid


def _stop_sleep(process: subprocess.Popen[str], sleep_pid: str) -> None:
    subprocess.run(
        [_bash(), "-c", 'kill "$1" 2>/dev/null || true', "--", sleep_pid],
        check=True,
    )
    process.wait(timeout=5)


def test_dry_run_emits_three_isolated_default_training_commands() -> None:
    completed = _run("--dry-run")

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert len(lines) == 3
    expected = (("8x8", "0"), ("16x16", "1"), ("32x32", "2"))
    for line, (array_size, gpu) in zip(lines, expected):
        assert line.startswith(f"CUDA_VISIBLE_DEVICES={gpu} python ")
        assert "run_same_frequency_multiscale_uno.py train" in line
        assert "--device cuda:0" in line
        assert "--resume auto" in line
        assert f"manifest_samefreq_6.7ghz_{array_size}_0deg.jsonl" in line
        assert f"multiscale_uno_samefreq_6.7ghz/runs/{array_size}" in line.replace("\\", "/")
    assert "/home/wys/radioflow_20260823/multiscale-uno-singlebeam" in lines[0]


def test_dry_run_exposes_root_overrides_in_the_emitted_commands(tmp_path: Path) -> None:
    environment, invocation_log = _launcher_environment(tmp_path / "root with spaces")
    environment["RADIOFLOW_TEST_TRAIN_HOLD_SECONDS"] = "0"

    completed = _run("--dry-run", environment=environment)

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert len(lines) == 3
    for line in lines:
        parsed = subprocess.run(
            [_bash(), "-n", "-c", line],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert parsed.returncode == 0, parsed.stderr
    executed = subprocess.run(
        [_bash(), "-c", lines[0]],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    assert executed.returncode == 0, executed.stderr
    assert len(_commands(invocation_log)) == 1
    output = completed.stdout
    assert "root\\ with\\ spaces/code" in output
    assert "root\\ with\\ spaces/dataset" in output
    assert "root\\ with\\ spaces/results" in output


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
    commands = _commands(invocation_log)
    assert len(commands) == 3
    for array_size, command in zip(("8x8", "16x16", "32x32"), commands):
        assert f"run_same_frequency_multiscale_uno.py {expected_command} " in command
        assert f"--array-size {array_size}" in command
        assert expected_extra in command


def test_smoke_waits_for_every_array_and_propagates_a_failure(tmp_path: Path) -> None:
    environment, invocation_log = _launcher_environment(tmp_path)
    environment["RADIOFLOW_TEST_SMOKE_FAIL_ARRAY"] = "16x16"
    environment["RADIOFLOW_TEST_SLOW_SMOKE_ARRAY"] = "32x32"
    sentinel = tmp_path / "slow-smoke-complete"
    environment["RADIOFLOW_TEST_SMOKE_SENTINEL"] = _bash_path(sentinel)

    completed = _run("--smoke", environment=environment)

    assert completed.returncode == 1
    assert sentinel.is_file()
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    commands = [line for line in invocations if not line.startswith("complete ")]
    assert len(commands) == 3
    assert all("train" in command and "--smoke-optimizer-steps 1" in command for command in commands)
    assert {line.removeprefix("complete ") for line in invocations if line.startswith("complete ")} == {
        "8x8",
        "32x32",
    }


def test_train_lock_failure_releases_earlier_locks_without_launching(tmp_path: Path) -> None:
    environment, invocation_log = _launcher_environment(tmp_path)
    result_root = tmp_path / "results"
    pid_dir = result_root / "pids"
    pid_dir.mkdir(parents=True)
    blocked_lock = pid_dir / "train_16x16.lock"
    blocked_lock.mkdir()

    completed = _run("--train", environment=environment)

    assert completed.returncode == 1
    assert not invocation_log.exists()
    assert not (pid_dir / "train_8x8.lock").exists()
    assert blocked_lock.is_dir()


def test_train_preflight_checks_all_owned_pids_before_any_launch(tmp_path: Path) -> None:
    environment, invocation_log = _launcher_environment(tmp_path)
    pid_dir = tmp_path / "results" / "pids"
    pid_dir.mkdir(parents=True)
    process, sleep_pid = _start_sleep()
    start_ticks = 123456
    try:
        _write_proc_stat(tmp_path / "proc", sleep_pid, start_ticks)
        _write_pid_metadata(
            pid_dir / "train_16x16.pid",
            pid=sleep_pid,
            start_ticks=start_ticks,
            owner_fingerprint=_owner_fingerprint(environment, "16x16"),
        )

        completed = _run("--train", environment=environment)

        assert completed.returncode == 1, completed.stderr
        assert "already live for 16x16" in completed.stderr
        assert not invocation_log.exists()
        assert not any(pid_dir.glob("*.lock"))
    finally:
        _stop_sleep(process, sleep_pid)


@pytest.mark.parametrize("mismatch", ["birth", "owner"])
def test_train_replaces_stale_reused_or_unowned_pid_metadata(
    tmp_path: Path,
    mismatch: str,
) -> None:
    environment, invocation_log = _launcher_environment(tmp_path)
    environment["RADIOFLOW_TEST_TRAIN_HOLD_SECONDS"] = "0.6"
    pid_dir = tmp_path / "results" / "pids"
    pid_dir.mkdir(parents=True)
    unrelated = pid_dir / "unrelated.pid"
    unrelated.write_text("keep\n", encoding="utf-8")
    process, sleep_pid = _start_sleep()
    live_start_ticks = 777777
    expected_owner = _owner_fingerprint(environment, "8x8")
    try:
        _write_proc_stat(tmp_path / "proc", sleep_pid, live_start_ticks)
        _write_pid_metadata(
            pid_dir / "train_8x8.pid",
            pid=sleep_pid,
            start_ticks=live_start_ticks + (1 if mismatch == "birth" else 0),
            owner_fingerprint=("0" * 64 if mismatch == "owner" else expected_owner),
        )

        completed = _run("--train", environment=environment)

        assert completed.returncode == 0, completed.stderr
        assert len(_commands(invocation_log)) == 3
        pid, _start_ticks, owner = _read_pid_metadata(pid_dir / "train_8x8.pid")
        assert pid != sleep_pid
        assert owner == expected_owner
        assert unrelated.read_text(encoding="utf-8") == "keep\n"
        assert not any(pid_dir.glob("*.lock"))
    finally:
        _stop_sleep(process, sleep_pid)
        time.sleep(0.7)


def test_two_concurrent_train_launchers_create_at_most_one_launch_set(
    tmp_path: Path,
) -> None:
    environment, invocation_log = _launcher_environment(tmp_path)
    environment["RADIOFLOW_TEST_PROC_WRITE_DELAY"] = "0.15"
    environment["RADIOFLOW_TEST_TRAIN_HOLD_SECONDS"] = "0.8"

    first = subprocess.Popen(
        _launcher_command("--train"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    time.sleep(0.03)
    second = subprocess.Popen(
        _launcher_command("--train"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    first_output = first.communicate(timeout=10)
    second_output = second.communicate(timeout=10)

    assert sorted((first.returncode, second.returncode)) == [0, 1], (
        first_output,
        second_output,
    )
    assert len(_commands(invocation_log)) == 3
    assert not any((tmp_path / "results" / "pids").glob("*.lock"))
    time.sleep(1.0)
