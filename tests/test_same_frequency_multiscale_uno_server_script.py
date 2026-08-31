from __future__ import annotations

import functools
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
        errors="replace",
        env=environment,
    )


def _server_path(path: Path) -> str:
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)
    drive, tail = os.path.splitdrive(str(resolved))
    assert drive and len(drive) == 2, resolved
    linux_tail = tail.lstrip("\\/").replace("\\", "/")
    return f"/mnt/{drive[0].lower()}/{linux_tail}"


@functools.lru_cache(maxsize=1)
def _server_prefix() -> list[str]:
    if os.name != "nt":
        if not Path("/bin/bash").is_file():
            pytest.skip("/bin/bash is required for server launcher tests")
        return []
    wsl = shutil.which("wsl.exe")
    if wsl is None:
        pytest.skip("WSL is required for server launcher train tests")
    probe = subprocess.run(
        [wsl, "--exec", "/bin/bash", "-lc", "test -x /usr/bin/flock"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if probe.returncode != 0:
        pytest.skip("WSL util-linux flock is required for server launcher train tests")
    return [wsl, "--exec"]


def _server_command(
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> list[str]:
    assignments = [] if environment is None else [
        f"{key}={value}" for key, value in sorted(environment.items())
    ]
    return [
        *_server_prefix(),
        "/usr/bin/env",
        *assignments,
        "/bin/bash",
        _server_path(SCRIPT),
        *arguments,
    ]


def _server_run(
    *arguments: str,
    environment: dict[str, str],
    timeout: float = 15,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _server_command(*arguments, environment=environment),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _server_shell(
    script: str,
    *arguments: str,
    environment: dict[str, str] | None = None,
    timeout: float = 10,
) -> subprocess.CompletedProcess[str]:
    assignments = [] if environment is None else [
        f"{key}={value}" for key, value in sorted(environment.items())
    ]
    return subprocess.run(
        [
            *_server_prefix(),
            "/usr/bin/env",
            *assignments,
            "/bin/bash",
            "-c",
            script,
            "radioflow-test",
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _launcher_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    code_root = tmp_path / "code"
    dataset_root = tmp_path / "dataset"
    result_root = tmp_path / "results"
    proc_root = tmp_path / "proc"
    env_file = tmp_path / "remote_env.sh"
    invocation_log = tmp_path / "python-invocations.log"
    started_pid_log = tmp_path / "started-pids.log"
    alive_dir = tmp_path / "alive"
    bin_dir = tmp_path / "bin"
    code_root.mkdir()
    dataset_root.mkdir()
    proc_root.mkdir()
    alive_dir.mkdir()
    bin_dir.mkdir()
    (code_root / "run_same_frequency_multiscale_uno.py").touch()
    env_file.write_text(
        "#!/usr/bin/env bash\n"
        "process_dir=\"${RADIOFLOW_MULTISCALE_UNO_TEST_PROC_ROOT:?}/$$\"\n"
        "mkdir -p \"$process_dir\"\n"
        "fields=(S)\n"
        "for _field in {4..21}; do fields+=(0); done\n"
        "fields+=(\"${RADIOFLOW_TEST_LAUNCHER_START_TICKS:-313131}\")\n"
        "printf '%s (test-launcher) %s\\n' \"$$\" \"${fields[*]}\" > \"$process_dir/stat\"\n"
        "mv() {\n"
        "  for argument in \"$@\"; do\n"
        "    if [[ -n \"${RADIOFLOW_TEST_FAIL_PID_PUBLICATION_ARRAY:-}\" "
        "&& \"$argument\" == */train_${RADIOFLOW_TEST_FAIL_PID_PUBLICATION_ARRAY}.pid ]]; then\n"
        "      return 73\n"
        "    fi\n"
        "  done\n"
        "  /usr/bin/mv \"$@\"\n"
        "}\n",
        encoding="utf-8",
    )
    (bin_dir / "flock").write_text(
        "#!/usr/bin/env bash\n"
        "exit 1\n",
        encoding="utf-8",
    )
    (bin_dir / "flock").chmod(0o755)
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
        "  printf '%s %s\\n' \"$array_size\" \"$$\" >> \"${RADIOFLOW_TEST_STARTED_PID_LOG:?}\"\n"
        "  alive_path=\"${RADIOFLOW_TEST_ALIVE_DIR:?}/${array_size}.$$\"\n"
        "  : > \"$alive_path\"\n"
        "  hold_pid=''\n"
        "  trap 'rm -f \"$alive_path\"' EXIT\n"
        "  trap '[[ -z \"$hold_pid\" ]] || kill \"$hold_pid\" 2>/dev/null; exit 143' TERM\n"
        "  sleep \"${RADIOFLOW_TEST_PROC_WRITE_DELAY:-0}\"\n"
        "  if [[ \"${RADIOFLOW_TEST_SKIP_PROC_ARRAY:-}\" != \"$array_size\" ]]; then\n"
        "    process_dir=\"${RADIOFLOW_MULTISCALE_UNO_TEST_PROC_ROOT:?}/$$\"\n"
        "    mkdir -p \"$process_dir\"\n"
        "    fields=(S)\n"
        "    for _field in {4..21}; do fields+=(0); done\n"
        "    fields+=(\"${RADIOFLOW_TEST_PROCESS_START_TICKS:-424242}\")\n"
        "    printf '%s (fake-python) %s\\n' \"$$\" \"${fields[*]}\" > \"$process_dir/stat\"\n"
        "  fi\n"
        "  sleep \"${RADIOFLOW_TEST_TRAIN_HOLD_SECONDS:-0.6}\" &\n"
        "  hold_pid=$!\n"
        "  wait \"$hold_pid\"\n"
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
            "RADIOFLOW_TEST_STARTED_PID_LOG": _bash_path(started_pid_log),
            "RADIOFLOW_TEST_ALIVE_DIR": _bash_path(alive_dir),
        }
    )
    return environment, invocation_log


def _server_launcher_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    _server_prefix()
    tmp_path.mkdir(parents=True, exist_ok=True)
    code_root = tmp_path / "code"
    dataset_root = tmp_path / "dataset"
    result_root = tmp_path / "results"
    env_file = tmp_path / "remote_env.sh"
    invocation_log = tmp_path / "python-invocations.log"
    started_pid_log = tmp_path / "started-pids.log"
    spawned_pid_log = tmp_path / "spawned-pids.log"
    alive_dir = tmp_path / "alive"
    bin_dir = tmp_path / "bin"
    code_root.mkdir()
    dataset_root.mkdir()
    alive_dir.mkdir()
    bin_dir.mkdir()
    env_file.write_text(
        "#!/bin/bash\n"
        "if [[ -n \"${RADIOFLOW_TEST_LAUNCHER_PID_LOG:-}\" ]]; then\n"
        "  printf '%s\\n' \"$$\" > \"$RADIOFLOW_TEST_LAUNCHER_PID_LOG\"\n"
        "fi\n"
        "if [[ -n \"${RADIOFLOW_TEST_CLEANUP_SIGNAL_MARKER:-}\" "
        "|| -n \"${RADIOFLOW_TEST_REUSED_PID_SIGNAL_LOG:-}\" ]]; then\n"
        "  _radioflow_cleanup_signal_marked=0\n"
        "  kill() {\n"
        "    local signal=\"${1:-}\" target=\"${!#}\"\n"
        "    if [[ -n \"${RADIOFLOW_TEST_REUSED_PID_SIGNAL_LOG:-}\" "
        "&& -n \"${RADIOFLOW_TEST_FAST_EXIT_ARRAY:-}\" "
        "&& -f \"${RADIOFLOW_TEST_STARTED_PID_LOG:?}\" ]] "
        "&& /usr/bin/grep -Fqx "
        "\"${RADIOFLOW_TEST_FAST_EXIT_ARRAY} ${target}\" "
        "\"$RADIOFLOW_TEST_STARTED_PID_LOG\"; then\n"
        "      if [[ \"$signal\" == '-0' ]]; then\n"
        "        return 0\n"
        "      fi\n"
        "      printf '%s %s\\n' \"$signal\" \"$target\" >> "
        "\"$RADIOFLOW_TEST_REUSED_PID_SIGNAL_LOG\"\n"
        "      builtin kill \"$signal\" "
        "\"${RADIOFLOW_TEST_UNRELATED_SENTINEL_PID:?}\" 2>/dev/null || true\n"
        "      return 0\n"
        "    fi\n"
        "    if [[ -n \"${RADIOFLOW_TEST_CLEANUP_SIGNAL_MARKER:-}\" "
        "&& \"$signal\" == '-TERM' "
        "&& \"$_radioflow_cleanup_signal_marked\" -eq 0 ]]; then\n"
        "      _radioflow_cleanup_signal_marked=1\n"
        "      printf 'cleanup\\n' > \"$RADIOFLOW_TEST_CLEANUP_SIGNAL_MARKER\"\n"
        "      /bin/sleep \"${RADIOFLOW_TEST_CLEANUP_SIGNAL_DELAY:-0.4}\"\n"
        "    fi\n"
        "    builtin kill \"$@\"\n"
        "  }\n"
        "fi\n"
        "mv() {\n"
        "  local argument publication_target='' source_path=''\n"
        "  local pid_line publication_pid attempt\n"
        "  for argument in \"$@\"; do\n"
        "    if [[ \"$argument\" == */train_*.pid ]]; then\n"
        "      publication_target=\"$argument\"\n"
        "    elif [[ \"$argument\" == */train_*.pid.tmp.* ]]; then\n"
        "      source_path=\"$argument\"\n"
        "    fi\n"
        "  done\n"
        "  if [[ -n \"${RADIOFLOW_TEST_PID_PUBLICATION_DELAY:-}\" "
        "&& -n \"$publication_target\" ]]; then\n"
        "    /bin/sleep \"$RADIOFLOW_TEST_PID_PUBLICATION_DELAY\"\n"
        "  fi\n"
        "  if [[ -n \"${RADIOFLOW_TEST_FAIL_PID_PUBLICATION_ARRAY:-}\" "
        "&& \"$publication_target\" == "
        "*/train_${RADIOFLOW_TEST_FAIL_PID_PUBLICATION_ARRAY}.pid ]]; then\n"
        "    IFS= read -r pid_line < \"$source_path\" || return 74\n"
        "    [[ \"$pid_line\" =~ ^pid=([1-9][0-9]*)$ ]] || return 74\n"
        "    publication_pid=\"${BASH_REMATCH[1]}\"\n"
        "    for ((attempt = 0; attempt < 200; attempt++)); do\n"
        "      if /usr/bin/grep -Fqx "
        "\"${RADIOFLOW_TEST_FAIL_PID_PUBLICATION_ARRAY} $publication_pid\" "
        "\"${RADIOFLOW_TEST_STARTED_PID_LOG:?}\" 2>/dev/null; then\n"
        "        break\n"
        "      fi\n"
        "      kill -0 \"$publication_pid\" 2>/dev/null || return 74\n"
        "      /bin/sleep 0.01\n"
        "    done\n"
        "    /usr/bin/grep -Fqx "
        "\"${RADIOFLOW_TEST_FAIL_PID_PUBLICATION_ARRAY} $publication_pid\" "
        "\"$RADIOFLOW_TEST_STARTED_PID_LOG\" || return 74\n"
        "    /usr/bin/mv \"$@\" || return\n"
        "    return 73\n"
        "  fi\n"
        "  /usr/bin/mv \"$@\"\n"
        "}\n"
        "if [[ -n \"${RADIOFLOW_TEST_FAIL_START_TICKS_ARRAY:-}\" ]]; then\n"
        "  capture_process_start_ticks() {\n"
        "    local pid=\"$1\" attempt start_ticks\n"
        "    for ((attempt = 0; attempt < 100; attempt++)); do\n"
        "      if [[ -f \"${RADIOFLOW_TEST_STARTED_PID_LOG:?}\" ]] "
        "&& /usr/bin/grep -Eq \"^[^ ]+ ${pid}$\" "
        "\"$RADIOFLOW_TEST_STARTED_PID_LOG\"; then\n"
        "        if /usr/bin/grep -Fqx "
        "\"${RADIOFLOW_TEST_FAIL_START_TICKS_ARRAY} ${pid}\" "
        "\"$RADIOFLOW_TEST_STARTED_PID_LOG\"; then\n"
        "          if [[ \"${RADIOFLOW_TEST_WAIT_FOR_FAILED_PROCESS_EXIT:-0}\" "
        "== '1' ]]; then\n"
        "            while [[ -e \"/proc/$pid\" ]]; do\n"
        "              /bin/sleep 0.01\n"
        "            done\n"
        "          fi\n"
        "          return 1\n"
        "        fi\n"
        "        break\n"
        "      fi\n"
        "      kill -0 \"$pid\" 2>/dev/null || return 1\n"
        "      /bin/sleep 0.01\n"
        "    done\n"
        "    for ((attempt = 0; attempt < 100; attempt++)); do\n"
        "      if start_ticks=\"$(read_process_start_ticks \"$pid\")\"; then\n"
        "        printf '%s\\n' \"$start_ticks\"\n"
        "        return 0\n"
        "      fi\n"
        "      kill -0 \"$pid\" 2>/dev/null || return 1\n"
        "      /bin/sleep 0.01\n"
        "    done\n"
        "    return 1\n"
        "  }\n"
        "fi\n"
        "_radioflow_debug_signal_hook() {\n"
        "  if [[ \"$BASH_COMMAND\" == 'pid=\"$!\"' ]]; then\n"
        "    ((_radioflow_pid_capture_count += 1))\n"
        "    if [[ -n \"${RADIOFLOW_TEST_SIGNAL_BEFORE_PID_CAPTURE_LOG:-}\" "
        "&& \"${_radioflow_signal_injected:-0}\" -eq 0 "
        "&& \"$_radioflow_pid_capture_count\" -eq "
        "\"${RADIOFLOW_TEST_SIGNAL_BEFORE_PID_CAPTURE_INDEX:-1}\" ]]; then\n"
        "      _radioflow_signal_injected=1\n"
        "      printf '%s\\n' \"$!\" > "
        "\"$RADIOFLOW_TEST_SIGNAL_BEFORE_PID_CAPTURE_LOG\"\n"
        "      kill -TERM \"$$\"\n"
        "    fi\n"
        "  fi\n"
        "}\n"
        "if [[ -n \"${RADIOFLOW_TEST_SIGNAL_BEFORE_PID_CAPTURE_LOG:-}\" ]]; then\n"
        "  _radioflow_signal_injected=0\n"
        "  _radioflow_pid_capture_count=0\n"
        "  trap _radioflow_debug_signal_hook DEBUG\n"
        "fi\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_program = code_root / "run_same_frequency_multiscale_uno.py"
    fake_program.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >> \"${RADIOFLOW_TEST_INVOCATION_LOG:?}\"\n"
        "array_size=''\n"
        "for ((index = 1; index <= $#; index++)); do\n"
        "  if [[ \"${!index}\" == '--array-size' ]]; then\n"
        "    next=$((index + 1))\n"
        "    array_size=\"${!next}\"\n"
        "  fi\n"
        "done\n"
        "if [[ \"${1:-}\" == 'train' && \"$*\" != *'--preflight-only'* "
        "&& \"$*\" != *'--smoke-optimizer-steps'* ]]; then\n"
        "  printf '%s %s\\n' \"$array_size\" \"$$\" >> "
        "\"${RADIOFLOW_TEST_STARTED_PID_LOG:?}\"\n"
        "  if [[ \"$array_size\" == \"${RADIOFLOW_TEST_FAST_EXIT_ARRAY:-}\" ]]; then\n"
        "    exit 0\n"
        "  fi\n"
        "  alive_path=\"${RADIOFLOW_TEST_ALIVE_DIR:?}/${array_size}.$$\"\n"
        "  hold_pid=''\n"
        "  cleanup() {\n"
        "    rm -f -- \"$alive_path\"\n"
        "    if [[ -n \"$hold_pid\" ]]; then\n"
        "      kill \"$hold_pid\" 2>/dev/null || true\n"
        "      wait \"$hold_pid\" 2>/dev/null || true\n"
        "    fi\n"
        "  }\n"
        "  trap cleanup EXIT\n"
        "  trap 'exit 143' TERM HUP INT\n"
        "  : > \"$alive_path\"\n"
        "  /bin/sleep \"${RADIOFLOW_TEST_TRAIN_HOLD_SECONDS:-5}\" &\n"
        "  hold_pid=$!\n"
        "  wait \"$hold_pid\"\n"
        "fi\n"
        "printf 'complete %s\\n' \"$array_size\" >> "
        "\"${RADIOFLOW_TEST_INVOCATION_LOG:?}\"\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_program.chmod(0o755)
    fake_nohup = bin_dir / "nohup"
    fake_nohup.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "array_size=''\n"
        "for ((index = 1; index <= $#; index++)); do\n"
        "  if [[ \"${!index}\" == '--array-size' ]]; then\n"
        "    next=$((index + 1))\n"
        "    array_size=\"${!next}\"\n"
        "  fi\n"
        "done\n"
        "printf '%s %s\\n' \"$array_size\" \"$$\" >> "
        "\"${RADIOFLOW_TEST_SPAWNED_PID_LOG:?}\"\n"
        "exec /usr/bin/nohup \"$@\"\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_nohup.chmod(0o755)
    server_bin = _server_path(bin_dir)
    environment = {
        "LC_ALL": "C",
        "PATH": f"{server_bin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHON_BIN": "/bin/bash",
        "RADIOFLOW_MULTISCALE_UNO_CODE_ROOT": _server_path(code_root),
        "RADIOFLOW_MULTISCALE_UNO_DATASET_ROOT": _server_path(dataset_root),
        "RADIOFLOW_MULTISCALE_UNO_RESULT_ROOT": _server_path(result_root),
        "RADIOFLOW_MULTISCALE_UNO_ENV_FILE": _server_path(env_file),
        "RADIOFLOW_TEST_INVOCATION_LOG": _server_path(invocation_log),
        "RADIOFLOW_TEST_STARTED_PID_LOG": _server_path(started_pid_log),
        "RADIOFLOW_TEST_SPAWNED_PID_LOG": _server_path(spawned_pid_log),
        "RADIOFLOW_TEST_ALIVE_DIR": _server_path(alive_dir),
        "RADIOFLOW_TEST_PID_PUBLICATION_DELAY": "0.05",
    }
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


def _started_pids(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    return [tuple(line.split()) for line in path.read_text(encoding="utf-8").splitlines()]  # type: ignore[list-item]


def _wait_for_started_pids(
    path: Path,
    minimum: int,
    *,
    timeout: float = 2,
) -> list[tuple[str, str]]:
    deadline = time.monotonic() + timeout
    started = _started_pids(path)
    while len(started) < minimum and time.monotonic() < deadline:
        time.sleep(0.02)
        started = _started_pids(path)
    return started


def _wait_for_file_text(path: Path, *, timeout: float = 3) -> str:
    deadline = time.monotonic() + timeout
    value = ""
    while time.monotonic() < deadline:
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
        time.sleep(0.02)
    return value


def _pid_is_alive(pid: str) -> bool:
    completed = subprocess.run(
        [_bash(), "-c", 'kill -0 "$1" 2>/dev/null', "--", pid],
        check=False,
    )
    return completed.returncode == 0


def _terminate_pids(pids: list[str]) -> None:
    for pid in pids:
        subprocess.run(
            [_bash(), "-c", 'kill "$1" 2>/dev/null || true', "--", pid],
            check=True,
        )
    deadline = time.monotonic() + 3
    while any(_pid_is_alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.05)


def _server_pid_is_alive(pid: str) -> bool:
    completed = _server_shell('kill -0 "$1" 2>/dev/null && test -e "/proc/$1" ', pid)
    return completed.returncode == 0


def _terminate_server_pids(pids: list[str]) -> None:
    unique_pids = list(dict.fromkeys(pids))
    if not unique_pids:
        return
    _server_shell(
        'for pid in "$@"; do kill -TERM "$pid" 2>/dev/null || true; done',
        *unique_pids,
    )
    deadline = time.monotonic() + 3
    while (
        any(_server_pid_is_alive(pid) for pid in unique_pids)
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    remaining = [pid for pid in unique_pids if _server_pid_is_alive(pid)]
    if remaining:
        _server_shell(
            'for pid in "$@"; do kill -KILL "$pid" 2>/dev/null || true; done',
            *remaining,
        )


def _start_server_sleep() -> tuple[subprocess.Popen[str], str]:
    process = subprocess.Popen(
        [
            *_server_prefix(),
            "/bin/bash",
            "-c",
            '/bin/sleep 30 & child=$!; printf "%s\\n" "$child"; wait "$child"',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    sleep_pid = process.stdout.readline().strip()
    assert sleep_pid.isdecimal()
    return process, sleep_pid


def _stop_server_sleep(process: subprocess.Popen[str], sleep_pid: str) -> None:
    _server_shell('kill -TERM "$1" 2>/dev/null || true', sleep_pid)
    process.wait(timeout=5)


def _start_server_signal_sentinel(marker: Path) -> tuple[subprocess.Popen[str], str]:
    process = subprocess.Popen(
        [
            *_server_prefix(),
            "/bin/bash",
            "-c",
            "trap 'printf \"signal\\n\" >> \"$1\"' HUP INT TERM; "
            "printf '%s\\n' \"$$\"; "
            "while :; do /bin/sleep 0.1; done",
            "radioflow-sentinel",
            _server_path(marker),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    sentinel_pid = process.stdout.readline().strip()
    assert sentinel_pid.isdecimal()
    return process, sentinel_pid


def _stop_server_signal_sentinel(
    process: subprocess.Popen[str], sentinel_pid: str
) -> None:
    _server_shell('kill -KILL "$1" 2>/dev/null || true', sentinel_pid)
    process.wait(timeout=5)


def _server_process_start_ticks(pid: str) -> int:
    completed = _server_shell(
        'stat_line="$(<"/proc/$1/stat")"; '
        'stat_tail="${stat_line##*) }"; '
        "set -- $stat_tail; "
        'printf "%s\\n" "${20}"',
        pid,
    )
    assert completed.returncode == 0, completed.stderr
    return int(completed.stdout.strip())


def _assert_server_locks_reacquirable(pid_dir: Path) -> None:
    lock_paths = [
        _server_path(pid_dir / f"train_{array_size}.flock")
        for array_size in ARRAYS
    ]
    completed = _server_shell(
        'for lock_path in "$@"; do '
        'exec {lock_fd}>> "$lock_path" || exit 90; '
        '/usr/bin/flock -n "$lock_fd" || exit 91; '
        "done",
        *lock_paths,
    )
    assert completed.returncode == 0, completed.stderr


def _path_without_flock(tmp_path: Path) -> str:
    bin_dir = tmp_path / "no-flock-bin"
    bin_dir.mkdir()
    for command in ("mkdir", "sha256sum"):
        wrapper = bin_dir / command
        wrapper.write_text(
            f"#!/bin/bash\nexec /usr/bin/{command} \"$@\"\n",
            encoding="utf-8",
            newline="\n",
        )
        wrapper.chmod(0o755)
    return _server_path(bin_dir)


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
        errors="replace",
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


def test_train_requires_util_linux_flock_before_launching(tmp_path: Path) -> None:
    environment, invocation_log = _server_launcher_environment(tmp_path)
    environment["PATH"] = _path_without_flock(tmp_path)

    completed = _server_run("--train", environment=environment)

    assert completed.returncode == 1
    assert "util-linux flock is required for --train" in completed.stderr
    assert not invocation_log.exists()
    assert not (tmp_path / "started-pids.log").exists()


def test_train_preflight_checks_all_owned_pids_before_any_launch(tmp_path: Path) -> None:
    environment, invocation_log = _server_launcher_environment(tmp_path)
    pid_dir = tmp_path / "results" / "pids"
    pid_dir.mkdir(parents=True)
    process, sleep_pid = _start_server_sleep()
    try:
        _write_pid_metadata(
            pid_dir / "train_16x16.pid",
            pid=sleep_pid,
            start_ticks=_server_process_start_ticks(sleep_pid),
            owner_fingerprint=_owner_fingerprint(environment, "16x16"),
        )

        completed = _server_run("--train", environment=environment)

        assert completed.returncode == 1, completed.stderr
        assert "already live for 16x16" in completed.stderr
        assert not invocation_log.exists()
        _assert_server_locks_reacquirable(pid_dir)
    finally:
        _stop_server_sleep(process, sleep_pid)


@pytest.mark.parametrize("mismatch", ["birth", "owner"])
def test_train_replaces_stale_reused_or_unowned_pid_metadata(
    tmp_path: Path,
    mismatch: str,
) -> None:
    environment, _invocation_log = _server_launcher_environment(tmp_path)
    pid_dir = tmp_path / "results" / "pids"
    pid_dir.mkdir(parents=True)
    unrelated = pid_dir / "unrelated.pid"
    unrelated.write_text("keep\n", encoding="utf-8")
    process, sleep_pid = _start_server_sleep()
    expected_owner = _owner_fingerprint(environment, "8x8")
    launched_pids: list[str] = []
    try:
        live_start_ticks = _server_process_start_ticks(sleep_pid)
        _write_pid_metadata(
            pid_dir / "train_8x8.pid",
            pid=sleep_pid,
            start_ticks=live_start_ticks + (1 if mismatch == "birth" else 0),
            owner_fingerprint=("0" * 64 if mismatch == "owner" else expected_owner),
        )

        completed = _server_run("--train", environment=environment)
        launched_pids = [
            pid
            for _array_size, pid in _wait_for_started_pids(
                tmp_path / "spawned-pids.log", 3
            )
        ]

        assert completed.returncode == 0, completed.stderr
        assert len(launched_pids) == 3
        assert completed.stdout.count("LAUNCHED ARRAY=") == 3
        pid, _start_ticks, owner = _read_pid_metadata(pid_dir / "train_8x8.pid")
        assert pid != sleep_pid
        assert owner == expected_owner
        assert unrelated.read_text(encoding="utf-8") == "keep\n"
    finally:
        _stop_server_sleep(process, sleep_pid)
        _terminate_server_pids(launched_pids)


def test_two_concurrent_train_launchers_create_at_most_one_launch_set(
    tmp_path: Path,
) -> None:
    environment, _invocation_log = _server_launcher_environment(tmp_path)
    environment["RADIOFLOW_TEST_PID_PUBLICATION_DELAY"] = "0.15"
    environment["RADIOFLOW_TEST_TRAIN_HOLD_SECONDS"] = "5"

    first = subprocess.Popen(
        _server_command("--train", environment=environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    time.sleep(0.03)
    second = subprocess.Popen(
        _server_command("--train", environment=environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    launched_pids: list[str] = []
    try:
        first_output = first.communicate(timeout=15)
        second_output = second.communicate(timeout=15)
        launched_pids = [
            pid
            for _array_size, pid in _wait_for_started_pids(
                tmp_path / "spawned-pids.log", 3
            )
        ]

        assert sorted((first.returncode, second.returncode)) == [0, 1], (
            first_output,
            second_output,
        )
        assert len(launched_pids) == 3
    finally:
        _terminate_server_pids(launched_pids)


def test_train_children_do_not_inherit_launcher_flock_descriptors(
    tmp_path: Path,
) -> None:
    environment, _invocation_log = _server_launcher_environment(tmp_path)
    environment["RADIOFLOW_TEST_TRAIN_HOLD_SECONDS"] = "30"
    started: list[tuple[str, str]] = []
    try:
        completed = _server_run("--train", environment=environment)
        started = _wait_for_started_pids(tmp_path / "spawned-pids.log", 3)

        assert completed.returncode == 0, completed.stderr
        assert len(started) == 3
        assert completed.stdout.count("LAUNCHED ARRAY=") == 3
        assert all(_server_pid_is_alive(pid) for _array_size, pid in started)
        _assert_server_locks_reacquirable(tmp_path / "results" / "pids")
    finally:
        _terminate_server_pids([pid for _array_size, pid in started])


def test_second_array_start_identity_failure_rolls_back_every_launch(
    tmp_path: Path,
) -> None:
    environment, _invocation_log = _server_launcher_environment(tmp_path)
    environment["RADIOFLOW_TEST_FAIL_START_TICKS_ARRAY"] = "16x16"
    environment["RADIOFLOW_TEST_TRAIN_HOLD_SECONDS"] = "5"
    started_log = tmp_path / "started-pids.log"
    started: list[tuple[str, str]] = []
    spawned: list[tuple[str, str]] = []
    try:
        completed = _server_run("--train", environment=environment)
        started = _started_pids(started_log)
        spawned = _started_pids(tmp_path / "spawned-pids.log")

        assert completed.returncode != 0
        assert [array_size for array_size, _pid in started] == ["8x8", "16x16"]
        assert all(not _server_pid_is_alive(pid) for _array_size, pid in spawned)
        assert not list((tmp_path / "alive").iterdir())
        pid_dir = tmp_path / "results" / "pids"
        assert not list(pid_dir.glob("train_*.pid"))
        assert not list(pid_dir.glob("train_*.pid.tmp.*"))
        _assert_server_locks_reacquirable(pid_dir)
    finally:
        _terminate_server_pids([pid for _array_size, pid in [*started, *spawned]])


def test_signal_before_second_pid_capture_defers_until_rollback_record_is_complete(
    tmp_path: Path,
) -> None:
    environment, _invocation_log = _server_launcher_environment(tmp_path)
    environment["RADIOFLOW_TEST_TRAIN_HOLD_SECONDS"] = "5"
    environment["RADIOFLOW_TEST_PID_PUBLICATION_DELAY"] = "0.05"
    signal_pid_log = tmp_path / "signal-child.pid"
    environment["RADIOFLOW_TEST_SIGNAL_BEFORE_PID_CAPTURE_LOG"] = _server_path(
        signal_pid_log
    )
    environment["RADIOFLOW_TEST_SIGNAL_BEFORE_PID_CAPTURE_INDEX"] = "2"
    tracked_pids: list[str] = []
    try:
        completed = _server_run("--train", environment=environment)
        assert signal_pid_log.is_file(), completed.stderr
        signal_pid = signal_pid_log.read_text(encoding="utf-8").strip()
        tracked_pids = [
            pid
            for _array_size, pid in [
                *_started_pids(tmp_path / "started-pids.log"),
                *_started_pids(tmp_path / "spawned-pids.log"),
            ]
        ]
        tracked_pids.append(signal_pid)

        assert completed.returncode == 143
        assert signal_pid.isdecimal()
        assert all(not _server_pid_is_alive(pid) for pid in set(tracked_pids))
        pid_dir = tmp_path / "results" / "pids"
        assert not list(pid_dir.glob("train_*.pid"))
        assert not list(pid_dir.glob("train_*.pid.tmp.*"))
        _assert_server_locks_reacquirable(pid_dir)
    finally:
        _terminate_server_pids(tracked_pids)


def test_cleanup_ignores_repeated_handled_signals_until_rollback_finishes(
    tmp_path: Path,
) -> None:
    environment, _invocation_log = _server_launcher_environment(tmp_path)
    launcher_pid_log = tmp_path / "launcher.pid"
    cleanup_marker = tmp_path / "cleanup-started"
    environment["RADIOFLOW_TEST_LAUNCHER_PID_LOG"] = _server_path(launcher_pid_log)
    environment["RADIOFLOW_TEST_CLEANUP_SIGNAL_MARKER"] = _server_path(
        cleanup_marker
    )
    environment["RADIOFLOW_TEST_CLEANUP_SIGNAL_DELAY"] = "0.5"
    environment["RADIOFLOW_TEST_PID_PUBLICATION_DELAY"] = "0.2"
    environment["RADIOFLOW_TEST_TRAIN_HOLD_SECONDS"] = "5"
    process = subprocess.Popen(
        _server_command("--train", environment=environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    launcher_pid = ""
    spawned: list[tuple[str, str]] = []
    try:
        launcher_pid = _wait_for_file_text(launcher_pid_log)
        assert launcher_pid.isdecimal()
        started = _wait_for_started_pids(tmp_path / "started-pids.log", 2, timeout=5)
        assert len(started) >= 2

        initial_signal = _server_shell('kill -TERM "$1"', launcher_pid)
        assert initial_signal.returncode == 0, initial_signal.stderr
        assert _wait_for_file_text(cleanup_marker, timeout=5) == "cleanup"
        repeated_signals = _server_shell(
            'for signal in HUP INT TERM HUP INT TERM; do '
            'kill "-$signal" "$1" 2>/dev/null || true; '
            "/bin/sleep 0.02; "
            "done",
            launcher_pid,
        )
        assert repeated_signals.returncode == 0, repeated_signals.stderr

        stdout, stderr = process.communicate(timeout=10)
        spawned = _started_pids(tmp_path / "spawned-pids.log")

        assert process.returncode == 143, (stdout, stderr)
        assert len(spawned) >= 2
        assert all(not _server_pid_is_alive(pid) for _array_size, pid in spawned)
        assert not list((tmp_path / "alive").iterdir())
        pid_dir = tmp_path / "results" / "pids"
        assert not list(pid_dir.glob("train_*.pid"))
        assert not list(pid_dir.glob("train_*.pid.tmp.*"))
        _assert_server_locks_reacquirable(pid_dir)
    finally:
        if process.poll() is None and launcher_pid:
            _server_shell('kill -KILL "$1" 2>/dev/null || true', launcher_pid)
            process.wait(timeout=5)
        _terminate_server_pids(
            [
                pid
                for _array_size, pid in [
                    *spawned,
                    *_started_pids(tmp_path / "spawned-pids.log"),
                ]
            ]
        )


def test_unknown_birth_fast_exit_never_signals_reused_unrelated_pid(
    tmp_path: Path,
) -> None:
    environment, _invocation_log = _server_launcher_environment(tmp_path)
    sentinel_marker = tmp_path / "unrelated-sentinel-signaled"
    reused_pid_signal_log = tmp_path / "reused-pid-signals.log"
    sentinel_process, sentinel_pid = _start_server_signal_sentinel(sentinel_marker)
    environment["RADIOFLOW_TEST_FAIL_START_TICKS_ARRAY"] = "16x16"
    environment["RADIOFLOW_TEST_FAST_EXIT_ARRAY"] = "16x16"
    environment["RADIOFLOW_TEST_WAIT_FOR_FAILED_PROCESS_EXIT"] = "1"
    environment["RADIOFLOW_TEST_REUSED_PID_SIGNAL_LOG"] = _server_path(
        reused_pid_signal_log
    )
    environment["RADIOFLOW_TEST_UNRELATED_SENTINEL_PID"] = sentinel_pid
    environment["RADIOFLOW_TEST_TRAIN_HOLD_SECONDS"] = "5"
    spawned: list[tuple[str, str]] = []
    try:
        completed = _server_run("--train", environment=environment)
        spawned = _started_pids(tmp_path / "spawned-pids.log")

        assert completed.returncode != 0
        assert [array_size for array_size, _pid in spawned] == ["8x8", "16x16"]
        assert all(not _server_pid_is_alive(pid) for _array_size, pid in spawned)
        assert not reused_pid_signal_log.exists()
        assert not sentinel_marker.exists()
        assert _server_pid_is_alive(sentinel_pid)
        pid_dir = tmp_path / "results" / "pids"
        assert not list(pid_dir.glob("train_*.pid"))
        assert not list(pid_dir.glob("train_*.pid.tmp.*"))
        _assert_server_locks_reacquirable(pid_dir)
    finally:
        _terminate_server_pids([pid for _array_size, pid in spawned])
        _stop_server_signal_sentinel(sentinel_process, sentinel_pid)


def test_second_array_pid_publication_failure_rolls_back_every_launch(
    tmp_path: Path,
) -> None:
    environment, _invocation_log = _server_launcher_environment(tmp_path)
    environment["RADIOFLOW_TEST_TRAIN_HOLD_SECONDS"] = "5"
    environment["RADIOFLOW_TEST_PID_PUBLICATION_DELAY"] = "0.05"
    environment["RADIOFLOW_TEST_FAIL_PID_PUBLICATION_ARRAY"] = "16x16"
    started_log = tmp_path / "started-pids.log"
    started: list[tuple[str, str]] = []
    spawned: list[tuple[str, str]] = []
    try:
        completed = _server_run("--train", environment=environment)
        started = _started_pids(started_log)
        spawned = _started_pids(tmp_path / "spawned-pids.log")

        assert completed.returncode != 0
        assert [array_size for array_size, _pid in started] == ["8x8", "16x16"]
        assert all(not _server_pid_is_alive(pid) for _array_size, pid in spawned)
        assert not list((tmp_path / "alive").iterdir())
        pid_dir = tmp_path / "results" / "pids"
        assert not list(pid_dir.glob("train_*.pid"))
        assert not list(pid_dir.glob("train_*.pid.tmp.*"))
        _assert_server_locks_reacquirable(pid_dir)
    finally:
        _terminate_server_pids([pid for _array_size, pid in [*started, *spawned]])
