#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "an explicit mode is required: --dry-run, --preflight, --smoke, --train, --select-cfg, or --test" >&2
  exit 2
fi

MODE="$1"
case "$MODE" in
  --dry-run|--preflight|--smoke|--train|--select-cfg|--test) ;;
  *)
    echo "unsupported mode: $MODE" >&2
    exit 2
    ;;
esac

CODE_ROOT="${RADIOFLOW_MULTISCALE_UNO_CODE_ROOT:-/home/wys/radioflow_20260823/multiscale-uno-singlebeam}"
DATASET_ROOT="${RADIOFLOW_MULTISCALE_UNO_DATASET_ROOT:-/home/wys/radioflow_20260823/datasets/MultiConfigRadiomap}"
RESULT_ROOT="${RADIOFLOW_MULTISCALE_UNO_RESULT_ROOT:-/home/wys/radioflow_20260823/results/multiscale_uno_samefreq_6.7ghz}"
ENV_FILE="${RADIOFLOW_MULTISCALE_UNO_ENV_FILE:-/home/wys/radioflow_20260823/radioflow_remote_env.sh}"
PROC_ROOT="${RADIOFLOW_MULTISCALE_UNO_TEST_PROC_ROOT:-/proc}"
MANIFEST_ROOT="$DATASET_ROOT/manifests"
HEIGHT_STATS="$MANIFEST_ROOT/height_stats_train.json"

ARRAYS=(8x8 16x16 32x32)
GPUS=(0 1 2)

build_train_command() {
  local array_size="$1"
  TRAIN_COMMAND=(
    python -u "$CODE_ROOT/run_same_frequency_multiscale_uno.py" train
    --dataset-root "$DATASET_ROOT"
    --manifest-path "$MANIFEST_ROOT/manifest_samefreq_6.7ghz_${array_size}_0deg.jsonl"
    --height-stats-path "$HEIGHT_STATS"
    --run-root "$RESULT_ROOT/runs/$array_size"
    --array-size "$array_size"
    --device cuda:0
    --resume auto
  )
}

build_evaluation_command() {
  local array_size="$1"
  local command="$2"
  EVALUATION_COMMAND=(
    python -u "$CODE_ROOT/run_same_frequency_multiscale_uno.py" "$command"
    --dataset-root "$DATASET_ROOT"
    --manifest-path "$MANIFEST_ROOT/manifest_samefreq_6.7ghz_${array_size}_0deg.jsonl"
    --height-stats-path "$HEIGHT_STATS"
    --run-root "$RESULT_ROOT/runs/$array_size"
    --results-root "$RESULT_ROOT/results/$array_size"
    --array-size "$array_size"
    --device cuda:0
  )
}

print_train_command() {
  local array_size="$1"
  local gpu="$2"
  build_train_command "$array_size"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
  printf '%q ' "${TRAIN_COMMAND[@]}"
  printf '\n'
}

build_owner_fingerprint() {
  local array_size="$1"
  local run_root="$RESULT_ROOT/runs/$array_size"
  local digest
  digest="$(printf '%s\0%s\0%s\0' "$CODE_ROOT" "$run_root" "$array_size" | sha256sum)"
  OWNER_FINGERPRINT="${digest%% *}"
  [[ "$OWNER_FINGERPRINT" =~ ^[0-9a-f]{64}$ ]]
}

read_pid_metadata() {
  local pid_path="$1"
  local pid_line start_line owner_line extra_line
  {
    IFS= read -r pid_line || return 1
    IFS= read -r start_line || return 1
    IFS= read -r owner_line || return 1
    if IFS= read -r extra_line; then
      return 1
    fi
  } < "$pid_path"
  [[ "$pid_line" =~ ^pid=([1-9][0-9]*)$ ]] || return 1
  METADATA_PID="${BASH_REMATCH[1]}"
  [[ "$start_line" =~ ^start_ticks=([1-9][0-9]*)$ ]] || return 1
  METADATA_START_TICKS="${BASH_REMATCH[1]}"
  [[ "$owner_line" =~ ^owner_fingerprint=([0-9a-f]{64})$ ]] || return 1
  METADATA_OWNER_FINGERPRINT="${BASH_REMATCH[1]}"
}

read_process_start_ticks() {
  local pid="$1"
  local stat_path="$PROC_ROOT/$pid/stat"
  local stat_line stat_tail
  [[ -r "$stat_path" ]] || return 1
  IFS= read -r stat_line < "$stat_path" || return 1
  [[ "$stat_line" == *") "* ]] || return 1
  stat_tail="${stat_line##*) }"
  set -- $stat_tail
  [[ $# -ge 20 && "${20}" =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s\n' "${20}"
}

pid_metadata_is_owned_live() {
  local pid_path="$1"
  local expected_owner="$2"
  local current_start_ticks
  read_pid_metadata "$pid_path" || return 1
  [[ "$METADATA_OWNER_FINGERPRINT" == "$expected_owner" ]] || return 1
  current_start_ticks="$(read_process_start_ticks "$METADATA_PID")" || return 1
  [[ "$current_start_ticks" == "$METADATA_START_TICKS" ]] || return 1
}

capture_process_start_ticks() {
  local pid="$1"
  local attempt start_ticks
  for ((attempt = 0; attempt < 100; attempt++)); do
    if start_ticks="$(read_process_start_ticks "$pid")"; then
      printf '%s\n' "$start_ticks"
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 0.01
  done
  return 1
}

FLOCK_FDS=()
FALLBACK_LOCKS=()
FALLBACK_LOCK_FINGERPRINTS=()
LAUNCHED_PIDS=()
PUBLISHED_PID_PATHS=()
PUBLISHED_PIDS=()
PUBLISHED_START_TICKS=()
PUBLISHED_FINGERPRINTS=()
LOCK_OWNER_TEMP=""
PID_METADATA_TEMP=""
LAUNCHER_START_TICKS=""
TRANSACTION_COMMITTED=0

write_pid_metadata_atomic() {
  local pid_path="$1"
  local pid="$2"
  local start_ticks="$3"
  local owner_fingerprint="$4"
  PID_METADATA_TEMP="${pid_path}.tmp.$$"
  (
    umask 077
    printf 'pid=%s\nstart_ticks=%s\nowner_fingerprint=%s\n' \
      "$pid" "$start_ticks" "$owner_fingerprint" > "$PID_METADATA_TEMP"
  ) || return 1
  mv -f -- "$PID_METADATA_TEMP" "$pid_path" || return 1
  PID_METADATA_TEMP=""
}

flock_is_available() {
  command -v flock >/dev/null 2>&1 && flock --version >/dev/null 2>&1
}

close_flock_fd() {
  local lock_fd="$1"
  eval "exec ${lock_fd}>&-"
}

acquire_flock_locks() {
  local array_size lock_fd lock_path
  for array_size in "${ARRAYS[@]}"; do
    lock_path="$RESULT_ROOT/pids/train_${array_size}.flock"
    exec {lock_fd}>> "$lock_path" || return 1
    if ! flock -n "$lock_fd"; then
      close_flock_fd "$lock_fd"
      echo "training launch lock is already held for $array_size: $lock_path" >&2
      return 1
    fi
    FLOCK_FDS+=("$lock_fd")
  done
}

fallback_lock_owner_is_active() {
  local owner_path="$1"
  local current_start_ticks
  read_pid_metadata "$owner_path" || return 1
  current_start_ticks="$(read_process_start_ticks "$METADATA_PID")" || return 1
  [[ "$current_start_ticks" == "$METADATA_START_TICKS" ]]
}

wait_for_fallback_lock_owner() {
  local owner_path="$1"
  local attempt
  for ((attempt = 0; attempt < 50; attempt++)); do
    [[ -f "$owner_path" ]] && return 0
    sleep 0.01
  done
  return 1
}

reclaim_stale_fallback_lock() {
  local lock_path="$1"
  local owner_path="$lock_path/owner"
  local owner_snapshot current_snapshot
  if [[ -f "$owner_path" ]]; then
    fallback_lock_owner_is_active "$owner_path" && return 1
    owner_snapshot="$(<"$owner_path")"
    current_snapshot="$(<"$owner_path")"
    [[ "$current_snapshot" == "$owner_snapshot" ]] || return 1
    rm -f -- "$owner_path" || return 1
  fi
  rmdir -- "$lock_path" 2>/dev/null
}

acquire_one_fallback_lock() {
  local lock_path="$1"
  local owner_fingerprint="$2"
  local owner_path="$lock_path/owner"
  local attempt
  for ((attempt = 0; attempt < 3; attempt++)); do
    LOCK_OWNER_TEMP="${lock_path}.owner.tmp.$$"
    (
      umask 077
      printf 'pid=%s\nstart_ticks=%s\nowner_fingerprint=%s\n' \
        "$$" "$LAUNCHER_START_TICKS" "$owner_fingerprint" > "$LOCK_OWNER_TEMP"
    ) || return 1
    if mkdir -- "$lock_path" 2>/dev/null; then
      if ln -- "$LOCK_OWNER_TEMP" "$owner_path" 2>/dev/null; then
        rm -f -- "$LOCK_OWNER_TEMP"
        LOCK_OWNER_TEMP=""
        FALLBACK_LOCKS+=("$lock_path")
        FALLBACK_LOCK_FINGERPRINTS+=("$owner_fingerprint")
        return 0
      fi
      rm -f -- "$LOCK_OWNER_TEMP"
      LOCK_OWNER_TEMP=""
      rmdir -- "$lock_path" 2>/dev/null || true
      return 1
    fi
    rm -f -- "$LOCK_OWNER_TEMP"
    LOCK_OWNER_TEMP=""
    if wait_for_fallback_lock_owner "$owner_path" && \
      fallback_lock_owner_is_active "$owner_path"; then
      return 1
    fi
    reclaim_stale_fallback_lock "$lock_path" || return 1
  done
  return 1
}

acquire_fallback_locks() {
  local index array_size lock_path
  LAUNCHER_START_TICKS="$(read_process_start_ticks "$$")" || {
    echo "could not capture fallback lock owner identity for launcher PID $$" >&2
    return 1
  }
  for index in "${!ARRAYS[@]}"; do
    array_size="${ARRAYS[$index]}"
    lock_path="$RESULT_ROOT/pids/train_${array_size}.lock"
    if ! acquire_one_fallback_lock "$lock_path" "${OWNER_FINGERPRINTS[$index]}"; then
      echo "training launch lock is already held for $array_size: $lock_path" >&2
      return 1
    fi
  done
}

release_fallback_locks() {
  local index lock_path owner_path expected_fingerprint
  for ((index = ${#FALLBACK_LOCKS[@]} - 1; index >= 0; index--)); do
    lock_path="${FALLBACK_LOCKS[$index]}"
    owner_path="$lock_path/owner"
    expected_fingerprint="${FALLBACK_LOCK_FINGERPRINTS[$index]}"
    if read_pid_metadata "$owner_path" && \
      [[ "$METADATA_PID" == "$$" ]] && \
      [[ "$METADATA_START_TICKS" == "$LAUNCHER_START_TICKS" ]] && \
      [[ "$METADATA_OWNER_FINGERPRINT" == "$expected_fingerprint" ]]; then
      rm -f -- "$owner_path"
      rmdir -- "$lock_path" 2>/dev/null || true
    fi
  done
  FALLBACK_LOCKS=()
  FALLBACK_LOCK_FINGERPRINTS=()
}

release_flock_locks() {
  local index
  for ((index = ${#FLOCK_FDS[@]} - 1; index >= 0; index--)); do
    close_flock_fd "${FLOCK_FDS[$index]}"
  done
  FLOCK_FDS=()
}

remove_matching_published_pid_files() {
  local index pid_path
  for ((index = ${#PUBLISHED_PID_PATHS[@]} - 1; index >= 0; index--)); do
    pid_path="${PUBLISHED_PID_PATHS[$index]}"
    if read_pid_metadata "$pid_path" && \
      [[ "$METADATA_PID" == "${PUBLISHED_PIDS[$index]}" ]] && \
      [[ "$METADATA_START_TICKS" == "${PUBLISHED_START_TICKS[$index]}" ]] && \
      [[ "$METADATA_OWNER_FINGERPRINT" == "${PUBLISHED_FINGERPRINTS[$index]}" ]]; then
      rm -f -- "$pid_path"
    fi
  done
}

rollback_launched_processes() {
  local pid
  for pid in "${LAUNCHED_PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  if [[ ${#LAUNCHED_PIDS[@]} -gt 0 ]]; then
    sleep 0.05
  fi
  for pid in "${LAUNCHED_PIDS[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
  for pid in "${LAUNCHED_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}

finish_train_transaction() {
  local status="$?"
  set +e
  trap - EXIT HUP INT TERM
  if [[ "$TRANSACTION_COMMITTED" -ne 1 ]]; then
    rollback_launched_processes
    remove_matching_published_pid_files
    if [[ -n "$PID_METADATA_TEMP" ]]; then
      rm -f -- "$PID_METADATA_TEMP"
      PID_METADATA_TEMP=""
    fi
  fi
  if [[ -n "$LOCK_OWNER_TEMP" ]]; then
    rm -f -- "$LOCK_OWNER_TEMP"
    LOCK_OWNER_TEMP=""
  fi
  release_fallback_locks
  release_flock_locks
  exit "$status"
}

if [[ "$MODE" == "--dry-run" ]]; then
  for index in "${!ARRAYS[@]}"; do
    print_train_command "${ARRAYS[$index]}" "${GPUS[$index]}"
  done
  exit 0
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "environment bootstrap is missing: $ENV_FILE" >&2
  exit 1
fi
if [[ ! -d "$CODE_ROOT" ]]; then
  echo "multiscale UNO checkout is missing: $CODE_ROOT" >&2
  exit 1
fi

source "$ENV_FILE"
export RADIOFLOW_CODE="$CODE_ROOT"
cd "$CODE_ROOT"
mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/pids" "$RESULT_ROOT/runs"

if [[ "$MODE" == "--preflight" ]]; then
  for index in "${!ARRAYS[@]}"; do
    array_size="${ARRAYS[$index]}"
    gpu="${GPUS[$index]}"
    build_train_command "$array_size"
    CUDA_VISIBLE_DEVICES="$gpu" "${TRAIN_COMMAND[@]}" --preflight-only
  done
  exit 0
fi

if [[ "$MODE" == "--smoke" ]]; then
  smoke_pids=()
  for index in "${!ARRAYS[@]}"; do
    array_size="${ARRAYS[$index]}"
    gpu="${GPUS[$index]}"
    build_train_command "$array_size"
    log_path="$RESULT_ROOT/logs/smoke_${array_size}.log"
    CUDA_VISIBLE_DEVICES="$gpu" "${TRAIN_COMMAND[@]}" --smoke-optimizer-steps 1 \
      >"$log_path" 2>&1 &
    smoke_pids+=("$!")
    echo "SMOKE_LAUNCHED ARRAY=$array_size GPU=$gpu PID=$! LOG=$log_path"
  done
  failed=0
  for pid in "${smoke_pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "one or more multiscale UNO smoke runs failed; inspect smoke logs" >&2
    exit 1
  fi
  echo "SMOKE_COMPLETE arrays=3"
  exit 0
fi

if [[ "$MODE" == "--select-cfg" || "$MODE" == "--test" ]]; then
  command="${MODE#--}"
  for index in "${!ARRAYS[@]}"; do
    array_size="${ARRAYS[$index]}"
    gpu="${GPUS[$index]}"
    build_evaluation_command "$array_size" "$command"
    CUDA_VISIBLE_DEVICES="$gpu" "${EVALUATION_COMMAND[@]}"
  done
  exit 0
fi

OWNER_FINGERPRINTS=()
for index in "${!ARRAYS[@]}"; do
  array_size="${ARRAYS[$index]}"
  build_owner_fingerprint "$array_size"
  OWNER_FINGERPRINTS[$index]="$OWNER_FINGERPRINT"
done

trap finish_train_transaction EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if flock_is_available; then
  acquire_flock_locks || exit 1
else
  acquire_fallback_locks || exit 1
fi

for index in "${!ARRAYS[@]}"; do
  array_size="${ARRAYS[$index]}"
  pid_path="$RESULT_ROOT/pids/train_${array_size}.pid"
  OWNER_FINGERPRINT="${OWNER_FINGERPRINTS[$index]}"
  if [[ -f "$pid_path" ]]; then
    if pid_metadata_is_owned_live "$pid_path" "$OWNER_FINGERPRINT"; then
      echo "training is already live for $array_size at PID $METADATA_PID" >&2
      exit 1
    fi
    rm -f -- "$pid_path"
  fi
done

for index in "${!ARRAYS[@]}"; do
  array_size="${ARRAYS[$index]}"
  gpu="${GPUS[$index]}"
  build_train_command "$array_size"
  log_path="$RESULT_ROOT/logs/train_${array_size}.log"
  pid_path="$RESULT_ROOT/pids/train_${array_size}.pid"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "${TRAIN_COMMAND[@]}" \
    >"$log_path" 2>&1 </dev/null &
  pid="$!"
  LAUNCHED_PIDS+=("$pid")
  if ! start_ticks="$(capture_process_start_ticks "$pid")"; then
    echo "could not capture process birth identity for $array_size at PID $pid" >&2
    exit 1
  fi
  if ! write_pid_metadata_atomic \
    "$pid_path" "$pid" "$start_ticks" "${OWNER_FINGERPRINTS[$index]}"; then
    echo "could not publish PID metadata for $array_size at PID $pid" >&2
    exit 1
  fi
  PUBLISHED_PID_PATHS+=("$pid_path")
  PUBLISHED_PIDS+=("$pid")
  PUBLISHED_START_TICKS+=("$start_ticks")
  PUBLISHED_FINGERPRINTS+=("${OWNER_FINGERPRINTS[$index]}")
  echo "LAUNCHED ARRAY=$array_size GPU=$gpu PID=$pid LOG=$log_path"
done
TRANSACTION_COMMITTED=1
