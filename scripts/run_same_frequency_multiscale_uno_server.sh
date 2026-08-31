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
PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST_ROOT="$DATASET_ROOT/manifests"
HEIGHT_STATS="$MANIFEST_ROOT/height_stats_train.json"

ARRAYS=(8x8 16x16 32x32)
GPUS=(0 1 2)

build_train_command() {
  local array_size="$1"
  TRAIN_COMMAND=(
    "$PYTHON_BIN" -u "$CODE_ROOT/run_same_frequency_multiscale_uno.py" train
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
    "$PYTHON_BIN" -u "$CODE_ROOT/run_same_frequency_multiscale_uno.py" "$command"
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
CHILD_ROLLBACK_RECORDS=()
CHILD_RECORD_SEPARATOR=$'\034'
SPAWN_RECORD_CRITICAL=0
PENDING_SIGNAL_STATUS=0
TRANSACTION_COMMITTED=0

write_pid_metadata_atomic() {
  local pid_path="$1"
  local temp_path="$2"
  local pid="$3"
  local start_ticks="$4"
  local owner_fingerprint="$5"
  (
    umask 077
    printf 'pid=%s\nstart_ticks=%s\nowner_fingerprint=%s\n' \
      "$pid" "$start_ticks" "$owner_fingerprint" > "$temp_path"
  ) || return 1
  mv -f -- "$temp_path" "$pid_path" || return 1
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

release_flock_locks() {
  local index
  for ((index = ${#FLOCK_FDS[@]} - 1; index >= 0; index--)); do
    close_flock_fd "${FLOCK_FDS[$index]}"
  done
  FLOCK_FDS=()
}

close_flock_fds_for_child() {
  local lock_fd
  for lock_fd in "${FLOCK_FDS[@]}"; do
    close_flock_fd "$lock_fd"
  done
}

read_child_rollback_record() {
  local record="$1"
  local extra
  IFS="$CHILD_RECORD_SEPARATOR" read -r \
    CHILD_PID CHILD_PID_PATH CHILD_TEMP_PATH CHILD_OWNER_FINGERPRINT \
    CHILD_START_TICKS extra <<< "$record"
  [[ "$CHILD_PID" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -n "$CHILD_PID_PATH" && -n "$CHILD_TEMP_PATH" ]] || return 1
  [[ "$CHILD_OWNER_FINGERPRINT" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ "$CHILD_START_TICKS" == unknown || \
    "$CHILD_START_TICKS" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -z "$extra" ]]
}

set_child_record_start_ticks() {
  local index="$1"
  local start_ticks="$2"
  read_child_rollback_record "${CHILD_ROLLBACK_RECORDS[$index]}" || return 1
  CHILD_ROLLBACK_RECORDS[$index]="${CHILD_PID}${CHILD_RECORD_SEPARATOR}\
${CHILD_PID_PATH}${CHILD_RECORD_SEPARATOR}${CHILD_TEMP_PATH}\
${CHILD_RECORD_SEPARATOR}${CHILD_OWNER_FINGERPRINT}\
${CHILD_RECORD_SEPARATOR}${start_ticks}"
}

child_pid_is_active_job() {
  local expected_pid="$1"
  local active_pid
  while IFS= read -r active_pid; do
    if [[ "$active_pid" == "$expected_pid" ]]; then
      return 0
    fi
  done < <(jobs -pr)
  return 1
}

child_process_matches_record() {
  local current_start_ticks
  if [[ "$CHILD_START_TICKS" == unknown ]]; then
    child_pid_is_active_job "$CHILD_PID"
    return
  fi
  current_start_ticks="$(read_process_start_ticks "$CHILD_PID")" || return 1
  [[ "$current_start_ticks" == "$CHILD_START_TICKS" ]]
}

rollback_tracked_children() {
  local record
  for record in "${CHILD_ROLLBACK_RECORDS[@]}"; do
    read_child_rollback_record "$record" || continue
    if child_process_matches_record; then
      kill -TERM "$CHILD_PID" 2>/dev/null || true
    fi
  done
  if [[ ${#CHILD_ROLLBACK_RECORDS[@]} -gt 0 ]]; then
    sleep 0.05
  fi
  for record in "${CHILD_ROLLBACK_RECORDS[@]}"; do
    read_child_rollback_record "$record" || continue
    if child_process_matches_record; then
      kill -KILL "$CHILD_PID" 2>/dev/null || true
    fi
  done
  for record in "${CHILD_ROLLBACK_RECORDS[@]}"; do
    read_child_rollback_record "$record" || continue
    wait "$CHILD_PID" 2>/dev/null || true
  done
}

remove_matching_child_metadata() {
  local index record
  for ((index = ${#CHILD_ROLLBACK_RECORDS[@]} - 1; index >= 0; index--)); do
    record="${CHILD_ROLLBACK_RECORDS[$index]}"
    read_child_rollback_record "$record" || continue
    if [[ "$CHILD_START_TICKS" != unknown ]] && \
      [[ -f "$CHILD_PID_PATH" ]] && \
      read_pid_metadata "$CHILD_PID_PATH" && \
      [[ "$METADATA_PID" == "$CHILD_PID" ]] && \
      [[ "$METADATA_START_TICKS" == "$CHILD_START_TICKS" ]] && \
      [[ "$METADATA_OWNER_FINGERPRINT" == "$CHILD_OWNER_FINGERPRINT" ]]; then
      rm -f -- "$CHILD_PID_PATH"
    fi
    rm -f -- "$CHILD_TEMP_PATH"
  done
}

handle_train_signal() {
  local status="$1"
  if [[ "$SPAWN_RECORD_CRITICAL" -eq 1 ]]; then
    if [[ "$PENDING_SIGNAL_STATUS" -eq 0 ]]; then
      PENDING_SIGNAL_STATUS="$status"
    fi
    return 0
  fi
  exit "$status"
}

dispatch_pending_train_signal() {
  local status="$PENDING_SIGNAL_STATUS"
  PENDING_SIGNAL_STATUS=0
  if [[ "$status" -ne 0 ]]; then
    exit "$status"
  fi
}

finish_train_transaction() {
  trap '' HUP INT TERM
  local status="$1"
  set +e
  set +u
  trap - EXIT
  SPAWN_RECORD_CRITICAL=0
  if [[ "$TRANSACTION_COMMITTED" -ne 1 ]]; then
    rollback_tracked_children
    remove_matching_child_metadata
  fi
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

if ! command -v flock >/dev/null 2>&1; then
  echo "util-linux flock is required for --train" >&2
  exit 1
fi

OWNER_FINGERPRINTS=()
for index in "${!ARRAYS[@]}"; do
  array_size="${ARRAYS[$index]}"
  build_owner_fingerprint "$array_size"
  OWNER_FINGERPRINTS[$index]="$OWNER_FINGERPRINT"
done

trap 'finish_train_transaction "$?"' EXIT
trap 'handle_train_signal 129' HUP
trap 'handle_train_signal 130' INT
trap 'handle_train_signal 143' TERM

acquire_flock_locks || exit 1

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
  pid_temp_path="${pid_path}.tmp.$$.$index"
  owner_fingerprint="${OWNER_FINGERPRINTS[$index]}"
  SPAWN_RECORD_CRITICAL=1
  (
    close_flock_fds_for_child
    exec nohup env CUDA_VISIBLE_DEVICES="$gpu" "${TRAIN_COMMAND[@]}"
  ) >"$log_path" 2>&1 </dev/null &
  pid="$!"
  record_index="${#CHILD_ROLLBACK_RECORDS[@]}"
  CHILD_ROLLBACK_RECORDS+=(
    "${pid}${CHILD_RECORD_SEPARATOR}${pid_path}${CHILD_RECORD_SEPARATOR}\
${pid_temp_path}${CHILD_RECORD_SEPARATOR}${owner_fingerprint}\
${CHILD_RECORD_SEPARATOR}unknown"
  )
  SPAWN_RECORD_CRITICAL=0
  dispatch_pending_train_signal
  if ! start_ticks="$(capture_process_start_ticks "$pid")"; then
    echo "could not capture process birth identity for $array_size at PID $pid" >&2
    exit 1
  fi
  set_child_record_start_ticks "$record_index" "$start_ticks" || exit 1
  if ! write_pid_metadata_atomic \
    "$pid_path" "$pid_temp_path" "$pid" "$start_ticks" "$owner_fingerprint"; then
    echo "could not publish PID metadata for $array_size at PID $pid" >&2
    exit 1
  fi
  echo "LAUNCHED ARRAY=$array_size GPU=$gpu PID=$pid LOG=$log_path"
done
TRANSACTION_COMMITTED=1
