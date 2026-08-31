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

ACQUIRED_LOCKS=()
PID_METADATA_TEMP=""

release_launch_locks() {
  local index
  if [[ -n "$PID_METADATA_TEMP" ]]; then
    rm -f -- "$PID_METADATA_TEMP"
    PID_METADATA_TEMP=""
  fi
  for ((index = ${#ACQUIRED_LOCKS[@]} - 1; index >= 0; index--)); do
    rmdir -- "${ACQUIRED_LOCKS[$index]}" 2>/dev/null || true
  done
  ACQUIRED_LOCKS=()
}

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
  )
  mv -f -- "$PID_METADATA_TEMP" "$pid_path"
  PID_METADATA_TEMP=""
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

trap release_launch_locks EXIT
for array_size in "${ARRAYS[@]}"; do
  lock_path="$RESULT_ROOT/pids/train_${array_size}.lock"
  if ! mkdir -- "$lock_path"; then
    echo "training launch lock is already held for $array_size: $lock_path" >&2
    exit 1
  fi
  ACQUIRED_LOCKS+=("$lock_path")
done

OWNER_FINGERPRINTS=()
for index in "${!ARRAYS[@]}"; do
  array_size="${ARRAYS[$index]}"
  pid_path="$RESULT_ROOT/pids/train_${array_size}.pid"
  build_owner_fingerprint "$array_size"
  OWNER_FINGERPRINTS[$index]="$OWNER_FINGERPRINT"
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
  if ! start_ticks="$(capture_process_start_ticks "$pid")"; then
    echo "could not capture process birth identity for $array_size at PID $pid" >&2
    kill "$pid" 2>/dev/null || true
    exit 1
  fi
  write_pid_metadata_atomic \
    "$pid_path" "$pid" "$start_ticks" "${OWNER_FINGERPRINTS[$index]}"
  echo "LAUNCHED ARRAY=$array_size GPU=$gpu PID=$pid LOG=$log_path"
done
