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
  printf 'TRAIN ARRAY=%s CUDA_VISIBLE_DEVICES=%s ' "$array_size" "$gpu"
  printf '%q ' "${TRAIN_COMMAND[@]}"
  printf '\n'
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

for index in "${!ARRAYS[@]}"; do
  array_size="${ARRAYS[$index]}"
  gpu="${GPUS[$index]}"
  build_train_command "$array_size"
  log_path="$RESULT_ROOT/logs/train_${array_size}.log"
  pid_path="$RESULT_ROOT/pids/train_${array_size}.pid"
  if [[ -f "$pid_path" ]]; then
    existing_pid="$(<"$pid_path")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "training is already live for $array_size at PID $existing_pid" >&2
      exit 1
    fi
  fi
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "${TRAIN_COMMAND[@]}" \
    >"$log_path" 2>&1 </dev/null &
  pid="$!"
  printf '%s\n' "$pid" >"$pid_path"
  echo "LAUNCHED ARRAY=$array_size GPU=$gpu PID=$pid LOG=$log_path"
done
