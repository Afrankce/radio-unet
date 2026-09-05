#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "train" && "$MODE" != "evaluate" ]]; then
  echo "usage: $0 {train|evaluate}" >&2
  exit 2
fi

: "${DATASET_ROOT:?set DATASET_ROOT to the extracted dataset root}"
: "${MANIFEST_DIR:?set MANIFEST_DIR to the manifest directory}"
: "${HEIGHT_STATS:?set HEIGHT_STATS to height_stats_train.json}"
: "${RUN_ROOT:?set RUN_ROOT to the beam-zero parent run directory}"
: "${RESULTS_ROOT:?set RESULTS_ROOT to the beam-zero parent result directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_8X8="${GPU_8X8:-0}"
GPU_16X16="${GPU_16X16:-1}"
GPU_32X32="${GPU_32X32:-2}"

arrays=("8x8" "16x16" "32x32")
gpus=("$GPU_8X8" "$GPU_16X16" "$GPU_32X32")
pids=()

for index in "${!arrays[@]}"; do
  array="${arrays[$index]}"
  gpu="${gpus[$index]}"
  manifest="$MANIFEST_DIR/manifest_samefreq_6.7ghz_${array}_0deg.jsonl"
  run_dir="$RUN_ROOT/$array"
  result_dir="$RESULTS_ROOT/$array"

  if [[ "$MODE" == "train" ]]; then
    mkdir -p "$run_dir"
    "$PYTHON_BIN" -u run_same_frequency_multiscale_uno.py train \
      --dataset-root "$DATASET_ROOT" \
      --manifest-path "$manifest" \
      --height-stats-path "$HEIGHT_STATS" \
      --run-root "$run_dir" \
      --array-size "$array" \
      --device "cuda:$gpu" \
      --condition-variant beam_zero \
      --resume auto \
      >"$run_dir/launcher.log" 2>&1 &
  else
    mkdir -p "$RESULTS_ROOT"
    (
      "$PYTHON_BIN" -u run_same_frequency_multiscale_uno.py select-cfg \
        --dataset-root "$DATASET_ROOT" \
        --manifest-path "$manifest" \
        --height-stats-path "$HEIGHT_STATS" \
        --run-root "$run_dir" \
        --results-root "$result_dir" \
        --array-size "$array" \
        --device "cuda:$gpu" \
        --condition-variant beam_zero
      "$PYTHON_BIN" -u run_same_frequency_multiscale_uno.py test \
        --dataset-root "$DATASET_ROOT" \
        --manifest-path "$manifest" \
        --height-stats-path "$HEIGHT_STATS" \
        --run-root "$run_dir" \
        --results-root "$result_dir" \
        --array-size "$array" \
        --device "cuda:$gpu" \
        --condition-variant beam_zero
    ) >"$RUN_ROOT/${array}_evaluation.log" 2>&1 &
  fi
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
