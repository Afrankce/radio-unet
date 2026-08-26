from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.multiconfig_manifest import canonical_json_bytes


ARRAY_SIZES = ("8x8", "16x16", "32x32")
UNET_LITE_DB_RMSE = {
    "8x8": 11.627,
    "16x16": 11.657,
    "32x32": 11.700,
}
EXPERIMENT = "same_frequency_6.7_single_beam_paper_fno"
MODEL_SIZE = "paper_fno_lite"


class FNOExperimentSummaryError(RuntimeError):
    """The registered three-array result set is missing or inconsistent."""


def _validated_fno_metrics(values: Mapping[str, float]) -> dict[str, float]:
    if set(values) != set(ARRAY_SIZES):
        raise FNOExperimentSummaryError(
            "FNO metrics must contain exactly 8x8, 16x16, and 32x32"
        )
    normalized: dict[str, float] = {}
    for array_size in ARRAY_SIZES:
        try:
            value = float(values[array_size])
        except (TypeError, ValueError) as error:
            raise FNOExperimentSummaryError(
                f"{array_size} dB-RMSE is not numeric"
            ) from error
        if not math.isfinite(value) or value < 0.0:
            raise FNOExperimentSummaryError(
                f"{array_size} dB-RMSE must be finite and non-negative"
            )
        normalized[array_size] = value
    return normalized


def apply_registered_decision(fno_rmse: Mapping[str, float]) -> dict[str, Any]:
    """Apply the preregistered mean/two-of-three/worst-array decision rule."""

    normalized = _validated_fno_metrics(fno_rmse)
    deltas = {
        array_size: UNET_LITE_DB_RMSE[array_size] - normalized[array_size]
        for array_size in ARRAY_SIZES
    }
    mean_delta = sum(deltas.values()) / len(ARRAY_SIZES)
    n_improved = sum(delta > 0.0 for delta in deltas.values())
    worst_delta = min(deltas.values())
    guards = {
        "mean_delta_at_least_0.3_db": mean_delta >= 0.3,
        "at_least_two_arrays_improved": n_improved >= 2,
        "no_array_regressed_more_than_0.5_db": worst_delta >= -0.5,
    }
    return {
        "delta_definition": "unet_lite_db_rmse_minus_fno_db_rmse",
        "delta_db": deltas,
        "mean_delta_db": mean_delta,
        "n_improved": n_improved,
        "worst_delta_db": worst_delta,
        "guards": guards,
        "h1_confirmed": all(guards.values()),
    }


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FNOExperimentSummaryError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise FNOExperimentSummaryError(f"{label} must be a JSON object")
    return payload


def collect_registered_results(results_root: str | Path) -> dict[str, float]:
    root = Path(results_root).resolve()
    expected_metrics = {
        (root / "test" / array_size / "metrics_test.json").resolve()
        for array_size in ARRAY_SIZES
    }
    observed_metrics = {path.resolve() for path in root.rglob("metrics_test.json")}
    if observed_metrics != expected_metrics:
        raise FNOExperimentSummaryError(
            "result root must contain exactly the three registered metrics_test.json paths"
        )

    collected: dict[str, float] = {}
    for array_size in ARRAY_SIZES:
        directory = root / "test" / array_size
        metrics = _read_json(directory / "metrics_test.json", "test metrics")
        manifest = _read_json(directory / "run_manifest.json", "run manifest")
        expected_common = {
            "experiment": EXPERIMENT,
            "array_size": array_size,
            "model_size": MODEL_SIZE,
            "split": "test",
            "n_samples": 160,
        }
        for key, expected in expected_common.items():
            if metrics.get(key) != expected or manifest.get(key) != expected:
                raise FNOExperimentSummaryError(
                    f"{array_size} metrics/manifest mismatch at {key}"
                )
        if manifest.get("status") != "complete":
            raise FNOExperimentSummaryError(
                f"{array_size} run manifest is not complete"
            )
        try:
            rmse = float(metrics["db_rmse"])
        except (KeyError, TypeError, ValueError) as error:
            raise FNOExperimentSummaryError(
                f"{array_size} metrics lack dB-RMSE"
            ) from error
        collected[array_size] = rmse
    return _validated_fno_metrics(collected)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_registered_summary(
    output_dir: str | Path,
    fno_rmse: Mapping[str, float],
) -> dict[str, Path]:
    normalized = _validated_fno_metrics(fno_rmse)
    decision = apply_registered_decision(normalized)
    output = Path(output_dir)
    json_path = output / "registered_decision.json"
    csv_path = output / "registered_comparison.csv"
    payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "model_size": MODEL_SIZE,
        "metric": "db_rmse",
        "unet_lite_db_rmse": UNET_LITE_DB_RMSE,
        "fno_db_rmse": normalized,
        "decision": decision,
    }
    _atomic_write(json_path, canonical_json_bytes(payload))

    rows = [
        {
            "array_size": array_size,
            "unet_lite_db_rmse": UNET_LITE_DB_RMSE[array_size],
            "fno_db_rmse": normalized[array_size],
            "delta_db": decision["delta_db"][array_size],
            "improved": decision["delta_db"][array_size] > 0.0,
        }
        for array_size in ARRAY_SIZES
    ]
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, csv_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"json": json_path, "csv": csv_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the registered three-array paper-FNO decision rule"
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    metrics = collect_registered_results(arguments.results_root)
    outputs = write_registered_summary(arguments.output_dir, metrics)
    result = {
        "metrics": metrics,
        "decision": apply_registered_decision(metrics),
        "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
    }
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
