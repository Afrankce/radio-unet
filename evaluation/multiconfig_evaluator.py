from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import shutil
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from data_loaders.multiconfig import multiconfig_collate
from evaluation.radioflow_sampling import euler_cfg_sample, make_sample_noise
from evaluation.radiomap_metrics import (
    MetricAccumulator,
    PerBeamMetricAccumulators,
    metrics_for_json,
)
from evaluation.runtime_benchmark import benchmark_generation
from evaluation.visualization import (
    render_comparison,
    render_error_map,
    save_prediction_npz,
    stable_case_stem,
)
from experiments.multiconfig_manifest import canonical_json_bytes
from experiments.provenance import sha256_file
from training.checkpointing import (
    CheckpointIdentity,
    TrainerState,
    load_ema_for_evaluation,
)
from training.config import MultiConfigTrainConfig, TEST_SAMPLES, VAL_SAMPLES
from training.hardware_evidence import (
    LargeHardwareGateContext,
    validate_large_hardware_gate,
)
from training.model_factory import EXPECTED_PARAMETER_COUNTS, build_locked_radioflow
from training.multiconfig_trainer import (
    BenchmarkContext,
    build_checkpoint_identity,
    preflight_benchmark,
    seed_everything,
    seed_worker,
)


CFG_CANDIDATES = (1.0, 1.5, 2.0, 2.5)
COMMON_ANGLES_DEG = (-28.0, -21.0, -14.0, -7.0, 0.0, 7.0, 14.0, 21.0)
VISUALIZATION_ANGLES_DEG = (-28.0, -7.0, 7.0, 21.0)
PER_BEAM_COLUMNS = (
    "angle_deg",
    "beam_id",
    "n_samples",
    "n_valid_pixels",
    "db_rmse",
    "db_mae",
    "mse",
    "nmse",
    "psnr",
    "ssim",
)
ARRAY_NAMES = ("8x8", "16x16", "32x32")
MODEL_SIZES = ("lite", "large")
SUMMARY_COLUMNS = (
    "array_size",
    "model_size",
    "status",
    "selected_epoch",
    "best_validation_db_rmse_cfg1",
    "selected_cfg_scale",
    "selected_validation_db_rmse",
    "db_rmse",
    "db_mae",
    "mse",
    "nmse",
    "psnr",
    "ssim",
    "parameter_count",
    "checkpoint_size_bytes",
    "peak_training_allocated_bytes",
    "peak_inference_allocated_bytes",
    "latency_ms_p50",
    "latency_ms_p95",
    "hardware_gate_sha256",
)
ANGLE_SUMMARY_COLUMNS = (
    "array_size",
    "model_size",
    "status",
    *PER_BEAM_COLUMNS,
    "hardware_gate_sha256",
)
SELECTION_IDENTITY_KEYS = {
    "checkpoint_sha256",
    "config_sha256",
    "manifest_sha256",
    "split_sha256",
    "schema_sha256",
    "archive_sha256",
    "dataset_revision",
    "radioflow_upstream_base",
    "git_commit",
}


class EvaluationContractError(RuntimeError):
    """Evaluation input or output violates the frozen one-time protocol."""


def _validate_selection_identity(identity: Mapping[str, Any]) -> dict[str, str]:
    if set(identity) != SELECTION_IDENTITY_KEYS:
        raise EvaluationContractError("CFG selection identity keys mismatch")
    normalized = {key: str(value) for key, value in identity.items()}
    for key, value in normalized.items():
        length = 40 if key in {
            "dataset_revision",
            "radioflow_upstream_base",
            "git_commit",
        } else 64
        if len(value) != length or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise EvaluationContractError(
                f"CFG selection identity {key} must be {length} lowercase hex digits"
            )
    return normalized


def select_cfg_candidate(
    candidate_metrics: Mapping[float, Mapping[str, Any]],
) -> float:
    if set(candidate_metrics) != set(CFG_CANDIDATES):
        raise EvaluationContractError(
            "CFG candidate grid must be exactly "
            f"{list(CFG_CANDIDATES)}, got {sorted(candidate_metrics)}"
        )
    ranked: list[tuple[float, float]] = []
    for scale in CFG_CANDIDATES:
        metrics = candidate_metrics[scale]
        try:
            rmse = float(metrics["db_rmse"])
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationContractError(
                f"CFG {scale} has no valid db_rmse"
            ) from error
        if not math.isfinite(rmse) or rmse < 0.0:
            raise EvaluationContractError(f"CFG {scale} db_rmse is invalid: {rmse}")
        ranked.append((rmse, scale))
    return min(ranked)[1]


def build_cfg_selection_payload(
    *,
    array_size: str,
    model_size: str,
    selected_epoch: int,
    candidate_metrics: Mapping[float, Mapping[str, Any]],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    selected_scale = select_cfg_candidate(candidate_metrics)
    if selected_epoch <= 0:
        raise EvaluationContractError("selected_epoch must be positive")
    locked_identity = _validate_selection_identity(identity)
    serialized_metrics = {
        f"{scale:.1f}": metrics_for_json(candidate_metrics[scale])
        for scale in CFG_CANDIDATES
    }
    return {
        "schema_version": 1,
        "array_size": array_size,
        "model_size": model_size,
        "candidates": list(CFG_CANDIDATES),
        "candidate_metrics": serialized_metrics,
        "selected_epoch": selected_epoch,
        "best_validation_db_rmse_cfg1": float(
            candidate_metrics[1.0]["db_rmse"]
        ),
        "selected_scale": selected_scale,
        "selected_validation_db_rmse": float(
            candidate_metrics[selected_scale]["db_rmse"]
        ),
        "tie_break_rule": "minimum_db_rmse_then_smaller_cfg",
        "solver": "euler",
        "euler_steps": 2,
        "ema": True,
        "identity": locked_identity,
    }


def _completed_manifest(path: Path | None) -> bool:
    if path is None or not Path(path).is_file():
        return False
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationContractError(
            f"cannot read existing test completion manifest {path}: {error}"
        ) from error
    return isinstance(payload, dict) and payload.get("status") == "complete"


def freeze_cfg_selection(
    path: Path,
    payload: Mapping[str, Any],
    *,
    completed_manifest_path: Path | None = None,
) -> str:
    path = Path(path)
    expected = canonical_json_bytes(payload)
    if _completed_manifest(completed_manifest_path) and not path.is_file():
        raise EvaluationContractError(
            "a completed test manifest forbids creating or changing CFG selection"
        )
    if path.exists():
        if path.read_bytes() != expected:
            raise EvaluationContractError(
                f"CFG selection already exists with different bytes: {path}"
            )
        return hashlib.sha256(expected).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as output:
            output.write(expected)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(expected).hexdigest()


def load_cfg_selection(
    path: Path,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationContractError(f"cannot read CFG selection {path}: {error}") from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise EvaluationContractError("CFG selection must be a canonical JSON object")
    required = {
        "schema_version",
        "array_size",
        "model_size",
        "candidates",
        "candidate_metrics",
        "selected_epoch",
        "best_validation_db_rmse_cfg1",
        "selected_scale",
        "selected_validation_db_rmse",
        "tie_break_rule",
        "solver",
        "euler_steps",
        "ema",
        "identity",
    }
    if set(payload) != required:
        raise EvaluationContractError("CFG selection keys mismatch")
    if payload["schema_version"] != 1 or payload["candidates"] != list(CFG_CANDIDATES):
        raise EvaluationContractError("CFG selection fixed protocol mismatch")
    if payload["selected_scale"] not in CFG_CANDIDATES:
        raise EvaluationContractError("CFG selection scale is outside the fixed grid")
    if payload["tie_break_rule"] != "minimum_db_rmse_then_smaller_cfg":
        raise EvaluationContractError("CFG selection tie-break rule mismatch")
    if payload["solver"] != "euler" or payload["euler_steps"] != 2 or payload["ema"] is not True:
        raise EvaluationContractError("CFG selection generation protocol mismatch")
    locked_identity = _validate_selection_identity(expected_identity)
    if payload["identity"] != locked_identity:
        raise EvaluationContractError("CFG selection identity mismatch")
    candidate_payload = payload["candidate_metrics"]
    if not isinstance(candidate_payload, dict):
        raise EvaluationContractError("CFG selection candidate metrics must be an object")
    expected_candidate_keys = {f"{scale:.1f}" for scale in CFG_CANDIDATES}
    if set(candidate_payload) != expected_candidate_keys:
        raise EvaluationContractError("CFG selection candidate metric grid mismatch")
    candidate_metrics: dict[float, Mapping[str, Any]] = {}
    for scale in CFG_CANDIDATES:
        metrics = candidate_payload[f"{scale:.1f}"]
        if not isinstance(metrics, Mapping):
            raise EvaluationContractError(f"CFG {scale} metrics must be an object")
        if metrics.get("n_samples") != VAL_SAMPLES:
            raise EvaluationContractError(
                f"CFG {scale} must contain exactly {VAL_SAMPLES} validation samples"
            )
        _validate_accuracy_values(metrics, label=f"CFG {scale} validation metrics")
        candidate_metrics[scale] = metrics
    recomputed = select_cfg_candidate(candidate_metrics)
    if float(payload["selected_scale"]) != recomputed:
        raise EvaluationContractError(
            "CFG selection selected scale does not match its frozen metrics"
        )
    try:
        selected_epoch = int(payload["selected_epoch"])
        best_cfg1 = float(payload["best_validation_db_rmse_cfg1"])
        selected_rmse = float(payload["selected_validation_db_rmse"])
    except (TypeError, ValueError) as error:
        raise EvaluationContractError("CFG selection summary values are invalid") from error
    if selected_epoch <= 0:
        raise EvaluationContractError("CFG selection epoch must be positive")
    if not math.isclose(
        best_cfg1,
        float(candidate_metrics[1.0]["db_rmse"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise EvaluationContractError("CFG 1.0 validation summary mismatch")
    if not math.isclose(
        selected_rmse,
        float(candidate_metrics[recomputed]["db_rmse"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise EvaluationContractError("selected CFG validation summary mismatch")
    return payload


def validate_test_metric_counts(
    overall: Mapping[str, Any],
    per_beam_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed unless the complete fixed 160-scene by eight-beam test ran."""

    try:
        overall_samples = int(overall["n_samples"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationContractError("overall test metrics lack n_samples") from error
    if overall_samples != 1_280:
        raise EvaluationContractError(
            f"test evaluation must contain exactly 1280 samples, got {overall_samples}"
        )
    if len(per_beam_rows) != 8:
        raise EvaluationContractError(
            f"test evaluation must contain eight beam rows, got {len(per_beam_rows)}"
        )
    beam_ids: set[int] = set()
    angles: list[float] = []
    row_total = 0
    for index, row in enumerate(per_beam_rows):
        missing = set(PER_BEAM_COLUMNS) - set(row)
        if missing:
            raise EvaluationContractError(
                f"per-beam row {index} is missing columns {sorted(missing)}"
            )
        try:
            beam_id = int(row["beam_id"])
            angle = float(row["angle_deg"])
            count = int(row["n_samples"])
        except (TypeError, ValueError) as error:
            raise EvaluationContractError(f"invalid per-beam row {index}") from error
        if beam_id in beam_ids:
            raise EvaluationContractError(f"duplicate beam ID {beam_id}")
        if count != 160:
            raise EvaluationContractError(
                f"each selected beam must contain exactly 160 samples; "
                f"beam {beam_id} has {count}"
            )
        beam_ids.add(beam_id)
        angles.append(angle)
        row_total += count
    if sorted(angles) != list(COMMON_ANGLES_DEG):
        raise EvaluationContractError(
            f"per-beam angles must equal {list(COMMON_ANGLES_DEG)}"
        )
    if row_total != overall_samples:
        raise EvaluationContractError("per-beam sample counts do not equal overall count")


def write_metrics_per_beam_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=PER_BEAM_COLUMNS,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                missing = set(PER_BEAM_COLUMNS) - set(row)
                if missing:
                    raise EvaluationContractError(
                        f"per-beam CSV row is missing {sorted(missing)}"
                    )
                serialized = {column: row[column] for column in PER_BEAM_COLUMNS}
                psnr = float(serialized["psnr"])
                if math.isinf(psnr) and psnr > 0.0:
                    serialized["psnr"] = ""
                writer.writerow(serialized)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def fixed_visualization_sample_keys(
    metadata: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Choose three shared test scenes crossed with four shared angles."""

    scenes: list[str] = []
    seen_scenes: set[str] = set()
    available: set[tuple[str, float]] = set()
    for index, item in enumerate(metadata):
        try:
            scene_id = str(item["scene_id"])
            angle = float(item["steering_deg"])
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationContractError(
                f"invalid visualization metadata at index {index}"
            ) from error
        if scene_id not in seen_scenes:
            scenes.append(scene_id)
            seen_scenes.add(scene_id)
        available.add((scene_id, angle))
    if len(scenes) != 160:
        raise EvaluationContractError(
            f"fixed visualizations require 160 test scenes, got {len(scenes)}"
        )
    chosen_scenes = (scenes[0], scenes[len(scenes) // 2], scenes[-1])
    selected: list[str] = []
    for scene_id in chosen_scenes:
        for angle in VISUALIZATION_ANGLES_DEG:
            if (scene_id, angle) not in available:
                raise EvaluationContractError(
                    f"visualization case is missing: {scene_id} at {angle} degrees"
                )
            selected.append(f"{scene_id}|{angle:.1f}")
    return tuple(selected)


def _atomic_canonical_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as output:
            output.write(canonical_json_bytes(payload))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_files(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise EvaluationContractError(f"result artifacts may not be symlinks: {path}")
        if not path.is_file() or path.name == "run_manifest.json":
            continue
        relative = path.relative_to(directory).as_posix()
        files[relative] = path
    return files


def write_run_manifest(
    staging_dir: str | Path,
    base_payload: Mapping[str, Any],
) -> Path:
    directory = Path(staging_dir)
    path = directory / "run_manifest.json"
    if path.exists():
        raise EvaluationContractError("run_manifest.json must be written exactly once")
    if "artifacts" in base_payload:
        raise EvaluationContractError("run manifest base must not predefine artifacts")
    if base_payload.get("schema_version") != 1 or base_payload.get("status") != "complete":
        raise EvaluationContractError("run manifest must publish schema 1 status complete")
    artifacts = {
        relative: sha256_file(artifact)
        for relative, artifact in _artifact_files(directory).items()
    }
    if not artifacts:
        raise EvaluationContractError("run manifest cannot publish an empty result")
    payload = {**dict(base_payload), "artifacts": artifacts}
    _atomic_canonical_json(path, payload)
    return path


def validate_run_manifest_artifacts(result_dir: str | Path) -> dict[str, Any]:
    directory = Path(result_dir)
    path = directory / "run_manifest.json"
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationContractError(f"cannot read run_manifest.json: {error}") from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise EvaluationContractError("run_manifest.json must be canonical JSON")
    if payload.get("schema_version") != 1 or payload.get("status") != "complete":
        raise EvaluationContractError("run_manifest.json is not a complete schema-1 receipt")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise EvaluationContractError("run_manifest.json has no artifact hashes")
    actual_files = _artifact_files(directory)
    if set(artifacts) != set(actual_files):
        raise EvaluationContractError(
            "run manifest artifact inventory mismatch: "
            f"missing={sorted(set(actual_files) - set(artifacts))}, "
            f"extra={sorted(set(artifacts) - set(actual_files))}"
        )
    for relative, artifact in actual_files.items():
        expected = artifacts[relative]
        if not isinstance(expected, str) or sha256_file(artifact) != expected:
            raise EvaluationContractError(f"artifact hash mismatch: {relative}")
    return payload


def atomic_result_transaction(
    final_dir: str | Path,
    builder: Callable[[Path], Any],
) -> Path:
    """Publish a test result only after its complete receipt validates."""

    final = Path(final_dir)
    if final.exists():
        raise EvaluationContractError(f"final result directory already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f"{final.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        builder(staging)
        validate_run_manifest_artifacts(staging)
        if final.exists():
            raise EvaluationContractError(
                f"final result directory appeared during evaluation: {final}"
            )
        os.replace(staging, final)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return final


@dataclass(frozen=True)
class PreparedEvaluation:
    cfg: MultiConfigTrainConfig
    context: BenchmarkContext
    model: torch.nn.Module
    device: torch.device
    checkpoint_path: Path
    checkpoint_identity: CheckpointIdentity
    selection_identity: Mapping[str, str]
    trainer_state: TrainerState


def _read_canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationContractError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise EvaluationContractError(f"{label} must be a canonical JSON object")
    return payload


def _validate_training_run_config(cfg: MultiConfigTrainConfig) -> None:
    path = cfg.run_dir / "config.json"
    try:
        stored = MultiConfigTrainConfig.from_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise EvaluationContractError(f"invalid completed training config {path}: {error}") from error
    comparisons = {
        "array_size": (stored.array_size, cfg.array_size),
        "model_size": (stored.model_size, cfg.model_size),
        "dataset_root": (stored.dataset_root.resolve(), cfg.dataset_root.resolve()),
        "manifest_dir": (stored.manifest_dir.resolve(), cfg.manifest_dir.resolve()),
        "run_root": (stored.run_root.resolve(), cfg.run_root.resolve()),
        "config_sha256": (stored.config_sha256, cfg.config_sha256),
    }
    for label, (actual, expected) in comparisons.items():
        if actual != expected:
            raise EvaluationContractError(
                f"training config {label} mismatch: expected {expected}, got {actual}"
            )
    runtime = _read_canonical_json(
        cfg.run_dir / "training_runtime.json", label="training runtime"
    )
    if runtime.get("status") != "complete":
        raise EvaluationContractError(
            "CFG/test evaluation requires a completed training run"
        )


def _selection_identity(
    checkpoint_path: Path,
    identity: CheckpointIdentity,
) -> dict[str, str]:
    return {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": identity.config_sha256,
        "manifest_sha256": identity.manifest_sha256,
        "split_sha256": identity.split_sha256,
        "schema_sha256": identity.schema_sha256,
        "archive_sha256": identity.archive_sha256,
        "dataset_revision": identity.dataset_revision,
        "radioflow_upstream_base": identity.radioflow_upstream_base,
        "git_commit": identity.git_commit,
    }


def _validate_best_checkpoint_state(state: TrainerState) -> tuple[int, float]:
    if state.completed_epochs <= 0 or not state.history:
        raise EvaluationContractError("best checkpoint has no completed validation epoch")
    selected_epoch = state.completed_epochs
    last = state.history[-1]
    try:
        history_epoch = int(last["epoch"])
        history_rmse = float(last["val_db_rmse"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationContractError("best checkpoint history is incomplete") from error
    if history_epoch != selected_epoch:
        raise EvaluationContractError("best checkpoint epoch and history disagree")
    if not math.isclose(
        history_rmse,
        float(state.best_val_db_rmse),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise EvaluationContractError(
            "best checkpoint does not end at its recorded best validation dB-RMSE"
        )
    return selected_epoch, history_rmse


def _prepare_evaluation(
    cfg: MultiConfigTrainConfig,
    device: torch.device,
) -> PreparedEvaluation:
    if not isinstance(cfg, MultiConfigTrainConfig):
        raise EvaluationContractError("evaluation requires MultiConfigTrainConfig")
    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        raise EvaluationContractError("evaluation device must be CPU or CUDA")
    _validate_training_run_config(cfg)
    context = preflight_benchmark(cfg)
    seed_everything(cfg.seed)
    model = build_locked_radioflow(cfg.model_size).to(device)
    checkpoint_path = cfg.run_dir / "best.pt"
    identity = build_checkpoint_identity(cfg, context, model)
    state = load_ema_for_evaluation(
        checkpoint_path,
        model=model,
        expected_identity=identity,
    )
    _validate_best_checkpoint_state(state)
    model.eval()
    return PreparedEvaluation(
        cfg=cfg,
        context=context,
        model=model,
        device=device,
        checkpoint_path=checkpoint_path,
        checkpoint_identity=identity,
        selection_identity=_selection_identity(checkpoint_path, identity),
        trainer_state=state,
    )


def _evaluation_loader(
    dataset: Any,
    cfg: MultiConfigTrainConfig,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=multiconfig_collate,
        worker_init_fn=seed_worker,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )


def _batch_prediction(
    prepared: PreparedEvaluation,
    batch: Mapping[str, Any],
    *,
    cfg_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    condition = batch["condition"].to(prepared.device, non_blocking=True)
    target = batch["target"].to(prepared.device, non_blocking=True)
    valid_mask = batch["valid_mask"].to(prepared.device, non_blocking=True)
    noises = [
        make_sample_noise(
            str(metadata["scene_id"]),
            float(metadata["steering_deg"]),
            shape=tuple(target.shape[1:]),
            base_seed=prepared.cfg.seed,
        )
        for metadata in batch["metadata"]
    ]
    noise = torch.stack(noises).to(prepared.device)
    prediction = euler_cfg_sample(
        prepared.model,
        condition,
        noise,
        cfg_scale=cfg_scale,
        steps=2,
        use_amp=prepared.cfg.use_amp,
    )
    return condition, target, valid_mask, prediction


def _evaluate_validation_candidate(
    prepared: PreparedEvaluation,
    cfg_scale: float,
) -> dict[str, int | float]:
    loader = _evaluation_loader(prepared.context.val_dataset, prepared.cfg)
    if len(loader) != VAL_SAMPLES:
        raise EvaluationContractError(
            f"validation loader must have {VAL_SAMPLES} samples, got {len(loader)}"
        )
    accumulator = MetricAccumulator()
    for batch in loader:
        _condition, target, valid_mask, prediction = _batch_prediction(
            prepared, batch, cfg_scale=cfg_scale
        )
        accumulator.update(prediction, target, valid_mask)
    metrics = accumulator.compute()
    if metrics["n_samples"] != VAL_SAMPLES:
        raise EvaluationContractError("CFG selection did not evaluate all 640 samples")
    return metrics


def _validate_selection_against_checkpoint(
    prepared: PreparedEvaluation,
    selection: Mapping[str, Any],
) -> None:
    epoch, best_rmse = _validate_best_checkpoint_state(prepared.trainer_state)
    if selection.get("array_size") != prepared.cfg.array_size:
        raise EvaluationContractError("CFG selection array identity mismatch")
    if selection.get("model_size") != prepared.cfg.model_size:
        raise EvaluationContractError("CFG selection model identity mismatch")
    if int(selection.get("selected_epoch", -1)) != epoch:
        raise EvaluationContractError("CFG selection epoch differs from best.pt")
    if not math.isclose(
        float(selection.get("best_validation_db_rmse_cfg1", math.nan)),
        best_rmse,
        rel_tol=1e-9,
        abs_tol=1e-7,
    ):
        raise EvaluationContractError(
            "CFG selection CFG-1 validation does not match best.pt history"
        )


def run_cfg_selection(
    cfg: MultiConfigTrainConfig,
    device: torch.device,
    results_root: str | Path,
) -> dict[str, Any]:
    prepared = _prepare_evaluation(cfg, device)
    selection_path = cfg.run_dir / "cfg_selection.json"
    final_manifest = Path(results_root) / cfg.array_size / cfg.model_size / "run_manifest.json"
    if selection_path.exists():
        selection = load_cfg_selection(selection_path, prepared.selection_identity)
        _validate_selection_against_checkpoint(prepared, selection)
        return {
            "status": "validated_existing",
            "selection_path": str(selection_path.resolve()),
            "cfg_selection_sha256": sha256_file(selection_path),
            "selected_scale": selection["selected_scale"],
            "selected_epoch": selection["selected_epoch"],
        }
    if _completed_manifest(final_manifest):
        raise EvaluationContractError(
            "a completed test result forbids creating a missing CFG selection"
        )
    if final_manifest.parent.exists():
        raise EvaluationContractError(
            "an existing non-complete test result forbids CFG selection"
        )
    candidate_metrics = {
        scale: _evaluate_validation_candidate(prepared, scale)
        for scale in CFG_CANDIDATES
    }
    epoch, best_rmse = _validate_best_checkpoint_state(prepared.trainer_state)
    if not math.isclose(
        float(candidate_metrics[1.0]["db_rmse"]),
        best_rmse,
        rel_tol=1e-9,
        abs_tol=1e-7,
    ):
        raise EvaluationContractError(
            "recomputed CFG-1 validation dB-RMSE differs from best.pt history"
        )
    payload = build_cfg_selection_payload(
        array_size=cfg.array_size,
        model_size=cfg.model_size,
        selected_epoch=epoch,
        candidate_metrics=candidate_metrics,
        identity=prepared.selection_identity,
    )
    digest = freeze_cfg_selection(
        selection_path,
        payload,
        completed_manifest_path=final_manifest,
    )
    return {
        "status": "selected",
        "selection_path": str(selection_path.resolve()),
        "cfg_selection_sha256": digest,
        "selected_scale": payload["selected_scale"],
        "selected_epoch": payload["selected_epoch"],
        "candidate_metrics": payload["candidate_metrics"],
    }


def _beam_angles_from_test_dataset(dataset: Any) -> dict[int, float]:
    pairs: dict[int, float] = {}
    for record in dataset.records:
        beam_id = int(record.beam_id)
        angle = float(record.steering_deg)
        if beam_id in pairs and not math.isclose(pairs[beam_id], angle, abs_tol=1e-9):
            raise EvaluationContractError(f"beam {beam_id} has inconsistent angles")
        pairs[beam_id] = angle
    if len(pairs) != 8 or sorted(pairs.values()) != list(COMMON_ANGLES_DEG):
        raise EvaluationContractError("test dataset does not contain the common eight beams")
    return dict(sorted(pairs.items(), key=lambda item: item[1]))


def _test_metadata(dataset: Any) -> list[dict[str, Any]]:
    return [record.to_dict() for record in dataset.records]


def _case_key(metadata: Mapping[str, Any]) -> str:
    return f"{metadata['scene_id']}|{float(metadata['steering_deg']):.1f}"


def _metrics_test_payload(
    prepared: PreparedEvaluation,
    selection: Mapping[str, Any],
    selection_sha256: str,
    overall: Mapping[str, int | float],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "array_size": prepared.cfg.array_size,
        "model_size": prepared.cfg.model_size,
        "split": "test",
        "frequency_hz": 6_700_000_000,
        "common_angles_deg": list(COMMON_ANGLES_DEG),
        "resolution": 256,
        "solver": "euler",
        "euler_steps": 2,
        "ema": True,
        "batch_size": 1,
        "selected_epoch": int(selection["selected_epoch"]),
        "best_validation_db_rmse_cfg1": float(
            selection["best_validation_db_rmse_cfg1"]
        ),
        "selected_cfg_scale": float(selection["selected_scale"]),
        "selected_validation_db_rmse": float(
            selection["selected_validation_db_rmse"]
        ),
        "cfg_selection_sha256": selection_sha256,
        "identity": dict(prepared.selection_identity),
        **metrics_for_json(overall),
    }


def _write_test_transaction(
    staging: Path,
    prepared: PreparedEvaluation,
    selection: Mapping[str, Any],
    selection_path: Path,
    selection_sha256: str,
) -> None:
    (staging / "cfg_selection.json").write_bytes(selection_path.read_bytes())
    loader = _evaluation_loader(prepared.context.test_dataset, prepared.cfg)
    if len(loader) != TEST_SAMPLES:
        raise EvaluationContractError(
            f"test loader must have {TEST_SAMPLES} samples, got {len(loader)}"
        )
    beam_metrics = PerBeamMetricAccumulators(
        _beam_angles_from_test_dataset(prepared.context.test_dataset)
    )
    visualization_keys = set(
        fixed_visualization_sample_keys(_test_metadata(prepared.context.test_dataset))
    )
    visualization_cases: dict[str, dict[str, Any]] = {}
    runtime_case: tuple[torch.Tensor, Mapping[str, Any]] | None = None
    prediction_paths: set[str] = set()
    cfg_scale = float(selection["selected_scale"])
    for batch in loader:
        condition, target, valid_mask, prediction = _batch_prediction(
            prepared, batch, cfg_scale=cfg_scale
        )
        metadata = batch["metadata"]
        beam_metrics.update(prediction, target, valid_mask, metadata)
        for index, item in enumerate(metadata):
            stem = stable_case_stem(
                item,
                model_size=prepared.cfg.model_size,
                cfg_scale=cfg_scale,
            )
            relative = f"predictions/{stem}.npz"
            if relative in prediction_paths:
                raise EvaluationContractError(f"duplicate prediction artifact: {relative}")
            prediction_paths.add(relative)
            save_prediction_npz(
                staging / relative,
                prediction=prediction[index],
                target=target[index],
                valid_mask=valid_mask[index],
                metadata=item,
            )
            key = _case_key(item)
            if key in visualization_keys:
                if key in visualization_cases:
                    raise EvaluationContractError(f"duplicate visualization case: {key}")
                visualization_cases[key] = {
                    "condition": condition[index].detach().cpu(),
                    "target": target[index].detach().cpu(),
                    "prediction": prediction[index].detach().cpu(),
                    "valid_mask": valid_mask[index].detach().cpu(),
                    "metadata": dict(item),
                }
            if runtime_case is None:
                runtime_case = (condition[index : index + 1].detach(), dict(item))
    if len(prediction_paths) != TEST_SAMPLES:
        raise EvaluationContractError(
            f"expected {TEST_SAMPLES} prediction files, got {len(prediction_paths)}"
        )
    overall = beam_metrics.compute_overall()
    rows = beam_metrics.compute_rows()
    validate_test_metric_counts(overall, rows)
    _atomic_canonical_json(
        staging / "metrics_test.json",
        _metrics_test_payload(
            prepared,
            selection,
            selection_sha256,
            overall,
        ),
    )
    write_metrics_per_beam_csv(staging / "metrics_per_beam.csv", rows)
    if runtime_case is None:
        raise EvaluationContractError("test loader produced no runtime case")
    runtime_condition, runtime_metadata = runtime_case
    runtime_noise = make_sample_noise(
        str(runtime_metadata["scene_id"]),
        float(runtime_metadata["steering_deg"]),
        base_seed=prepared.cfg.seed,
    ).unsqueeze(0).to(prepared.device)

    def generate() -> torch.Tensor:
        return euler_cfg_sample(
            prepared.model,
            runtime_condition,
            runtime_noise,
            cfg_scale=cfg_scale,
            steps=2,
            use_amp=prepared.cfg.use_amp,
        )

    runtime = benchmark_generation(
        generate=generate,
        model=prepared.model,
        device=prepared.device,
        checkpoint_path=prepared.checkpoint_path,
    )
    runtime.update(
        {
            "batch_size": 1,
            "solver": "euler",
            "euler_steps": 2,
            "ema": True,
            "cfg_scale": cfg_scale,
            "includes_condition_encoding": True,
            "includes_conditional_and_unconditional_cfg_branches": True,
        }
    )
    _atomic_canonical_json(staging / "runtime_generation.json", runtime)
    if set(visualization_cases) != visualization_keys:
        raise EvaluationContractError(
            "fixed visualization cases were not all observed: "
            f"missing={sorted(visualization_keys - set(visualization_cases))}"
        )
    for key in sorted(visualization_cases):
        case = visualization_cases[key]
        stem = stable_case_stem(
            case["metadata"],
            model_size=prepared.cfg.model_size,
            cfg_scale=cfg_scale,
        )
        render_comparison(
            staging / "visualizations" / "comparisons" / f"{stem}.png",
            condition=case["condition"],
            target=case["target"],
            prediction=case["prediction"],
            valid_mask=case["valid_mask"],
            metadata=case["metadata"],
            model_size=prepared.cfg.model_size,
            cfg_scale=cfg_scale,
        )
        render_error_map(
            staging / "visualizations" / "error_maps" / f"{stem}.png",
            target=case["target"],
            prediction=case["prediction"],
            valid_mask=case["valid_mask"],
            metadata=case["metadata"],
            model_size=prepared.cfg.model_size,
            cfg_scale=cfg_scale,
        )
    write_run_manifest(
        staging,
        {
            "schema_version": 1,
            "status": "complete",
            "array_size": prepared.cfg.array_size,
            "model_size": prepared.cfg.model_size,
            "split": "test",
            "n_samples": TEST_SAMPLES,
            "cfg_selection_sha256": selection_sha256,
            "selected_epoch": int(selection["selected_epoch"]),
            "selected_cfg_scale": cfg_scale,
            "solver": "euler",
            "euler_steps": 2,
            "ema": True,
            "identity": dict(prepared.selection_identity),
        },
    )


def run_test_evaluation(
    cfg: MultiConfigTrainConfig,
    device: torch.device,
    results_root: str | Path,
) -> dict[str, Any]:
    final_dir = Path(results_root) / cfg.array_size / cfg.model_size
    if final_dir.exists():
        raise EvaluationContractError(
            f"final result directory already exists; test cannot be rerun: {final_dir}"
        )
    prepared = _prepare_evaluation(cfg, device)
    selection_path = cfg.run_dir / "cfg_selection.json"
    selection = load_cfg_selection(selection_path, prepared.selection_identity)
    _validate_selection_against_checkpoint(prepared, selection)
    selection_sha256 = sha256_file(selection_path)

    def builder(staging: Path) -> None:
        _write_test_transaction(
            staging,
            prepared,
            selection,
            selection_path,
            selection_sha256,
        )

    published = atomic_result_transaction(final_dir, builder)
    return {
        "status": "complete",
        "result_dir": str(published.resolve()),
        "run_manifest_sha256": sha256_file(published / "run_manifest.json"),
        "cfg_selection_sha256": selection_sha256,
        "selected_scale": selection["selected_scale"],
        "selected_epoch": selection["selected_epoch"],
        "n_samples": TEST_SAMPLES,
    }


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(payload) != expected:
        raise EvaluationContractError(
            f"{label} keys mismatch: missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )


def _read_per_beam_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != list(PER_BEAM_COLUMNS):
                raise EvaluationContractError(
                    f"per-beam CSV columns mismatch: {reader.fieldnames}"
                )
            raw_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise EvaluationContractError(f"cannot read per-beam CSV {path}: {error}") from error
    numeric_ints = {"beam_id", "n_samples", "n_valid_pixels"}
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        row: dict[str, Any] = {}
        try:
            for column in PER_BEAM_COLUMNS:
                value = raw[column]
                if column in numeric_ints:
                    row[column] = int(value)
                elif column == "psnr" and value == "":
                    row[column] = None
                else:
                    row[column] = float(value)
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationContractError(
                f"invalid per-beam CSV row {index}"
            ) from error
        rows.append(row)
    return rows


def _validate_accuracy_values(
    metrics: Mapping[str, Any],
    *,
    label: str,
    allow_unmarked_infinite_psnr: bool = False,
) -> None:
    nonnegative = ("db_rmse", "db_mae", "mse", "nmse")
    for key in nonnegative:
        try:
            value = float(metrics[key])
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationContractError(f"{label} {key} is invalid") from error
        if not math.isfinite(value) or value < 0.0:
            raise EvaluationContractError(f"{label} {key} must be finite and non-negative")
    try:
        ssim = float(metrics["ssim"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationContractError(f"{label} ssim is invalid") from error
    if not math.isfinite(ssim) or not -1.0 <= ssim <= 1.0:
        raise EvaluationContractError(f"{label} ssim must be finite in [-1,1]")
    for key in ("raw_fraction_below_zero", "raw_fraction_above_one"):
        if key not in metrics:
            continue
        value = float(metrics[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise EvaluationContractError(f"{label} {key} must be in [0,1]")
    psnr = metrics.get("psnr")
    if psnr is None:
        if not allow_unmarked_infinite_psnr and metrics.get("psnr_infinite") is not True:
            raise EvaluationContractError(f"{label} null PSNR lacks infinite marker")
    else:
        try:
            psnr_value = float(psnr)
        except (TypeError, ValueError) as error:
            raise EvaluationContractError(f"{label} psnr is invalid") from error
        if not math.isfinite(psnr_value):
            raise EvaluationContractError(f"{label} psnr must be finite or marked infinite")


def _validate_runtime_values(runtime: Mapping[str, Any], *, label: str) -> None:
    timings: dict[str, float] = {}
    for key in (
        "latency_ms_p50",
        "latency_ms_p95",
        "latency_ms_mean",
        "latency_ms_std",
    ):
        try:
            value = float(runtime[key])
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationContractError(f"{label} {key} is invalid") from error
        if not math.isfinite(value) or value < 0.0:
            raise EvaluationContractError(f"{label} {key} must be finite and non-negative")
        timings[key] = value
    if timings["latency_ms_p95"] < timings["latency_ms_p50"]:
        raise EvaluationContractError(f"{label} p95 latency is below p50")
    peak = runtime.get("max_memory_allocated_bytes")
    if peak is not None and (isinstance(peak, bool) or not isinstance(peak, int) or peak < 0):
        raise EvaluationContractError(f"{label} peak allocation is invalid")
    checkpoint_size = runtime.get("checkpoint_size_bytes")
    if (
        isinstance(checkpoint_size, bool)
        or not isinstance(checkpoint_size, int)
        or checkpoint_size <= 0
    ):
        raise EvaluationContractError(f"{label} checkpoint size is invalid")


def _load_completed_pair(
    run_root: Path,
    results_root: Path,
    array_size: str,
    model_size: str,
) -> dict[str, Any]:
    result_dir = results_root / array_size / model_size
    run_dir = run_root / array_size / model_size
    manifest = validate_run_manifest_artifacts(result_dir)
    manifest_keys = {
        "schema_version",
        "status",
        "array_size",
        "model_size",
        "split",
        "n_samples",
        "cfg_selection_sha256",
        "selected_epoch",
        "selected_cfg_scale",
        "solver",
        "euler_steps",
        "ema",
        "identity",
        "artifacts",
    }
    _require_exact_keys(manifest, manifest_keys, label="completion manifest")
    fixed_manifest = {
        "array_size": array_size,
        "model_size": model_size,
        "split": "test",
        "n_samples": TEST_SAMPLES,
        "solver": "euler",
        "euler_steps": 2,
        "ema": True,
    }
    for key, expected in fixed_manifest.items():
        if manifest[key] != expected:
            raise EvaluationContractError(
                f"completion manifest {key} mismatch for {array_size}/{model_size}"
            )
    identity_raw = manifest["identity"]
    if not isinstance(identity_raw, Mapping):
        raise EvaluationContractError("completion manifest identity must be an object")
    identity = _validate_selection_identity(identity_raw)
    if manifest["cfg_selection_sha256"] != sha256_file(
        result_dir / "cfg_selection.json"
    ):
        raise EvaluationContractError("completion manifest selection hash mismatch")
    source_selection = run_dir / "cfg_selection.json"
    if not source_selection.is_file():
        raise EvaluationContractError(f"current source selection is missing: {source_selection}")
    selection_sha = sha256_file(source_selection)
    if selection_sha != manifest["cfg_selection_sha256"]:
        raise EvaluationContractError("current CFG selection differs from completed test")
    if source_selection.read_bytes() != (result_dir / "cfg_selection.json").read_bytes():
        raise EvaluationContractError("published CFG selection copy differs from source")
    selection = load_cfg_selection(source_selection, identity)
    if selection["array_size"] != array_size or selection["model_size"] != model_size:
        raise EvaluationContractError("CFG selection pair identity mismatch")
    if (
        selection["selected_epoch"] != manifest["selected_epoch"]
        or selection["selected_scale"] != manifest["selected_cfg_scale"]
    ):
        raise EvaluationContractError("completion manifest inference choice mismatch")

    metrics = _read_canonical_json(
        result_dir / "metrics_test.json", label="test metrics"
    )
    required_metrics = {
        "schema_version",
        "array_size",
        "model_size",
        "split",
        "frequency_hz",
        "common_angles_deg",
        "resolution",
        "solver",
        "euler_steps",
        "ema",
        "batch_size",
        "selected_epoch",
        "best_validation_db_rmse_cfg1",
        "selected_cfg_scale",
        "selected_validation_db_rmse",
        "cfg_selection_sha256",
        "identity",
        "n_samples",
        "n_valid_pixels",
        "n_ssim_windows",
        "db_rmse",
        "db_mae",
        "mse",
        "nmse",
        "psnr",
        "ssim",
        "raw_fraction_below_zero",
        "raw_fraction_above_one",
        "psnr_infinite",
    }
    _require_exact_keys(metrics, required_metrics, label="test metrics")
    fixed_metrics = {
        "schema_version": 1,
        "array_size": array_size,
        "model_size": model_size,
        "split": "test",
        "frequency_hz": 6_700_000_000,
        "common_angles_deg": list(COMMON_ANGLES_DEG),
        "resolution": 256,
        "solver": "euler",
        "euler_steps": 2,
        "ema": True,
        "batch_size": 1,
        "n_samples": TEST_SAMPLES,
        "cfg_selection_sha256": selection_sha,
        "identity": identity,
        "selected_epoch": selection["selected_epoch"],
        "selected_cfg_scale": selection["selected_scale"],
        "best_validation_db_rmse_cfg1": selection[
            "best_validation_db_rmse_cfg1"
        ],
        "selected_validation_db_rmse": selection[
            "selected_validation_db_rmse"
        ],
    }
    for key, expected in fixed_metrics.items():
        if metrics[key] != expected:
            raise EvaluationContractError(
                f"test metrics {key} mismatch for {array_size}/{model_size}"
            )
    _validate_accuracy_values(metrics, label="overall test metrics")
    rows = _read_per_beam_csv(result_dir / "metrics_per_beam.csv")
    validate_test_metric_counts(metrics, rows)
    for row in rows:
        _validate_accuracy_values(
            row,
            label=f"beam {row['beam_id']} metrics",
            allow_unmarked_infinite_psnr=True,
        )

    runtime = _read_canonical_json(
        result_dir / "runtime_generation.json", label="generation runtime"
    )
    runtime_fixed = {
        "batch_size": 1,
        "solver": "euler",
        "euler_steps": 2,
        "ema": True,
        "cfg_scale": selection["selected_scale"],
        "includes_condition_encoding": True,
        "includes_conditional_and_unconditional_cfg_branches": True,
        "warmup_calls": 20,
        "measured_calls": 100,
        "parameter_count": EXPECTED_PARAMETER_COUNTS[model_size],
    }
    for key, expected in runtime_fixed.items():
        if runtime.get(key) != expected:
            raise EvaluationContractError(
                f"generation runtime {key} mismatch for {array_size}/{model_size}"
            )
    _validate_runtime_values(runtime, label="generation runtime")
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != identity[
        "checkpoint_sha256"
    ]:
        raise EvaluationContractError("current best.pt differs from completed test")
    if runtime.get("checkpoint_size_bytes") != checkpoint_path.stat().st_size:
        raise EvaluationContractError("generation runtime checkpoint size mismatch")
    config_path = run_dir / "config.json"
    try:
        cfg = MultiConfigTrainConfig.from_json(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise EvaluationContractError(f"invalid source training config: {error}") from error
    if (
        cfg.array_size != array_size
        or cfg.model_size != model_size
        or cfg.run_root.resolve() != run_root.resolve()
        or cfg.config_sha256 != identity["config_sha256"]
    ):
        raise EvaluationContractError("current training config differs from completed test")
    training_runtime = _read_canonical_json(
        run_dir / "training_runtime.json", label="training runtime"
    )
    if training_runtime.get("status") != "complete":
        raise EvaluationContractError("summary requires complete training runtime evidence")

    summary_row = {
        "array_size": array_size,
        "model_size": model_size,
        "status": "complete",
        "selected_epoch": selection["selected_epoch"],
        "best_validation_db_rmse_cfg1": selection[
            "best_validation_db_rmse_cfg1"
        ],
        "selected_cfg_scale": selection["selected_scale"],
        "selected_validation_db_rmse": selection[
            "selected_validation_db_rmse"
        ],
        "db_rmse": metrics["db_rmse"],
        "db_mae": metrics["db_mae"],
        "mse": metrics["mse"],
        "nmse": metrics["nmse"],
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "parameter_count": runtime["parameter_count"],
        "checkpoint_size_bytes": runtime["checkpoint_size_bytes"],
        "peak_training_allocated_bytes": training_runtime.get(
            "peak_training_allocated_bytes"
        ),
        "peak_inference_allocated_bytes": runtime.get(
            "max_memory_allocated_bytes"
        ),
        "latency_ms_p50": runtime.get("latency_ms_p50"),
        "latency_ms_p95": runtime.get("latency_ms_p95"),
        "hardware_gate_sha256": "",
    }
    angle_rows = [
        {
            "array_size": array_size,
            "model_size": model_size,
            "status": "complete",
            **row,
            "hardware_gate_sha256": "",
        }
        for row in sorted(rows, key=lambda row: float(row["angle_deg"]))
    ]
    return {
        "summary_row": summary_row,
        "angle_rows": angle_rows,
        "identity": identity,
        "cfg": cfg,
    }


def _validate_common_completed_contract(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise EvaluationContractError("summary has no completed model records")
    shared_identity_keys = (
        "split_sha256",
        "schema_sha256",
        "archive_sha256",
        "dataset_revision",
        "radioflow_upstream_base",
        "git_commit",
    )
    reference = records[0]["identity"]
    for record in records[1:]:
        identity = record["identity"]
        for key in shared_identity_keys:
            if identity[key] != reference[key]:
                raise EvaluationContractError(
                    f"completed model shared identity mismatch: {key}"
                )
    per_array_manifest: dict[str, str] = {}
    for record in records:
        row = record["summary_row"]
        array_size = str(row["array_size"])
        manifest_sha = record["identity"]["manifest_sha256"]
        if array_size in per_array_manifest and per_array_manifest[array_size] != manifest_sha:
            raise EvaluationContractError(
                f"Lite/Large manifest mismatch for {array_size}"
            )
        per_array_manifest[array_size] = manifest_sha


def _terminal_state(
    completed_pairs: set[tuple[str, str]],
    *,
    gate_exists: bool,
) -> str:
    all_pairs = {(array, size) for array in ARRAY_NAMES for size in MODEL_SIZES}
    lite_pairs = {(array, "lite") for array in ARRAY_NAMES}
    if completed_pairs == all_pairs and not gate_exists:
        return "complete"
    if completed_pairs == lite_pairs and gate_exists:
        return "large_hardware_blocked"
    raise EvaluationContractError(
        "benchmark terminal state must be six completed pairs, or three Lite "
        "pairs plus one global Large hardware gate"
    )


def _validate_global_large_gate(
    gate_path: Path,
    lite_records: Sequence[Mapping[str, Any]],
    run_root: Path,
) -> str:
    gate_raw = _read_canonical_json(gate_path, label="Large hardware gate")
    trigger = gate_raw.get("trigger_array")
    if trigger not in ARRAY_NAMES:
        raise EvaluationContractError("Large hardware gate trigger array is invalid")
    by_array = {
        record["summary_row"]["array_size"]: record for record in lite_records
    }
    if set(by_array) != set(ARRAY_NAMES):
        raise EvaluationContractError("Large hardware gate requires all three Lite records")
    reference = lite_records[0]["identity"]
    config_hashes: dict[str, str] = {}
    manifest_hashes: dict[str, str] = {}
    for array in ARRAY_NAMES:
        lite_cfg = by_array[array]["cfg"]
        large_cfg = MultiConfigTrainConfig(
            array_size=array,
            model_size="large",
            dataset_root=lite_cfg.dataset_root,
            manifest_dir=lite_cfg.manifest_dir,
            run_root=run_root,
            train_scale=lite_cfg.train_scale,
        )
        config_hashes[array] = large_cfg.config_sha256
        manifest_hashes[array] = by_array[array]["identity"]["manifest_sha256"]
    context = LargeHardwareGateContext(
        trigger_array=str(trigger),
        config_sha256_by_array=config_hashes,
        manifest_sha256_by_array=manifest_hashes,
        split_sha256=reference["split_sha256"],
        schema_sha256=reference["schema_sha256"],
        archive_sha256=reference["archive_sha256"],
        dataset_revision=reference["dataset_revision"],
        radioflow_upstream_base=reference["radioflow_upstream_base"],
        git_commit=reference["git_commit"],
    )
    try:
        validate_large_hardware_gate(gate_path, context)
    except Exception as error:
        raise EvaluationContractError(f"invalid global Large hardware gate: {error}") from error
    return sha256_file(gate_path)


def _blocked_summary_row(array_size: str, gate_sha256: str) -> dict[str, Any]:
    return {
        column: (
            array_size
            if column == "array_size"
            else "large"
            if column == "model_size"
            else "hardware_blocked"
            if column == "status"
            else gate_sha256
            if column == "hardware_gate_sha256"
            else ""
        )
        for column in SUMMARY_COLUMNS
    }


def _blocked_angle_rows(array_size: str, gate_sha256: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for angle in COMMON_ANGLES_DEG:
        row = {column: "" for column in ANGLE_SUMMARY_COLUMNS}
        row.update(
            {
                "array_size": array_size,
                "model_size": "large",
                "status": "hardware_blocked",
                "angle_deg": angle,
                "hardware_gate_sha256": gate_sha256,
            }
        )
        rows.append(row)
    return rows


def _write_csv_atomic(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=columns,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for index, row in enumerate(rows):
                missing = set(columns) - set(row)
                if missing:
                    raise EvaluationContractError(
                        f"summary CSV row {index} is missing {sorted(missing)}"
                    )
                writer.writerow({column: row[column] for column in columns})
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _markdown_cell(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def _write_summary_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    display_columns = (
        "array_size",
        "model_size",
        "status",
        "selected_epoch",
        "selected_cfg_scale",
        "db_rmse",
        "db_mae",
        "ssim",
        "parameter_count",
        "latency_ms_p50",
        "latency_ms_p95",
        "peak_inference_allocated_bytes",
        "hardware_gate_sha256",
    )
    lines = [
        "# Multi-config SRM RadioFlow benchmark",
        "",
        "| " + " | ".join(display_columns) + " |",
        "| " + " | ".join("---" for _ in display_columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown_cell(row[column]) for column in display_columns) + " |"
        for row in rows
    )
    lines.append("")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            output.write("\n".join(lines))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize_benchmark(
    run_root: str | Path,
    results_root: str | Path,
) -> dict[str, Any]:
    run_root = Path(run_root)
    results_root = Path(results_root)
    existing_pairs = {
        (array, size)
        for array in ARRAY_NAMES
        for size in MODEL_SIZES
        if (results_root / array / size).exists()
    }
    gate_path = run_root / "_hardware" / "large_hardware_gate.json"
    terminal = _terminal_state(existing_pairs, gate_exists=gate_path.exists())
    records = [
        _load_completed_pair(run_root, results_root, array, size)
        for array in ARRAY_NAMES
        for size in MODEL_SIZES
        if (array, size) in existing_pairs
    ]
    _validate_common_completed_contract(records)
    record_by_pair = {
        (record["summary_row"]["array_size"], record["summary_row"]["model_size"]): record
        for record in records
    }
    gate_sha256 = ""
    if terminal == "large_hardware_blocked":
        lite_records = [record_by_pair[(array, "lite")] for array in ARRAY_NAMES]
        gate_sha256 = _validate_global_large_gate(gate_path, lite_records, run_root)
    summary_rows: list[dict[str, Any]] = []
    angle_rows: list[dict[str, Any]] = []
    for array in ARRAY_NAMES:
        for size in MODEL_SIZES:
            pair = (array, size)
            if pair in record_by_pair:
                summary_rows.append(dict(record_by_pair[pair]["summary_row"]))
                angle_rows.extend(record_by_pair[pair]["angle_rows"])
            else:
                summary_rows.append(_blocked_summary_row(array, gate_sha256))
                angle_rows.extend(_blocked_angle_rows(array, gate_sha256))
    if len(summary_rows) != 6:
        raise EvaluationContractError("benchmark summary must contain six rows")
    summary_csv = results_root / "benchmark_summary.csv"
    summary_md = results_root / "benchmark_summary.md"
    angle_csv = results_root / "metrics_per_angle_comparison.csv"
    _write_csv_atomic(summary_csv, SUMMARY_COLUMNS, summary_rows)
    _write_csv_atomic(angle_csv, ANGLE_SUMMARY_COLUMNS, angle_rows)
    _write_summary_markdown(summary_md, summary_rows)
    return {
        "status": terminal,
        "completed_pairs": len(records),
        "blocked_pairs": 6 - len(records),
        "hardware_gate_sha256": gate_sha256 or None,
        "summary_csv": str(summary_csv.resolve()),
        "summary_markdown": str(summary_md.resolve()),
        "per_angle_csv": str(angle_csv.resolve()),
    }
