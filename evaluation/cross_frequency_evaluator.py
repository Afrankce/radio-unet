from __future__ import annotations

import csv
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from data_loaders.multiconfig import multiconfig_collate
from evaluation.multiconfig_evaluator import (
    CFG_CANDIDATES,
    atomic_result_transaction,
    select_cfg_candidate,
    write_run_manifest,
)
from evaluation.radioflow_sampling import euler_cfg_sample, make_sample_noise
from evaluation.radiomap_metrics import (
    MetricAccumulator,
    PerFrequencyMetricAccumulators,
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
from training.cross_frequency_config import CrossFrequencyTrainConfig
from training.cross_frequency_trainer import (
    CrossFrequencyContext,
    build_cross_frequency_checkpoint_identity,
    preflight_cross_frequency,
    seed_everything,
)
from training.multiconfig_trainer import seed_worker
from training.model_factory import build_locked_radioflow


METRICS_PER_FREQUENCY_COLUMNS = (
    "frequency_hz",
    "angle_deg",
    "n_samples",
    "n_valid_pixels",
    "db_rmse",
    "db_mae",
    "mse",
    "nmse",
    "psnr",
    "ssim",
)


class CrossFrequencyEvaluationError(RuntimeError):
    """Cross-frequency selection or test output violates the frozen protocol."""


@dataclass(frozen=True)
class PreparedCrossFrequencyEvaluation:
    cfg: CrossFrequencyTrainConfig
    context: CrossFrequencyContext
    model: torch.nn.Module
    device: torch.device
    checkpoint_path: Path
    checkpoint_identity: CheckpointIdentity
    selection_identity: Mapping[str, str]
    trainer_state: TrainerState


def _read_canonical_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CrossFrequencyEvaluationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise CrossFrequencyEvaluationError(f"{label} must be a canonical JSON object")
    return payload


def _validate_identity(identity: Mapping[str, Any]) -> dict[str, str]:
    expected = {
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
    if set(identity) != expected:
        raise CrossFrequencyEvaluationError("selection identity keys mismatch")
    normalized = {key: str(value) for key, value in identity.items()}
    for key, value in normalized.items():
        length = 40 if key in {
            "dataset_revision",
            "radioflow_upstream_base",
            "git_commit",
        } else 64
        if len(value) != length or any(character not in "0123456789abcdef" for character in value):
            raise CrossFrequencyEvaluationError(
                f"selection identity {key} is not a lowercase hexadecimal digest"
            )
    return normalized


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
        raise CrossFrequencyEvaluationError(
            "best checkpoint has no completed validation epoch"
        )
    try:
        epoch = int(state.history[-1]["epoch"])
        rmse = float(state.history[-1]["val_db_rmse"])
    except (KeyError, TypeError, ValueError) as error:
        raise CrossFrequencyEvaluationError(
            "best checkpoint history is incomplete"
        ) from error
    if epoch != state.completed_epochs or not math.isclose(
        rmse,
        float(state.best_val_db_rmse),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise CrossFrequencyEvaluationError(
            "best checkpoint epoch and validation history disagree"
        )
    return epoch, rmse


def _prepare_evaluation(
    cfg: CrossFrequencyTrainConfig,
    device: torch.device,
) -> PreparedCrossFrequencyEvaluation:
    if not isinstance(cfg, CrossFrequencyTrainConfig):
        raise CrossFrequencyEvaluationError(
            "cross-frequency evaluation requires CrossFrequencyTrainConfig"
        )
    if device.type not in {"cpu", "cuda"}:
        raise CrossFrequencyEvaluationError("evaluation device must be CPU or CUDA")
    context = preflight_cross_frequency(cfg)
    seed_everything(cfg.seed)
    model = build_locked_radioflow(cfg.model_size).to(device)
    checkpoint_path = cfg.run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise CrossFrequencyEvaluationError(
            f"best checkpoint is missing: {checkpoint_path}"
        )
    identity = build_cross_frequency_checkpoint_identity(cfg, context, model)
    try:
        state = load_ema_for_evaluation(
            checkpoint_path,
            model=model,
            expected_identity=identity,
        )
    except Exception as error:
        raise CrossFrequencyEvaluationError(
            f"cannot load EMA checkpoint strictly: {error}"
        ) from error
    _validate_best_checkpoint_state(state)
    model.eval()
    return PreparedCrossFrequencyEvaluation(
        cfg=cfg,
        context=context,
        model=model,
        device=device,
        checkpoint_path=checkpoint_path,
        checkpoint_identity=identity,
        selection_identity=_selection_identity(checkpoint_path, identity),
        trainer_state=state,
    )


def _evaluation_loader(dataset: Any, cfg: CrossFrequencyTrainConfig) -> DataLoader:
    # Evaluation is deterministic and batch-one.  On Windows, repeatedly
    # spawning two workers for each of the four CFG candidates can retain large
    # torch worker processes between iterators; the single-process path avoids
    # that resource leak without changing samples or metric aggregation.
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=multiconfig_collate,
        worker_init_fn=seed_worker,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )


def _batch_prediction(
    prepared: PreparedCrossFrequencyEvaluation,
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
    prepared: PreparedCrossFrequencyEvaluation,
    cfg_scale: float,
) -> dict[str, int | float]:
    loader = _evaluation_loader(prepared.context.val_dataset, prepared.cfg)
    if len(loader) != prepared.cfg.val_samples:
        raise CrossFrequencyEvaluationError(
            f"validation loader must contain {prepared.cfg.val_samples} samples"
        )
    accumulator = MetricAccumulator()
    for batch in loader:
        _condition, target, valid_mask, prediction = _batch_prediction(
            prepared,
            batch,
            cfg_scale=cfg_scale,
        )
        accumulator.update(prediction, target, valid_mask)
    metrics = accumulator.compute()
    if metrics["n_samples"] != prepared.cfg.val_samples:
        raise CrossFrequencyEvaluationError(
            "CFG selection did not evaluate all validation samples"
        )
    return metrics


def build_cfg_selection_payload(
    *,
    prepared: PreparedCrossFrequencyEvaluation,
    candidate_metrics: Mapping[float, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(candidate_metrics) != set(CFG_CANDIDATES):
        raise CrossFrequencyEvaluationError("CFG candidate grid is not the locked grid")
    selected_scale = select_cfg_candidate(candidate_metrics)
    epoch, best_rmse = _validate_best_checkpoint_state(prepared.trainer_state)
    return {
        "schema_version": 1,
        "experiment": "cross_frequency_8x8",
        "array_size": prepared.cfg.array_size,
        "model_size": prepared.cfg.model_size,
        "candidates": list(CFG_CANDIDATES),
        "candidate_metrics": {
            f"{scale:.1f}": metrics_for_json(candidate_metrics[scale])
            for scale in CFG_CANDIDATES
        },
        "train_frequency_hz": prepared.cfg.train_frequency_hz,
        "validation_frequency_hz": prepared.cfg.val_frequency_hz,
        "validation_samples": prepared.cfg.val_samples,
        "test_frequency_hz": prepared.cfg.test_frequency_hz,
        "test_samples": prepared.cfg.test_samples,
        "steering_deg": prepared.cfg.steering_deg,
        "selected_epoch": epoch,
        "best_validation_db_rmse_cfg1": best_rmse,
        "selected_scale": selected_scale,
        "selected_validation_db_rmse": float(candidate_metrics[selected_scale]["db_rmse"]),
        "tie_break_rule": "minimum_db_rmse_then_smaller_cfg",
        "solver": "euler",
        "euler_steps": 2,
        "ema": True,
        "identity": _validate_identity(prepared.selection_identity),
    }


def _freeze_canonical_json(path: Path, payload: Mapping[str, Any]) -> str:
    expected = canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != expected:
            raise CrossFrequencyEvaluationError(
                f"immutable CFG selection differs: {path}"
            )
    else:
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
    return sha256_file(path)


def load_cfg_selection(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
    cfg: CrossFrequencyTrainConfig,
) -> dict[str, Any]:
    payload = _read_canonical_json(path, "cross-frequency CFG selection")
    required = {
        "schema_version",
        "experiment",
        "array_size",
        "model_size",
        "candidates",
        "candidate_metrics",
        "train_frequency_hz",
        "validation_frequency_hz",
        "validation_samples",
        "test_frequency_hz",
        "test_samples",
        "steering_deg",
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
        raise CrossFrequencyEvaluationError("cross-frequency CFG selection keys mismatch")
    if (
        payload["schema_version"] != 1
        or payload["experiment"] != "cross_frequency_8x8"
        or payload["array_size"] != cfg.array_size
        or payload["model_size"] != cfg.model_size
        or payload["candidates"] != list(CFG_CANDIDATES)
        or payload["train_frequency_hz"] != cfg.train_frequency_hz
        or payload["validation_frequency_hz"] != cfg.val_frequency_hz
        or payload["validation_samples"] != cfg.val_samples
        or payload["test_frequency_hz"] != cfg.test_frequency_hz
        or payload["test_samples"] != cfg.test_samples
        or float(payload["steering_deg"]) != cfg.steering_deg
    ):
        raise CrossFrequencyEvaluationError("cross-frequency CFG protocol mismatch")
    if payload["solver"] != "euler" or payload["euler_steps"] != 2 or payload["ema"] is not True:
        raise CrossFrequencyEvaluationError("cross-frequency generation protocol mismatch")
    identity = _validate_identity(expected_identity)
    if payload["identity"] != identity:
        raise CrossFrequencyEvaluationError("cross-frequency CFG selection identity mismatch")
    raw_metrics = payload["candidate_metrics"]
    if not isinstance(raw_metrics, Mapping):
        raise CrossFrequencyEvaluationError("candidate metrics must be an object")
    expected_keys = {f"{scale:.1f}" for scale in CFG_CANDIDATES}
    if set(raw_metrics) != expected_keys:
        raise CrossFrequencyEvaluationError("candidate metric grid mismatch")
    candidate_metrics: dict[float, Mapping[str, Any]] = {}
    for scale in CFG_CANDIDATES:
        metrics = raw_metrics[f"{scale:.1f}"]
        if not isinstance(metrics, Mapping) or metrics.get("n_samples") != cfg.val_samples:
            raise CrossFrequencyEvaluationError(
                f"CFG {scale} must contain exactly {cfg.val_samples} validation samples"
            )
        try:
            rmse = float(metrics["db_rmse"])
        except (KeyError, TypeError, ValueError) as error:
            raise CrossFrequencyEvaluationError("candidate metric lacks db_rmse") from error
        if not math.isfinite(rmse) or rmse < 0.0:
            raise CrossFrequencyEvaluationError("candidate db_rmse is invalid")
        candidate_metrics[scale] = metrics
    selected = select_cfg_candidate(candidate_metrics)
    if float(payload["selected_scale"]) != selected:
        raise CrossFrequencyEvaluationError("selected CFG does not match candidate metrics")
    if int(payload["selected_epoch"]) <= 0:
        raise CrossFrequencyEvaluationError("selected epoch must be positive")
    return payload


def run_cfg_selection(
    cfg: CrossFrequencyTrainConfig,
    device: torch.device,
    results_root: str | Path,
) -> dict[str, Any]:
    prepared = _prepare_evaluation(cfg, device)
    selection_path = cfg.run_dir / "cfg_selection.json"
    final_manifest = Path(results_root) / "run_manifest.json"
    if selection_path.exists():
        selection = load_cfg_selection(
            selection_path,
            expected_identity=prepared.selection_identity,
            cfg=cfg,
        )
        epoch, _rmse = _validate_best_checkpoint_state(prepared.trainer_state)
        if int(selection["selected_epoch"]) != epoch:
            raise CrossFrequencyEvaluationError(
                "existing CFG selection epoch differs from best.pt"
            )
        return {
            "status": "validated_existing",
            "selection_path": str(selection_path.resolve()),
            "cfg_selection_sha256": sha256_file(selection_path),
            "selected_scale": selection["selected_scale"],
            "selected_epoch": selection["selected_epoch"],
        }
    if final_manifest.exists() or Path(results_root).exists():
        raise CrossFrequencyEvaluationError(
            "an existing test result forbids creating a missing CFG selection"
        )
    candidate_metrics = {
        scale: _evaluate_validation_candidate(prepared, scale)
        for scale in CFG_CANDIDATES
    }
    if not math.isclose(
        float(candidate_metrics[1.0]["db_rmse"]),
        float(prepared.trainer_state.best_val_db_rmse),
        rel_tol=1e-9,
        abs_tol=1e-7,
    ):
        raise CrossFrequencyEvaluationError(
            "recomputed CFG-1 validation dB-RMSE differs from best.pt"
        )
    payload = build_cfg_selection_payload(
        prepared=prepared,
        candidate_metrics=candidate_metrics,
    )
    digest = _freeze_canonical_json(selection_path, payload)
    return {
        "status": "selected",
        "selection_path": str(selection_path.resolve()),
        "cfg_selection_sha256": digest,
        "selected_scale": payload["selected_scale"],
        "selected_epoch": payload["selected_epoch"],
        "candidate_metrics": payload["candidate_metrics"],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=METRICS_PER_FREQUENCY_COLUMNS,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                serialized = dict(row)
                if math.isinf(float(serialized["psnr"])):
                    serialized["psnr"] = ""
                writer.writerow({column: serialized[column] for column in METRICS_PER_FREQUENCY_COLUMNS})
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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
        if temporary.exists():
            temporary.unlink()


def _metrics_test_payload(
    prepared: PreparedCrossFrequencyEvaluation,
    selection: Mapping[str, Any],
    selection_sha256: str,
    overall: Mapping[str, int | float],
    *,
    test_beam_id: int,
    test_config_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "cross_frequency_8x8",
        "array_size": prepared.cfg.array_size,
        "model_size": prepared.cfg.model_size,
        "split": "test",
        "train_frequency_hz": prepared.cfg.train_frequency_hz,
        "validation_frequency_hz": prepared.cfg.val_frequency_hz,
        "test_frequency_hz": prepared.cfg.test_frequency_hz,
        "steering_deg": prepared.cfg.steering_deg,
        "test_beam_id": test_beam_id,
        "test_config_ids": list(test_config_ids),
        "resolution": prepared.cfg.resolution,
        "solver": "euler",
        "euler_steps": 2,
        "ema": True,
        "batch_size": 1,
        "selected_epoch": int(selection["selected_epoch"]),
        "best_validation_db_rmse_cfg1": float(selection["best_validation_db_rmse_cfg1"]),
        "selected_cfg_scale": float(selection["selected_scale"]),
        "selected_validation_db_rmse": float(selection["selected_validation_db_rmse"]),
        "cfg_selection_sha256": selection_sha256,
        "identity": dict(prepared.selection_identity),
        **metrics_for_json(overall),
    }


def _write_test_transaction(
    staging: Path,
    prepared: PreparedCrossFrequencyEvaluation,
    selection: Mapping[str, Any],
    selection_path: Path,
    selection_sha256: str,
) -> None:
    (staging / "cfg_selection.json").write_bytes(selection_path.read_bytes())
    loader = _evaluation_loader(prepared.context.test_dataset, prepared.cfg)
    if len(loader) != prepared.cfg.test_samples:
        raise CrossFrequencyEvaluationError(
            f"test loader must contain {prepared.cfg.test_samples} samples"
        )
    expected_group = ((prepared.cfg.test_frequency_hz, prepared.cfg.steering_deg),)
    grouped = PerFrequencyMetricAccumulators(expected_groups=expected_group)
    records = tuple(prepared.context.test_dataset.records)
    test_beam_ids = {int(record.beam_id) for record in records}
    test_config_ids = sorted({str(record.config_id) for record in records})
    if test_beam_ids != {4} or test_config_ids == []:
        raise CrossFrequencyEvaluationError(
            f"test source must use 6.7 GHz zero-degree beam ID 4, got {test_beam_ids}"
        )
    chosen_scenes = {
        records[0].scene_id,
        records[len(records) // 2].scene_id,
        records[-1].scene_id,
    }
    visualization_cases: dict[str, dict[str, Any]] = {}
    runtime_case: tuple[torch.Tensor, Mapping[str, Any]] | None = None
    prediction_paths: set[str] = set()
    cfg_scale = float(selection["selected_scale"])
    for batch in loader:
        condition, target, valid_mask, prediction = _batch_prediction(
            prepared,
            batch,
            cfg_scale=cfg_scale,
        )
        metadata = batch["metadata"]
        grouped.update(prediction, target, valid_mask, metadata)
        for index, item in enumerate(metadata):
            stem = stable_case_stem(
                item,
                model_size=prepared.cfg.model_size,
                cfg_scale=cfg_scale,
            )
            relative = f"predictions/{stem}.npz"
            if relative in prediction_paths:
                raise CrossFrequencyEvaluationError(
                    f"duplicate prediction artifact: {relative}"
                )
            prediction_paths.add(relative)
            save_prediction_npz(
                staging / relative,
                prediction=prediction[index],
                target=target[index],
                valid_mask=valid_mask[index],
                metadata=item,
            )
            if str(item["scene_id"]) in chosen_scenes:
                visualization_cases[str(item["scene_id"])] = {
                    "condition": condition[index].detach().cpu(),
                    "target": target[index].detach().cpu(),
                    "prediction": prediction[index].detach().cpu(),
                    "valid_mask": valid_mask[index].detach().cpu(),
                    "metadata": dict(item),
                }
            if runtime_case is None:
                runtime_case = (condition[index : index + 1].detach(), dict(item))
    if len(prediction_paths) != prepared.cfg.test_samples:
        raise CrossFrequencyEvaluationError(
            f"expected {prepared.cfg.test_samples} prediction artifacts, got {len(prediction_paths)}"
        )
    overall = grouped.compute_overall()
    rows = grouped.compute_rows()
    if overall["n_samples"] != prepared.cfg.test_samples:
        raise CrossFrequencyEvaluationError("test metric sample count is incomplete")
    if len(rows) != 1 or rows[0]["n_samples"] != prepared.cfg.test_samples:
        raise CrossFrequencyEvaluationError("frequency metric row is incomplete")
    _atomic_canonical_json(
        staging / "metrics_test.json",
        _metrics_test_payload(
            prepared,
            selection,
            selection_sha256,
            overall,
            test_beam_id=4,
            test_config_ids=test_config_ids,
        ),
    )
    _write_csv(staging / "metrics_per_frequency.csv", rows)
    if runtime_case is None:
        raise CrossFrequencyEvaluationError("test loader produced no runtime case")
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
            "test_frequency_hz": prepared.cfg.test_frequency_hz,
            "steering_deg": prepared.cfg.steering_deg,
            "includes_condition_encoding": True,
            "includes_conditional_and_unconditional_cfg_branches": True,
        }
    )
    _atomic_canonical_json(staging / "runtime_generation.json", runtime)
    if set(visualization_cases) != chosen_scenes:
        raise CrossFrequencyEvaluationError(
            "not all deterministic cross-frequency visualization cases were observed"
        )
    for scene_id in sorted(visualization_cases):
        case = visualization_cases[scene_id]
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
            "experiment": "cross_frequency_8x8",
            "array_size": prepared.cfg.array_size,
            "model_size": prepared.cfg.model_size,
            "split": "test",
            "n_samples": prepared.cfg.test_samples,
            "test_frequency_hz": prepared.cfg.test_frequency_hz,
            "steering_deg": prepared.cfg.steering_deg,
            "test_beam_id": 4,
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
    cfg: CrossFrequencyTrainConfig,
    device: torch.device,
    results_root: str | Path,
) -> dict[str, Any]:
    final_dir = Path(results_root)
    if final_dir.exists():
        raise CrossFrequencyEvaluationError(
            f"final cross-frequency result directory already exists: {final_dir}"
        )
    prepared = _prepare_evaluation(cfg, device)
    selection_path = cfg.run_dir / "cfg_selection.json"
    selection = load_cfg_selection(
        selection_path,
        expected_identity=prepared.selection_identity,
        cfg=cfg,
    )
    epoch, best_rmse = _validate_best_checkpoint_state(prepared.trainer_state)
    if int(selection["selected_epoch"]) != epoch or not math.isclose(
        float(selection["best_validation_db_rmse_cfg1"]),
        best_rmse,
        rel_tol=1e-9,
        abs_tol=1e-7,
    ):
        raise CrossFrequencyEvaluationError(
            "CFG selection does not match the best checkpoint"
        )
    selection_sha256 = sha256_file(selection_path)

    def builder(staging: Path) -> None:
        _write_test_transaction(
            staging,
            prepared,
            selection,
            selection_path,
            selection_sha256,
        )

    try:
        published = atomic_result_transaction(final_dir, builder)
    except Exception as error:
        if isinstance(error, CrossFrequencyEvaluationError):
            raise
        raise CrossFrequencyEvaluationError(
            f"cross-frequency test transaction failed: {error}"
        ) from error
    return {
        "status": "complete",
        "result_dir": str(published.resolve()),
        "run_manifest_sha256": sha256_file(published / "run_manifest.json"),
        "cfg_selection_sha256": selection_sha256,
        "selected_scale": selection["selected_scale"],
        "selected_epoch": selection["selected_epoch"],
        "n_samples": cfg.test_samples,
    }


__all__ = [
    "CrossFrequencyEvaluationError",
    "PreparedCrossFrequencyEvaluation",
    "build_cfg_selection_payload",
    "load_cfg_selection",
    "run_cfg_selection",
    "run_test_evaluation",
]
