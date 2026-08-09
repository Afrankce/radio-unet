from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch import Tensor

from data_loaders.multiconfig import denormalize_db


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


POWER_DB_RANGE = (-300.0, 0.0)
ERROR_DB_RANGE = (0.0, 300.0)
INVALID_COLOR = "#777777"


class VisualizationInputError(ValueError):
    """A saved or rendered benchmark case does not satisfy the locked format."""


def _cpu_tensor(value: Tensor, *, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise VisualizationInputError(f"{label} must be a torch tensor")
    return value.detach().to(device="cpu")


def _single_channel(value: Tensor, *, label: str) -> Tensor:
    value = _cpu_tensor(value, label=label)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise VisualizationInputError(
            f"{label} must have shape [1,H,W] or [H,W], got {tuple(value.shape)}"
        )
    return value


def _condition_tensor(condition: Tensor) -> Tensor:
    condition = _cpu_tensor(condition, label="condition")
    if condition.ndim == 4 and condition.shape[0] == 1:
        condition = condition[0]
    if condition.ndim != 3 or condition.shape[0] != 3:
        raise VisualizationInputError(
            "condition must have shape [3,H,W] or [1,3,H,W]"
        )
    return condition


def prepare_visualization_arrays(
    condition: Tensor,
    target: Tensor,
    prediction: Tensor,
    valid_mask: Tensor,
) -> dict[str, np.ndarray | np.ma.MaskedArray]:
    """Convert one normalized benchmark case to fixed-range display arrays."""

    condition_cpu = _condition_tensor(condition).to(dtype=torch.float32)
    target_cpu = _single_channel(target, label="target").to(dtype=torch.float32)
    prediction_cpu = _single_channel(prediction, label="prediction").to(
        dtype=torch.float32
    )
    mask_cpu = _single_channel(valid_mask, label="valid_mask")
    if mask_cpu.dtype != torch.bool:
        raise VisualizationInputError("valid_mask must have boolean dtype")
    spatial_shape = tuple(condition_cpu.shape[-2:])
    if (
        tuple(target_cpu.shape) != spatial_shape
        or tuple(prediction_cpu.shape) != spatial_shape
        or tuple(mask_cpu.shape) != spatial_shape
    ):
        raise VisualizationInputError("all visualization tensors must share H and W")
    if not bool(mask_cpu.any()):
        raise VisualizationInputError("valid_mask must contain a valid pixel")
    if not bool(torch.isfinite(condition_cpu).all()):
        raise VisualizationInputError("condition contains a non-finite value")
    if not bool(torch.isfinite(target_cpu[mask_cpu]).all()):
        raise VisualizationInputError("target contains a non-finite valid value")
    if not bool(torch.isfinite(prediction_cpu[mask_cpu]).all()):
        raise VisualizationInputError("prediction contains a non-finite valid value")

    target_db = denormalize_db(target_cpu.clamp(0.0, 1.0))
    prediction_db = denormalize_db(prediction_cpu.clamp(0.0, 1.0))
    invalid = (~mask_cpu).numpy()
    target_numpy = target_db.numpy()
    prediction_numpy = prediction_db.numpy()
    return {
        "tx_mask": condition_cpu[0].numpy(),
        "height": condition_cpu[1].numpy(),
        "beam_db": denormalize_db(condition_cpu[2].clamp(0.0, 1.0)).numpy(),
        "ground_truth_db": np.ma.array(target_numpy, mask=invalid, copy=False),
        "prediction_db": np.ma.array(prediction_numpy, mask=invalid, copy=False),
        "absolute_error_db": np.ma.array(
            np.abs(prediction_numpy - target_numpy),
            mask=invalid,
            copy=False,
        ),
    }


def _metadata_json(metadata: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            dict(metadata),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise VisualizationInputError(f"metadata is not canonical JSON: {error}") from error


def _numpy_channel(value: Tensor, *, label: str) -> np.ndarray:
    return _single_channel(value, label=label).unsqueeze(0).numpy()


def save_prediction_npz(
    path: str | Path,
    *,
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically save one compressed, pickle-free prediction artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prediction_array = _numpy_channel(prediction, label="prediction")
    target_array = _numpy_channel(target, label="target")
    mask_tensor = _single_channel(valid_mask, label="valid_mask")
    if mask_tensor.dtype != torch.bool:
        raise VisualizationInputError("valid_mask must have boolean dtype")
    mask_array = mask_tensor.unsqueeze(0).numpy()
    if prediction_array.shape != target_array.shape or prediction_array.shape != mask_array.shape:
        raise VisualizationInputError("prediction, target, and valid_mask shapes must match")
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                prediction=prediction_array,
                target=target_array,
                valid_mask=mask_array,
                metadata_json=np.asarray(_metadata_json(metadata)),
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _safe_token(value: Any, *, label: str) -> str:
    token = str(value)
    if not token or re.fullmatch(r"[A-Za-z0-9_.-]+", token) is None:
        raise VisualizationInputError(f"unsafe {label} token: {token!r}")
    return token


def _compact_number(value: float) -> str:
    return f"{float(value):g}"


def stable_case_stem(
    metadata: Mapping[str, Any],
    *,
    model_size: str,
    cfg_scale: float,
) -> str:
    """Build a deterministic filename from the case identity and inference lock."""

    try:
        scene_id = _safe_token(metadata["scene_id"], label="scene ID")
        array_name = _safe_token(metadata["array_name"], label="array name")
        beam_id = int(metadata["beam_id"])
        angle = float(metadata["steering_deg"])
    except (KeyError, TypeError, ValueError) as error:
        raise VisualizationInputError("case metadata is incomplete") from error
    if beam_id < 0 or beam_id > 99 or not np.isfinite(angle):
        raise VisualizationInputError("beam ID or steering angle is invalid")
    model_token = _safe_token(str(model_size).lower(), label="model size")
    if not np.isfinite(float(cfg_scale)):
        raise VisualizationInputError("CFG scale must be finite")
    angle_token = f"{'p' if angle >= 0.0 else 'm'}{abs(angle):04.1f}"
    return (
        f"{scene_id}__{array_name}__beam{beam_id:02d}__angle_{angle_token}"
        f"__{model_token}__cfg{_compact_number(cfg_scale)}"
    )


def _power_cmap():
    colormap = matplotlib.colormaps["viridis"].copy()
    colormap.set_bad(INVALID_COLOR)
    return colormap


def _error_cmap():
    colormap = matplotlib.colormaps["magma"].copy()
    colormap.set_bad(INVALID_COLOR)
    return colormap


def _case_title(
    metadata: Mapping[str, Any], *, model_size: str, cfg_scale: float
) -> str:
    frequency_hz = metadata.get("frequency_hz")
    frequency_label = ""
    if frequency_hz is not None:
        try:
            frequency_label = f"{float(frequency_hz) / 1_000_000_000:g} GHz | "
        except (TypeError, ValueError):
            raise VisualizationInputError("frequency_hz must be numeric when provided")
    return (
        f"{frequency_label}{metadata.get('scene_id', '?')} | "
        f"{metadata.get('array_name', '?')} | "
        f"beam {int(metadata.get('beam_id', -1)):02d} "
        f"({float(metadata.get('steering_deg', float('nan'))):g} deg) | "
        f"{str(model_size).title()} | CFG {_compact_number(cfg_scale)}"
    )


def _save_figure_atomic(figure: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        figure.savefig(temporary, format="png", dpi=160, bbox_inches="tight")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def render_comparison(
    path: str | Path,
    *,
    condition: Tensor,
    target: Tensor,
    prediction: Tensor,
    valid_mask: Tensor,
    metadata: Mapping[str, Any],
    model_size: str,
    cfg_scale: float,
) -> Path:
    arrays = prepare_visualization_arrays(condition, target, prediction, valid_mask)
    figure, axes = plt.subplots(1, 5, figsize=(19, 4.2), constrained_layout=True)
    height_image = axes[0].imshow(arrays["height"], cmap="gray", vmin=0.0, vmax=1.0)
    tx_rows, tx_columns = np.nonzero(np.asarray(arrays["tx_mask"]) > 0.0)
    if len(tx_rows):
        axes[0].scatter(tx_columns, tx_rows, marker="+", s=70, c="#ff3b30", linewidths=1.5)
    figure.colorbar(height_image, ax=axes[0], fraction=0.046, pad=0.04)
    beam_image = axes[1].imshow(
        arrays["beam_db"], cmap=_power_cmap(), vmin=POWER_DB_RANGE[0], vmax=POWER_DB_RANGE[1]
    )
    figure.colorbar(beam_image, ax=axes[1], fraction=0.046, pad=0.04)
    for axis, key in zip(
        axes[2:4], ("ground_truth_db", "prediction_db"), strict=True
    ):
        image = axis.imshow(
            arrays[key], cmap=_power_cmap(), vmin=POWER_DB_RANGE[0], vmax=POWER_DB_RANGE[1]
        )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    error_image = axes[4].imshow(
        arrays["absolute_error_db"],
        cmap=_error_cmap(),
        vmin=ERROR_DB_RANGE[0],
        vmax=ERROR_DB_RANGE[1],
    )
    figure.colorbar(error_image, ax=axes[4], fraction=0.046, pad=0.04)
    for axis, title in zip(
        axes,
        ("Height", "Beam map (dB)", "Ground truth (dB)", "Prediction (dB)", "Absolute error (dB)"),
        strict=True,
    ):
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(_case_title(metadata, model_size=model_size, cfg_scale=cfg_scale))
    destination = Path(path)
    try:
        _save_figure_atomic(figure, destination)
    finally:
        plt.close(figure)
    return destination


def render_error_map(
    path: str | Path,
    *,
    target: Tensor,
    prediction: Tensor,
    valid_mask: Tensor,
    metadata: Mapping[str, Any],
    model_size: str,
    cfg_scale: float,
) -> Path:
    target_cpu = _single_channel(target, label="target")
    spatial_shape = target_cpu.shape
    condition = torch.zeros((3, *spatial_shape), dtype=torch.float32)
    arrays = prepare_visualization_arrays(condition, target_cpu, prediction, valid_mask)
    figure, axis = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    image = axis.imshow(
        arrays["absolute_error_db"],
        cmap=_error_cmap(),
        vmin=ERROR_DB_RANGE[0],
        vmax=ERROR_DB_RANGE[1],
    )
    figure.colorbar(image, ax=axis, label="Absolute error (dB)")
    axis.set_title(_case_title(metadata, model_size=model_size, cfg_scale=cfg_scale))
    axis.set_xticks([])
    axis.set_yticks([])
    destination = Path(path)
    try:
        _save_figure_atomic(figure, destination)
    finally:
        plt.close(figure)
    return destination
