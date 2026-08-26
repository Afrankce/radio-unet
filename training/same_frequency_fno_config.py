from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from experiments.multiconfig_manifest import canonical_json_bytes
from training.config import InvocationControls
from training.same_frequency_config import SameFrequencyTrainConfig


PAPER_FNO_MODEL_SIZE = "paper_fno_lite"
PAPER_FNO_BACKBONE = "paper_fno2d"
PAPER_FNO_WIDTH = 40
PAPER_FNO_MODES = (12, 12)
PAPER_FNO_PADDING = 9
PAPER_FNO_LAYERS = 4
PAPER_FNO_CFG_CANDIDATES = (1.0,)
PAPER_FNO_TENSOR_PARAMETERS = 1_855_457
PAPER_FNO_REAL_SCALAR_PARAMETERS = 3_698_657


class PaperFNOConfigError(ValueError):
    """The paper-FNO scientific configuration differs from its registration."""


@dataclass(frozen=True)
class PaperFNOTrainConfig:
    """FNO identity layered over the unchanged same-frequency Lite controls."""

    base: SameFrequencyTrainConfig
    backbone: str = PAPER_FNO_BACKBONE
    fno_width: int = PAPER_FNO_WIDTH
    fno_modes: tuple[int, int] = PAPER_FNO_MODES
    fno_padding: int = PAPER_FNO_PADDING
    fno_layers: int = PAPER_FNO_LAYERS
    cfg_candidates: tuple[float, ...] = PAPER_FNO_CFG_CANDIDATES
    tensor_parameter_count: int = PAPER_FNO_TENSOR_PARAMETERS
    real_scalar_parameter_count: int = PAPER_FNO_REAL_SCALAR_PARAMETERS

    def __post_init__(self) -> None:
        if not isinstance(self.base, SameFrequencyTrainConfig):
            raise PaperFNOConfigError("base must be a SameFrequencyTrainConfig")
        if self.base.model_size != "lite":
            raise PaperFNOConfigError("base model_size must be 'lite'")
        locked: dict[str, Any] = {
            "backbone": PAPER_FNO_BACKBONE,
            "fno_width": PAPER_FNO_WIDTH,
            "fno_modes": PAPER_FNO_MODES,
            "fno_padding": PAPER_FNO_PADDING,
            "fno_layers": PAPER_FNO_LAYERS,
            "cfg_candidates": PAPER_FNO_CFG_CANDIDATES,
            "tensor_parameter_count": PAPER_FNO_TENSOR_PARAMETERS,
            "real_scalar_parameter_count": PAPER_FNO_REAL_SCALAR_PARAMETERS,
        }
        labels = {
            "fno_width": "width",
            "fno_modes": "modes",
            "cfg_candidates": "CFG candidates",
        }
        for name, expected in locked.items():
            actual = getattr(self, name)
            if actual != expected or type(actual) is not type(expected):
                raise PaperFNOConfigError(
                    f"{labels.get(name, name)} is locked to {expected!r}, "
                    f"got {actual!r}"
                )

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    @property
    def model_size(self) -> str:
        return PAPER_FNO_MODEL_SIZE

    @property
    def run_dir(self) -> Path:
        return self.base.run_dir

    def with_run_root(self, run_root: str | Path) -> "PaperFNOTrainConfig":
        return PaperFNOTrainConfig(
            replace(self.base, run_root=Path(run_root)),
            backbone=self.backbone,
            fno_width=self.fno_width,
            fno_modes=self.fno_modes,
            fno_padding=self.fno_padding,
            fno_layers=self.fno_layers,
            cfg_candidates=self.cfg_candidates,
            tensor_parameter_count=self.tensor_parameter_count,
            real_scalar_parameter_count=self.real_scalar_parameter_count,
        )

    def scientific_payload(self) -> dict[str, Any]:
        payload = dict(self.base.scientific_payload())
        payload.update(
            {
                "experiment": "same_frequency_6.7_single_beam_paper_fno",
                "model_size": self.model_size,
                "backbone": self.backbone,
                "fno_width": self.fno_width,
                "fno_modes": list(self.fno_modes),
                "fno_padding": self.fno_padding,
                "fno_layers": self.fno_layers,
                "fno_normalization": "none",
                "fno_activation": "gelu_first_three_blocks",
                "fno_projection": [self.fno_width, 128, 1],
                "fno_dense_complex_weights": True,
                "fno_input_order": [
                    "x_t",
                    "tx_mask",
                    "height",
                    "beam_map",
                    "t_map",
                    "grid_x",
                    "grid_y",
                ],
                "cfg_drop_prob": 0.25,
                "cfg_candidates": list(self.cfg_candidates),
                "tensor_parameter_count": self.tensor_parameter_count,
                "real_scalar_parameter_count": self.real_scalar_parameter_count,
                "fft_precision": "float32",
            }
        )
        return payload

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.scientific_payload())
        ).hexdigest()

    def to_record(
        self,
        invocation: InvocationControls | None = None,
    ) -> dict[str, Any]:
        controls = invocation or InvocationControls()
        return {
            **self.scientific_payload(),
            "dataset_root": str(self.dataset_root.resolve()),
            "manifest_path": str(self.manifest_path.resolve()),
            "height_stats_path": str(self.height_stats_path.resolve()),
            "run_root": str(self.run_root.resolve()),
            "config_sha256": self.config_sha256,
            "invocation": controls.to_dict(),
        }

    @classmethod
    def from_json(cls, text: str) -> "PaperFNOTrainConfig":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise PaperFNOConfigError(
                f"run config is not valid JSON: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise PaperFNOConfigError("run config root must be an object")
        try:
            controls = InvocationControls.from_dict(payload["invocation"])
            template = cls(
                SameFrequencyTrainConfig(
                    dataset_root=payload["dataset_root"],
                    manifest_path=payload["manifest_path"],
                    height_stats_path=payload["height_stats_path"],
                    run_root=payload["run_root"],
                    array_size=payload["array_size"],
                    beam_id=int(payload["beam_id"]),
                    model_size="lite",
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PaperFNOConfigError(f"invalid run config fields: {error}") from error
        expected = template.to_record(controls)
        if set(payload) != set(expected):
            raise PaperFNOConfigError(
                "run config keys mismatch: "
                f"unknown={sorted(set(payload) - set(expected))}, "
                f"missing={sorted(set(expected) - set(payload))}"
            )
        for key, value in expected.items():
            if payload[key] != value:
                raise PaperFNOConfigError(
                    f"run config {key} mismatch: expected {value!r}, "
                    f"got {payload[key]!r}"
                )
        return template


__all__ = [
    "PAPER_FNO_BACKBONE",
    "PAPER_FNO_CFG_CANDIDATES",
    "PAPER_FNO_LAYERS",
    "PAPER_FNO_MODEL_SIZE",
    "PAPER_FNO_MODES",
    "PAPER_FNO_PADDING",
    "PAPER_FNO_REAL_SCALAR_PARAMETERS",
    "PAPER_FNO_TENSOR_PARAMETERS",
    "PAPER_FNO_WIDTH",
    "PaperFNOConfigError",
    "PaperFNOTrainConfig",
]

