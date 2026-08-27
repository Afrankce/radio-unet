from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from experiments.multiconfig_manifest import canonical_json_bytes
from training.config import InvocationControls
from training.same_frequency_config import SameFrequencyTrainConfig


ATTENTION_FNO_MODEL_SIZE = "attention_fno_lite"
ATTENTION_FNO_BACKBONE = "attention_conditioned_full_resolution_fno2d"
ATTENTION_FNO_WIDTH = 40
ATTENTION_FNO_MODES = (12, 12)
ATTENTION_FNO_PADDING = 9
ATTENTION_FNO_LAYERS = 4
ATTENTION_FNO_CFG_CANDIDATES = (1.0,)
ATTENTION_FNO_TENSOR_PARAMETERS = 3_487_273
ATTENTION_FNO_REAL_SCALAR_PARAMETERS = 5_330_473


class AttentionFNOConfigError(ValueError):
    """The attention-FNO configuration differs from its locked registration."""


@dataclass(frozen=True)
class AttentionFNOTrainConfig:
    """Attention-FNO identity layered over the dense single-beam protocol."""

    base: SameFrequencyTrainConfig
    backbone: str = ATTENTION_FNO_BACKBONE
    fno_width: int = ATTENTION_FNO_WIDTH
    fno_modes: tuple[int, int] = ATTENTION_FNO_MODES
    fno_padding: int = ATTENTION_FNO_PADDING
    fno_layers: int = ATTENTION_FNO_LAYERS
    cfg_candidates: tuple[float, ...] = ATTENTION_FNO_CFG_CANDIDATES
    tensor_parameter_count: int = ATTENTION_FNO_TENSOR_PARAMETERS
    real_scalar_parameter_count: int = ATTENTION_FNO_REAL_SCALAR_PARAMETERS

    def __post_init__(self) -> None:
        if not isinstance(self.base, SameFrequencyTrainConfig):
            raise AttentionFNOConfigError("base must be a SameFrequencyTrainConfig")
        if self.base.model_size != "lite":
            raise AttentionFNOConfigError("base model_size must be 'lite'")
        locked: dict[str, Any] = {
            "backbone": ATTENTION_FNO_BACKBONE,
            "fno_width": ATTENTION_FNO_WIDTH,
            "fno_modes": ATTENTION_FNO_MODES,
            "fno_padding": ATTENTION_FNO_PADDING,
            "fno_layers": ATTENTION_FNO_LAYERS,
            "cfg_candidates": ATTENTION_FNO_CFG_CANDIDATES,
            "tensor_parameter_count": ATTENTION_FNO_TENSOR_PARAMETERS,
            "real_scalar_parameter_count": ATTENTION_FNO_REAL_SCALAR_PARAMETERS,
        }
        labels = {
            "fno_width": "width",
            "fno_modes": "modes",
            "cfg_candidates": "CFG candidates",
        }
        for name, expected in locked.items():
            actual = getattr(self, name)
            if actual != expected or type(actual) is not type(expected):
                raise AttentionFNOConfigError(
                    f"{labels.get(name, name)} is locked to {expected!r}, "
                    f"got {actual!r}"
                )

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    @property
    def model_size(self) -> str:
        return ATTENTION_FNO_MODEL_SIZE

    @property
    def run_dir(self) -> Path:
        return self.base.run_dir

    def with_run_root(self, run_root: str | Path) -> "AttentionFNOTrainConfig":
        return AttentionFNOTrainConfig(
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
                "experiment": "same_frequency_6.7_single_beam_attention_fno",
                "model_size": self.model_size,
                "backbone": self.backbone,
                "condition_encoder": "BasicUNetEncoder_lite",
                "condition_encoder_features": [32, 32, 64, 128, 256, 32],
                "condition_aggregation": "project_resize_sum",
                "attention": "RadioFlow_CA_SA_each_block",
                "time_conditioning": "sinusoidal_mlp_each_block",
                "fno_width": self.fno_width,
                "fno_modes": list(self.fno_modes),
                "fno_padding": self.fno_padding,
                "fno_layers": self.fno_layers,
                "fno_activation": "gelu_all_four_blocks",
                "fno_projection": [self.fno_width, 128, 1],
                "fno_dense_complex_weights": True,
                "fno_input_order": [
                    "x_t",
                    "tx_mask",
                    "height",
                    "beam_map",
                    "grid_x",
                    "grid_y",
                ],
                "state_downsampling": False,
                "state_upsampling": False,
                "state_skip_connections": False,
                "cfg_drop_prob": 0.25,
                "cfg_dropout_scope": "raw_and_encoded_condition",
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
    def from_json(cls, text: str) -> "AttentionFNOTrainConfig":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise AttentionFNOConfigError(
                f"run config is not valid JSON: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise AttentionFNOConfigError("run config root must be an object")
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
            raise AttentionFNOConfigError(
                f"invalid run config fields: {error}"
            ) from error
        expected = template.to_record(controls)
        if set(payload) != set(expected):
            raise AttentionFNOConfigError(
                "run config keys mismatch: "
                f"unknown={sorted(set(payload) - set(expected))}, "
                f"missing={sorted(set(expected) - set(payload))}"
            )
        for key, value in expected.items():
            if payload[key] != value:
                raise AttentionFNOConfigError(
                    f"run config {key} mismatch: expected {value!r}, "
                    f"got {payload[key]!r}"
                )
        return template


__all__ = [
    "ATTENTION_FNO_BACKBONE",
    "ATTENTION_FNO_CFG_CANDIDATES",
    "ATTENTION_FNO_LAYERS",
    "ATTENTION_FNO_MODEL_SIZE",
    "ATTENTION_FNO_MODES",
    "ATTENTION_FNO_PADDING",
    "ATTENTION_FNO_REAL_SCALAR_PARAMETERS",
    "ATTENTION_FNO_TENSOR_PARAMETERS",
    "ATTENTION_FNO_WIDTH",
    "AttentionFNOConfigError",
    "AttentionFNOTrainConfig",
]

