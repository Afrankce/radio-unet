from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from experiments.multiconfig_manifest import canonical_json_bytes
from training.config import InvocationControls
from training.same_frequency_config import SameFrequencyTrainConfig


MULTISCALE_UNO_MODEL_SIZE = "attention_multiscale_uno_lite"
MULTISCALE_UNO_BACKBONE = "attention_conditioned_multiscale_uno2d"
MULTISCALE_UNO_STATE_CHANNELS = (32, 64, 128, 256, 256)
MULTISCALE_UNO_OPERATOR_WIDTH = 24
MULTISCALE_UNO_OPERATOR_MODES = (12, 12, 8, 4, 4)
MULTISCALE_UNO_OPERATOR_PADDING = (9, 5, 3, 2, 1)
MULTISCALE_UNO_OPERATOR_STAGES = 9
MULTISCALE_UNO_CONDITION_ENCODER = "BasicUNetEncoder_lite"
MULTISCALE_UNO_CONDITION_ENCODER_FEATURES = (32, 32, 64, 128, 256, 32)
MULTISCALE_UNO_ATTENTION_MODULES = 9
MULTISCALE_UNO_CONDITION_INJECTION = "native_scale_CA_SA_encoder_decoder"
MULTISCALE_UNO_DOWNSAMPLE = "avgpool2_plus_1x1"
MULTISCALE_UNO_UPSAMPLE = "bilinear_concat_1x1"
MULTISCALE_UNO_STATE_SKIP_CONNECTIONS = True
MULTISCALE_UNO_CFG_DROP_PROB = 0.25
MULTISCALE_UNO_CFG_CANDIDATES = (1.0,)
MULTISCALE_UNO_TENSOR_PARAMETERS = 3_059_355
MULTISCALE_UNO_REAL_SCALAR_PARAMETERS = 3_925_659


def _exact_type_and_value(actual: Any, expected: Any) -> bool:
    """Compare nested configuration values without Python numeric coercion."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        if len(actual) != len(expected):
            return False
        unmatched = list(actual.items())
        for expected_key, expected_value in expected.items():
            for index, (actual_key, actual_value) in enumerate(unmatched):
                if _exact_type_and_value(actual_key, expected_key):
                    if not _exact_type_and_value(actual_value, expected_value):
                        return False
                    unmatched.pop(index)
                    break
            else:
                return False
        return not unmatched
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _exact_type_and_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


class MultiscaleUNOConfigError(ValueError):
    """The multiscale-UNO configuration differs from its locked registration."""


@dataclass(frozen=True)
class MultiscaleUNOTrainConfig:
    """Multiscale-UNO identity layered over the dense single-beam protocol."""

    base: SameFrequencyTrainConfig
    backbone: str = MULTISCALE_UNO_BACKBONE
    state_channels: tuple[int, int, int, int, int] = MULTISCALE_UNO_STATE_CHANNELS
    operator_width: int = MULTISCALE_UNO_OPERATOR_WIDTH
    operator_modes: tuple[int, int, int, int, int] = MULTISCALE_UNO_OPERATOR_MODES
    operator_padding: tuple[int, int, int, int, int] = MULTISCALE_UNO_OPERATOR_PADDING
    operator_stages: int = MULTISCALE_UNO_OPERATOR_STAGES
    condition_encoder: str = MULTISCALE_UNO_CONDITION_ENCODER
    condition_encoder_features: tuple[int, int, int, int, int, int] = (
        MULTISCALE_UNO_CONDITION_ENCODER_FEATURES
    )
    attention_modules: int = MULTISCALE_UNO_ATTENTION_MODULES
    condition_injection: str = MULTISCALE_UNO_CONDITION_INJECTION
    downsample: str = MULTISCALE_UNO_DOWNSAMPLE
    upsample: str = MULTISCALE_UNO_UPSAMPLE
    state_skip_connections: bool = MULTISCALE_UNO_STATE_SKIP_CONNECTIONS
    cfg_drop_prob: float = MULTISCALE_UNO_CFG_DROP_PROB
    cfg_candidates: tuple[float, ...] = MULTISCALE_UNO_CFG_CANDIDATES
    tensor_parameter_count: int = MULTISCALE_UNO_TENSOR_PARAMETERS
    real_scalar_parameter_count: int = MULTISCALE_UNO_REAL_SCALAR_PARAMETERS

    def __post_init__(self) -> None:
        if not isinstance(self.base, SameFrequencyTrainConfig):
            raise MultiscaleUNOConfigError("base must be a SameFrequencyTrainConfig")
        if self.base.model_size != "lite":
            raise MultiscaleUNOConfigError("base model_size must be 'lite'")
        locked: dict[str, Any] = {
            "backbone": MULTISCALE_UNO_BACKBONE,
            "state_channels": MULTISCALE_UNO_STATE_CHANNELS,
            "operator_width": MULTISCALE_UNO_OPERATOR_WIDTH,
            "operator_modes": MULTISCALE_UNO_OPERATOR_MODES,
            "operator_padding": MULTISCALE_UNO_OPERATOR_PADDING,
            "operator_stages": MULTISCALE_UNO_OPERATOR_STAGES,
            "condition_encoder": MULTISCALE_UNO_CONDITION_ENCODER,
            "condition_encoder_features": MULTISCALE_UNO_CONDITION_ENCODER_FEATURES,
            "attention_modules": MULTISCALE_UNO_ATTENTION_MODULES,
            "condition_injection": MULTISCALE_UNO_CONDITION_INJECTION,
            "downsample": MULTISCALE_UNO_DOWNSAMPLE,
            "upsample": MULTISCALE_UNO_UPSAMPLE,
            "state_skip_connections": MULTISCALE_UNO_STATE_SKIP_CONNECTIONS,
            "cfg_drop_prob": MULTISCALE_UNO_CFG_DROP_PROB,
            "cfg_candidates": MULTISCALE_UNO_CFG_CANDIDATES,
            "tensor_parameter_count": MULTISCALE_UNO_TENSOR_PARAMETERS,
            "real_scalar_parameter_count": MULTISCALE_UNO_REAL_SCALAR_PARAMETERS,
        }
        labels = {"cfg_candidates": "CFG candidates"}
        for name, expected in locked.items():
            actual = getattr(self, name)
            if not _exact_type_and_value(actual, expected):
                raise MultiscaleUNOConfigError(
                    f"{labels.get(name, name)} is locked to {expected!r}, "
                    f"got {actual!r}"
                )

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    @property
    def model_size(self) -> str:
        return MULTISCALE_UNO_MODEL_SIZE

    @property
    def run_dir(self) -> Path:
        return self.base.run_dir

    def with_run_root(self, run_root: str | Path) -> "MultiscaleUNOTrainConfig":
        return MultiscaleUNOTrainConfig(
            replace(self.base, run_root=Path(run_root)),
            backbone=self.backbone,
            state_channels=self.state_channels,
            operator_width=self.operator_width,
            operator_modes=self.operator_modes,
            operator_padding=self.operator_padding,
            operator_stages=self.operator_stages,
            condition_encoder=self.condition_encoder,
            condition_encoder_features=self.condition_encoder_features,
            attention_modules=self.attention_modules,
            condition_injection=self.condition_injection,
            downsample=self.downsample,
            upsample=self.upsample,
            state_skip_connections=self.state_skip_connections,
            cfg_drop_prob=self.cfg_drop_prob,
            cfg_candidates=self.cfg_candidates,
            tensor_parameter_count=self.tensor_parameter_count,
            real_scalar_parameter_count=self.real_scalar_parameter_count,
        )

    def scientific_payload(self) -> dict[str, Any]:
        payload = dict(self.base.scientific_payload())
        payload.update(
            {
                "experiment": "same_frequency_6.7_single_beam_attention_multiscale_uno",
                "model_size": self.model_size,
                "backbone": self.backbone,
                "state_channels": list(self.state_channels),
                "operator_width": self.operator_width,
                "operator_modes": list(self.operator_modes),
                "operator_padding": list(self.operator_padding),
                "operator_stages": self.operator_stages,
                "condition_encoder": self.condition_encoder,
                "condition_encoder_features": list(self.condition_encoder_features),
                "attention_modules": self.attention_modules,
                "condition_injection": self.condition_injection,
                "downsample": self.downsample,
                "upsample": self.upsample,
                "state_skip_connections": self.state_skip_connections,
                "cfg_drop_prob": self.cfg_drop_prob,
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
    def from_json(cls, text: str) -> "MultiscaleUNOTrainConfig":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise MultiscaleUNOConfigError(
                f"run config is not valid JSON: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise MultiscaleUNOConfigError("run config root must be an object")
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
            raise MultiscaleUNOConfigError(
                f"invalid run config fields: {error}"
            ) from error
        expected = template.to_record(controls)
        if set(payload) != set(expected):
            raise MultiscaleUNOConfigError(
                "run config keys mismatch: "
                f"unknown={sorted(set(payload) - set(expected))}, "
                f"missing={sorted(set(expected) - set(payload))}"
            )
        for key, value in expected.items():
            if not _exact_type_and_value(payload[key], value):
                raise MultiscaleUNOConfigError(
                    f"run config {key} mismatch: expected {value!r}, "
                    f"got {payload[key]!r}"
                )
        return template


__all__ = [
    "MULTISCALE_UNO_ATTENTION_MODULES",
    "MULTISCALE_UNO_BACKBONE",
    "MULTISCALE_UNO_CFG_CANDIDATES",
    "MULTISCALE_UNO_CFG_DROP_PROB",
    "MULTISCALE_UNO_CONDITION_ENCODER",
    "MULTISCALE_UNO_CONDITION_ENCODER_FEATURES",
    "MULTISCALE_UNO_CONDITION_INJECTION",
    "MULTISCALE_UNO_DOWNSAMPLE",
    "MULTISCALE_UNO_MODEL_SIZE",
    "MULTISCALE_UNO_OPERATOR_MODES",
    "MULTISCALE_UNO_OPERATOR_PADDING",
    "MULTISCALE_UNO_OPERATOR_STAGES",
    "MULTISCALE_UNO_OPERATOR_WIDTH",
    "MULTISCALE_UNO_REAL_SCALAR_PARAMETERS",
    "MULTISCALE_UNO_STATE_CHANNELS",
    "MULTISCALE_UNO_STATE_SKIP_CONNECTIONS",
    "MULTISCALE_UNO_TENSOR_PARAMETERS",
    "MULTISCALE_UNO_UPSAMPLE",
    "MultiscaleUNOConfigError",
    "MultiscaleUNOTrainConfig",
]
