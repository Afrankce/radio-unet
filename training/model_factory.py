from __future__ import annotations

from typing import Literal

import torch

from config import MODEL_FEATURES
from model.fno import (
    ConditionalFNO2d,
    count_real_scalar_parameters,
    count_tensor_parameters,
)
from model.model import DiffUNet
from model.unet.basic_unet import BasicUNetEncoder
from model.unet.basic_unet_denose import BasicUNetDe
from training.same_frequency_fno_config import (
    PAPER_FNO_MODEL_SIZE,
    PAPER_FNO_REAL_SCALAR_PARAMETERS,
    PAPER_FNO_TENSOR_PARAMETERS,
)


EXPECTED_FEATURES = {
    "lite": (32, 32, 64, 128, 256, 32),
    "large": (128, 128, 256, 512, 1024, 128),
}
EXPECTED_PARAMETER_COUNTS = {
    "lite": 3_994_859,
    "large": 54_126_059,
}
SPARSE_PARAMETER_COUNTS = {
    ("lite", 5): 3_996_011,
}
TASK2_SPARSE_PARAMETER_COUNTS = {
    ("lite", "feature5_mask"): 3_996_011,
}


class FrameworkLockError(RuntimeError):
    """The live RadioFlow architecture differs from the approved benchmark."""


def build_paper_fno() -> ConditionalFNO2d:
    network = ConditionalFNO2d()
    if network.condition_channels != 3:
        raise FrameworkLockError("paper FNO must consume three condition channels")
    if network.width != 40 or (network.modes1, network.modes2) != (12, 12):
        raise FrameworkLockError("paper FNO width or retained modes changed")
    if network.padding != 9 or len(network.spectral_layers) != 4:
        raise FrameworkLockError("paper FNO padding or layer count changed")
    if network.cfg_drop_prob != 0.25:
        raise FrameworkLockError("paper FNO CFG dropout must remain 0.25")
    if count_tensor_parameters(network) != PAPER_FNO_TENSOR_PARAMETERS:
        raise FrameworkLockError("paper FNO tensor parameter count changed")
    if count_real_scalar_parameters(network) != PAPER_FNO_REAL_SCALAR_PARAMETERS:
        raise FrameworkLockError("paper FNO real scalar parameter count changed")
    return network


def build_same_frequency_backbone(model_size: str) -> torch.nn.Module:
    if model_size == PAPER_FNO_MODEL_SIZE:
        return build_paper_fno()
    if model_size in EXPECTED_PARAMETER_COUNTS:
        return build_locked_radioflow(model_size)
    raise FrameworkLockError(
        f"unsupported same-frequency model size: {model_size!r}"
    )


def build_locked_radioflow(
    model_size: Literal["lite", "large"],
) -> DiffUNet:
    if model_size not in EXPECTED_PARAMETER_COUNTS:
        raise FrameworkLockError(f"unsupported locked model size: {model_size!r}")
    if tuple(MODEL_FEATURES.get(model_size, ())) != EXPECTED_FEATURES[model_size]:
        raise FrameworkLockError(
            f"RadioFlow feature tuple changed for {model_size}: "
            f"{MODEL_FEATURES.get(model_size)!r}"
        )
    activation_checkpointing = model_size == "large"
    network = DiffUNet(
        con_channels=3,
        model_size=model_size,
        activation_checkpointing=activation_checkpointing,
    )
    if type(network.embed_model) is not BasicUNetEncoder:
        raise FrameworkLockError("unexpected condition encoder")
    if type(network.model) is not BasicUNetDe:
        raise FrameworkLockError("unexpected velocity decoder")
    if network.cfg_drop_prob != 0.25:
        raise FrameworkLockError("RadioFlow CFG dropout must remain 0.25")
    if network.activation_checkpointing is not activation_checkpointing:
        raise FrameworkLockError("activation-checkpoint policy changed")
    if network.embed_model.activation_checkpointing is not activation_checkpointing:
        raise FrameworkLockError("encoder checkpoint policy changed")
    if network.model.activation_checkpointing is not activation_checkpointing:
        raise FrameworkLockError("decoder checkpoint policy changed")
    actual = sum(parameter.numel() for parameter in network.parameters())
    expected = EXPECTED_PARAMETER_COUNTS[model_size]
    if actual != expected:
        raise FrameworkLockError(
            f"parameter count changed: expected {expected}, got {actual}"
        )
    return network


def build_sparse_radioflow(
    *,
    variant: Literal["beam_masked", "no_beam_masked"],
    model_size: Literal["lite", "large"] = "lite",
) -> DiffUNet:
    if variant != "beam_masked":
        raise FrameworkLockError(
            "sparse RadioFlow factory only supports the formal 'beam_masked' variant"
        )
    if model_size != "lite":
        raise FrameworkLockError("sparse RadioFlow factory is locked to model_size='lite'")
    if tuple(MODEL_FEATURES.get("lite", ())) != EXPECTED_FEATURES["lite"]:
        raise FrameworkLockError(
            f"RadioFlow feature tuple changed for lite: {MODEL_FEATURES.get('lite')!r}"
        )
    network = DiffUNet(
        con_channels=5,
        model_size="lite",
        activation_checkpointing=False,
    )
    if type(network.embed_model) is not BasicUNetEncoder:
        raise FrameworkLockError("unexpected condition encoder")
    if type(network.model) is not BasicUNetDe:
        raise FrameworkLockError("unexpected velocity decoder")
    if network.embed_model.conv_0.conv_0.conv.in_channels != 5:
        raise FrameworkLockError("sparse condition encoder must consume 5 channels")
    if network.cfg_drop_prob != 0.25:
        raise FrameworkLockError("RadioFlow CFG dropout must remain 0.25")
    if network.activation_checkpointing is not False:
        raise FrameworkLockError("sparse lite model must not enable activation checkpointing")
    if network.embed_model.activation_checkpointing is not False:
        raise FrameworkLockError("sparse lite encoder must not enable activation checkpointing")
    if network.model.activation_checkpointing is not False:
        raise FrameworkLockError("sparse lite decoder must not enable activation checkpointing")
    actual = sum(parameter.numel() for parameter in network.parameters())
    expected = SPARSE_PARAMETER_COUNTS[("lite", 5)]
    if actual != expected:
        raise FrameworkLockError(
            f"sparse parameter count changed: expected {expected}, got {actual}"
        )
    return network


def build_task2_sparse_radioflow(
    *,
    condition_variant: Literal["feature5_mask"] = "feature5_mask",
    model_size: Literal["lite"] = "lite",
) -> DiffUNet:
    """Build the locked five-channel model for the single-beam Task 2 run."""

    if condition_variant != "feature5_mask":
        raise FrameworkLockError(
            "Task 2 sparse factory is locked to condition_variant='feature5_mask'"
        )
    if model_size != "lite":
        raise FrameworkLockError("Task 2 sparse factory is locked to model_size='lite'")
    if tuple(MODEL_FEATURES.get("lite", ())) != EXPECTED_FEATURES["lite"]:
        raise FrameworkLockError(
            f"RadioFlow feature tuple changed for lite: {MODEL_FEATURES.get('lite')!r}"
        )
    network = DiffUNet(
        con_channels=5,
        model_size="lite",
        activation_checkpointing=False,
    )
    if type(network.embed_model) is not BasicUNetEncoder:
        raise FrameworkLockError("unexpected condition encoder")
    if type(network.model) is not BasicUNetDe:
        raise FrameworkLockError("unexpected velocity decoder")
    if network.embed_model.conv_0.conv_0.conv.in_channels != 5:
        raise FrameworkLockError("Task 2 condition encoder must consume 5 channels")
    if network.cfg_drop_prob != 0.25:
        raise FrameworkLockError("RadioFlow CFG dropout must remain 0.25")
    if network.activation_checkpointing is not False:
        raise FrameworkLockError("Task 2 Lite model must not enable activation checkpointing")
    if network.embed_model.activation_checkpointing is not False:
        raise FrameworkLockError("Task 2 Lite encoder must not enable activation checkpointing")
    if network.model.activation_checkpointing is not False:
        raise FrameworkLockError("Task 2 Lite decoder must not enable activation checkpointing")
    actual = sum(parameter.numel() for parameter in network.parameters())
    expected = TASK2_SPARSE_PARAMETER_COUNTS[("lite", "feature5_mask")]
    if actual != expected:
        raise FrameworkLockError(
            f"Task 2 sparse parameter count changed: expected {expected}, got {actual}"
        )
    return network

