from __future__ import annotations

from typing import Literal

from config import MODEL_FEATURES
from model.model import DiffUNet
from model.unet.basic_unet import BasicUNetEncoder
from model.unet.basic_unet_denose import BasicUNetDe


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


class FrameworkLockError(RuntimeError):
    """The live RadioFlow architecture differs from the approved benchmark."""


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

