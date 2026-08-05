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

