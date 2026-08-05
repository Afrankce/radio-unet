from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch

from config import MODEL_FEATURES
from model.model import DiffUNet
from model.unet.basic_unet import BasicUNetEncoder
from model.unet.basic_unet_denose import BasicUNetDe


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_checkout_descends_from_locked_radioflow_source() -> None:
    provenance = importlib.import_module("experiments.provenance")

    checkout = provenance.assert_radioflow_checkout(REPO_ROOT)

    assert checkout.origin_url == "https://github.com/Hxxxz0/RadioFlow.git"
    assert (
        checkout.upstream_base
        == "8944e3160f6a7a85b5451ae58e337186a4d98771"
    )
    assert len(checkout.head_commit) == 40


def test_sha256_file_matches_hand_computed_digest(tmp_path: Path) -> None:
    provenance = importlib.import_module("experiments.provenance")
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abc")

    assert provenance.sha256_file(payload) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


@pytest.mark.parametrize(
    ("model_size", "expected_features", "expected_parameters"),
    [
        ("lite", (32, 32, 64, 128, 256, 32), 3_994_859),
        ("large", (128, 128, 256, 512, 1024, 128), 54_126_059),
    ],
)
def test_three_channel_model_is_locked_radioflow_diffunet(
    model_size: str,
    expected_features: tuple[int, int, int, int, int, int],
    expected_parameters: int,
) -> None:
    network = DiffUNet(con_channels=3, model_size=model_size)

    assert MODEL_FEATURES[model_size] == expected_features
    assert type(network) is DiffUNet
    assert type(network.embed_model) is BasicUNetEncoder
    assert type(network.model) is BasicUNetDe
    assert sum(parameter.numel() for parameter in network.parameters()) == (
        expected_parameters
    )


def test_benchmark_factory_locks_three_condition_channels() -> None:
    from training.model_factory import build_locked_radioflow

    network = build_locked_radioflow("lite")

    assert network.embed_model.conv_0.conv_0.conv.in_channels == 3
    assert network.activation_checkpointing is False

def test_channel_attention_uses_deterministic_global_max_pool() -> None:
    """Global max pooling must avoid CUDA-nondeterministic AdaptiveMaxPool2d."""

    from model.unet.basic_unet_denose import ChannelAttention

    module = ChannelAttention(in_channels=16)
    assert not any(
        isinstance(child, torch.nn.AdaptiveMaxPool2d)
        for child in module.modules()
    )
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        tensor = torch.randn(2, 16, 8, 8, requires_grad=True)
        output = module(tensor)
        output.square().mean().backward()
        assert tensor.grad is not None
    finally:
        torch.use_deterministic_algorithms(previous)
