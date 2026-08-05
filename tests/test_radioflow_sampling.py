from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest
import torch


def _sampling_module():
    from evaluation import radioflow_sampling

    return radioflow_sampling


class RecordingModel:
    def __init__(self) -> None:
        self.embed_calls = 0
        self.forward_calls = 0
        self.cfg_calls: list[dict[str, object]] = []

    def embed_model(self, condition: torch.Tensor) -> list[torch.Tensor]:
        self.embed_calls += 1
        return [condition.mean(dim=1, keepdim=True)]

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        raise AssertionError("ordinary forward must not be used for CFG sampling")

    def forward_with_cfg(
        self,
        *,
        image: torch.Tensor,
        x: torch.Tensor,
        step: torch.Tensor,
        embedding: list[torch.Tensor],
        cfg_scale: float,
    ) -> torch.Tensor:
        self.cfg_calls.append(
            {
                "image": image,
                "step": step.detach().clone(),
                "embedding": embedding,
                "cfg_scale": cfg_scale,
            }
        )
        return torch.ones_like(x) * float(cfg_scale)


def test_two_step_euler_uses_radioflow_cfg_and_one_embedding() -> None:
    module = _sampling_module()
    model = RecordingModel()
    condition = torch.randn(2, 3, 8, 8)
    noise = torch.zeros(2, 1, 8, 8)

    result = module.euler_cfg_sample(
        model,
        condition,
        noise,
        cfg_scale=2.5,
        steps=2,
        use_amp=False,
    )

    assert model.embed_calls == 1
    assert model.forward_calls == 0
    assert len(model.cfg_calls) == 2
    assert torch.equal(model.cfg_calls[0]["step"], torch.tensor([0.0, 0.0]))
    assert torch.equal(model.cfg_calls[1]["step"], torch.tensor([0.5, 0.5]))
    assert model.cfg_calls[0]["embedding"] is model.cfg_calls[1]["embedding"]
    assert torch.allclose(result, torch.full_like(result, 2.5))
    assert result.dtype == torch.float32


def test_cfg_scale_changes_generated_output() -> None:
    module = _sampling_module()
    condition = torch.zeros(1, 3, 8, 8)
    noise = torch.zeros(1, 1, 8, 8)

    weak = module.euler_cfg_sample(
        RecordingModel(), condition, noise, cfg_scale=1.0, use_amp=False
    )
    strong = module.euler_cfg_sample(
        RecordingModel(), condition, noise, cfg_scale=2.0, use_amp=False
    )

    assert not torch.equal(weak, strong)
    assert torch.allclose(weak, torch.ones_like(weak))
    assert torch.allclose(strong, torch.full_like(strong, 2.0))


@pytest.mark.parametrize("steps", [0, -1])
def test_euler_rejects_nonpositive_steps(steps: int) -> None:
    module = _sampling_module()

    with pytest.raises(ValueError, match="steps must be positive"):
        module.euler_cfg_sample(
            RecordingModel(),
            torch.zeros(1, 3, 8, 8),
            torch.zeros(1, 1, 8, 8),
            cfg_scale=1.0,
            steps=steps,
            use_amp=False,
        )


def test_sample_noise_depends_only_on_scene_angle_and_base_seed() -> None:
    module = _sampling_module()
    reference = module.make_sample_noise("u731", -14.0, shape=(1, 8, 8))

    for _array in ("8x8", "16x16", "32x32"):
        for _size in ("lite", "large"):
            for _cfg in (1.0, 1.5, 2.0, 2.5):
                assert torch.equal(
                    reference,
                    module.make_sample_noise("u731", -14.0, shape=(1, 8, 8)),
                )
    assert not torch.equal(
        reference,
        module.make_sample_noise("u731", -7.0, shape=(1, 8, 8)),
    )
    assert not torch.equal(
        reference,
        module.make_sample_noise("u732", -14.0, shape=(1, 8, 8)),
    )


def test_sample_noise_is_stable_across_processes() -> None:
    module = _sampling_module()
    expected = hashlib.sha256(
        module.make_sample_noise("u731", 21.0, shape=(1, 8, 8)).numpy().tobytes()
    ).hexdigest()
    script = """
import hashlib, json
from evaluation.radioflow_sampling import make_sample_noise
value = make_sample_noise('u731', 21.0, shape=(1, 8, 8))
print(json.dumps({'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest()}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["sha256"] == expected

