from __future__ import annotations

import pytest
import torch

from training.complex_grad_scaler import ComplexGradScaler


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_complex_grad_scaler_steps_real_and_complex_parameters() -> None:
    device = torch.device("cuda:0")
    real = torch.nn.Parameter(torch.tensor([1.5], device=device))
    spectral = torch.nn.Parameter(
        torch.tensor([1.0 + 2.0j], dtype=torch.cfloat, device=device)
    )
    optimizer = torch.optim.AdamW([real, spectral], lr=1e-2, weight_decay=0.0)
    scaler = ComplexGradScaler("cuda", enabled=True)
    before_real = real.detach().clone()
    before_spectral = spectral.detach().clone()

    loss = real.square().sum() + spectral.abs().square().sum()
    scaler.scale(loss).backward()

    assert spectral.grad is not None and spectral.grad.is_complex()
    scaler.step(optimizer)
    scaler.update()

    assert torch.isfinite(real).all()
    assert torch.isfinite(torch.view_as_real(spectral)).all()
    assert not torch.equal(real, before_real)
    assert not torch.equal(spectral, before_spectral)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_complex_grad_scaler_detects_nonfinite_complex_gradient_and_skips_step() -> None:
    device = torch.device("cuda:0")
    spectral = torch.nn.Parameter(
        torch.tensor([1.0 + 2.0j], dtype=torch.cfloat, device=device)
    )
    optimizer = torch.optim.AdamW([spectral], lr=1e-2, weight_decay=0.0)
    scaler = ComplexGradScaler("cuda", enabled=True)
    before_parameter = spectral.detach().clone()
    before_scale = scaler.get_scale()

    loss = spectral.real.sum() * torch.tensor(float("inf"), device=device)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert torch.equal(spectral, before_parameter)
    assert scaler.get_scale() < before_scale


def test_disabled_complex_grad_scaler_has_standard_checkpoint_state() -> None:
    scaler = ComplexGradScaler("cuda", enabled=False)

    assert scaler.state_dict() == {}
    scaler.load_state_dict({})
