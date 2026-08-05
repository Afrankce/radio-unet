from __future__ import annotations

import copy
import math

import pytest
import torch


def _optimization_module():
    from training import optimization

    return optimization


def _micro_batches() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "x": torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
            "target": torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]]),
            "valid_mask": torch.tensor([[[[True, True], [False, False]]]]),
        },
        {
            "x": torch.tensor([[[[2.0, 1.0], [1.0, 3.0]]]]),
            "target": torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]]),
            "valid_mask": torch.tensor([[[[True, True], [True, False]]]]),
        },
    ]


class CountingEMA:
    def __init__(self) -> None:
        self.calls = 0

    def update(self, _model: torch.nn.Module) -> None:
        self.calls += 1


class CountingScheduler:
    def __init__(self) -> None:
        self.calls = 0

    def step(self) -> None:
        self.calls += 1


class OverflowScaler:
    def __init__(self) -> None:
        self.scale_value = 1024.0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def get_scale(self) -> float:
        return self.scale_value

    def step(self, _optimizer: torch.optim.Optimizer) -> None:
        return None

    def update(self) -> None:
        self.scale_value /= 2


def _predict(model: torch.nn.Module, batch: dict[str, torch.Tensor]):
    return model(batch["x"]), batch["target"], batch["valid_mask"]


def test_accumulated_update_matches_true_pixel_weighted_effective_batch() -> None:
    module = _optimization_module()
    accumulated = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
    accumulated.weight.data.fill_(0.25)
    direct = copy.deepcopy(accumulated)
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.1)
    direct_optimizer = torch.optim.SGD(direct.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batches = _micro_batches()

    result = module.run_accumulation_window(
        batches,
        model=accumulated,
        optimizer=accumulated_optimizer,
        scaler=scaler,
        scheduler=CountingScheduler(),
        ema=CountingEMA(),
        predict=_predict,
    )

    direct_optimizer.zero_grad(set_to_none=True)
    predictions = torch.cat([direct(batch["x"]) for batch in batches])
    targets = torch.cat([batch["target"] for batch in batches])
    masks = torch.cat([batch["valid_mask"] for batch in batches])
    difference = (predictions - targets).masked_select(masks)
    difference.square().mean().backward()
    direct_optimizer.step()

    assert result.optimizer_ran is True
    assert result.valid_pixels == 5
    assert result.micro_batches == 2
    assert result.samples == 2
    assert torch.allclose(accumulated.weight, direct.weight, atol=1e-7, rtol=0)


def test_scheduler_and_ema_advance_only_after_successful_optimizer_step() -> None:
    module = _optimization_module()
    model = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = CountingScheduler()
    ema = CountingEMA()

    skipped = module.run_accumulation_window(
        _micro_batches(),
        model=model,
        optimizer=optimizer,
        scaler=OverflowScaler(),
        scheduler=scheduler,
        ema=ema,
        predict=_predict,
    )

    assert skipped.optimizer_ran is False
    assert scheduler.calls == 0
    assert ema.calls == 0

    successful = module.run_accumulation_window(
        _micro_batches(),
        model=model,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        scheduler=scheduler,
        ema=ema,
        predict=_predict,
    )

    assert successful.optimizer_ran is True
    assert scheduler.calls == 1
    assert ema.calls == 1


def test_final_short_window_uses_actual_valid_pixel_denominator() -> None:
    module = _optimization_module()
    model = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
    model.weight.data.fill_(0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    result = module.run_accumulation_window(
        _micro_batches()[:1],
        model=model,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        scheduler=CountingScheduler(),
        ema=CountingEMA(),
        predict=_predict,
    )

    assert result.micro_batches == 1
    assert result.valid_pixels == 2
    assert result.mean_squared_error == pytest.approx(0.5)


def test_lr_schedule_controls_the_lr_used_by_upcoming_optimizer_step() -> None:
    module = _optimization_module()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1e-3)
    scheduler = module.build_optimizer_step_scheduler(
        optimizer,
        total_steps=56_000,
        warmup_steps=5_600,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3 / 5_600)
    for _ in range(5_599):
        optimizer.step()
        scheduler.step()
    assert scheduler.completed_steps == 5_599
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    for _ in range(5_599, 55_999):
        optimizer.step()
        scheduler.step()
    assert scheduler.completed_steps == 55_999
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-15)


def test_lr_multiplier_matches_locked_boundaries() -> None:
    module = _optimization_module()

    assert module.lr_multiplier(0, 56_000, 5_600) == pytest.approx(1 / 5_600)
    assert module.lr_multiplier(5_599, 56_000, 5_600) == 1.0
    assert module.lr_multiplier(55_999, 56_000, 5_600) == pytest.approx(
        0.0, abs=1e-15
    )
    assert math.isfinite(module.lr_multiplier(20_000, 56_000, 5_600))

