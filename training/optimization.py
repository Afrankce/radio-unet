from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar

import torch
from torch import Tensor

from training.masked_flow_loss import MaskedLossError, masked_velocity_mse


Batch = TypeVar("Batch")


class OptimizationContractError(RuntimeError):
    """Gradient accumulation or optimizer-step state violates the benchmark."""


class ScalerLike(Protocol):
    def scale(self, loss: Tensor) -> Tensor: ...

    def step(self, optimizer: torch.optim.Optimizer) -> Any: ...

    def update(self) -> None: ...

    def get_scale(self) -> float: ...


class SchedulerLike(Protocol):
    def step(self) -> None: ...


class EMALike(Protocol):
    def update(self, model: torch.nn.Module) -> None: ...


def lr_multiplier(step_index: int, total_steps: int, warmup_steps: int) -> float:
    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise OptimizationContractError("step_index must be a non-negative integer")
    if (
        isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or total_steps <= 0
        or warmup_steps <= 0
        or warmup_steps >= total_steps
    ):
        raise OptimizationContractError(
            "scheduler requires total_steps > warmup_steps > 0"
        )
    optimizer_step_number = min(step_index + 1, total_steps)
    if optimizer_step_number <= warmup_steps:
        return optimizer_step_number / warmup_steps
    progress = (optimizer_step_number - warmup_steps) / (
        total_steps - warmup_steps
    )
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class OptimizerStepScheduler:
    """Set LR for the upcoming successful optimizer step, not for an epoch."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
    ) -> None:
        # Validate the schedule before capturing or changing optimizer state.
        lr_multiplier(0, total_steps, warmup_steps)
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        if not self.base_lrs or any(
            not math.isfinite(value) or value < 0.0 for value in self.base_lrs
        ):
            raise OptimizationContractError("optimizer base learning rates are invalid")
        self.completed_steps = 0
        self._apply_upcoming_lr()

    def _apply_upcoming_lr(self) -> None:
        step_index = min(self.completed_steps, self.total_steps - 1)
        multiplier = lr_multiplier(
            step_index,
            self.total_steps,
            self.warmup_steps,
        )
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * multiplier

    def step(self) -> None:
        if self.completed_steps >= self.total_steps:
            raise OptimizationContractError("scheduler advanced beyond total_steps")
        self.completed_steps += 1
        self._apply_upcoming_lr()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "base_lrs": list(self.base_lrs),
            "completed_steps": self.completed_steps,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        expected = {
            "schema_version",
            "total_steps",
            "warmup_steps",
            "base_lrs",
            "completed_steps",
        }
        if set(payload) != expected:
            raise OptimizationContractError("scheduler state keys mismatch")
        if payload["schema_version"] != 1:
            raise OptimizationContractError("scheduler schema version mismatch")
        if payload["total_steps"] != self.total_steps:
            raise OptimizationContractError("scheduler total_steps mismatch")
        if payload["warmup_steps"] != self.warmup_steps:
            raise OptimizationContractError("scheduler warmup_steps mismatch")
        try:
            base_lrs = tuple(float(value) for value in payload["base_lrs"])
            completed = int(payload["completed_steps"])
        except (TypeError, ValueError) as error:
            raise OptimizationContractError("invalid scheduler state values") from error
        if base_lrs != self.base_lrs:
            raise OptimizationContractError("scheduler base_lrs mismatch")
        if not 0 <= completed <= self.total_steps:
            raise OptimizationContractError("scheduler completed_steps is invalid")
        self.completed_steps = completed
        self._apply_upcoming_lr()


def build_optimizer_step_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
) -> OptimizerStepScheduler:
    return OptimizerStepScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )


@dataclass(frozen=True)
class AccumulationWindowResult:
    optimizer_ran: bool
    micro_batches: int
    samples: int
    valid_pixels: int
    squared_error_sum: float

    @property
    def mean_squared_error(self) -> float:
        if self.valid_pixels <= 0:
            raise OptimizationContractError("window contains no valid pixels")
        return self.squared_error_sum / self.valid_pixels


def _batch_valid_mask(batch: Any) -> Tensor:
    if not isinstance(batch, Mapping) or "valid_mask" not in batch:
        raise OptimizationContractError(
            "each micro-batch must expose a valid_mask tensor"
        )
    mask = batch["valid_mask"]
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
        raise OptimizationContractError("micro-batch valid_mask must be boolean")
    return mask


def _scaled_optimizer_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: ScalerLike,
    scheduler: SchedulerLike,
    ema: EMALike,
) -> bool:
    before = float(scaler.get_scale())
    if not math.isfinite(before) or before <= 0.0:
        raise OptimizationContractError(f"invalid GradScaler scale before step: {before}")
    scaler.step(optimizer)
    scaler.update()
    after = float(scaler.get_scale())
    if not math.isfinite(after) or after <= 0.0:
        raise OptimizationContractError(f"invalid GradScaler scale after step: {after}")
    optimizer_ran = after >= before
    if optimizer_ran:
        ema.update(model)
        scheduler.step()
    return optimizer_ran


def run_accumulation_window(
    micro_batches: Sequence[Batch],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: ScalerLike,
    scheduler: SchedulerLike,
    ema: EMALike,
    predict: Callable[[torch.nn.Module, Batch], tuple[Tensor, Tensor, Tensor]],
) -> AccumulationWindowResult:
    """Backpropagate one exact valid-pixel-weighted accumulation window."""

    batches = tuple(micro_batches)
    if not batches:
        raise OptimizationContractError("accumulation window must not be empty")
    total_valid = sum(int(_batch_valid_mask(batch).sum().item()) for batch in batches)
    if total_valid <= 0:
        raise OptimizationContractError("accumulation window has zero valid pixels")

    optimizer.zero_grad(set_to_none=True)
    squared_error_sum = 0.0
    samples = 0
    try:
        for batch in batches:
            predicted, target, valid_mask = predict(model, batch)
            micro_valid = int(valid_mask.sum().item())
            if micro_valid <= 0:
                raise MaskedLossError("valid_mask contains zero valid pixels")
            loss = masked_velocity_mse(predicted, target, valid_mask)
            weight = torch.tensor(
                micro_valid / total_valid,
                device=loss.device,
                dtype=torch.float32,
            )
            scaler.scale(loss * weight).backward()
            squared_error_sum += float(loss.detach().float().item()) * micro_valid
            samples += int(predicted.shape[0])
        optimizer_ran = _scaled_optimizer_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            ema=ema,
        )
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    optimizer.zero_grad(set_to_none=True)
    return AccumulationWindowResult(
        optimizer_ran=optimizer_ran,
        micro_batches=len(batches),
        samples=samples,
        valid_pixels=total_valid,
        squared_error_sum=squared_error_sum,
    )


def accumulation_windows(
    values: Sequence[Batch],
    accumulation_steps: int,
) -> tuple[tuple[Batch, ...], ...]:
    if (
        isinstance(accumulation_steps, bool)
        or not isinstance(accumulation_steps, int)
        or accumulation_steps <= 0
    ):
        raise OptimizationContractError("accumulation_steps must be positive")
    return tuple(
        tuple(values[start : start + accumulation_steps])
        for start in range(0, len(values), accumulation_steps)
    )

