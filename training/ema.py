from __future__ import annotations

import copy

import torch


class ModelEMA:
    """Maintain an exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.ema_model = copy.deepcopy(model)
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)
        self.decay = decay

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            model_state = model.state_dict()
            ema_state = self.ema_model.state_dict()
            for key in ema_state:
                if key in model_state:
                    ema_state[key].mul_(self.decay).add_(
                        model_state[key], alpha=1 - self.decay
                    )
            self.ema_model.load_state_dict(ema_state)
