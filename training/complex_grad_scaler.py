from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch
from torch.amp.grad_scaler import _MultiDeviceReplicator


class ComplexGradScaler(torch.amp.GradScaler):
    """GradScaler whose fused unscale path accepts dense complex gradients.

    PyTorch 2.5 groups gradients by dtype and sends each group to a fused
    non-finite-check/unscale kernel.  That CUDA kernel does not accept
    ``complex64``.  A complex tensor's real view aliases the same storage, so
    checking and unscaling the float view preserves the exact complex gradient
    while retaining the standard GradScaler state machine and overflow policy.
    """

    def _unscale_grads_(
        self,
        optimizer: torch.optim.Optimizer,
        inv_scale: torch.Tensor,
        found_inf: torch.Tensor,
        allow_fp16: bool,
    ) -> Dict[torch.device, torch.Tensor]:
        per_device_inv_scale = _MultiDeviceReplicator(inv_scale)
        per_device_found_inf = _MultiDeviceReplicator(found_inf)
        per_device_and_dtype_grads: Dict[
            torch.device,
            Dict[torch.dtype, List[torch.Tensor]],
        ] = defaultdict(lambda: defaultdict(list))

        with torch.no_grad():
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    gradient = parameter.grad
                    if not allow_fp16 and gradient.dtype == torch.float16:
                        raise ValueError("Attempting to unscale FP16 gradients.")
                    if gradient.is_sparse:
                        if gradient.dtype == torch.float16:
                            parameter.grad = gradient.coalesce()
                            gradient = parameter.grad
                        to_unscale = gradient._values()
                    else:
                        to_unscale = gradient
                    if to_unscale.is_complex():
                        to_unscale = torch.view_as_real(to_unscale)
                    per_device_and_dtype_grads[to_unscale.device][
                        to_unscale.dtype
                    ].append(to_unscale)

            for device, per_dtype_grads in per_device_and_dtype_grads.items():
                for gradients in per_dtype_grads.values():
                    torch._amp_foreach_non_finite_check_and_unscale_(
                        gradients,
                        per_device_found_inf.get(device),
                        per_device_inv_scale.get(device),
                    )

        return per_device_found_inf._per_device_tensors


__all__ = ["ComplexGradScaler"]
