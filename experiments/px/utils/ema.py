import copy

import numpy as np
import torch


@torch.no_grad()
def update_ema(model: torch.nn.Module, ema: torch.nn.Module, beta: float):
    for p_net, p_ema in zip(model.parameters(), ema.parameters()):
        p_ema.lerp_(p_net, 1 - beta)


def std_to_exp(std):
    std = np.float64(std)
    tmp = std.flatten() ** -2
    exp = [np.roots([1, 7, 16 - t, 12 - t]).real.max() for t in tmp]
    exp = np.float64(exp).reshape(std.shape)  # type: ignore
    return exp


def power_function_beta(std: float, step: int):
    return (1 - 1 / step) ** (std_to_exp(std) + 1)


class KarrasEMA(torch.nn.Module):
    """Power-function EMA as used in EDM2:
    - https://arxiv.org/abs/2312.02696
    - github.com/NVlabs/edm2

    This effectively schedules the decay beta throughout training:
    e.g. with std = 0.1 (good for longer runs):
    step 10k: beta = 0.999
    step 100k: beta = 0.9999
    step 1M: beta = 0.99999

    with std = 0.01:
    step 10k: beta = 0.99
    step 100k: beta = 0.999
    step 1M: beta = 0.9999
    """

    def __init__(self, net: torch.nn.Module, std: float = 0.01):
        super().__init__()
        self.module = copy.deepcopy(net)
        self.module.eval()
        self.std = std

    def _get_model_device(self, model: torch.nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int):
        model_device = self._get_model_device(model)
        ema_device = self._get_model_device(self.module)
        if model_device != ema_device:
            self.module.to(model_device)

        beta = power_function_beta(std=self.std, step=step)
        for p_net, p_ema in zip(model.parameters(), self.module.parameters()):
            p_ema.lerp_(p_net, 1 - beta)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
