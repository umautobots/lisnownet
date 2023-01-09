from typing import Tuple, Union
import numpy as np
import torch
from . import utils


def sparsity_loss(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x - x.mean(dim=-1, keepdim=True)
    x = utils.flip_cat(x)

    return dwt_loss(x), fft_loss(x)


def dwt_loss(x: torch.Tensor, levels: Union[None, int] = None) -> torch.Tensor:
    h, w = x.shape[2:]

    if levels is None:
        levels = (np.log2(h) - 2).astype(np.uint8)

    assert (h % 2**levels == 0) & (w % 2**levels == 0)

    loss = torch.tensor(0, dtype=x.dtype, device=x.device)
    for k in range(levels):
        y = utils.dwt2(x)
        c1 = y.shape[1] // 4
        x = 0.5 * y[:, :c1, :, :]

        loss += y[:, c1:, :, :].abs().mean() / 4**k

    return loss


def fft_loss(x: torch.Tensor) -> torch.Tensor:
    y = utils.fft2(x)

    return y.abs().log1p().mean()
