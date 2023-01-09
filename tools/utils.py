from typing import Tuple, List
import numpy as np
import torch
from torch import fft
import torch.nn.functional as F


haar_basis = 0.5 * torch.tensor([
    [+1, +1, +1, +1],
    [+1, +1, -1, -1],
    [+1, -1, +1, -1],
    [+1, -1, -1, +1]
], dtype=torch.float32, requires_grad=False)
haar_basis = haar_basis.unsqueeze(0)


def dwt2(x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    h2, w2 = h // 2, w // 2

    x = x.reshape(n, c, h2, 2, w2, 2).permute(0, 3, 5, 1, 2, 4)
    y = torch.matmul(
        haar_basis.to(x.device),
        x.reshape(n, 4, c * h2 * w2)
    )

    return y.reshape(n, 4 * c, h2, w2)


def idwt2(y: torch.Tensor) -> torch.Tensor:
    n, c, h, w = y.shape
    c4 = c // 4

    x = torch.matmul(
        haar_basis.to(y.device),
        y.reshape(n, 4, c4 * h * w)
    )
    x = x.reshape(n, 2, 2, c4, h, w).permute(0, 3, 4, 1, 5, 2)

    return x.reshape(n, c4, h * 2, w * 2)


def fft2(x: torch.Tensor) -> torch.Tensor:
    y = fft.rfft2(x, norm='ortho')
    y = torch.cat([y.real, y.imag.flip(-1)], dim=-1)

    return fft.fftshift(y, dim=[-2, -1])


def image2points(point_img: torch.Tensor, fill_value: float = -np.inf) -> torch.Tensor:
    '''
    Convert range images to point clouds.

    Parameters:
        point_img:  (batch_size, num_features, height, width)
                    The features are usually in the order of (x, y, z, i, ...)

    Returns:
        points:     (batch_size, max_num_points, num_features)
    '''

    batch_size = point_img.size(0)
    dtype, device = point_img.dtype, point_img.device

    idx_valid = (point_img >= 0).any(1)     # (batch_size, H, W)
    max_num_points = idx_valid.sum(dim=[-2, -1]).max()
    points = torch.full(
        [batch_size, max_num_points, point_img.size(1)],
        fill_value,
        dtype=dtype,
        device=device
    )

    for b in range(batch_size):
        idx = idx_valid[b, :, :]    # (H, W)
        points[b, :idx.sum(), :] = point_img[b, :, idx].transpose(-1, -2)

    return points


def dwt2_image(x: torch.Tensor, levels: int = 1, scale : bool = True) -> torch.Tensor:
    y = dwt2(x)
    n, c, h, w = y.shape
    y = y.reshape(n, 4, c // 4, h, w)

    if levels == 1:
        y0 = y[:, 0, :, :, :]
    else:
        y0 = dwt2_image(y[:, 0, :, :, :], levels - 1, scale)

    # scale the top-right by 0.5 to maintain at the same magnitude
    if scale:
        y0 *= 0.5

    y = torch.cat([
        torch.cat([y0, y[:, 1, :, :, :]], dim=-2),
        torch.cat([y[:, 2, :, :, :], y[:, 3, :, :, :]], dim=-2),
    ], dim=-1)

    return y


def circular_pad(x: torch.Tensor, padding: List[int] = [1, 1, 1, 1]) -> torch.Tensor:
    # padding: [left, right, top, bottom]
    x = F.pad(x, [padding[0], padding[1], 0, 0], mode='circular')
    x = F.pad(x, [0, 0, padding[2], padding[3]], mode='reflect')
    return x


def laplacian(x: torch.Tensor) -> torch.Tensor:
    w = 0.125 * torch.tensor([
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1],
        ], dtype=x.dtype, device=x.device, requires_grad=False)
    w = w.reshape(1, 1, 3, 3).tile(x.size(1), 1, 1, 1)

    x = circular_pad(x, [1] * 4)

    return F.conv2d(x, w, groups=x.size(1))


def checkerboard_split(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n, c, h, w = x.shape
    i0, i1 = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    mask = (i0 + i1).remainder(2).type(torch.bool).to(x.device)
    mask = mask.reshape(1, 1, h, w).tile(n, c, 1, 1)

    return x[mask].reshape(n, c, h, w // 2), x[~mask].reshape(n, c, h, w // 2)


def flip_cat(x: torch.Tensor) -> torch.Tensor:
    x0, x1 = checkerboard_split(x)
    return torch.cat([x0, x1.flip(2)], dim=2)


def get_valid_indices(range_img: torch.Tensor) -> torch.Tensor:
    return (range_img >= 0).any(1, keepdim=True).expand_as(range_img)
