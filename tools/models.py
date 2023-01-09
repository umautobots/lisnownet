import torch
import torch.nn as nn
from . import utils


class Fill2d(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.requires_grad = False

        self.maxpool = CircMaxPool2d(kernel_size)
        self.avgpool = CircAvgPool2d(2 * kernel_size - 1)

    def forward(self, x, idx_valid):
        n, c, h, w = x.shape

        y = self.maxpool(x)
        y[idx_valid] = x[idx_valid]

        idx_y = utils.get_valid_indices(y)

        mu = (y * idx_y).sum(dim=3) / idx_y.sum(dim=3)
        mu = mu.reshape(n, c, h, 1).tile(1, 1, 1, w)
        sigma2 = ((y - mu).pow(2) * idx_y).sum(dim=3) / idx_y.sum(dim=3)
        sigma2 = sigma2.reshape(n, c, h, 1).tile(1, 1, 1, w)
        y[~idx_y] = (mu + sigma2.sqrt())[~idx_y]

        dy = utils.laplacian(y).relu()
        y = self.avgpool(y + dy)
        y[idx_valid] = x[idx_valid]

        return y


class CircAvgPool2d(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.requires_grad = False

        self.padding = kernel_size // 2
        self.pool = nn.AvgPool2d(kernel_size, stride=1)

    def forward(self, x):
        x = utils.circular_pad(x, [self.padding] * 4)

        return self.pool(x)


class CircMaxPool2d(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.requires_grad = False

        self.padding = kernel_size // 2
        self.pool = nn.MaxPool2d(kernel_size, stride=1)

    def forward(self, x):
        x = utils.circular_pad(x, [self.padding] * 4)

        return self.pool(x)


class DWT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def forward(self, x):
        return utils.dwt2(x)


class IDWT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def forward(self, y):
        return utils.idwt2(y)


class ConvBN2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = utils.circular_pad(x, [self.padding] * 4)

        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d()

        self.main_block = nn.Sequential(
            ConvBN2d(in_channels, out_channels, 3),
            self.activation,
            self.dropout,
            ConvBN2d(out_channels, out_channels, 3),
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = ConvBN2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.activation(self.main_block(x) + self.shortcut(x))


class LiSnowNet(nn.Module):
    def __init__(self, c0=8):
        super().__init__()

        self.fill = Fill2d()

        self.enc0 = nn.Sequential(
            ResidualBlock(2 + 1, c0),
            ResidualBlock(c0, c0)
        )   # output shape: (N, C, H, W)

        self.enc1 = nn.Sequential(
            DWT2(),
            ResidualBlock(4 * c0, 2 * c0),
            ResidualBlock(2 * c0, 2 * c0)
        )   # output shape: (N, C * 2, H // 2, W // 2)

        self.enc2 = nn.Sequential(
            DWT2(),
            ResidualBlock(8 * c0, 4 * c0),
            ResidualBlock(4 * c0, 4 * c0)
        )   # output shape: (N, C * 4, H // 4, W // 4)

        self.enc3 = nn.Sequential(
            DWT2(),
            ResidualBlock(16 * c0, 8 * c0),
            ResidualBlock(8 * c0, 8 * c0)
        )   # output shape: (N, C * 8, H // 8, W // 8)

        self.enc4 = nn.Sequential(
            DWT2(),
            ResidualBlock(32 * c0, 16 * c0),
            ResidualBlock(16 * c0, 16 * c0)
        )   # output shape: (N, C * 16, H // 16, W // 16)

        self.dec4 = nn.Sequential(
            ConvBN2d(c0 * 16, c0 * 32, 1),
            IDWT2()
        )   # output shape: (N, C * 8, H // 8, W // 8)

        self.dec3 = nn.Sequential(
            ResidualBlock(8 * c0, 8 * c0),
            ResidualBlock(8 * c0, 16 * c0),
            IDWT2()
        )   # output shape: (N, C * 4, H // 4, W // 4)

        self.dec2 = nn.Sequential(
            ResidualBlock(4 * c0, 4 * c0),
            ResidualBlock(4 * c0, 8 * c0),
            IDWT2()
        )   # output shape: (N, C * 2, H // 2, W // 2)

        self.dec1 = nn.Sequential(
            ResidualBlock(2 * c0, 2 * c0),
            ResidualBlock(2 * c0, 4 * c0),
            IDWT2()
        )   # output shape: (N, C, H, W)

        self.dec0 = nn.Sequential(
            ResidualBlock(c0, c0),
            ResidualBlock(c0, c0),
            nn.Conv2d(c0, 2, 1)
        )   # output shape: (N, 2, H, W)

    def forward(self, x):
        # x.shape: (N, 2, H, W)

        idx_valid = utils.get_valid_indices(x)
        x = self.fill(x, idx_valid)

        x0 = torch.cat([
            x,
            idx_valid[:, 0, :, :].unsqueeze(1)
        ], dim=1)

        f0 = self.enc0(x0)
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        y = self.dec4(f4)
        y = self.dec3(y + f3)
        y = self.dec2(y + f2)
        y = self.dec1(y + f1)
        y = self.dec0(y + f0) + x

        return idx_valid, y
