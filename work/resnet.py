import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Два 3×3 свёрточных слоя + optional shortcut 1×1."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.25):
        super().__init__()

        self.same_channels = in_channels == out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.drop  = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # 1×1 shortcut, если число каналов меняется
        if self.same_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))

        out += identity
        return F.relu(out)


class ResNetLike(nn.Module):
    """Небольшой ResNet-подобный CNN для бинарной классификации."""

    def __init__(self):
        super().__init__()

        self.block1 = ResBlock(3,   32)
        self.pool1  = nn.MaxPool2d(2)

        self.block2 = ResBlock(32,  64)
        self.pool2  = nn.MaxPool2d(2)

        self.block3 = ResBlock(64, 128)
        self.pool3  = nn.MaxPool2d(2)

        self.block4 = ResBlock(128, 256)
        self.pool4  = nn.MaxPool2d(2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout     = nn.Dropout(0.5)
        self.fc1         = nn.Linear(256, 128)
        self.fc2         = nn.Linear(128, 1)   # → логит

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))

        x = self.global_pool(x).flatten(1)     # B × 256
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x).squeeze(1)        # B
        return logits


# Алиас, чтобы train.py нашёл правильный класс
CNNModel = ResNetLike
