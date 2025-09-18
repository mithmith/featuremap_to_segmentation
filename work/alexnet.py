import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    """
    Упрощённая AlexNet-подобная сеть для бинарной классификации.
    Последний слой выдаёт один логит; используйте BCEWithLogitsLoss.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.2),

            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.2),

            nn.Conv2d(64, 96, kernel_size=3), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(3), nn.Dropout(0.2),
        )

        self._fc1: nn.Linear | None = None        # создаётся при первом forward
        self.fc2 = nn.Linear(128, 1)              # логит

    # ―――― helpers ――――――――――――――――――――――――――――――――――――――――――
    def _init_fc1(self, x: torch.Tensor) -> None:
        flat_dim = x.shape[1] * x.shape[2] * x.shape[3]
        self._fc1 = nn.Linear(flat_dim, 128).to(x.device)

    # ―――― forward ――――――――――――――――――――――――――――――――――――――――――
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self._fc1 is None:
            self._init_fc1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self._fc1(x))
        logits = self.fc2(x).squeeze(1)  # (B,)
        return logits


# Алиас, который ищут train.py / analyze_model.py
CNNModel = AlexNet
