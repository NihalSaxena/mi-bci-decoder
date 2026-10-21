"""EEGNet (Lawhern et al., 2018), sized for BCI IV 2a. See docs/DESIGN.md, "EEGNet architecture"."""
import torch
from torch import nn


def _pad_time(kernel_length: int) -> nn.ZeroPad2d:
    """Zero-pad the time axis so a convolution keeps the same number of samples."""
    return nn.ZeroPad2d((kernel_length // 2 - 1, kernel_length // 2, 0, 0))


class EEGNet(nn.Module):
    """(batch, n_channels, n_times) EEG -> (batch, n_classes) class scores (logits)."""

    def __init__(
        self,
        n_channels: int = 22,
        n_times: int = 500,
        n_classes: int = 4,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.features = nn.Sequential(
            # 1. temporal filters: F1 learned frequency bands, same filter on every electrode
            #    shape: (1, 22, 500) -> (8, 22, 500)
            _pad_time(kernel_length),
            nn.Conv2d(1, F1, (1, kernel_length), bias=False),
            nn.BatchNorm2d(F1),

            # 2. spatial filters: D blends of all electrodes per band (depthwise)
            #    shape: (8, 22, 500) -> (16, 1, 500)
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),

            # 3. average over time (÷4)
            #    shape: (16, 1, 500) -> (16, 1, 125)
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),

            # 4. per-map temporal filter, then mix the maps
            #    shape: (16, 1, 125) -> (16, 1, 15) -> (240,)
            _pad_time(16),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        # 5. classifier: one score per class
        #    shape: (240,) -> (4,)
        self.classifier = nn.Linear(F2 * (n_times // 4 // 8), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, 22, 500) -> (batch, 1, 22, 500)
        return self.classifier(self.features(x))