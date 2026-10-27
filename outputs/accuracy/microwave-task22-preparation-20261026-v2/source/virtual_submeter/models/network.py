"""A small offline sequence-to-point CNN; not a claimed reproduction of the paper."""

import torch
from torch import nn


class SmallSeq2Point(nn.Module):
    def __init__(self, width: int, outputs: int, channels: int = 8, hidden: int = 32):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(1, channels, 5, padding=2), nn.ReLU(),
                                      nn.Conv1d(channels, channels, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels * width, hidden), nn.ReLU(), nn.Linear(hidden, outputs))
        nn.init.constant_(self.head[-1].bias, -3.0)

    def forward(self, values):
        return torch.nn.functional.softplus(self.head(self.features(values[:, None, :])))
