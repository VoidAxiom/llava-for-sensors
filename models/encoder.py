"""Toy 1D-CNN time-series encoder.

Input:  (B, 2048) float32 -- mono sensor window
Output: (B, 32, 512) float32 -- 32 patches x d=512
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ToyTSEncoder(nn.Module):
    """Toy 1D-CNN encoder: 2048-sample window to 32 patches of d=512."""

    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=8, stride=8, padding=0),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=4, padding=0),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=2, stride=2, padding=0),
            nn.GELU(),
        )
        self.proj = nn.Linear(256, 512)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a batch of mono 2048-sample windows into 32 sensor patches."""
        x = x.unsqueeze(1)
        x = self.convs(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x
