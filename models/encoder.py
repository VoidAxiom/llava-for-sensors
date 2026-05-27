"""Minimal PatchTST time-series encoder.

Input:  (B, 2048) float32 -- mono sensor window
Output: (B, 32, 512) float32 -- 32 patches x d=512
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ToyTSEncoder(nn.Module):
    """Minimal PatchTST encoder: 2048-sample window to 32 patches of d=512."""

    def __init__(
        self,
        *,
        n_patches: int = 32,
        patch_size: int = 64,
        d_model: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a batch of mono 2048-sample windows into 32 sensor patches."""
        x = x.reshape(x.shape[0], self.n_patches, self.patch_size)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.encoder(x)
        return x
