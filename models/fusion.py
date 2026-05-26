"""Fusion adapter: sensor embeddings to VLM token space.

Input:  (B, 32, 512) -- encoder output
Output: (B, 16, 1536) -- 16 sensor tokens at Qwen2-VL-2B hidden dim=1536
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

T_SENSOR_TOKENS: int = 16
D_ENC: int = 512
D_VLM: int = 1536


class FusionAdapter(nn.Module):
    """Cross-attention adapter from encoder patches into fixed VLM sensor tokens."""

    def __init__(
        self,
        d_enc: int = D_ENC,
        d_vlm: int = D_VLM,
        t_sensor: int = T_SENSOR_TOKENS,
        n_heads: int = 8,
    ) -> None:
        super().__init__()
        self.d_vlm = d_vlm
        self.t_sensor = t_sensor

        self.kv_proj = nn.Linear(d_enc, d_vlm)
        self.query_tokens = nn.Parameter(torch.randn(t_sensor, d_vlm) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_vlm,
            num_heads=n_heads,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_vlm, d_vlm),
            nn.GELU(),
            nn.Linear(d_vlm, d_vlm),
        )
        self.norm = nn.LayerNorm(d_vlm)

    def forward(self, enc_out: Tensor) -> Tensor:
        """Adapt encoder patches into fixed-size VLM sensor tokens."""
        batch_size = enc_out.shape[0]
        kv = self.kv_proj(enc_out)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        attn_out, _ = self.cross_attn(queries, kv, kv, need_weights=False)
        out = self.norm(attn_out + self.mlp(attn_out))
        return out
