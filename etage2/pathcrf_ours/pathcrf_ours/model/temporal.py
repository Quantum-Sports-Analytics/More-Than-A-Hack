"""Temporal module: how each node's situation evolves over the window.

Linear + ReLU -> add a fixed sinusoidal positional encoding (tells the model which time step a
vector comes from) -> dropout (the only dropout in the whole model) -> two self-attention blocks
over the T time steps, run independently for every node. No causal mask: every step sees the
whole window in both directions.

Input [B, T, N, in_dim] -> output [B, T, N, d].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .set_transformer import MAB


def sinusoidal_encoding(max_len: int, d: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d)
    position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)[:, : d // 2]
    return pe


class TemporalModule(nn.Module):
    def __init__(self, in_dim: int, d: int = 128, heads: int = 4, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d = d
        self.proj = nn.Linear(in_dim, d)
        self.drop = nn.Dropout(dropout)
        self.sab1 = MAB(d, heads)
        self.sab2 = MAB(d, heads)
        self.register_buffer("pe", sinusoidal_encoding(max_len, d), persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, T, N, _ = z.shape
        x = F.relu(self.proj(z))
        x = x + self.pe[:T][None, :, None, :]
        x = self.drop(x)
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, self.d)
        x = self.sab1(x, x)
        x = self.sab2(x, x)
        return x.view(B, N, T, self.d).permute(0, 2, 1, 3)
