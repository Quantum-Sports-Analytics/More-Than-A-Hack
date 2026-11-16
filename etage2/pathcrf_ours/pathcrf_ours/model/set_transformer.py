"""Set Transformer blocks (Lee et al., 2019) as used by PathCRF.

A "set" block treats its inputs as an unordered collection: shuffling the rows shuffles the outputs
the same way (permutation equivariance). The blocks are:

- MultiheadAttention: standard scaled dot-product attention with an optional key mask.
- MAB (multihead attention block): post-norm residual attention + a row-wise feed-forward net.
- SAB (self attention block): MAB(x, x).
- ISAB (induced self attention block): the rows first write into `m` learned "inducing points",
  then read back from them. Cost grows with n*m instead of n*n.
- SetEncoder: Linear -> ReLU -> ISAB -> ISAB -> Linear, applied row by row (equivariant output).

Faithfulness notes (needed so the authors' weights load and give identical outputs):
- the feed-forward net has three Linear-ReLU pairs and ENDS with a ReLU;
- LayerNorm is applied after the residual sum (post-norm);
- attention logits are divided by sqrt(d / heads);
- SetEncoder's Linear layers use Kaiming-normal weights and zero biases; LayerNorm starts at (1, 0).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0
        self.heads = heads
        self.scale = math.sqrt(d // heads)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, key_mask: Optional[torch.Tensor] = None):
        """queries [b, n, d], keys [b, m, d], key_mask [b, m] bool (True = usable key). Returns [b, n, d]."""
        b, n, d = queries.shape
        m = keys.shape[1]
        h, p = self.heads, d // self.heads
        q = self.q(queries).view(b, n, h, p).transpose(1, 2)  # [b, h, n, p]
        k = self.k(keys).view(b, m, h, p).transpose(1, 2)  # [b, h, m, p]
        v = self.v(keys).view(b, m, h, p).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-1, -2)) / self.scale  # [b, h, n, m]
        if key_mask is not None:
            att = att.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        att = att.softmax(dim=-1)
        if key_mask is not None:
            att = torch.nan_to_num(att)  # a query with no usable key gets zeros instead of NaN
        out = torch.matmul(att, v).transpose(1, 2).reshape(b, n, d)
        return self.o(out)


class FeedForward(nn.Sequential):
    """Row-wise feed-forward: (Linear -> ReLU) x 3, ending in a ReLU (official quirk)."""

    def __init__(self, d: int):
        super().__init__(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())


class MAB(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.attn = MultiheadAttention(d, heads)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = FeedForward(d)

    def forward(self, x: torch.Tensor, y: torch.Tensor, key_mask: Optional[torch.Tensor] = None):
        h = self.norm1(x + self.attn(x, y, key_mask))
        return self.norm2(h + self.ff(h))


class ISAB(nn.Module):
    def __init__(self, d: int, inducing: int, heads: int):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, inducing, d))
        self.mab1 = MAB(d, heads)
        self.mab2 = MAB(d, heads)

    def forward(self, x: torch.Tensor, key_mask: Optional[torch.Tensor] = None):
        p = self.inducing.expand(x.shape[0], -1, -1)
        h = self.mab1(p, x, key_mask)  # [b, m, d]: the inducing points summarise the (masked) set
        return self.mab2(x, h)  # every row reads from the summary; absent rows get an output too


class SetEncoder(nn.Module):
    """Equivariant Set Transformer encoder: per-row output of size out_dim."""

    def __init__(self, in_dim: int, out_dim: int, d: int = 128, inducing: int = 16, heads: int = 4):
        super().__init__()
        self.embed = nn.Linear(in_dim, d)
        self.isab1 = ISAB(d, inducing, heads)
        self.isab2 = ISAB(d, inducing, heads)
        self.out = nn.Linear(d, out_dim)
        self.apply(_init_set_encoder)

    def forward(self, x: torch.Tensor, key_mask: Optional[torch.Tensor] = None):
        h = F.relu(self.embed(x))
        h = self.isab1(h, key_mask)
        h = self.isab2(h, key_mask)
        return self.out(h)


def _init_set_encoder(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
