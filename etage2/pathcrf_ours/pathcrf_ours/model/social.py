"""Social module: who is near whom at one moment.

Two branches run side by side on the 26 nodes of one time step and their outputs are concatenated:

- PPE (partially permutation equivariant): one SetEncoder per team (separate weights), and a plain
  Linear layer for each of the 4 outside nodes. Players can be reordered inside a team, but a home
  player is never interchangeable with an away player.
- FPE (fully permutation equivariant): one SetEncoder over all 26 nodes together.

Input [b, N, in_dim] with node order: team on the left 0..ts-1, team on the right ts..2ts-1,
outside nodes last. Output [b, N, 2 * out_dim].
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .set_transformer import SetEncoder


class SocialModule(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        team_size: int = 11,
        n_outside: int = 4,
        outside_relu: bool = False,
        set_dim: int = 128,
        inducing: int = 16,
        heads: int = 4,
    ):
        super().__init__()
        self.team_size = team_size
        self.n_outside = n_outside
        self.outside_relu = outside_relu
        self.team_a = SetEncoder(in_dim, out_dim, set_dim, inducing, heads)
        self.team_b = SetEncoder(in_dim, out_dim, set_dim, inducing, heads)
        self.outside = nn.Linear(in_dim, out_dim)
        self.fpe = SetEncoder(in_dim, out_dim, set_dim, inducing, heads)

    def forward(self, x: torch.Tensor, present: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x [b, N, in_dim]; present [b, N] bool (Phase B key mask) or None."""
        ts = self.team_size
        mask_a = mask_b = mask_all = None
        if present is not None:
            mask_a, mask_b, mask_all = present[:, :ts], present[:, ts : 2 * ts], present
        za = self.team_a(x[:, :ts], mask_a)
        zb = self.team_b(x[:, ts : 2 * ts], mask_b)
        zo = self.outside(x[:, 2 * ts :])
        if self.outside_relu:
            zo = F.relu(zo)
        ppe = torch.cat([za, zb, zo], dim=1)
        fpe = self.fpe(x, mask_all)
        return torch.cat([ppe, fpe], dim=-1)
