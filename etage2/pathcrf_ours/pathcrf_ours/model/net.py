"""PathCRFNet: coarse pass -> fine pass -> edge scores -> CRF.

Shapes with the default config (B windows, T = 50 steps, N = 26 nodes, F = 8 features):

coarse pass
    x [B,T,N,8] (+2 zero channels -> 10) -> SocialModule -> PPE 16 | FPE 16
    cat[x, ppe, fpe] = 42 -> TemporalModule -> h1 [B,T,N,128]
    coarse_head Linear(128,4) + GLU -> macro logits [B,T,N,2] = (sender logit, receiver logit)
fine pass
    cat[x 8, h1 128, macro 2] = 138 -> SocialModule -> 64 | 64 -> Linear(128,64)+ReLU (+2 zeros -> 66)
    -> TemporalModule -> h2 [B,T,N,128]
edges
    for every ordered pair (i, j): cat[h2_i, h2_j] 256 -> Linear(256,128)+ReLU = edge_h
    edge_head Linear(128,2)+GLU -> emission logit per edge [B,T,N*N], flat index i*N+j
    crf_proj Linear(128,16)+ReLU -> LayerNorm(16) -> CRF edge embedding [B,T,N*N,16]
CRF
    keeps the 576 valid states (see crf.py)
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .crf import MaskedDynamicCRF
from .social import SocialModule
from .temporal import TemporalModule


class PathCRFNet(nn.Module):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        ts, z = cfg.team_size, cfg.extra_zero_channels
        self.team_size, self.n_nodes, self.zero_channels = ts, cfg.n_nodes, z
        fin = cfg.in_dim

        c_in = fin + z
        self.coarse_social = SocialModule(c_in, cfg.coarse_dim, ts, cfg.n_outside, False, cfg.set_dim, cfg.inducing_points, cfg.heads)
        self.coarse_temporal = TemporalModule(c_in + 2 * cfg.coarse_dim, cfg.seq_dim, cfg.heads, cfg.dropout)
        self.coarse_head = nn.Linear(cfg.seq_dim, 4)

        f_in = fin + cfg.seq_dim + 2
        self.fine_social = SocialModule(f_in, cfg.fine_dim, ts, cfg.n_outside, True, cfg.set_dim, cfg.inducing_points, cfg.heads)
        self.fine_node_proj = nn.Linear(2 * cfg.fine_dim, cfg.fine_dim)
        self.fine_temporal = TemporalModule(cfg.fine_dim + z, cfg.seq_dim, cfg.heads, cfg.dropout)

        self.edge_proj = nn.Linear(2 * cfg.seq_dim, cfg.seq_dim)
        self.edge_head = nn.Linear(cfg.seq_dim, 2)
        self.crf_proj = nn.Linear(cfg.seq_dim, cfg.crf_edge_dim)
        self.crf_norm = nn.LayerNorm(cfg.crf_edge_dim)
        self.crf = MaskedDynamicCRF(cfg.n_players, cfg.n_outside, cfg.crf_edge_dim, cfg.transition, cfg.trans_hidden, cfg.forbid_score)

    # ------------------------------------------------------------------------------------------------
    def _zeros(self, x: torch.Tensor) -> torch.Tensor:
        if self.zero_channels == 0:
            return x
        return torch.cat([x, x.new_zeros(*x.shape[:-1], self.zero_channels)], dim=-1)

    def forward(self, x: torch.Tensor, present: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """x [B, T, N, F] node features; present [B, T, N] bool (Phase B) or None."""
        B, T, N, _ = x.shape
        assert N == self.n_nodes
        xs = x.reshape(B * T, N, -1)
        pm = present.reshape(B * T, N).bool() if present is not None else None

        # coarse pass
        xc = self._zeros(xs)
        soc = self.coarse_social(xc, pm)
        h1 = self.coarse_temporal(torch.cat([xc, soc], dim=-1).view(B, T, N, -1))  # [B,T,N,seq]
        macro = F.glu(self.coarse_head(h1), dim=-1)  # [B,T,N,2]

        # fine pass
        fin = torch.cat([x, h1, macro], dim=-1).reshape(B * T, N, -1)
        fs = F.relu(self.fine_node_proj(self.fine_social(fin, pm)))
        h2 = self.fine_temporal(self._zeros(fs).view(B, T, N, -1))  # [B,T,N,seq]

        # edges
        src = h2.unsqueeze(3).expand(B, T, N, N, h2.shape[-1])
        dst = h2.unsqueeze(2).expand(B, T, N, N, h2.shape[-1])
        edge_h = F.relu(self.edge_proj(torch.cat([src, dst], dim=-1)))  # [B,T,N,N,seq]
        edge_logits = F.glu(self.edge_head(edge_h), dim=-1).reshape(B, T, N * N)
        crf_embed = self.crf_norm(F.relu(self.crf_proj(edge_h))).reshape(B, T, N * N, -1)
        return {"macro_logits": macro, "edge_logits": edge_logits, "crf_embed": crf_embed, "node_h": h2}

    # ------------------------------------------------------------------------------------- CRF helpers
    def crf_inputs(self, out: Dict[str, torch.Tensor]):
        """(emissions [B,T,K], embeds [B,T,K,D]) restricted to the valid CRF states."""
        return self.crf.select(out["edge_logits"]), self.crf.select(out["crf_embed"])

    def decode(self, out: Dict[str, torch.Tensor], last_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        emis, emb = self.crf_inputs(out)
        return self.crf.decode(emis, emb, last_state)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
