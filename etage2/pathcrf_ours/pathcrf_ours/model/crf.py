"""Masked, dynamic conditional random field (CRF) over possession-edge states.

A CRF scores a whole sequence of labels (here: one edge per time step) as the sum of per-step
emission scores and per-transition scores, and normalises over all possible sequences.

State space (team_size 11): the 22 x 26 = 572 edges that leave a player (self-loop "A holds",
"A -> B in flight", "A -> outside node") plus the 4 outside self-loops ("ball is out"): K = 576.
Edges leaving an outside node to somewhere else do not exist. Original flat index = src * N + dst.

Allowed transitions (anything else is impossible and gets `forbid_score`):
    A->A  ->  any edge leaving A           (A keeps it, or kicks it to B / outside)
    A->B  ->  A->B, or any edge leaving B  (still in flight, or B receives / plays first time)
    A->O  ->  A->O, or O->O                (still flying out, or now out)
    O->O  ->  O->O                         (absorbing)

Transition score psi_t(e_prev, e_curr) is computed from the two edge embeddings at t-1 and t:
    "linear": Linear(2D, 1) on [e_prev; e_curr]   (the official code; prev and curr never interact)
    "mlp":    Linear(2D, h) - ReLU - Linear(h, 1)  (what the paper describes)

Forward algorithm (log-sum-exp) gives log Z; Viterbi (max) gives the best path. Both only visit
the <= 23 allowed predecessors of each state, stored as padded "incoming" index tables.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def build_state_space(n_players: int, n_outside: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (valid_ids [K], orig2comp [N*N], src [K], dst [K]) for the compressed state space."""
    n = n_players + n_outside
    valid, src, dst = [], [], []
    for s in range(n_players):
        for d in range(n):
            valid.append(s * n + d)
            src.append(s)
            dst.append(d)
    for o in range(n_players, n):
        valid.append(o * n + o)
        src.append(o)
        dst.append(o)
    valid_t = torch.tensor(valid, dtype=torch.long)
    orig2comp = torch.full((n * n,), -1, dtype=torch.long)
    orig2comp[valid_t] = torch.arange(len(valid), dtype=torch.long)
    return valid_t, orig2comp, torch.tensor(src), torch.tensor(dst)


def allowed_matrix(src: torch.Tensor, dst: torch.Tensor, n_players: int) -> torch.Tensor:
    """allowed[i, j] = True if state i at t-1 may be followed by state j at t."""
    K = src.numel()
    src_l, dst_l = src.tolist(), dst.tolist()
    allowed = torch.zeros(K, K, dtype=torch.bool)
    for i in range(K):
        s, d = src_l[i], dst_l[i]
        allowed[i, i] = True  # stay on the same edge is always allowed
        if s >= n_players:  # O->O absorbing
            continue
        if s == d:  # A->A: any edge leaving A
            for j in range(K):
                if src_l[j] == s:
                    allowed[i, j] = True
        elif d < n_players:  # A->B: any edge leaving B
            for j in range(K):
                if src_l[j] == d:
                    allowed[i, j] = True
        else:  # A->O: O->O
            for j in range(K):
                if src_l[j] == d and dst_l[j] == d:
                    allowed[i, j] = True
    return allowed


def incoming_tables(allowed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Padded lists of allowed predecessors: inc_idx [K, M] (-1 padded), inc_mask [K, M]."""
    K = allowed.shape[0]
    preds = [allowed[:, j].nonzero(as_tuple=False).flatten().tolist() for j in range(K)]
    M = max(len(p) for p in preds)
    inc_idx = torch.full((K, M), -1, dtype=torch.long)
    inc_mask = torch.zeros((K, M), dtype=torch.bool)
    for j, p in enumerate(preds):
        inc_idx[j, : len(p)] = torch.tensor(p, dtype=torch.long)
        inc_mask[j, : len(p)] = True
    return inc_idx, inc_mask


class MaskedDynamicCRF(nn.Module):
    def __init__(
        self,
        n_players: int = 22,
        n_outside: int = 4,
        embed_dim: int = 16,
        transition: str = "linear",
        trans_hidden: int = 32,
        forbid_score: float = -1e4,
    ):
        super().__init__()
        self.n_players, self.n_outside, self.embed_dim = n_players, n_outside, embed_dim
        self.forbid_score = float(forbid_score)
        valid, orig2comp, src, dst = build_state_space(n_players, n_outside)
        self.K = valid.numel()
        self.register_buffer("valid_ids", valid)
        self.register_buffer("orig2comp", orig2comp)
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        allowed = allowed_matrix(src, dst, n_players)
        self.register_buffer("allowed", allowed)
        inc_idx, inc_mask = incoming_tables(allowed)
        self.register_buffer("inc_idx", inc_idx)
        self.register_buffer("inc_mask", inc_mask)
        # spellF1 option 1 / 5: (state, predecessor slot) pairs whose sender changes team (home <-> away, both players)
        half = n_players // 2
        team = torch.where(src < half, 0, torch.where(src < n_players, 1, 2))
        pred_team = team[inc_idx.clamp(min=0)]
        switch_pair = (team[:, None] != pred_team) & (team[:, None] < 2) & (pred_team < 2) & inc_mask
        self.register_buffer("switch_pair", switch_pair)
        self.transition = transition
        if transition == "linear":
            self.trans_linear = nn.Linear(2 * embed_dim, 1)
        elif transition == "mlp":
            self.trans_mlp = nn.Sequential(nn.Linear(2 * embed_dim, trans_hidden), nn.ReLU(), nn.Linear(trans_hidden, 1))
        else:
            raise ValueError(f"unknown transition type {transition!r}")

    # ----------------------------------------------------------------------------------------- scores
    def trans_score(self, e_prev: torch.Tensor, e_curr: torch.Tensor) -> torch.Tensor:
        """[..., D], [..., D] -> [...]."""
        z = torch.cat([e_prev, e_curr], dim=-1)
        if self.transition == "linear":
            return self.trans_linear(z).squeeze(-1)
        return self.trans_mlp(z).squeeze(-1)

    def _step_transitions(self, emb_prev: torch.Tensor, emb_curr: torch.Tensor) -> torch.Tensor:
        """emb_prev, emb_curr [B, K, D] -> transition scores for every (state, predecessor slot) [B, K, M]."""
        M = self.inc_idx.shape[1]
        prev_g = emb_prev[:, self.inc_idx.clamp(min=0)]  # [B, K, M, D]
        curr_e = emb_curr.unsqueeze(2).expand(-1, -1, M, -1)
        return self.trans_score(prev_g, curr_e)

    def select(self, edge_tensor: torch.Tensor) -> torch.Tensor:
        """Keep only the K valid states along dim 2 of a [B, T, N*N, ...] tensor."""
        return edge_tensor.index_select(2, self.valid_ids)

    # ----------------------------------------------------------------------------------- forward algorithm
    def log_partition(self, emissions: torch.Tensor, embeds: torch.Tensor, step_cost: Optional[torch.Tensor] = None,
                      switch_cost: Optional[torch.Tensor] = None) -> torch.Tensor:
        """emissions [B, T, K], embeds [B, T, K, D] -> log Z [B].
        Softmax-margin (spellF1 option 5): `step_cost` [B, T, K] is added to the emissions and `switch_cost` [B, T]
        to every team-switching transition (switch_pair) entering step t, so log Z runs over cost-augmented paths."""
        B, T, K = emissions.shape
        forbid = emissions.new_full((B, K, self.inc_idx.shape[1]), self.forbid_score)
        alpha = emissions[:, 0] + (step_cost[:, 0] if step_cost is not None else 0.0)
        sp = self.switch_pair.to(emissions.dtype) if switch_cost is not None else None
        for t in range(1, T):
            prev_alpha = alpha[:, self.inc_idx.clamp(min=0)]  # [B, K, M]
            trans = self._step_transitions(embeds[:, t - 1], embeds[:, t])
            if sp is not None:
                trans = trans + switch_cost[:, t, None, None] * sp
            score = torch.where(self.inc_mask[None], prev_alpha + trans, forbid)
            alpha = torch.logsumexp(score, dim=2) + emissions[:, t]
            if step_cost is not None:
                alpha = alpha + step_cost[:, t]
        return torch.logsumexp(alpha, dim=1)

    def gold_score(self, emissions: torch.Tensor, embeds: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """Score of the given path tags [B, T] (compressed ids). Forbidden steps score forbid_score."""
        B, T, K = emissions.shape
        ar = torch.arange(B, device=emissions.device)
        score = emissions[ar, 0, tags[:, 0]]
        for t in range(1, T):
            prev, curr = tags[:, t - 1], tags[:, t]
            trans = self.trans_score(embeds[ar, t - 1, prev], embeds[ar, t, curr])
            ok = self.allowed[prev, curr]
            trans = torch.where(ok, trans, trans.new_full(trans.shape, self.forbid_score))
            score = score + emissions[ar, t, curr] + trans
        return score

    def nll(self, emissions: torch.Tensor, embeds: torch.Tensor, tags: torch.Tensor, step_cost: Optional[torch.Tensor] = None,
            switch_cost: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Negative log-likelihood per sequence [B] (not divided by T). With costs: softmax-margin loss (the gold path
        must carry zero cost: costs are defined against the gold labels)."""
        return self.log_partition(emissions, embeds, step_cost, switch_cost) - self.gold_score(emissions, embeds, tags)

    # ------------------------------------------------------------------------------------------ Viterbi
    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, embeds: Optional[torch.Tensor], last_state: Optional[torch.Tensor] = None,
               team_switch_penalty: float = 0.0, trans_scale: float = 1.0, trans_fn=None):
        """Best path [B, T]. `last_state` [B] (compressed id) forces the final state when given.
        `team_switch_penalty` (option 1): subtracted from every transition whose sender changes team (0 = the paper).
        `trans_scale` (weekend bundle, board item 28): multiplies the learned transition scores (1 = the paper).
        `trans_fn(t) -> [B, K, M]` replaces the transitions computed from `embeds` (ensembles: the mean of the members'
        transitions); `embeds` may then be None."""
        B, T, K = emissions.shape
        M = self.inc_idx.shape[1]
        forbid = emissions.new_full((B, K, M), self.forbid_score)
        delta = emissions[:, 0]
        back = []
        pen = team_switch_penalty * self.switch_pair.to(emissions.dtype) if team_switch_penalty else None
        for t in range(1, T):
            prev_delta = delta[:, self.inc_idx.clamp(min=0)]
            trans = trans_fn(t) if trans_fn is not None else self._step_transitions(embeds[:, t - 1], embeds[:, t])
            if trans_scale != 1.0:
                trans = trans * trans_scale
            if pen is not None:
                trans = trans - pen
            score = torch.where(self.inc_mask[None], prev_delta + trans, forbid)
            best, slot = score.max(dim=2)
            delta = best + emissions[:, t]
            # keep the scores near 0: on a long episode (thousands of steps) delta drifts below forbid_score and the
            # forbidden slots would win the max, sending the back-pointers to the -1 padding. The argmax path is unchanged.
            delta = delta - delta.max(dim=1, keepdim=True).values
            back.append(slot)
        path = emissions.new_zeros((B, T), dtype=torch.long)
        path[:, -1] = delta.argmax(dim=1) if last_state is None else last_state
        for t in range(T - 1, 0, -1):
            slot = back[t - 1].gather(1, path[:, t : t + 1]).squeeze(1)
            path[:, t - 1] = self.inc_idx[path[:, t], slot]
        return path

    def self_loop_state(self, node: int) -> int:
        """Compressed id of the self-loop of `node` (player or outside node)."""
        hit = ((self.src == node) & (self.dst == node)).nonzero(as_tuple=False)
        return int(hit[0, 0])

    def forbidden_rate(self, path: torch.Tensor) -> float:
        """Share of consecutive steps in `path` [B, T] that are not allowed."""
        if path.shape[1] < 2:
            return 0.0
        ok = self.allowed[path[:, :-1], path[:, 1:]]
        return float((~ok).float().mean())
