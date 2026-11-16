"""Run a model over a whole processed match and decode one possession path per episode.

Per phase (constant line-up), per in-play episode:
  node order = sorted left-keeper-team players, sorted right-team players, outside nodes;
  features at 25 Hz -> sampled every `sample_every` frames (5 fps) -> padded to 26 nodes;
  windows of `window_steps` with 50 % overlap (the last one stretched to the episode end), each run
  through the network; macro logits, edge logits and CRF embeddings are blended across overlaps with
  linear ramps; ONE Viterbi pass over the stitched episode; the decoded edge is held for the 5 source
  frames of each step (step-hold upsampling).
Optionally the final state of an episode is forced to the outside node implied by the first event of
the following episode (goal kick / corner / throw-in), as in the official `correct_episode_lasts`.

    .venv/bin/python -m pathcrf_ours.infer --weights outputs/ported_140.pt --match J03WR9
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .config import OUTSIDE_NODES, PITCH_X, PITCH_Y, Config, DataConfig, ModelConfig
from .data import labels as L
from .data.pipeline import process_match
from .data.sportec import detect_keepers, label_phases, team_of
from .data.windows import I_VX, I_VY, I_X, I_Y, PHYS, add_outside_columns
from .model import PathCRFNet
from .paths import CACHE, OUTPUTS, SPORTEC_DIR, SYNCED_DIR


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_net(weights: Path, device: torch.device) -> PathCRFNet:
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["config"]) if "config" in ck else ModelConfig()
    net = PathCRFNet(cfg)
    merged = dict(net.state_dict())  # state tables are recomputed; only parameters come from the file
    merged.update({k: v for k, v in ck["model"].items() if k in merged})
    net.load_state_dict(merged, strict=True)
    return net.to(device).eval()


# ------------------------------------------------------------------------------------------ windows
def episode_windows(n: int, W: int, stride: int) -> List[Tuple[int, int]]:
    if n <= W:
        return [(0, n)]
    starts = list(range(0, n - W + 1, stride)) or [0]
    return [(s, n if i == len(starts) - 1 else s + W) for i, s in enumerate(starts)]


def ramp_weights(windows: List[Tuple[int, int]], i: int) -> torch.Tensor:
    s, e = windows[i]
    w = torch.ones(e - s)
    if i > 0:
        ov = min(windows[i - 1][1] - s, e - s)
        if ov > 0:
            w[:ov] = torch.linspace(0.0, 1.0, ov)
    if i < len(windows) - 1:
        ov = min(e - windows[i + 1][0], e - s)
        if ov > 0:
            w[-ov:] = torch.linspace(1.0, 0.0, ov)
    return w


@torch.no_grad()
def run_episode(net: PathCRFNet, x: np.ndarray, present: Optional[np.ndarray], dcfg: DataConfig, device: torch.device, windowed: bool = True,
                stride_div: int = 2):
    """x [T, N, F] downsampled features -> (macro [T,N,2], edge_logits [T,N*N], crf_embed [T,K,D]) blended.
    `stride_div`: window stride = window_steps // stride_div (2 = the paper's 50 % overlap)."""
    T = x.shape[0]
    W, stride = dcfg.window_steps, max(1, dcfg.window_steps // max(1, stride_div))
    windows = episode_windows(T, W, stride) if windowed else [(0, T)]
    acc = {}
    wsum = torch.zeros(T)
    for i, (s, e) in enumerate(windows):
        xt = torch.from_numpy(x[s:e]).unsqueeze(0).to(device)
        pt = torch.from_numpy(present[s:e]).unsqueeze(0).to(device) if present is not None else None
        out = net(xt, pt)
        emis, emb = net.crf_inputs(out)
        parts = {"macro": out["macro_logits"][0].cpu(), "emis": emis[0].cpu(), "emb": emb[0].cpu()}
        w = ramp_weights(windows, i)
        for k, v in parts.items():
            if k not in acc:
                acc[k] = torch.zeros((T,) + tuple(v.shape[1:]))
            acc[k][s:e] += v * w.view(-1, *([1] * (v.dim() - 1)))
        wsum[s:e] += w
    wsum = torch.where(wsum == 0, torch.ones_like(wsum), wsum)
    return {k: v / wsum.view(-1, *([1] * (v.dim() - 1))) for k, v in acc.items()}


# ------------------------------------------------------------------------- test-time augmentation (board item 28)
FLIPS_TTA = [(False, False), (True, False), (False, True), (True, True)]


def flip_inputs(x: np.ndarray, pm: Optional[np.ndarray], bxy: Optional[np.ndarray], flip_x: bool, flip_y: bool, ts: int):
    """Mirror the inference inputs exactly as training did in legacy flip mode (data/windows.apply_flip): x [T, N, 8]
    (flags + PHYS), pm [T, N] presence or None, bxy [T, 2] ball or None. Team blocks swap slots under an x flip; the
    outside-node slots keep their index (the official code's behaviour, learnt by every checkpoint)."""
    x = x.copy()
    pm = None if pm is None else pm.copy()
    bxy = None if bxy is None else bxy.copy()
    if flip_x:
        x[:, :, I_X] = PITCH_X - x[:, :, I_X]
        x[:, :, I_VX] = -x[:, :, I_VX]
        if bxy is not None:
            bxy[:, 0] = PITCH_X - bxy[:, 0]
        x = np.concatenate([x[:, ts : 2 * ts], x[:, :ts], x[:, 2 * ts :]], axis=1)
        if pm is not None:
            pm = np.concatenate([pm[:, ts : 2 * ts], pm[:, :ts], pm[:, 2 * ts :]], axis=1)
    if flip_y:
        x[:, :, I_Y] = PITCH_Y - x[:, :, I_Y]
        x[:, :, I_VY] = -x[:, :, I_VY]
        if bxy is not None:
            bxy[:, 1] = PITCH_Y - bxy[:, 1]
    return x, pm, bxy


def node_perm(ts: int, n_outside: int, flip_x: bool, flip_y: bool) -> np.ndarray:
    """Slot k of a flipped run holds original node perm[k] (an involution): team blocks and out_left/out_right under an
    x flip, out_bottom/out_top under a y flip, as the training labels were remapped."""
    p = np.arange(2 * ts + n_outside)
    if flip_x:
        p[:ts], p[ts : 2 * ts] = np.arange(ts, 2 * ts), np.arange(ts)
        p[2 * ts], p[2 * ts + 1] = 2 * ts + 1, 2 * ts
    if flip_y:
        p[2 * ts + 2], p[2 * ts + 3] = 2 * ts + 3, 2 * ts + 2
    return p


def state_perm(crf, p: np.ndarray) -> torch.Tensor:
    """Compressed-state permutation induced by the node permutation `p`: original state i <- flipped state perm[i]."""
    n = len(p)
    src, dst = crf.src.cpu().numpy(), crf.dst.cpu().numpy()
    comp = crf.orig2comp.cpu().numpy()[p[src] * n + p[dst]]
    if (comp < 0).any():
        raise ValueError("the CRF state space is not symmetric under the pitch flip")
    return torch.from_numpy(comp)


# ------------------------------------------------------------------------------------------- match
def infer_match(
    net: PathCRFNet,
    tracking: pd.DataFrame,
    dcfg: DataConfig,
    device: torch.device,
    windowed: bool = True,
    force_last: bool = False,
    evaluate: bool = True,
    team_switch: float = 0.0,
    tta: bool = False,
    stride_div: int = 2,
    variants: Optional[List[Tuple[str, float, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Returns (edges: frame_id, edge_src, edge_dst, src_prob_home; stats).
    Weekend bundle (board item 28): `net` may be a list of same-architecture nets (emission ensemble: emissions and
    macro logits averaged, transitions averaged member by member); `tta` adds the 3 mirrored pitches of every net as
    members; `stride_div` sets the inference window stride; `variants` = [(name, team_switch, trans_scale)] decodes the
    same network outputs several ways and returns {name: edges} instead of one edges table (stats from the first)."""
    t0 = time.time()
    nets = list(net) if isinstance(net, (list, tuple)) else [net]
    net = nets[0]
    for other in nets[1:]:
        if (other.cfg.presence, other.cfg.ball_features, other.cfg.mask_absent, other.n_nodes) != (net.cfg.presence, net.cfg.ball_features, net.cfg.mask_absent, net.n_nodes):
            raise ValueError("ensemble members must share presence / ball / node configuration")
    flips = FLIPS_TTA if tta else FLIPS_TTA[:1]
    multi = variants is not None
    variants = variants or [("", team_switch, 1.0)]
    ts, N = net.team_size, net.n_nodes
    every, W = dcfg.sample_every, dcfg.window_steps
    tracking = tracking.copy()
    if "phase_id" not in tracking.columns:
        tracking, _ = label_phases(tracking)
    tracking = add_outside_columns(tracking).set_index("frame_id").sort_index()
    edges = pd.DataFrame(index=tracking.index, columns=["edge_src", "edge_dst"], dtype=object)
    edges["p_home"] = np.nan          # soft sender probability mass of the home team (coarse head, blended)
    edges["p_self"] = np.nan          # softmax mass of the player self-loop states in the blended edge emissions ("someone holds the ball")
    edges["step"] = False             # True on the 5 fps step frames the model actually scored
    edges_v = {name: edges.copy() for name, _, _ in variants}   # one edge table per decode variant
    edges = edges_v[variants[0][0]]
    counts = dict(n=0, edge=0, src=0, dst=0, macro=0, team_n=0, team=0, home_pred=0, home_true=0, forbidden=0.0, trans=0, skipped=0)

    for phase in sorted(tracking["phase_id"].unique(), reverse=True):
        if phase == 0:
            continue
        pt = tracking[tracking["phase_id"] == phase]
        left_gk, right_gk = detect_keepers(pt)
        left_team, right_team = team_of(left_gk), team_of(right_gk)
        next_first_event = None
        for ep in sorted([e for e in pt["episode_id"].unique() if e > 0], reverse=True):
            et = pt[pt["episode_id"] == ep]
            valid = et.dropna(axis=1, how="all").columns
            left = sorted({c[:-2] for c in valid if re.match(rf"^{left_team}_\d+_x$", c)})
            right = sorted({c[:-2] for c in valid if re.match(rf"^{right_team}_\d+_x$", c)})
            order = L.node_order(left, right)
            cols = [f"{p}_{f}" for p in order for f in PHYS]
            if et[cols].isna().any().any():
                counts["skipped"] += 1
                continue
            flags = np.zeros((len(order), 2), dtype=np.float32)
            for i, p in enumerate(order):
                flags[i, 0] = -1.0 if p.startswith(left_team) else (1.0 if p.startswith(right_team) else 0.0)
                flags[i, 1] = 1.0 if p in (left_gk, right_gk) else 0.0
            feats = et[cols].to_numpy(dtype=np.float32).reshape(len(et), len(order), len(PHYS))
            feats = np.concatenate([np.broadcast_to(flags[None], (len(et), len(order), 2)), feats], -1)
            down = L.downsample_index(len(et), every)
            fd = feats[down]
            # pad to 26 nodes in the model's slot layout
            x = np.zeros((len(fd), N, feats.shape[-1]), dtype=np.float32)
            slots: Dict[int, int] = {}
            for k, i in enumerate([i for i, p in enumerate(order) if p in left][:ts]):
                x[:, k] = fd[:, i]
                slots[i] = k
            for k, i in enumerate([i for i, p in enumerate(order) if p in right][:ts]):
                x[:, ts + k] = fd[:, i]
                slots[i] = ts + k
            for k, i in enumerate([i for i, p in enumerate(order) if p in OUTSIDE_NODES]):
                x[:, 2 * ts + k] = fd[:, i]
                slots[i] = 2 * ts + k
            slot_names = [None] * N
            for i, k in slots.items():
                slot_names[k] = order[i]
            pm = None
            if net.cfg.presence:
                pm = np.zeros((len(fd), N), dtype=bool)
                pcols = {p: f"{p}_present" for p in order if f"{p}_present" in et.columns}
                for i, k in slots.items():
                    p = order[i]
                    pm[:, k] = et[pcols[p]].to_numpy(dtype=bool)[down] if p in pcols else True
            bxy = None
            if net.cfg.ball_features:
                bxy = et[["ball_x", "ball_y"]].to_numpy(dtype=np.float32)[down]
                if "ball_missing" in et.columns:
                    bxy[et["ball_missing"].to_numpy(dtype=bool)[down]] = np.nan

            # members = nets x pitch flips; emissions and macro logits averaged, transitions averaged in decode
            members, emis_sum, macro_sum = [], None, None
            for m_net in nets:
                for fx, fy in flips:
                    xi, pmi, bi = flip_inputs(x, pm, bxy, fx, fy, ts)
                    if pmi is not None:
                        xi = np.concatenate([xi, pmi[..., None].astype(np.float32)], -1)
                    if bi is not None:
                        from .data.ball import ball_node_features
                        xi = np.concatenate([xi, ball_node_features(bi, xi[:, :, [I_X, I_Y]], dcfg.target_fps)], -1)
                    bl = run_episode(m_net, xi, pmi if m_net.cfg.mask_absent else None, dcfg, device, windowed, stride_div)
                    if fx or fy:
                        p = node_perm(ts, N - 2 * ts, fx, fy)
                        sp = state_perm(m_net.crf, p)
                        bl = {"macro": bl["macro"][:, p], "emis": bl["emis"][:, sp], "emb": bl["emb"][:, sp]}
                    emis_sum = bl["emis"] if emis_sum is None else emis_sum + bl["emis"]
                    macro_sum = bl["macro"] if macro_sum is None else macro_sum + bl["macro"]
                    members.append((m_net.crf, bl["emb"].unsqueeze(0).to(device)))
            blended = {"emis": emis_sum / len(members), "macro": macro_sum / len(members)}
            emis = blended["emis"].unsqueeze(0).to(device)
            if len(members) == 1:
                emb, trans_fn = members[0][1], None
            else:
                emb, _cache = None, {}

                def trans_fn(t, _m=members, _c=_cache):
                    if t not in _c:
                        _c[t] = sum(c._step_transitions(e[:, t - 1], e[:, t]) for c, e in _m) / len(_m)
                    return _c[t]
            last = None
            if force_last and next_first_event is not None:
                node = _last_node_from_next_event(next_first_event)
                if node is not None:
                    last = torch.tensor([net.crf.self_loop_state(slots[order.index(node)])], device=device)
            step_frames = et.index.to_numpy()[down]
            path = full = None
            for name, lam, alpha in variants:
                path_v = net.crf.decode(emis, emb, last, team_switch_penalty=lam, trans_scale=alpha, trans_fn=trans_fn)[0].cpu().numpy()
                src_v = net.crf.src.cpu().numpy()[path_v]
                dst_v = net.crf.dst.cpu().numpy()[path_v]
                dd = pd.DataFrame({"edge_src": [slot_names[i] for i in src_v], "edge_dst": [slot_names[i] for i in dst_v]}, index=step_frames)
                full_v = dd.reindex(et.index).ffill().bfill()
                edges_v[name].loc[et.index, ["edge_src", "edge_dst"]] = full_v.values
                if path is None:
                    path, full, src_nodes, dst_nodes = path_v, full_v, src_v, dst_v
            prob = torch.softmax(blended["macro"][:, :, 0], dim=1).numpy()       # [T, N] sender distribution
            home_slots = [k for k, nm in enumerate(slot_names) if nm is not None and nm.startswith("home")]
            away_slots = [k for k, nm in enumerate(slot_names) if nm is not None and nm.startswith("away")]
            ph, pa = prob[:, home_slots].sum(1), prob[:, away_slots].sum(1)
            p_home = pd.Series(ph / np.maximum(ph + pa, 1e-9), index=step_frames)
            pe = torch.softmax(blended["emis"].float(), dim=1).numpy()            # [T, K] edge-state distribution
            self_mask = ((net.crf.src == net.crf.dst) & (net.crf.src < 2 * ts)).cpu().numpy()
            p_self = pd.Series(pe[:, self_mask].sum(1), index=step_frames)
            for ev in edges_v.values():
                ev.loc[et.index, "p_home"] = p_home.reindex(et.index).ffill().bfill().to_numpy()
                ev.loc[et.index, "p_self"] = p_self.reindex(et.index).ffill().bfill().to_numpy()
                ev.loc[step_frames, "step"] = True

            if evaluate and "player_id" in et.columns:
                index = {order[i]: k for i, k in slots.items()}
                index["goal_left"], index["goal_right"] = index["out_left"], index["out_right"]
                src_t, dst_t = L.step_labels(et["player_id"], down, index)
                ok = ~(np.isnan(src_t) | np.isnan(dst_t))
                if ok.any():
                    comp = net.crf.orig2comp.cpu().numpy()[(src_t[ok] * N + dst_t[ok]).astype(int)]
                    valid_lab = comp >= 0
                    n = int(valid_lab.sum())
                    counts["n"] += n
                    counts["edge"] += int((path[ok][valid_lab] == comp[valid_lab]).sum())
                    counts["src"] += int((src_nodes[ok][valid_lab] == src_t[ok][valid_lab]).sum())
                    counts["dst"] += int((dst_nodes[ok][valid_lab] == dst_t[ok][valid_lab]).sum())
                    mp = blended["macro"].numpy()
                    counts["macro"] += int(((mp[:, :, 0].argmax(1)[ok][valid_lab] == src_t[ok][valid_lab]) & (mp[:, :, 1].argmax(1)[ok][valid_lab] == dst_t[ok][valid_lab])).sum())
                    # team possession = team of the sender, on steps whose true sender is a player
                    ts_true = src_t[ok][valid_lab].astype(int)
                    ts_pred = src_nodes[ok][valid_lab].astype(int)
                    is_player = ts_true < 2 * ts
                    tt = np.where(ts_true[is_player] < ts, left_team, right_team)
                    tp = np.where(ts_pred[is_player] < ts, left_team, np.where(ts_pred[is_player] < 2 * ts, right_team, "out"))
                    counts["team_n"] += int(is_player.sum())
                    counts["team"] += int((tt == tp).sum())
                    counts["home_true"] += int((tt == "home").sum())
                    counts["home_pred"] += int((tp == "home").sum())
                counts["forbidden"] += net.crf.forbidden_rate(torch.from_numpy(path)[None]) * max(len(path) - 1, 0)
                counts["trans"] += max(len(path) - 1, 0)

            if force_last:
                next_first_event = _first_event(et, full)

    stats = {"inference_time_s": round(time.time() - t0, 1), "episodes_skipped_nan": counts["skipped"]}
    if evaluate and counts["n"]:
        stats.update(
            n_steps=counts["n"],
            edge_acc=round(counts["edge"] / counts["n"], 4),
            src_acc=round(counts["src"] / counts["n"], 4),
            dst_acc=round(counts["dst"] / counts["n"], 4),
            macro_acc=round(counts["macro"] / counts["n"], 4),
            forbid_rate=round(counts["forbidden"] / max(counts["trans"], 1), 6),
        )
        if counts["team_n"]:
            stats.update(
                team_acc=round(counts["team"] / counts["team_n"], 4),
                home_share_true=round(counts["home_true"] / counts["team_n"], 4),
                home_share_pred=round(counts["home_pred"] / counts["team_n"], 4),
            )
    if multi:
        return {name: ev.reset_index() for name, ev in edges_v.items()}, stats
    return edges.reset_index(), stats


# ------------------------------------------------------------------------- forced last state helpers
def _first_event(et: pd.DataFrame, edges: pd.DataFrame, min_dur: int = 10) -> Optional[dict]:
    """First event of an episode from its decoded edges (same rules as the official `_detect_first_event`)."""
    from .evaluate import is_player, node_xy, overwrite_short_self_loops

    tuples = pd.Series(list(zip(edges["edge_src"], edges["edge_dst"])), index=edges.index)
    tuples = overwrite_short_self_loops(tuples, min_dur)
    if tuples.empty:
        return None
    prev = None
    for i, (f, cur) in enumerate(tuples.items()):
        if i == 0:
            if cur[0] == cur[1]:  # a self-loop at the very first frame is not an event
                prev = cur
                continue
        elif cur == prev:
            continue
        if cur[0] == cur[1]:
            pid = cur[0] if is_player(cur[0]) else str(cur[0]).split("_")[-1]
            kind = "control" if is_player(cur[0]) else "out"
        elif is_player(cur[0]):
            pid, kind = cur[0], "pass"
        else:
            prev = cur
            continue
        x, y = node_xy(et, f, cur[0])
        return {"player_id": pid, "event_type": _setpiece_type(pid, x, y, kind), "x": x, "y": y}
    return None


def _setpiece_type(pid: str, x: float, y: float, default: str) -> str:
    """corner / throw_in / goalkick from the first event's position (official `classify_setpieces`)."""
    from .config import PITCH_X

    if not (np.isfinite(x) and np.isfinite(y)):
        return default
    is_home, is_away = pid.startswith("home_"), pid.startswith("away_")
    corners = [(0.0, 0.0), (0.0, PITCH_Y), (PITCH_X, 0.0), (PITCH_X, PITCH_Y)]
    if is_home and any((x - cx) ** 2 + (y - cy) ** 2 <= 4.0 for cx, cy in corners[2:]):
        return "corner"
    if is_away and any((x - cx) ** 2 + (y - cy) ** 2 <= 4.0 for cx, cy in corners[:2]):
        return "corner"
    if y < 1.0 or y > PITCH_Y - 1.0:
        return "throw_in"
    gk_len, gk_w = 5.5, 18.3
    near_y = PITCH_Y / 2 - gk_w / 2 <= y <= PITCH_Y / 2 + gk_w / 2
    if near_y and ((is_home and abs(x - gk_len) <= 3.0) or (is_away and abs(x - (PITCH_X - gk_len)) <= 3.0)):
        return "goalkick"
    return default


def _last_node_from_next_event(ev: dict) -> Optional[str]:
    is_home = ev["player_id"].startswith("home_")
    is_away = ev["player_id"].startswith("away_")
    if ev["event_type"] == "goalkick":
        return "out_left" if is_home else "out_right"
    if ev["event_type"] == "corner":
        return "out_left" if is_away else "out_right"
    if ev["event_type"] == "throw_in":
        return "out_top" if float(ev["y"]) >= PITCH_Y / 2 else "out_bottom"
    return None


# --------------------------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--match", default="J03WR9")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-window", action="store_true", help="one pass over each whole episode")
    ap.add_argument("--force-last", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    device = pick_device(a.device)
    net = load_net(Path(a.weights), device)
    dcfg = DataConfig()
    tracking = process_match(a.match, SPORTEC_DIR, SYNCED_DIR, CACHE, dcfg)
    edges, stats = infer_match(net, tracking, dcfg, device, windowed=not a.no_window, force_last=a.force_last)
    OUTPUTS.mkdir(exist_ok=True)
    tag = f"_{a.tag}" if a.tag else ""
    edges.to_parquet(OUTPUTS / f"edges_{a.match}{tag}.parquet")
    from .evaluate import event_scores, events_from_edges

    pred_events = events_from_edges(tracking, edges)
    true_touches = pd.read_parquet(CACHE / "touches" / f"{a.match}.parquet")
    ev_stats = event_scores(true_touches, pred_events)
    stats["events"] = ev_stats.round(4).to_dict(orient="index")
    (OUTPUTS / f"stats_{a.match}{tag}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "events"}, indent=2))
    print(ev_stats.round(4).to_string())


if __name__ == "__main__":
    main()
