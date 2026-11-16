"""Training loop. Phase A = the official trial-140 recipe; Phase B = the same loop on damaged data.

Phase A (replication, fold 6 = test J03WR9, val J03WQQ, train the other 5 Sportec matches):
    python -m pathcrf_ours.train --fold 6 --tag replica
    python -m pathcrf_ours.train --fold all                   # 7-fold leave-one-match-out
    python -m pathcrf_ours.train --fold 6 --smoke             # 3 batches, checks the code

Phase B (Metrica-like damage, presence flag, SkillCorner + Sportec):
    python -m pathcrf_ours.preprocess_b --variants 3          # once: caches cache/processed/sc_* and cache/damaged/*
    python -m pathcrf_ours.train --split phaseb --presence --damaged 3 --init-from outputs/runs/replica_fold6/model/epoch_010.pt \
                                 --tag phaseb --window-stride 2 --lr 5e-4 --epochs 10
    --split phaseb: train = 6 Sportec (all but J03WR9) + 17 SkillCorner (train + test of splits.json),
                    val = the 3 SkillCorner validation matches (damaged) and J03WR9 (clean and damaged).
    --train-ids / --val-ids: explicit comma-separated lists (Sportec ids or sc_<match id>).
    --damaged K: index K damaged draws per training match (cache/damaged/<key>_v1..vK.parquet) besides the
                 clean table (dropped with --no-clean); every epoch picks one variant per match at random.
    --init-from: start from a Phase A checkpoint; the new "present" input column starts at zero weight.
    --mask-absent: also mask absent nodes out of the social attention (off: the flag is a feature only).

spellF1 options (2026-11-12): --macro-weight / --micro-weight / --crf-weight reweight the three
terms (option 2); --team-loss W adds W x BCE(P(sender team = home) from the edge logits, true team) and
--switch-loss W adds W x BCE(P(team changes between t-1 and t), true switch), true switches up-weighted by
--switch-pos-weight (option 3). Validation also logs team_acc, spell_f1 (spells inside the window matched by
IoU >= 0.5) and switch_f1 (switches within 2 steps); best_spell.pt keeps the best validation spell_f1 (option 0).

Loss per batch = macro (CE sender + CE receiver over the 26 nodes, from the coarse head)
               + micro (CE over the 676 edge logits + CE of the sender marginal + CE of the receiver marginal)
               + CRF negative log-likelihood / T.
Adam, lr 1e-3, weight decay 1e-4, batch 32, 50 epochs; the lr halves (floor 1e-5) after 3 epochs
without a new best validation loss or accuracy. Checkpoints: best_loss.pt (lowest sum of the three
validation losses), best_acc.pt (highest validation Viterbi edge accuracy on the main validation set; the
one that is evaluated), every `save_every` epochs, and last.pt (model + optimizer + schedule state) after
every epoch so that `--resume` can continue an interrupted run. `--amp` runs the backbone in bfloat16
autocast; the CRF (forward algorithm, Viterbi) and all losses stay in float32.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config, DataConfig, ModelConfig, TrainConfig
from .data.pipeline import load_matches
from .data.skillcorner import SPLITS, is_skillcorner, process_skillcorner
from .data.sportec import match_ids
from .data.windows import WindowDataset, collate
from .infer import pick_device
from .model import PathCRFNet
from .paths import CACHE, OUTPUTS, SPORTEC_DIR, SYNCED_DIR


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lomocv_split(ids: List[str], fold: int) -> Tuple[List[str], List[str], List[str]]:
    n = len(ids)
    test, val = ids[fold], ids[(fold - 1) % n]
    train = [m for i, m in enumerate(ids) if i not in (fold, (fold - 1) % n)]
    return train, [val], [test]


# ------------------------------------------------------------------------------------------ data
def load_table(key: str, cfg: DataConfig, variant: int = 0) -> pd.DataFrame:
    """Clean (variant 0) or damaged (variant k >= 1, from preprocess_b) processed table of one match."""
    if variant > 0:
        f = CACHE / "damaged" / f"{key}_v{variant}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"{f}: run `python -m pathcrf_ours.preprocess_b --variants {variant}` first")
        return pd.read_parquet(f)
    if is_skillcorner(key):
        return process_skillcorner(key[3:], cfg, CACHE)
    return load_matches([key], SPORTEC_DIR, SYNCED_DIR, CACHE, cfg)[key]


def load_variants(keys: Sequence[str], cfg: DataConfig, n_damaged: int, keep_clean: bool = True) -> Dict[str, List[pd.DataFrame]]:
    out = {}
    for k in keys:
        tables = [load_table(k, cfg, 0)] if keep_clean or n_damaged == 0 else []
        tables += [load_table(k, cfg, v) for v in range(1, n_damaged + 1)]
        out[k] = tables
    return out


def phaseb_split() -> Tuple[List[str], Dict[str, List[str]]]:
    sportec = match_ids(SPORTEC_DIR)
    if len(sportec) != 7:                      # 2026-11-15: a package copy without external/pathcrf/data/sportec silently trained on 17 matches
        raise FileNotFoundError(f"phaseb split expects the 7 Sportec matches under {SPORTEC_DIR}, found {sportec}; "
                                f"link external/ next to the package or pass --train-ids explicitly")
    train = [m for m in sportec if m != "J03WR9"] + [f"sc_{m}" for m in SPLITS["train"] + SPLITS["test"]]
    val = {"val": [f"sc_{m}" for m in SPLITS["validation"]], "idsse": ["J03WR9"]}
    return train, val


# ------------------------------------------------------------------------------------------ model
def init_from_checkpoint(net: PathCRFNet, path: Path) -> None:
    """Load a checkpoint trained without the presence input into a net that may have it (zero weight)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    src = ck["model"]
    dst = net.state_dict()
    old_in = ModelConfig(**ck["config"]).in_dim if "config" in ck else net.cfg.node_in_dim
    new_in = net.cfg.in_dim
    n_new = new_in - old_in
    merged = {}
    for k, v in dst.items():
        if k not in src:
            merged[k] = v
            continue
        w = src[k]
        if w.shape == v.shape:
            merged[k] = w
        elif w.dim() == 2 and v.dim() == 2 and v.shape[0] == w.shape[0] and v.shape[1] == w.shape[1] + n_new:
            # a Linear whose input starts with the node features: zero columns for the new features
            nw = torch.zeros_like(v)
            nw[:, :old_in] = w[:, :old_in]
            nw[:, new_in:] = w[:, old_in:]
            merged[k] = nw
        else:
            raise ValueError(f"cannot map {k}: {tuple(w.shape)} -> {tuple(v.shape)}")
    net.load_state_dict(merged, strict=True)


def team_sequence_counts(team_pred: torch.Tensor, team_true: torch.Tensor, tol: int = 2, iou: float = 0.5) -> Dict[str, float]:
    """Spell-level counts per batch from the Viterbi path's sender team ([B, T] bool). Returns per-window means of:
    team accuracy, true / predicted / matched switches (matched = within `tol` steps, one to one), true / predicted /
    matched spells (one-to-one by overlap-over-union >= iou inside the window). F1 values are derived in fit()."""
    tp_, tt_, pp_, sw_tp, sw_t, sw_p = 0, 0, 0, 0, 0, 0
    pred_np, true_np = team_pred.cpu().numpy(), team_true.cpu().numpy()
    for b in range(pred_np.shape[0]):
        pt, tt = pred_np[b], true_np[b]
        st = np.flatnonzero(tt[1:] != tt[:-1]) + 1
        sp = np.flatnonzero(pt[1:] != pt[:-1]) + 1
        sw_t += len(st); sw_p += len(sp)
        used = set()
        for a in st:
            j = next((k for k, c in enumerate(sp) if k not in used and abs(int(c) - int(a)) <= tol), None)
            if j is not None:
                used.add(j); sw_tp += 1
        def runs(v):
            edges = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1, len(v)]
            return [(int(edges[i]), int(edges[i + 1]), bool(v[edges[i]])) for i in range(len(edges) - 1)]
        rt, rp = runs(tt), runs(pt)
        tt_ += len(rt); pp_ += len(rp)
        pairs = sorted(((min(e1, e2) - max(s1, s2)) / (max(e1, e2) - min(s1, s2)), i, j)
                       for i, (s1, e1, c1) in enumerate(rt) for j, (s2, e2, c2) in enumerate(rp)
                       if c1 == c2 and min(e1, e2) > max(s1, s2))
        ut, up = set(), set()
        for o, i, j in reversed(pairs):
            if o >= iou and i not in ut and j not in up:
                ut.add(i); up.add(j); tp_ += 1
    n = pred_np.shape[0]
    return {"team_acc": float((pred_np == true_np).mean()), "cnt_spell_tp": tp_ / n, "cnt_spell_true": tt_ / n,
            "cnt_spell_pred": pp_ / n, "cnt_sw_tp": sw_tp / n, "cnt_sw_true": sw_t / n, "cnt_sw_pred": sw_p / n}


def derive_f1(d: Dict[str, float]) -> Dict[str, float]:
    """Add spell_f1 and switch_f1 (ratios of the pooled counts) to an epoch's aggregated logs."""
    for name, key in (("spell_f1", "spell"), ("switch_f1", "sw")):
        tp, t, p = d.get(f"cnt_{key}_tp"), d.get(f"cnt_{key}_true"), d.get(f"cnt_{key}_pred")
        if tp is None:
            continue
        prec = tp / p if p else 0.0
        rec = tp / t if t else 0.0
        d[name] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return d


def batch_losses(net: PathCRFNet, x: torch.Tensor, y: torch.Tensor, tcfg: TrainConfig, present=None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """x [B,T,N,F] (F already includes the presence feature if the model uses it), y [B,T,2] (src, dst node ids),
    present [B,T,N] bool or None (attention mask)."""
    B, T, N, _ = x.shape
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=tcfg.amp):
        out = net(x, present)
    if tcfg.amp:
        out = {k: v.float() for k, v in out.items()}
    src, dst = y[:, :, 0].reshape(-1), y[:, :, 1].reshape(-1)
    # spellF1 option 4: steps within +-boundary_steps of a true team switch weigh boundary_weight in the cross-entropies
    ts0 = net.team_size
    tt = (y[:, :, 0] < ts0).float()                                       # [B, T] 1 = home sender
    if tcfg.boundary_weight != 1.0:
        sw = (tt[:, 1:] != tt[:, :-1]).float()                            # switch between t-1 and t
        ind = torch.zeros_like(tt)
        ind[:, 1:] = sw
        ind[:, :-1] = torch.maximum(ind[:, :-1], sw)
        k = 2 * tcfg.boundary_steps + 1
        near = F.max_pool1d(ind[:, None], k, stride=1, padding=tcfg.boundary_steps)[:, 0]
        w = torch.where(near > 0, tt.new_tensor(float(tcfg.boundary_weight)), tt.new_tensor(1.0)).reshape(-1)
        w = w / w.mean()

        def ce(logits, target):
            return (F.cross_entropy(logits, target, reduction="none") * w).mean()
    else:
        ce = F.cross_entropy
    ml = out["macro_logits"]
    macro = ce(ml[..., 0].reshape(-1, N), src) + ce(ml[..., 1].reshape(-1, N), dst)
    edge_lab = (y[:, :, 0] * N + y[:, :, 1]).reshape(-1)
    el = out["edge_logits"].reshape(-1, N * N)
    micro = el.new_tensor(0.0)
    if tcfg.emit_loss:
        micro = micro + ce(el, edge_lab)
    el3 = el.view(-1, N, N)
    if tcfg.src_loss:
        micro = micro + ce(torch.logsumexp(el3, dim=2), src)
    if tcfg.dst_loss:
        micro = micro + ce(torch.logsumexp(el3, dim=1), dst)
    emis, emb = net.crf_inputs(out)
    tags = net.crf.orig2comp[edge_lab].view(B, T)
    if (tags < 0).any():
        raise ValueError("a target edge leaves an outside node; sanitise the labels")
    step_cost = switch_cost = None
    if tcfg.margin_c1 > 0 or tcfg.margin_c2 > 0:                          # spellF1 option 5: softmax-margin costs
        src_k = net.crf.src
        st_team = torch.where(src_k < ts0, 0, torch.where(src_k < 2 * ts0, 1, 2))          # [K]
        tt_long = (1 - tt).long()                                            # true team id: 0 home, 1 away
        if tcfg.margin_c1 > 0:
            wrong = (st_team[None, None, :] != tt_long[:, :, None]) & (st_team[None, None, :] < 2)
            step_cost = tcfg.margin_c1 * wrong.to(emis.dtype)
        if tcfg.margin_c2 > 0:
            no_switch = torch.ones_like(tt)
            no_switch[:, 1:] = (tt[:, 1:] == tt[:, :-1]).float()
            switch_cost = tcfg.margin_c2 * no_switch.to(emis.dtype)
    crf = net.crf.nll(emis, emb, tags, step_cost, switch_cost).mean() / T if tcfg.crf_weight > 0 else el.new_tensor(0.0)
    loss = tcfg.macro_weight * macro + tcfg.micro_weight * micro + tcfg.crf_weight * crf
    logs = {"macro_loss": float(macro.detach()), "micro_loss": float(micro.detach()), "crf_loss": float(crf.detach())}
    # ---- spellF1 option 3: team-level and switch losses from the edge logits (nodes 0..ts-1 home, ts..2ts-1 away)
    ts = net.team_size
    team_true = (src < ts).float().view(B, T)                       # 1 = home sender (targets never leave an outside node)
    logit_home = torch.logsumexp(el3[:, :ts, :].reshape(B * T, -1), 1) - torch.logsumexp(el3[:, ts:2 * ts, :].reshape(B * T, -1), 1)
    logit_home = logit_home.view(B, T)
    team = F.binary_cross_entropy_with_logits(logit_home, team_true)
    p_home = torch.sigmoid(logit_home)
    p_switch = p_home[:, 1:] * (1 - p_home[:, :-1]) + (1 - p_home[:, 1:]) * p_home[:, :-1]
    sw_true = (team_true[:, 1:] != team_true[:, :-1]).float()
    w = torch.where(sw_true > 0, p_switch.new_tensor(tcfg.switch_pos_weight), p_switch.new_tensor(1.0))
    switch = F.binary_cross_entropy(p_switch.clamp(1e-6, 1 - 1e-6), sw_true, weight=w)
    if tcfg.team_weight > 0:
        loss = loss + tcfg.team_weight * team
    if tcfg.switch_weight > 0:
        loss = loss + tcfg.switch_weight * switch
    logs["team_loss"], logs["switch_loss"] = float(team.detach()), float(switch.detach())
    with torch.no_grad():
        if tcfg.crf_weight > 0:
            pred = net.crf.decode(emis, emb)
            logs["edge_acc"] = float((pred == tags).float().mean())
            logs["src_acc"] = float((net.crf.src[pred].reshape(-1) == src).float().mean())
            logs["dst_acc"] = float((net.crf.dst[pred].reshape(-1) == dst).float().mean())
            logs.update(team_sequence_counts((net.crf.src[pred] < ts).view(B, T), team_true.bool()))
        else:
            logs["edge_acc"] = float((el.argmax(1) == edge_lab).float().mean())
        logs["macro_acc"] = float(((ml[..., 0].reshape(-1, N).argmax(1) == src) & (ml[..., 1].reshape(-1, N).argmax(1) == dst)).float().mean())
    return loss, logs


def run_epoch(net, loader, device, tcfg, optimizer=None, max_batches=None) -> Dict[str, float]:
    train = optimizer is not None
    net.train(train)
    agg: Dict[str, List[float]] = {}
    for i, (x, y, present, ballf) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y, present = x.to(device), y.to(device), present.to(device)
        mask = None
        if net.cfg.presence:
            x = torch.cat([x, present.unsqueeze(-1)], -1)
            if net.cfg.mask_absent:
                mask = present.bool()
        if net.cfg.ball_features:
            x = torch.cat([x, ballf.to(device)], -1)
        with torch.set_grad_enabled(train):
            loss, logs = batch_losses(net, x, y, tcfg, mask)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), tcfg.grad_clip)
            optimizer.step()
        logs["loss"] = float(loss.detach())
        for k, v in logs.items():
            agg.setdefault(k, []).append(v)
    return derive_f1({k: float(np.mean(v)) for k, v in agg.items()})


# ------------------------------------------------------------------------------------------ fit
def fit(
    cfg: Config,
    train_ids: List[str],
    val_sets: Dict[str, List[str]],
    out_dir: Path,
    device: torch.device,
    smoke: bool = False,
    resume: bool = False,
    n_damaged: int = 0,
    keep_clean: bool = True,
    init_from: Optional[Path] = None,
    main_val: str = "val",
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model").mkdir(exist_ok=True)
    last_path = out_dir / "model" / "last.pt"
    if resume and not last_path.exists():
        raise FileNotFoundError(f"--resume asked but {last_path} does not exist")
    if not resume and last_path.exists() and not smoke:
        raise FileExistsError(f"{out_dir} already holds a run; pass --resume to continue it or use another --tag")
    cfg.to_json(out_dir / "config.json")
    seed_all(cfg.train.seed)

    t0 = time.time()
    train_tables = load_variants(train_ids, cfg.data, n_damaged, keep_clean)
    train_ds = WindowDataset(train_tables, cfg.data, cfg.model.team_size, seed=cfg.train.seed)
    val_loaders: Dict[str, DataLoader] = {}
    nw = 0 if smoke else cfg.train.num_workers
    for name, ids in val_sets.items():
        # "<name>" uses the first damaged draw when there is one (a damaged validation), "<name>_clean" the clean table
        for suffix, variant in (("", 1 if n_damaged > 0 else 0), ("_clean", 0)):
            if suffix == "_clean" and n_damaged == 0:
                continue
            tables = {}
            for k in ids:
                try:
                    tables[k] = load_table(k, cfg.data, variant)
                except FileNotFoundError:
                    if variant == 0:
                        raise
            if not tables:
                continue
            ds = WindowDataset(tables, cfg.data, cfg.model.team_size, flip=cfg.data.flip if cfg.data.flip_valid else "none", seed=cfg.train.seed + 1,
                               ball_damage=(suffix != "_clean"), deterministic=True)
            val_loaders[name + suffix] = DataLoader(ds, cfg.train.batch_size, shuffle=False, num_workers=nw, collate_fn=collate)
    print(f"train windows per epoch {len(train_ds)} ({train_ds.n_variants()} variants per match) | "
          + ", ".join(f"{k} {len(v.dataset)}" for k, v in val_loaders.items()) + f" | data {time.time() - t0:.0f}s", flush=True)
    if main_val not in val_loaders:
        main_val = next(iter(val_loaders))
    train_loader = DataLoader(train_ds, cfg.train.batch_size, shuffle=True, num_workers=nw, collate_fn=collate, pin_memory=device.type == "cuda")

    net = PathCRFNet(cfg.model).to(device)
    if init_from is not None and not resume:
        init_from_checkpoint(net, init_from)
        print(f"initialised from {init_from}")
    print(f"parameters {net.num_parameters():,}")
    lr = max(cfg.train.lr, cfg.train.min_lr)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=cfg.train.weight_decay)
    best_loss, best_acc, since_best, start_epoch = np.inf, 0.0, 0, 1
    best_spell = 0.0                 # spellF1 option 0: checkpoint with the best window spell F1 on the main validation
    history = []
    rng = np.random.default_rng(cfg.train.seed + 7)
    if resume:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        lr, best_loss, best_acc, since_best = ck["lr"], ck["best_loss"], ck["best_acc"], ck["since_best"]
        best_spell = ck.get("best_spell", 0.0)
        start_epoch = ck["epoch"] + 1
        torch.set_rng_state(ck["rng_cpu"].cpu())
        if device.type == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        if ck.get("rng_np") is not None:
            rng.bit_generator.state = ck["rng_np"]
        print(f"resumed from epoch {ck['epoch']} (lr {lr:.1e}, best acc {best_acc:.4f}, best loss {best_loss:.3f})")
    log_f = open(out_dir / "log.jsonl", "a")
    for epoch in range(start_epoch, (1 if smoke else cfg.train.epochs) + 1):
        if since_best >= cfg.train.plateau_patience and lr > cfg.train.min_lr:
            lr = max(lr * 0.5, cfg.train.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            since_best = 0
        if train_ds.n_variants() > 1:
            train_ds.resample(rng)
        t0 = time.time()
        tr = run_epoch(net, train_loader, device, cfg.train, opt, max_batches=3 if smoke else None)
        vals = {name: run_epoch(net, dl, device, cfg.train, None, max_batches=2 if smoke else None) for name, dl in val_loaders.items()}
        va = vals[main_val]
        val_total = sum(v for k, v in va.items() if k.endswith("_loss"))
        rec = {"epoch": epoch, "lr": lr, "time_s": round(time.time() - t0, 1), "train": tr, "val": va, "val_total_loss": val_total,
               "vals": vals, "n_train_windows": len(train_ds)}
        history.append(rec)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        extra = " ".join(f"{k} {v['edge_acc']:.3f}" for k, v in vals.items() if k != main_val)
        print(f"epoch {epoch:3d} lr {lr:.1e} {rec['time_s']:6.0f}s | train loss {tr['loss']:.3f} acc {tr['edge_acc']:.3f} | "
              f"{main_val} loss {val_total:.3f} acc {va['edge_acc']:.3f} spell {va.get('spell_f1', 0):.3f} "
              f"switch {va.get('switch_f1', 0):.3f} team {va.get('team_acc', 0):.3f} | {extra}", flush=True)
        if va.get("spell_f1", 0.0) > best_spell:
            best_spell = va["spell_f1"]
            torch.save({"model": net.state_dict(), "config": asdict(cfg.model), "epoch": epoch}, out_dir / "model" / "best_spell.pt")
        if epoch == 1 or val_total < best_loss:
            best_loss, since_best = val_total, 0
            torch.save({"model": net.state_dict(), "config": asdict(cfg.model), "epoch": epoch}, out_dir / "model" / "best_loss.pt")
        else:
            since_best += 1
        if va["edge_acc"] > best_acc:
            best_acc, since_best = va["edge_acc"], 0
            torch.save({"model": net.state_dict(), "config": asdict(cfg.model), "epoch": epoch}, out_dir / "model" / "best_acc.pt")
        if epoch % cfg.train.save_every == 0:
            torch.save({"model": net.state_dict(), "config": asdict(cfg.model), "epoch": epoch}, out_dir / "model" / f"epoch_{epoch:03d}.pt")
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "config": asdict(cfg.model), "epoch": epoch,
                    "lr": lr, "best_loss": best_loss, "best_acc": best_acc, "since_best": since_best, "best_spell": best_spell,
                    "rng_cpu": torch.get_rng_state(), "rng_np": rng.bit_generator.state,
                    "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None}, last_path)
    log_f.close()
    return {"best_val_loss": best_loss, "best_val_edge_acc": best_acc, "best_val_spell_f1": best_spell, "epochs": start_epoch - 1 + len(history)}


# ------------------------------------------------------------------------------------------ CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="6", help="LOMOCV fold index (test match), or 'all'")
    ap.add_argument("--split", default=None, choices=["phaseb"], help="preset split instead of --fold")
    ap.add_argument("--train-ids", default=None, help="comma-separated match keys (Sportec id or sc_<id>)")
    ap.add_argument("--val-ids", default=None, help="comma-separated match keys for the main validation set")
    ap.add_argument("--tag", default="replica")
    ap.add_argument("--config", default=None, help="json written by Config.to_json; default = saved/140 recipe")
    ap.add_argument("--transition", choices=["linear", "mlp"], default=None)
    ap.add_argument("--flip", choices=["none", "legacy", "fixed"], default=None)
    ap.add_argument("--zero-channels", type=int, default=None)
    ap.add_argument("--presence", action="store_true", help="9th input feature: 1 = real observation, 0 = filled / padded")
    ap.add_argument("--mask-absent", action="store_true", help="also mask absent nodes out of the social attention")
    ap.add_argument("--ball", action="store_true", help="Phase B.B: per-node ball channels at the input (data/ball.py), damaged on the fly")
    ap.add_argument("--no-ball-damage", action="store_true")
    ap.add_argument("--damaged", type=int, default=0, help="damaged draws per training match (cache/damaged/<key>_v<k>.parquet)")
    ap.add_argument("--no-clean", action="store_true", help="drop the clean table from the training variants")
    ap.add_argument("--init-from", default=None, help="checkpoint to start from (Phase A weights)")
    ap.add_argument("--window-stride", type=int, default=None)
    ap.add_argument("--target-fps", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true", help="3 training batches + 2 validation batches, then stop")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast on the backbone (CRF and losses stay fp32)")
    ap.add_argument("--resume", action="store_true", help="continue the run in outputs/runs/<tag>[_fold<k>] from its last.pt")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--macro-weight", type=float, default=None, help="spellF1 option 2: weight of the coarse-head losses (default 1)")
    ap.add_argument("--micro-weight", type=float, default=None, help="option 2: weight of the edge-head losses (default 1)")
    ap.add_argument("--crf-weight", type=float, default=None, help="option 2: weight of the CRF path likelihood (default 1)")
    ap.add_argument("--team-loss", type=float, default=None, help="option 3: weight of the BCE on P(sender team = home)")
    ap.add_argument("--switch-loss", type=float, default=None, help="option 3: weight of the BCE on the per-step team-switch probability")
    ap.add_argument("--switch-pos-weight", type=float, default=None, help="option 3: weight of true switches in the switch loss (default 5)")
    ap.add_argument("--boundary-weight", type=float, default=None, help="option 4: CE weight within +-1 s of a true team switch (default 1 = off)")
    ap.add_argument("--margin-c1", type=float, default=None, help="option 5: softmax-margin cost per wrong-team step")
    ap.add_argument("--margin-c2", type=float, default=None, help="option 5: softmax-margin cost per false team switch")
    a = ap.parse_args()

    cfg = Config.from_json(a.config) if a.config else Config()
    if a.transition:
        cfg.model.transition = a.transition
    if a.flip:
        cfg.data.flip = a.flip
    if a.zero_channels is not None:
        cfg.model.extra_zero_channels = a.zero_channels
    if a.presence:
        cfg.model.presence = True
    if a.mask_absent:
        cfg.model.presence = True
        cfg.model.mask_absent = True
    if a.ball:
        from .data.ball import N_BALL_FEATURES
        cfg.model.ball_features = N_BALL_FEATURES
        cfg.data.ball = True
    if a.no_ball_damage:
        cfg.data.ball_damage = False
    if a.window_stride:
        cfg.data.window_stride = a.window_stride
    if a.target_fps:
        cfg.data.target_fps = a.target_fps
    if a.epochs:
        cfg.train.epochs = a.epochs
    if a.batch_size:
        cfg.train.batch_size = a.batch_size
    if a.lr:
        cfg.train.lr = a.lr
    if a.seed is not None:
        cfg.train.seed = a.seed
    if a.amp:
        cfg.train.amp = True
    if a.num_workers is not None:
        cfg.train.num_workers = a.num_workers
    for name in ("macro_weight", "micro_weight", "crf_weight", "switch_pos_weight", "boundary_weight", "margin_c1", "margin_c2"):
        v = getattr(a, name)
        if v is not None:
            setattr(cfg.train, name, v)
    if a.team_loss is not None:
        cfg.train.team_weight = a.team_loss
    if a.switch_loss is not None:
        cfg.train.switch_weight = a.switch_loss
    device = pick_device(a.device)
    init_from = Path(a.init_from) if a.init_from else None

    if a.split == "phaseb" or a.train_ids:
        if a.train_ids:
            train_ids = a.train_ids.split(",")
            val_sets = {"val": a.val_ids.split(",")} if a.val_ids else {}
        else:
            train_ids, val_sets = phaseb_split()
        out_dir = OUTPUTS / "runs" / a.tag
        print(f"train {train_ids}\nval {val_sets}\n-> {out_dir}")
        res = fit(cfg, train_ids, val_sets, out_dir, device, smoke=a.smoke, resume=a.resume, n_damaged=a.damaged,
                  keep_clean=not a.no_clean, init_from=init_from)
        (out_dir / "result.json").write_text(json.dumps({**res, "train": train_ids, "val": val_sets}, indent=2))
        print(res)
        return

    ids = match_ids(SPORTEC_DIR)
    folds = range(len(ids)) if a.fold == "all" else [int(a.fold)]
    for fold in folds:
        train_ids, val_ids, test_ids = lomocv_split(ids, fold)
        out_dir = OUTPUTS / "runs" / f"{a.tag}_fold{fold}"
        print(f"fold {fold}: train {train_ids} val {val_ids} test {test_ids} -> {out_dir}")
        res = fit(cfg, train_ids, {"val": val_ids}, out_dir, device, smoke=a.smoke, resume=a.resume, n_damaged=a.damaged,
                  keep_clean=not a.no_clean, init_from=init_from)
        (out_dir / "result.json").write_text(json.dumps({**res, "train": train_ids, "val": val_ids, "test": test_ids}, indent=2))
        print(res)


if __name__ == "__main__":
    main()
