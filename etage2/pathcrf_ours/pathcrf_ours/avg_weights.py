"""Weight averaging of same-basin checkpoints (weekend bundle, board item 28): the parameters are averaged tensor by
tensor; the config of the first checkpoint is kept.

    .venv/bin/python -m pathcrf_ours.avg_weights --weights A.pt B.pt ... --out outputs/runs/avg/<name>.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cks = [torch.load(w, map_location="cpu", weights_only=False) for w in a.weights]
    # keys common to every checkpoint (newer checkpoints carry extra CRF buffers such as switch_pair; load_net recomputes them)
    keys = [k for k in cks[0]["model"] if all(k in ck["model"] for ck in cks)]
    dropped = sorted({k for ck in cks for k in ck["model"]} - set(keys))
    if dropped:
        print("not averaged (missing in some checkpoints, recomputed at load):", dropped)
    avg = {}
    for k in keys:
        v = cks[0]["model"][k]
        if any(ck["model"][k].shape != v.shape for ck in cks):
            raise ValueError(f"shape mismatch on {k}")
        avg[k] = sum(ck["model"][k].float() for ck in cks).div(len(cks)).to(v.dtype) if v.is_floating_point() else v
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": avg, "config": cks[0].get("config"), "epoch": None, "averaged_from": a.weights}, out)
    print(f"averaged {len(cks)} checkpoints -> {out}")


if __name__ == "__main__":
    main()
