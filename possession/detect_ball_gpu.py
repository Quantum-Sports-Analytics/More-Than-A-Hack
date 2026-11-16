"""Ball candidate detection on the tactical video (GPU, Cloud Batch).

Per frame, independently (so frames can be batched): RF-DETR medium (COCO, class 37 'sports ball')
on the full frame at FULLRES, plus x2-upscaled 192-px crops (CROPRES) centred on (a) the ATD ball
projected to pixels and (b) the two best full-frame candidates. Raw candidates only; tracking,
static-mark blacklist and fusion with the ATD are done offline.
Outputs: <out>/part_<start>.npz (frame, src, u, v, conf) every CHUNK frames + <out>/status.json.
src: 0 full frame, 1 crop around ATD ball, 2 crop around full-frame candidate.
"""
import argparse, json, os, time
import numpy as np, cv2, torch

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--atd", required=True)
ap.add_argument("--start", type=int, required=True); ap.add_argument("--end", type=int, required=True)
ap.add_argument("--out", required=True); ap.add_argument("--fullres", type=int, default=1440)
ap.add_argument("--cropres", type=int, default=384); ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--chunk", type=int, default=5000)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
T0 = time.time()
def status(**kw):
    kw.update(elapsed_s=round(time.time() - T0, 1), cuda=torch.cuda.is_available(),
              device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    json.dump(kw, open(os.path.join(a.out, "status.json.tmp"), "w")); os.replace(os.path.join(a.out, "status.json.tmp"), os.path.join(a.out, "status.json"))
if not torch.cuda.is_available():
    status(state="FAILED_NO_GPU"); raise SystemExit(2)
status(state="LOADING")
from rfdetr import RFDETRMedium
full = RFDETRMedium(resolution=a.fullres, device="cuda"); crop = RFDETRMedium(resolution=a.cropres, device="cuda")
CB = 3 * a.batch   # max crops per batch of frames (1 ATD + 2 candidates per frame)
for m, bs in ((full, a.batch), (crop, CB)):
    try: m.inference(compile=True, batch_size=bs)
    except Exception as e: print("inference() compile skipped:", e, flush=True)
atd = np.load(a.atd)
cap = cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
def balls(det, ox=0.0, oy=0.0, s=1.0, k=8):
    m = det.class_id == 37
    if not m.any(): return np.zeros((0, 3), np.float32)
    xy = det.xyxy[m] / s + [ox, oy, ox, oy]; c = det.confidence[m]
    out = np.column_stack([(xy[:, 0] + xy[:, 2]) / 2, (xy[:, 1] + xy[:, 3]) / 2, c]); return out[np.argsort(-c)][:k]
rows, t_dec, t_full, t_crop, part_start = [], 0.0, 0.0, 0.0, a.start
def flush(upto):
    global rows, part_start
    arr = np.array(rows, np.float32).reshape(-1, 5)
    np.savez_compressed(os.path.join(a.out, f"part_{part_start:06d}.npz"), frame=arr[:, 0].astype(np.int32), src=arr[:, 1].astype(np.int8),
                        u=arr[:, 2], v=arr[:, 3], conf=arr[:, 4], first=part_start, last=upto)
    rows, part_start = [], upto + 1
f = a.start
while f < a.end:
    t = time.time(); frames, idx = [], []
    while len(frames) < a.batch and f < a.end:
        ok, img = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); idx.append(f); f += 1
    t_dec += time.time() - t
    if not frames: break
    n_real = len(frames); pad = frames + [frames[-1]] * (a.batch - n_real)
    t = time.time(); dets = full.predict(pad, threshold=0.1); t_full += time.time() - t
    if not isinstance(dets, list): dets = [dets]
    dets = dets[:n_real]
    crops, meta = [], []
    for fr, rgb, d in zip(idx, frames, dets):
        cand = balls(d)
        for u, v, c in cand: rows.append((fr, 0, u, v, c))
        centres = []
        if np.isfinite(atd[fr]).all(): centres.append((1, atd[fr][0], atd[fr][1]))
        for u, v, c in cand[:2]: centres.append((2, u, v))
        for src, cu, cv in centres:
            x0 = int(np.clip(cu - 96, 0, 1920 - 192)); y0 = int(np.clip(cv - 96, 0, 1080 - 192))
            crops.append(cv2.resize(rgb[y0:y0 + 192, x0:x0 + 192], (a.cropres, a.cropres), interpolation=cv2.INTER_CUBIC)); meta.append((fr, src, x0, y0))
    if crops:
        n_c = len(crops); blank = np.zeros((a.cropres, a.cropres, 3), np.uint8)
        t = time.time(); cd = crop.predict(crops + [blank] * (CB - n_c), threshold=0.15); t_crop += time.time() - t
        if not isinstance(cd, list): cd = [cd]
        cd = cd[:n_c]
        for (fr, src, x0, y0), d in zip(meta, cd):
            for u, v, c in balls(d, x0, y0, a.cropres / 192.0, k=3): rows.append((fr, src, u, v, c))
    done = f - a.start
    if done % 500 < a.batch or f >= a.end:
        fps = done / max(time.time() - T0, 1e-6)
        status(state="RUNNING", done=done, total=a.end - a.start, fps=round(fps, 2), eta_s=round((a.end - f) / max(fps, 1e-6)),
               t_decode=round(t_dec, 1), t_full=round(t_full, 1), t_crop=round(t_crop, 1))
    if f - part_start >= a.chunk: flush(f - 1)
flush(f - 1)
status(state="DONE", done=f - a.start, total=a.end - a.start, t_decode=round(t_dec, 1), t_full=round(t_full, 1), t_crop=round(t_crop, 1))
