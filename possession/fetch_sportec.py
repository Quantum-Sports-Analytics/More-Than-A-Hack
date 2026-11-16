"""Download IDSSE (Sportec Open DFL) positions from the HF mirror, parse to compact npz, delete xml.

Run from a directory OUTSIDE the repo: python fetch_sportec.py J03WMX J03WN1 J03WOH J03WOY J03WPY J03WQQ J03WR9
"""
import os, re, sys, subprocess, numpy as np, xml.etree.ElementTree as ET
BASE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
COM = {"J03WMX": "000001", "J03WN1": "000001"}
rx = re.compile(r'N="(\d+)".*?X="([-\d.]+)" Y="([-\d.]+)"(?: Z="([-\d.]+)")?')
rx_attr = re.compile(r'(\w+)="([^"]*)"')

def get(fname):
    if not os.path.exists(fname):
        r = subprocess.run(["curl", "-sL", "--fail", "-o", fname + ".part", BASE + fname])
        if r.returncode: raise RuntimeError(f"download failed {fname}")
        os.replace(fname + ".part", fname)
    return fname

def parse(mid):
    com = COM.get(mid, "000002")
    meta = get(f"DFL_02_01_matchinformation_DFL-COM-{com}_DFL-MAT-{mid}.xml")
    root = ET.parse(meta).getroot()
    g = root.find(".//General"); env = root.find(".//Environment")
    home, away = g.get("HomeTeamId"), g.get("GuestTeamId")
    L, W = float(env.get("PitchX")), float(env.get("PitchY"))
    pos = get(f"DFL_04_03_positions_raw_observed_DFL-COM-{com}_DFL-MAT-{mid}.xml")
    sets = {}                  # (section, person) -> dict
    cur = None
    with open(pos) as fh:
        for line in fh:
            if line.startswith("<Frame "):
                if cur is None: continue
                if cur["ball"]:
                    a = dict(rx_attr.findall(line))
                    cur["rows"].append((int(a["N"]), float(a["X"]), float(a["Y"]), float(a.get("Z", "nan")),
                                        int(a.get("BallStatus", -1)), int(a.get("BallPossession", -1))))
                else:
                    m = rx.search(line)
                    cur["rows"].append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
            elif line.startswith("<FrameSet "):
                a = dict(rx_attr.findall(line))
                team = a["TeamId"]
                cur = dict(section=a["GameSection"], team=team, ball=(team.upper() == "BALL"), rows=[])
                if team not in (home, away) and not cur["ball"]:
                    cur = None                       # referees etc.
                else:
                    sets[(a["GameSection"], a["PersonId"])] = cur
            elif line.startswith("</FrameSet"):
                cur = None
    out = {}
    for sec in ("firstHalf", "secondHalf"):
        keys = [k for k in sets if k[0] == sec]
        ball = [sets[k] for k in keys if sets[k]["ball"]][0]
        br = np.array(ball["rows"], dtype=float)
        n0, n1 = int(br[:, 0].min()), int(br[:, 0].max())
        T = n1 - n0 + 1
        pk = [k for k in keys if not sets[k]["ball"]]
        PX = np.full((T, len(pk)), np.nan, np.float32); PY = PX.copy()
        team = np.array([0 if sets[k]["team"] == home else 1 for k in pk], np.int8)
        for j, k in enumerate(pk):
            r = np.array(sets[k]["rows"], dtype=float)
            ok = (r[:, 0] >= n0) & (r[:, 0] <= n1); r = r[ok]
            idx = r[:, 0].astype(int) - n0
            PX[idx, j], PY[idx, j] = r[:, 1], r[:, 2]
        BX = np.full(T, np.nan, np.float32); BY = BX.copy(); BZ = BX.copy()
        BS = np.full(T, -1, np.int8); BP = BS.copy()
        idx = br[:, 0].astype(int) - n0
        BX[idx], BY[idx], BZ[idx], BS[idx], BP[idx] = br[:, 1], br[:, 2], br[:, 3], br[:, 4], br[:, 5]
        p = "h1" if sec == "firstHalf" else "h2"
        out.update({f"{p}_PX": PX, f"{p}_PY": PY, f"{p}_team": team, f"{p}_BX": BX, f"{p}_BY": BY,
                    f"{p}_BZ": BZ, f"{p}_BS": BS, f"{p}_BP": BP, f"{p}_n0": n0})
    np.savez_compressed(f"{mid}.npz", L=L, W=W, **out)
    os.remove(pos)
    print(mid, "ok", {k: v.shape for k, v in out.items() if hasattr(v, "shape") and v.ndim}, flush=True)

if __name__ == "__main__":
    for mid in sys.argv[1:]:
        if os.path.exists(f"{mid}.npz"): print(mid, "cached"); continue
        try: parse(mid)
        except Exception as e: print(mid, "FAILED", repr(e), flush=True)
