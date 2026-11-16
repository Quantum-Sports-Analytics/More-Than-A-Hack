"""Where things live. Everything is relative to the hackathon root (the parent of this package folder)."""
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]          # .../pathcrf_ours
HACK_ROOT = PKG_ROOT.parent                              # .../Metrica X Barca Hackathon
OFFICIAL = HACK_ROOT / "external" / "pathcrf"
SPORTEC_DIR = OFFICIAL / "data" / "sportec"              # metadata/, event/, tracking/ (symlinks to the IDSSE XML)
SYNCED_DIR = OFFICIAL / "data" / "sportec_synced"        # ELASTIC-synchronised events (authors' files)
OFFICIAL_PROCESSED = OFFICIAL / "data" / "sportec" / "tracking_processed"
OFFICIAL_TOUCHES = OFFICIAL / "data" / "sportec" / "event_processed"
OFFICIAL_WEIGHTS = OFFICIAL / "saved" / "140" / "model" / "state_dict_best_acc.pt"
CACHE = PKG_ROOT / "cache"
OUTPUTS = PKG_ROOT / "outputs"
