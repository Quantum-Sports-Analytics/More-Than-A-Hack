"""Every hyper-parameter in one place. Defaults reproduce the authors' trained model `saved/140`."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

PITCH_X, PITCH_Y = 105.0, 68.0
OUTSIDE_NODES = ("out_left", "out_right", "out_bottom", "out_top")
OUTSIDE_XY = {
    "out_left": (0.0, PITCH_Y / 2),
    "out_right": (PITCH_X, PITCH_Y / 2),
    "out_bottom": (PITCH_X / 2, 0.0),
    "out_top": (PITCH_X / 2, PITCH_Y),
}
FEATURE_NAMES = ("team", "gk", "x", "y", "vx", "vy", "speed", "accel")


@dataclass
class ModelConfig:
    node_in_dim: int = 8            # team flag, gk flag, x, y, vx, vy, speed, accel
    team_size: int = 11
    n_outside: int = 4
    coarse_dim: int = 16            # PPE and FPE output size of the coarse pass
    fine_dim: int = 64              # PPE and FPE output size of the fine pass
    seq_dim: int = 128              # temporal (SAB) width, also the edge width
    set_dim: int = 128              # hidden width inside every Set Transformer encoder
    inducing_points: int = 16
    heads: int = 4
    crf_edge_dim: int = 16
    dropout: float = 0.1
    # SHORTCUT 3 (official quirk): two always-zero channels appended to the coarse input and to the
    # fine temporal input. Keep 2 for the weight port; 0 for new training.
    extra_zero_channels: int = 2
    # SHORTCUT 2: "linear" = the official Linear(32, 1) on [e_prev; e_curr] (prev and curr never interact);
    # "mlp" = Linear(32, h) - ReLU - Linear(h, 1), the paper's description.
    transition: str = "linear"
    trans_hidden: int = 32
    forbid_score: float = -1e4
    # Phase B: a 9th "present" feature (1 = real observation, 0 = filled / padded). Off = exact replication.
    presence: bool = False
    # Phase B option: also mask absent nodes out of the social attention (key mask). Off = flag only.
    mask_absent: bool = False
    # Phase B.B: per-node ball channels appended to the input (data/ball.py: dx, dy, dist, ball speed, missing). 0 = no ball.
    ball_features: int = 0

    @property
    def n_players(self) -> int:
        return 2 * self.team_size

    @property
    def n_nodes(self) -> int:
        return 2 * self.team_size + self.n_outside

    @property
    def in_dim(self) -> int:
        return self.node_in_dim + (1 if self.presence else 0) + self.ball_features


@dataclass
class DataConfig:
    source_fps: float = 25.0
    target_fps: float = 5.0
    window_seconds: float = 10.0
    window_stride: int = 1          # in downsampled steps
    flip: str = "legacy"            # "none" | "legacy" (official, with its outside-node bug) | "fixed"
    flip_valid: bool = True         # the official code augments the validation set too
    min_episode_frames: int = 250   # episodes shorter than window_seconds * fps are dropped
    # feature extraction
    speed_outlier: float = 12.0     # m/s; faster frame-to-frame jumps are removed and interpolated
    savgol_vel: tuple = (15, 2)
    savgol_acc: tuple = (9, 2)
    # touch refinement
    rdp_epsilon: float = 0.5
    rdp_min_angle: float = 15.0
    rdp_frame_scale: float = 1e-6
    touch_max_dist: float = 5.0
    # label sanitising (SHORTCUT 4): cut an episode at its first out/goal touch
    sanitize_after_out: bool = True
    # Phase B.B: carry the ball through the windows (data/ball.py) and damage it on the fly during training
    ball: bool = False
    ball_damage: bool = True

    @property
    def sample_every(self) -> int:
        k = self.source_fps / self.target_fps
        assert abs(k - round(k)) < 1e-9, "source_fps must be a multiple of target_fps"
        return int(round(k))

    @property
    def window_steps(self) -> int:
        return int(round(self.window_seconds * self.target_fps))


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    plateau_patience: int = 3       # halve the lr after this many epochs without a new best
    grad_clip: Optional[float] = None
    seed: int = 100
    num_workers: int = 4
    macro_weight: float = 1.0
    micro_weight: float = 1.0
    crf_weight: float = 1.0
    emit_loss: bool = True
    src_loss: bool = True
    dst_loss: bool = True
    save_every: int = 10
    team_weight: float = 0.0      # spellF1 option 3: BCE on P(sender team = home) from the edge logits
    switch_weight: float = 0.0    # option 3: BCE on the per-step 'team changes' probability vs the true switches
    switch_pos_weight: float = 5.0  # weight of the true switches in the switch loss (they are rare)
    boundary_weight: float = 1.0    # option 4: per-step CE weight within +-1 s of a true team switch (1 = off)
    boundary_steps: int = 5         # half-width of the boundary zone, in 5 fps steps
    margin_c1: float = 0.0          # option 5: cost per step whose sender team differs from the true team
    margin_c2: float = 0.0          # option 5: cost per team-switching transition where the truth does not switch
    device: str = "auto"
    amp: bool = False               # bf16 autocast on the backbone; CRF and losses stay fp32


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=list))

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        d = json.loads(Path(path).read_text())
        return cls(
            model=ModelConfig(**d.get("model", {})),
            data=DataConfig(**_tuples(DataConfig, d.get("data", {}))),
            train=TrainConfig(**d.get("train", {})),
        )

    @classmethod
    def from_official_args(cls, path: str | Path) -> "Config":
        """Read the authors' `args.json` (e.g. external/pathcrf/saved/140/args.json)."""
        a = json.loads(Path(path).read_text())
        m = ModelConfig(
            node_in_dim=a.get("node_in_dim", 8),
            team_size=a.get("team_size", 11),
            coarse_dim=a.get("coarse_dim", 16),
            fine_dim=a.get("fine_dim", 64),
            seq_dim=a.get("seq_dim", 128),
            crf_edge_dim=a.get("crf_edge_dim", 16),
            dropout=a.get("dropout", 0.1),
        )
        d = DataConfig(
            target_fps=a.get("target_fps", 5.0),
            window_seconds=a.get("window_seconds", 10.0),
            window_stride=a.get("window_stride", 1),
            flip="legacy" if a.get("flip_pitch", False) else "none",
        )
        t = TrainConfig(
            epochs=a.get("n_epochs", 50),
            batch_size=a.get("batch_size", 32),
            lr=a.get("start_lr", 1e-3),
            min_lr=a.get("min_lr", 1e-5),
            weight_decay=a.get("weight_decay", 1e-4),
            grad_clip=a.get("clip"),
            seed=a.get("seed", 100),
            macro_weight=a.get("macro_weight", 1.0),
            crf_weight=a.get("crf_weight", 1.0),
            emit_loss=a.get("emit_loss", True),
            src_loss=a.get("src_loss", True),
            dst_loss=a.get("dst_loss", True),
        )
        return cls(model=m, data=d, train=t)


def _tuples(cls, d: dict) -> dict:
    """JSON turns tuples into lists; turn them back for the fields declared as tuples."""
    out = dict(d)
    for f in fields(cls):
        if f.type == "tuple" and f.name in out and isinstance(out[f.name], list):
            out[f.name] = tuple(out[f.name])
    return out
