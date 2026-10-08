"""Paths, constants and the knobs that other modules are allowed to read.

Anything here that changes a fitted number is a tunable and must be echoed into
the run manifest (see `src/runlog.py`) so a result can be reproduced exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

for _d in (RAW, INTERIM, PROCESSED, RESULTS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data scope

# Understat league keys as `soccerdata` names them.
LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]

# Season labels are the starting year: 2014 == 2014/15.
DEV_SEASONS = [2014, 2015, 2016, 2017, 2018]
HOLDOUT_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
FIRST_HOLDOUT_SEASON = 2019

# --------------------------------------------------------------- match clock

# Nominal regulation length. Per-match T is taken as max(T_NOMINAL, last event + 1)
# so stoppage-time shots are inside the observation window.
T_NOMINAL = 95.0
BIN_MINUTES = 1.0

SITUATIONS = ("open", "setpiece", "penalty")
PENALTY_XG = 0.76  # documented in CLAUDE.md; used only as a sanity check


@dataclass(frozen=True)
class DedupConfig:
    """Collapse rule for mechanical follow-ups (rebounds, corner chains)."""

    # Calibrated on StatsBomb Premier League 2015/16: the excess over the renewal
    # baseline is confined to 0-4 s and is gone by 5 s. See results/dedup_calibration.json.
    threshold_seconds: float = 4.0
    # Penalties are never collapsed into a neighbouring shot.
    protect_penalties: bool = True
    # Refuse the rule if it eats more than this fraction of shots (CLAUDE.md step 2).
    max_removed_fraction: float = 0.15


@dataclass(frozen=True)
class BackgroundConfig:
    """Inhomogeneous-Poisson background mu_k(t)."""

    minute_df: int = 6  # natural cubic spline df for f(minute)
    score_bins: tuple[int, ...] = (-2, -1, 0, 1, 2)  # clipped score differential
    red_bins: tuple[int, ...] = (-1, 0, 1)  # clipped red-card differential
    team_ridge: float = 1.0  # L2 on team attack/defence effects only
    include_score: bool = True  # sensitivity fit sets this False
    include_red: bool = True
    maxiter: int = 500
    tol: float = 1e-8


@dataclass(frozen=True)
class HawkesConfig:
    """Single slow exponential kernel. A fast component is deliberately absent."""

    # 1/beta box, in minutes: the spec's 3-8 min band (CLAUDE.md, "Model
    # specification"). Fixed at this before the pooled slate was first fitted
    # (2026-09-17). The earlier 1-20 box pinned at 20 on 380 StatsBomb matches
    # while the tau-profile showed eta and tau trading off almost freely; a wide
    # box therefore reports whatever its edge is. The wide box is kept as a
    # sensitivity fit and the profile is reported alongside.
    tau_min: float = 3.0
    tau_max: float = 8.0
    tau_init: float = 5.0
    rho_max: float = 0.99  # spectral-radius cap enforcing stationarity
    eta_self_init: float = 0.05
    eta_cross_init: float = 0.05
    symmetric: bool = True  # eta_HH == eta_AA, eta_HA == eta_AH
    maxiter: int = 400
    # Penalties excite nothing and are excited by nothing: dropped entirely.
    exclude_penalties: bool = True


@dataclass(frozen=True)
class JitterConfig:
    """Uniform jitter inside the recorded minute, for Understat's integer clock."""

    n_draws: int = 5
    base_seed: int = 20260910


@dataclass(frozen=True)
class PowerConfig:
    eta_grid: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
    replicates: int = 30
    null_replicates: int = 200  # the eta*=0 arm sets the critical value
    alpha: float = 0.05
    target_power: float = 0.80
    seed: int = 5694


@dataclass(frozen=True)
class Config:
    dedup: DedupConfig = field(default_factory=DedupConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    hawkes: HawkesConfig = field(default_factory=HawkesConfig)
    jitter: JitterConfig = field(default_factory=JitterConfig)
    power: PowerConfig = field(default_factory=PowerConfig)

    def to_dict(self) -> dict:
        return asdict(self)


CONFIG = Config()
