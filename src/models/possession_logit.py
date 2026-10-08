"""Model A of docs/spec_possessions.md: possession-sequence logistic regression.

One row per possession p, D_p = 1 if it reaches the danger zone:

    logit P(D_p = 1) = θ0 + θ_home + a_team + d_opp + f(minute) + g(score) + h(red)
                     + c · setpiece(p) + rho_self · S_self(p) + rho_cross · S_cross(p)

    S_self(p)  = Σ_{q<p, same team, D_q=1} exp(-(t_p - t_q)/τ)
    S_cross(p) = Σ_{q<p, opponent,  D_q=1} exp(-(t_p - t_q)/τ)

with t the possession start. The background terms copy the shot model
(`models/baseline.py`): team effects per (source, team, season) with the same ridge,
the same natural spline in minute evaluated at one-minute bin centres, and the
same clipped score and red-card bins with 0 as the reference level.

The alternation that defeats a shot-style Hawkes fit on possessions (the pilot's
refractory gap) is not a problem here: the unit of analysis is the possession, so
"the opponent has the ball in between" is the data's structure, not something the
background has to absorb.

Unlike the Hawkes η, rho is not bounded below. A negative rho_self is a legitimate
answer ("a dangerous possession makes the next one less likely"), so it is
estimated unconstrained and the test is one-sided only at the reporting stage.

The excitation sums are a recursion over possessions, run across all matches at
once on a padded (match x possession) grid so that simulation, which has to draw
D_p sequentially, costs one vector step per possession index rather than one
Python step per possession.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from src.config import BackgroundConfig
from src.models.baseline import BIN_MINUTES
from src.models.spline import natural_cubic_basis, natural_cubic_knots


@dataclass
class PossDesign:
    """Everything the likelihood needs, with rows sorted by (match, t_start)."""

    cfg: BackgroundConfig
    match_id: np.ndarray
    t: np.ndarray
    is_home: np.ndarray
    team_idx: np.ndarray
    opp_idx: np.ndarray
    team_labels: np.ndarray
    bin_idx: np.ndarray
    basis: np.ndarray
    score_idx: np.ndarray
    red_idx: np.ndarray
    score_levels: np.ndarray
    red_levels: np.ndarray
    setpiece: np.ndarray
    y: np.ndarray
    grid: np.ndarray  # (n_matches, max_len) row index, -1 = padding
    include_score: bool = True
    include_red: bool = True
    include_setpiece: bool = True
    labels: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.t.size


def build_design(
    table: pd.DataFrame,
    cfg: BackgroundConfig,
    *,
    danger: str = "ft",
    include_setpiece: bool = True,
) -> PossDesign:
    """`table` is one rule's rows from `ingest.possessions.load`. `danger` picks the
    definition: ft (headline), ft75 or box."""
    df = table.sort_values(["match_id", "t_start"], kind="stable").reset_index(drop=True)
    src = df["source"].astype(str)
    key_t = src + "|" + df["team"].astype(str) + "|" + df["season"].astype(str)
    key_o = src + "|" + df["opponent"].astype(str) + "|" + df["season"].astype(str)
    team_labels = np.sort(pd.unique(pd.concat([key_t, key_o])))
    pos = {lab: i for i, lab in enumerate(team_labels)}

    t = df["t_start"].to_numpy(dtype=np.float64)
    bin_idx = np.floor(t / BIN_MINUTES).astype(np.int32)
    n_bins = int(bin_idx.max()) + 1
    centres = (np.arange(n_bins, dtype=np.float64) + 0.5) * BIN_MINUTES
    basis = natural_cubic_basis(centres, natural_cubic_knots(cfg.minute_df, 0.0, centres[-1]))

    score_levels = np.array(cfg.score_bins, dtype=np.int64)
    red_levels = np.array(cfg.red_bins, dtype=np.int64)
    s_pos = {v: i for i, v in enumerate(score_levels)}
    r_pos = {v: i for i, v in enumerate(red_levels)}

    mcodes, mid = pd.factorize(df["match_id"], sort=False)
    counts = np.bincount(mcodes)
    grid = np.full((counts.size, int(counts.max())), -1, dtype=np.int64)
    within = df.groupby(mcodes).cumcount().to_numpy()
    grid[mcodes, within] = np.arange(len(df))

    return PossDesign(
        cfg=cfg,
        match_id=mcodes.astype(np.int64),
        t=t,
        is_home=(df["side"] == "H").to_numpy(dtype=np.float64),
        team_idx=key_t.map(pos).to_numpy(dtype=np.int32),
        opp_idx=key_o.map(pos).to_numpy(dtype=np.int32),
        team_labels=team_labels,
        bin_idx=bin_idx,
        basis=basis,
        score_idx=np.array([s_pos[int(v)] for v in df["score_diff"]], dtype=np.int32),
        red_idx=np.array([r_pos[int(v)] for v in df["red_diff"]], dtype=np.int32),
        score_levels=score_levels,
        red_levels=red_levels,
        setpiece=df[f"sp_{danger}"].to_numpy(dtype=np.float64),
        y=df[f"t_{danger}"].notna().to_numpy(dtype=np.float64),
        grid=grid,
        include_score=cfg.include_score,
        include_red=cfg.include_red,
        include_setpiece=include_setpiece,
        labels={"match_ids": np.asarray(mid), "danger": danger},
    )


# ---------------------------------------------------------------- excitation


def excitation(design: PossDesign, y: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """S_self, S_cross for every row, by the decayed-sum recursion."""
    s_self = np.zeros(design.n)
    s_cross = np.zeros(design.n)
    _walk(design, tau, y=y, out=(s_self, s_cross))
    return s_self, s_cross


def excitation_bruteforce(design: PossDesign, y: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """O(n²) reference for the tests."""
    s_self = np.zeros(design.n)
    s_cross = np.zeros(design.n)
    for row in design.grid:
        idx = row[row >= 0]
        for k, p in enumerate(idx):
            for q in idx[:k]:
                if y[q]:
                    w = np.exp(-(design.t[p] - design.t[q]) / tau)
                    if design.is_home[q] == design.is_home[p]:
                        s_self[p] += w
                    else:
                        s_cross[p] += w
    return s_self, s_cross


def _walk(design, tau, *, y=None, out=None, eta_bg=None, rho=(0.0, 0.0), rng=None):
    """One pass down the padded grid. Either reads `y` and writes the sums into
    `out`, or draws y from `eta_bg + rho*S` (simulation) and returns it."""
    grid = design.grid
    n_m, L = grid.shape
    e_home = np.zeros(n_m)  # decayed count of home dangerous possessions so far
    e_away = np.zeros(n_m)
    t_prev = np.zeros(n_m)
    started = np.zeros(n_m, dtype=bool)
    draw = y is None
    y_out = np.zeros(design.n) if draw else None
    for k in range(L):
        rows = grid[:, k]
        live = rows >= 0
        r = rows[live]
        tk = design.t[r]
        decay = np.where(started[live], np.exp(-(tk - t_prev[live]) / tau), 1.0)
        e_home[live] *= decay
        e_away[live] *= decay
        home = design.is_home[r] > 0.5
        s_self = np.where(home, e_home[live], e_away[live])
        s_cross = np.where(home, e_away[live], e_home[live])
        if draw:
            p = expit(eta_bg[r] + rho[0] * s_self + rho[1] * s_cross)
            yk = (rng.random(r.size) < p).astype(np.float64)
            y_out[r] = yk
        else:
            out[0][r] = s_self
            out[1][r] = s_cross
            yk = y[r]
        e_home[live] += yk * home
        e_away[live] += yk * ~home
        t_prev[live] = tk
        started[live] = True
    return y_out


def simulate(
    design: PossDesign, eta_bg: np.ndarray, rho_self: float, rho_cross: float, tau: float, rng
) -> np.ndarray:
    """Draw D along the observed possession path, sequentially within each match."""
    return _walk(design, tau, eta_bg=eta_bg, rho=(rho_self, rho_cross), rng=rng)


# ---------------------------------------------------------------- likelihood


def _slices(design: PossDesign, with_exc: bool) -> dict:
    n_team = design.team_labels.size
    sizes = [
        ("theta0", 1),
        ("theta_home", 1),
        ("a", n_team),
        ("d", n_team),
        ("f", design.basis.shape[1]),
        ("g", design.score_levels.size - 1 if design.include_score else 0),
        ("h", design.red_levels.size - 1 if design.include_red else 0),
        ("c", 1 if design.include_setpiece else 0),
        ("rho", 2 if with_exc else 0),
    ]
    out, i = {}, 0
    for name, k in sizes:
        out[name] = slice(i, i + k)
        i += k
    out["_n"] = i
    return out


def _ref(levels: np.ndarray) -> int:
    return int(np.flatnonzero(levels == 0)[0])


def _full(vals: np.ndarray, levels: np.ndarray) -> np.ndarray:
    return np.insert(vals, _ref(levels), 0.0) if vals.size else np.zeros(levels.size)


def linear_predictor(p, design: PossDesign, s: dict, S=None) -> np.ndarray:
    eta = p[s["theta0"]][0] + p[s["theta_home"]][0] * design.is_home
    eta = eta + p[s["a"]][design.team_idx] + p[s["d"]][design.opp_idx]
    eta = eta + (design.basis @ p[s["f"]])[design.bin_idx]
    eta = eta + _full(p[s["g"]], design.score_levels)[design.score_idx]
    eta = eta + _full(p[s["h"]], design.red_levels)[design.red_idx]
    if design.include_setpiece:
        eta = eta + p[s["c"]][0] * design.setpiece
    if S is not None:
        eta = eta + p[s["rho"]][0] * S[0] + p[s["rho"]][1] * S[1]
    return eta


def objective(p, design: PossDesign, y: np.ndarray, S, s: dict):
    eta = linear_predictor(p, design, s, S)
    ridge = design.cfg.team_ridge
    a, d = p[s["a"]], p[s["d"]]
    # -loglik of Bernoulli(expit(eta)): log(1 + e^eta) - y·eta, computed stably
    nll = float(np.sum(np.logaddexp(0.0, eta) - y * eta)) + 0.5 * ridge * float(a @ a + d @ d)
    r = expit(eta) - y
    g = np.zeros_like(p)
    g[s["theta0"]] = r.sum()
    g[s["theta_home"]] = r @ design.is_home
    n_team = design.team_labels.size
    g[s["a"]] = np.bincount(design.team_idx, weights=r, minlength=n_team) + ridge * a
    g[s["d"]] = np.bincount(design.opp_idx, weights=r, minlength=n_team) + ridge * d
    g[s["f"]] = design.basis.T @ np.bincount(design.bin_idx, weights=r, minlength=design.basis.shape[0])
    if design.include_score:
        g[s["g"]] = np.delete(
            np.bincount(design.score_idx, weights=r, minlength=design.score_levels.size),
            _ref(design.score_levels),
        )
    if design.include_red:
        g[s["h"]] = np.delete(
            np.bincount(design.red_idx, weights=r, minlength=design.red_levels.size), _ref(design.red_levels)
        )
    if design.include_setpiece:
        g[s["c"]] = r @ design.setpiece
    if S is not None:
        g[s["rho"]] = [r @ S[0], r @ S[1]]
    return nll, g


@dataclass
class PossFit:
    params: np.ndarray
    rho_self: float
    rho_cross: float
    tau: float
    loglik: float  # Bernoulli log-likelihood, ridge excluded
    ridge_penalty: float
    converged: bool
    n_iter: int
    eta_background: np.ndarray  # linear predictor without the excitation terms
    terms: dict

    def as_dict(self) -> dict:
        return {
            "rho_self": self.rho_self,
            "odds_ratio_self": float(np.exp(self.rho_self)),
            "rho_cross": self.rho_cross,
            "odds_ratio_cross": float(np.exp(self.rho_cross)),
            "tau_minutes": self.tau,
            "loglik": self.loglik,
            "ridge_penalty": self.ridge_penalty,
            "converged": self.converged,
            "n_iter": self.n_iter,
            **self.terms,
        }


def fit(
    design: PossDesign,
    *,
    tau: float = 5.0,
    y: np.ndarray | None = None,
    excitation_terms: bool = True,
    maxiter: int = 3000,
) -> PossFit:
    """Penalised MLE by L-BFGS-B with the analytic gradient. `y` overrides the
    observed outcomes (simulation); `excitation_terms=False` fits the background only."""
    y = design.y if y is None else y
    s = _slices(design, excitation_terms)
    S = excitation(design, y, tau) if excitation_terms else None
    p0 = np.zeros(s["_n"])
    p0[s["theta0"]] = np.log(y.mean() / (1.0 - y.mean()))
    res = minimize(
        objective,
        p0,
        args=(design, y, S, s),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-6, "maxcor": 20},
    )
    p = res.x
    a, d = p[s["a"]], p[s["d"]]
    pen = 0.5 * design.cfg.team_ridge * float(a @ a + d @ d)
    rho = p[s["rho"]] if excitation_terms else np.zeros(2)
    terms = {
        "theta0": float(p[s["theta0"]][0]),
        "theta_home": float(p[s["theta_home"]][0]),
        "setpiece": float(p[s["c"]][0]) if design.include_setpiece else None,
        "g_score": dict(
            zip(map(str, design.score_levels), _full(p[s["g"]], design.score_levels).tolist(), strict=True)
        )
        if design.include_score
        else None,
        "h_red": dict(
            zip(map(str, design.red_levels), _full(p[s["h"]], design.red_levels).tolist(), strict=True)
        )
        if design.include_red
        else None,
        "n_possessions": int(design.n),
        "n_dangerous": int(y.sum()),
        "n_team_effects": int(design.team_labels.size),
    }
    return PossFit(
        params=p,
        rho_self=float(rho[0]),
        rho_cross=float(rho[1]),
        tau=float(tau),
        loglik=float(-res.fun + pen),
        ridge_penalty=pen,
        converged=bool(res.success),
        n_iter=int(res.nit),
        eta_background=linear_predictor(p, design, s, None),
        terms=terms,
    )
