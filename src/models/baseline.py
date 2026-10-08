"""Inhomogeneous Poisson background mu_k(t) -- the null model, and the carrier of
every piece of domain knowledge in the project.

    log mu_k(t) = theta0 + theta_home*1[k=H] + theta_source + a_team(k) + d_opp(k)
                + f(minute) + g(score_diff) + h(red_diff)

`theta_source` is a per-provider intercept, present only on a pooled slate: two
providers do not count exactly the same things as shots, and that is a level
difference, not football. Team effects are keyed by (source, team, season).

Piecewise constant on one-minute bins, so the compensator is a finite sum and no
numerical integration appears anywhere (docs/math.md section 2).

The design is stored as index arrays rather than a sparse matrix: at ~4M rows an
explicit matrix costs half a gigabyte, while `np.bincount` over int32 indices
costs nothing and is faster.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import BIN_MINUTES, BackgroundConfig
from src.features import state as state_mod
from src.models.spline import natural_cubic_basis, natural_cubic_knots


@dataclass
class BackgroundDesign:
    """Row = (match, side, minute-bin). Rows are contiguous within a segment."""

    seg_match: np.ndarray  # (n_seg,) match_id strings
    seg_side: np.ndarray  # (n_seg,) 'H'/'A'
    seg_start: np.ndarray  # (n_seg,) first row index
    seg_nbins: np.ndarray  # (n_seg,) number of bins
    seg_T: np.ndarray  # (n_seg,) match_T

    is_home: np.ndarray  # (n_rows,) float64 0/1
    team_idx: np.ndarray  # (n_rows,) int32 into team_labels
    opp_idx: np.ndarray  # (n_rows,) int32 into team_labels
    bin_idx: np.ndarray  # (n_rows,) int32 minute bin
    score_idx: np.ndarray  # (n_rows,) int32 into score_levels
    red_idx: np.ndarray  # (n_rows,) int32 into red_levels
    exposure: np.ndarray  # (n_rows,) minutes

    team_labels: np.ndarray
    score_levels: np.ndarray
    red_levels: np.ndarray
    source_idx: np.ndarray  # (n_rows,) int32 into source_labels
    source_labels: np.ndarray
    basis: np.ndarray  # (n_bins_max, df)
    cfg: BackgroundConfig
    seg_lookup: dict
    # A covariate that never varies on this slate is not estimable. Understat carries no
    # dismissal times, so red_diff is identically 0 there and the h() terms would sit at
    # whatever the optimiser was initialised to while looking like estimates. Drop them
    # from the design and say so, rather than reporting zeros.
    dropped_terms: tuple[str, ...] = ()

    @property
    def n_rows(self) -> int:
        return self.exposure.size

    @property
    def n_seg(self) -> int:
        return self.seg_start.size

    def row_of(self, match_id: str, side: str, t: np.ndarray | float) -> np.ndarray:
        """Row index containing time `t` for one segment."""
        s = self.seg_lookup[(match_id, side)]
        b = np.floor(np.asarray(t, dtype=np.float64) / BIN_MINUTES).astype(np.int64)
        b = np.clip(b, 0, self.seg_nbins[s] - 1)
        return self.seg_start[s] + b

    def event_rows(self, events: pd.DataFrame) -> np.ndarray:
        """Vectorised row lookup for a whole event table."""
        seg = np.array(
            [self.seg_lookup[(m, s)] for m, s in zip(events["match_id"], events["side"], strict=True)],
            dtype=np.int64,
        )
        b = np.floor(events["t"].to_numpy(dtype=np.float64) / BIN_MINUTES).astype(np.int64)
        b = np.minimum(b, self.seg_nbins[seg] - 1)
        return self.seg_start[seg] + b


def build_design(slate, cfg: BackgroundConfig) -> BackgroundDesign:
    """One design per data slate. Team effects are indexed by (team, season)."""
    from src.ingest.schema import match_frame

    events = slate.events
    matches = match_frame(events).sort_values("match_id").reset_index(drop=True)
    goals = slate.goals
    cards = slate.cards

    src = events["source"].astype(str)
    team_season = pd.unique(
        pd.concat(
            [
                src + "|" + events["team"].astype(str) + "|" + events["season"].astype(str),
                src + "|" + events["opponent"].astype(str) + "|" + events["season"].astype(str),
            ]
        )
    )
    team_labels = np.sort(team_season)
    team_pos = {lab: i for i, lab in enumerate(team_labels)}
    source_labels = np.sort(pd.unique(src))
    source_pos = {lab: i for i, lab in enumerate(source_labels)}

    score_levels = np.array(cfg.score_bins, dtype=np.int64)
    red_levels = np.array(cfg.red_bins, dtype=np.int64)
    score_pos = {v: i for i, v in enumerate(score_levels)}
    red_pos = {v: i for i, v in enumerate(red_levels)}

    goals_by_match = dict(list(goals.groupby("match_id", sort=False))) if len(goals) else {}
    cards_by_match = (
        dict(list(cards.groupby("match_id", sort=False))) if cards is not None and len(cards) else {}
    )

    seg_match, seg_side, seg_start, seg_nbins, seg_T = [], [], [], [], []
    is_home, team_idx, opp_idx, bin_idx, score_idx, red_idx, exposure = [], [], [], [], [], [], []
    source_idx = []
    cursor = 0
    for row in matches.itertuples(index=False):
        T = float(row.match_T)
        n_bins = int(np.ceil(T / BIN_MINUTES))
        widths = np.full(n_bins, BIN_MINUTES, dtype=np.float64)
        widths[-1] = T - BIN_MINUTES * (n_bins - 1)
        g = goals_by_match.get(row.match_id)
        c = cards_by_match.get(row.match_id)
        for side, team, opp in (("H", row.home_team, row.away_team), ("A", row.away_team, row.home_team)):
            sc, rd = state_mod.state_trajectory(T, g, c, side, BIN_MINUTES)
            seg_match.append(row.match_id)
            seg_side.append(side)
            seg_start.append(cursor)
            seg_nbins.append(n_bins)
            seg_T.append(T)
            is_home.append(np.full(n_bins, 1.0 if side == "H" else 0.0))
            key_t = f"{row.source}|{team}|{row.season}"
            key_o = f"{row.source}|{opp}|{row.season}"
            team_idx.append(np.full(n_bins, team_pos[key_t], dtype=np.int32))
            opp_idx.append(np.full(n_bins, team_pos[key_o], dtype=np.int32))
            source_idx.append(np.full(n_bins, source_pos[str(row.source)], dtype=np.int32))
            bin_idx.append(np.arange(n_bins, dtype=np.int32))
            score_idx.append(np.array([score_pos[int(v)] for v in sc], dtype=np.int32))
            red_idx.append(np.array([red_pos[int(v)] for v in rd], dtype=np.int32))
            exposure.append(widths)
            cursor += n_bins

    n_bins_max = int(max(seg_nbins))
    centres = (np.arange(n_bins_max, dtype=np.float64) + 0.5) * BIN_MINUTES
    knots = natural_cubic_knots(cfg.minute_df, 0.0, centres[-1])
    basis = natural_cubic_basis(centres, knots)

    score_idx_arr = np.concatenate(score_idx)
    red_idx_arr = np.concatenate(red_idx)
    dropped = []
    if cfg.include_score and np.unique(score_idx_arr).size < 2:
        cfg = replace(cfg, include_score=False)
        dropped.append("g_score (score differential never varies on this slate)")
    if cfg.include_red and np.unique(red_idx_arr).size < 2:
        cfg = replace(cfg, include_red=False)
        dropped.append("h_red (no dismissal data: red_diff is identically 0)")

    seg_start_arr = np.asarray(seg_start, dtype=np.int64)
    seg_match_arr = np.asarray(seg_match, dtype=object)
    seg_side_arr = np.asarray(seg_side, dtype=object)
    lookup = {(m, s): i for i, (m, s) in enumerate(zip(seg_match_arr, seg_side_arr, strict=True))}

    return BackgroundDesign(
        seg_match=seg_match_arr,
        seg_side=seg_side_arr,
        seg_start=seg_start_arr,
        seg_nbins=np.asarray(seg_nbins, dtype=np.int64),
        seg_T=np.asarray(seg_T, dtype=np.float64),
        is_home=np.concatenate(is_home),
        team_idx=np.concatenate(team_idx),
        opp_idx=np.concatenate(opp_idx),
        bin_idx=np.concatenate(bin_idx),
        score_idx=score_idx_arr,
        red_idx=red_idx_arr,
        exposure=np.concatenate(exposure),
        team_labels=team_labels,
        score_levels=score_levels,
        red_levels=red_levels,
        source_idx=np.concatenate(source_idx),
        source_labels=source_labels,
        basis=basis,
        cfg=cfg,
        seg_lookup=lookup,
        dropped_terms=tuple(dropped),
    )


def counts(design: BackgroundDesign, events: pd.DataFrame) -> np.ndarray:
    y = np.zeros(design.n_rows, dtype=np.float64)
    if len(events):
        np.add.at(y, design.event_rows(events), 1.0)
    return y


# ------------------------------------------------------------------ the fit


@dataclass
class BackgroundFit:
    params: np.ndarray
    design: BackgroundDesign
    # The point-process log-likelihood of the data, *without* the team ridge, so it
    # is on the same scale as the Hawkes loglik it is compared with in gof.compare.
    # The ridge is what the optimiser minimised alongside it; it is reported apart.
    loglik: float
    converged: bool
    n_iter: int
    message: str
    ridge_penalty: float = 0.0

    def unpack(self) -> dict:
        return _unpack(self.params, self.design)

    def log_rate_rows(self) -> np.ndarray:
        return _log_rate(self.params, self.design)

    def rate_rows(self) -> np.ndarray:
        return np.exp(self.log_rate_rows())

    def integral_per_segment(self) -> np.ndarray:
        """int_0^T mu_k(s) ds for every (match, side)."""
        contrib = self.rate_rows() * self.design.exposure
        return _segment_sum(contrib, self.design)

    def cumulative_integral(self) -> np.ndarray:
        """Cumulative int_0^{bin edge} mu, per row, measured from the segment start."""
        contrib = self.rate_rows() * self.design.exposure
        c = np.cumsum(contrib)
        starts = self.design.seg_start
        base = np.repeat(np.concatenate([[0.0], c[starts[1:] - 1]]), self.design.seg_nbins)
        return c - base  # cumulative through the END of each row's bin

    def as_dict(self) -> dict:
        d = self.unpack()
        return {
            "theta0": float(d["theta0"]),
            "theta_home": float(d["theta_home"]),
            "f_minute": d["f"].tolist(),
            "g_score": (
                dict(zip(map(int, self.design.score_levels), map(float, d["g"]), strict=True))
                if self.design.cfg.include_score
                else None
            ),
            "h_red": (
                dict(zip(map(int, self.design.red_levels), map(float, d["h"]), strict=True))
                if self.design.cfg.include_red
                else None
            ),
            "theta_source": dict(zip(map(str, self.design.source_labels), map(float, d["s"]), strict=True)),
            "dropped_terms": list(self.design.dropped_terms),
            "n_team_effects": int(d["a"].size),
            "loglik": float(self.loglik),
            "ridge_penalty": float(self.ridge_penalty),
            "converged": bool(self.converged),
            "n_iter": int(self.n_iter),
            "team_ridge": float(self.design.cfg.team_ridge),
            "minute_df": int(self.design.cfg.minute_df),
            "include_score": bool(self.design.cfg.include_score),
            "include_red": bool(self.design.cfg.include_red),
        }


def _slices(design: BackgroundDesign) -> dict:
    n_team = design.team_labels.size
    n_f = design.basis.shape[1]
    n_g = design.score_levels.size - 1 if design.cfg.include_score else 0
    n_h = design.red_levels.size - 1 if design.cfg.include_red else 0
    n_s = design.source_labels.size - 1  # first provider is the reference
    i = 0
    out = {}
    for name, width in (
        ("theta0", 1),
        ("theta_home", 1),
        ("a", n_team),
        ("d", n_team),
        ("f", n_f),
        ("g", n_g),
        ("h", n_h),
        ("s", n_s),
    ):
        out[name] = slice(i, i + width)
        i += width
    out["_n"] = i
    return out


def n_params(design: BackgroundDesign) -> int:
    return _slices(design)["_n"]


def _ref_index(levels: np.ndarray) -> int:
    """Reference category: score/red differential 0."""
    return int(np.where(levels == 0)[0][0])


def _unpack(p: np.ndarray, design: BackgroundDesign) -> dict:
    s = _slices(design)
    g_free = p[s["g"]]
    h_free = p[s["h"]]
    g = np.zeros(design.score_levels.size)
    h = np.zeros(design.red_levels.size)
    if g_free.size:
        mask = np.ones(g.size, dtype=bool)
        mask[_ref_index(design.score_levels)] = False
        g[mask] = g_free
    if h_free.size:
        mask = np.ones(h.size, dtype=bool)
        mask[_ref_index(design.red_levels)] = False
        h[mask] = h_free
    src_eff = np.zeros(design.source_labels.size)
    src_eff[1:] = p[s["s"]]
    return {
        "theta0": p[s["theta0"]][0],
        "theta_home": p[s["theta_home"]][0],
        "a": p[s["a"]],
        "d": p[s["d"]],
        "f": p[s["f"]],
        "g": g,
        "h": h,
        "s": src_eff,
    }


def _log_rate(p: np.ndarray, design: BackgroundDesign) -> np.ndarray:
    u = _unpack(p, design)
    fvals = design.basis @ u["f"]
    eta = (
        u["theta0"]
        + u["theta_home"] * design.is_home
        + u["a"][design.team_idx]
        + u["d"][design.opp_idx]
        + fvals[design.bin_idx]
        + u["g"][design.score_idx]
        + u["h"][design.red_idx]
        + u["s"][design.source_idx]
    )
    return eta


def _objective(p: np.ndarray, design: BackgroundDesign, y: np.ndarray):
    s = _slices(design)
    u = _unpack(p, design)
    eta = _log_rate(p, design)
    rate = np.exp(eta)
    mean = rate * design.exposure
    ridge = design.cfg.team_ridge
    nll = float(mean.sum() - (y * eta).sum())
    nll += 0.5 * ridge * (float(u["a"] @ u["a"]) + float(u["d"] @ u["d"]))

    resid = mean - y  # d nll / d eta
    grad = np.zeros_like(p)
    grad[s["theta0"]] = resid.sum()
    grad[s["theta_home"]] = float(resid @ design.is_home)
    n_team = design.team_labels.size
    grad[s["a"]] = np.bincount(design.team_idx, weights=resid, minlength=n_team) + ridge * u["a"]
    grad[s["d"]] = np.bincount(design.opp_idx, weights=resid, minlength=n_team) + ridge * u["d"]
    bin_resid = np.bincount(design.bin_idx, weights=resid, minlength=design.basis.shape[0])
    grad[s["f"]] = design.basis.T @ bin_resid
    if s["g"].stop > s["g"].start:
        gr = np.bincount(design.score_idx, weights=resid, minlength=design.score_levels.size)
        grad[s["g"]] = np.delete(gr, _ref_index(design.score_levels))
    if s["h"].stop > s["h"].start:
        hr = np.bincount(design.red_idx, weights=resid, minlength=design.red_levels.size)
        grad[s["h"]] = np.delete(hr, _ref_index(design.red_levels))
    if s["s"].stop > s["s"].start:
        sr = np.bincount(design.source_idx, weights=resid, minlength=design.source_labels.size)
        grad[s["s"]] = sr[1:]
    return nll, grad


def fit(
    design: BackgroundDesign,
    y: np.ndarray,
    *,
    init: np.ndarray | None = None,
) -> BackgroundFit:
    cfg = design.cfg
    p0 = np.zeros(n_params(design)) if init is None else np.asarray(init, dtype=np.float64).copy()
    if init is None:
        total_rate = max(y.sum(), 1.0) / design.exposure.sum()
        p0[_slices(design)["theta0"]] = np.log(total_rate)
    res = minimize(
        _objective,
        p0,
        args=(design, y),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": cfg.maxiter, "ftol": cfg.tol, "gtol": 1e-7, "maxcor": 20},
    )
    pen = _ridge_penalty(res.x, design)
    return BackgroundFit(
        params=res.x,
        design=design,
        loglik=float(-res.fun + pen),
        converged=bool(res.success),
        n_iter=int(res.nit),
        message=str(res.message),
        ridge_penalty=pen,
    )


def _ridge_penalty(p: np.ndarray, design: BackgroundDesign) -> float:
    """0.5 * ridge * (|a|^2 + |d|^2): the part of the objective that is not likelihood."""
    u = _unpack(p, design)
    return 0.5 * design.cfg.team_ridge * (float(u["a"] @ u["a"]) + float(u["d"] @ u["d"]))


def _segment_sum(values: np.ndarray, design: BackgroundDesign) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(values)])
    return c[design.seg_start + design.seg_nbins] - c[design.seg_start]


def gather_design(design: BackgroundDesign, pairs: list[tuple[str, str]]) -> BackgroundDesign:
    """A design for a slate of *relabelled copies* of matches already in `design`.

    The cluster bootstrap resamples whole matches, so every block of rows it needs
    already exists: the covariates, the exposure and the game-state path of a
    duplicated match are identical to the original's. Rebuilding the design from
    scratch per replicate costs ~15 s on a 20,000-match slate and buys nothing, so
    gather the rows instead.

    `pairs` is [(new_match_id, original_match_id), ...] and must already be in the
    order the resampled event table sorts to, because `hawkes.prepare` requires the
    design's match order and the event order to agree.
    """
    take, seg_match, seg_side, seg_start, seg_nbins, seg_T = [], [], [], [], [], []
    cursor = 0
    for new_id, old_id in pairs:
        for side in ("H", "A"):
            seg = design.seg_lookup[(old_id, side)]
            n = int(design.seg_nbins[seg])
            start = int(design.seg_start[seg])
            take.append(np.arange(start, start + n, dtype=np.int64))
            seg_match.append(new_id)
            seg_side.append(side)
            seg_start.append(cursor)
            seg_nbins.append(n)
            seg_T.append(float(design.seg_T[seg]))
            cursor += n

    idx = np.concatenate(take) if take else np.zeros(0, dtype=np.int64)
    seg_match_arr = np.asarray(seg_match, dtype=object)
    seg_side_arr = np.asarray(seg_side, dtype=object)
    return BackgroundDesign(
        seg_match=seg_match_arr,
        seg_side=seg_side_arr,
        seg_start=np.asarray(seg_start, dtype=np.int64),
        seg_nbins=np.asarray(seg_nbins, dtype=np.int64),
        seg_T=np.asarray(seg_T, dtype=np.float64),
        is_home=design.is_home[idx],
        team_idx=design.team_idx[idx],
        opp_idx=design.opp_idx[idx],
        bin_idx=design.bin_idx[idx],
        score_idx=design.score_idx[idx],
        red_idx=design.red_idx[idx],
        exposure=design.exposure[idx],
        team_labels=design.team_labels,
        score_levels=design.score_levels,
        red_levels=design.red_levels,
        source_idx=design.source_idx[idx],
        source_labels=design.source_labels,
        basis=design.basis,
        cfg=design.cfg,
        seg_lookup={(m, s): i for i, (m, s) in enumerate(zip(seg_match_arr, seg_side_arr, strict=True))},
        dropped_terms=design.dropped_terms,
    )
