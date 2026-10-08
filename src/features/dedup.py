"""Collapse mechanical follow-ups (rebounds, corner chains) before fitting.

CLAUDE.md, constraint 1: this structure is removed in preprocessing, never
modelled. A fast kernel component is not identifiable at Understat's minute
resolution, so an optimiser given raw data will lock onto the mechanical
clustering and report a branching ratio that means nothing.

The threshold is calibrated on StatsBomb (second resolution) and then applied to
Understat, where -- because Understat records integer minutes and any sane
threshold is under a minute -- it reduces to "collapse same-team shots recorded in
the same minute".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DedupConfig


def within_team_gaps(events: pd.DataFrame, period_col: str | None = None) -> pd.DataFrame:
    """Consecutive shot-to-shot gaps, in seconds, within one team within one match.

    If `period_col` is given, gaps that span a period boundary (i.e. the half-time
    break) are dropped: they are not a football timescale, just a clock artefact.
    """
    cols = ["match_id", "team", "t"] + ([period_col] if period_col else [])
    df = events.loc[:, cols].sort_values(["match_id", "team", "t"], kind="stable")
    g = df.groupby(["match_id", "team"], sort=False)
    gap = (df["t"] - g["t"].shift(1)) * 60.0
    out = pd.DataFrame({"gap_s": gap.to_numpy()}, index=df.index)
    if period_col:
        same_period = df[period_col].eq(g[period_col].shift(1))
        out.loc[~same_period.to_numpy(), "gap_s"] = np.nan
    return out.dropna().reset_index(drop=True)


def hazard_profile(
    gaps_s: np.ndarray,
    *,
    bin_s: float = 2.0,
    max_s: float = 180.0,
    baseline_window_s: tuple[float, float] = (120.0, 900.0),
) -> pd.DataFrame:
    """Empirical short-gap density against an exponential baseline extrapolated
    from long gaps.

    Independent attacks arrive roughly as a renewal process; mechanical follow-ups
    are a spike on top of it at a few seconds. Fitting the baseline rate on gaps
    well beyond any mechanical timescale and extrapolating down gives the excess.
    """
    gaps_s = np.asarray(gaps_s, dtype=np.float64)
    lo, hi = baseline_window_s
    tail = gaps_s[(gaps_s >= lo) & (gaps_s < hi)]
    if tail.size < 50:
        raise ValueError("not enough long gaps to fit the baseline")
    # MLE for a left-truncated, right-censored exponential on [lo, hi).
    rate = 1.0 / (tail.mean() - lo)
    n = gaps_s.size
    surv_lo = np.exp(-rate * lo)
    # scale so the fitted density integrates to the observed mass beyond `lo`
    mass_beyond_lo = float((gaps_s >= lo).sum()) / n
    scale = mass_beyond_lo / surv_lo

    edges = np.arange(0.0, max_s + bin_s, bin_s)
    counts, _ = np.histogram(gaps_s, bins=edges)
    emp = counts / n / bin_s
    centres = edges[:-1] + bin_s / 2
    base = scale * rate * np.exp(-rate * centres)
    return pd.DataFrame(
        {
            "gap_s": centres,
            "n": counts,
            "empirical_density": emp,
            "baseline_density": base,
            "excess_ratio": np.where(base > 0, emp / base, np.nan),
        }
    )


def choose_threshold(
    gaps_s: np.ndarray,
    *,
    bin_s: float = 2.0,
    ratio_tol: float = 1.25,
    consecutive: int = 3,
    max_s: float = 180.0,
) -> dict:
    """Smallest gap beyond which the excess over the renewal baseline has died.

    The threshold is the left edge of the first run of `consecutive` bins whose
    empirical/baseline density ratio is at or below `ratio_tol`.
    """
    prof = hazard_profile(gaps_s, bin_s=bin_s, max_s=max_s)
    ratio = prof["excess_ratio"].to_numpy()
    ok = np.nan_to_num(ratio, nan=np.inf) <= ratio_tol
    run = 0
    threshold = float(max_s)
    for i, flag in enumerate(ok):
        run = run + 1 if flag else 0
        if run >= consecutive:
            threshold = float(prof["gap_s"].iloc[i - consecutive + 1] - bin_s / 2)
            break
    excess_mass = float(
        ((prof["empirical_density"] - prof["baseline_density"]).clip(lower=0) * bin_s)[
            prof["gap_s"] < threshold
        ].sum()
    )
    return {
        "threshold_seconds": threshold,
        "ratio_tol": ratio_tol,
        "bin_seconds": bin_s,
        "consecutive_bins": consecutive,
        "excess_mass_below_threshold": excess_mass,
        "n_gaps": int(np.asarray(gaps_s).size),
        "profile": prof,
    }


def _run_ids(df: pd.DataFrame, threshold_min: float, protect_penalties: bool) -> np.ndarray:
    """Label each maximal run of same-team shots separated by <= the threshold.

    Vectorised across the whole slate rather than looped per (match, team): the power
    study calls this once per replicate, and on a 20,000-match slate a Python loop over
    40,000 groups costs tens of seconds a call.

    `df` must already be sorted by (match_id, team, t).
    """
    t = df["t"].to_numpy(dtype=np.float64)
    is_pen = (df["situation"].to_numpy() == "penalty") if protect_penalties else np.zeros(len(df), dtype=bool)
    group = (df["match_id"].astype(str) + "|" + df["team"].astype(str)).to_numpy()

    new_run = np.ones(len(df), dtype=bool)
    if len(df) > 1:
        same_group = group[1:] == group[:-1]
        close = (t[1:] - t[:-1]) <= threshold_min
        # a penalty neither joins the run before it nor admits the shot after it
        no_penalty = ~is_pen[1:] & ~is_pen[:-1]
        new_run[1:] = ~(same_group & close & no_penalty)
    return np.cumsum(new_run) - 1


def apply_dedup(
    events: pd.DataFrame, cfg: DedupConfig, *, enforce_guard: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Collapse runs of same-team shots separated by <= the threshold.

    The surviving event keeps the *first* shot's time and game state; its mark is
    the probability the sequence produced a goal, 1 - prod(1 - xG_i), which is the
    only combination that stays a probability.

    `enforce_guard=False` is for simulated slates only. The 15% ceiling exists to
    catch a mis-calibrated threshold on real data; inside a power replicate with a
    large planted eta, the extra clustering genuinely produces more same-minute
    collisions, and refusing to proceed there would silently truncate the sweep at
    exactly the effect sizes it is meant to measure. The fraction is still reported.
    """
    from src.ingest import schema

    threshold_min = cfg.threshold_seconds / 60.0
    df = events.sort_values(["match_id", "team", "t"], kind="stable").reset_index(drop=True)
    df = df.assign(
        _run=_run_ids(df, threshold_min, cfg.protect_penalties),
        _log_no_goal=np.log1p(-df["xg"].to_numpy(dtype=np.float64).clip(0.0, 1 - 1e-12)),
    )

    agg = df.groupby("_run", sort=True).agg(
        match_id=("match_id", "first"),
        source=("source", "first"),
        league=("league", "first"),
        season=("season", "first"),
        date=("date", "first"),
        side=("side", "first"),
        team=("team", "first"),
        opponent=("opponent", "first"),
        t=("t", "first"),
        minute=("minute", "first"),
        second=("second", "first"),
        situation=("situation", "first"),
        is_goal=("is_goal", "max"),
        score_diff=("score_diff", "first"),
        red_diff=("red_diff", "first"),
        match_T=("match_T", "first"),
        red_cards_known=("red_cards_known", "first"),
        n_merged=("n_merged", "sum"),
        # the run's mark is P(at least one goal), the only combination that stays a
        # probability: 1 - prod(1 - xG_i), summed in log space so the aggregation is a
        # plain groupby-sum rather than a per-group Python callable
        _log_no_goal=("_log_no_goal", "sum"),
    )
    agg["xg"] = -np.expm1(agg.pop("_log_no_goal"))
    out = schema.validate(agg.reset_index(drop=True))

    n_before, n_after = len(events), len(out)
    removed = 1.0 - n_after / max(n_before, 1)
    stats = {
        "threshold_seconds": cfg.threshold_seconds,
        "n_before": n_before,
        "n_after": n_after,
        "removed_fraction": removed,
        "max_run_length": int(out["n_merged"].max()) if n_after else 0,
        "collapsed_events": int((out["n_merged"] > 1).sum()),
        # Understat compresses first-half and second-half stoppage into minutes 45/90,
        # so collapses there are the ones most likely to be a clock artefact.
        "collapses_at_half_boundaries": int(((out["n_merged"] > 1) & out["minute"].isin([45, 90])).sum()),
    }
    stats["guard_enforced"] = enforce_guard
    if enforce_guard and removed > cfg.max_removed_fraction:
        raise ValueError(
            f"de-duplication removed {removed:.1%} of shots (> {cfg.max_removed_fraction:.0%}); "
            "the threshold is eating real chances (CLAUDE.md build step 2)"
        )
    return out, stats
