"""Goodness of fit by time rescaling.

Under the true conditional intensity the rescaled gaps

    tau_i = Lambda(t_i) - Lambda(t_{i-1})

are i.i.d. Exp(1). Applied to *both* models: if the Poisson null already fits,
there is no residual structure for excitation to explain, and that is the answer.
If neither fits, the comparison between them is weak evidence and must be
presented as such (CLAUDE.md, "Statistical discipline").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.models import hawkes


def rescaled_gaps(data: hawkes.HawkesData, eta_self: float, eta_cross: float, beta: float) -> pd.DataFrame:
    """One row per event: the rescaled gap since that team's previous chance."""
    eta = hawkes.branching_matrix(eta_self, eta_cross)
    comp = hawkes.compensator_at_events(data, eta, beta)
    df = pd.DataFrame(
        {"match": comp["match"], "team": comp["team"], "t": comp["t"], "Lambda": comp["Lambda"]}
    ).sort_values(["match", "team", "t"], kind="stable")
    prev = df.groupby(["match", "team"], sort=False)["Lambda"].shift(1).fillna(0.0)
    df["tau"] = df["Lambda"] - prev
    # A compensator that runs backwards is a bug, not a small number.
    if (df["tau"] < -1e-8).any():
        raise ValueError("negative rescaled gap: the compensator is not monotone")
    df["tau"] = df["tau"].clip(lower=0.0)
    return df.reset_index(drop=True)


# Note on the mean of tau. Under a correct model the *complete* rescaled gaps are
# Exp(1), but the final interval of every process -- from its last chance to the
# whistle -- is censored and is not a gap at all, so it is excluded. The pooled mean
# is therefore (n_events - n_processes * E[censored tail]) / n_events, a little under
# 1 by construction. A mean near 0.9 on ~15 events per process is right, not a bug;
# a mean near 1.0 would mean the censored tails were being counted by mistake.


def ks_report(tau: np.ndarray, *, max_points: int = 200_000, seed: int = 0) -> dict:
    """KS against Exp(1), via u = 1 - exp(-tau) ~ U(0,1).

    The KS *p-value* on half a million pooled gaps is not informative -- any model
    of football is wrong at that sample size and the test will say so. The KS
    statistic is the useful number: it is a distance, and it is comparable between
    the two models on the same data.
    """
    tau = np.asarray(tau, dtype=np.float64)
    tau = tau[np.isfinite(tau)]
    n = tau.size
    u = 1.0 - np.exp(-tau)
    stat_full = float(stats.kstest(u, "uniform").statistic)
    if n > max_points:
        rng = np.random.default_rng(seed)
        u_sub = rng.choice(u, size=max_points, replace=False)
    else:
        u_sub = u
    res = stats.kstest(u_sub, "uniform")
    return {
        "n": int(n),
        "ks_stat": stat_full,
        "ks_stat_subsample": float(res.statistic),
        "ks_pvalue_subsample": float(res.pvalue),
        "n_subsample": int(u_sub.size),
        "mean_tau": float(tau.mean()),
        "var_tau": float(tau.var()),
    }


def qq_points(tau: np.ndarray, n_points: int = 200) -> pd.DataFrame:
    """Theoretical vs empirical quantiles of the rescaled gaps, for the Q-Q plot."""
    tau = np.sort(np.asarray(tau, dtype=np.float64))
    probs = (np.arange(1, n_points + 1) - 0.5) / n_points
    return pd.DataFrame(
        {
            "prob": probs,
            "theoretical": stats.expon.ppf(probs),
            "empirical": np.quantile(tau, probs),
        }
    )


def per_match_ks(gaps: pd.DataFrame, min_events: int = 8) -> dict:
    """KS per team-process, then the distribution of those p-values.

    Pooling half a million gaps hides where a model fails. Per-process KS p-values
    should be uniform under a correct model; their own KS statistic against U(0,1)
    is a much better-calibrated summary at this sample size.
    """
    rows = []
    for _key, g in gaps.groupby(["match", "team"], sort=False):
        if len(g) < min_events:
            continue
        u = 1.0 - np.exp(-g["tau"].to_numpy())
        rows.append(stats.kstest(u, "uniform").pvalue)
    p = np.asarray(rows, dtype=np.float64)
    if p.size == 0:
        return {"n_processes": 0}
    return {
        "n_processes": int(p.size),
        "frac_reject_05": float((p < 0.05).mean()),
        "uniformity_ks_stat": float(stats.kstest(p, "uniform").statistic),
        "uniformity_ks_pvalue": float(stats.kstest(p, "uniform").pvalue),
    }


def compare(
    null_data: hawkes.HawkesData,
    null_fit,
    alt_data: hawkes.HawkesData,
    alt_fit,
) -> dict:
    """Time-rescaling report for the Poisson null and the Hawkes alternative.

    **Each model is rescaled by its own compensator.** The two arms need separate
    `HawkesData` objects because they carry different backgrounds: the null's mu is
    the Poisson MLE, while the alternative's mu was fitted jointly with the kernel
    and is smaller, because the excitation term is explaining some of the events.

    Taking the joint fit's mu and setting eta = 0 is not the null model -- it is a
    misspecified hybrid whose intensity is too low everywhere, and it makes the null
    look worse than it is. (Doing exactly that reported 8.3% of team-processes
    rejecting where the real null gives 5.5%.)
    """
    null_gaps = rescaled_gaps(null_data, 0.0, 0.0, alt_fit.beta)
    alt_gaps = rescaled_gaps(alt_data, alt_fit.eta_self, alt_fit.eta_cross, alt_fit.beta)
    return {
        "poisson": {**ks_report(null_gaps["tau"].to_numpy()), **per_match_ks(null_gaps)},
        "hawkes": {**ks_report(alt_gaps["tau"].to_numpy()), **per_match_ks(alt_gaps)},
        "loglik_poisson": float(null_fit.loglik) if null_fit is not None else None,
        "loglik_hawkes": float(alt_fit.loglik),
        "note": "each model is rescaled by its own fitted compensator",
    }
