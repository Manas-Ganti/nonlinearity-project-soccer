"""Secondary analyses (docs/math.md section 9).

These are extensions, not the deliverable. They are here because the intensity
machinery already exists and each is a few lines on top of it -- and because
`eta_self` alone answers a narrower question than the one people argue about.

None of these run as part of the build order. Nothing in them may be used to
revise the background model: adding covariates to `mu` after looking at `eta_hat`
is fitting the null until the alternative disappears (CLAUDE.md, "Do not").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG, Config
from src.inference.pipeline import Pipeline
from src.models import hawkes


def quality_regression(data: hawkes.HawkesData, events: pd.DataFrame, fit) -> dict:
    """Do teams create *better* chances during a spell, or only more of them?

        E[x_i] = psi_0 + psi_1 * (excitation at t_i) + game-state controls

    `psi_1` tests the quality reading of momentum directly, and it is orthogonal to
    the arrival-time question: a team can have eta_self = 0 and still shoot from
    better positions while on top.
    """
    exc = hawkes.excitation_at_events(data, hawkes.branching_matrix(fit.eta_self, fit.eta_cross), fit.beta)
    ev = events.sort_values(["match_id", "t"], kind="stable").reset_index(drop=True)
    if len(ev) != exc.size:
        raise ValueError("event table and intensity trace are not aligned")

    y = ev["xg"].to_numpy(dtype=np.float64)
    cols = [np.ones(len(ev)), exc, (ev["side"] == "H").to_numpy(float)]
    names = ["intercept", "excitation", "home"]
    for level in sorted(ev["score_diff"].unique()):
        if level == 0:
            continue
        cols.append((ev["score_diff"] == level).to_numpy(float))
        names.append(f"score_{level:+d}")
    for level in sorted(ev["red_diff"].unique()):
        if level == 0:
            continue
        cols.append((ev["red_diff"] == level).to_numpy(float))
        names.append(f"red_{level:+d}")

    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    cov = np.linalg.pinv(X.T @ X) * float(resid @ resid) / dof
    se = np.sqrt(np.diag(cov))
    i = names.index("excitation")
    return {
        "coefficients": dict(zip(names, map(float, beta), strict=True)),
        "std_errors": dict(zip(names, map(float, se), strict=True)),
        "psi1": float(beta[i]),
        "psi1_se": float(se[i]),
        "psi1_t": float(beta[i] / se[i]) if se[i] > 0 else np.nan,
        "n": len(y),
        "mean_excitation": float(exc.mean()),
        "note": (
            "OLS with heteroskedasticity ignored and events inside a match treated as "
            "independent, so the standard error is optimistic. Use the cluster bootstrap "
            "before reporting psi1 as significant."
        ),
    }


def state_dependent_eta(slate, cfg: Config = CONFIG, *, min_events: int = 500) -> dict:
    """Does self-excitation differ when level, ahead or behind?

    The tactical reading predicts excitation is suppressed when leading (the block
    drops deeper); the psychological reading predicts it is elevated. They disagree,
    which makes this worth measuring.

    Implemented by splitting the slate on the score state that held for most of the
    match and refitting, rather than by making `eta` a function of state inside the
    likelihood. Splitting costs power and is honest about it; a state-dependent
    kernel would need the recursion to carry a time-varying `eta` and is a larger
    change than an extension warrants.
    """
    ev = slate.events
    dominant = (
        ev.groupby("match_id")["score_diff"]
        .agg(lambda s: int(np.sign(s.mean())))
        .rename("state")
        .reset_index()
    )
    out = {}
    for label, want in (("behind", -1), ("level", 0), ("ahead", 1)):
        ids = dominant.loc[dominant["state"] == want, "match_id"]
        sub = slate.filter_matches(ids)
        if len(sub.events) < min_events:
            out[label] = {"n_events": len(sub.events), "skipped": "too few events"}
            continue
        pipe = Pipeline(sub, cfg)
        res = pipe.run(sub.events, warm_start=False)
        out[label] = {
            "n_matches": int(sub.events["match_id"].nunique()),
            "n_events": len(sub.events),
            **res.hawkes_fit.as_dict(),
        }
    out["note"] = (
        "Split by the sign of each match's mean score differential from the shooting "
        "team's view, so a match contributes to exactly one arm. The arms are much "
        "smaller than the full slate, so each carries its own, much higher, detection "
        "floor -- which this function does not compute. Do not read a difference "
        "between arms as real without running the floor for each."
    )
    return out


def marked_excitation_profile(
    data: hawkes.HawkesData, events: pd.DataFrame, fit, gammas=(0.0, 0.5, 1.0)
) -> dict:
    """Does a big chance generate more momentum than a speculative one?

    The full version reweights the kernel by `w(x) = (x/xbar)^gamma` and refits.
    What is implemented here is the cheap diagnostic that says whether it is worth
    doing: the correlation between an event's own xG and the excitation it is
    sitting in. `gamma = 0` is the unmarked model, so the marked model nests it.
    """
    exc = hawkes.excitation_at_events(data, hawkes.branching_matrix(fit.eta_self, fit.eta_cross), fit.beta)
    ev = events.sort_values(["match_id", "t"], kind="stable").reset_index(drop=True)
    x = ev["xg"].to_numpy(dtype=np.float64)
    xbar = float(x.mean())
    # eta_hat = 0 makes the excitation trace identically zero, and a correlation with a
    # constant is undefined rather than zero. Say so instead of emitting a nan.
    corr = float(np.corrcoef(x, exc)[0, 1]) if exc.std() > 0 else None
    return {
        "mean_xg": xbar,
        "corr_xg_excitation": corr,
        "excitation_is_degenerate": corr is None,
        "weights_at_mean": {str(g): float((xbar / xbar) ** g) for g in gammas},
        "note": (
            "A diagnostic, not the marked fit. The marked kernel needs w(x) folded into "
            "the O(n) recursion, which is a change to models/hawkes.py, not a wrapper "
            "around it."
        ),
    }
