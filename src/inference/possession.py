"""Gates 2-4 of docs/spec_possessions.md for Model A: recovery, floor, development fit.

The same discipline as the shot pipeline:

- Every simulated dataset follows the observed possession path and the observed
  game-state path. Only D (dangerous or not) is redrawn, sequentially, so that
  planted excitation feeds through the sums exactly as the model says it would.
- The background is re-estimated inside every replicate. A floor computed with the
  true background would overstate power.
- The null for the p-value and for the power study's critical value is the pure
  background fit (rho_self = rho_cross = 0). Nothing is planted in the cross term.

rho is a log-odds coefficient, so the grid is in log-odds too: rho = 0.05 is a 5%
lift in the odds of a dangerous possession per recent dangerous possession.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.inference.bootstrap import percentile_ci
from src.inference.parallel import pmap
from src.inference.power import detection_floor
from src.models import possession_logit as pl

log = logging.getLogger(__name__)

RECOVERY_TOLERANCE = 0.08
POWER_GRID = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
TAU_PROFILE = (2.0, 3.0, 5.0, 8.0, 14.0, 30.0)


def _replicate(design, bg_eta, rho_self, tau, seed):
    y = pl.simulate(design, bg_eta, rho_self, 0.0, tau, np.random.default_rng(seed))
    f = pl.fit(design, tau=tau, y=y)
    return {
        "rho_planted": rho_self,
        "seed": seed,
        "rho_hat_self": f.rho_self,
        "rho_hat_cross": f.rho_cross,
        "converged": f.converged,
        "n_iter": f.n_iter,
        "dangerous_frac": float(y.mean()),
    }


def recovery(design, *, tau=5.0, plant=0.30, n_replicates=10, seed=5694, workers=None) -> dict:
    """Gate 2. Plant rho_self = 0.30 (no cross term), refit everything, 10 times."""
    bg = pl.fit(design, tau=tau, excitation_terms=False)
    cells = [seed + r for r in range(n_replicates)]
    rows = pmap(
        lambda s: _replicate(design, bg.eta_background, plant, tau, s),
        cells,
        workers=workers,
        label="poss recovery",
        every=5,
    )
    detail = pd.DataFrame(rows)
    mean = float(detail["rho_hat_self"].mean())
    return {
        "rho_planted": plant,
        "tau_minutes": tau,
        "n_replicates": n_replicates,
        "mean_rho_hat_self": mean,
        "sd_rho_hat_self": float(detail["rho_hat_self"].std(ddof=1)),
        "mean_rho_hat_cross": float(detail["rho_hat_cross"].mean()),
        "bias": mean - plant,
        "n_unconverged": int((~detail["converged"]).sum()),
        "tolerance": RECOVERY_TOLERANCE,
        "passed": bool(abs(mean - plant) <= RECOVERY_TOLERANCE),
        "detail": detail,
    }


def power(
    design,
    *,
    tau=5.0,
    grid=POWER_GRID,
    replicates=30,
    null_replicates=200,
    alpha=0.05,
    target=0.80,
    seed=5694,
    workers=None,
) -> dict:
    """Gate 3. Critical value from the pure-null arm; power = P(rho_hat > crit)."""
    bg = pl.fit(design, tau=tau, excitation_terms=False)
    cells = []
    for rho in grid:
        n = null_replicates if rho == 0.0 else replicates
        cells += [(float(rho), seed + 1_000_003 * round(rho * 1000) + r) for r in range(n)]
    rows = pmap(
        lambda c: _replicate(design, bg.eta_background, c[0], tau, c[1]),
        cells,
        workers=workers,
        label="poss power",
        every=20,
    )
    sweep = pd.DataFrame(rows)
    null = sweep.loc[sweep["rho_planted"] == 0.0, "rho_hat_self"].to_numpy()
    crit = float(np.quantile(null, 1.0 - alpha))
    curve = []
    for rho, g in sweep.groupby("rho_planted", sort=True):
        hat = g["rho_hat_self"].to_numpy()
        curve.append(
            {
                "eta_star": float(rho),  # named for power.detection_floor
                "n_replicates": int(hat.size),
                "power": float((hat > crit).mean()),
                "mean_rho_hat": float(hat.mean()),
                "sd_rho_hat": float(hat.std(ddof=1)),
                "bias": float(hat.mean() - rho),
                "frac_converged": float(g["converged"].mean()),
            }
        )
    curve = pd.DataFrame(curve)
    floor = detection_floor(curve, target)
    return {
        "tau_minutes": tau,
        "critical_value": crit,
        "alpha": alpha,
        "target_power": target,
        "detection_floor": floor["floor"],
        "detection_floor_interpolated": floor["interpolated"],
        "curve": curve.rename(columns={"eta_star": "rho_planted"}),
        "config": {
            "grid": list(grid),
            "replicates": replicates,
            "null_replicates": null_replicates,
            "seed": seed,
        },
        "sweep": sweep,
    }


def _resample(table: pd.DataFrame, rng) -> pd.DataFrame:
    """Whole matches with replacement; a match drawn twice becomes two matches."""
    ids = table["match_id"].unique()
    drawn = rng.choice(ids, size=ids.size, replace=True)
    by = dict(tuple(table.groupby("match_id", sort=False)))
    return pd.concat([by[m].assign(match_id=f"{m}#{k}") for k, m in enumerate(drawn)], ignore_index=True)


def fit_dev(
    table: pd.DataFrame,
    *,
    tau=5.0,
    bootstrap=500,
    cluster_bootstrap=200,
    seed=5694,
    workers=None,
    sensitivity_tables: dict | None = None,
) -> dict:
    """Gate 4. Headline fit, parametric-bootstrap p-value, match-bootstrap CI, tau
    profile and the sensitivities fixed in the spec."""
    cfg = CONFIG.background
    design = pl.build_design(table, cfg)
    head = pl.fit(design, tau=tau)
    bg = pl.fit(design, tau=tau, excitation_terms=False)
    log.info("headline rho_self=%.5f rho_cross=%.5f", head.rho_self, head.rho_cross)

    out: dict = {"headline": head.as_dict(), "background_only": bg.as_dict()}
    out["lr_vs_background"] = 2.0 * (head.loglik - bg.loglik)

    out["tau_profile"] = [
        {
            "tau_minutes": t,
            **{
                k: v
                for k, v in pl.fit(design, tau=t).as_dict().items()
                if k in ("rho_self", "rho_cross", "loglik", "converged")
            },
        }
        for t in TAU_PROFILE
    ]

    sens = {}
    variants = {
        "box": dict(danger="box"),
        "final_third_x75": dict(danger="ft75"),
        "no_setpiece_term": dict(include_setpiece=False),
    }
    for name, kw in variants.items():
        sens[name] = pl.fit(pl.build_design(table, cfg, **kw), tau=tau).as_dict()
    from dataclasses import replace

    for name, bcfg in {
        "no_score_state": replace(cfg, include_score=False),
        "minute_spline_df_3": replace(cfg, minute_df=3),
        "minute_spline_df_9": replace(cfg, minute_df=9),
    }.items():
        sens[name] = pl.fit(pl.build_design(table, bcfg), tau=tau).as_dict()
    for name, tab in (sensitivity_tables or {}).items():
        sens[name] = pl.fit(pl.build_design(tab, cfg), tau=tau).as_dict()
    out["sensitivities"] = {
        k: {kk: v[kk] for kk in ("rho_self", "rho_cross", "odds_ratio_self", "converged", "n_possessions")}
        for k, v in sens.items()
    }

    if bootstrap:
        rows = pmap(
            lambda s: _replicate(design, bg.eta_background, 0.0, tau, s),
            [seed + 7_000_000 + b for b in range(bootstrap)],
            workers=workers,
            label="poss null bootstrap",
            every=50,
        )
        null = pd.DataFrame(rows)
        hat = null["rho_hat_self"].to_numpy()
        out["parametric_bootstrap"] = {
            "rho_self_observed": head.rho_self,
            "p_value_upper": float((1 + np.sum(hat >= head.rho_self)) / (hat.size + 1)),
            "p_value_lower": float((1 + np.sum(hat <= head.rho_self)) / (hat.size + 1)),
            "null_q05": float(np.quantile(hat, 0.05)),
            "null_q50": float(np.quantile(hat, 0.50)),
            "null_q95": float(np.quantile(hat, 0.95)),
            "n_replicates": int(hat.size),
            "n_unconverged": int((~null["converged"]).sum()),
            "seed": seed,
        }
        out["_null_detail"] = null

    if cluster_bootstrap:

        def one(b):
            rs = _resample(table, np.random.default_rng(seed + 9_000_000 + b))
            f = pl.fit(pl.build_design(rs, cfg), tau=tau)
            return {"b": b, "rho_self": f.rho_self, "rho_cross": f.rho_cross, "converged": f.converged}

        cb = pd.DataFrame(
            pmap(one, range(cluster_bootstrap), workers=workers, label="poss cluster bootstrap", every=25)
        )
        lo, hi = percentile_ci(cb["rho_self"].to_numpy())
        out["cluster_bootstrap"] = {
            "rho_self_ci95": [lo, hi],
            "odds_ratio_self_ci95": [float(np.exp(lo)), float(np.exp(hi))],
            "rho_cross_ci95": list(percentile_ci(cb["rho_cross"].to_numpy())),
            "n_replicates": len(cb),
            "n_unconverged": int((~cb["converged"]).sum()),
        }
        out["_cluster_detail"] = cb
    return out
