"""Detection floor: the smallest branching ratio this design can actually find.

A branching ratio reported without this number is uninterpretable (CLAUDE.md), so
the sweep is a first-class deliverable, not a diagnostic.

For every planted eta* the *entire* pipeline runs on the simulated slate -- mu is
re-estimated from the simulated events, exactly as it would be on real data.
Using the true background here would overstate power badly, and measuring how much
of a real effect background estimation absorbs is the whole point.

**One documented deviation from docs/math.md section 8.** That sketch runs a full
parametric bootstrap inside every replicate: 7 eta values x 30 replicates x 500
bootstrap draws is ~105,000 pipeline runs, which is not affordable. Instead the
eta* = 0 arm is run at `null_replicates` (>= 200) and its (1 - alpha) quantile
becomes the critical value for every arm. This is the same Monte Carlo test with
the null calibrated once rather than 210 times; the level is unchanged, and the
only thing lost is per-replicate variation in the critical value.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import PowerConfig
from src.inference.parallel import pmap
from src.inference.pipeline import Pipeline

log = logging.getLogger(__name__)


def degrade_to_minute_clock(events: pd.DataFrame, cfg, seed: int) -> pd.DataFrame:
    """Put a simulated slate through Understat's clock.

    Without this, a floor computed for Understat is the floor for a hypothetical
    Understat that recorded seconds. The real feed rounds to integer minutes, which
    forces de-duplication to collapse every same-minute pair and then needs jitter to
    make the likelihood well defined -- and steps 2 and 7 both show that is where the
    measurable excitation goes. So the degradation belongs inside the replicate loop,
    not outside it.
    """
    from src.features import dedup, jitter
    from src.ingest import schema

    coarse = events.copy()
    coarse["t"] = np.floor(coarse["t"]).astype(float)
    coarse["minute"] = coarse["t"].astype("int16")
    coarse["second"] = np.nan
    collapsed, stats = dedup.apply_dedup(schema.validate(coarse), cfg.dedup, enforce_guard=False)
    out = jitter.jitter_events(collapsed, seed)
    out.attrs["dedup_removed_fraction"] = stats["removed_fraction"]
    return out


def run_sweep(
    pipeline: Pipeline,
    background,
    *,
    eta_cross: float,
    beta: float,
    cfg: PowerConfig,
    progress: bool = True,
    minute_clock: bool = False,
    workers: int | None = None,
) -> pd.DataFrame:
    """Plant each eta* in the grid, run the full pipeline per replicate, record eta_hat.

    `minute_clock=True` rounds each simulated slate to integer minutes and re-applies
    de-duplication and jitter before fitting, which is what the floor for an
    Understat-resolution dataset actually means.
    """
    sim = pipeline.simulator(background)
    cells = []
    for eta_star in cfg.eta_grid:
        n_rep = cfg.null_replicates if eta_star == 0.0 else cfg.replicates
        for r in range(n_rep):
            # Seeded off (eta*, replicate) so any single cell can be reproduced alone.
            cells.append((float(eta_star), r, cfg.seed + 1_000_003 * round(eta_star * 1000) + r))

    def one(cell):
        eta_star, r, seed = cell
        rng = np.random.default_rng(seed)
        events = pipeline.simulate_events(sim, eta_star, eta_cross, beta, rng)
        n_simulated = len(events)
        removed = None
        if minute_clock:
            events = degrade_to_minute_clock(events, pipeline.cfg, seed)
            removed = events.attrs.get("dedup_removed_fraction")
        res = pipeline.run(events)
        return {
            "eta_star": eta_star,
            "n_simulated": n_simulated,
            "minute_clock": minute_clock,
            "dedup_removed_fraction": removed,
            "replicate": r,
            "seed": seed,
            "n_events": len(events),
            "eta_hat_self": res.hawkes_fit.eta_self,
            "eta_hat_cross": res.hawkes_fit.eta_cross,
            "beta_hat": res.hawkes_fit.beta,
            "tau_hat": res.hawkes_fit.tau_minutes,
            # With the joint fit these are one optimiser's flag; kept as two columns
            # because the two-stage estimator sets them separately.
            "mu_converged": res.background.converged,
            "hawkes_converged": res.hawkes_fit.converged,
            "n_iter": res.hawkes_fit.n_iter,
        }

    rows = pmap(one, cells, workers=workers, label="power sweep" if progress else "", every=10)
    return pd.DataFrame(rows)


def summarise(sweep: pd.DataFrame, cfg: PowerConfig) -> dict:
    null = sweep.loc[sweep["eta_star"] == 0.0, "eta_hat_self"].to_numpy()
    if null.size == 0:
        raise ValueError("the sweep has no eta*=0 arm, so there is no critical value")
    crit = float(np.quantile(null, 1.0 - cfg.alpha))

    rows = []
    for eta_star, g in sweep.groupby("eta_star", sort=True):
        hat = g["eta_hat_self"].to_numpy()
        rows.append(
            {
                "eta_star": float(eta_star),
                "n_replicates": int(hat.size),
                "power": float((hat > crit).mean()),
                "mean_eta_hat": float(hat.mean()),
                "median_eta_hat": float(np.median(hat)),
                "sd_eta_hat": float(hat.std(ddof=1)) if hat.size > 1 else np.nan,
                "bias": float(hat.mean() - eta_star),
                "mean_n_events": float(g["n_events"].mean()),
                "frac_converged": float(g["hawkes_converged"].astype(bool).mean())
                if "hawkes_converged" in g
                else None,
            }
        )
    curve = pd.DataFrame(rows)
    floor = detection_floor(curve, cfg.target_power)
    return {
        "critical_value": crit,
        "alpha": cfg.alpha,
        "target_power": cfg.target_power,
        "detection_floor": floor["floor"],
        "detection_floor_interpolated": floor["interpolated"],
        "curve": curve,
    }


def detection_floor(curve: pd.DataFrame, target_power: float) -> dict:
    """Smallest eta* on the grid reaching the target power, plus a linear read-off
    between the two bracketing grid points."""
    c = curve.sort_values("eta_star").reset_index(drop=True)
    hit = c[c["power"] >= target_power]
    if hit.empty:
        return {"floor": None, "interpolated": None}
    i = int(hit.index[0])
    floor = float(c.loc[i, "eta_star"])
    if i == 0:
        return {"floor": floor, "interpolated": floor}
    x0, y0 = c.loc[i - 1, "eta_star"], c.loc[i - 1, "power"]
    x1, y1 = c.loc[i, "eta_star"], c.loc[i, "power"]
    interp = floor if y1 == y0 else float(x0 + (target_power - y0) * (x1 - x0) / (y1 - y0))
    return {"floor": floor, "interpolated": interp}


def recovery_check(
    pipeline: Pipeline,
    background,
    *,
    eta_plant: float = 0.30,
    eta_cross: float,
    beta: float,
    n_replicates: int = 10,
    seed: int = 5694,
    tolerance: float = 0.08,
    workers: int | None = None,
) -> dict:
    """Build step 5. Plant a large, unmissable eta and confirm the pipeline finds it.

    If this fails, nothing downstream means anything and the run must stop.
    """
    sim = pipeline.simulator(background)

    def one(r):
        rng = np.random.default_rng(seed + r)
        events = pipeline.simulate_events(sim, eta_plant, eta_cross, beta, rng)
        res = pipeline.run(events)
        return {
            "replicate": r,
            "n_events": len(events),
            "eta_hat_self": res.hawkes_fit.eta_self,
            "eta_hat_cross": res.hawkes_fit.eta_cross,
            "beta_hat": res.hawkes_fit.beta,
            "tau_hat": res.hawkes_fit.tau_minutes,
            "fit_converged": res.hawkes_fit.converged,
            "n_iter": res.hawkes_fit.n_iter,
        }

    detail = pd.DataFrame(pmap(one, range(n_replicates), workers=workers, label="recovery", every=5))
    mean_hat = float(detail["eta_hat_self"].mean())
    return {
        "eta_planted": eta_plant,
        "eta_cross_planted": eta_cross,
        "beta_planted": beta,
        "tau_planted_minutes": 1.0 / beta,
        "n_replicates": n_replicates,
        "mean_eta_hat_self": mean_hat,
        "sd_eta_hat_self": float(detail["eta_hat_self"].std(ddof=1)) if n_replicates > 1 else np.nan,
        "bias": mean_hat - eta_plant,
        "mean_tau_hat": float(detail["tau_hat"].mean()),
        "n_unconverged": int((~detail["fit_converged"].astype(bool)).sum()),
        "tolerance": tolerance,
        "passed": bool(abs(mean_hat - eta_plant) <= tolerance),
        "detail": detail,
    }
