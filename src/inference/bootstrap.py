"""Inference on eta.

No likelihood-ratio test: eta = 0 sits on the boundary of the parameter space, so
the chi-squared asymptotics do not apply and beta is unidentified under the null
(docs/math.md section 7). Two procedures instead:

- `parametric_bootstrap_null` for the p-value: simulate from the fitted Poisson
  null, **re-estimate mu inside every replicate**, refit the Hawkes model, and
  compare eta_hat against that distribution. Skipping the re-estimation makes the
  test anticonservative, which is the failure mode this whole design exists to
  avoid.
- `cluster_bootstrap` for the interval: resample whole matches with replacement.
  Matches are the independent unit here; events inside one are not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import CONFIG, Config
from src.inference.parallel import pmap
from src.inference.pipeline import Pipeline
from src.models import baseline

log = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    eta_obs: float
    draws: np.ndarray
    p_value: float
    n_replicates: int
    seed: int
    detail: pd.DataFrame

    def as_dict(self) -> dict:
        q = np.percentile(self.draws, [50, 90, 95, 99]) if self.draws.size else [np.nan] * 4
        return {
            "eta_self_observed": float(self.eta_obs),
            "p_value": float(self.p_value),
            "n_replicates": int(self.n_replicates),
            "seed": int(self.seed),
            "null_median": float(q[0]),
            "null_q90": float(q[1]),
            "null_q95": float(q[2]),
            "null_q99": float(q[3]),
        }


def parametric_bootstrap_null(
    pipeline: Pipeline,
    background,
    eta_obs: float,
    *,
    n_replicates: int = 500,
    seed: int = 5694,
    beta_for_sim: float | None = None,
    progress_every: int = 25,
    workers: int | None = None,
) -> BootstrapResult:
    """Simulate from mu_hat with eta = 0; refit everything; collect eta_hat."""
    rng = np.random.default_rng(seed)
    sim = pipeline.simulator(background)
    beta_sim = beta_for_sim if beta_for_sim is not None else 1.0 / pipeline.cfg.hawkes.tau_init
    # Seeds drawn up front, in order: the same draws whatever the worker count.
    rep_seeds = [int(rng.integers(0, 2**63 - 1)) for _ in range(n_replicates)]

    def one(arg):
        b, rep_seed = arg
        events = pipeline.simulate_events(sim, 0.0, 0.0, beta_sim, np.random.default_rng(rep_seed))
        res = pipeline.run(events)
        return {
            "replicate": b,
            "n_events": len(events),
            "eta_self": res.hawkes_fit.eta_self,
            "eta_cross": res.hawkes_fit.eta_cross,
            "beta": res.hawkes_fit.beta,
            "mu_converged": res.background.converged,
        }

    rows = pmap(
        one, list(enumerate(rep_seeds)), workers=workers, label="null bootstrap", every=progress_every
    )
    detail = pd.DataFrame(rows)
    draws = detail["eta_self"].to_numpy()
    p = (1.0 + float((draws >= eta_obs).sum())) / (n_replicates + 1.0)
    return BootstrapResult(
        eta_obs=float(eta_obs), draws=draws, p_value=p, n_replicates=n_replicates, seed=seed, detail=detail
    )


def cluster_bootstrap(
    slate,
    *,
    n_replicates: int = 200,
    seed: int = 5694,
    cfg: Config = CONFIG,
    progress_every: int = 25,
    workers: int | None = None,
) -> pd.DataFrame:
    """Resample matches with replacement and rerun the whole pipeline on each draw.

    Matches, not events, are the resampling block: chances inside one match are
    exactly the dependence the study is about, so an i.i.d. bootstrap over events
    would hide it and halve the error bars.
    """
    rng = np.random.default_rng(seed)
    match_ids = np.sort(slate.events["match_id"].unique())
    base = Pipeline(slate, cfg)  # built once; every replicate gathers rows from it
    draws = [rng.choice(match_ids, size=match_ids.size, replace=True) for _ in range(n_replicates)]

    def one(arg):
        b, draw = arg
        sub, pairs = _resample_slate(slate, draw, with_pairs=True)
        pipe = Pipeline(sub, cfg, design=baseline.gather_design(base.design, pairs))
        res = pipe.run(sub.events, warm_start=False)
        return {
            "replicate": b,
            "eta_self": res.hawkes_fit.eta_self,
            "eta_cross": res.hawkes_fit.eta_cross,
            "beta": res.hawkes_fit.beta,
        }

    rows = pmap(one, list(enumerate(draws)), workers=workers, label="cluster bootstrap", every=progress_every)
    return pd.DataFrame(rows)


def _resample_slate(slate, drawn_ids: np.ndarray, *, with_pairs: bool = False):
    """Duplicate whole matches, relabelling copies so each is its own process.

    Copies are numbered with a zero-padded suffix so that sorting the new ids as
    strings gives the same order as sorting (original id, copy number) -- which is
    what lets `gather_design` line up with the resampled event table.
    """
    from src.ingest.schema import Slate

    ev_by, go_by = dict(list(slate.events.groupby("match_id"))), dict(list(slate.goals.groupby("match_id")))
    ca_by = dict(list(slate.cards.groupby("match_id"))) if slate.cards is not None else {}
    ev, go, ca = [], [], []
    pairs = []
    for rep, mid in enumerate(drawn_ids):
        new_id = f"{mid}#b{rep:06d}"
        pairs.append((new_id, str(mid)))
        e = ev_by[mid].copy()
        e["match_id"] = new_id
        ev.append(e)
        if mid in go_by:
            g = go_by[mid].copy()
            g["match_id"] = new_id
            go.append(g)
        if mid in ca_by:
            c = ca_by[mid].copy()
            c["match_id"] = new_id
            ca.append(c)
    out = Slate(
        pd.concat(ev, ignore_index=True),
        pd.concat(go, ignore_index=True) if go else slate.goals.iloc[:0],
        pd.concat(ca, ignore_index=True) if ca else (None if slate.cards is None else slate.cards.iloc[:0]),
    )
    if not with_pairs:
        return out
    order = {mid: i for i, mid in enumerate(sorted(p[0] for p in pairs))}
    return out, sorted(pairs, key=lambda p: order[p[0]])


def percentile_ci(draws: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
