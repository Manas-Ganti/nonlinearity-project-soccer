"""The full estimation pipeline, as one callable.

The power study and the bootstrap both need to run *the entire thing* per
replicate -- re-estimating mu is the point, not an optimisation to skip
(docs/math.md sections 7 and 8) -- so it lives in one place and every caller uses
the same one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import CONFIG, Config
from src.models import baseline, hawkes, joint, simulate


@dataclass
class PipelineResult:
    background: baseline.BackgroundFit
    hawkes_fit: hawkes.HawkesFit
    data: hawkes.HawkesData
    em_trace: list | None = None

    def as_dict(self) -> dict:
        return {
            "background": self.background.as_dict(),
            "hawkes": self.hawkes_fit.as_dict(),
            "em_trace": self.em_trace,
        }


class Pipeline:
    """Bound to one design (one slate of matches); reusable across replicates.

    The design encodes the match calendar, the team/season index and the observed
    game-state paths. Those are held fixed across simulation replicates, so the
    design is built once and only the event counts change.
    """

    def __init__(self, slate, cfg: Config = CONFIG, *, design=None):
        self.cfg = cfg
        self.slate = slate
        # `design` lets a caller hand in a design gathered from another one, which is
        # what the cluster bootstrap does: a resampled slate is duplicated matches, so
        # every row it needs already exists.
        self.design = design if design is not None else baseline.build_design(slate, cfg.background)
        self.template = _template(slate)
        self._mu_warm: np.ndarray | None = None

    def fit_background(self, events: pd.DataFrame, *, warm_start: bool = True) -> baseline.BackgroundFit:
        y = baseline.counts(self.design, _modelled(events, self.cfg))
        init = self._mu_warm if warm_start else None
        fit = baseline.fit(self.design, y, init=init)
        if warm_start:
            self._mu_warm = fit.params.copy()
        return fit

    def run(
        self,
        events: pd.DataFrame,
        *,
        warm_start: bool = True,
        method: str = "joint",
        n_em: int = 10,
        tol: float = 1e-3,
    ) -> PipelineResult:
        """Fit the background and the kernel to one event set.

        Three estimators, all reachable, because the difference between them is
        itself a result (see results/estimator_comparison.json):

        - ``joint`` (default): maximum likelihood over mu, eta and beta together.
        - ``em``: alternate between splitting events background/offspring and
          refitting mu on the fractional counts. Same fixed point, far slower.
        - ``two_stage``: fit mu under the Poisson null and freeze it. This is what
          docs/math.md sketches and it is **badly biased downwards** -- a Poisson mu
          fitted to data containing excitation absorbs the offspring, leaving nothing
          for eta. The recovery test measures it: eta = 0.30 comes back as 0.08.
        """
        slate = self.slate.with_events(events)
        bg = self.fit_background(events, warm_start=warm_start)
        data = hawkes.prepare(slate, bg, self.cfg.hawkes)

        if method == "joint":
            # Deliberately *not* warm-started from the two-stage fit. That fit collapses
            # onto the short-tau boundary, and starting the joint optimiser there leaves
            # it in the wrong basin: eta_self came back 0.12 instead of 0.30. Starting
            # from the configured defaults recovers 0.30 from any of tau0 in {2, 5, 10}.
            jf = joint.fit(self.design, data, self.cfg.hawkes, theta_init=bg.params, z_init=None)
            self._mu_warm = jf.background.params.copy()
            # `update_mu` leaves the cached compensator stale; rebuild it for the GOF pass
            final = hawkes.prepare(slate, jf.background, self.cfg.hawkes)
            return PipelineResult(
                background=jf.background,
                hawkes_fit=jf.hawkes_fit,
                data=final,
                em_trace=[
                    {
                        "iter": jf.n_iter,
                        "loglik": jf.hawkes_fit.loglik,
                        "eta_self": jf.hawkes_fit.eta_self,
                        "eta_cross": jf.hawkes_fit.eta_cross,
                        "tau": jf.hawkes_fit.tau_minutes,
                    }
                ],
            )

        hk = hawkes.fit(data, self.cfg.hawkes)
        if method == "two_stage":
            return PipelineResult(background=bg, hawkes_fit=hk, data=data, em_trace=None)
        if method != "em":
            raise ValueError(f"unknown fit method {method!r}")

        trace = [
            {
                "iter": 0,
                "loglik": hk.loglik,
                "eta_self": hk.eta_self,
                "eta_cross": hk.eta_cross,
                "tau": hk.tau_minutes,
            }
        ]
        z = hawkes.natural_to_unconstrained(hk.eta_self, hk.eta_cross, hk.beta, self.cfg.hawkes)
        for it in range(1, n_em + 1):
            eta = hawkes.branching_matrix(hk.eta_self, hk.eta_cross)
            lam = hawkes.intensity_at_events(data, eta, hk.beta)
            mu_ev = np.exp(data.log_mu[data.mask])
            p_bg = np.clip(mu_ev / np.maximum(lam, 1e-300), 0.0, 1.0)

            y = np.zeros(self.design.n_rows)
            np.add.at(y, data.design_rows, p_bg)
            bg = baseline.fit(self.design, y, init=bg.params)
            self._mu_warm = bg.params.copy()

            data = hawkes.prepare(slate, bg, self.cfg.hawkes)
            hk = hawkes.fit(data, self.cfg.hawkes, init=z)
            z = hawkes.natural_to_unconstrained(hk.eta_self, hk.eta_cross, hk.beta, self.cfg.hawkes)
            trace.append(
                {
                    "iter": it,
                    "loglik": hk.loglik,
                    "eta_self": hk.eta_self,
                    "eta_cross": hk.eta_cross,
                    "tau": hk.tau_minutes,
                }
            )
            if abs(trace[-1]["loglik"] - trace[-2]["loglik"]) < tol:
                break

        hk.method = f"EM ({len(trace) - 1} iterations, mu re-estimated with fractional counts)"
        return PipelineResult(background=bg, hawkes_fit=hk, data=data, em_trace=trace)

    # -------------------------------------------------------------- simulation

    def simulator(
        self, background: baseline.BackgroundFit, marks: np.ndarray | None = None
    ) -> simulate.SlateSimulator:
        if marks is None:
            ev = _modelled(self.slate.events, self.cfg)
            marks = ev["xg"].to_numpy(dtype=np.float64)
        return simulate.SlateSimulator(self.design, background.rate_rows(), marks=marks)

    def simulate_events(
        self,
        sim: simulate.SlateSimulator,
        eta_self: float,
        eta_cross: float,
        beta: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        raw = sim.simulate(eta_self, eta_cross, beta, rng)
        return sim.to_events(raw, self.template, rng)


def _template(slate) -> pd.DataFrame:
    from src.ingest.schema import match_frame

    return match_frame(slate.events)


def _modelled(events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The event set the models actually see.

    Penalties arrive from a foul, have no build-up and excite nothing, so they are
    dropped entirely -- from the background too, which would otherwise absorb them
    and inflate the rate they are supposed to be excluded from (CLAUDE.md,
    constraint 2).
    """
    if cfg.hawkes.exclude_penalties:
        return events[events["situation"] != "penalty"]
    return events
