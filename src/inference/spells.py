"""Models S and G: pipeline, recovery gate, detection floors, development fit, Holm
(docs/spec_spells.md section 6). Exploratory throughout (section 1).

Every replicate runs the **whole** pipeline: the Poisson background is refitted
from the simulated shots, then the joint fit of background, gamma and kernel, exactly
as on real data (CLAUDE.md, "Re-estimate mu inside every simulation").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.config import CONFIG, Config
from src.features.spells import SpellState
from src.inference.bootstrap import _resample_slate
from src.inference.parallel import pmap
from src.inference.pipeline import _modelled, _template
from src.models import baseline, hawkes, split_hawkes
from src.models.split_simulate import SplitSimulator, score_class

log = logging.getLogger(__name__)

LABELS = {"S": ("out", "in"), "G": ("level", "trail", "lead")}
RECOVERY_PLANT = {"S": (0.10, 0.30), "G": (0.30, 0.10, 0.20)}  # in LABELS order
POWER_GRID = (0.0, 0.02, 0.05, 0.10, 0.20)
ALPHA_FLOOR = 0.01  # 0.05 / 5 primary tests (section 6.4)
TAU_PROFILE = (3.0, 5.0, 8.0, 14.0, 20.0, 30.0)


@dataclass
class SpellResult:
    fit: split_hawkes.SplitFit
    labels: tuple[str, ...]

    def as_dict(self) -> dict:
        out = self.fit.as_dict(self.labels)
        if len(self.labels) == 2:
            out["contrast_in_minus_out"] = out["eta_in"] - out["eta_out"]
        return out


class SpellPipeline:
    """Bound to one slate (one design) and, for Model S, one spell path."""

    def __init__(
        self,
        slate,
        model: str,
        *,
        spell: SpellState | None = None,
        gamma: bool = True,
        cfg: Config = CONFIG,
        design=None,
    ):
        if model not in LABELS:
            raise ValueError(f"model must be S or G, got {model!r}")
        if model == "S" and spell is None:
            raise ValueError("Model S needs a spell path")
        self.model = model
        self.labels = LABELS[model]
        self.n_class = len(self.labels)
        self.cfg = cfg
        self.slate = slate
        self.spell = spell
        self.use_gamma = bool(gamma and model == "S")
        self.design = design if design is not None else baseline.build_design(slate, cfg.background)
        self.w = spell.row_weights(self.design) if self.use_gamma else None
        self.template = _template(slate)
        self._theta_warm: np.ndarray | None = None

    # --------------------------------------------------------- annotation

    def _sorted(self, events: pd.DataFrame) -> pd.DataFrame:
        ev = _modelled(events, self.cfg)
        return ev.sort_values(["match_id", "t"], kind="stable").reset_index(drop=True)

    def classes(self, ev: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(class, z_own, z_opp) per event, in sorted order."""
        if self.spell is not None and len(ev):
            zo, zp = self.spell.at(ev["match_id"].to_numpy(), ev["side"].to_numpy(), ev["t"].to_numpy())
        else:
            zo = zp = np.zeros(len(ev))
        cls = zo.astype(np.int64) if self.model == "S" else score_class(ev["score_diff"].to_numpy())
        return cls, zo, zp

    # ------------------------------------------------------------ fitting

    def _prepare(self, events: pd.DataFrame):
        ev = self._sorted(events)
        y = baseline.counts(self.design, ev)
        bg = baseline.fit(self.design, y, init=self._theta_warm)
        base = hawkes.prepare(self.slate.with_events(ev), bg, self.cfg.hawkes)
        cls, zo, zp = self.classes(ev)
        sd = split_hawkes.make_data(
            base,
            cls,
            self.n_class,
            z_own=zo if self.use_gamma else None,
            z_opp=zp if self.use_gamma else None,
            w=self.w,
        )
        return bg, sd

    def run(self, events: pd.DataFrame, *, kernel: bool = True, hawkes_cfg=None) -> SpellResult:
        bg, sd = self._prepare(events)
        fit = split_hawkes.fit(
            self.design, sd, hawkes_cfg or self.cfg.hawkes, theta_init=bg.params, kernel=kernel
        )
        self._theta_warm = fit.theta.copy()
        return SpellResult(fit=fit, labels=self.labels)

    # ---------------------------------------------------------- simulation

    def simulator(self, null_fit: split_hawkes.SplitFit) -> SplitSimulator:
        marks = _modelled(self.slate.events, self.cfg)["xg"].to_numpy(dtype=np.float64)
        return SplitSimulator(
            self.design,
            split_hawkes.rate_rows(null_fit, self.design),
            mode="spell" if self.model == "S" else "score",
            spell=self.spell if (self.model == "S" or self.use_gamma) else None,
            gamma=(null_fit.gamma_own, null_fit.gamma_opp),
            marks=marks,
        )

    def simulate_events(self, sim: SplitSimulator, eta_own, eta_cross, beta, rng) -> pd.DataFrame:
        raw = sim.simulate_split(eta_own, eta_cross, beta, rng)
        return sim.to_events(raw, self.template, rng)


# --------------------------------------------------------------- the gates


def _row(res: SpellResult, **extra) -> dict:
    d = res.as_dict()
    return {
        **extra,
        **{
            k: d[k]
            for k in d
            if k.startswith("eta_")
            or k in ("tau_minutes", "gamma_own", "gamma_opp", "converged", "n_iter", "n_events")
        },
    }


def recovery(
    pipe: SpellPipeline,
    null_fit,
    *,
    tau: float = 5.0,
    eta_cross: float = 0.05,
    n_replicates: int = 10,
    seed: int = 5694,
    tolerance: float = 0.08,
    workers=None,
) -> dict:
    """Gate (section 6.2): plant the spec's values, full pipeline, +-0.08 on each."""
    plant = np.array(RECOVERY_PLANT[pipe.model])
    sim = pipe.simulator(null_fit)

    def one(r):
        rng = np.random.default_rng(seed + 21_000_000 + r)
        ev = pipe.simulate_events(sim, plant, eta_cross, 1.0 / tau, rng)
        return _row(pipe.run(ev), replicate=r)

    detail = pd.DataFrame(pmap(one, range(n_replicates), workers=workers, label="spell recovery", every=5))
    per = {}
    for lab, eta in zip(pipe.labels, plant, strict=True):
        m = float(detail[f"eta_{lab}"].mean())
        per[lab] = {
            "planted": float(eta),
            "mean_hat": m,
            "sd_hat": float(detail[f"eta_{lab}"].std(ddof=1)),
            "bias": m - float(eta),
            "passed": bool(abs(m - eta) <= tolerance),
        }
    return {
        "model": pipe.model,
        "planted": dict(zip(pipe.labels, map(float, plant), strict=True)),
        "eta_cross_planted": eta_cross,
        "tau_planted": tau,
        "gamma_planted": [null_fit.gamma_own, null_fit.gamma_opp],
        "n_replicates": n_replicates,
        "tolerance": tolerance,
        "per_parameter": per,
        "mean_eta_cross_hat": float(detail["eta_cross"].mean()),
        "mean_tau_hat": float(detail["tau_minutes"].mean()),
        "n_unconverged": int((~detail["converged"].astype(bool)).sum()),
        "passed": bool(all(v["passed"] for v in per.values())),
        "detail": detail,
    }


def power(
    pipe: SpellPipeline,
    null_fit,
    *,
    tau: float = 5.0,
    eta_cross: float = 0.05,
    grid=POWER_GRID,
    replicates: int = 30,
    null_replicates: int = 200,
    alpha: float = ALPHA_FLOOR,
    target_power: float = 0.8,
    seed: int = 5694,
    workers=None,
) -> dict:
    """Floor per parameter (section 6.5): plant one eta, others 0; one shared pure-null arm."""
    sim = pipe.simulator(null_fit)
    C = pipe.n_class
    cells = [("null", -1, 0.0, r) for r in range(null_replicates)]
    for c, lab in enumerate(pipe.labels):
        for eta in grid:
            if eta == 0.0:
                continue
            cells += [(lab, c, float(eta), r) for r in range(replicates)]

    def one(cell):
        lab, c, eta, r = cell
        plant = np.zeros(C)
        if c >= 0:
            plant[c] = eta
        s = seed + 22_000_000 + 100_000 * (c + 1) + 1000 * round(eta * 100) + r
        ev = pipe.simulate_events(sim, plant, eta_cross, 1.0 / tau, np.random.default_rng(s))
        return _row(pipe.run(ev), target=lab, eta_planted=eta, replicate=r, seed=s)

    sweep = pd.DataFrame(pmap(one, cells, workers=workers, label="spell power", every=20))
    null = sweep[sweep["target"] == "null"]
    floors, curves = {}, {}
    for lab in pipe.labels:
        col = f"eta_{lab}"
        crit = float(np.quantile(null[col], 1.0 - alpha))
        rows = [
            {
                "eta_planted": 0.0,
                "power": float((null[col] > crit).mean()),
                "mean_hat": float(null[col].mean()),
            }
        ]
        for eta, g in sweep[sweep["target"] == lab].groupby("eta_planted"):
            rows.append(
                {
                    "eta_planted": float(eta),
                    "power": float((g[col] > crit).mean()),
                    "mean_hat": float(g[col].mean()),
                    "bias": float(g[col].mean() - eta),
                    "frac_converged": float(g["converged"].mean()),
                }
            )
        curve = pd.DataFrame(rows).sort_values("eta_planted")
        hit = curve[curve["power"] >= target_power]
        floors[lab] = {
            "critical_value": crit,
            "detection_floor": float(hit["eta_planted"].iloc[0]) if len(hit) else None,
        }
        curves[lab] = curve.to_dict("records")
    return {
        "model": pipe.model,
        "alpha": alpha,
        "target_power": target_power,
        "grid": list(grid),
        "replicates": replicates,
        "null_replicates": null_replicates,
        "eta_cross_planted": eta_cross,
        "tau_planted": tau,
        "floors": floors,
        "curves": curves,
        "n_unconverged": int((~sweep["converged"].astype(bool)).sum()),
        "sweep": sweep,
    }


def parametric_bootstrap(
    pipe: SpellPipeline, null_fit, observed: dict, *, n_replicates=500, seed=5694, workers=None
) -> dict:
    """p-value per eta under the fitted Poisson null with gamma (section 6.6)."""
    sim = pipe.simulator(null_fit)
    zeros = np.zeros(pipe.n_class)

    def one(b):
        rng = np.random.default_rng(seed + 23_000_000 + b)
        ev = pipe.simulate_events(sim, zeros, 0.0, 1.0 / pipe.cfg.hawkes.tau_init, rng)
        return _row(pipe.run(ev), replicate=b)

    det = pd.DataFrame(
        pmap(one, range(n_replicates), workers=workers, label="spell null bootstrap", every=25)
    )
    out = {
        "n_replicates": n_replicates,
        "n_unconverged": int((~det["converged"].astype(bool)).sum()),
        "p_value": {},
        "null_q95": {},
    }
    for lab in pipe.labels:
        draws = det[f"eta_{lab}"].to_numpy()
        out["p_value"][lab] = float((1.0 + (draws >= observed[f"eta_{lab}"]).sum()) / (n_replicates + 1.0))
        out["null_q95"][lab] = float(np.quantile(draws, 0.95))
    out["_detail"] = det
    return out


def cluster_bootstrap(pipe: SpellPipeline, *, n_replicates=200, seed=5694, workers=None) -> dict:
    """Match-level resampling CI. Copies read their original match's spell path."""
    rng = np.random.default_rng(seed)
    slate = pipe.slate
    ids = np.sort(slate.events["match_id"].unique())
    draws = [rng.choice(ids, size=ids.size, replace=True) for _ in range(n_replicates)]

    def one(arg):
        b, draw = arg
        sub, pairs = _resample_slate(slate, draw, with_pairs=True)
        p = SpellPipeline(
            sub,
            pipe.model,
            spell=pipe.spell,
            gamma=pipe.use_gamma,
            cfg=pipe.cfg,
            design=baseline.gather_design(pipe.design, pairs),
        )
        return _row(p.run(sub.events), replicate=b)

    det = pd.DataFrame(
        pmap(one, list(enumerate(draws)), workers=workers, label="spell cluster bootstrap", every=25)
    )
    ci = {}
    for lab in (*pipe.labels, "cross"):
        lo, hi = np.percentile(det[f"eta_{lab}"], [2.5, 97.5])
        ci[lab] = [float(lo), float(hi)]
    if pipe.model == "S":
        diff = det["eta_in"] - det["eta_out"]
        ci["in_minus_out"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
    return {
        "ci95": ci,
        "n_replicates": n_replicates,
        "n_unconverged": int((~det["converged"].astype(bool)).sum()),
        "_detail": det,
    }


def tau_profile(pipe: SpellPipeline, events, taus=TAU_PROFILE) -> list[dict]:
    out = []
    for tau in taus:
        cfg = replace(pipe.cfg.hawkes, tau_min=tau, tau_max=tau, tau_init=tau)
        r = pipe.run(events, hawkes_cfg=cfg).as_dict()
        out.append(
            {
                "tau_minutes": tau,
                **{k: r[k] for k in r if k.startswith("eta_") or k in ("loglik", "converged", "gamma_own")},
            }
        )
    return out


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm step-down over the five primary tests. Returns adjusted p and rejections."""
    names = sorted(pvalues, key=lambda k: pvalues[k])
    m = len(names)
    adj, running = {}, 0.0
    for i, k in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        adj[k] = running
    return {
        "alpha": alpha,
        "m": m,
        "raw": dict(pvalues),
        "adjusted": adj,
        "reject": {k: bool(adj[k] < alpha) for k in pvalues},
    }
