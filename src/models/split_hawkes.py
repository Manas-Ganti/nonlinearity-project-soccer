"""Bivariate Hawkes with the own-team kernel split by a class of the source shot
(docs/spec_spells.md sections 4 and 5).

    lambda_k(t) = mu_k(t)
                + sum_{i: k_i = k}  eta_own[c_i] * beta e^{-beta (t - t_i)}
                + sum_{i: k_i != k} eta_cross    * beta e^{-beta (t - t_i)}

- Model S: c_i = Z_{k_i}(t_i), the shooting team's spell state at the shot
  (0 = out, 1 = in). The background carries gamma_own * Z_k(t) + gamma_opp * Z_{-k}(t).
- Model G: c_i = the shooting team's score state at the shot (0 = level,
  1 = trailing, 2 = leading). No gamma terms; the background is the headline one.

`eta_cross` is never split. With every eta_own[c] equal this is exactly the
headline model, which the tests check against `hawkes.loglik`.

**Gamma inside a minute bin.** Z switches at possession boundaries, not minute
edges, so each background row carries the minutes it spends in each of the four
(Z_own, Z_opp) states (`spells.SpellState.row_weights`). Then

    int over the row of mu = rate_row * sum_s w_row,s * exp(gamma . s)

exactly, and mu at a shot is rate_row * exp(gamma . s_shot).

Joint MLE of the background, gamma and the kernel, as in `models/joint.py`:
the background gradient is the Poisson-regression gradient at fractional counts
p_bg = mu / lambda with the effective exposure above; gamma has an analytic
gradient; the kernel parameters use central differences.

Parameterisation (stationarity max_c eta_own[c] + eta_cross < rho_max by
construction):  eta_cross = rho_max * expit(z_x),
                eta_own[c] = (rho_max - eta_cross) * expit(z_c),  tau in the box.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from src.config import HawkesConfig
from src.features.spells import STATE_OPP, STATE_OWN
from src.models import baseline, hawkes


@dataclass
class SplitData:
    base: hawkes.HawkesData  # padding, times, teams, design rows
    cls: np.ndarray  # (M, N) int class of each event (0 where padded)
    n_class: int
    z_own: np.ndarray  # (n_events,) in pad order; zeros when gamma is off
    z_opp: np.ndarray
    w: np.ndarray | None  # (n_rows, 4) exposure by (Z_own, Z_opp) state, or None = no gamma

    @property
    def use_gamma(self) -> bool:
        return self.w is not None


def make_data(base: hawkes.HawkesData, cls_ev: np.ndarray, n_class: int, z_own=None, z_opp=None, w=None):
    """`cls_ev`, `z_own`, `z_opp` are per event in `hawkes.prepare`'s sorted order."""
    cls = np.zeros(base.t.shape, dtype=np.int64)
    cls[base.pad_r, base.pad_c] = np.asarray(cls_ev, dtype=np.int64)
    if cls_ev.size and (cls_ev.min() < 0 or cls_ev.max() >= n_class):
        raise ValueError("event class out of range")
    n = base.n_events
    zo = np.zeros(n) if z_own is None else np.asarray(z_own, dtype=np.float64)
    zp = np.zeros(n) if z_opp is None else np.asarray(z_opp, dtype=np.float64)
    return SplitData(base=base, cls=cls, n_class=n_class, z_own=zo, z_opp=zp, w=w)


# --------------------------------------------------------------- likelihood


def loglik_and_intensity(sd: SplitData, eta_own: np.ndarray, eta_cross: float, beta: float):
    """O(n) recursion with one running sum per (team, class). Uses sd.base.log_mu / int_mu."""
    d = sd.base
    t, team, mask, cls = d.t, d.team, d.mask, sd.cls
    M, N = t.shape
    C = sd.n_class
    eta_own = np.asarray(eta_own, dtype=np.float64)
    E = np.zeros((M, 2, C))
    lam_pad = np.zeros((M, N))
    total = 0.0
    prev_t = np.zeros(M)
    rows = np.arange(M)
    for i in range(N):
        cm = mask[:, i]
        if not cm.any():
            break
        ti = np.where(cm, t[:, i], prev_t)
        E *= np.exp(-beta * (ti - prev_t))[:, None, None]
        k = team[:, i].astype(np.int64)
        own = E[rows, k, :] @ eta_own
        cross = eta_cross * E[rows, 1 - k, :].sum(axis=1)
        lam = np.exp(d.log_mu[:, i]) + beta * (own + cross)
        lam_pad[:, i] = lam
        total += float(np.log(lam[cm]).sum())
        E[rows, k, cls[:, i]] += np.where(cm, 1.0, 0.0)
        prev_t = ti
    edge = np.where(mask, 1.0 - np.exp(-beta * (d.T[:, None] - t)), 0.0)
    comp_exc = float(((eta_own[cls] + eta_cross) * edge)[mask].sum())
    ll = total - d.int_mu.sum() - comp_exc
    return ll, lam_pad[d.pad_r, d.pad_c]


def loglik(sd: SplitData, eta_own, eta_cross, beta) -> float:
    return loglik_and_intensity(sd, eta_own, eta_cross, beta)[0]


def naive_loglik(sd: SplitData, eta_own, eta_cross, beta) -> float:
    """O(n^2) reference. Tests only."""
    d = sd.base
    eta_own = np.asarray(eta_own, dtype=np.float64)
    total = 0.0
    for m in range(d.n_matches):
        idx = np.flatnonzero(d.mask[m])
        tm, km, cm = d.t[m, idx], d.team[m, idx], sd.cls[m, idx]
        for a in range(idx.size):
            exc = 0.0
            for b in range(a):
                e = eta_own[cm[b]] if km[b] == km[a] else eta_cross
                exc += e * beta * np.exp(-beta * (tm[a] - tm[b]))
            total += np.log(np.exp(d.log_mu[m, idx[a]]) + exc)
        for a in range(idx.size):
            total -= (eta_own[cm[a]] + eta_cross) * (1.0 - np.exp(-beta * (d.T[m] - tm[a])))
    return total - d.int_mu.sum()


# ---------------------------------------------------------- parameterisation


def to_natural(z: np.ndarray, cfg: HawkesConfig, n_class: int):
    z = np.asarray(z, dtype=np.float64)
    eta_cross = cfg.rho_max * expit(z[0])
    eta_own = (cfg.rho_max - eta_cross) * expit(z[1 : 1 + n_class])
    tau = cfg.tau_min + (cfg.tau_max - cfg.tau_min) * expit(z[1 + n_class])
    return eta_own, float(eta_cross), float(1.0 / tau)


def to_unconstrained(eta_own, eta_cross: float, tau: float, cfg: HawkesConfig) -> np.ndarray:
    eps = 1e-6
    x = float(np.clip(eta_cross / cfg.rho_max, eps, 1 - eps))
    rem = cfg.rho_max - eta_cross
    own = np.clip(np.asarray(eta_own, dtype=np.float64) / rem, eps, 1 - eps)
    span = cfg.tau_max - cfg.tau_min
    frac = 0.5 if span <= 0 else float(np.clip((tau - cfg.tau_min) / span, eps, 1 - eps))
    return np.concatenate([[logit(x)], logit(own), [logit(frac)]])


# ------------------------------------------------------------------ the fit


@dataclass
class SplitFit:
    eta_own: np.ndarray
    eta_cross: float
    beta: float
    gamma_own: float
    gamma_opp: float
    theta: np.ndarray
    loglik: float
    converged: bool
    n_iter: int
    message: str
    n_events: int
    n_matches: int
    kernel: bool

    @property
    def tau_minutes(self) -> float:
        return 1.0 / self.beta

    def as_dict(self, labels: tuple[str, ...]) -> dict:
        out = {f"eta_{lab}": float(v) for lab, v in zip(labels, self.eta_own, strict=True)}
        out.update(
            {
                "eta_cross": float(self.eta_cross),
                "tau_minutes": float(self.tau_minutes),
                "gamma_own": float(self.gamma_own),
                "gamma_opp": float(self.gamma_opp),
                "loglik": float(self.loglik),
                "converged": bool(self.converged),
                "n_iter": int(self.n_iter),
                "n_events": int(self.n_events),
                "n_matches": int(self.n_matches),
                "kernel": bool(self.kernel),
            }
        )
        return out


def _state_factor(gamma: np.ndarray) -> np.ndarray:
    return np.exp(gamma[0] * STATE_OWN + gamma[1] * STATE_OPP)  # (4,)


def _background(theta, gamma, design, sd: SplitData):
    """Effective exposure, rate per row, log mu at events, int mu."""
    log_rate = baseline._log_rate(theta, design)
    rate = np.exp(log_rate)
    if sd.use_gamma:
        f = _state_factor(gamma)
        e_eff = sd.w @ f
        log_mu_ev = log_rate[sd.base.design_rows] + gamma[0] * sd.z_own + gamma[1] * sd.z_opp
    else:
        f = None
        e_eff = design.exposure
        log_mu_ev = log_rate[sd.base.design_rows]
    return rate, e_eff, f, log_mu_ev, float((rate * e_eff).sum())


def _split(p, n_mu, n_g):
    return p[:n_mu], (p[n_mu : n_mu + n_g] if n_g else np.zeros(2)), p[n_mu + n_g :]


def objective(p, design, sd: SplitData, cfg: HawkesConfig, kernel: bool, fd_step: float = 1e-5):
    n_mu = baseline.n_params(design)
    n_g = 2 if sd.use_gamma else 0
    theta, gamma, z = _split(p, n_mu, n_g)
    rate, e_eff, f, log_mu_ev, int_mu = _background(theta, gamma, design, sd)
    d = sd.base
    d.update_mu(log_mu_ev, int_mu)

    if kernel:
        eta_own, eta_cross, beta = to_natural(z, cfg, sd.n_class)
        ll, lam = loglik_and_intensity(sd, eta_own, eta_cross, beta)
        p_bg = np.exp(log_mu_ev) / np.maximum(lam, 1e-300)
    else:
        ll = float(log_mu_ev.sum()) - int_mu
        p_bg = np.ones(d.n_events)

    u = baseline._unpack(theta, design)
    ridge = design.cfg.team_ridge
    nll = -ll + 0.5 * ridge * (float(u["a"] @ u["a"]) + float(u["d"] @ u["d"]))

    y = np.zeros(design.n_rows)
    np.add.at(y, d.design_rows, p_bg)
    _, grad_mu = baseline._objective(theta, replace(design, exposure=e_eff), y)
    grads = [grad_mu]

    if n_g:
        g_own = -float(p_bg @ sd.z_own) + float(rate @ (sd.w @ (f * STATE_OWN)))
        g_opp = -float(p_bg @ sd.z_opp) + float(rate @ (sd.w @ (f * STATE_OPP)))
        grads.append(np.array([g_own, g_opp]))

    if kernel:
        gz = np.zeros(z.size)
        for i in range(z.size):
            lls = []
            for sgn in (+1, -1):
                zz = z.copy()
                zz[i] += sgn * fd_step
                eo, ec, bb = to_natural(zz, cfg, sd.n_class)
                lls.append(loglik(sd, eo, ec, bb))
            gz[i] = -(lls[0] - lls[1]) / (2 * fd_step)
        grads.append(gz)
    return nll, np.concatenate(grads)


def fit(
    design: baseline.BackgroundDesign,
    sd: SplitData,
    cfg: HawkesConfig,
    *,
    theta_init: np.ndarray,
    kernel: bool = True,
    z_init: np.ndarray | None = None,
    gamma_init: np.ndarray | None = None,
    maxiter: int = 3000,
) -> SplitFit:
    """Joint MLE. `kernel=False` is the Poisson null with gamma (the bootstrap's null)."""
    n_mu = baseline.n_params(design)
    n_g = 2 if sd.use_gamma else 0
    parts = [np.asarray(theta_init, dtype=np.float64)]
    if n_g:
        parts.append(np.zeros(2) if gamma_init is None else np.asarray(gamma_init, dtype=np.float64))
    if kernel:
        if z_init is None:
            z_init = to_unconstrained(
                np.full(sd.n_class, cfg.eta_self_init), cfg.eta_cross_init, cfg.tau_init, cfg
            )
        parts.append(np.asarray(z_init, dtype=np.float64))
    p0 = np.concatenate(parts)
    res = minimize(
        objective,
        p0,
        args=(design, sd, cfg, kernel),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6, "maxcor": 20},
    )
    theta, gamma, z = _split(res.x, n_mu, n_g)
    _, _, _, log_mu_ev, int_mu = _background(theta, gamma, design, sd)
    sd.base.update_mu(log_mu_ev, int_mu)
    if kernel:
        eta_own, eta_cross, beta = to_natural(z, cfg, sd.n_class)
        ll = loglik(sd, eta_own, eta_cross, beta)
    else:
        eta_own, eta_cross, beta = np.zeros(sd.n_class), 0.0, 1.0 / cfg.tau_init
        ll = float(log_mu_ev.sum()) - int_mu
    return SplitFit(
        eta_own=np.asarray(eta_own),
        eta_cross=float(eta_cross),
        beta=float(beta),
        gamma_own=float(gamma[0]) if n_g else 0.0,
        gamma_opp=float(gamma[1]) if n_g else 0.0,
        theta=theta.copy(),
        loglik=float(ll),
        converged=bool(res.success),
        n_iter=int(res.nit),
        message=str(res.message),
        n_events=sd.base.n_events,
        n_matches=sd.base.n_matches,
        kernel=kernel,
    )


def rate_rows(fit: SplitFit, design) -> np.ndarray:
    return np.exp(baseline._log_rate(fit.theta, design))
