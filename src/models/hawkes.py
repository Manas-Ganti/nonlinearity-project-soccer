"""Bivariate Hawkes with a single exponential kernel.

    lambda_k(t) = mu_k(t) + sum_j sum_{t_i^j < t} eta_kj * beta * exp(-beta (t - t_i^j))

The kernel is normalised so that `eta` IS the branching ratio (int phi = eta);
the alpha/beta parameterisation is not used (docs/math.md section 3).

There is deliberately **no fast kernel component**. At Understat's minute
resolution it is not identifiable, and an optimiser handed one will return a
confident number for it anyway. Mechanical clustering is removed by
`features.dedup` instead (CLAUDE.md, constraint 1).

Likelihood: Ogata's O(n) recursion, run over a padded (n_matches x n_max) array so
that all matches step in lockstep and the whole pass is a few dozen vectorised
operations. `naive_loglik` is the O(n^2) double sum, kept only as the reference
the recursion is tested against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from src.config import BIN_MINUTES, CONFIG, HawkesConfig
from src.models import baseline


@dataclass
class HawkesData:
    """A slate flattened into padded arrays, ready for the recursion.

    Rows are matches; columns are the merged (both teams) event sequence in time
    order. `mask` marks real entries.
    """

    t: np.ndarray  # (M, N) float64 event times, 0 where padded
    team: np.ndarray  # (M, N) int8: 0 = home, 1 = away
    mask: np.ndarray  # (M, N) bool
    log_mu: np.ndarray  # (M, N) float64 log background at each event time
    cum_mu: np.ndarray  # (M, N) float64 int_0^{t_i} mu_{k_i}(s) ds
    T: np.ndarray  # (M,) match length
    int_mu: np.ndarray  # (M, 2) integral of mu over [0, T] for home/away
    design_rows: np.ndarray  # (n_events,) design row per event, in flattened-mask order
    pad_r: np.ndarray  # (n_events,) padded-array row for each event, same order
    pad_c: np.ndarray  # (n_events,) padded-array column for each event, same order
    n_events: int

    @property
    def n_matches(self) -> int:
        return self.t.shape[0]

    def update_mu(self, log_mu_events: np.ndarray, int_mu_total: float) -> None:
        """Swap in a new background without rebuilding the padded layout.

        The padding depends only on the event times, which do not move while the
        background is being fitted, so the joint optimiser can call this every
        iteration for the cost of a scatter.
        """
        self.log_mu[self.pad_r, self.pad_c] = log_mu_events
        self.int_mu[:] = 0.0
        self.int_mu[0, 0] = float(int_mu_total)


def prepare(slate, fit_mu: baseline.BackgroundFit, cfg: HawkesConfig) -> HawkesData:
    events = slate.events
    if cfg.exclude_penalties:
        events = events[events["situation"] != "penalty"]
    events = events.sort_values(["match_id", "t"], kind="stable").reset_index(drop=True)

    design = fit_mu.design
    log_rate = fit_mu.log_rate_rows()
    int_seg = fit_mu.integral_per_segment()

    rows = design.event_rows(events) if len(events) else np.zeros(0, dtype=np.int64)
    ev_log_mu = log_rate[rows] if len(events) else np.zeros(0)

    # int_0^t mu = (integral through the end of t's bin) - (rate * the unused tail of that bin)
    cum_through_bin = fit_mu.cumulative_integral()
    rate_rows = np.exp(log_rate)
    if len(events):
        bin_end = (design.bin_idx[rows] + 1) * BIN_MINUTES
        seg_of_row = np.repeat(np.arange(design.n_seg), design.seg_nbins)[rows]
        bin_end = np.minimum(bin_end, design.seg_T[seg_of_row])
        ev_cum_mu = cum_through_bin[rows] - rate_rows[rows] * (bin_end - events["t"].to_numpy())
    else:
        ev_cum_mu = np.zeros(0)

    match_ids = pd.unique(design.seg_match)
    m_pos = {m: i for i, m in enumerate(match_ids)}
    M = len(match_ids)

    mapped = events["match_id"].map(m_pos)
    if mapped.isna().any():
        raise ValueError("event references a match that is not in the background design")
    ev_match = mapped.to_numpy(dtype=np.int64)
    if len(events) and np.any(np.diff(ev_match) < 0):
        raise ValueError("events must be grouped by match in design order")
    ev_team = (events["side"].to_numpy() == "A").astype(np.int8)
    ev_t = events["t"].to_numpy(dtype=np.float64)

    counts_per_match = np.bincount(ev_match, minlength=M)
    N = int(counts_per_match.max()) if M and counts_per_match.size else 0
    N = max(N, 1)

    col = (
        np.concatenate([np.arange(c) for c in counts_per_match])
        if len(events)
        else np.zeros(0, dtype=np.int64)
    )
    t = np.zeros((M, N))
    team = np.zeros((M, N), dtype=np.int8)
    mask = np.zeros((M, N), dtype=bool)
    log_mu = np.zeros((M, N))
    cum_mu = np.zeros((M, N))
    if len(events):
        t[ev_match, col] = ev_t
        team[ev_match, col] = ev_team
        mask[ev_match, col] = True
        log_mu[ev_match, col] = ev_log_mu
        cum_mu[ev_match, col] = ev_cum_mu

    T = np.zeros(M)
    int_mu = np.zeros((M, 2))
    for (m, side), seg in design.seg_lookup.items():
        T[m_pos[m]] = design.seg_T[seg]
        int_mu[m_pos[m], 0 if side == "H" else 1] = int_seg[seg]

    # `events` is sorted by (match, t) and the padding is filled in that order, so
    # this is aligned with `mask`-flattened order and stays aligned.
    return HawkesData(
        t=t,
        team=team,
        mask=mask,
        log_mu=log_mu,
        cum_mu=cum_mu,
        T=T,
        int_mu=int_mu,
        design_rows=np.asarray(rows, dtype=np.int64),
        pad_r=np.asarray(ev_match, dtype=np.int64),
        pad_c=np.asarray(col, dtype=np.int64),
        n_events=int(mask.sum()),
    )


# --------------------------------------------------------------- likelihood


def loglik(data: HawkesData, eta: np.ndarray, beta: float) -> float:
    """Ogata's recursion, vectorised across matches.

    `eta` is the 2x2 branching matrix, eta[k, j] = offspring of team k triggered by
    one team-j event.
    """
    t, team, mask = data.t, data.team, data.mask
    M, N = t.shape
    eta = np.asarray(eta, dtype=np.float64).reshape(2, 2)

    # E[m, j] = sum over team-j events strictly earlier than "now", decayed to now.
    E = np.zeros((M, 2))
    total_log_lambda = 0.0
    prev_t = np.zeros(M)

    for i in range(N):
        col_mask = mask[:, i]
        if not col_mask.any():
            break
        ti = np.where(col_mask, t[:, i], prev_t)
        decay = np.exp(-beta * (ti - prev_t))
        E *= decay[:, None]

        ki = team[:, i]
        # eta[k_i, 0] * E[:,0] + eta[k_i, 1] * E[:,1]
        excite = beta * (eta[ki, 0] * E[:, 0] + eta[ki, 1] * E[:, 1])
        lam = np.exp(data.log_mu[:, i]) + excite
        total_log_lambda += float(np.log(np.where(col_mask, lam, 1.0))[col_mask].sum())

        # the event itself joins the history only after contributing to lambda
        E[np.arange(M), ki] += np.where(col_mask, 1.0, 0.0)
        prev_t = ti

    # Compensator. int_0^T mu + sum_j sum_i eta_kj (1 - exp(-beta (T - t_i^j)))
    comp_mu = data.int_mu.sum()
    edge = np.where(mask, 1.0 - np.exp(-beta * (data.T[:, None] - t)), 0.0)
    # per match, per source team j: sum of edge factors
    src = np.zeros((M, 2))
    for j in (0, 1):
        src[:, j] = np.where(mask & (team == j), edge, 0.0).sum(axis=1)
    # every source excites both targets
    comp_exc = float((src * eta.sum(axis=0)[None, :]).sum())
    return total_log_lambda - comp_mu - comp_exc


def naive_loglik(data: HawkesData, eta: np.ndarray, beta: float) -> float:
    """The O(n^2) double sum. Reference implementation only -- never in a hot path."""
    eta = np.asarray(eta, dtype=np.float64).reshape(2, 2)
    total = 0.0
    for m in range(data.n_matches):
        idx = np.flatnonzero(data.mask[m])
        tm, km = data.t[m, idx], data.team[m, idx]
        for a, (ta, ka) in enumerate(zip(tm, km, strict=True)):
            exc = 0.0
            for b in range(a):
                exc += eta[ka, km[b]] * beta * np.exp(-beta * (ta - tm[b]))
            total += np.log(np.exp(data.log_mu[m, idx[a]]) + exc)
        for ta, ka in zip(tm, km, strict=True):
            total -= eta[:, ka].sum() * (1.0 - np.exp(-beta * (data.T[m] - ta)))
    return total - data.int_mu.sum()


def intensity_at_events(data: HawkesData, eta: np.ndarray, beta: float) -> np.ndarray:
    """lambda at each event, flattened in the same order as `mask.nonzero()`."""
    t, team, mask = data.t, data.team, data.mask
    M, N = t.shape
    eta = np.asarray(eta, dtype=np.float64).reshape(2, 2)
    out = np.full((M, N), np.nan)
    E = np.zeros((M, 2))
    prev_t = np.zeros(M)
    for i in range(N):
        col_mask = mask[:, i]
        if not col_mask.any():
            break
        ti = np.where(col_mask, t[:, i], prev_t)
        E *= np.exp(-beta * (ti - prev_t))[:, None]
        ki = team[:, i]
        out[:, i] = np.exp(data.log_mu[:, i]) + beta * (eta[ki, 0] * E[:, 0] + eta[ki, 1] * E[:, 1])
        E[np.arange(M), ki] += np.where(col_mask, 1.0, 0.0)
        prev_t = ti
    return out[mask]


def excitation_at_events(data: HawkesData, eta: np.ndarray, beta: float) -> np.ndarray:
    """The excitation term alone, for the marked/quality secondary analyses."""
    lam = intensity_at_events(data, eta, beta)
    return lam - np.exp(data.log_mu[data.mask])


# ---------------------------------------------------------- parameterisation


def unconstrained_to_natural(z: np.ndarray, cfg: HawkesConfig) -> tuple[float, float, float]:
    """(z0, z1, w) -> (eta_self, eta_cross, beta), all constraints satisfied by
    construction (docs/math.md section 4)."""
    rho = cfg.rho_max * expit(z[0])
    p = expit(z[1])
    eta_self = rho * p
    eta_cross = rho * (1.0 - p)
    frac = expit(z[2])  # overflow-safe: the optimiser can push z far past +-700 when tau pins
    tau = cfg.tau_min + (cfg.tau_max - cfg.tau_min) * frac
    return float(eta_self), float(eta_cross), float(1.0 / tau)


def natural_to_unconstrained(eta_self: float, eta_cross: float, beta: float, cfg: HawkesConfig) -> np.ndarray:
    rho = eta_self + eta_cross
    rho = min(max(rho, 1e-6), cfg.rho_max - 1e-6)
    p = min(max(eta_self / rho if rho > 0 else 0.5, 1e-6), 1 - 1e-6)
    tau = min(max(1.0 / beta, cfg.tau_min + 1e-9), cfg.tau_max - 1e-9)
    frac = (tau - cfg.tau_min) / (cfg.tau_max - cfg.tau_min)
    logit = lambda x: float(np.log(x / (1 - x)))  # noqa: E731
    return np.array([logit(rho / cfg.rho_max), logit(p), logit(frac)])


def branching_matrix(eta_self: float, eta_cross: float) -> np.ndarray:
    return np.array([[eta_self, eta_cross], [eta_cross, eta_self]], dtype=np.float64)


def asym_unconstrained_to_natural(z: np.ndarray, cfg: HawkesConfig) -> tuple[np.ndarray, float]:
    """Five parameters -> a full 2x2 branching matrix with a controlled spectral radius.

    CLAUDE.md allows the symmetry reduction to be relaxed as a robustness check and
    nowhere else. Enforcing `rho(N) < 1` on a general non-negative matrix is not as
    simple as the symmetric case, where rho is just the row sum, so the shape and the
    scale are separated: four logits set the relative sizes of the entries, and a
    fifth sets rho, which the matrix is then rescaled to hit exactly.

        z[0:4] -> shape,  z[4] -> rho,  z[5] -> tau
    """
    shape = expit(np.asarray(z[:4], dtype=np.float64))
    N0 = shape.reshape(2, 2)
    rho0 = float(np.max(np.abs(np.linalg.eigvals(N0))))
    rho = cfg.rho_max * expit(z[4])
    N = N0 * (rho / rho0) if rho0 > 0 else N0
    frac = expit(z[5])
    tau = cfg.tau_min + (cfg.tau_max - cfg.tau_min) * frac
    return N, float(1.0 / tau)


def fit_asymmetric(data: HawkesData, cfg: HawkesConfig, *, mu_is_fixed: bool = True) -> dict:
    """Robustness check only: eta_HH, eta_HA, eta_AH, eta_AA all free.

    Reported as a check on whether the symmetry reduction is costing anything, never
    as the headline. `mu` is held at whatever the caller fitted, so this is a
    two-stage fit and inherits that bias -- it is a comparison of shapes at a fixed
    background, not an estimate to quote.
    """
    from scipy.optimize import minimize as _minimize

    def nll(z):
        N, beta = asym_unconstrained_to_natural(z, cfg)
        return -loglik(data, N, beta)

    z0 = np.array([0.0, -1.0, -1.0, 0.0, -2.5, 0.0])
    res = _minimize(nll, z0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-6})
    N, beta = asym_unconstrained_to_natural(res.x, cfg)
    return {
        "eta_HH": float(N[0, 0]),
        "eta_HA": float(N[0, 1]),
        "eta_AH": float(N[1, 0]),
        "eta_AA": float(N[1, 1]),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(N)))),
        "tau_minutes": float(1.0 / beta),
        "loglik": float(-res.fun),
        "converged": bool(res.success),
        "note": "robustness check on the H/A symmetry reduction; mu is held fixed, so this "
        "inherits the two-stage bias and is not a headline estimate",
    }


@dataclass
class HawkesFit:
    eta_self: float
    eta_cross: float
    beta: float
    loglik: float
    converged: bool
    n_iter: int
    message: str
    n_events: int
    n_matches: int
    method: str

    @property
    def tau_minutes(self) -> float:
        return 1.0 / self.beta

    @property
    def spectral_radius(self) -> float:
        return self.eta_self + self.eta_cross

    def tau_at_boundary(self, cfg: HawkesConfig, tol: float = 1e-3) -> bool:
        """True if the kernel timescale is pinned to its box.

        A tau pinned at the short end means the optimiser is trying to fit
        sub-resolution structure it cannot see, and the estimate should not be
        reported as if the data chose it.
        """
        tau = self.tau_minutes
        return bool(abs(tau - cfg.tau_min) < tol or abs(tau - cfg.tau_max) < tol)

    def as_dict(self) -> dict:
        return {
            "eta_self": self.eta_self,
            "eta_cross": self.eta_cross,
            "beta": self.beta,
            "tau_minutes": self.tau_minutes,
            "spectral_radius": self.spectral_radius,
            "loglik": self.loglik,
            "converged": self.converged,
            "n_iter": self.n_iter,
            "n_events": self.n_events,
            "n_matches": self.n_matches,
            "mu_fit": self.method,
            "tau_at_boundary": self.tau_at_boundary(CONFIG.hawkes),
        }


def fit(data: HawkesData, cfg: HawkesConfig, *, init: np.ndarray | None = None) -> HawkesFit:
    """Maximum likelihood for (eta_self, eta_cross, beta) with mu held fixed.

    Two-stage: mu was fitted under the Poisson null and is not re-estimated here.
    That is biased *towards* finding excitation, because mu was fitted without an
    excitation term to compete with; the recovery test measures the size and sign
    of that bias directly.
    """
    z0 = (
        natural_to_unconstrained(cfg.eta_self_init, cfg.eta_cross_init, 1.0 / cfg.tau_init, cfg)
        if init is None
        else np.asarray(init, dtype=np.float64)
    )

    def nll(z):
        eta_self, eta_cross, beta = unconstrained_to_natural(z, cfg)
        return -loglik(data, branching_matrix(eta_self, eta_cross), beta)

    res = minimize(
        nll, z0, method="Nelder-Mead", options={"maxiter": cfg.maxiter, "xatol": 1e-5, "fatol": 1e-6}
    )
    eta_self, eta_cross, beta = unconstrained_to_natural(res.x, cfg)
    return HawkesFit(
        eta_self=eta_self,
        eta_cross=eta_cross,
        beta=beta,
        loglik=float(-res.fun),
        converged=bool(res.success),
        n_iter=int(res.nit),
        message=str(res.message),
        n_events=data.n_events,
        n_matches=data.n_matches,
        method="two-stage (mu fitted under the Poisson null, then held fixed)",
    )


def compensator_at_events(data: HawkesData, eta: np.ndarray, beta: float) -> dict:
    """Lambda_k(t_i) at every event, for the time-rescaling check.

        Lambda_k(t) = int_0^t mu_k + sum_j eta_kj * sum_{t_i^j < t} (1 - exp(-beta(t - t_i^j)))

    The inner sum is (count of earlier team-j events) minus (their decayed sum),
    both of which the same O(n) recursion already carries.
    """
    t, team, mask = data.t, data.team, data.mask
    M, N = t.shape
    eta = np.asarray(eta, dtype=np.float64).reshape(2, 2)

    D = np.zeros((M, 2))  # decayed sum
    A = np.zeros((M, 2))  # plain count
    out = np.full((M, N), np.nan)
    prev_t = np.zeros(M)
    for i in range(N):
        col_mask = mask[:, i]
        if not col_mask.any():
            break
        ti = np.where(col_mask, t[:, i], prev_t)
        D *= np.exp(-beta * (ti - prev_t))[:, None]
        ki = team[:, i]
        S = A - D
        out[:, i] = data.cum_mu[:, i] + eta[ki, 0] * S[:, 0] + eta[ki, 1] * S[:, 1]
        rows = np.arange(M)
        add = np.where(col_mask, 1.0, 0.0)
        D[rows, ki] += add
        A[rows, ki] += add
        prev_t = ti
    return {
        "Lambda": out[mask],
        "match": np.repeat(np.arange(M), mask.sum(axis=1)),
        "team": team[mask],
        "t": t[mask],
    }


def loglik_and_intensity(data: HawkesData, eta: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    """The log-likelihood and lambda at every event, in one pass of the recursion."""
    t, team, mask = data.t, data.team, data.mask
    M, N = t.shape
    eta = np.asarray(eta, dtype=np.float64).reshape(2, 2)

    E = np.zeros((M, 2))
    lam_pad = np.zeros((M, N))
    total_log_lambda = 0.0
    prev_t = np.zeros(M)
    rows = np.arange(M)
    for i in range(N):
        col_mask = mask[:, i]
        if not col_mask.any():
            break
        ti = np.where(col_mask, t[:, i], prev_t)
        E *= np.exp(-beta * (ti - prev_t))[:, None]
        ki = team[:, i]
        lam = np.exp(data.log_mu[:, i]) + beta * (eta[ki, 0] * E[:, 0] + eta[ki, 1] * E[:, 1])
        lam_pad[:, i] = lam
        total_log_lambda += float(np.log(lam[col_mask]).sum())
        E[rows, ki] += np.where(col_mask, 1.0, 0.0)
        prev_t = ti

    edge = np.where(mask, 1.0 - np.exp(-beta * (data.T[:, None] - t)), 0.0)
    src = np.stack([np.where(mask & (team == j), edge, 0.0).sum(axis=1) for j in (0, 1)], axis=1)
    comp_exc = float((src * eta.sum(axis=0)[None, :]).sum())
    ll = total_log_lambda - data.int_mu.sum() - comp_exc
    return ll, lam_pad[data.pad_r, data.pad_c]
