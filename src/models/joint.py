"""Joint maximum likelihood for (mu, eta, beta).

Why this and not the two-stage fit docs/math.md sketches: fitting a Poisson
background to data that *contains* excitation inflates mu by roughly 1/(1 - rho),
because every offspring event has to be explained as background. Freezing that
inflated mu leaves nothing for eta, and the recovery test measures the damage --
a planted eta_self = 0.30 comes back as 0.08. The bias is downward and large.
(math.md argues the two-stage fit is biased *towards* excitation; that is true of
the kernel's shape but the level effect dominates and reverses the sign.)

EM (Veen & Schoenberg 2008) fixes the bias but crawls. The useful observation is
that the gradient of the *observed* log-likelihood with respect to the background
parameters is exactly the EM M-step gradient:

    d/dtheta [ sum_i log(mu_i + exc_i) - int mu ]
        = sum_i (mu_i / lambda_i) dlog mu_i/dtheta - sum_rows exposure * mu_row * ...

which is a Poisson-regression gradient with the fractional counts p_bg = mu/lambda.
So the whole thing goes into one L-BFGS over [theta_mu, z], with analytic gradients
for the ~2000 background parameters and central differences for the three kernel
parameters. Same fixed point as EM, an order of magnitude fewer likelihood
evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from src.config import HawkesConfig
from src.models import baseline, hawkes


@dataclass
class JointFit:
    background: baseline.BackgroundFit
    hawkes_fit: hawkes.HawkesFit
    data: hawkes.HawkesData
    n_iter: int
    converged: bool
    message: str


def _mu_pieces(theta: np.ndarray, design: baseline.BackgroundDesign):
    log_rate = baseline._log_rate(theta, design)
    rate = np.exp(log_rate)
    int_mu = float((rate * design.exposure).sum())
    return log_rate, rate, int_mu


def objective(
    p: np.ndarray,
    design: baseline.BackgroundDesign,
    data: hawkes.HawkesData,
    cfg: HawkesConfig,
    *,
    fd_step: float = 1e-5,
):
    n_mu = baseline.n_params(design)
    theta, z = p[:n_mu], p[n_mu:]
    eta_self, eta_cross, beta = hawkes.unconstrained_to_natural(z, cfg)
    eta = hawkes.branching_matrix(eta_self, eta_cross)

    log_rate, _rate, int_mu = _mu_pieces(theta, design)
    data.update_mu(log_rate[data.design_rows], int_mu)
    ll, lam = hawkes.loglik_and_intensity(data, eta, beta)

    ridge = design.cfg.team_ridge
    u = baseline._unpack(theta, design)
    nll = -ll + 0.5 * ridge * (float(u["a"] @ u["a"]) + float(u["d"] @ u["d"]))

    # background gradient == Poisson-regression gradient at the fractional counts p_bg
    mu_ev = np.exp(data.log_mu[data.pad_r, data.pad_c])
    p_bg = mu_ev / np.maximum(lam, 1e-300)
    y = np.zeros(design.n_rows)
    np.add.at(y, data.design_rows, p_bg)
    _, grad_mu = baseline._objective(theta, design, y)

    # kernel gradient: three central differences, each one recursion pass
    grad_z = np.zeros(z.size)
    for i in range(z.size):
        lls = []
        for sign in (+1, -1):
            zz = z.copy()
            zz[i] += sign * fd_step
            es, ec, bb = hawkes.unconstrained_to_natural(zz, cfg)
            lls.append(hawkes.loglik(data, hawkes.branching_matrix(es, ec), bb))
        grad_z[i] = -(lls[0] - lls[1]) / (2 * fd_step)

    return nll, np.concatenate([grad_mu, grad_z])


def fit(
    design: baseline.BackgroundDesign,
    data: hawkes.HawkesData,
    cfg: HawkesConfig,
    *,
    theta_init: np.ndarray,
    z_init: np.ndarray | None = None,
    maxiter: int = 400,
) -> JointFit:
    if z_init is None:
        z_init = hawkes.natural_to_unconstrained(
            cfg.eta_self_init, cfg.eta_cross_init, 1.0 / cfg.tau_init, cfg
        )
    p0 = np.concatenate([np.asarray(theta_init, dtype=np.float64), np.asarray(z_init, dtype=np.float64)])
    res = minimize(
        objective,
        p0,
        args=(design, data, cfg),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6, "maxcor": 20},
    )
    n_mu = baseline.n_params(design)
    theta, z = res.x[:n_mu], res.x[n_mu:]
    eta_self, eta_cross, beta = hawkes.unconstrained_to_natural(z, cfg)

    log_rate, _rate, int_mu = _mu_pieces(theta, design)
    data.update_mu(log_rate[data.design_rows], int_mu)
    ll = hawkes.loglik(data, hawkes.branching_matrix(eta_self, eta_cross), beta)

    bg = baseline.BackgroundFit(
        params=theta,
        design=design,
        loglik=float(ll),
        converged=bool(res.success),
        n_iter=int(res.nit),
        message=str(res.message),
    )
    hk = hawkes.HawkesFit(
        eta_self=eta_self,
        eta_cross=eta_cross,
        beta=beta,
        loglik=float(ll),
        converged=bool(res.success),
        n_iter=int(res.nit),
        message=str(res.message),
        n_events=data.n_events,
        n_matches=data.n_matches,
        method="joint MLE (mu and the kernel fitted together)",
    )
    return JointFit(
        background=bg,
        hawkes_fit=hk,
        data=data,
        n_iter=int(res.nit),
        converged=bool(res.success),
        message=str(res.message),
    )
