"""The joint fit, and the identity it rests on.

`joint.objective` claims that the gradient of the observed Hawkes log-likelihood
with respect to the background parameters is a Poisson-regression gradient
evaluated at the fractional counts p_bg = mu/lambda. That is the whole reason the
estimator is affordable, and it is exactly the kind of derivation that is wrong in
a way nothing else notices. So it is checked against finite differences of the
actual likelihood.
"""

import numpy as np
import pytest

from src.config import CONFIG
from src.inference.pipeline import Pipeline
from src.models import baseline, hawkes, joint


@pytest.fixture(scope="module")
def setup():
    from src.ingest import synthetic

    slate = synthetic.make_slate(n_matches=40, n_teams=8, seed=77, eta_self=0.2, beta=1 / 5.0)
    pipe = Pipeline(slate, CONFIG)
    bg = pipe.fit_background(slate.events, warm_start=False)
    data = hawkes.prepare(slate, bg, CONFIG.hawkes)
    return pipe, bg, data


def test_joint_gradient_matches_finite_differences(setup):
    pipe, bg, data = setup
    n_mu = baseline.n_params(pipe.design)
    rng = np.random.default_rng(2)
    z = hawkes.natural_to_unconstrained(0.15, 0.05, 1 / 6.0, CONFIG.hawkes)
    p = np.concatenate([bg.params + rng.normal(0, 0.02, n_mu), z])

    _, grad = joint.objective(p, pipe.design, data, CONFIG.hawkes)
    eps = 1e-6
    probe = [*rng.choice(n_mu, size=10, replace=False), n_mu, n_mu + 1, n_mu + 2]
    for i in probe:
        lo, hi = p.copy(), p.copy()
        lo[i] -= eps
        hi[i] += eps
        f_lo, _ = joint.objective(lo, pipe.design, data, CONFIG.hawkes)
        f_hi, _ = joint.objective(hi, pipe.design, data, CONFIG.hawkes)
        fd = (f_hi - f_lo) / (2 * eps)
        assert fd == pytest.approx(grad[i], rel=2e-4, abs=1e-4), f"parameter {i}"


def test_objective_agrees_with_the_plain_likelihood(setup):
    """The joint objective must be the negative log-likelihood plus the stated ridge,
    with nothing else quietly added."""
    pipe, bg, data = setup
    z = hawkes.natural_to_unconstrained(0.15, 0.05, 1 / 6.0, CONFIG.hawkes)
    p = np.concatenate([bg.params, z])
    nll, _ = joint.objective(p, pipe.design, data, CONFIG.hawkes)

    log_rate, _rate, int_mu = joint._mu_pieces(bg.params, pipe.design)
    data.update_mu(log_rate[data.design_rows], int_mu)
    ll = hawkes.loglik(data, hawkes.branching_matrix(0.15, 0.05), 1 / 6.0)
    u = baseline._unpack(bg.params, pipe.design)
    ridge = 0.5 * CONFIG.background.team_ridge * (u["a"] @ u["a"] + u["d"] @ u["d"])
    assert nll == pytest.approx(-ll + ridge, rel=1e-10)


def test_joint_fit_beats_the_two_stage_likelihood(setup):
    pipe, bg, data = setup
    two_stage = hawkes.fit(data, CONFIG.hawkes)
    jf = joint.fit(pipe.design, data, CONFIG.hawkes, theta_init=bg.params)
    assert jf.hawkes_fit.loglik >= two_stage.loglik - 1e-6


def test_joint_fit_respects_the_stationarity_and_box_constraints(setup):
    pipe, bg, data = setup
    jf = joint.fit(pipe.design, data, CONFIG.hawkes, theta_init=bg.params)
    hk = jf.hawkes_fit
    assert 0 <= hk.spectral_radius < 1.0
    assert CONFIG.hawkes.tau_min - 1e-6 <= hk.tau_minutes <= CONFIG.hawkes.tau_max + 1e-6


def test_update_mu_is_equivalent_to_rebuilding(setup):
    """`update_mu` is the shortcut the optimiser takes every iteration; it must give
    the same likelihood as rebuilding the padded arrays from scratch."""
    pipe, bg, data = setup
    theta = bg.params
    log_rate, _rate, int_mu = joint._mu_pieces(theta, pipe.design)
    data.update_mu(log_rate[data.design_rows], int_mu)
    via_update = hawkes.loglik(data, hawkes.branching_matrix(0.1, 0.05), 1 / 5.0)
    rebuilt = hawkes.prepare(pipe.slate, bg, CONFIG.hawkes)
    via_rebuild = hawkes.loglik(rebuilt, hawkes.branching_matrix(0.1, 0.05), 1 / 5.0)
    assert via_update == pytest.approx(via_rebuild, rel=1e-12)


def test_all_three_estimators_agree_when_there_is_no_excitation():
    from src.ingest import synthetic

    slate = synthetic.make_slate(n_matches=80, n_teams=10, seed=91)
    pipe = Pipeline(slate, CONFIG)
    fits = {m: pipe.run(slate.events, warm_start=False, method=m).hawkes_fit for m in
            ("joint", "two_stage", "em")}
    for m, f in fits.items():
        assert f.eta_self < 0.05, f"{m} manufactured excitation from a Poisson process"
