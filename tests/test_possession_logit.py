"""Model A (docs/spec_possessions.md section 3): the recursion, the gradient, recovery.

Gate 1 of the spec: the excitation sums match a brute-force double sum, the analytic
gradient matches finite differences, and a fit on data simulated from known
parameters gets them back.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import CONFIG
from src.models import possession_logit as pl


def _table(n_matches=40, n_poss=60, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    for m in range(n_matches):
        home, away = rng.choice(teams, 2, replace=False)
        t = np.sort(rng.uniform(0, 95, n_poss))
        for k, tk in enumerate(t):
            side = "H" if k % 2 == 0 else "A"
            danger = rng.random() < 0.5
            rows.append(
                {
                    "match_id": f"m{m}",
                    "t_start": tk,
                    "side": side,
                    "team": home if side == "H" else away,
                    "opponent": away if side == "H" else home,
                    "season": 2017,
                    "source": "test",
                    "score_diff": int(rng.integers(-2, 3)),
                    "red_diff": int(rng.integers(-1, 2)),
                    "sp_ft": bool(rng.random() < 0.3),
                    "t_ft": tk + 0.1 if danger else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_excitation_recursion_matches_the_double_sum():
    d = pl.build_design(_table(n_matches=6, n_poss=25), CONFIG.background)
    for tau in (0.5, 5.0, 30.0):
        fast = pl.excitation(d, d.y, tau)
        slow = pl.excitation_bruteforce(d, d.y, tau)
        np.testing.assert_allclose(fast[0], slow[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fast[1], slow[1], rtol=1e-12, atol=1e-12)


def test_the_first_possession_of_a_match_sees_no_history():
    d = pl.build_design(_table(n_matches=3, n_poss=10), CONFIG.background)
    s_self, s_cross = pl.excitation(d, d.y, 5.0)
    first = d.grid[:, 0]
    assert np.all(s_self[first] == 0) and np.all(s_cross[first] == 0)


def test_gradient_matches_finite_differences():
    d = pl.build_design(_table(n_matches=8, n_poss=30), CONFIG.background)
    s = pl._slices(d, True)
    S = pl.excitation(d, d.y, 5.0)
    rng = np.random.default_rng(1)
    p = rng.normal(scale=0.1, size=s["_n"])
    _, g = pl.objective(p, d, d.y, S, s)
    h = 1e-6
    for i in [*rng.choice(s["_n"], 25, replace=False).tolist(), s["rho"].start, s["rho"].start + 1]:
        e = np.zeros_like(p)
        e[i] = h
        fd = (pl.objective(p + e, d, d.y, S, s)[0] - pl.objective(p - e, d, d.y, S, s)[0]) / (2 * h)
        assert g[i] == pytest.approx(fd, rel=1e-5, abs=1e-6)


def test_simulation_at_rho_zero_draws_from_the_background():
    d = pl.build_design(_table(n_matches=200, n_poss=60), CONFIG.background)
    eta = np.full(d.n, 0.4)
    y = pl.simulate(d, eta, 0.0, 0.0, 5.0, np.random.default_rng(2))
    assert y.mean() == pytest.approx(1 / (1 + np.exp(-0.4)), abs=0.01)


def test_a_planted_rho_is_recovered():
    d = pl.build_design(_table(n_matches=300, n_poss=80, seed=3), CONFIG.background)
    bg = pl.fit(d, excitation_terms=False)
    hits = []
    for r in range(3):
        y = pl.simulate(d, bg.eta_background, 0.4, -0.1, 5.0, np.random.default_rng(10 + r))
        f = pl.fit(d, y=y)
        assert f.converged
        hits.append((f.rho_self, f.rho_cross))
    m = np.mean(hits, axis=0)
    assert m[0] == pytest.approx(0.4, abs=0.08)
    assert m[1] == pytest.approx(-0.1, abs=0.08)


def test_background_only_fit_ignores_excitation_and_reports_unpenalised_loglik():
    d = pl.build_design(_table(n_matches=20, n_poss=40), CONFIG.background)
    f = pl.fit(d, excitation_terms=False)
    p = 1 / (1 + np.exp(-f.eta_background))
    ll = float(np.sum(d.y * np.log(p) + (1 - d.y) * np.log(1 - p)))
    assert f.loglik == pytest.approx(ll, rel=1e-9)
    assert f.rho_self == 0.0 and f.ridge_penalty >= 0.0
