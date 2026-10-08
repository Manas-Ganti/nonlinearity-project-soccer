"""The background model. Most of the work and most of the value."""

import numpy as np
import pytest

from src.config import CONFIG, BackgroundConfig
from src.inference.pipeline import Pipeline
from src.ingest import synthetic
from src.models import baseline, spline


@pytest.fixture(scope="module")
def big_synth():
    return synthetic.make_slate(n_matches=600, n_teams=16, seed=31)


def test_spline_basis_is_finite_and_full_rank():
    x = np.linspace(0, 95, 96)
    B = spline.natural_cubic_basis(x, spline.natural_cubic_knots(6, 0.0, 95.0))
    assert B.shape == (96, 6)
    assert np.isfinite(B).all()
    assert np.linalg.matrix_rank(B) == 6


def test_natural_spline_is_linear_beyond_the_outer_knots():
    knots = spline.natural_cubic_knots(5, 10.0, 80.0)
    x = np.array([85.0, 90.0, 95.0, 100.0])
    B = spline.natural_cubic_basis(x, knots)
    second_diff = np.diff(B, n=2, axis=0)
    assert np.allclose(second_diff, 0, atol=1e-8)


def test_design_row_count_matches_bins(big_synth):
    design = baseline.build_design(big_synth, CONFIG.background)
    assert design.n_rows == int(design.seg_nbins.sum())
    assert design.n_seg == 2 * big_synth.events["match_id"].nunique()
    assert np.all(design.exposure > 0)
    assert design.exposure.sum() == pytest.approx(design.seg_T.sum())


def test_every_event_lands_in_a_row(big_synth):
    design = baseline.build_design(big_synth, CONFIG.background)
    y = baseline.counts(design, big_synth.events)
    assert y.sum() == len(big_synth.events)


def test_fit_recovers_planted_effects(big_synth):
    """The synthetic generator uses hand-set effects that share no code with the
    estimator, so agreement is evidence about the estimator."""
    design = baseline.build_design(big_synth, CONFIG.background)
    y = baseline.counts(design, big_synth.events)
    fit = baseline.fit(design, y)
    assert fit.converged
    d = fit.unpack()

    assert d["theta_home"] == pytest.approx(synthetic.TRUE["home"], abs=0.06)

    g = dict(zip(map(int, design.score_levels), d["g"], strict=True))
    truth = synthetic.TRUE["score"]
    # leading teams create less: the ordering is the thing that must survive
    assert g[2] < g[1] < g[0] < g[-1]
    for level in (-1, 1, 2):
        assert g[level] == pytest.approx(truth[level], abs=0.10)


def test_integral_of_mu_matches_observed_counts(big_synth):
    """A Poisson MLE with an intercept makes total fitted mass equal total events."""
    design = baseline.build_design(big_synth, CONFIG.background)
    y = baseline.counts(design, big_synth.events)
    fit = baseline.fit(design, y)
    assert fit.integral_per_segment().sum() == pytest.approx(y.sum(), rel=1e-3)


def test_cumulative_integral_is_monotone_within_segments(big_synth):
    design = baseline.build_design(big_synth, CONFIG.background)
    fit = baseline.fit(design, baseline.counts(design, big_synth.events))
    cum = fit.cumulative_integral()
    for s in range(min(design.n_seg, 50)):
        lo = design.seg_start[s]
        hi = lo + design.seg_nbins[s]
        seg = cum[lo:hi]
        assert np.all(np.diff(seg) > 0)
        assert seg[-1] == pytest.approx(fit.integral_per_segment()[s])


def test_dropping_score_state_changes_the_fit(big_synth):
    """Constraint 3: omit g_score and the model must visibly change. If it does not,
    the covariate is not doing anything and something is wired wrong."""
    with_score = baseline.build_design(big_synth, CONFIG.background)
    fit_a = baseline.fit(with_score, baseline.counts(with_score, big_synth.events))
    cfg_b = BackgroundConfig(include_score=False)
    without = baseline.build_design(big_synth, cfg_b)
    fit_b = baseline.fit(without, baseline.counts(without, big_synth.events))
    assert fit_a.loglik > fit_b.loglik


def test_warm_start_reaches_the_same_optimum(big_synth):
    pipe = Pipeline(big_synth, CONFIG)
    cold = pipe.fit_background(big_synth.events, warm_start=False)
    pipe.fit_background(big_synth.events, warm_start=True)
    warm2 = pipe.fit_background(big_synth.events, warm_start=True)
    assert warm2.loglik == pytest.approx(cold.loglik, rel=1e-6)


def test_gradient_matches_finite_differences(big_synth):
    sub = big_synth.filter_matches(sorted(big_synth.events["match_id"].unique())[:20])
    design = baseline.build_design(sub, CONFIG.background)
    y = baseline.counts(design, sub.events)
    rng = np.random.default_rng(0)
    p = rng.normal(0, 0.05, baseline.n_params(design))
    p[0] = np.log(0.12)
    _, g0 = baseline._objective(p, design, y)
    eps = 1e-5
    for i in rng.choice(p.size, size=12, replace=False):
        lo, hi = p.copy(), p.copy()
        lo[i] -= eps
        hi[i] += eps
        f_lo, _ = baseline._objective(lo, design, y)
        f_hi, _ = baseline._objective(hi, design, y)
        assert (f_hi - f_lo) / (2 * eps) == pytest.approx(g0[i], rel=1e-5, abs=1e-6)
