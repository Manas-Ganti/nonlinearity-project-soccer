"""The O(n) recursion against the naive O(n^2) double sum.

docs/math.md: "Implement and unit-test the recursion against the naive O(n^2) sum
on small inputs. This is the single most likely place for a silent bug."
"""

import numpy as np
import pytest

from src.config import CONFIG
from src.inference.pipeline import Pipeline
from src.models import hawkes

CASES = [
    (0.0, 0.0, 1 / 5.0),
    (0.10, 0.05, 1 / 5.0),
    (0.30, 0.20, 1 / 3.0),
    (0.45, 0.40, 1 / 8.0),
    (0.02, 0.00, 1 / 20.0),
]


@pytest.fixture(scope="module")
def data(request):
    from src.ingest import synthetic

    slate = synthetic.make_slate(n_matches=6, n_teams=6, seed=11)
    pipe = Pipeline(slate, CONFIG)
    bg = pipe.fit_background(slate.events, warm_start=False)
    return hawkes.prepare(slate, bg, CONFIG.hawkes)


@pytest.mark.parametrize(("eta_self", "eta_cross", "beta"), CASES)
def test_recursion_matches_naive(data, eta_self, eta_cross, beta):
    eta = hawkes.branching_matrix(eta_self, eta_cross)
    fast = hawkes.loglik(data, eta, beta)
    slow = hawkes.naive_loglik(data, eta, beta)
    assert fast == pytest.approx(slow, rel=0, abs=1e-8)


def test_recursion_matches_naive_asymmetric(data):
    """The symmetry reduction is a modelling choice, not something the recursion
    may quietly assume."""
    eta = np.array([[0.25, 0.05], [0.30, 0.12]])
    assert hawkes.loglik(data, eta, 1 / 4.0) == pytest.approx(
        hawkes.naive_loglik(data, eta, 1 / 4.0), abs=1e-8
    )


def test_zero_eta_reduces_to_poisson(data):
    """With eta = 0 the log-likelihood must be exactly the Poisson one."""
    ll = hawkes.loglik(data, np.zeros((2, 2)), 1 / 5.0)
    expected = float(data.log_mu[data.mask].sum() - data.int_mu.sum())
    assert ll == pytest.approx(expected, abs=1e-9)


def test_intensity_is_positive_and_above_background(data):
    lam = hawkes.intensity_at_events(data, hawkes.branching_matrix(0.2, 0.1), 1 / 5.0)
    mu = np.exp(data.log_mu[data.mask])
    assert np.all(lam > 0)
    assert np.all(lam >= mu - 1e-12)


def test_excitation_grows_with_eta(data):
    small = hawkes.excitation_at_events(data, hawkes.branching_matrix(0.05, 0.0), 1 / 5.0)
    big = hawkes.excitation_at_events(data, hawkes.branching_matrix(0.25, 0.0), 1 / 5.0)
    assert np.all(big >= small - 1e-12)
    assert big.sum() > small.sum()
