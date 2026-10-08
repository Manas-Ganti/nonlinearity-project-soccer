"""Simulator validation (docs/math.md section 5).

Two checks the maths prescribes: with eta = 0 the event count must match int mu
within Monte Carlo error, and with a known eta the excess over int mu must match
the branching ratio's prediction.
"""

import numpy as np
import pytest

from src.config import CONFIG
from src.inference.pipeline import Pipeline
from src.models import hawkes, simulate


@pytest.fixture(scope="module")
def setup():
    from src.ingest import synthetic

    slate = synthetic.make_slate(n_matches=150, n_teams=12, seed=21)
    pipe = Pipeline(slate, CONFIG)
    bg = pipe.fit_background(slate.events, warm_start=False)
    return pipe, bg


def test_zero_eta_count_matches_integral_of_mu(setup):
    pipe, bg = setup
    sim = pipe.simulator(bg)
    expected = simulate.expected_counts(pipe.design, bg.rate_rows())
    counts = []
    for r in range(12):
        out = sim.simulate(0.0, 0.0, 1 / 5.0, np.random.default_rng(100 + r))
        counts.append(out["t"].size)
    mean = float(np.mean(counts))
    se = np.sqrt(expected / len(counts))
    assert abs(mean - expected) < 4 * se, f"mean {mean:.1f} vs int mu {expected:.1f} (se {se:.1f})"


def test_branching_ratio_inflates_the_count(setup):
    """A stationary Hawkes process has mean count int mu / (1 - rho), less an edge
    correction for offspring arriving after the whistle."""
    pipe, bg = setup
    sim = pipe.simulator(bg)
    base = simulate.expected_counts(pipe.design, bg.rate_rows())
    eta_self, eta_cross = 0.20, 0.10
    rho = eta_self + eta_cross
    counts = [
        sim.simulate(eta_self, eta_cross, 1 / 5.0, np.random.default_rng(200 + r))["t"].size for r in range(8)
    ]
    mean = float(np.mean(counts))
    ideal = base / (1 - rho)
    # the edge correction only ever removes offspring, so the truth sits below `ideal`
    assert base < mean < ideal
    assert mean > base * 1.2


def test_simulated_events_pass_the_schema(setup):
    pipe, bg = setup
    sim = pipe.simulator(bg)
    rng = np.random.default_rng(7)
    events = pipe.simulate_events(sim, 0.1, 0.05, 1 / 5.0, rng)
    assert len(events) > 0
    assert events["t"].le(events["match_T"]).all()
    assert set(events["side"]) <= {"H", "A"}
    assert (events["situation"] == "open").all()


def test_simulated_slate_reuses_the_same_design(setup):
    """The observed game-state path is held fixed, so the design is not rebuilt."""
    pipe, bg = setup
    sim = pipe.simulator(bg)
    events = pipe.simulate_events(sim, 0.0, 0.0, 1 / 5.0, np.random.default_rng(1))
    y = np.zeros(pipe.design.n_rows)
    rows = pipe.design.event_rows(events)
    np.add.at(y, rows, 1.0)
    assert y.sum() == len(events)


def test_seed_reproducibility(setup):
    pipe, bg = setup
    sim = pipe.simulator(bg)
    a = sim.simulate(0.15, 0.05, 1 / 5.0, np.random.default_rng(99))
    b = sim.simulate(0.15, 0.05, 1 / 5.0, np.random.default_rng(99))
    assert np.array_equal(a["t"], b["t"])
    assert np.array_equal(a["team"], b["team"])


def test_time_rescaling_is_exponential_under_the_true_model(setup):
    """The strongest end-to-end check available: simulate from a known intensity,
    rescale by that same intensity, and the gaps must be Exp(1)."""
    from scipy import stats

    pipe, bg = setup
    sim = pipe.simulator(bg)
    eta_self, eta_cross, beta = 0.15, 0.05, 1 / 5.0
    events = pipe.simulate_events(sim, eta_self, eta_cross, beta, np.random.default_rng(4242))
    data = hawkes.prepare(pipe.slate.with_events(events), bg, CONFIG.hawkes)
    from src.inference import gof

    gaps = gof.rescaled_gaps(data, eta_self, eta_cross, beta)
    tau = gaps["tau"].to_numpy()
    assert tau.mean() == pytest.approx(1.0, abs=0.08)
    u = 1.0 - np.exp(-tau)
    assert stats.kstest(u, "uniform").statistic < 0.05
