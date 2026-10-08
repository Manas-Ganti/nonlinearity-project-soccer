"""Build step 5: plant eta = 0.3, run the full pipeline, confirm it comes back.

CLAUDE.md: "If this fails, everything downstream is meaningless." This runs on a
small synthetic slate so it can live in the test suite; the full-scale version is
`make recovery`, which is the gate the real analysis has to pass.
"""

import numpy as np
import pytest

from src.config import CONFIG
from src.inference import power
from src.inference.pipeline import Pipeline
from src.ingest import synthetic


@pytest.fixture(scope="module")
def pipe_and_bg():
    slate = synthetic.make_slate(n_matches=400, n_teams=16, seed=41)
    pipe = Pipeline(slate, CONFIG)
    bg = pipe.fit_background(slate.events, warm_start=False)
    return pipe, bg


def test_recovers_a_planted_branching_ratio(pipe_and_bg):
    pipe, bg = pipe_and_bg
    res = power.recovery_check(
        pipe, bg, eta_plant=0.30, eta_cross=0.05, beta=1 / 5.0, n_replicates=4, seed=17, tolerance=0.08
    )
    assert res["passed"], res


def test_recovers_the_kernel_timescale(pipe_and_bg):
    pipe, bg = pipe_and_bg
    res = power.recovery_check(
        pipe, bg, eta_plant=0.30, eta_cross=0.05, beta=1 / 6.0, n_replicates=4, seed=23
    )
    assert res["mean_tau_hat"] == pytest.approx(6.0, rel=0.5)


def test_zero_excitation_is_not_manufactured(pipe_and_bg):
    """The two-stage fit is biased *towards* excitation. Measure it; do not assume
    it away."""
    pipe, bg = pipe_and_bg
    res = power.recovery_check(
        pipe, bg, eta_plant=0.0, eta_cross=0.0, beta=1 / 5.0, n_replicates=6, seed=29, tolerance=0.05
    )
    assert res["mean_eta_hat_self"] < 0.05, res


def test_detection_floor_reads_off_the_power_curve():
    import pandas as pd

    curve = pd.DataFrame({"eta_star": [0.0, 0.02, 0.05, 0.10, 0.15], "power": [0.05, 0.20, 0.55, 0.90, 0.99]})
    out = power.detection_floor(curve, 0.80)
    assert out["floor"] == 0.10
    assert 0.05 < out["interpolated"] < 0.10


def test_detection_floor_is_none_when_nothing_reaches_target():
    import pandas as pd

    curve = pd.DataFrame({"eta_star": [0.0, 0.1], "power": [0.05, 0.4]})
    assert power.detection_floor(curve, 0.80)["floor"] is None


def test_minute_clock_degradation_matches_what_understat_does():
    """The degradation applied inside the power loop must be the same transformation
    the real Understat feed imposes: integer minutes, then the same de-duplication and
    jitter the pipeline applies to real data."""
    from src.inference.power import degrade_to_minute_clock
    from src.ingest import synthetic

    slate = synthetic.make_slate(n_matches=30, n_teams=8, seed=13)
    fine = slate.events.copy()
    fine["second"] = (fine["t"] % 1.0) * 60.0  # pretend the source had a real clock
    coarse = degrade_to_minute_clock(fine, CONFIG, seed=1)

    assert len(coarse) < len(fine), "rounding must collapse at least some same-minute pairs"
    assert np.all(np.floor(coarse["t"]) == coarse["minute"])
    key = coarse["match_id"].astype(str) + "|" + coarse["t"].astype(str)
    assert key.is_unique, "jitter must leave the likelihood well defined"
    # no event may be invented, and none may leave its match
    assert set(coarse["match_id"]) <= set(fine["match_id"])
    assert coarse["n_merged"].sum() == len(fine)


def test_minute_clock_costs_power():
    """The whole point of measuring Understat's floor separately: rounding the clock
    must make a planted effect harder to see, not easier."""
    from src.inference.pipeline import Pipeline
    from src.inference.power import degrade_to_minute_clock
    from src.ingest import synthetic

    slate = synthetic.make_slate(n_matches=250, n_teams=12, seed=57)
    pipe = Pipeline(slate, CONFIG)
    bg = pipe.fit_background(slate.events, warm_start=False)
    sim = pipe.simulator(bg)
    events = pipe.simulate_events(sim, 0.30, 0.05, 1 / 5.0, np.random.default_rng(8))
    events = events.assign(second=(events["t"] % 1.0) * 60.0)

    fine = pipe.run(events, warm_start=False).hawkes_fit.eta_self
    coarse_events = degrade_to_minute_clock(events, CONFIG, seed=2)
    coarse = pipe.run(coarse_events, warm_start=False).hawkes_fit.eta_self
    assert coarse < fine, f"minute rounding should shrink eta_hat: {coarse:.4f} vs {fine:.4f}"
