import numpy as np
import pytest

from src.config import CONFIG
from src.inference import secondary


def test_quality_regression_recovers_a_planted_relationship(fitted):
    """Plant xG that rises with the excitation state and psi1 must find it."""
    pipe, res = fitted
    events = pipe.slate.events.sort_values(["match_id", "t"], kind="stable").reset_index(drop=True)
    events = events[events["situation"] != "penalty"].reset_index(drop=True)
    from src.models import hawkes

    # eta_hat on this slate may be 0, which makes the excitation trace constant and the
    # planted relationship untestable. Use a definitely-nonzero kernel for the trace.
    probe = hawkes.HawkesFit(
        eta_self=0.25, eta_cross=0.05, beta=1 / 5.0, loglik=0.0, converged=True, n_iter=0,
        message="probe", n_events=res.data.n_events, n_matches=res.data.n_matches, method="probe",
    )
    exc = hawkes.excitation_at_events(res.data, hawkes.branching_matrix(0.25, 0.05), 1 / 5.0)
    planted = events.copy()
    rng = np.random.default_rng(0)
    planted["xg"] = np.clip(0.08 + 0.05 * exc + rng.normal(0, 0.01, len(planted)), 0.001, 0.999)
    out = secondary.quality_regression(res.data, planted, probe)
    assert out["psi1"] == pytest.approx(0.05, abs=0.01)
    assert out["n"] == len(planted)


def test_quality_regression_refuses_misaligned_inputs(fitted):
    pipe, res = fitted
    events = pipe.slate.events.iloc[:5]
    with pytest.raises(ValueError, match="not aligned"):
        secondary.quality_regression(res.data, events, res.hawkes_fit)


def test_state_dependent_eta_partitions_the_matches(small_slate):
    out = secondary.state_dependent_eta(small_slate, CONFIG, min_events=1_000_000)
    arms = [k for k in out if k != "note"]
    assert set(arms) == {"behind", "level", "ahead"}
    total = sum(out[a]["n_events"] for a in arms)
    assert total == len(small_slate.events)


def test_marked_excitation_profile_is_a_diagnostic(fitted):
    from src.models import hawkes

    pipe, res = fitted
    events = pipe.slate.events.sort_values(["match_id", "t"], kind="stable").reset_index(drop=True)
    events = events[events["situation"] != "penalty"].reset_index(drop=True)
    probe = hawkes.HawkesFit(
        eta_self=0.25, eta_cross=0.05, beta=1 / 5.0, loglik=0.0, converged=True, n_iter=0,
        message="probe", n_events=res.data.n_events, n_matches=res.data.n_matches, method="probe",
    )
    out = secondary.marked_excitation_profile(res.data, events, probe)
    assert -1.0 <= out["corr_xg_excitation"] <= 1.0
    assert out["mean_xg"] > 0

    zero = hawkes.HawkesFit(
        eta_self=0.0, eta_cross=0.0, beta=1 / 5.0, loglik=0.0, converged=True, n_iter=0,
        message="probe", n_events=res.data.n_events, n_matches=res.data.n_matches, method="probe",
    )
    degenerate = secondary.marked_excitation_profile(res.data, events, zero)
    assert degenerate["excitation_is_degenerate"] is True
    assert degenerate["corr_xg_excitation"] is None
