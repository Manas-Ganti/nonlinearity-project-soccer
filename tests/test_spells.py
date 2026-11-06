"""Models S and G (docs/spec_spells.md section 6.1): the spell indicator, the split
recursion, the gamma background, the simulator, and the recovery gate in miniature."""

import sys

import numpy as np
import pandas as pd
import pytest

from src.config import CONFIG
from src.features import spells
from src.inference import spells as sp_inf
from src.ingest import synthetic
from src.models import baseline, hawkes, split_hawkes
from src.models.split_simulate import score_class


def synthetic_possessions(slate, seed=0, p_danger=0.55, mean_s=40.0):
    """Alternating possessions over each match, two periods split at T/2."""
    rng = np.random.default_rng(seed)
    rows = []
    for mid, g in slate.events.groupby("match_id"):
        T = float(g["match_T"].iloc[0])
        for period, (a, b) in enumerate(((0.0, T / 2), (T / 2, T))):
            t = a
            side = "H" if rng.random() < 0.5 else "A"
            while t < b:
                dur = rng.exponential(mean_s / 60.0) + 0.05
                nxt = min(t + dur, b)
                rows.append(
                    {
                        "match_id": mid,
                        "period": period,
                        "side": side,
                        "t_start": t,
                        "t_next": nxt,
                        "t_ft": t + 0.5 * (nxt - t) if rng.random() < p_danger else np.nan,
                        "match_T": T,
                    }
                )
                t = nxt
                side = "A" if side == "H" else "H"
    return pd.DataFrame(rows)


def _poss(rows):
    """rows: (period, side, t_start, t_next, dangerous)"""
    return pd.DataFrame(
        [
            {
                "match_id": "m",
                "period": p,
                "side": s,
                "t_start": a,
                "t_next": b,
                "t_ft": a if d else np.nan,
                "match_T": 20.0,
            }
            for p, s, a, b, d in rows
        ]
    )


# ---------------------------------------------------------------- indicator


def test_z_counts_only_earlier_possessions_and_resets_each_half():
    rows = []
    t = 0.0
    for _ in range(5):  # H dangerous five times, A never
        rows += [(0, "H", t, t + 0.5, True), (0, "A", t + 0.5, t + 1.0, False)]
        t += 1.0
    rows += [(1, "H", 10.0, 10.5, True), (1, "A", 10.5, 11.0, False)]
    st = spells.build(_poss(rows), need=4, window=5)
    zo, zp = st.at(["m"] * 4, ["H"] * 4, [3.49, 3.5, 4.2, 10.2])
    # H's possessions start at 0, 1, 2, 3, 4 (each dangerous)
    assert zo[0] == 0  # inside H's 4th possession (3.0-3.5): only 3 earlier H possessions
    assert zo[1] == 1  # A's possession from 3.5: H's four earlier possessions all dangerous
    assert zo[2] == 1 and zp[2] == 0
    assert zo[3] == 0  # new half: nothing carried over
    # a shot inside H's 4th possession (t=3.2) does not count that possession
    assert st.at(["m"], ["H"], [3.2])[0][0] == 0


def test_z_never_uses_the_possession_containing_t():
    rng = np.random.default_rng(3)
    slate = synthetic.make_slate(n_matches=5, n_teams=6, seed=3)
    poss = synthetic_possessions(slate, seed=4)
    st = spells.build(poss)
    # recompute Z for every possession start from strictly earlier own possessions
    for (mid, _period), h in poss.sort_values("t_start").groupby(["match_id", "period"]):
        for j in range(len(h)):
            for k in ("H", "A"):
                prev = h.iloc[:j]
                prev = prev[prev["side"] == k].tail(5)
                want = int(prev["t_ft"].notna().sum() >= 4)
                t = h["t_start"].iloc[j] + 1e-9
                got = st.at([mid], [k], [t])[0][0]
                assert got == want
    assert rng is not None


def test_row_weights_partition_the_exposure():
    slate = synthetic.make_slate(n_matches=6, n_teams=6, seed=5)
    st = spells.build(synthetic_possessions(slate, seed=1))
    design = baseline.build_design(slate, CONFIG.background)
    w = st.row_weights(design)
    assert np.allclose(w.sum(axis=1), design.exposure, atol=1e-9)
    assert (w >= -1e-12).all()
    # spell time for H in w equals the step function's own integral
    share = spells.prevalence(st)["share_time_z1"]
    own_on = (w[:, 2] + w[:, 3]).sum()
    assert own_on / design.exposure.sum() == pytest.approx(share, rel=1e-9)


# ---------------------------------------------------------------- recursion


@pytest.fixture(scope="module")
def small():
    slate = synthetic.make_slate(n_matches=8, n_teams=6, seed=11)
    st = spells.build(synthetic_possessions(slate, seed=2))
    pipe = sp_inf.SpellPipeline(slate, "S", spell=st)
    bg, sd = pipe._prepare(slate.events)
    return slate, st, pipe, bg, sd


@pytest.mark.parametrize(
    ("eta_own", "eta_cross", "beta"),
    [([0.0, 0.0], 0.0, 0.2), ([0.1, 0.4], 0.05, 0.2), ([0.35, 0.02], 0.3, 1 / 3), ([0.0, 0.6], 0.1, 1 / 8)],
)
def test_split_recursion_matches_naive(small, eta_own, eta_cross, beta):
    *_, sd = small
    assert split_hawkes.loglik(sd, eta_own, eta_cross, beta) == pytest.approx(
        split_hawkes.naive_loglik(sd, eta_own, eta_cross, beta), abs=1e-8
    )


def test_three_classes_match_naive(small):
    slate, *_ = small
    pipe = sp_inf.SpellPipeline(slate, "G")
    _, sd = pipe._prepare(slate.events)
    e = [0.3, 0.05, 0.15]
    assert split_hawkes.loglik(sd, e, 0.07, 0.25) == pytest.approx(
        split_hawkes.naive_loglik(sd, e, 0.07, 0.25), abs=1e-8
    )


@pytest.mark.parametrize("model", ["S", "G"])
def test_equal_split_is_the_headline_likelihood(small, model):
    slate, st, _, bg, _ = small
    pipe = sp_inf.SpellPipeline(slate, model, spell=st, gamma=False)
    _, sd = pipe._prepare(slate.events)
    ref = hawkes.prepare(slate, bg, CONFIG.hawkes)
    for e, ec, b in ((0.2, 0.1, 0.2), (0.05, 0.3, 1 / 7)):
        got = split_hawkes.loglik(sd, np.full(pipe.n_class, e), ec, b)
        assert got == pytest.approx(hawkes.loglik(ref, hawkes.branching_matrix(e, ec), b), abs=1e-8)


def test_gamma_zero_leaves_the_background_unchanged(small):
    _, _, pipe, bg, sd = small
    _, e_eff, _, _log_mu_ev, int_mu = split_hawkes._background(bg.params, np.zeros(2), pipe.design, sd)
    assert np.allclose(e_eff, pipe.design.exposure)
    assert int_mu == pytest.approx(float((bg.rate_rows() * pipe.design.exposure).sum()))


def test_joint_gradient_matches_finite_differences(small):
    _, _, pipe, bg, sd = small
    cfg = CONFIG.hawkes
    z = split_hawkes.to_unconstrained([0.1, 0.25], 0.05, 5.0, cfg)
    p = np.concatenate([bg.params, [0.3, -0.2], z])
    _, g = split_hawkes.objective(p, pipe.design, sd, cfg, True)
    rng = np.random.default_rng(0)
    n_mu = baseline.n_params(pipe.design)
    for i in [0, 1, 5, n_mu, n_mu + 1]:  # some background terms, then both gammas
        h = 1e-6
        pp, pm = p.copy(), p.copy()
        pp[i] += h
        pm[i] -= h
        fd = (
            split_hawkes.objective(pp, pipe.design, sd, cfg, True)[0]
            - split_hawkes.objective(pm, pipe.design, sd, cfg, True)[0]
        ) / (2 * h)
        assert g[i] == pytest.approx(fd, rel=1e-4, abs=1e-5)
    assert rng is not None


def test_parameterisation_round_trips_and_is_stationary():
    cfg = CONFIG.hawkes
    z = split_hawkes.to_unconstrained([0.1, 0.4], 0.2, 6.0, cfg)
    eo, ec, b = split_hawkes.to_natural(z, cfg, 2)
    assert np.allclose(eo, [0.1, 0.4], atol=1e-6) and ec == pytest.approx(0.2) and 1 / b == pytest.approx(6.0)
    for zz in np.random.default_rng(1).normal(0, 8, size=(200, 4)):
        eo, ec, _ = split_hawkes.to_natural(zz, cfg, 2)
        assert eo.max() + ec < 1.0


def test_score_class():
    assert list(score_class(np.array([0, -1, -3, 1, 2]))) == [0, 1, 1, 2, 2]


# ---------------------------------------------------------------- simulator


def test_simulator_at_zero_eta_matches_the_gamma_background():
    slate = synthetic.make_slate(n_matches=60, n_teams=10, seed=9)
    st = spells.build(synthetic_possessions(slate, seed=3))
    pipe = sp_inf.SpellPipeline(slate, "S", spell=st)
    null = pipe.run(slate.events, kernel=False).fit
    null.gamma_own, null.gamma_opp = 0.6, -0.4  # make gamma matter
    sim = pipe.simulator(null)
    rate = split_hawkes.rate_rows(null, pipe.design)
    expected = float((rate * (pipe.w @ split_hawkes._state_factor(np.array([0.6, -0.4])))).sum())
    n = [sim.simulate_split(np.zeros(2), 0.0, 0.2, np.random.default_rng(s))["t"].size for s in range(30)]
    assert np.mean(n) == pytest.approx(expected, rel=0.03)
    # and the null fit gets gamma back from a simulated slate
    ev = pipe.simulate_events(sim, np.zeros(2), 0.0, 0.2, np.random.default_rng(99))
    refit = pipe.run(ev, kernel=False).fit
    assert refit.gamma_own == pytest.approx(0.6, abs=0.15)
    assert refit.gamma_opp == pytest.approx(-0.4, abs=0.15)


def test_simulated_classes_agree_with_the_refit_annotation():
    slate = synthetic.make_slate(n_matches=20, n_teams=8, seed=4)
    st = spells.build(synthetic_possessions(slate, seed=8))
    for model in ("S", "G"):
        pipe = sp_inf.SpellPipeline(slate, model, spell=st)
        null = pipe.run(slate.events, kernel=False).fit
        sim = pipe.simulator(null)
        raw = sim.simulate_split(np.full(pipe.n_class, 0.2), 0.05, 0.2, np.random.default_rng(1))
        ev = sim.to_events(raw, pipe.template, np.random.default_rng(2))
        cls, *_ = pipe.classes(pipe._sorted(ev))
        assert (cls == raw["cls"]).all()


# ---------------------------------------------------------------- gates


@pytest.fixture(scope="module")
def recovery_slate():
    slate = synthetic.make_slate(n_matches=400, n_teams=16, seed=41)
    st = spells.build(synthetic_possessions(slate, seed=6))
    return slate, st


@pytest.mark.parametrize("model", ["S", "G"])
def test_recovery_gate_in_miniature(recovery_slate, model):
    slate, st = recovery_slate
    pipe = sp_inf.SpellPipeline(slate, model, spell=st)
    null = pipe.run(slate.events, kernel=False).fit
    res = sp_inf.recovery(pipe, null, n_replicates=3, seed=17, tolerance=0.10, workers=1)
    assert res["passed"], res["per_parameter"]
    assert res["n_unconverged"] == 0


def test_holm():
    h = sp_inf.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5, "e": 0.002})
    assert h["adjusted"]["e"] == pytest.approx(0.01)
    assert h["adjusted"]["a"] == pytest.approx(0.04)
    assert h["adjusted"]["c"] == pytest.approx(0.09)
    assert h["adjusted"]["b"] == pytest.approx(0.09)  # monotone
    assert h["reject"] == {"a": True, "b": False, "c": False, "d": False, "e": True}


def test_cli_end_to_end_on_a_synthetic_slate(tmp_path, monkeypatch):
    """recovery -> power -> fit -> holm through the real commands, tiny settings."""
    import types

    from src import runlog
    from src.cli import spell_cmds

    slate = synthetic.make_slate(n_matches=120, n_teams=10, seed=13)
    st = spells.build(synthetic_possessions(slate, seed=5))
    monkeypatch.setattr(spell_cmds, "RESULTS", tmp_path)
    monkeypatch.setattr(runlog, "RESULTS", tmp_path)
    monkeypatch.setattr(spell_cmds, "_inputs", lambda args, caller, **kw: (slate, st, {"synthetic": True}))
    for model in ("S", "G"):
        a = types.SimpleNamespace(
            source="wyscout",
            exclude_one_club=False,
            model=model,
            tau=5.0,
            seed=1,
            workers=1,
            replicates=2,
            null_replicates=4,
            bootstrap=3,
            cluster_bootstrap=3,
            no_sensitivities=True,
        )
        spell_cmds.cmd_spell_recovery(a)
        a.replicates = 1
        spell_cmds.cmd_spell_power(a)
        spell_cmds.cmd_spell_fit(a)
        fit = runlog.read_manifest(f"spell_fit_{model}_wyscout_dev")["payload"]
        assert set(fit["parametric_bootstrap"]["p_value"]) == set(sp_inf.LABELS[model])
        assert set(fit["floors"]) == set(sp_inf.LABELS[model])
    spell_cmds.cmd_spell_holm(types.SimpleNamespace(source="wyscout", exclude_one_club=False))
    h = runlog.read_manifest("spell_holm_SG_wyscout_dev")["payload"]
    assert set(h["adjusted"]) == {"eta_out", "eta_in", "eta_level", "eta_trail", "eta_lead"}


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Accelerate BLAS segfaults in forked children on macOS; runs at ARC's pytest gate",
)
def test_sensitivities_run_across_forked_workers(monkeypatch):
    """ARC job for 5e9522f: every spell-fit died in `_sensitivities` with workers > 1,
    because the (name, closure) jobs were pickled. workers=1 runs inline and hid it."""
    import types

    from src.cli import spell_cmds

    slate = synthetic.make_slate(n_matches=60, n_teams=8, seed=21)
    st = spells.build(synthetic_possessions(slate, seed=8))
    monkeypatch.setattr(spell_cmds, "_inputs", lambda args, caller, **kw: (slate, st, {"prevalence": {}}))
    a = types.SimpleNamespace(source="wyscout", exclude_one_club=False, model="S", workers=2)
    out = spell_cmds._sensitivities(a, slate, st)
    assert len(out["tau_profile"]) == len(sp_inf.TAU_PROFILE)
    errors = {k: v["error"] for k, v in out.items() if isinstance(v, dict) and "error" in v}
    errors |= {f"tau[{i}]": v["error"] for i, v in enumerate(out["tau_profile"]) if "error" in v}
    assert not errors, errors
    assert {"box_1_20", "no_gamma_confounded", "rule_3_of_5", "strict_possession_rule"} <= set(out)
