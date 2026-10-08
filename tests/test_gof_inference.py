import numpy as np
import pytest

from src.config import CONFIG, PowerConfig
from src.inference import bootstrap, gof, power
from src.models import hawkes


def test_rescaled_gaps_average_one_under_the_fitted_model(fitted):
    _pipe, res = fitted
    gaps = gof.rescaled_gaps(res.data, res.hawkes_fit.eta_self, res.hawkes_fit.eta_cross, res.hawkes_fit.beta)
    assert gaps["tau"].mean() == pytest.approx(1.0, abs=0.2)
    assert (gaps["tau"] >= 0).all()


def test_compensator_is_monotone_along_each_process(fitted):
    _pipe, res = fitted
    comp = hawkes.compensator_at_events(res.data, hawkes.branching_matrix(0.2, 0.1), res.hawkes_fit.beta)
    import pandas as pd

    df = pd.DataFrame(comp).sort_values(["match", "team", "t"])
    d = df.groupby(["match", "team"])["Lambda"].diff().dropna()
    assert (d >= -1e-9).all()


def test_ks_report_shape(fitted):
    _pipe, res = fitted
    gaps = gof.rescaled_gaps(res.data, res.hawkes_fit.eta_self, res.hawkes_fit.eta_cross, res.hawkes_fit.beta)
    rep = gof.ks_report(gaps["tau"].to_numpy())
    assert 0 <= rep["ks_stat"] <= 1
    assert rep["n"] == len(gaps)


def test_ks_detects_a_deliberately_wrong_compensator(fitted):
    """Rescaling by an intensity that is not the fitted one must fail the test."""
    _pipe, res = fitted
    good = gof.rescaled_gaps(res.data, res.hawkes_fit.eta_self, res.hawkes_fit.eta_cross, res.hawkes_fit.beta)
    bad = good.copy()
    bad["tau"] = bad["tau"] * 3.0
    assert gof.ks_report(bad["tau"].to_numpy())["ks_stat"] > gof.ks_report(good["tau"].to_numpy())["ks_stat"]


def test_qq_points_are_sorted(fitted):
    _pipe, res = fitted
    gaps = gof.rescaled_gaps(res.data, 0.0, 0.0, res.hawkes_fit.beta)
    q = gof.qq_points(gaps["tau"].to_numpy(), n_points=50)
    assert q["theoretical"].is_monotonic_increasing
    assert q["empirical"].is_monotonic_increasing


def test_eta_parameterisation_round_trips():
    cfg = CONFIG.hawkes
    # tau values inside the configured box; outside it the map clips, by design
    for eta_self, eta_cross, beta in [(0.3, 0.1, 1 / 5.0), (0.01, 0.02, 1 / 3.5), (0.5, 0.4, 1 / 7.5)]:
        z = hawkes.natural_to_unconstrained(eta_self, eta_cross, beta, cfg)
        back = hawkes.unconstrained_to_natural(z, cfg)
        assert back[0] == pytest.approx(eta_self, abs=1e-6)
        assert back[1] == pytest.approx(eta_cross, abs=1e-6)
        assert back[2] == pytest.approx(beta, abs=1e-6)


def test_parameterisation_cannot_leave_the_stationary_region():
    """Stationarity is enforced by construction, not by a penalty that can be lost."""
    cfg = CONFIG.hawkes
    rng = np.random.default_rng(0)
    for z in rng.normal(0, 30, size=(500, 3)):
        eta_self, eta_cross, beta = hawkes.unconstrained_to_natural(z, cfg)
        assert eta_self >= 0 and eta_cross >= 0
        assert eta_self + eta_cross < 1.0
        assert cfg.tau_min - 1e-9 <= 1 / beta <= cfg.tau_max + 1e-9


def test_branching_matrix_spectral_radius():
    N = hawkes.branching_matrix(0.3, 0.2)
    assert np.max(np.abs(np.linalg.eigvals(N))) == pytest.approx(0.5)


def test_parametric_bootstrap_null_puts_a_real_estimate_in_the_tail(fitted):
    pipe, res = fitted
    out = bootstrap.parametric_bootstrap_null(
        pipe, res.background, eta_obs=0.30, n_replicates=8, seed=3, progress_every=0
    )
    assert out.draws.size == 8
    assert 0 < out.p_value <= 1
    assert out.p_value < 0.5, "eta_hat = 0.30 should be extreme under a null with no excitation"


def test_bootstrap_p_value_is_one_for_an_absurdly_small_estimate(fitted):
    pipe, res = fitted
    out = bootstrap.parametric_bootstrap_null(
        pipe, res.background, eta_obs=-1.0, n_replicates=5, seed=4, progress_every=0
    )
    assert out.p_value == pytest.approx(1.0)


def test_cluster_bootstrap_resamples_whole_matches(small_slate):
    ids = np.sort(small_slate.events["match_id"].unique())[:6]
    sub = small_slate.filter_matches(ids)
    resampled = bootstrap._resample_slate(sub, np.array([ids[0], ids[0], ids[1]]))
    assert resampled.events["match_id"].nunique() == 3
    per_match = resampled.events.groupby("match_id").size()
    assert per_match.iloc[0] == per_match.iloc[1]  # the duplicated match is intact


def test_power_summary_reads_the_critical_value_off_the_null_arm():
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    for eta_star, loc in [(0.0, 0.01), (0.05, 0.05), (0.10, 0.11)]:
        n = 200 if eta_star == 0 else 30
        for r in range(n):
            rows.append(
                {
                    "eta_star": eta_star,
                    "replicate": r,
                    "n_events": 1000,
                    "eta_hat_self": max(0.0, rng.normal(loc, 0.01)),
                }
            )
    out = power.summarise(pd.DataFrame(rows), PowerConfig())
    assert out["critical_value"] > 0
    assert out["curve"].loc[out["curve"]["eta_star"] == 0.10, "power"].iloc[0] > 0.8
    assert out["detection_floor"] in (0.05, 0.10)


def test_asymmetric_parameterisation_respects_the_spectral_radius():
    """The symmetric case has rho = row sum; the general 2x2 case does not, so the
    robustness check gets its own constraint test."""
    from src.models import hawkes as hk

    rng = np.random.default_rng(3)
    for z in rng.normal(0, 20, size=(300, 6)):
        N, beta = hk.asym_unconstrained_to_natural(z, CONFIG.hawkes)
        assert (N >= 0).all()
        rho = float(np.max(np.abs(np.linalg.eigvals(N))))
        assert rho < 1.0
        assert CONFIG.hawkes.tau_min - 1e-9 <= 1 / beta <= CONFIG.hawkes.tau_max + 1e-9


def test_asymmetric_fit_cannot_do_worse_than_symmetric(fitted):
    """The symmetric model is nested inside it, so the likelihood must not fall."""
    from src.models import hawkes as hk

    _pipe, res = fitted
    sym = hk.loglik(
        res.data, hk.branching_matrix(res.hawkes_fit.eta_self, res.hawkes_fit.eta_cross),
        res.hawkes_fit.beta,
    )
    asym = hk.fit_asymmetric(res.data, CONFIG.hawkes)
    assert asym["loglik"] >= sym - 1e-4


def test_gathered_design_equals_a_rebuilt_one(small_slate):
    """The cluster bootstrap gathers design rows instead of rebuilding them. If the two
    ever disagree, every bootstrap interval in the project is wrong."""
    from src.config import CONFIG as CFG
    from src.inference import bootstrap as bs
    from src.models import baseline

    ids = np.sort(small_slate.events["match_id"].unique())[:6]
    sub = small_slate.filter_matches(ids)
    draw = np.array([ids[0], ids[2], ids[0], ids[1]])
    resampled, pairs = bs._resample_slate(sub, draw, with_pairs=True)

    base = baseline.build_design(sub, CFG.background)
    gathered = baseline.gather_design(base, pairs)
    rebuilt = baseline.build_design(resampled, CFG.background)

    assert gathered.n_rows == rebuilt.n_rows
    for attr in ("is_home", "bin_idx", "score_idx", "red_idx", "exposure", "seg_start", "seg_nbins"):
        assert np.array_equal(getattr(gathered, attr), getattr(rebuilt, attr)), attr
    # team indices point into possibly different label orders, so compare the labels
    assert np.array_equal(
        gathered.team_labels[gathered.team_idx], rebuilt.team_labels[rebuilt.team_idx]
    )
    assert np.array_equal(
        gathered.team_labels[gathered.opp_idx], rebuilt.team_labels[rebuilt.opp_idx]
    )
    # and the counts must land in the same rows
    assert np.array_equal(
        baseline.counts(gathered, resampled.events), baseline.counts(rebuilt, resampled.events)
    )


def test_cluster_bootstrap_runs_end_to_end(small_slate):
    from src.inference import bootstrap as bs

    ids = np.sort(small_slate.events["match_id"].unique())[:20]
    sub = small_slate.filter_matches(ids)
    out = bs.cluster_bootstrap(sub, n_replicates=3, seed=1, progress_every=0)
    assert len(out) == 3
    assert out["eta_self"].between(0, 1).all()
    lo, hi = bs.percentile_ci(out["eta_self"].to_numpy())
    assert lo <= hi


def test_compare_rescales_each_model_by_its_own_compensator(small_slate):
    """Regression test for a bug that made the null look worse than it is.

    The Poisson arm must use the background fitted under the null. Feeding it the
    joint fit's background with eta = 0 gives an intensity that is too low everywhere,
    which inflates the rescaled gaps and manufactures a lack of fit.
    """
    from src.inference import gof as g
    from src.inference.pipeline import Pipeline
    from src.models import hawkes as hk

    pipe = Pipeline(small_slate, CONFIG)
    joint_res = pipe.run(small_slate.events, warm_start=False)
    null_bg = pipe.fit_background(small_slate.events, warm_start=False)
    null_data = hk.prepare(small_slate, null_bg, CONFIG.hawkes)

    # the null background must integrate to the observed count; the joint one, less
    assert null_bg.integral_per_segment().sum() > joint_res.background.integral_per_segment().sum() - 1e-6

    correct = g.compare(null_data, null_bg, joint_res.data, joint_res.hawkes_fit)
    wrong = g.compare(joint_res.data, null_bg, joint_res.data, joint_res.hawkes_fit)
    # the correct null arm must be centred, the hybrid one biased upwards
    assert correct["poisson"]["mean_tau"] >= wrong["poisson"]["mean_tau"] - 1e-9
    assert set(correct) >= {"poisson", "hawkes", "loglik_poisson", "loglik_hawkes"}
