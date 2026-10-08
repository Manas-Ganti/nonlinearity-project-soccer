import numpy as np
import pandas as pd
import pytest

from src.config import CONFIG, DedupConfig
from src.features import dedup, jitter
from src.ingest import schema


def _events(rows):
    base = {
        "match_id": "m",
        "source": "test",
        "league": "L",
        "season": 2015,
        "date": pd.Timestamp("2015-01-01"),
        "side": "H",
        "team": "A",
        "opponent": "B",
        "second": np.nan,
        "xg": 0.1,
        "situation": "open",
        "is_goal": False,
        "score_diff": 0,
        "red_diff": 0,
        "match_T": 95.0,
        "n_merged": 1,
        "red_cards_known": False,
    }
    out = []
    for r in rows:
        d = dict(base)
        d.update(r)
        d.setdefault("minute", int(d["t"]))
        out.append(d)
    return schema.validate(pd.DataFrame(out))


def test_collapses_a_rebound_run():
    ev = _events([{"t": 10.0}, {"t": 10.05}, {"t": 10.1}, {"t": 40.0}])  # 3s apart
    out, stats = dedup.apply_dedup(ev, DedupConfig(threshold_seconds=5.0, max_removed_fraction=1.0))
    assert len(out) == 2
    assert out["n_merged"].tolist() == [3, 1]
    assert out["t"].tolist() == [10.0, 40.0]  # the run keeps the FIRST shot's time
    assert stats["removed_fraction"] == pytest.approx(0.5)


def test_combined_mark_is_probability_of_at_least_one_goal():
    ev = _events([{"t": 10.0, "xg": 0.2}, {"t": 10.02, "xg": 0.5}])
    out, _ = dedup.apply_dedup(ev, DedupConfig(threshold_seconds=5.0, max_removed_fraction=1.0))
    assert out["xg"].iloc[0] == pytest.approx(1 - 0.8 * 0.5)
    assert 0.0 <= out["xg"].iloc[0] <= 1.0


def test_goal_survives_a_collapse():
    ev = _events([{"t": 10.0}, {"t": 10.02, "is_goal": True}])
    out, _ = dedup.apply_dedup(ev, DedupConfig(threshold_seconds=5.0, max_removed_fraction=1.0))
    assert bool(out["is_goal"].iloc[0])


def test_penalties_are_never_collapsed():
    ev = _events([{"t": 10.0}, {"t": 10.02, "situation": "penalty", "xg": 0.76}, {"t": 10.04}])
    out, _ = dedup.apply_dedup(
        ev, DedupConfig(threshold_seconds=5.0, protect_penalties=True, max_removed_fraction=1.0)
    )
    assert len(out) == 3


def test_opposing_teams_are_never_collapsed():
    ev = _events([{"t": 10.0}, {"t": 10.01, "side": "A", "team": "B", "opponent": "A"}])
    out, _ = dedup.apply_dedup(ev, DedupConfig(threshold_seconds=30.0, max_removed_fraction=1.0))
    assert len(out) == 2


def test_over_aggressive_threshold_is_refused():
    """CLAUDE.md build step 2: above ~15% removed, the rule is eating real chances."""
    rng = np.random.default_rng(0)
    ev = _events([{"t": float(t)} for t in np.sort(rng.uniform(0, 90, 200))])
    with pytest.raises(ValueError, match="de-duplication removed"):
        dedup.apply_dedup(ev, DedupConfig(threshold_seconds=600.0))


def test_threshold_calibration_finds_a_planted_spike():
    rng = np.random.default_rng(1)
    independent = rng.exponential(240.0, 8000)
    mechanical = rng.uniform(0.5, 4.0, 700)
    gaps = np.concatenate([independent, mechanical])
    res = dedup.choose_threshold(gaps)
    assert 3.0 <= res["threshold_seconds"] <= 12.0


def test_jitter_stays_inside_the_recorded_minute():
    ev = _events([{"t": 10.0}, {"t": 10.0}, {"t": 55.0}])
    out = jitter.jitter_events(ev, seed=1)
    assert np.all(np.floor(out["t"]) == out["minute"])
    assert out["t"].is_unique


def test_jitter_is_order_preserving():
    ev = _events([{"t": 10.0, "xg": 0.11}, {"t": 10.0, "xg": 0.22}, {"t": 10.0, "xg": 0.33}])
    out = jitter.jitter_events(ev, seed=5)
    assert out["xg"].tolist() == [0.11, 0.22, 0.33]
    assert out["t"].is_monotonic_increasing


def test_jitter_leaves_second_resolution_alone():
    ev = _events([{"t": 10.25, "second": 15.0}, {"t": 20.5, "second": 30.0}])
    out = jitter.jitter_events(ev, seed=2)
    assert out["t"].tolist() == [10.25, 20.5]


def test_jitter_draws_differ_but_are_reproducible():
    ev = _events([{"t": 10.0}, {"t": 10.0}])
    a = jitter.jitter_events(ev, seed=1)["t"].tolist()
    b = jitter.jitter_events(ev, seed=1)["t"].tolist()
    c = jitter.jitter_events(ev, seed=2)["t"].tolist()
    assert a == b and a != c


def test_config_threshold_is_the_calibrated_one():
    """The default must be whatever `make calibrate-dedup` measured, not a guess --
    and one rule is applied to the pooled slate, so every provider must agree."""
    import json
    from pathlib import Path

    paths = sorted(Path("results").glob("dedup_calibration_*.json"))
    if not paths:
        pytest.skip("calibration has not been run in this checkout")
    for path in paths:
        measured = json.loads(path.read_text())["payload"]["threshold_seconds"]
        assert CONFIG.dedup.threshold_seconds == pytest.approx(measured), path.name


def test_vectorised_runs_match_a_reference_loop():
    """`_run_ids` is vectorised across the whole slate for speed. If it ever disagrees
    with the obvious per-group loop, every de-duplicated dataset in the project is
    wrong, so the two are compared directly on a slate with plenty of edge cases."""
    from src.ingest import synthetic

    slate = synthetic.make_slate(n_matches=40, n_teams=8, seed=99)
    ev = slate.events.copy()
    # sprinkle penalties so the protection rule is exercised
    rng = np.random.default_rng(0)
    ev = ev.reset_index(drop=True)
    ev.loc[rng.choice(len(ev), size=len(ev) // 20, replace=False), "situation"] = "penalty"
    ev = schema.validate(ev)

    threshold_min = 45.0 / 60.0  # deliberately loose, to force long runs
    df = ev.sort_values(["match_id", "team", "t"], kind="stable").reset_index(drop=True)
    fast = dedup._run_ids(df, threshold_min, protect_penalties=True)

    slow, run, prev_key, prev_t, prev_pen = [], -1, None, None, None
    for row in df.itertuples(index=False):
        key = (row.match_id, row.team)
        pen = row.situation == "penalty"
        if key != prev_key or (row.t - prev_t) > threshold_min or pen or prev_pen:
            run += 1
        slow.append(run)
        prev_key, prev_t, prev_pen = key, row.t, pen

    assert np.array_equal(fast, np.array(slow))
