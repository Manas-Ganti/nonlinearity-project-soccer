"""Wyscout loader: the clock, the sides, the goal rules, and the scorelines.

The unit tests build tiny event lists by hand. The last test runs against the real
extraction when it is on disk and checks every reconstructed final score against
the official one in the match file -- the reconstruction is exact or it is wrong.
"""

import numpy as np
import pandas as pd
import pytest

from src.ingest import wyscout

META = {
    "match_id": 1,
    "competition": "Test League",
    "international": False,
    "season": 2017,
    "date": pd.Timestamp("2017-09-01"),
    "home_id": 10,
    "away_id": 20,
    "home_team": "Home",
    "away_team": "Away",
    "final_home": 1,
    "final_away": 1,
}


def _ev(period, sec, team, name, sub, tags=(), pos=None):
    return {
        "matchPeriod": period,
        "eventSec": float(sec),
        "teamId": team,
        "eventName": name,
        "subEventName": sub,
        "tags": [{"id": t} for t in tags],
        "positions": pos or [{"x": 88, "y": 50}],
    }


def test_periods_are_laid_end_to_end_and_sides_come_from_ids():
    events = [
        _ev("1H", 600, 10, "Shot", "Shot", tags=(402,)),
        _ev("1H", 2850, 20, "Pass", "Simple pass"),  # 47.5 min: bounds the half
        _ev("2H", 30, 20, "Shot", "Shot", tags=(401, 101)),
        _ev("2H", 2700, 10, "Pass", "Simple pass"),
        _ev("P", 10, 10, "Shot", "Penalty", tags=(101,)),  # shootout: dropped
    ]
    meta = dict(META, final_home=0, final_away=1)
    df = wyscout._extract_match(events, meta)
    shots = df[df["kind"] == "shot"].sort_values("t")
    assert len(shots) == 2
    assert shots["t"].iloc[0] == pytest.approx(10.0)
    assert shots["t"].iloc[1] == pytest.approx(47.5 + 0.5)
    assert list(shots["side"]) == ["H", "A"]
    assert list(shots["team"]) == ["Home", "Away"]
    assert df["match_T"].iloc[0] == pytest.approx(47.5 + 45.0)


def test_goal_tag_on_keeper_event_is_not_a_second_goal():
    events = [
        _ev("1H", 100, 10, "Shot", "Shot", tags=(101,)),
        _ev("1H", 102, 20, "Save attempt", "Reflexes", tags=(101,)),
    ]
    meta = dict(META, final_home=1, final_away=0)
    df = wyscout._extract_match(events, meta)
    assert (df["kind"] == "shot").sum() == 1
    assert "conceded_marker" not in set(df["kind"])
    assert "goal_conceded_unmatched" not in set(df["kind"])


def test_lost_shot_is_recovered_only_when_the_official_score_says_so():
    marker_only = [_ev("1H", 100, 20, "Save attempt", "Reflexes", tags=(101,))]
    # official says the home side scored once: the away keeper's marker is it
    df = wyscout._extract_match(marker_only, dict(META, final_home=1, final_away=0))
    assert list(df["kind"]) == ["goal_conceded_unmatched"]
    assert df["side"].iloc[0] == "A"  # recorded for the keeper's side; credited to H downstream
    # official says 0-0: the marker is a disallowed goal and must vanish
    df = wyscout._extract_match(marker_only, dict(META, final_home=0, final_away=0))
    assert df.empty or "goal_conceded_unmatched" not in set(df["kind"])


def test_own_goal_and_corner_goal_land_on_the_right_side():
    events = [
        _ev("1H", 100, 10, "Others on the ball", "Touch", tags=(102,)),  # home puts it in own net
        _ev("1H", 900, 20, "Free Kick", "Corner", tags=(101, 801)),  # away scores direct from corner
        _ev("2H", 100, 10, "Foul", "Foul", tags=(1701,)),
        _ev("2H", 200, 10, "Shot", "Shot", tags=(402,)),
    ]
    meta = dict(META, final_home=0, final_away=2)
    raw = wyscout._extract_match(events, meta)
    slate, _ = wyscout.build_slate(raw.assign(x=88.0, y=50.0))
    goals = slate.goals.sort_values("t")
    assert list(goals["side"]) == ["A", "A"]
    assert len(slate.cards) == 1 and slate.cards["side"].iloc[0] == "H"
    shot = slate.events.iloc[0]
    assert shot["score_diff"] == -2
    assert shot["red_diff"] == -1  # home lost a player before shooting


def test_location_xg_is_monotone_in_distance_and_pins_penalties():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(60, 99, n)
    y = rng.uniform(20, 80, n)
    dist, _ = wyscout.shot_geometry(x, y)
    p = 1 / (1 + np.exp(-(2.5 - 0.15 * dist)))
    shots = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "is_header": rng.random(n) < 0.2,
            "situation": "open",
            "is_goal": rng.random(n) < p,
        }
    )
    model = wyscout.fit_location_xg(shots)
    assert model["converged"]
    assert model["coef"]["log_dist"] < 0
    probe = pd.DataFrame(
        {
            "x": [95.0, 70.0, 95.0],
            "y": [50.0, 50.0, 50.0],
            "is_header": False,
            "situation": ["open", "open", "penalty"],
        }
    )
    xg = wyscout.apply_location_xg(probe, model)
    assert xg[0] > xg[1]
    assert xg[2] == pytest.approx(0.76)


@pytest.mark.skipif(not (wyscout.RAW_DIR / "events.parquet").exists(), reason="no Wyscout extraction on disk")
def test_real_wyscout_scorelines_are_exact():
    raw = wyscout.load_raw()
    slate, model = wyscout.build_slate(raw)
    rec = (
        slate.goals.groupby(["match_id", "side"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["H", "A"], fill_value=0)
    )
    official = wyscout.official_scores(raw).set_index("match_id")
    joined = official.join(rec, how="left").fillna(0)
    bad = joined[(joined["final_home"] != joined["H"]) | (joined["final_away"] != joined["A"])]
    assert bad.empty, f"{len(bad)} matches with a wrong reconstructed scoreline"
    ev = slate.events
    assert ev["match_id"].nunique() == 1941
    assert 18 < ev.groupby("match_id").size().mean() < 34
    assert 0.08 < ev["xg"].mean() < 0.15
    assert model["coef"]["log_dist"] < 0
