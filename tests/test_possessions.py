"""Possession table: the two rules, own-possession time, danger flags, game state.

The unit tests build tiny on-ball sequences by hand. The last two run against the
real table when it is on disk: every shot must sit inside its own team's
possession time, and the game state rebuilt here must equal the shot slate's at
every shot. Either failing means the two tables are not on one clock.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import INTERIM
from src.ingest import possessions

H, A = 10, 20
ENDS = {0: 45.0, 1: 92.0}


def _ob(rows):
    """rows: (period, t, team, name, sub, accurate, x0, y0, x1, y1)"""
    return pd.DataFrame(
        rows, columns=["period", "t", "team_id", "name", "sub", "accurate", "x0", "y0", "x1", "y1"]
    )


def _ev(period, t, team, name="Pass", sub="Simple pass", acc=True, x0=50, y0=50, x1=55, y1=50):
    return (period, t, team, name, sub, acc, x0, y0, x1, y1)


def _poss(rows, tolerant=True):
    return possessions.summarise(possessions.assign_possessions(_ob(rows), tolerant=tolerant), ENDS)


def test_single_stray_touch_does_not_end_a_possession_under_the_tolerant_rule():
    rows = [
        _ev(0, 1.0, H),
        _ev(0, 1.1, H),
        _ev(0, 1.2, A, "Others on the ball", "Clearance", acc=False),
        _ev(0, 1.3, H),
        _ev(0, 1.4, H),
    ]
    assert len(_poss(rows, tolerant=True)) == 1
    assert len(_poss(rows, tolerant=False)) == 3


def test_an_accurate_pass_or_a_shot_is_never_a_touch():
    for opp in (_ev(0, 1.2, A), _ev(0, 1.2, A, "Shot", "Shot", acc=False, x0=80, y0=50)):
        rows = [_ev(0, 1.0, H), opp, _ev(0, 1.4, H)]
        assert len(_poss(rows)) == 3


def test_two_opponent_events_take_the_ball():
    rows = [
        _ev(0, 1.0, H),
        _ev(0, 1.2, A, acc=False),
        _ev(0, 1.3, A, acc=False),
        _ev(0, 1.4, H),
    ]
    assert len(_poss(rows)) == 3


def test_a_period_break_ends_a_possession_and_own_time_tiles_each_period():
    rows = [_ev(0, 1.0, H), _ev(0, 20.0, A), _ev(1, 46.0, A), _ev(1, 60.0, H)]
    p = _poss(rows)
    assert len(p) == 4
    np.testing.assert_allclose(p["t_next"], [20.0, 45.0, 60.0, 92.0])
    # own time covers each period from its first on-ball event to its end
    total = (p["t_next"] - p["t_start"]).sum()
    assert total == pytest.approx((45.0 - 1.0) + (92.0 - 46.0))


def test_a_touch_does_not_carry_its_own_location_into_the_danger_flags():
    rows = [
        _ev(0, 1.0, H, x0=40, x1=45),
        _ev(0, 1.1, A, "Others on the ball", "Clearance", acc=False, x0=90, y0=50, x1=95, y1=50),
        _ev(0, 1.2, H, x0=45, x1=50),
    ]
    p = _poss(rows)
    assert len(p) == 1 and np.isnan(p["t_ft"].iloc[0])


def test_danger_flags_need_an_accurate_pass_or_an_event_that_starts_there():
    rows = [
        _ev(0, 1.0, H, x0=40, x1=70, acc=False),  # inaccurate into the final third: no
        _ev(0, 1.1, H, x0=40, x1=70),  # accurate into the final third: yes
        _ev(0, 1.2, H, x0=78, y0=50, x1=79),  # starts beyond x = 75
        _ev(0, 1.3, H, "Shot", "Shot", acc=False, x0=88, y0=50, x1=100, y1=50),  # in the box
    ]
    p = _poss(rows).iloc[0]
    assert p["t_ft"] == pytest.approx(1.1)
    assert p["t_ft75"] == pytest.approx(1.2)
    assert p["t_box"] == pytest.approx(1.3)
    assert p["n_shots"] == 1


def test_setpiece_flag_counts_only_restarts_at_or_before_the_first_danger_moment():
    before = [_ev(0, 1.0, H, "Free Kick", "Throw in", x0=60, x1=70), _ev(0, 1.1, H, x0=70, x1=75)]
    after = [_ev(0, 1.0, H, x0=40, x1=70), _ev(0, 1.1, H, "Free Kick", "Corner", x0=100, y0=0, x1=90)]
    never = [_ev(0, 1.0, H, x0=20, x1=30), _ev(0, 1.1, H, "Free Kick", "Throw in", x0=30, x1=35)]
    goal_kick = [_ev(0, 1.0, H, "Free Kick", "Goal kick", x0=5, x1=40)]
    assert bool(_poss(before)["sp_ft"].iloc[0]) is True
    assert bool(_poss(after)["sp_ft"].iloc[0]) is False
    assert bool(_poss(never)["sp_ft"].iloc[0]) is True
    assert bool(_poss(goal_kick)["sp_ft"].iloc[0]) is False


def test_start_type_comes_from_the_first_own_event():
    rows = [_ev(0, 1.0, H, "Free Kick", "Corner", x0=100, y0=0), _ev(0, 2.0, A), _ev(0, 2.1, A)]
    assert list(_poss(rows)["start_type"]) == ["corner", "open"]


def test_state_counts_goals_strictly_before_and_cards_from_their_instant():
    goals = pd.DataFrame({"side": ["H"], "t": [10.0]})
    cards = pd.DataFrame({"side": ["A"], "t": [30.0]})  # away lose a player
    t = np.array([10.0, 10.0, 11.0, 30.0, 30.0])
    side = np.array(["H", "A", "A", "H", "A"])
    sd, rd = possessions.state_at(t, side, goals, cards)
    assert list(sd) == [0, 0, -1, 1, -1]
    assert list(rd) == [0, 0, 0, 1, -1]


TABLE = INTERIM / "wyscout_possessions" / "possessions.parquet"
SHOTS = INTERIM / "wyscout" / "events.parquet"
needs_data = pytest.mark.skipif(not (TABLE.exists() and SHOTS.exists()), reason="possession table not built")


@needs_data
@pytest.mark.parametrize("rule", ["tolerant", "strict"])
def test_every_shot_lies_in_its_own_teams_possession_time(rule):
    tab = pd.read_parquet(TABLE)
    tab = tab[tab["rule"] == rule]
    shots = pd.read_parquet(SHOTS)
    # A shot that opens a possession can come out ~1e-12 min before that possession's
    # start (the shot clock's offset is recovered as t - local), and two events by
    # different teams can share one eventSec. Allow 1e-7 min (6 microseconds).
    eps = 1e-7
    bad = 0
    for mid, s in shots.groupby("match_id"):
        p = tab[tab["match_id"] == mid].sort_values("t_start")
        starts, sides = p["t_start"].to_numpy(), p["side"].to_numpy()
        t = s["t"].to_numpy()
        i = np.searchsorted(starts, t + eps, side="right") - 1
        ok = (i >= 0) & (t < p["t_next"].to_numpy()[i] + eps)
        own = sides[i] == s["side"].to_numpy()
        # or the shot's own possession starts at the same instant as the one found
        j = np.clip(i - 1, 0, None)
        tie = (np.abs(starts[i] - t) <= eps) & (sides[j] == s["side"].to_numpy())
        bad += int((~(ok & (own | tie))).sum())
    assert bad == 0


@needs_data
def test_rebuilt_game_state_equals_the_shot_slate_at_every_shot():
    shots = pd.read_parquet(SHOTS)
    goals = pd.read_parquet(INTERIM / "wyscout" / "goals.parquet")
    cards = pd.read_parquet(INTERIM / "wyscout" / "cards.parquet")
    gb, cb = dict(tuple(goals.groupby("match_id"))), dict(tuple(cards.groupby("match_id")))
    empty = goals.iloc[0:0]
    mism = 0
    for mid, s in shots.groupby("match_id"):
        sd, rd = possessions.state_at(
            s["t"].to_numpy(), s["side"].to_numpy(), gb.get(mid, empty), cb.get(mid, empty)
        )
        mism += int(((sd != s["score_diff"].to_numpy()) | (rd != s["red_diff"].to_numpy())).sum())
    assert mism == 0
