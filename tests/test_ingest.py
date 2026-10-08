"""Ingest-layer tests that do not need the network.

The loaders' hard parts are the bits that are easy to get silently wrong: which
side took the shot, whether a penalty is labelled, where an own goal is credited,
and how a match clock that restarts every period becomes one continuous line.
"""

import pandas as pd
import pytest

from src.ingest import statsbomb, understat


def test_season_labels_parse_both_conventions():
    assert understat._season_start_year("1415") == 2014
    assert understat._season_start_year(2014) == 2014
    assert understat._season_start_year("2014") == 2014
    assert understat._season_start_year("14") == 2014
    assert understat._season_start_year("9900") == 1999
    with pytest.raises(ValueError):
        understat._season_start_year("nonsense")


def test_penalty_is_recovered_when_the_label_is_missing():
    """soccerdata 1.9.1 drops Understat's `Penalty` situation, so every penalty
    arrives as <NA>. The constant penalty xG is the fallback."""
    sit = pd.Series(["Open Play", None, None], dtype="object")
    xg = pd.Series([0.05, 0.7608, 0.31])
    out = understat._situation(sit, xg)
    assert out.iloc[0] == "open"
    assert out.iloc[1] == "penalty"
    assert pd.isna(out.iloc[2])  # unmapped: warned about and treated as open play


def test_situation_map_covers_understats_vocabulary():
    from soccerdata import understat as us

    for value in us.SHOT_SITUATIONS.values():
        assert value in understat.SITUATION_MAP, value
    assert "Penalty" in understat._EXTRA_SITUATIONS


def test_statsbomb_periods_become_one_continuous_clock():
    """Period 2 starts at minute 45 on StatsBomb's clock even when the first half ran
    to 45+3, so the raw minutes overlap and must be laid end to end."""
    events = [
        {"period": 1, "minute": 10, "second": 0, "type": {"name": "Shot"},
         "team": {"name": "H", "id": 1}, "shot": {"statsbomb_xg": 0.1, "type": {"name": "Open Play"},
                                         "outcome": {"name": "Saved"}}},
        {"period": 1, "minute": 47, "second": 30, "type": {"name": "Pass"}, "team": {"name": "H", "id": 1}},
        {"period": 2, "minute": 45, "second": 10, "type": {"name": "Shot"},
         "team": {"name": "A", "id": 2}, "shot": {"statsbomb_xg": 0.2, "type": {"name": "Penalty"},
                                         "outcome": {"name": "Goal"}}},
        {"period": 2, "minute": 90, "second": 0, "type": {"name": "Pass"}, "team": {"name": "A", "id": 2}},
    ]
    meta = {"match_id": 1, "competition": "C", "season_name": "2015/2016", "date": "2015-08-08",
            "home_team": "H", "away_team": "A", "home_id": 1, "away_id": 2, "gender": "male"}
    df = statsbomb._extract(events, meta)
    shots = df[df["kind"] == "shot"].sort_values("t")
    first_half_end = 47.5
    assert shots["t"].iloc[0] == pytest.approx(10.0)
    # the second-half shot sits after the first half ended, not on top of it
    assert shots["t"].iloc[1] == pytest.approx(first_half_end + 10 / 60)
    assert df["match_T"].iloc[0] == pytest.approx(first_half_end + 45.0)


def test_statsbomb_red_cards_are_picked_up_from_both_event_types():
    events = [
        {"period": 1, "minute": 30, "second": 0, "type": {"name": "Foul Committed"},
         "team": {"name": "H", "id": 1}, "foul_committed": {"card": {"name": "Second Yellow"}}},
        {"period": 1, "minute": 40, "second": 0, "type": {"name": "Bad Behaviour"},
         "team": {"name": "A", "id": 2}, "bad_behaviour": {"card": {"name": "Red Card"}}},
        {"period": 1, "minute": 44, "second": 0, "type": {"name": "Foul Committed"},
         "team": {"name": "H", "id": 1}, "foul_committed": {"card": {"name": "Yellow Card"}}},
        {"period": 1, "minute": 46, "second": 0, "type": {"name": "Pass"}, "team": {"name": "H", "id": 1}},
    ]
    meta = {"match_id": 2, "competition": "C", "season_name": "2015/2016", "date": "2015-08-08",
            "home_team": "H", "away_team": "A", "home_id": 1, "away_id": 2, "gender": "male"}
    df = statsbomb._extract(events, meta)
    cards = df[df["kind"] == "card"]
    assert len(cards) == 2  # the plain yellow is not a dismissal
    assert set(cards["side"]) == {"H", "A"}


def test_statsbomb_shootouts_are_dropped():
    events = [
        {"period": 1, "minute": 10, "second": 0, "type": {"name": "Shot"}, "team": {"name": "H", "id": 1},
         "shot": {"statsbomb_xg": 0.1, "type": {"name": "Open Play"}, "outcome": {"name": "Saved"}}},
        {"period": 5, "minute": 120, "second": 0, "type": {"name": "Shot"}, "team": {"name": "H", "id": 1},
         "shot": {"statsbomb_xg": 0.76, "type": {"name": "Penalty"}, "outcome": {"name": "Goal"}}},
    ]
    meta = {"match_id": 3, "competition": "C", "season_name": "2015/2016", "date": "2015-08-08",
            "home_team": "H", "away_team": "A", "home_id": 1, "away_id": 2, "gender": "male"}
    df = statsbomb._extract(events, meta)
    assert len(df[df["kind"] == "shot"]) == 1


def test_own_goal_is_credited_to_the_other_side():
    events = [
        {"period": 1, "minute": 10, "second": 0, "type": {"name": "Own Goal Against"},
         "team": {"name": "H", "id": 1}},
        {"period": 1, "minute": 20, "second": 0, "type": {"name": "Shot"}, "team": {"name": "H", "id": 1},
         "shot": {"statsbomb_xg": 0.1, "type": {"name": "Open Play"}, "outcome": {"name": "Saved"}}},
        {"period": 1, "minute": 46, "second": 0, "type": {"name": "Pass"}, "team": {"name": "H", "id": 1}},
    ]
    meta = {"match_id": 4, "competition": "C", "season_name": "2015/2016", "date": "2015-08-08",
            "home_team": "H", "away_team": "A", "home_id": 1, "away_id": 2, "gender": "male"}
    slate = statsbomb.build_slate(statsbomb._extract(events, meta))
    # the home team conceded, so at its next shot it is a goal down
    assert slate.events["score_diff"].iloc[0] == -1
    assert slate.goals["side"].iloc[0] == "A"
    # and the own goal itself is not a chance created by anybody
    assert len(slate.events) == 1


@pytest.mark.skipif(
    not (statsbomb.RAW_DIR / "events.parquet").exists(), reason="statsbomb pull not present"
)
def test_real_statsbomb_slate_is_football_shaped():
    slate = statsbomb.build_slate(statsbomb.load_raw())
    ev = slate.events
    per_match = ev.groupby("match_id").size()
    assert 18 < per_match.mean() < 34, "shots per match is not football"
    assert 0.08 < ev["xg"].mean() < 0.15
    pens = ev[ev["situation"] == "penalty"]
    assert 0.70 < pens["xg"].mean() < 0.85, "penalty xG should sit near 0.76"
    assert 85 < ev["match_T"].mean() < 105
    assert ev.groupby("match_id")["side"].nunique().eq(2).mean() > 0.99
    goals_per_match = ev.groupby("match_id")["is_goal"].sum().mean()
    assert 2.0 < goals_per_match < 3.5, goals_per_match
