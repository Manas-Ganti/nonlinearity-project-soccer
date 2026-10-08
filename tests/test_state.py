"""Game-state reconstruction against known scorelines.

CLAUDE.md build step 1: "Write reconstruction tests against known scorelines."
Getting these wrong produces a confident wrong answer -- score state is the
dominant confound and it points the wrong way.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import state as st


def _shots(rows):
    df = pd.DataFrame(rows, columns=["match_id", "side", "t"])
    df["order_idx"] = np.arange(len(df))
    return df


def _events(rows):
    df = pd.DataFrame(rows, columns=["match_id", "side", "t"])
    df["order_idx"] = np.arange(len(df))
    return df


def test_score_is_state_before_the_shot():
    """A goal is itself a shot; it must not condition on its own outcome."""
    shots = _shots([("m", "H", 10.0), ("m", "H", 20.0), ("m", "A", 30.0)])
    goals = _events([("m", "H", 10.0)])
    out = st.reconstruct(shots, goals, None)
    assert out["score_diff"].tolist() == [0, 1, -1]


def test_two_one_scoreline():
    shots = _shots([("m", "H", 5.0), ("m", "A", 25.0), ("m", "H", 60.0), ("m", "A", 80.0), ("m", "H", 88.0)])
    goals = _events([("m", "H", 5.0), ("m", "A", 25.0), ("m", "H", 60.0)])
    out = st.reconstruct(shots, goals, None)
    # before each shot: 0, then H lead 1 (from A's view -1), then 0, then H lead 1 -> A sees -1, then +1
    assert out["score_diff"].tolist() == [0, -1, 0, -1, 1]


def test_score_is_clipped():
    shots = _shots([("m", "H", 90.0)])
    goals = _events([("m", "H", t) for t in (5.0, 10.0, 15.0, 20.0)])
    out = st.reconstruct(shots, goals, None)
    assert out["score_diff"].tolist() == [st.SCORE_CLIP]


def test_red_card_is_a_man_advantage_for_the_other_side():
    shots = _shots([("m", "H", 40.0), ("m", "H", 60.0), ("m", "A", 70.0)])
    cards = _events([("m", "A", 50.0)])  # away team loses a player
    out = st.reconstruct(shots, _events([]), cards)
    assert out["red_diff"].tolist() == [0, 1, -1]


def test_card_takes_effect_at_the_same_instant():
    shots = _shots([("m", "H", 50.0)])
    cards = _events([("m", "A", 50.0)])
    out = st.reconstruct(shots, _events([]), cards)
    assert out["red_diff"].tolist() == [1]


def test_trajectory_matches_event_level_reconstruction():
    goals = _events([("m", "H", 12.4), ("m", "A", 67.9)])
    cards = _events([("m", "A", 30.2)])
    score, red = st.state_trajectory(95.0, goals, cards, "H")
    assert score[0] == 0 and score[13] == 1 and score[-1] == 0
    assert red[30] == 0 and red[31] == 1 and red[-1] == 1
    assert len(score) == 95


def test_trajectory_is_constant_inside_a_bin():
    """mu must be exactly constant within a one-minute bin, which is what makes the
    compensator a finite sum (docs/math.md section 2)."""
    goals = _events([("m", "H", 12.4)])
    score, _ = st.state_trajectory(95.0, goals, None, "H")
    assert score[12] == 0  # the bin the goal falls in keeps the state it opened with
    assert score[13] == 1


def test_empty_streams():
    shots = _shots([("m", "H", 10.0)])
    out = st.reconstruct(shots, _events([]), None)
    assert out["score_diff"].tolist() == [0]
    assert out["red_diff"].tolist() == [0]


@pytest.mark.parametrize("side", ["H", "A"])
def test_sign_conventions_are_mirror_images(side):
    shots = _shots([("m", "H", 80.0), ("m", "A", 80.0)])
    goals = _events([("m", side, 10.0)])
    out = st.reconstruct(shots, goals, None)
    assert out["score_diff"].iloc[0] == -out["score_diff"].iloc[1]
