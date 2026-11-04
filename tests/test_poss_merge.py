"""Quick-regain merge and within-half shuffle (docs/spec_possessions.md section 9)."""

import numpy as np
import pandas as pd
import pytest

from src.features import poss_merge


def _table(rows, match="m1"):
    """rows: (period, side, t_start_min, dur_s, t_ft or None, sp_ft, n_shots)"""
    out = []
    for i, (period, side, t0, dur, tft, sp, ns) in enumerate(rows):
        out.append(
            {
                "match_id": match,
                "poss": i,
                "period": period,
                "side": side,
                "team": side,
                "opponent": "A" if side == "H" else "H",
                "t_start": t0,
                "t_last": t0 + dur / 120.0,
                "t_next": t0 + dur / 60.0,
                "n_events": 3,
                "start_type": "open",
                "t_ft": np.nan if tft is None else tft,
                "sp_ft": sp,
                "n_shots": ns,
                "score_diff": 0,
                "red_diff": 0,
            }
        )
    df = pd.DataFrame(out)
    # make t_next consistent with the following possession's start, as the table does
    nxt = df["t_start"].shift(-1)
    same = df["period"].eq(df["period"].shift(-1))
    df.loc[same, "t_next"] = nxt[same]
    return df


def test_short_opponent_possession_joins_the_two_sides_of_it():
    #   H 0.0 (30 s)   A 0.5 (5 s, short)   H 0.5833 (30 s)   A 1.0833 (60 s)
    t = _table(
        [
            (0, "H", 0.0, 30, None, False, 0),
            (0, "A", 0.5, 5, None, False, 0),
            (0, "H", 0.5 + 5 / 60, 30, 0.6, False, 1),
            (0, "A", 0.5 + 35 / 60, 60, None, False, 0),
        ]
    )
    out, stats = poss_merge.merge_quick_regains(t, 10.0, require_high_restart=False)
    assert list(out["side"]) == ["H", "A"]
    assert out.loc[0, "n_merged"] == 3
    assert out.loc[0, "t_start"] == 0.0
    assert out.loc[0, "t_ft"] == pytest.approx(0.6)  # dangerous because its second part was
    assert out.loc[0, "n_shots"] == 1
    assert stats["n_before"] == 4 and stats["n_after"] == 2


def test_chains_merge_transitively_and_alternation_survives():
    rows = [(0, "H", 0.0, 20, None, False, 0)]
    t0 = 20 / 60
    for _k in range(3):  # three short opponent touches, each followed by H again
        rows.append((0, "A", t0, 4, None, False, 0))
        t0 += 4 / 60
        rows.append((0, "H", t0, 20, None, False, 0))
        t0 += 20 / 60
    rows.append((0, "A", t0, 90, None, False, 0))
    out, _ = poss_merge.merge_quick_regains(_table(rows), 10.0, require_high_restart=False)
    assert list(out["side"]) == ["H", "A"]
    assert out.loc[0, "n_merged"] == 7
    assert (out["side"] != out["side"].shift()).iloc[1:].all()


def test_opponent_danger_inside_a_merged_attack_is_not_credited_to_the_attacker():
    t = _table(
        [
            (0, "H", 0.0, 30, None, False, 0),
            (0, "A", 0.5, 5, 0.52, False, 0),  # opponent reaches its final third in 5 s
            (0, "H", 0.5 + 5 / 60, 30, None, False, 0),
            (0, "A", 1.2, 60, None, False, 0),
        ]
    )
    out, _ = poss_merge.merge_quick_regains(t, 10.0, require_high_restart=False)
    assert np.isnan(out.loc[0, "t_ft"])


def test_no_merge_across_a_period_boundary_or_at_the_edges():
    t = _table(
        [
            (0, "H", 44.0, 30, None, False, 0),
            (0, "A", 44.5, 3, None, False, 0),  # last of the half: no possession after it
            (1, "H", 45.0, 3, None, False, 0),  # first of the half: none before it
            (1, "A", 45.05, 60, None, False, 0),
        ]
    )
    t.loc[1, "t_next"] = 44.5 + 3 / 60
    out, _ = poss_merge.merge_quick_regains(t, 10.0, require_high_restart=False)
    assert len(out) == 4


def test_setpiece_after_the_first_danger_moment_does_not_count():
    t = _table(
        [
            (0, "H", 0.0, 30, 0.2, False, 0),  # dangerous at 0.2
            (0, "A", 0.5, 5, None, False, 0),
            (0, "H", 0.5 + 5 / 60, 30, 0.7, True, 0),  # corner, but after the first danger
            (0, "A", 1.2, 60, None, False, 0),
        ]
    )
    out, _ = poss_merge.merge_quick_regains(t, 10.0, require_high_restart=False)
    assert out.loc[0, "t_ft"] == pytest.approx(0.2)
    assert not out.loc[0, "sp_ft"]


def test_threshold_zero_is_the_identity():
    t = _table(
        [
            (0, "H", 0.0, 30, None, False, 0),
            (0, "A", 0.5, 2, None, False, 0),
            (0, "H", 0.6, 9, None, False, 0),
        ]
    )
    out, stats = poss_merge.merge_quick_regains(t, 0.0)
    assert len(out) == len(t) and stats["n_after"] == len(t)


def test_shuffle_keeps_counts_per_team_half_and_moves_setpiece_with_danger():
    rng = np.random.default_rng(1)
    rows = []
    t = 0.0
    for period in (0, 1):
        for i in range(40):
            side = "H" if i % 2 == 0 else "A"
            danger = i % 3 == 0
            rows.append((period, side, t, 30, t + 0.1 if danger else None, danger and i % 6 == 0, 0))
            t += 0.5
    tab = _table(rows)
    sh = poss_merge.shuffle_danger(tab, rng)
    key = ["match_id", "period", "side"]
    before = tab.assign(d=tab["t_ft"].notna()).groupby(key)["d"].sum()
    after = sh.assign(d=sh["t_ft"].notna()).groupby(key)["d"].sum()
    pd.testing.assert_series_equal(before, after)
    # a set-piece flag only ever sits on a dangerous row in this table, before and after
    assert not (sh["sp_ft"] & sh["t_ft"].isna()).any()
    assert (sh["t_start"] == tab["t_start"]).all()
    assert not sh["t_ft"].isna().equals(tab["t_ft"].isna())  # something moved


def test_spec_rule_merges_only_when_the_ball_is_won_back_high():
    base = [
        (0, "H", 0.0, 30, None, False, 0),
        (0, "A", 0.5, 5, None, False, 0),
        None,
        (0, "A", 0.5 + 35 / 60, 60, None, False, 0),
    ]
    t_next_h = 0.5 + 5 / 60
    high = list(base)
    high[2] = (0, "H", t_next_h, 30, t_next_h, False, 0)  # first own event already in the final third
    low = list(base)
    low[2] = (0, "H", t_next_h, 30, t_next_h + 0.2, False, 0)  # gets there later: a new attack
    out_high, _ = poss_merge.merge_quick_regains(_table(high), 10.0)
    out_low, _ = poss_merge.merge_quick_regains(_table(low), 10.0)
    assert len(out_high) == 2
    assert len(out_low) == 4
