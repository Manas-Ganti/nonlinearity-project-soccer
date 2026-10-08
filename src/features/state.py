"""Game-state reconstruction: score differential and man advantage at every event time.

Both are piecewise constant and change only at goals and dismissals, so the state
attached to an event is the state *strictly before* that event. A goal is itself a
shot, and it must not condition on its own outcome.

Sign conventions (also in `ingest.schema`):
  score_diff = team goals - opponent goals
  red_diff   = opponent dismissals - team dismissals   (positive == man advantage)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCORE_CLIP = 2
RED_CLIP = 1


def _ordering_key(df: pd.DataFrame) -> pd.DataFrame:
    """Stable within-match ordering: time first, then the source's own order.

    `order_idx` is whatever the source used to sequence events inside one minute.
    Dedup keeps the earliest shot of a collapsed run and jitter is order-preserving,
    so this ordering survives the rest of the pipeline unchanged.
    """
    return df.sort_values(["match_id", "t", "row_kind", "order_idx"], kind="stable")


def reconstruct(
    shots: pd.DataFrame,
    goals: pd.DataFrame,
    cards: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach `score_diff` and `red_diff` to `shots`.

    Parameters
    ----------
    shots : columns match_id, side, t, order_idx
    goals : columns match_id, side (side *credited* with the goal), t, order_idx.
        Own goals must already be re-attributed to the side that benefits.
    cards : columns match_id, side (the side that lost a player), t, order_idx.
        None means no dismissal data is available; red_diff is then 0 everywhere
        and `red_cards_known` is False.

    Returns the `shots` frame with score_diff/red_diff added, original row order.
    """
    shots = shots.copy()
    shots["_row"] = np.arange(len(shots), dtype=np.int64)

    pieces = [
        shots.assign(row_kind=1, d_goal=0, d_card=0)[
            ["match_id", "side", "t", "order_idx", "row_kind", "d_goal", "d_card", "_row"]
        ]
    ]
    if len(goals):
        pieces.append(
            goals.assign(row_kind=1, d_goal=1, d_card=0, _row=-1)[
                ["match_id", "side", "t", "order_idx", "row_kind", "d_goal", "d_card", "_row"]
            ]
        )
    if cards is not None and len(cards):
        # Dismissals take effect before any shot recorded at the same instant.
        pieces.append(
            cards.assign(row_kind=0, d_goal=0, d_card=1, _row=-1)[
                ["match_id", "side", "t", "order_idx", "row_kind", "d_goal", "d_card", "_row"]
            ]
        )

    comb = _ordering_key(pd.concat(pieces, ignore_index=True))
    is_home = (comb["side"] == "H").to_numpy()

    match_key = comb["match_id"]
    for name in ("goal", "card"):
        delta = comb[f"d_{name}"].to_numpy()
        h = np.where(is_home, delta, 0)
        a = np.where(is_home, 0, delta)
        # count strictly before this row = inclusive cumsum minus this row's own delta
        comb[f"h_{name}"] = pd.Series(h, index=comb.index).groupby(match_key).cumsum() - h
        comb[f"a_{name}"] = pd.Series(a, index=comb.index).groupby(match_key).cumsum() - a

    out = comb[comb["_row"] >= 0].copy()
    home = (out["side"] == "H").to_numpy()
    own_goals = np.where(home, out["h_goal"], out["a_goal"])
    opp_goals = np.where(home, out["a_goal"], out["h_goal"])
    own_cards = np.where(home, out["h_card"], out["a_card"])
    opp_cards = np.where(home, out["a_card"], out["h_card"])

    score = np.clip(own_goals - opp_goals, -SCORE_CLIP, SCORE_CLIP)
    red = np.clip(opp_cards - own_cards, -RED_CLIP, RED_CLIP)

    res = pd.DataFrame({"_row": out["_row"].to_numpy(), "score_diff": score, "red_diff": red}).sort_values(
        "_row"
    )
    shots = shots.merge(res, on="_row", how="left").drop(columns=["_row"])
    shots["score_diff"] = shots["score_diff"].fillna(0).astype("int16")
    shots["red_diff"] = shots["red_diff"].fillna(0).astype("int16")
    return shots


def state_trajectory(
    match_T: float,
    goals: pd.DataFrame,
    cards: pd.DataFrame | None,
    side: str,
    bin_minutes: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-minute-bin (score_diff, red_diff) for one side of one match.

    Used by the background model, which is piecewise constant on one-minute bins.
    A goal or card inside bin b takes effect from bin b+1: the bin in which it
    happens keeps the state that held when the bin opened, which is what makes
    `mu` exactly constant within the bin (docs/math.md section 2).
    """
    n_bins = int(np.ceil(match_T / bin_minutes))
    edges = np.arange(n_bins + 1, dtype=np.float64) * bin_minutes
    score = np.zeros(n_bins, dtype=np.int16)
    red = np.zeros(n_bins, dtype=np.int16)

    def _cum(frame: pd.DataFrame, want_side: str) -> np.ndarray:
        if frame is None or len(frame) == 0:
            return np.zeros(n_bins, dtype=np.int64)
        t = frame.loc[frame["side"] == want_side, "t"].to_numpy(dtype=np.float64)
        if t.size == 0:
            return np.zeros(n_bins, dtype=np.int64)
        # count of events strictly before the *opening* of each bin
        return np.searchsorted(np.sort(t), edges[:-1], side="left").astype(np.int64)

    opp = "A" if side == "H" else "H"
    own_g, opp_g = _cum(goals, side), _cum(goals, opp)
    own_c = _cum(cards, side) if cards is not None else np.zeros(n_bins, dtype=np.int64)
    opp_c = _cum(cards, opp) if cards is not None else np.zeros(n_bins, dtype=np.int64)

    score[:] = np.clip(own_g - opp_g, -SCORE_CLIP, SCORE_CLIP)
    red[:] = np.clip(opp_c - own_c, -RED_CLIP, RED_CLIP)
    return score, red
