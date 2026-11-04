"""Post-hoc possession checks, docs/spec_possessions.md section 9 (amendment 1).

Both are labelled post hoc: they were designed after the Model A development fit.

- `merge_quick_regains` (section 9.1): the possession analogue of the 4 s shot
  rebound rule. An opponent possession shorter than `threshold_s` between two of a
  team's possessions means the opponent never escaped: the three become one
  attack for the team.
- `shuffle_danger` (section 9.2): permute danger labels within (match, half, team),
  which keeps each team's count of dangerous possessions per half (dominance) and
  destroys their order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DANGER_COLS = ("ft", "ft75", "box")


def merge_quick_regains(
    table: pd.DataFrame, threshold_s: float = 10.0, *, require_high_restart: bool = True
) -> tuple[pd.DataFrame, dict]:
    """One rule's possession rows -> the merged table, plus counts for the manifest.

    `table` must alternate sides within each (match, period), which the tolerant and
    strict rules both guarantee. A possession is *short* if it lasts less than
    `threshold_s` (t_start to t_next) and has a possession on each side of it in the
    same period. With `require_high_restart` (the spec rule), the team's possession
    after it must also *start* in the final third: the ball was won back high. A
    short possession and the one after it join the group before it, so chains
    merge transitively.

    Without `require_high_restart`, any short opponent spell anywhere on the pitch
    merges. On Wyscout that removes 42% of possessions at 10 s, which collapses
    ordinary midfield turnovers. It is kept only so the counts can be reported.
    """
    if threshold_s <= 0:
        return table.copy(), {"threshold_s": 0.0, "n_before": len(table), "n_after": len(table)}

    df = table.sort_values(["match_id", "period", "t_start"], kind="stable").reset_index(drop=True)
    key = df["match_id"].astype(str) + "|" + df["period"].astype(str)
    same_prev = key.eq(key.shift(1))
    same_next = key.eq(key.shift(-1))
    alternates = df["side"].ne(df["side"].shift(1))
    if not bool(alternates[same_prev].all()):
        raise ValueError("possessions do not alternate sides within a period; cannot merge")

    dur_s = (df["t_next"] - df["t_start"]) * 60.0
    short = (dur_s < threshold_s) & same_prev & same_next
    if require_high_restart:
        starts_high = df["t_ft"].notna() & np.isclose(df["t_ft"], df["t_start"])
        short &= starts_high.shift(-1, fill_value=False)
    start = ~(short | short.shift(1, fill_value=False)) | ~same_prev
    group = start.cumsum() - 1
    df["_g"] = group.to_numpy()

    first = df.groupby("_g", sort=True).head(1).set_index("_g")
    owner_side = df["_g"].map(first["side"])
    own = df[df["side"].eq(owner_side)]  # members belonging to the attacking team
    g_own = own.groupby("_g", sort=True)
    g_all = df.groupby("_g", sort=True)

    out = first.copy()
    out["t_next"] = g_all["t_next"].last()
    out["t_last"] = g_all["t_last"].last()
    out["n_events"] = g_all["n_events"].sum()
    out["n_shots"] = g_own["n_shots"].sum().reindex(out.index, fill_value=0).astype("int64")
    out["n_merged"] = g_all.size()
    for col in DANGER_COLS:
        tcol, scol = f"t_{col}", f"sp_{col}"
        if tcol not in df:
            continue
        t_first = g_own[tcol].min().reindex(out.index)
        out[tcol] = t_first
        # A set piece in an own member that starts at or before the merged attack's
        # first danger moment (anywhere, if it never becomes dangerous).
        lim = own["_g"].map(t_first)
        sp_ok = own[scol] & (lim.isna() | (own["t_start"] <= lim))
        out[scol] = sp_ok.groupby(own["_g"]).any().reindex(out.index, fill_value=False).astype(bool)

    out = out.reset_index(drop=True)
    out["poss"] = out.groupby("match_id").cumcount()
    stats = {
        "threshold_s": float(threshold_s),
        "require_high_restart": bool(require_high_restart),
        "n_before": len(df),
        "n_after": len(out),
        "possessions_removed_fraction": float(1.0 - len(out) / len(df)),
        "attacks_merged": int((out["n_merged"] > 1).sum()),
        "dangerous_share_before": float(df["t_ft"].notna().mean()) if "t_ft" in df else None,
        "dangerous_share_after": float(out["t_ft"].notna().mean()) if "t_ft" in out else None,
    }
    return out, stats


def shuffle_danger(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Permute (t_danger, sp_danger) pairs within each (match, period, side).

    Times, sides and game state stay where they are. The set-piece flag moves with
    its danger time, because it is defined relative to it."""
    df = table.copy()
    cols = [c for col in DANGER_COLS for c in (f"t_{col}", f"sp_{col}") if c in df]
    keys = df[["match_id", "period", "side"]].astype(str).agg("|".join, axis=1)
    order = np.arange(len(df))
    perm = order.copy()
    for idx in pd.Series(order).groupby(keys.to_numpy()).indices.values():
        perm[idx] = rng.permutation(idx)
    for c in cols:
        df[c] = df[c].to_numpy()[perm]
    return df
