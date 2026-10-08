"""The canonical event table. Everything downstream reads this and nothing else."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# One row per shot. `t` is minutes from kickoff on a continuous clock; `minute`
# is the integer the source recorded. For Understat the two differ only by the
# jitter applied in `features.jitter`; for StatsBomb `t` is exact.
EVENT_COLUMNS: dict[str, str] = {
    "match_id": "string",  # unique across sources: "<source>:<league>:<season>:<game>"
    "source": "string",  # understat | statsbomb | synthetic
    "league": "string",
    "season": "int16",  # starting year, 2014 == 2014/15
    "date": "datetime64[ns]",
    "side": "string",  # H | A  -- the shooting team's side
    "team": "string",
    "opponent": "string",
    "t": "float64",  # minutes from kickoff
    "minute": "int16",  # integer minute as recorded by the source
    "second": "float64",  # NaN where the source has no sub-minute clock
    "xg": "float64",
    "situation": "string",  # open | setpiece | penalty
    "is_goal": "bool",  # goal credited to `team` (own goals are re-attributed)
    "score_diff": "int16",  # (team goals - opponent goals) immediately before t
    "red_diff": "int16",  # man advantage: (opponent dismissals - team dismissals) before t
    "match_T": "float64",  # observation window length in minutes
    "n_merged": "int16",  # shots collapsed into this event by dedup (1 = untouched)
    "red_cards_known": "bool",  # False => red_diff is an assumption, not data
    # Strata. Pooling across providers, leagues and genders is the design; these
    # let eta be reported per stratum as the check that the pooling was fair.
    "gender": "string",  # male | female
    "one_club_sample": "bool",  # league-season where one club is in (nearly) every match
}

# Columns a loader may omit; filled with these defaults in `coerce`.
STRATA_DEFAULTS = {"gender": "male", "one_club_sample": False}

REQUIRED = tuple(c for c in EVENT_COLUMNS if c not in STRATA_DEFAULTS)


class SchemaError(ValueError):
    pass


def empty_events() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=d) for c, d in EVENT_COLUMNS.items()})


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"event table missing columns: {missing}")
    df = df.copy()
    for col, default in STRATA_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    out = df.loc[:, list(EVENT_COLUMNS)].copy()
    for col, dtype in EVENT_COLUMNS.items():
        if dtype == "datetime64[ns]":
            out[col] = pd.to_datetime(out[col])
        else:
            out[col] = out[col].astype(dtype)
    return out.sort_values(["match_id", "t", "side"], kind="stable").reset_index(drop=True)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Hard invariants. Anything violated here is a bug upstream, not a warning."""
    df = coerce(df)
    if len(df) == 0:
        return df
    if not df["situation"].isin(["open", "setpiece", "penalty"]).all():
        bad = sorted(set(df["situation"]) - {"open", "setpiece", "penalty"})
        raise SchemaError(f"unknown situation values: {bad}")
    if not df["side"].isin(["H", "A"]).all():
        raise SchemaError("side must be H or A")
    if not df["gender"].isin(["male", "female"]).all():
        raise SchemaError("gender must be male or female")
    if (df["t"] < 0).any():
        raise SchemaError("negative event time")
    if (df["t"] > df["match_T"]).any():
        n = int((df["t"] > df["match_T"]).sum())
        raise SchemaError(f"{n} events after the end of the observation window")
    if not np.isfinite(df["xg"]).all() or (df["xg"] < 0).any() or (df["xg"] > 1).any():
        raise SchemaError("xg outside [0, 1] or non-finite")
    if (df["n_merged"] < 1).any():
        raise SchemaError("n_merged must be >= 1")
    per_match_sides = df.groupby("match_id")["side"].nunique()
    if (per_match_sides > 2).any():
        raise SchemaError("a match has more than two sides")
    # match_T must be a single value per match
    if df.groupby("match_id")["match_T"].nunique().gt(1).any():
        raise SchemaError("match_T is not constant within a match")
    return df


def flag_one_club(events: pd.DataFrame, share: float = 0.9) -> pd.Series:
    """True for events in a (source, league, season) where one club appears in at
    least `share` of the matches. StatsBomb's Barcelona-only La Liga seasons and
    Leverkusen-only Bundesliga 2015/16 are the cases; a full league or a
    tournament never trips it."""
    key = ["source", "league", "season"]
    pairs = events.loc[:, [*key, "match_id", "team"]].drop_duplicates()
    n_matches = pairs.groupby(key)["match_id"].nunique()
    top = pairs.groupby([*key, "team"])["match_id"].nunique().groupby(level=key).max()
    flag = (top / n_matches) >= share
    idx = pd.MultiIndex.from_frame(events.loc[:, key])
    return pd.Series(flag.reindex(idx).to_numpy(), index=events.index, dtype=bool)


def match_frame(events: pd.DataFrame) -> pd.DataFrame:
    """One row per match: the pieces the background model needs that are not per-shot."""
    g = events.groupby("match_id", sort=True)
    out = g.agg(
        source=("source", "first"),
        league=("league", "first"),
        season=("season", "first"),
        date=("date", "first"),
        match_T=("match_T", "first"),
        red_cards_known=("red_cards_known", "first"),
        gender=("gender", "first"),
        one_club_sample=("one_club_sample", "first"),
        n_events=("t", "size"),
    ).reset_index()
    home = (
        events[events["side"] == "H"]
        .groupby("match_id")
        .agg(home_team=("team", "first"), away_team=("opponent", "first"))
    )
    away = (
        events[events["side"] == "A"]
        .groupby("match_id")
        .agg(away_team_alt=("team", "first"), home_team_alt=("opponent", "first"))
    )
    out = out.merge(home, on="match_id", how="left").merge(away, on="match_id", how="left")
    out["home_team"] = out["home_team"].fillna(out.pop("home_team_alt"))
    out["away_team"] = out["away_team"].fillna(out.pop("away_team_alt"))
    return out


# --------------------------------------------------------------------- slate

GOAL_COLUMNS = {"match_id": "string", "side": "string", "t": "float64", "order_idx": "int64"}
CARD_COLUMNS = dict(GOAL_COLUMNS)


class Slate:
    """An event table plus the two state-change streams it cannot reconstruct itself.

    Own goals are goals for a side that never shot, and dismissals are not shots at
    all, so neither is recoverable from `events` alone. Everything that builds a
    background model needs all three.
    """

    def __init__(self, events: pd.DataFrame, goals: pd.DataFrame, cards: pd.DataFrame | None):
        self.events = validate(events)
        self.goals = _coerce_side_table(goals, GOAL_COLUMNS)
        self.cards = None if cards is None else _coerce_side_table(cards, CARD_COLUMNS)

    @property
    def red_cards_known(self) -> bool:
        return self.cards is not None

    def filter_matches(self, match_ids) -> Slate:
        keep = set(match_ids)
        ev = self.events[self.events["match_id"].isin(keep)]
        go = self.goals[self.goals["match_id"].isin(keep)]
        ca = None if self.cards is None else self.cards[self.cards["match_id"].isin(keep)]
        return Slate(ev, go, ca)

    def with_events(self, events: pd.DataFrame) -> Slate:
        """Same match state, a different (e.g. de-duplicated or simulated) event set."""
        return Slate(events, self.goals, self.cards)

    def save(self, directory) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.events.to_parquet(directory / "events.parquet", index=False)
        self.goals.to_parquet(directory / "goals.parquet", index=False)
        if self.cards is not None:
            self.cards.to_parquet(directory / "cards.parquet", index=False)

    @classmethod
    def load(cls, directory) -> Slate:
        directory = Path(directory)
        cards_path = directory / "cards.parquet"
        return cls(
            pd.read_parquet(directory / "events.parquet"),
            pd.read_parquet(directory / "goals.parquet"),
            pd.read_parquet(cards_path) if cards_path.exists() else None,
        )

    def __repr__(self) -> str:
        return (
            f"Slate(matches={self.events['match_id'].nunique()}, events={len(self.events)}, "
            f"goals={len(self.goals)}, cards={'none' if self.cards is None else len(self.cards)})"
        )


def _coerce_side_table(df: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame({c: pd.Series(dtype=d) for c, d in cols.items()})
    out = df.copy()
    if "order_idx" not in out.columns:
        out["order_idx"] = np.arange(len(out), dtype=np.int64)
    out = out.loc[:, list(cols)]
    for c, d in cols.items():
        out[c] = out[c].astype(d)
    return out.sort_values(["match_id", "t"], kind="stable").reset_index(drop=True)
