"""Understat shot events via `soccerdata`.

CLAUDE.md: do not write scrapers, do not re-scrape. `pull()` writes one parquet
per (league, season) into data/raw/understat/ and skips anything already there;
`soccerdata` keeps its own per-match HTML cache underneath. Everything downstream
reads the parquet, never the network.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW, T_NOMINAL
from src.features import state as state_mod
from src.ingest import schema

log = logging.getLogger(__name__)

RAW_DIR = RAW / "understat"
SOURCE = "understat"

# Understat's own `situation` vocabulary includes "Penalty", which soccerdata 1.9.1
# omits from its translation table, silently turning every penalty into <NA>.
# Penalties are load-bearing here (they must be excluded from the excitation term),
# so patch the table before reading and cross-check against the constant penalty xG.
_EXTRA_SITUATIONS = {"Penalty": "Penalty"}

SITUATION_MAP = {
    "Open Play": "open",
    "From Corner": "setpiece",
    "Set Piece": "setpiece",
    "Direct Freekick": "setpiece",
    "Penalty": "penalty",
}


def _patched_reader(**kwargs):
    import soccerdata as sd
    from soccerdata import understat as us

    for k, v in _EXTRA_SITUATIONS.items():
        us.SHOT_SITUATIONS.setdefault(k, v)
    return sd.Understat(**kwargs)


def raw_path(league: str, season: int) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", league).strip("-")
    return RAW_DIR / f"{slug}_{season}.parquet"


def pull(
    leagues: list[str],
    seasons: list[int],
    *,
    overwrite: bool = False,
    **reader_kwargs,
) -> list[Path]:
    """Fetch and cache. One request per match; run it once, overnight."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for league in leagues:
        for season in seasons:
            path = raw_path(league, season)
            if path.exists() and not overwrite:
                log.info("skip %s (cached)", path.name)
                written.append(path)
                continue
            reader = _patched_reader(leagues=league, seasons=season, **reader_kwargs)
            shots = reader.read_shot_events().reset_index()
            sched = reader.read_schedule().reset_index()
            keep = [
                c
                for c in ("league", "season", "game", "home_team", "away_team", "date")
                if c in sched.columns
            ]
            shots = shots.merge(
                sched[keep],
                on=[c for c in ("league", "season", "game") if c in keep],
                how="left",
                suffixes=("", "_sched"),
            )
            shots.to_parquet(path, index=False)
            log.info("wrote %s (%d shots)", path.name, len(shots))
            written.append(path)
    return written


def load_raw(leagues: list[str] | None = None, seasons: list[int] | None = None) -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no Understat pulls in {RAW_DIR}. Run `make ingest-understat` once (it takes hours)."
        )
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    if leagues is not None:
        df = df[df["league"].isin(leagues)]
    if seasons is not None:
        df = df[df["season"].astype(int).isin([int(s) for s in seasons])]
    return df.reset_index(drop=True)


def _season_start_year(season) -> int:
    """soccerdata labels seasons '1415', '2014' or '14'; all mean 2014/15.

    The two four-digit forms are genuinely ambiguous as strings, so the calendar
    range decides: 1900-2100 is a year, anything else is a <start><end> pair.
    """
    s = str(season).strip()
    if not s.isdigit():
        raise ValueError(f"unparseable season label {season!r}")
    v = int(s)
    if len(s) == 4:
        if 1900 <= v <= 2100:
            return v
        first = int(s[:2])  # '1415' -> 2014/15, '9900' -> 1999/2000
        return (2000 + first) if first < 50 else (1900 + first)
    if len(s) == 2:
        return (2000 + v) if v < 50 else (1900 + v)
    raise ValueError(f"unparseable season label {season!r}")


def _situation(raw_situation: pd.Series, xg: pd.Series) -> pd.Series:
    sit = raw_situation.map(SITUATION_MAP)
    # Penalty xG on Understat is a constant ~0.7608; any row the map missed at that
    # exact value is a penalty whose label was lost upstream.
    looks_penalty = (xg.sub(0.7608).abs() < 5e-4) & sit.isna()
    sit = sit.where(~looks_penalty, "penalty")
    return sit


def build_slate(raw: pd.DataFrame, cards: pd.DataFrame | None = None) -> schema.Slate:
    """Raw Understat shots -> the canonical event table, with game state attached."""
    df = raw.copy()
    df["season_year"] = df["season"].map(_season_start_year).astype("int16")
    df["match_id"] = (
        SOURCE
        + ":"
        + df["league"].astype(str)
        + ":"
        + df["season"].astype(str)
        + ":"
        + df["game"].astype(str)
    )

    if "home_team" not in df.columns or df["home_team"].isna().all():
        # Fall back to the "<date> <Home>-<Away>" game label soccerdata builds.
        parsed = df["game"].astype(str).str.extract(r"^\S+\s+(?P<home_team>.+?)-(?P<away_team>.+)$")
        df["home_team"] = df.get("home_team", pd.Series(index=df.index, dtype="object")).fillna(
            parsed["home_team"]
        )
        df["away_team"] = df.get("away_team", pd.Series(index=df.index, dtype="object")).fillna(
            parsed["away_team"]
        )
    unresolved = df["home_team"].isna()
    if unresolved.any():
        raise ValueError(f"{int(unresolved.sum())} shots with no home/away assignment")

    df["side"] = np.where(df["team"].astype(str) == df["home_team"].astype(str), "H", "A")
    df["opponent"] = np.where(df["side"] == "H", df["away_team"], df["home_team"])
    df["situation_std"] = _situation(df["situation"].astype("object"), df["xg"].astype(float))
    unknown = df["situation_std"].isna()
    if unknown.any():
        log.warning("%d shots with unmapped situation -> treated as open play", int(unknown.sum()))
        df.loc[unknown, "situation_std"] = "open"

    # Own goals are goals for the *other* side and are not chances for the shooter.
    df["result"] = df["result"].astype("object")
    is_own_goal = df["result"].eq("Own Goal")
    df["is_goal"] = df["result"].eq("Goal")

    df["minute"] = df["minute"].astype(int)
    df["t"] = df["minute"].astype(float)
    df["second"] = np.nan
    df = df.sort_values(["match_id", "minute"], kind="stable").reset_index(drop=True)
    df["order_idx"] = df.groupby("match_id").cumcount().astype("int64")

    m_T = df.groupby("match_id")["minute"].max().rename("max_minute")
    df = df.merge(m_T, on="match_id", how="left")
    df["match_T"] = np.maximum(T_NOMINAL, df["max_minute"].astype(float) + 1.0)

    goals = pd.concat(
        [
            df.loc[df["is_goal"], ["match_id", "side", "t", "order_idx"]],
            # own goal: credit the opposite side
            df.loc[is_own_goal, ["match_id", "side", "t", "order_idx"]].assign(
                side=lambda g: np.where(g["side"] == "H", "A", "H")
            ),
        ],
        ignore_index=True,
    )

    shots = df.loc[~is_own_goal].copy()  # an own goal is not a chance created
    shots = state_mod.reconstruct(shots, goals, cards)

    out = pd.DataFrame(
        {
            "match_id": shots["match_id"],
            "source": SOURCE,
            "league": shots["league"],
            "season": shots["season_year"],
            "date": pd.to_datetime(shots["date"]),
            "side": shots["side"],
            "team": shots["team"].astype(str),
            "opponent": shots["opponent"].astype(str),
            "t": shots["t"].astype(float),
            "minute": shots["minute"].astype("int16"),
            "second": shots["second"],
            "xg": shots["xg"].astype(float).clip(0.0, 1.0),
            "situation": shots["situation_std"],
            "is_goal": shots["is_goal"].astype(bool),
            "score_diff": shots["score_diff"],
            "red_diff": shots["red_diff"],
            "match_T": shots["match_T"].astype(float),
            "n_merged": 1,
            "red_cards_known": cards is not None,
        }
    )
    return schema.Slate(out, goals, cards)
