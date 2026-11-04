"""Possession table from StatsBomb full event files (docs/spec_possessions.md section 9.3).

The independent replication of the Wyscout possession table. The *same* code
groups events into possessions and summarises them (`ingest.possessions`
`assign_possessions` and `summarise`). Only the mapping from StatsBomb events to
the Wyscout-shaped on-ball frame lives here, and it is fixed by the spec:

- **On-ball events:** Pass, Shot, Carry, Dribble, Clearance, Miscontrol.
- **Accurate pass:** a pass with no `outcome`.
- **Set pieces:** pass type Corner / Free Kick / Throw-in, shot type Free Kick /
  Penalty. Goal kicks are restarts but not set pieces, as for Wyscout.
- **Pitch:** StatsBomb is 120 x 80 with every event in its own team's attacking
  direction. Coordinates are rescaled to Wyscout's 0-100, so the final third
  (x >= 80) lands on the same 66.7 line. The box differs slightly after rescaling
  (x >= 85 against Wyscout's 84); it is only a sensitivity.
- **Clock:** the shot extraction's own rule (`statsbomb._extract`): integer
  minute and second, each period laid end to end after the previous one's last
  event. So possessions and shots share one clock.
- **Game state:** from the StatsBomb development shot slate's goal and card tables.

Development matches only. The raw cache holds no holdout match, and the loader
refuses one anyway.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import FIRST_HOLDOUT_SEASON, INTERIM
from src.ingest import loader, possessions, statsbomb, statsbomb_events
from src.runlog import append_event

log = logging.getLogger(__name__)

OUT_DIR = INTERIM / "statsbomb_possessions"
SOURCE = statsbomb.SOURCE
ONBALL = {"Pass", "Shot", "Carry", "Dribble", "Clearance", "Miscontrol"}
SX, SY = 100.0 / 120.0, 100.0 / 80.0

PASS_RESTART = {
    "Corner": "Corner",
    "Free Kick": "Free Kick",
    "Throw-in": "Throw in",
    "Goal Kick": "Goal kick",
}
SHOT_RESTART = {"Free Kick": "Free kick shot", "Penalty": "Penalty"}


def _local(e: dict) -> float | None:
    period = int(e.get("period", 1))
    if period not in statsbomb.PERIOD_START_MINUTE:  # shootout
        return None
    return (int(e.get("minute", 0)) - statsbomb.PERIOD_START_MINUTE[period]) + float(
        e.get("second", 0.0)
    ) / 60.0


def clock(raw: list[dict]) -> tuple[dict[int, float], dict[int, float]]:
    """Per-period offsets and end times, exactly as `statsbomb._extract` lays them out."""
    durations: dict[int, float] = {}
    for e in raw:
        loc = _local(e)
        if loc is None:
            continue
        p = int(e["period"])
        durations[p] = max(durations.get(p, 0.0), loc)
    offsets, ends, running = {}, {}, 0.0
    for p in sorted(durations):
        offsets[p] = running
        running += durations[p]
        ends[p] = running
    return offsets, ends


def onball_events(raw: list[dict], offsets: dict[int, float]) -> pd.DataFrame:
    """StatsBomb events -> the Wyscout-shaped on-ball frame `possessions` expects."""
    rows = []
    for e in raw:
        kind = e["type"]["name"]
        loc = _local(e)
        if kind not in ONBALL or loc is None or "location" not in e:
            continue
        p = int(e["period"])
        x0, y0 = e["location"][0] * SX, e["location"][1] * SY
        x1 = y1 = np.nan
        name, sub, accurate = "Others on the ball", kind, False
        if kind == "Pass":
            ps = e.get("pass", {})
            ptype = ps.get("type", {}).get("name")
            accurate = "outcome" not in ps
            if "end_location" in ps:
                x1, y1 = ps["end_location"][0] * SX, ps["end_location"][1] * SY
            if ptype in PASS_RESTART:
                name, sub = "Free Kick", PASS_RESTART[ptype]
            else:
                name, sub = "Pass", "Simple pass"
        elif kind == "Shot":
            stype = e.get("shot", {}).get("type", {}).get("name")
            if stype in SHOT_RESTART:
                name, sub = "Free Kick", SHOT_RESTART[stype]
            else:
                name, sub = "Shot", "Shot"
        # p is 1-based in StatsBomb; possessions.summarise only needs it consistent
        rows.append(
            (p, offsets[p] + loc, int(e["index"]), int(e["team"]["id"]), name, sub, accurate, x0, y0, x1, y1)
        )
    df = pd.DataFrame(
        rows, columns=["period", "t", "index", "team_id", "name", "sub", "accurate", "x0", "y0", "x1", "y1"]
    )
    return (
        df.sort_values(["period", "t", "index"], kind="stable").drop(columns="index").reset_index(drop=True)
    )


def build_match(raw: list[dict], meta: dict, goals: pd.DataFrame, cards: pd.DataFrame) -> pd.DataFrame:
    offsets, ends = clock(raw)
    ob = onball_events(raw, offsets)
    frames = []
    for rule in ("tolerant", "strict"):
        p = possessions.summarise(possessions.assign_possessions(ob, tolerant=(rule == "tolerant")), ends)
        p["rule"] = rule
        frames.append(p)
    out = pd.concat(frames, ignore_index=True)
    out["side"] = np.where(out["owner"] == int(meta["home_id"]), "H", "A")
    out["team"] = np.where(out["side"] == "H", meta["home_team"], meta["away_team"])
    out["opponent"] = np.where(out["side"] == "H", meta["away_team"], meta["home_team"])
    out["score_diff"], out["red_diff"] = possessions.state_at(
        out["t_start"].to_numpy(), out["side"].to_numpy(), goals, cards
    )
    out["match_id"] = meta["slate_match_id"]
    out["league"] = meta["competition"]
    out["season"] = int(str(meta["season_name"])[:4])
    out["source"] = SOURCE
    out["match_T"] = max(ends.values())
    return out.drop(columns="owner")


def build() -> tuple[pd.DataFrame, dict]:
    """Every StatsBomb development match in the shot slate -> the possession table."""
    slate = loader.load_slate("statsbomb", split="dev", caller="build-possessions-statsbomb")
    ev = slate.events
    keep = ev.drop_duplicates("match_id").set_index("match_id")
    goals_by = dict(tuple(slate.goals.groupby("match_id")))
    cards_by = dict(tuple(slate.cards.groupby("match_id")))
    empty = slate.goals.iloc[0:0]
    one_club = keep["one_club_sample"].astype(bool).to_dict()

    # Home/away ids from the match listings (the shot rows carry side, not the id).
    listing = statsbomb.list_matches().set_index("match_id")
    raw_rows = pd.read_parquet(statsbomb.RAW_DIR / "events.parquet", columns=["sb_match_id", "match_id"])
    slate_id_of = raw_rows.drop_duplicates("sb_match_id").set_index("sb_match_id")["match_id"].to_dict()

    frames, missing = [], []
    for sb_id in statsbomb_events.dev_match_ids():
        mid = slate_id_of.get(sb_id)
        if mid not in keep.index:
            continue
        if not statsbomb_events.path_for(sb_id).exists():
            missing.append(sb_id)
            continue
        lm = listing.loc[sb_id]
        meta = {
            "slate_match_id": mid,
            "home_id": lm["home_id"],
            "home_team": lm["home_team"],
            "away_team": lm["away_team"],
            "competition": lm["competition"],
            "season_name": lm["season_name"],
        }
        frames.append(
            build_match(
                statsbomb_events.load_match(sb_id), meta, goals_by.get(mid, empty), cards_by.get(mid, empty)
            )
        )
        if len(frames) % 250 == 0:
            log.info("  %d matches", len(frames))
    if missing:
        raise RuntimeError(f"{len(missing)} development matches have no event file; finish the pull first")
    table = pd.concat(frames, ignore_index=True)
    # Team names must agree with the shot slate's, or team effects split in two.
    table["team"] = table["team"].astype(str)
    table["opponent"] = table["opponent"].astype(str)
    table["one_club_sample"] = table["match_id"].map(one_club).fillna(False).astype(bool)
    if (table["season"] >= FIRST_HOLDOUT_SEASON).any():
        raise RuntimeError("StatsBomb possessions reached a holdout season")
    return table, possessions.describe(table, ev)


def save(table: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_DIR / "possessions.parquet", index=False)


def load(
    *, split: str = "dev", final_run: bool = False, rule: str = "tolerant", caller: str = "unspecified"
) -> pd.DataFrame:
    """The StatsBomb possession table, behind the same holdout guard and audit log."""
    table = pd.read_parquet(OUT_DIR / "possessions.parquet")
    seasons = table["season"].astype(int)
    if split == "dev":
        keep = seasons < FIRST_HOLDOUT_SEASON
    elif split in ("holdout", "all"):
        keep = seasons >= FIRST_HOLDOUT_SEASON if split == "holdout" else pd.Series(True, index=table.index)
    else:
        raise ValueError(f"unknown split {split!r}")
    touches = split in ("holdout", "all") and bool((seasons >= FIRST_HOLDOUT_SEASON).any())
    append_event(
        loader.HOLDOUT_LOG,
        f"source=statsbomb_possessions split={split} final_run={final_run} caller={caller} "
        f"touches_holdout={touches} matches={table.loc[keep, 'match_id'].nunique()}",
    )
    if touches and not final_run:
        raise loader.HoldoutViolation(
            f"split={split!r} would return holdout seasons; pass --final-run to mean it"
        )
    return table[keep & (table["rule"] == rule)].reset_index(drop=True)
