"""Possession table from the Wyscout event streams (docs/spec_possessions.md section 2).

The shot slate keeps only shots, goals and cards. The richer-events analysis needs
every on-ball event, grouped into possessions, with the time each team has the ball.
Wyscout logs no possession id, so possessions are rebuilt here.

Two rules are built, and both go in the table under `rule`:

- **tolerant** (the analysis default): a run of one team's on-ball events within a
  period, where a single opponent event between two runs of the same team is a
  *touch* and does not end the possession -- unless it is an accurate pass or a
  shot. The feasibility pilot showed that without this, deflections and clearances
  won straight back split one attack into several (308 "possessions" a match).
- **strict** (a sensitivity): any change of team ends the possession.

Own-possession time runs from a possession's first event to the next possession's
first event, so P_H(t) + P_A(t) = 1 from each period's first on-ball event to its
end. That interval, not the span of the possession's own events, is what a
possession-aware intensity integrates over: a one-event possession still holds
the ball until the opponent's first touch.

**One clock with the shots.** Period offsets are read from the shot extraction
(`wyscout._extract_match`), not recomputed. The two disagree otherwise: the
extraction drops redundant keeper goal markers before measuring each half, so a
half whose last event was such a marker is shorter there. The feasibility pilot
found four Premier League matches off by 1-4 s that way.

**Game state** at each possession start comes from the shot slate's own goal and
card tables: goals strictly before t, dismissals at or before t, the same
precedence as `features.state.reconstruct`.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.config import FIRST_HOLDOUT_SEASON, INTERIM
from src.features.state import RED_CLIP, SCORE_CLIP
from src.ingest import loader, wyscout
from src.runlog import append_event

log = logging.getLogger(__name__)

OUT_DIR = INTERIM / "wyscout_possessions"
ONBALL = ("Pass", "Free Kick", "Shot", "Others on the ball")
TAG_ACCURATE = 1801

FINAL_THIRD_X = 200.0 / 3.0  # Wyscout pitch: 0-100 both axes, attacking towards x = 100
DEEP_FINAL_THIRD_X = 75.0  # sensitivity
BOX_X, BOX_Y = 84.0, (19.0, 81.0)

# Restarts that set up an attack. Goal kicks restart from the team's own box and
# are not counted.
SETPIECE_SUBS = ("Corner", "Free Kick", "Free kick cross", "Free kick shot", "Throw in", "Penalty")

START_TYPES = {
    "Throw in": "throw_in",
    "Goal kick": "goal_kick",
    "Corner": "corner",
    "Free Kick": "free_kick",
    "Free kick cross": "free_kick",
    "Free kick shot": "free_kick",
    "Penalty": "penalty",
}


def onball_events(raw: list[dict], offsets: dict[int, float]) -> pd.DataFrame:
    """One match's raw events -> its on-ball events, on the shot clock, in order."""
    rows = []
    for e in raw:
        period = e.get("matchPeriod")
        if period not in wyscout.PERIOD_ORDER or e.get("eventName") not in ONBALL:
            continue
        pos = e.get("positions") or []
        p0 = pos[0] if pos else {}
        p1 = pos[1] if len(pos) > 1 else {}
        p = wyscout.PERIOD_ORDER[period]
        rows.append(
            (
                p,
                offsets[p] + float(e["eventSec"]) / 60.0,
                int(e["teamId"]),
                e["eventName"],
                e.get("subEventName") or "",
                TAG_ACCURATE in {t["id"] for t in e.get("tags", [])},
                float(p0.get("x", np.nan)),
                float(p0.get("y", np.nan)),
                float(p1.get("x", np.nan)),
                float(p1.get("y", np.nan)),
            )
        )
    df = pd.DataFrame(
        rows, columns=["period", "t", "team_id", "name", "sub", "accurate", "x0", "y0", "x1", "y1"]
    )
    return df.sort_values(["period", "t"], kind="stable").reset_index(drop=True)


def clock(raw: list[dict], meta: dict) -> tuple[dict[int, float], dict[int, float]]:
    """Per-period start offset and end time, exactly as the shot extraction has them."""
    ext = wyscout._extract_match(raw, meta)
    present = sorted(
        {wyscout.PERIOD_ORDER[e["matchPeriod"]] for e in raw if e.get("matchPeriod") in wyscout.PERIOD_ORDER}
    )
    offsets: dict[int, float] = {}
    if len(ext):
        for p, g in ext.groupby("period"):
            offsets[int(p)] = float((g["t"] - g["local"]).iloc[0])
    # A period with no shot, goal or card row: lay it end to end after the last
    # known one, using its own last event (the extraction's rule, minus markers).
    local_max: dict[int, float] = {}
    for e in raw:
        p = wyscout.PERIOD_ORDER.get(e.get("matchPeriod"))
        if p is not None:
            local_max[p] = max(local_max.get(p, 0.0), float(e["eventSec"]) / 60.0)
    for p in present:
        if p not in offsets:
            prev = [q for q in present if q < p]
            offsets[p] = 0.0 if not prev else offsets[prev[-1]] + local_max[prev[-1]]
    match_T = float(ext["match_T"].iloc[0]) if len(ext) else offsets[present[-1]] + local_max[present[-1]]
    ends = {p: (offsets[present[i + 1]] if i + 1 < len(present) else match_T) for i, p in enumerate(present)}
    return offsets, ends


def assign_possessions(ob: pd.DataFrame, *, tolerant: bool) -> pd.DataFrame:
    """Add `owner` (team holding the ball), `is_touch` and `poss` (possession number)."""
    ob = ob.copy()
    new_run = (ob["team_id"] != ob["team_id"].shift()) | (ob["period"] != ob["period"].shift())
    run = new_run.cumsum()
    ob["is_touch"] = False
    if tolerant and len(ob):
        first = ob.groupby(run).head(1).set_index(run[new_run])
        r = pd.DataFrame(
            {
                "team": first["team_id"],
                "period": first["period"],
                "n": ob.groupby(run).size(),
                "keeps": (first["name"] == "Pass") & first["accurate"]
                | first["sub"].isin(wyscout.SHOT_SUBEVENTS),
            }
        )
        same_period = (r["period"] == r["period"].shift()) & (r["period"] == r["period"].shift(-1))
        touch = (r["n"] == 1) & ~r["keeps"] & (r["team"].shift() == r["team"].shift(-1)) & same_period
        ob["is_touch"] = run.map(touch).to_numpy(dtype=bool)
    ob["owner"] = ob["team_id"].where(~ob["is_touch"]).ffill().astype("int64")
    ob["poss"] = ((ob["owner"] != ob["owner"].shift()) | (ob["period"] != ob["period"].shift())).cumsum() - 1
    return ob


def _in_box(x, y):
    return (x >= BOX_X) & (y >= BOX_Y[0]) & (y <= BOX_Y[1])


def summarise(ob: pd.DataFrame, ends: dict[int, float]) -> pd.DataFrame:
    """Possession-level rows: timing, own-possession interval, start type, danger times."""
    own = ob[~ob["is_touch"]]
    passlike = own["name"].isin(["Pass", "Free Kick"]) & own["accurate"]
    flags = pd.DataFrame(
        {
            "poss": own["poss"],
            "t": own["t"],
            "ft": (own["x0"] >= FINAL_THIRD_X) | (passlike & (own["x1"] >= FINAL_THIRD_X)),
            "ft75": (own["x0"] >= DEEP_FINAL_THIRD_X) | (passlike & (own["x1"] >= DEEP_FINAL_THIRD_X)),
            "box": _in_box(own["x0"], own["y0"]) | (passlike & _in_box(own["x1"], own["y1"])),
            "shot": own["sub"].isin(wyscout.SHOT_SUBEVENTS),
            "setpiece": own["sub"].isin(SETPIECE_SUBS),
        }
    )
    g = ob.groupby("poss", sort=True)
    out = pd.DataFrame(
        {
            "period": g["period"].first(),
            "owner": g["owner"].first(),
            "t_start": g["t"].min(),
            "t_last": g["t"].max(),
            "n_events": g.size(),
            "start_sub": own.groupby("poss")["sub"].first(),
        }
    )
    nxt = out["t_start"].shift(-1)
    last_in_period = out["period"] != out["period"].shift(-1)
    out["t_next"] = np.where(last_in_period, out["period"].map(ends), nxt)
    out["start_type"] = out["start_sub"].map(START_TYPES).fillna("open")
    sp = flags.loc[flags["setpiece"]].groupby("poss")["t"].min()
    for col in ("ft", "ft75", "box"):
        out[f"t_{col}"] = flags.loc[flags[col]].groupby("poss")["t"].min()
        # A set-piece restart at or before the first danger moment (any, if the
        # possession never gets there). A corner won *by* an attack that already
        # reached the final third must not count: that would encode the outcome.
        first_sp = sp.reindex(out.index)
        out[f"sp_{col}"] = first_sp.notna() & (out[f"t_{col}"].isna() | (first_sp <= out[f"t_{col}"]))
    out["n_shots"] = flags.groupby("poss")["shot"].sum().reindex(out.index, fill_value=0).astype("int64")
    return out.drop(columns="start_sub").reset_index()


def state_at(
    t: np.ndarray, side: np.ndarray, goals: pd.DataFrame, cards: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Score and red-card differential from `side`'s view, for one match.

    Goals count strictly before t (a scoring shot sees the score before it); a
    dismissal counts from its own instant (it takes effect before a shot then).
    Clipped to the same bins as `features.state` (score ±2, red ±1)."""
    t = np.asarray(t, dtype=np.float64)
    home = np.asarray(side) == "H"

    def count(tab, s, strict):
        times = np.sort(tab.loc[tab["side"] == s, "t"].to_numpy(dtype=np.float64))
        return np.searchsorted(times, t, side="left" if strict else "right")

    gd = count(goals, "H", True) - count(goals, "A", True)
    # cards: `side` is the side that LOST a player, so a home dismissal is -1 for home
    rd = count(cards, "A", False) - count(cards, "H", False)
    sd, rd = np.where(home, gd, -gd), np.where(home, rd, -rd)
    return np.clip(sd, -SCORE_CLIP, SCORE_CLIP), np.clip(rd, -RED_CLIP, RED_CLIP)


def build_match(raw: list[dict], meta: dict, goals: pd.DataFrame, cards: pd.DataFrame) -> pd.DataFrame:
    offsets, ends = clock(raw, meta)
    ob = onball_events(raw, offsets)
    match_id = f"{wyscout.SOURCE}:{meta['competition']}:{meta['season']}:{meta['match_id']}"
    frames = []
    for rule in ("tolerant", "strict"):
        p = summarise(assign_possessions(ob, tolerant=(rule == "tolerant")), ends)
        p["rule"] = rule
        frames.append(p)
    out = pd.concat(frames, ignore_index=True)
    out["side"] = np.where(out["owner"] == meta["home_id"], "H", "A")
    out["team"] = np.where(out["side"] == "H", meta["home_team"], meta["away_team"])
    out["opponent"] = np.where(out["side"] == "H", meta["away_team"], meta["home_team"])
    out["score_diff"], out["red_diff"] = state_at(
        out["t_start"].to_numpy(), out["side"].to_numpy(), goals, cards
    )
    out["match_id"] = match_id
    out["league"] = meta["competition"]
    out["season"] = int(meta["season"])
    out["source"] = wyscout.SOURCE
    out["match_T"] = max(ends.values())
    return out.drop(columns="owner")


def build() -> tuple[pd.DataFrame, dict]:
    """Every Wyscout match in the development shot slate -> the possession table."""
    slate = loader.load_slate("wyscout", split="dev", caller="build-possessions")
    keep = set(slate.events["match_id"].unique())
    goals_by = dict(tuple(slate.goals.groupby("match_id")))
    cards_by = dict(tuple(slate.cards.groupby("match_id")))
    empty = slate.goals.iloc[0:0]

    teams = {int(t["wyId"]): t["name"] for t in json.loads((wyscout.RAW_DIR / "teams.json").read_text())}
    frames = []
    for suffix in wyscout.COMPETITIONS:
        metas = wyscout._load_matches(suffix, teams)
        by_match: dict[int, list[dict]] = {}
        for e in json.loads((wyscout.RAW_DIR / "events" / f"events_{suffix}.json").read_text()):
            by_match.setdefault(int(e["matchId"]), []).append(e)
        n = 0
        for wy_id, evs in by_match.items():
            meta = metas.get(wy_id)
            if meta is None:
                continue
            mid = f"{wyscout.SOURCE}:{meta['competition']}:{meta['season']}:{wy_id}"
            if mid not in keep:
                continue
            frames.append(build_match(evs, meta, goals_by.get(mid, empty), cards_by.get(mid, empty)))
            n += 1
        log.info("  %s: %d matches", suffix, n)
    table = pd.concat(frames, ignore_index=True)
    if (table["season"] >= FIRST_HOLDOUT_SEASON).any():
        raise RuntimeError("Wyscout possessions reached a holdout season; the release should be 2016-18 only")
    return table, describe(table, slate.events)


def describe(table: pd.DataFrame, shots: pd.DataFrame) -> dict:
    out: dict = {
        "matches": int(table["match_id"].nunique()),
        "seasons": sorted(table["season"].unique().tolist()),
    }
    for rule, g in table.groupby("rule"):
        tm = g.groupby(["match_id", "side"])
        own_time = (g["t_next"] - g["t_start"]).groupby(g["match_id"]).sum()
        out[rule] = {
            "possessions_per_match": float(g.groupby("match_id").size().mean()),
            "dangerous_final_third_per_team_match": float(tm["t_ft"].count().mean()),
            "dangerous_box_per_team_match": float(tm["t_box"].count().mean()),
            "frac_with_shot": float((g["n_shots"] > 0).mean()),
            "shots_in_table": int(g["n_shots"].sum()),
            "own_time_over_match_T_min": float((own_time / g.groupby("match_id")["match_T"].first()).min()),
            "start_types": g["start_type"].value_counts().to_dict(),
        }
    out["shots_in_slate"] = len(shots)
    return out


def save(table: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_DIR / "possessions.parquet", index=False)


def load(
    *, split: str = "dev", final_run: bool = False, rule: str = "tolerant", caller: str = "unspecified"
) -> pd.DataFrame:
    """The possession table, behind the same holdout guard and audit log as shots."""
    table = pd.read_parquet(OUT_DIR / "possessions.parquet")
    seasons = table["season"].astype(int)
    if split == "dev":
        keep = seasons < FIRST_HOLDOUT_SEASON
    elif split == "holdout":
        keep = seasons >= FIRST_HOLDOUT_SEASON
    elif split == "all":
        keep = pd.Series(True, index=table.index)
    else:
        raise ValueError(f"unknown split {split!r}")
    touches = split in ("holdout", "all") and bool((seasons >= FIRST_HOLDOUT_SEASON).any())
    append_event(
        loader.HOLDOUT_LOG,
        f"source=wyscout_possessions split={split} final_run={final_run} caller={caller} "
        f"touches_holdout={touches} matches={table.loc[keep, 'match_id'].nunique()}",
    )
    if touches and not final_run:
        raise loader.HoldoutViolation(
            f"split={split!r} would return holdout seasons; pass --final-run to mean it"
        )
    return table[keep & (table["rule"] == rule)].reset_index(drop=True)
