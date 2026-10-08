"""Wyscout open data (Pappalardo et al. 2019), from the figshare release on disk.

Second-resolution event logs for the 2017/18 season of the five big leagues plus
the 2018 World Cup and Euro 2016: ~1,940 matches, CC BY 4.0. It is the second
independent second-clock provider next to StatsBomb, and the two are pooled for
the headline fit; a per-provider intercept in the background model absorbs the
difference in what each counts as a shot.

The raw release is seven `events_<competition>.json` files (one row per event,
`eventSec` from the start of its period) and seven `matches_<competition>.json`
files. `extract()` reads them once into a per-match row table with the same shape
as the StatsBomb extraction, so the dedup calibration and the rest of the build
treat the two identically.

What Wyscout does not have, and what is done about it:

- **No xG.** A location-based stand-in is fitted on Wyscout's own goal outcomes
  (`fit_location_xg`): logistic in log-distance, angle, header and set-piece.
  xG is a mark carried for the simulator and the secondary analyses; the
  headline fit does not use it. Penalties get the documented constant.
- **No set-piece origin beyond direct free-kick shots.** A shot in the phase
  after a corner is `open` here and `setpiece` on Understat/StatsBomb. Nothing
  downstream distinguishes the two; only `penalty` is load-bearing.
- **Goals** are the `Goal` tag on a shot-like event only. The same tag is also
  stamped on the goalkeeper's save attempt, so counting it on every event
  doubles the score. Own goals are the `own_goal` tag on the conceding team's
  event, credited to the other side. Reconstructed scorelines are checked
  against the official ones in `tests/test_wyscout.py`.
- **Dismissals** are the red / second-yellow tags on a foul.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import PENALTY_XG, RAW
from src.features import state as state_mod
from src.ingest import schema

log = logging.getLogger(__name__)

RAW_DIR = RAW / "wyscout"
SOURCE = "wyscout"

# File suffix -> (competition label, international?)
COMPETITIONS: dict[str, tuple[str, bool]] = {
    "England": ("Premier League", False),
    "Spain": ("La Liga", False),
    "Italy": ("Serie A", False),
    "Germany": ("1. Bundesliga", False),
    "France": ("Ligue 1", False),
    "European_Championship": ("UEFA Euro", True),
    "World_Cup": ("FIFA World Cup", True),
}

PERIOD_ORDER = {"1H": 0, "2H": 1, "E1": 2, "E2": 3}  # 'P' (shootout) is dropped
SHOT_SUBEVENTS = {"Shot": "open", "Free kick shot": "setpiece", "Penalty": "penalty"}
TAG_GOAL, TAG_OWN_GOAL, TAG_HEAD = 101, 102, 403
RED_TAGS = {1701, 1703}

PITCH_LENGTH_M, PITCH_WIDTH_M, GOAL_WIDTH_M = 105.0, 68.0, 7.32


def _season_of(date_utc: str, international: bool) -> int:
    ts = pd.Timestamp(date_utc)
    if international:
        return int(ts.year)
    return int(ts.year if ts.month >= 7 else ts.year - 1)


def _load_matches(suffix: str, teams: dict[int, str]) -> dict[int, dict]:
    label, international = COMPETITIONS[suffix]
    out = {}
    for m in json.loads((RAW_DIR / "matches" / f"matches_{suffix}.json").read_text()):
        sides = {v["side"]: int(k) for k, v in m["teamsData"].items()}
        final = {}
        for k, v in m["teamsData"].items():
            # `score` is the 90-minute score; `scoreET` the cumulative score after
            # extra time when it was played. Shootout goals are not goals.
            final[int(k)] = int(v["scoreET"] if m.get("duration") != "Regular" else v["score"])
        out[int(m["wyId"])] = {
            "match_id": int(m["wyId"]),
            "competition": label,
            "international": international,
            "season": _season_of(m["dateutc"], international),
            "date": pd.Timestamp(m["dateutc"]).normalize(),
            "home_id": sides["home"],
            "away_id": sides["away"],
            "home_team": teams[sides["home"]],
            "away_team": teams[sides["away"]],
            "final_home": final[sides["home"]],
            "final_away": final[sides["away"]],
        }
    return out


def _extract_match(events: list[dict], meta: dict) -> pd.DataFrame:
    """One match's raw events -> shot/goal/card rows on a continuous clock."""
    rows = []
    for e in events:
        period = e.get("matchPeriod")
        if period not in PERIOD_ORDER:
            continue
        tags = {t["id"] for t in e.get("tags", [])}
        sub = e.get("subEventName")
        row = {
            "period": PERIOD_ORDER[period],
            "local": float(e["eventSec"]) / 60.0,
            "team_id": int(e["teamId"]),
            "kind": "clock",
            "xg": np.nan,
            "situation": None,
            "is_goal": False,
            "x": np.nan,
            "y": np.nan,
            "is_header": False,
        }
        if sub in SHOT_SUBEVENTS:
            pos = e.get("positions") or [{}]
            row.update(
                kind="shot",
                situation=SHOT_SUBEVENTS[sub],
                is_goal=TAG_GOAL in tags,
                x=float(pos[0].get("x", np.nan)),
                y=float(pos[0].get("y", np.nan)),
                is_header=TAG_HEAD in tags,
            )
        elif TAG_OWN_GOAL in tags:
            row["kind"] = "own_goal_conceded"
        elif TAG_GOAL in tags and e.get("eventName") == "Save attempt":
            # The keeper's side of a conceded goal. Normally redundant with the
            # scoring shot; kept so a goal whose shot event is missing from the
            # feed (it happens) can still be recovered below.
            row["kind"] = "conceded_marker"
        elif TAG_GOAL in tags:
            # A goal on a non-shot event: a corner swung straight in, a deflected
            # cross. It is a goal for the score state, not a shot for the model.
            row["kind"] = "goal_nonshot"
        elif tags & RED_TAGS:
            row["kind"] = "card"
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = _recover_missing_goals(df, meta)

    durations = df.groupby("period")["local"].max().to_dict()
    offsets, running = {}, 0.0
    for p in sorted(durations):
        offsets[p] = running
        running += durations[p]
    df["t"] = df["period"].map(offsets) + df["local"]
    match_T = float(running)

    df = df[df["kind"] != "clock"].copy()
    df["match_id"] = f"{SOURCE}:{meta['competition']}:{meta['season']}:{meta['match_id']}"
    df["wy_match_id"] = meta["match_id"]
    df["match_T"] = match_T
    df["home_team"] = meta["home_team"]
    df["away_team"] = meta["away_team"]
    df["date"] = meta["date"]
    df["competition"] = meta["competition"]
    df["season"] = int(meta["season"])
    df["gender"] = "male"
    df["side"] = np.where(df["team_id"] == meta["home_id"], "H", "A")
    df["team"] = np.where(df["side"] == "H", meta["home_team"], meta["away_team"])
    df["final_home"] = meta["final_home"]
    df["final_away"] = meta["final_away"]
    return df.drop(columns=["team_id"]).sort_values("t", kind="stable").reset_index(drop=True)


def _recover_missing_goals(df: pd.DataFrame, meta: dict, window_s: float = 15.0) -> pd.DataFrame:
    """Keeper markers are redundant with the scoring event -- except when the feed
    has lost that event. The official final score decides: a side whose
    reconstructed total falls short gets its missing goals from the opponent
    keeper's unmatched markers, earliest first. Every other marker is dropped
    (a marker with no goal behind it is a disallowed goal or a tagging slip)."""
    scoring = df[df["kind"].isin(["shot", "goal_nonshot", "own_goal_conceded"])]
    scoring = scoring[(scoring["kind"] != "shot") | scoring["is_goal"]]
    markers = df.index[df["kind"] == "conceded_marker"]
    unmatched = []
    for i in markers:
        p, loc = df.at[i, "period"], df.at[i, "local"]
        near = scoring[(scoring["period"] == p) & ((scoring["local"] - loc).abs() * 60.0 <= window_s)]
        if near.empty:
            unmatched.append(i)

    # reconstructed goals credited to each team id
    credited: dict[int, int] = {meta["home_id"]: 0, meta["away_id"]: 0}
    for row in scoring.itertuples(index=False):
        tid = int(row.team_id)
        if row.kind == "own_goal_conceded":
            tid = meta["away_id"] if tid == meta["home_id"] else meta["home_id"]
        credited[tid] = credited.get(tid, 0) + 1
    official = {meta["home_id"]: meta["final_home"], meta["away_id"]: meta["final_away"]}

    promote = []
    for tid, want in official.items():
        short = want - credited.get(tid, 0)
        if short <= 0:
            continue
        other = meta["away_id"] if tid == meta["home_id"] else meta["home_id"]
        cands = [i for i in unmatched if int(df.at[i, "team_id"]) == other]
        cands.sort(key=lambda i: (df.at[i, "period"], df.at[i, "local"]))
        promote.extend(cands[:short])

    df = df[(df["kind"] != "conceded_marker") | df.index.isin(promote)].copy()
    df.loc[df["kind"] == "conceded_marker", "kind"] = "goal_conceded_unmatched"
    return df


def extract(*, overwrite: bool = False) -> Path:
    """Raw JSON -> `events.parquet` of shot/goal/card rows. Idempotent."""
    out_path = RAW_DIR / "events.parquet"
    if out_path.exists() and not overwrite:
        log.info("wyscout extraction present at %s", out_path)
        return out_path
    teams = {int(t["wyId"]): t["name"] for t in json.loads((RAW_DIR / "teams.json").read_text())}
    frames = []
    for suffix in COMPETITIONS:
        path = RAW_DIR / "events" / f"events_{suffix}.json"
        if not path.exists():
            log.warning("missing %s; skipping", path.name)
            continue
        metas = _load_matches(suffix, teams)
        by_match: dict[int, list[dict]] = {}
        for e in json.loads(path.read_text()):
            by_match.setdefault(int(e["matchId"]), []).append(e)
        n = 0
        for mid, evs in by_match.items():
            if mid not in metas:
                continue
            frame = _extract_match(evs, metas[mid])
            if len(frame):
                frames.append(frame)
                n += 1
        log.info("  %s: %d matches", suffix, n)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out_path, index=False)
    log.info("wrote %s (%d rows, %d matches)", out_path, len(df), df["match_id"].nunique())
    return out_path


def load_raw() -> pd.DataFrame:
    path = RAW_DIR / "events.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no Wyscout extraction at {path}. Run `make ingest-wyscout`.")
    return pd.read_parquet(path)


# ------------------------------------------------------------- xG stand-in


def shot_geometry(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance (m) and opening angle (rad) to goal from Wyscout's 0-100 coordinates.

    Wyscout attacks left to right, goal centre at (100, 50)."""
    dx = (100.0 - np.asarray(x, dtype=np.float64)) * PITCH_LENGTH_M / 100.0
    dy = (np.asarray(y, dtype=np.float64) - 50.0) * PITCH_WIDTH_M / 100.0
    dx = np.maximum(dx, 0.1)
    dist = np.hypot(dx, dy)
    half = GOAL_WIDTH_M / 2.0
    angle = np.arctan2(GOAL_WIDTH_M * dx, dx**2 + dy**2 - half**2)
    angle = np.where(angle < 0, angle + np.pi, angle)
    return dist, angle


def _features(shots: pd.DataFrame) -> np.ndarray:
    dist, angle = shot_geometry(shots["x"].to_numpy(), shots["y"].to_numpy())
    return np.column_stack(
        [
            np.ones(len(shots)),
            np.log(dist),
            angle,
            shots["is_header"].to_numpy(dtype=np.float64),
            (shots["situation"] == "setpiece").to_numpy(dtype=np.float64),
        ]
    )


def fit_location_xg(shots: pd.DataFrame) -> dict:
    """Logistic P(goal | log-distance, angle, header, set piece), non-penalty shots."""
    ok = (shots["situation"] != "penalty") & shots["x"].notna() & shots["y"].notna()
    X = _features(shots[ok])
    y = shots.loc[ok, "is_goal"].to_numpy(dtype=np.float64)

    def nll(w):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        val = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).sum()
        grad = X.T @ (p - y)
        return val, grad

    res = minimize(nll, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
    w = res.x
    return {
        "coef": dict(
            zip(["intercept", "log_dist", "angle", "header", "setpiece"], map(float, w), strict=True)
        ),
        "n_shots": int(ok.sum()),
        "goal_rate": float(y.mean()),
        "converged": bool(res.success),
    }


def apply_location_xg(shots: pd.DataFrame, model: dict) -> np.ndarray:
    w = np.array([model["coef"][k] for k in ("intercept", "log_dist", "angle", "header", "setpiece")])
    filled = shots.copy()
    filled["x"] = filled["x"].fillna(85.0)
    filled["y"] = filled["y"].fillna(50.0)
    xg = 1.0 / (1.0 + np.exp(-(_features(filled) @ w)))
    xg = np.where(shots["situation"].to_numpy() == "penalty", PENALTY_XG, xg)
    return np.clip(xg, 1e-4, 1 - 1e-4)


# ------------------------------------------------------------------ slate


def build_slate(raw: pd.DataFrame) -> tuple[schema.Slate, dict]:
    """Extracted Wyscout rows -> the canonical event table, at second resolution.

    Returns the slate and the fitted xG stand-in so the build manifest can carry it.
    """
    df = raw.copy().sort_values(["match_id", "t"], kind="stable").reset_index(drop=True)
    df["order_idx"] = df.groupby("match_id").cumcount().astype("int64")

    conceded = df["kind"].isin(["own_goal_conceded", "goal_conceded_unmatched"])
    goals = pd.concat(
        [
            df.loc[(df["kind"] == "shot") & df["is_goal"], ["match_id", "side", "t", "order_idx"]],
            df.loc[df["kind"] == "goal_nonshot", ["match_id", "side", "t", "order_idx"]],
            df.loc[conceded, ["match_id", "side", "t", "order_idx"]].assign(
                side=lambda g: np.where(g["side"] == "H", "A", "H")
            ),
        ],
        ignore_index=True,
    )
    cards = df.loc[df["kind"] == "card", ["match_id", "side", "t", "order_idx"]].copy()

    shots = df[df["kind"] == "shot"].copy()
    xg_model = fit_location_xg(shots)
    shots["xg"] = apply_location_xg(shots, xg_model)
    shots = state_mod.reconstruct(shots, goals, cards)

    home = shots["side"] == "H"
    out = pd.DataFrame(
        {
            "match_id": shots["match_id"],
            "source": SOURCE,
            "league": shots["competition"],
            "season": shots["season"].astype("int16"),
            "date": pd.to_datetime(shots["date"]),
            "side": shots["side"],
            "team": shots["team"].astype(str),
            "opponent": np.where(home, shots["away_team"], shots["home_team"]),
            "t": shots["t"].astype(float),
            "minute": np.floor(shots["t"]).astype("int16"),
            "second": (shots["t"] % 1.0) * 60.0,
            "xg": shots["xg"],
            "situation": shots["situation"],
            "is_goal": shots["is_goal"].astype(bool),
            "score_diff": shots["score_diff"],
            "red_diff": shots["red_diff"],
            "match_T": shots["match_T"].astype(float),
            "n_merged": 1,
            "red_cards_known": True,
            "gender": "male",
        }
    )
    out["one_club_sample"] = schema.flag_one_club(out)
    return schema.Slate(out, goals, cards), xg_model


def official_scores(raw: pd.DataFrame) -> pd.DataFrame:
    """Per match, the official final score carried in the match file."""
    return (
        raw.groupby("match_id")
        .agg(final_home=("final_home", "first"), final_away=("final_away", "first"))
        .reset_index()
    )
