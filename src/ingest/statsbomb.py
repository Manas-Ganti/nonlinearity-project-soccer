"""StatsBomb open data, pulled as JSON straight from the public GitHub repo.

Volume is not what this source is for. It does two jobs (CLAUDE.md, "Data and
resolution"): calibrate the de-duplication threshold at second resolution, and
measure the rounding bias that Understat's integer clock induces.

We cache the *extracted* per-match rows rather than the raw JSON: the raw event
files are ~1.5 MB each and everything we need is a few hundred rows. The manifest
records the source URL and a digest of the raw bytes so the extraction is
reproducible and auditable.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import logging
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW
from src.features import state as state_mod
from src.ingest import schema

log = logging.getLogger(__name__)

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
RAW_DIR = RAW / "statsbomb"
SOURCE = "statsbomb"

# Every full-season league release plus the modern supplements, men's and women's.
# League quality is absorbed by the team effects in the background model; what
# has to be tracked is the stratum (competition, season, gender) so eta can be
# reported per stratum as a check on pooling. Seasons from 2019/20 onward are
# holdout and are pulled here but refused by the loader without --final-run.
DEFAULT_COMPETITIONS: tuple[tuple[int, int], ...] = (
    # ---- development era (<= 2018/19)
    (2, 27),  # Premier League 2015/16      -- 380
    (12, 27),  # Serie A 2015/16            -- 380
    (11, 27),  # La Liga 2015/16            -- 380
    (7, 27),  # Ligue 1 2015/16             -- 377
    (9, 27),  # 1. Bundesliga 2015/16       -- 34, one club
    (11, 37),  # La Liga 2004/05 .. 2018/19 -- Barcelona matches only, one club
    (11, 38),
    (11, 39),
    (11, 40),
    (11, 41),
    (11, 21),
    (11, 22),
    (11, 23),
    (11, 24),
    (11, 25),
    (11, 26),
    (11, 2),
    (11, 1),
    (11, 4),
    (37, 4),  # FA Women's Super League 2018/19 -- 107
    (49, 3),  # NWSL 2018                        -- 36
    # ---- holdout era (>= 2019/20): pulled, never read without --final-run
    (11, 42),  # La Liga 2019/20, one club
    (11, 90),  # La Liga 2020/21, one club
    (7, 108),  # Ligue 1 2021/22
    (7, 235),  # Ligue 1 2022/23
    (9, 281),  # 1. Bundesliga 2023/24
    (44, 107),  # MLS 2023
    (1238, 108),  # Indian Super League 2021/22
    (1267, 107),  # African Cup of Nations 2023
    (37, 42),  # FA WSL 2019/20
    (37, 90),  # FA WSL 2020/21
    (37, 281),  # FA WSL 2023/24
    (135, 281),  # Frauen Bundesliga 2023/24
    (182, 281),  # Liga F 2023/24
    (131, 281),  # Serie A Women 2023/24
    (49, 107),  # NWSL 2023
    (72, 107),  # Women's World Cup 2023
)

PERIOD_START_MINUTE = {1: 0, 2: 45, 3: 90, 4: 105}
SHOT_TYPE_MAP = {
    "Open Play": "open",
    "Penalty": "penalty",
    "Free Kick": "setpiece",
    "Corner": "setpiece",
    "Kick Off": "open",
}
RED_CARDS = {"Red Card", "Second Yellow"}


def _get(url: str, retries: int = 3, delay: float = 0.4) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as fh:
                return fh.read()
        except Exception as exc:
            last = exc
            time.sleep(delay * (2**attempt))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def list_matches(competitions=DEFAULT_COMPETITIONS) -> pd.DataFrame:
    gender = {
        (int(c["competition_id"]), int(c["season_id"])): c["competition_gender"]
        for c in json.loads(_get(f"{BASE}/competitions.json"))
    }
    rows = []
    for cid, sid in competitions:
        data = json.loads(_get(f"{BASE}/matches/{cid}/{sid}.json"))
        for m in data:
            rows.append(
                {
                    "match_id": int(m["match_id"]),
                    "competition_id": cid,
                    "season_id": sid,
                    "competition": m["competition"]["competition_name"],
                    "season_name": m["season"]["season_name"],
                    "gender": gender.get((cid, sid), "unknown"),
                    "date": m["match_date"],
                    "home_team": m["home_team"]["home_team_name"],
                    "away_team": m["away_team"]["away_team_name"],
                    "home_id": int(m["home_team"]["home_team_id"]),
                    "away_id": int(m["away_team"]["away_team_id"]),
                }
            )
    return pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)


def _extract(events: list[dict], meta: dict) -> pd.DataFrame:
    """One match's raw event list -> shot/goal/card rows on a continuous clock."""
    rows = []
    for e in events:
        period = int(e.get("period", 1))
        if period not in PERIOD_START_MINUTE:  # penalty shootout
            continue
        minute, second = int(e.get("minute", 0)), float(e.get("second", 0.0))
        local = (minute - PERIOD_START_MINUTE[period]) + second / 60.0
        team_id = e.get("team", {}).get("id")
        kind = e["type"]["name"]
        row = {
            "period": period,
            "local": local,
            "team_id": -1 if team_id is None else int(team_id),
            "kind": None,
            "xg": np.nan,
            "situation": None,
            "is_goal": False,
        }
        if kind == "Shot":
            shot = e.get("shot", {})
            row["kind"] = "shot"
            row["xg"] = float(shot.get("statsbomb_xg", np.nan))
            row["situation"] = SHOT_TYPE_MAP.get(shot.get("type", {}).get("name", ""), "open")
            row["is_goal"] = shot.get("outcome", {}).get("name") == "Goal"
        elif kind == "Own Goal Against":
            # Recorded for the team that conceded; the goal belongs to the other side.
            row["kind"] = "own_goal_conceded"
        elif (
            kind == "Foul Committed" and e.get("foul_committed", {}).get("card", {}).get("name") in RED_CARDS
        ) or (
            kind == "Bad Behaviour" and e.get("bad_behaviour", {}).get("card", {}).get("name") in RED_CARDS
        ):
            row["kind"] = "card"
        else:
            # Not an event we keep, but it still bounds the length of its period.
            row["kind"] = "clock"
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Contiguous clock: each period starts where the previous one ended.
    durations = df.groupby("period")["local"].max().to_dict()
    offsets, running = {}, 0.0
    for p in sorted(durations):
        offsets[p] = running
        running += durations[p]
    df["t"] = df["period"].map(offsets) + df["local"]
    match_T = float(running)

    df = df[df["kind"] != "clock"].copy()
    df["match_id"] = f"{SOURCE}:{meta['competition']}:{meta['season_name']}:{meta['match_id']}"
    df["sb_match_id"] = meta["match_id"]
    df["match_T"] = match_T
    df["home_team"] = meta["home_team"]
    df["away_team"] = meta["away_team"]
    df["date"] = meta["date"]
    df["competition"] = meta["competition"]
    df["season_name"] = meta["season_name"]
    df["gender"] = meta.get("gender", "unknown")
    # Sides by team *id*: the match file's names and the event file's names differ
    # for some teams ("Manchester United W" vs "Manchester United"), and a name
    # match silently puts every shot on one side.
    df["side"] = np.where(df["team_id"] == int(meta["home_id"]), "H", "A")
    df["team"] = np.where(df["side"] == "H", meta["home_team"], meta["away_team"])
    return df.sort_values(["t"], kind="stable").reset_index(drop=True)


def pull(
    competitions=DEFAULT_COMPETITIONS,
    *,
    max_matches: int | None = None,
    seed: int = 0,
    workers: int = 4,
    overwrite: bool = False,
) -> Path:
    """Fetch every match in `competitions` that is not already cached.

    Incremental by design: the cache is immutable input, so matches already on
    disk are kept exactly as they are and only the missing ones are fetched.
    `overwrite=True` discards the cache and starts from nothing.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "events.parquet"
    manifest_path = RAW_DIR / "manifest.json"

    existing: pd.DataFrame | None = None
    manifest: dict = {"base": BASE, "competitions": [], "seed": seed, "event_digests": {}}
    if out_path.exists() and not overwrite:
        existing = pd.read_parquet(out_path)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest.setdefault("event_digests", {})

    matches = list_matches(competitions)
    if max_matches is not None and len(matches) > max_matches:
        matches = matches.sample(max_matches, random_state=seed).sort_values("match_id")
    have = set(existing["sb_match_id"].astype(int)) if existing is not None else set()
    todo = matches[~matches["match_id"].isin(have)]
    log.info(
        "statsbomb: %d matches requested, %d cached, %d to fetch",
        len(matches),
        len(matches) - len(todo),
        len(todo),
    )
    if todo.empty:
        return out_path

    digests: dict[str, str] = {}

    def one(meta: dict) -> pd.DataFrame:
        url = f"{BASE}/events/{meta['match_id']}.json"
        raw = _get(url)
        digests[str(meta["match_id"])] = hashlib.sha256(raw).hexdigest()[:16]
        return _extract(json.loads(raw), meta)

    frames = [] if existing is None else [existing]
    metas = todo.to_dict("records")
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, frame in enumerate(pool.map(one, metas), start=1):
            if len(frame):
                frames.append(frame)
            if i % 50 == 0:
                log.info("  %d/%d", i, len(metas))

    df = pd.concat(frames, ignore_index=True)
    # Rows cached before `gender` was carried get it from the match listing.
    gender_of = matches.set_index("match_id")["gender"]
    df["gender"] = df["sb_match_id"].astype(int).map(gender_of).fillna(df.get("gender", "unknown"))
    df.to_parquet(out_path, index=False)
    seen = {tuple(c) for c in manifest["competitions"]}
    manifest["competitions"] = [list(c) for c in seen | {tuple(c) for c in competitions}]
    manifest["event_digests"].update(digests)
    manifest["n_matches"] = int(df["match_id"].nunique())
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("wrote %s (%d rows, %d matches)", out_path, len(df), df["match_id"].nunique())
    return out_path


def load_raw() -> pd.DataFrame:
    path = RAW_DIR / "events.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no StatsBomb pull at {path}. Run `make ingest-statsbomb`.")
    return pd.read_parquet(path)


def build_slate(raw: pd.DataFrame) -> schema.Slate:
    """Extracted StatsBomb rows -> the canonical event table, at second resolution."""
    df = raw.copy().sort_values(["match_id", "t"], kind="stable").reset_index(drop=True)
    df["order_idx"] = df.groupby("match_id").cumcount().astype("int64")

    goals = pd.concat(
        [
            df.loc[(df["kind"] == "shot") & df["is_goal"], ["match_id", "side", "t", "order_idx"]],
            df.loc[df["kind"] == "own_goal_conceded", ["match_id", "side", "t", "order_idx"]].assign(
                side=lambda g: np.where(g["side"] == "H", "A", "H")
            ),
        ],
        ignore_index=True,
    )
    cards = df.loc[df["kind"] == "card", ["match_id", "side", "t", "order_idx"]].copy()

    shots = df[df["kind"] == "shot"].copy()
    shots["xg"] = shots["xg"].astype(float).fillna(0.0).clip(0.0, 1.0)
    shots = state_mod.reconstruct(shots, goals, cards)

    home = shots["side"] == "H"
    out = pd.DataFrame(
        {
            "match_id": shots["match_id"],
            "source": SOURCE,
            "league": shots["competition"],
            "season": shots["season_name"].astype(str).str.slice(0, 4).astype("int16"),
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
            "gender": shots["gender"].astype(str) if "gender" in shots else "male",
        }
    )
    out["one_club_sample"] = schema.flag_one_club(out)
    return schema.Slate(out, goals, cards)
