"""Feasibility pilot: are Wyscout "dangerous possessions" a usable event?

Not part of the build order and not a momentum analysis. Nothing here estimates
whether one dangerous possession makes another more likely; it measures only what
a richer-events model would have to work with:

1. possession reconstruction (Wyscout has no possession id), under two rules
2. dangerous possessions per team-match, final-third and box definitions
3. how many shots fall inside a possession flagged dangerous
4. where mechanical clustering ends, with the same hazard calibration as shot dedup
5. data quality, and agreement of the rebuilt clock with the shot slate

Development data only (Wyscout is 2016-18). The match list comes through the
loader, so the read is in logs/holdout_access.log.

    PYTHONPATH=. .venv/bin/python scripts/pilot_wyscout_possessions.py
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.features import dedup
from src.ingest import loader, wyscout
from src.runlog import write_manifest

log = logging.getLogger("pilot")

ONBALL = {"Pass", "Free Kick", "Shot", "Others on the ball"}
TAG_ACCURATE = 1801
FINAL_THIRD_X = 200.0 / 3.0
BOX_X, BOX_Y = 84.0, (19.0, 81.0)  # Wyscout pitch is 0-100 on both axes, attacking to x = 100


def in_box(x, y):
    return (x >= BOX_X) & (y >= BOX_Y[0]) & (y <= BOX_Y[1])


def match_events(raw: list[dict]) -> pd.DataFrame:
    """One match's raw events on the shot slate's continuous clock."""
    rows = []
    for e in raw:
        period = e.get("matchPeriod")
        if period not in wyscout.PERIOD_ORDER:
            continue
        pos = e.get("positions") or []
        p0 = pos[0] if pos else {}
        p1 = pos[1] if len(pos) > 1 else {}
        rows.append(
            (
                wyscout.PERIOD_ORDER[period],
                float(e["eventSec"]) / 60.0,
                int(e["teamId"]),
                e.get("eventName"),
                e.get("subEventName"),
                TAG_ACCURATE in {t["id"] for t in e.get("tags", [])},
                p0.get("x", np.nan),
                p0.get("y", np.nan),
                p1.get("x", np.nan),
                p1.get("y", np.nan),
                len(pos),
            )
        )
    df = pd.DataFrame(
        rows,
        columns=["period", "local", "team_id", "name", "sub", "accurate", "x0", "y0", "x1", "y1", "npos"],
    )
    # Same offsets as wyscout._extract_match: each period's length is its last event.
    durations = df.groupby("period")["local"].max().sort_index()
    offsets = durations.cumsum().shift(1, fill_value=0.0)
    df["t"] = df["period"].map(offsets) + df["local"]
    return df


def possessions(ev: pd.DataFrame, *, tolerant: bool) -> pd.DataFrame:
    """On-ball events -> possession id. Strict: any change of team ends a possession.
    Tolerant: a single opponent event that is not an accurate pass, between two runs
    of the same team, is a touch and does not."""
    ob = ev[ev["name"].isin(ONBALL)].sort_values(["period", "local"], kind="stable").reset_index(drop=True)
    run = ((ob["team_id"] != ob["team_id"].shift()) | (ob["period"] != ob["period"].shift())).cumsum()
    if tolerant:
        r = ob.groupby(run).agg(
            team=("team_id", "first"),
            period=("period", "first"),
            n=("team_id", "size"),
            acc_pass=("accurate", lambda s: bool(s.iloc[0]) and ob.loc[s.index[0], "name"] == "Pass"),
            shot=("sub", lambda s: s.iloc[0] in wyscout.SHOT_SUBEVENTS),
        )
        prev_t, next_t = r["team"].shift(), r["team"].shift(-1)
        same_p = (r["period"] == r["period"].shift()) & (r["period"] == r["period"].shift(-1))
        touch = (r["n"] == 1) & ~r["acc_pass"] & ~r["shot"] & (prev_t == next_t) & same_p
        ob["is_touch"] = run.map(touch).to_numpy()
        # A touch takes the possessing team's id, then runs are recomputed.
        owner = ob["team_id"].where(~ob["is_touch"], np.nan).ffill()
        ob["owner"] = owner.astype(int)
        run = ((ob["owner"] != ob["owner"].shift()) | (ob["period"] != ob["period"].shift())).cumsum()
    else:
        ob["is_touch"] = False
        ob["owner"] = ob["team_id"]
    ob["poss"] = run.to_numpy()
    return ob


def flag_dangerous(ob: pd.DataFrame) -> pd.DataFrame:
    """Per possession: first time it reached the final third / the box, and whether
    it contains a shot. Only the possessing team's own events count."""
    own = ob[~ob["is_touch"]]
    passlike = own["name"].isin(["Pass", "Free Kick"]) & own["accurate"]
    ft = (own["x0"] >= FINAL_THIRD_X) | (passlike & (own["x1"] >= FINAL_THIRD_X))
    bx = in_box(own["x0"], own["y0"]) | (passlike & in_box(own["x1"], own["y1"]))
    shot = own["sub"].isin(wyscout.SHOT_SUBEVENTS)
    g = own.assign(ft=ft, bx=bx, shot=shot).groupby("poss")
    out = pd.DataFrame(
        {
            "owner": g["owner"].first(),
            "period": g["period"].first(),
            "t_start": g["t"].min(),
            "t_end": g["t"].max(),
            "n_events": g.size(),
            "has_shot": g["shot"].any(),
            "n_shots": g["shot"].sum(),
        }
    )
    for col in ("ft", "bx"):
        first = own.assign(ft=ft, bx=bx).loc[lambda d, c=col: d[c]].groupby("poss")["t"].min()
        out[f"t_{col}"] = first
    return out.reset_index()


def gaps_by_team(df: pd.DataFrame, tcol: str) -> np.ndarray:
    d = df.dropna(subset=[tcol]).sort_values(["match_id", "owner", tcol], kind="stable")
    g = d.groupby(["match_id", "owner"], sort=False)
    gap = (d[tcol] - g[tcol].shift()) * 60.0
    same = d["period"].eq(g["period"].shift())
    return gap[same].dropna().to_numpy()


def per_team_match(df: pd.DataFrame, tcol: str, ids: pd.DataFrame) -> pd.Series:
    n = df.dropna(subset=[tcol]).groupby(["match_id", "owner"]).size()
    return n.reindex(ids.index, fill_value=0)


def summary(s: pd.Series) -> dict:
    return {
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)),
        "p10": float(s.quantile(0.1)),
        "p50": float(s.median()),
        "p90": float(s.quantile(0.9)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    slate = loader.load_slate("wyscout", split="dev", caller="pilot-possessions")
    shots = slate.events
    shots = shots.assign(wy_id=shots["match_id"].str.rsplit(":", n=1).str[-1].astype(int))
    want = set(shots["wy_id"].unique()) | set(
        int(m.rsplit(":", 1)[-1]) for m in slate.goals["match_id"].unique()
    )
    log.info("dev slate: %d matches, %d shots", shots["match_id"].nunique(), len(shots))

    teams = {int(t["wyId"]): t["name"] for t in json.loads((wyscout.RAW_DIR / "teams.json").read_text())}
    meta = {}
    for suffix in wyscout.COMPETITIONS:
        meta.update(wyscout._load_matches(suffix, teams))
    assert max(m["season"] for m in meta.values()) < loader.FIRST_HOLDOUT_SEASON

    poss_rows = {"strict": [], "tolerant": []}
    quality = {"events": 0, "onball": 0, "onball_missing_end": 0, "nonmonotone_matches": 0, "matches": 0}
    clock_check = []
    for suffix in wyscout.COMPETITIONS:
        raw = json.loads((wyscout.RAW_DIR / "events" / f"events_{suffix}.json").read_text())
        by_match: dict[int, list] = {}
        for e in raw:
            by_match.setdefault(int(e["matchId"]), []).append(e)
        del raw
        for wy_id, evs in by_match.items():
            if wy_id not in want:
                continue
            m = meta[wy_id]
            match_id = f"{wyscout.SOURCE}:{m['competition']}:{m['season']}:{wy_id}"
            ev = match_events(evs)
            quality["matches"] += 1
            quality["events"] += len(ev)
            ob_mask = ev["name"].isin(ONBALL)
            quality["onball"] += int(ob_mask.sum())
            quality["onball_missing_end"] += int((ev.loc[ob_mask, "npos"] < 2).sum())
            if (ev.groupby("period")["local"].diff().dropna() < 0).any():
                quality["nonmonotone_matches"] += 1
            # Clock agreement with the shot slate (shots there are pre-dedup).
            mine = np.sort(ev.loc[ev["sub"].isin(wyscout.SHOT_SUBEVENTS), "t"].to_numpy())
            theirs = np.sort(shots.loc[shots["wy_id"] == wy_id, "t"].to_numpy())
            clock_check.append(
                (
                    len(mine),
                    len(theirs),
                    float(np.abs(mine - theirs).max()) if len(mine) == len(theirs) and len(mine) else 0.0,
                )
            )
            for rule in poss_rows:
                ob = possessions(ev, tolerant=(rule == "tolerant"))
                p = flag_dangerous(ob)
                p["match_id"] = match_id
                p["side"] = np.where(p["owner"] == m["home_id"], "H", "A")
                poss_rows[rule].append(p)
        log.info("%s done (%d matches so far)", suffix, quality["matches"])

    cc = np.array(clock_check)
    team_matches = shots.groupby(["match_id", "side"]).size()
    payload: dict = {
        "note": (
            "Feasibility only. No momentum statistic is computed: nothing here measures "
            "whether one dangerous possession predicts another."
        ),
        "definitions": {
            "onball_events": sorted(ONBALL),
            "strict_possession": "a new possession whenever the team of consecutive on-ball events changes, or the period does",
            "tolerant_possession": "as strict, but one opponent on-ball event that is not an accurate pass, between two runs of the same team, is a touch and does not end the possession; a shot is never a touch",
            "final_third": f"an own on-ball event starting at x >= {FINAL_THIRD_X:.1f}, or an accurate pass/free kick ending there",
            "box": f"an own on-ball event starting in the box (x >= {BOX_X}, {BOX_Y[0]} <= y <= {BOX_Y[1]}), or an accurate pass/free kick ending there",
            "event_time": "the first moment the possession met the definition",
        },
        "quality": {
            **quality,
            "frac_onball_missing_end": quality["onball_missing_end"] / max(quality["onball"], 1),
            "clock_shot_count_mismatch_matches": int((cc[:, 0] != cc[:, 1]).sum()),
            "clock_max_abs_diff_min": float(cc[:, 2].max()),
        },
        "shots_per_team_match": {
            "raw_incl_penalties": summary(team_matches),
            "n_team_matches": int(team_matches.size),
        },
    }

    profiles = []
    for rule, parts in poss_rows.items():
        p = pd.concat(parts, ignore_index=True)
        p["dur_s"] = (p["t_end"] - p["t_start"]) * 60.0
        idx = p.groupby(["match_id", "owner"]).size().index
        n_shots_total = int(p["n_shots"].sum())
        rec = {
            "possessions_per_match": float(p.groupby("match_id").size().mean()),
            "possession_duration_s": summary(p["dur_s"]),
            "events_per_possession_mean": float(p["n_events"].mean()),
            "frac_possessions_with_shot": float(p["has_shot"].mean()),
            "n_shots_in_possessions": n_shots_total,
        }
        for d, tcol in (("final_third", "t_ft"), ("box", "t_bx")):
            counts = per_team_match(p, tcol, pd.DataFrame(index=idx))
            gaps = gaps_by_team(p, tcol)
            calib = dedup.choose_threshold(gaps)
            thr = calib["threshold_seconds"]
            n_ev = int(p[tcol].notna().sum())
            removed = float((gaps < thr).sum() / n_ev)
            prof = calib.pop("profile")
            profiles.append(prof.assign(rule=rule, definition=d))
            shots_inside = int(p.loc[p[tcol].notna(), "n_shots"].sum())
            rec[d] = {
                "n_events": n_ev,
                "per_team_match": summary(counts),
                "per_team_match_after_collapse": float((n_ev - (gaps < thr).sum()) / len(idx)),
                "ratio_to_raw_shots": float(counts.mean() / team_matches.mean()),
                "frac_shots_inside": shots_inside / max(n_shots_total, 1),
                "excess_dies_at_s": thr,
                "excess_mass_below_threshold": calib["excess_mass_below_threshold"],
                "removed_fraction_if_collapsed": removed,
                "passes_15pct_guard": removed <= 0.15,
                "n_gaps": calib["n_gaps"],
                "gap_quantiles_s": {q: float(np.quantile(gaps, q)) for q in (0.05, 0.25, 0.5, 0.75)},
            }
        payload[rule] = rec

    pd.concat(profiles, ignore_index=True).to_csv("results/pilot_wyscout_possession_gaps.csv", index=False)
    path = write_manifest("pilot_wyscout_possessions", payload)
    print(json.dumps(json.loads(path.read_text())["payload"], indent=2))


if __name__ == "__main__":
    main()
