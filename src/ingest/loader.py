"""Slate loading, with the held-out split enforced in code.

CLAUDE.md: develop on 2014-15 to 2018-19; 2019-20 onward is touched exactly once,
at the end. The loader refuses to hand back holdout seasons unless `final_run=True`
is passed explicitly, and it logs every request either way. The log is the evidence
that the split was respected -- an honest split you cannot audit is not a split.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import CONFIG, FIRST_HOLDOUT_SEASON, INTERIM, Config
from src.ingest import schema
from src.runlog import append_event

log = logging.getLogger(__name__)

HOLDOUT_LOG = "holdout_access.log"


class HoldoutViolation(RuntimeError):
    pass


def slate_dir(source: str):
    return INTERIM / source


def save_slate(slate: schema.Slate, source: str) -> None:
    slate.save(slate_dir(source))


def load_slate(
    source: str = "understat",
    *,
    split: str = "dev",
    final_run: bool = False,
    leagues: list[str] | None = None,
    caller: str = "unspecified",
) -> schema.Slate:
    """`split` is one of dev | holdout | all."""
    slate = schema.Slate.load(slate_dir(source))
    seasons = slate.events["season"].astype(int)

    if split == "dev":
        # Everything before the holdout year. DEV_SEASONS names the Understat pull
        # range; StatsBomb's older seasons (2004/05 onward) are development data too.
        keep = seasons < FIRST_HOLDOUT_SEASON
    elif split == "holdout":
        keep = seasons >= FIRST_HOLDOUT_SEASON
    elif split == "all":
        keep = pd.Series(True, index=slate.events.index)
    else:
        raise ValueError(f"unknown split {split!r}")

    touches_holdout = split in ("holdout", "all") and bool((seasons >= FIRST_HOLDOUT_SEASON).any())
    append_event(
        HOLDOUT_LOG,
        f"source={source} split={split} final_run={final_run} caller={caller} "
        f"touches_holdout={touches_holdout} matches={slate.events.loc[keep, 'match_id'].nunique()}",
    )
    if touches_holdout and not final_run:
        raise HoldoutViolation(
            f"split={split!r} would return seasons from {FIRST_HOLDOUT_SEASON} onward. "
            "Holdout data is touched exactly once, at the end: pass --final-run to mean it."
        )

    ev = slate.events[keep]
    if leagues is not None:
        ev = ev[ev["league"].isin(leagues)]
    return slate.filter_matches(ev["match_id"].unique())


def prepare_model_slate(
    slate: schema.Slate,
    cfg: Config = CONFIG,
    *,
    jitter_seed: int | None = None,
) -> tuple[schema.Slate, dict]:
    """De-duplicate, then jitter integer-clock ties. The order the build requires."""
    from src.features import dedup, jitter

    deduped, stats = dedup.apply_dedup(slate.events, cfg.dedup)
    if jitter_seed is not None:
        deduped = jitter.jitter_events(deduped, jitter_seed)
        stats = {**stats, "jitter_seed": jitter_seed}
    ties = _tie_count(deduped)
    stats = {**stats, "remaining_exact_ties": ties}
    return slate.with_events(deduped), stats


def _tie_count(events: pd.DataFrame) -> int:
    key = events["match_id"].astype(str) + "|" + events["t"].astype(str)
    return int(len(key) - key.nunique())


def describe(slate: schema.Slate) -> dict:
    ev = slate.events
    per_match = ev.groupby("match_id").size()
    return {
        "matches": int(ev["match_id"].nunique()),
        "events": len(ev),
        "events_per_match_mean": float(per_match.mean()) if len(per_match) else 0.0,
        "seasons": sorted(map(int, ev["season"].unique())),
        "leagues": sorted(map(str, ev["league"].unique())),
        "situations": {str(k): int(v) for k, v in ev["situation"].value_counts().items()},
        "goals": int(ev["is_goal"].sum()),
        "red_cards_known": bool(slate.red_cards_known),
        "mean_xg": float(np.mean(ev["xg"])) if len(ev) else np.nan,
        "strata": {
            "matches_by_source": {
                str(k): int(v) for k, v in ev.groupby("source")["match_id"].nunique().items()
            },
            "matches_by_gender": {
                str(k): int(v) for k, v in ev.groupby("gender")["match_id"].nunique().items()
            },
            "one_club_matches": int(ev.loc[ev["one_club_sample"], "match_id"].nunique()),
        },
    }
