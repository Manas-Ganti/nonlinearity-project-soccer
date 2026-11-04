"""Full StatsBomb event files for the development matches (docs/spec_spells.md section 9).

The shot cache (`data/raw/statsbomb/events.parquet`) keeps only the extracted
shot, goal and card rows. The possession analyses need every on-ball event, so
the full event files are pulled here into a **separate** cache,
`data/raw/statsbomb_events/`. The shot cache is never read for writing and never
touched.

- **Which matches:** exactly the development matches already in the shot cache
  (season < FIRST_HOLDOUT_SEASON), so the possession and shot slates cover the
  same games. Holdout matches are not downloaded at all.
- **What is kept:** the raw bytes, gzipped, one file per match, plus a manifest
  with each file's SHA-256. The cache is immutable input: a match already on disk
  is never fetched again.

Run once:  uv run python -m src.ingest.statsbomb_events
"""

from __future__ import annotations

import concurrent.futures as cf
import gzip
import hashlib
import json
import logging
import sys
import time

import pandas as pd

from src.config import FIRST_HOLDOUT_SEASON, RAW
from src.ingest import statsbomb

log = logging.getLogger(__name__)

OUT_DIR = RAW / "statsbomb_events"
MANIFEST = OUT_DIR / "manifest.json"


def dev_match_ids() -> list[int]:
    """Development StatsBomb match ids, read from the existing shot cache."""
    raw = pd.read_parquet(statsbomb.RAW_DIR / "events.parquet", columns=["sb_match_id", "season_name"])
    m = raw.drop_duplicates("sb_match_id")
    season = m["season_name"].astype(str).str.slice(0, 4).astype(int)
    ids = sorted(m.loc[season < FIRST_HOLDOUT_SEASON, "sb_match_id"].astype(int).tolist())
    return ids


def path_for(match_id: int):
    return OUT_DIR / f"{int(match_id)}.json.gz"


def pull(*, workers: int = 4, delay: float = 0.25) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = dev_match_ids()
    todo = [m for m in ids if not path_for(m).exists()]
    log.info(
        "statsbomb events: %d dev matches, %d cached, %d to fetch", len(ids), len(ids) - len(todo), len(todo)
    )

    def one(mid: int) -> tuple[int, str]:
        raw = statsbomb._get(f"{statsbomb.BASE}/events/{mid}.json")
        json.loads(raw)  # refuse to cache anything that is not valid JSON
        tmp = path_for(mid).with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(raw, mtime=0))
        tmp.rename(path_for(mid))
        time.sleep(delay)
        return mid, hashlib.sha256(raw).hexdigest()

    manifest = (
        json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"base": statsbomb.BASE, "sha256": {}}
    )
    failed: list[int] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one, m): m for m in todo}
        for i, fut in enumerate(cf.as_completed(futs), start=1):
            try:
                mid, digest = fut.result()
                manifest["sha256"][str(mid)] = digest
            except Exception as exc:  # keep going; report at the end
                failed.append(futs[fut])
                log.warning("failed %s: %s", futs[fut], exc)
            if i % 100 == 0:
                log.info("  %d/%d", i, len(todo))
                MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    manifest["n_matches"] = len(manifest["sha256"])
    manifest["first_holdout_season"] = FIRST_HOLDOUT_SEASON
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    out = {"requested": len(ids), "cached": manifest["n_matches"], "failed": failed}
    log.info("statsbomb events: %s", out)
    return out


def load_match(match_id: int) -> list[dict]:
    return json.loads(gzip.decompress(path_for(match_id).read_bytes()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    res = pull()
    sys.exit(1 if res["failed"] else 0)
