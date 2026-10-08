"""Uniform jitter inside the recorded minute.

Understat records integer minutes, so after de-duplication the surviving
same-minute events are exact ties and the likelihood is not well defined (two
events at identical times make the excitation term degenerate). Jitter breaks the
ties; the estimate is then reported across several draws so it does not depend on
one realisation (CLAUDE.md, "Ties on Understat").

The jitter is **order-preserving**: within a minute the drawn offsets are sorted
and handed out in the table's existing order, so the sequence of events -- and
therefore the game state attached to each of them at ingest -- is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import JitterConfig


def jitter_events(events: pd.DataFrame, seed: int, *, only_integer_clock: bool = True) -> pd.DataFrame:
    """Return a copy of `events` with tied integer times spread inside their minute."""
    df = events.sort_values(["match_id", "t"], kind="stable").reset_index(drop=True).copy()
    target = df["second"].isna().to_numpy() if only_integer_clock else np.ones(len(df), dtype=bool)
    if not target.any():
        return df

    rng = np.random.default_rng(seed)
    sub = df.loc[target]
    key = (sub["match_id"].astype(str) + "|" + sub["minute"].astype(str)).to_numpy()
    _, group = np.unique(key, return_inverse=True)
    # The frame is already contiguous by (match, minute), so sorting the draws within
    # each group and writing them back positionally preserves the original order.
    u = rng.random(len(sub))
    order = np.lexsort((u, group))
    df.loc[target, "t"] = sub["minute"].to_numpy(dtype=np.float64) + u[order]

    df["t"] = np.minimum(df["t"], df["match_T"] - 1e-9)
    return df.sort_values(["match_id", "t"], kind="stable").reset_index(drop=True)


def jitter_draws(events: pd.DataFrame, cfg: JitterConfig):
    """Yield (draw_index, seed, jittered events) for the sensitivity report."""
    for i in range(cfg.n_draws):
        seed = cfg.base_seed + i
        yield i, seed, jitter_events(events, seed)
