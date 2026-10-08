"""The pooled second-resolution slate: every provider with a real clock, stacked.

League, gender and era are absorbed by the team effects in the background model
(keyed by provider, team and season) and by a per-provider intercept; what is
kept is the stratum labelling so `eta` can be reported per provider, competition
and gender as the check that pooling was fair. Understat is never pooled in: its
minute clock returns zero by construction (results/power_understat_dev.json) and
would only drag the estimate toward it.
"""

from __future__ import annotations

import pandas as pd

from src.ingest import schema

SOURCE = "pooled"
MEMBERS = ("statsbomb", "wyscout")


def build_slate(slates: dict[str, schema.Slate]) -> schema.Slate:
    missing = [m for m in MEMBERS if m not in slates]
    if missing:
        raise ValueError(f"pooled slate needs {MEMBERS}; missing {missing}")
    for name, s in slates.items():
        if s.events["second"].isna().any():
            raise ValueError(f"{name} has events without a sub-minute clock; it cannot be pooled")
        if not s.red_cards_known:
            raise ValueError(f"{name} has no dismissal data; the pooled null needs h_red everywhere")
    events = pd.concat([slates[m].events for m in MEMBERS], ignore_index=True)
    goals = pd.concat([slates[m].goals for m in MEMBERS], ignore_index=True)
    cards = pd.concat([slates[m].cards for m in MEMBERS], ignore_index=True)
    if events["match_id"].nunique() != sum(s.events["match_id"].nunique() for s in slates.values()):
        raise ValueError("match ids collide across providers")
    return schema.Slate(events, goals, cards)
