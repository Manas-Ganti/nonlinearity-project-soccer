"""One provider against another on the same matches.

PROPOSAL.md plans an FBref cross-check to ask how much the answer depends on whose
xG model is used. Where sources overlap -- Understat covers every season the two
second-clock providers do -- there is a stronger version of that check available
for free, and it is worth being precise about what it measures.

**The headline fit is unmarked.** xG never enters the branching-ratio likelihood;
only arrival times do. So this comparison is not a comparison of xG models. It is a
comparison of *which events each provider calls a shot* and *when it says they
happened* -- which is the part that can actually move `eta_hat`. The xG agreement is
reported alongside because it does bear on the quality regression and the marked
extension, but it is a separate quantity and is labelled as one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG, Config
from src.ingest import loader

# The two providers disagree only on how formal club names are.
TEAM_ALIASES = {
    "AFC Bournemouth": "Bournemouth",
    "Leicester City": "Leicester",
    "Norwich City": "Norwich",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Brighton & Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
}


_NOISE = {
    "fc",
    "cf",
    "afc",
    "sc",
    "ac",
    "as",
    "us",
    "ss",
    "ssc",
    "sd",
    "ud",
    "cd",
    "rcd",
    "rc",
    "club",
    "calcio",
    "de",
    "la",
    "le",
    "les",
    "1",
    "1.",
    "04",
    "05",
    "1899",
    "1900",
    "1909",
    "1913",
    "e",
    "v",
    "sv",
    "vfb",
    "vfl",
    "tsg",
    "fsv",
    "bsc",
    "sge",
    "ogc",
    "olympique",
    "stade",
    "en",
    "athletic",
    "association",
    "sportive",
    "sportiva",
    "football",
    "women",
    "w",
    "wfc",
    "lfc",
    "ladies",
    "feminas",
    "femminile",
}


def canonical_team(name: str) -> str:
    """Strip club-name furniture so 'FC Barcelona', 'Barcelona' and 'Barcelona CF' agree.

    Understat, StatsBomb and Wyscout spell the same club three ways; the alias
    table handles the English cases, and this handles the rest without a table."""
    import re
    import unicodedata

    s = TEAM_ALIASES.get(str(name), str(name))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    tokens = [t for t in re.split(r"[\s\-./&']+", s.lower()) if t and t not in _NOISE]
    return " ".join(tokens)


def _match_key(events: pd.DataFrame) -> pd.DataFrame:
    """(date, home team) identifies a fixture in both feeds."""
    home = events[events["side"] == "H"]
    key = (
        home.groupby("match_id")
        .agg(date=("date", "first"), home_team=("team", "first"), away_team=("opponent", "first"))
        .reset_index()
    )
    key["day"] = pd.to_datetime(key["date"]).dt.normalize()
    key["home_team"] = key["home_team"].map(canonical_team)
    key["away_team"] = key["away_team"].map(canonical_team)
    key["key"] = key["day"].dt.strftime("%Y-%m-%d") + "|" + key["home_team"]
    return key


def align(slate_a, slate_b) -> pd.DataFrame:
    """Fixtures present in both slates, with each side's match id."""
    a = _match_key(slate_a.events).rename(columns={"match_id": "match_id_a"})
    b = _match_key(slate_b.events).rename(columns={"match_id": "match_id_b"})
    merged = a.merge(b[["key", "match_id_b", "away_team"]], on="key", suffixes=("_a", "_b"))
    merged["away_agrees"] = merged["away_team_a"] == merged["away_team_b"]
    return merged


def compare(
    slate_a, slate_b, *, name_a: str = "understat", name_b: str = "statsbomb", cfg: Config = CONFIG
) -> dict:
    """Coverage, per-match agreement, and eta_hat from each source on the shared fixtures."""
    from src.inference.pipeline import Pipeline

    pairs = align(slate_a, slate_b)
    pairs = pairs[pairs["away_agrees"]]
    if pairs.empty:
        return {"error": "no fixtures aligned between the two sources"}

    sub_a = slate_a.filter_matches(pairs["match_id_a"])
    sub_b = slate_b.filter_matches(pairs["match_id_b"])

    counts_a = sub_a.events.groupby("match_id").size().reindex(pairs["match_id_a"]).to_numpy()
    counts_b = sub_b.events.groupby("match_id").size().reindex(pairs["match_id_b"]).to_numpy()
    xg_a = sub_a.events.groupby("match_id")["xg"].sum().reindex(pairs["match_id_a"]).to_numpy()
    xg_b = sub_b.events.groupby("match_id")["xg"].sum().reindex(pairs["match_id_b"]).to_numpy()

    fits = {}
    for label, slate in ((name_a, sub_a), (name_b, sub_b)):
        model, prep = loader.prepare_model_slate(slate, cfg, jitter_seed=cfg.jitter.base_seed)
        res = Pipeline(model, cfg).run(model.events, warm_start=False)
        fits[label] = {
            **res.hawkes_fit.as_dict(),
            "dedup": prep,
            "red_cards_known": bool(slate.red_cards_known),
            "matches": int(slate.events["match_id"].nunique()),
        }
    # If either side is itself pooled, fit each provider's share of the aligned set.
    by_provider = {}
    for label, slate in ((name_a, sub_a), (name_b, sub_b)):
        sources = slate.events["source"].unique()
        if len(sources) < 2:
            continue
        for src in sorted(sources):
            part = slate.filter_matches(slate.events.loc[slate.events["source"] == src, "match_id"].unique())
            model, prep = loader.prepare_model_slate(part, cfg, jitter_seed=cfg.jitter.base_seed)
            res = Pipeline(model, cfg).run(model.events, warm_start=False)
            by_provider[f"{label}:{src}"] = {
                **res.hawkes_fit.as_dict(),
                "dedup": prep,
                "matches": int(part.events["match_id"].nunique()),
            }

    return {
        "aligned_fixtures": len(pairs),
        "shots": {
            name_a: len(sub_a.events),
            name_b: len(sub_b.events),
            "ratio": float(len(sub_a.events) / max(len(sub_b.events), 1)),
            "per_match_correlation": float(np.corrcoef(counts_a, counts_b)[0, 1]),
            "mean_absolute_difference": float(np.abs(counts_a - counts_b).mean()),
        },
        "xg_totals": {
            name_a: float(np.nansum(xg_a)),
            name_b: float(np.nansum(xg_b)),
            "per_match_correlation": float(np.corrcoef(xg_a, xg_b)[0, 1]),
            "note": "reported for the quality regression and the marked extension; xG is not "
            "an input to the unmarked branching-ratio fit",
        },
        "fits": fits,
        "fits_by_provider": by_provider,
        "names": [name_a, name_b],
        "eta_self_difference": float(fits[name_a]["eta_self"] - fits[name_b]["eta_self"]),
        "note": (
            "The two feeds differ in clock resolution as well as in shot detection, so this "
            "difference is not attributable to either alone. Step 7 separates the resolution "
            "component by rounding one source's own clock."
        ),
    }
