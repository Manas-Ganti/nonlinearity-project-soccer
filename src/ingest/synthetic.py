"""A synthetic slate with known ground truth.

Two jobs: let the test suite exercise the whole pipeline without a network or an
overnight scrape, and give the recovery test a case where the true background is
known exactly rather than estimated.

The generative model here is deliberately *not* the fitted one -- it is a plain
inhomogeneous Poisson process with hand-set effects, so a test that recovers those
effects is testing the estimator rather than its own assumptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import T_NOMINAL
from src.ingest import schema

TRUE = {
    "log_rate_base": np.log(0.13),  # shots per team per minute
    "home": 0.15,
    "minute_slope": 0.004,  # chances drift up towards the whistle
    "score": {-2: 0.10, -1: 0.06, 0: 0.0, 1: -0.10, 2: -0.18},  # leading teams create less
    "red": {-1: -0.30, 0: 0.0, 1: 0.30},
    "team_sd": 0.20,
}


def make_slate(
    n_matches: int = 200,
    n_teams: int = 20,
    seasons: tuple[int, ...] = (2014, 2015),
    *,
    seed: int = 0,
    eta_self: float = 0.0,
    eta_cross: float = 0.0,
    beta: float = 1 / 5.0,
    with_cards: bool = True,
) -> schema.Slate:
    """Independent-Poisson slate (eta = 0) or, with eta > 0, a Hawkes one.

    Excitation is applied by simple sequential thinning per match: this generator
    is a *reference*, so it does not share code with `models.simulate`, and a test
    that agrees between the two is meaningful.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    attack = {t: rng.normal(0, TRUE["team_sd"]) for t in teams}
    defence = {t: rng.normal(0, TRUE["team_sd"]) for t in teams}

    ev_rows, goal_rows, card_rows = [], [], []
    for m in range(n_matches):
        season = int(seasons[m % len(seasons)])
        h, a = rng.choice(teams, size=2, replace=False)
        T = float(T_NOMINAL)
        match_id = f"synthetic:SYN:{season}:{m:05d}"

        cards: list[tuple[float, str]] = []
        if with_cards and rng.random() < 0.10:
            cards.append((float(rng.uniform(20, 85)), "H" if rng.random() < 0.5 else "A"))

        goals: list[tuple[float, str]] = []
        events: list[dict] = []
        state_h, state_a = 0, 0

        # sequential thinning on the superposition (Ogata), one match at a time
        lam_max = 1.2
        t = 0.0
        last_t = 0.0
        exc = {"H": 0.0, "A": 0.0}
        while True:
            t += rng.exponential(1.0 / lam_max)
            if t >= T:
                break
            decay = np.exp(-beta * (t - last_t))
            exc = {k: v * decay for k, v in exc.items()}
            last_t = t

            lam = {}
            for side in ("H", "A"):
                opp = "A" if side == "H" else "H"
                team, oteam = (h, a) if side == "H" else (a, h)
                sd = np.clip((state_h - state_a) if side == "H" else (state_a - state_h), -2, 2)
                own_c = sum(1 for ct, cs in cards if ct < t and cs == side)
                opp_c = sum(1 for ct, cs in cards if ct < t and cs == opp)
                rd = int(np.clip(opp_c - own_c, -1, 1))
                log_mu = (
                    TRUE["log_rate_base"]
                    + TRUE["home"] * (side == "H")
                    + attack[team]
                    + defence[oteam]
                    + TRUE["minute_slope"] * t
                    + TRUE["score"][int(sd)]
                    + TRUE["red"][rd]
                )
                lam[side] = np.exp(log_mu) + beta * (eta_self * exc[side] + eta_cross * exc[opp])
            tot = lam["H"] + lam["A"]
            if tot > lam_max:
                raise RuntimeError(f"thinning bound violated: {tot:.3f} > {lam_max}")
            if rng.random() > tot / lam_max:
                continue

            side = "H" if rng.random() < lam["H"] / tot else "A"
            team, oteam = (h, a) if side == "H" else (a, h)
            xg = float(np.clip(rng.beta(1.1, 8.0), 1e-4, 0.999))
            is_goal = bool(rng.random() < xg)
            events.append(
                {
                    "match_id": match_id,
                    "source": "synthetic",
                    "league": "SYN",
                    "season": season,
                    "date": pd.Timestamp("2014-08-16") + pd.Timedelta(days=7 * m),
                    "side": side,
                    "team": team,
                    "opponent": oteam,
                    "t": t,
                    "minute": int(t),
                    "second": np.nan,
                    "xg": xg,
                    "situation": "open",
                    "is_goal": is_goal,
                    "score_diff": 0,
                    "red_diff": 0,
                    "match_T": T,
                    "n_merged": 1,
                    "red_cards_known": with_cards,
                }
            )
            if is_goal:
                goals.append((t, side))
                if side == "H":
                    state_h += 1
                else:
                    state_a += 1
            exc[side] += 1.0

        ev_rows.extend(events)
        goal_rows.extend({"match_id": match_id, "side": s, "t": tt} for tt, s in goals)
        card_rows.extend({"match_id": match_id, "side": s, "t": tt} for tt, s in cards)

    events_df = pd.DataFrame(ev_rows)
    if len(events_df) == 0:
        raise RuntimeError("synthetic generator produced no events")
    # re-attach red_diff properly from the card stream
    from src.features import state as state_mod

    events_df = events_df.sort_values(["match_id", "t"], kind="stable").reset_index(drop=True)
    events_df["order_idx"] = events_df.groupby("match_id").cumcount()
    goals_df = pd.DataFrame(goal_rows, columns=["match_id", "side", "t"])
    cards_df = pd.DataFrame(card_rows, columns=["match_id", "side", "t"]) if with_cards else None
    for frame in (goals_df, cards_df):
        if frame is not None and len(frame):
            frame["order_idx"] = np.arange(len(frame))
        elif frame is not None:
            frame["order_idx"] = pd.Series(dtype="int64")
    events_df = state_mod.reconstruct(events_df.drop(columns=["score_diff", "red_diff"]), goals_df, cards_df)
    return schema.Slate(events_df, goals_df, cards_df)
