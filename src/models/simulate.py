"""Ogata thinning from a fitted background with a settable branching ratio.

This is what makes the detection floor possible, so it is worth being explicit
about what it does and does not reproduce.

**Game state is held at its observed path.** Score differential and dismissals are
taken from the real match they are simulating and are *not* regenerated from the
simulated shots. The background mu_k(t) is therefore a fixed, known function per
match and the re-fit inside each replicate reuses the same design matrix. The
consequence, stated plainly: the power study measures how much of a planted eta the
*background estimation* absorbs, not how much the score-state feedback loop absorbs.
`resample_goals=True` turns marks into simulated goals so the sensitivity of that
choice can be checked, but the headline sweep runs with the observed paths.

All matches are thinned in lockstep so the whole slate is a few hundred vectorised
operations rather than a Python loop over 20,000 matches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BIN_MINUTES
from src.models import baseline


class SlateSimulator:
    """Reusable simulator bound to one background design and fit."""

    def __init__(
        self, design: baseline.BackgroundDesign, rate_rows: np.ndarray, marks: np.ndarray | None = None
    ):
        self.design = design
        self.match_ids = pd.unique(design.seg_match)
        self.m_pos = {m: i for i, m in enumerate(self.match_ids)}
        M = len(self.match_ids)
        N = int(design.seg_nbins.max())

        self.T = np.zeros(M)
        self.n_bins = np.zeros(M, dtype=np.int64)
        self.mu = np.zeros((M, 2, N))  # 0 = home, 1 = away
        self.seg_of = np.full((M, 2), -1, dtype=np.int64)
        for (mid, side), seg in design.seg_lookup.items():
            m = self.m_pos[mid]
            k = 0 if side == "H" else 1
            nb = int(design.seg_nbins[seg])
            self.T[m] = design.seg_T[seg]
            self.n_bins[m] = nb
            self.seg_of[m, k] = seg
            start = design.seg_start[seg]
            self.mu[m, k, :nb] = rate_rows[start : start + nb]

        # Beyond the real bins the rate is zero, so a suffix max is a valid bound.
        tot = self.mu.sum(axis=1)  # (M, N)
        self.mu_suffix_max = np.maximum.accumulate(tot[:, ::-1], axis=1)[:, ::-1]
        self.marks = np.asarray(marks, dtype=np.float64) if marks is not None else None

    def simulate(
        self,
        eta_self: float,
        eta_cross: float,
        beta: float,
        rng: np.random.Generator,
        *,
        max_events_per_match: int = 400,
    ) -> dict:
        """Returns raw arrays: match index, team index (0/1) and time, per event."""
        M = len(self.match_ids)
        eta = np.array([[eta_self, eta_cross], [eta_cross, eta_self]])
        col_sum = eta.sum(axis=0)  # contribution of one team-j event to lambda_total

        t = np.zeros(M)
        E = np.zeros((M, 2))
        active = np.arange(M)
        out_m, out_k, out_t = [], [], []
        counts = np.zeros(M, dtype=np.int64)

        while active.size:
            ta = t[active]
            b = np.minimum((ta / BIN_MINUTES).astype(np.int64), self.n_bins[active] - 1)
            Ea = E[active]
            lam_bar = self.mu_suffix_max[active, b] + beta * (Ea @ col_sum)
            lam_bar = np.maximum(lam_bar, 1e-12)

            u = rng.exponential(1.0 / lam_bar)
            t_new = ta + u
            alive = t_new < self.T[active]

            Ea = Ea * np.exp(-beta * u)[:, None]
            E[active] = Ea
            t[active] = t_new

            b_new = np.minimum((t_new / BIN_MINUTES).astype(np.int64), self.n_bins[active] - 1)
            mu_h = self.mu[active, 0, b_new]
            mu_a = self.mu[active, 1, b_new]
            exc = beta * (Ea @ eta.T)  # (n_active, 2): excitation of team k
            lam_h = mu_h + exc[:, 0]
            lam_a = mu_a + exc[:, 1]
            lam_tot = lam_h + lam_a

            accept = alive & (rng.random(active.size) <= lam_tot / lam_bar)
            if accept.any():
                sel = np.flatnonzero(accept)
                p_home = np.where(lam_tot[sel] > 0, lam_h[sel] / lam_tot[sel], 0.5)
                k = (rng.random(sel.size) > p_home).astype(np.int8)  # 0 home, 1 away
                m_sel = active[sel]
                E[m_sel, k] += 1.0
                out_m.append(m_sel)
                out_k.append(k)
                out_t.append(t_new[sel])
                counts[m_sel] += 1

            keep = alive & (counts[active] < max_events_per_match)
            active = active[keep]

        if out_m:
            m_arr = np.concatenate(out_m)
            k_arr = np.concatenate(out_k)
            t_arr = np.concatenate(out_t)
            order = np.lexsort((t_arr, m_arr))
            return {"match": m_arr[order], "team": k_arr[order], "t": t_arr[order]}
        return {"match": np.zeros(0, np.int64), "team": np.zeros(0, np.int8), "t": np.zeros(0)}

    def to_events(self, sim: dict, template: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Wrap simulated arrivals in the canonical event schema.

        Game state comes from the design (the observed path), marks are resampled
        from the observed xG pool, and every simulated shot is open play: penalties
        are excluded from the model, so the simulator must not invent them.
        """
        design = self.design
        m = sim["match"]
        k = sim["team"]
        seg = self.seg_of[m, k]
        b = np.minimum((sim["t"] / BIN_MINUTES).astype(np.int64), design.seg_nbins[seg] - 1)
        rows = design.seg_start[seg] + b

        meta = template.drop_duplicates("match_id").set_index("match_id")
        ids = self.match_ids[m]
        meta_rows = meta.reindex(ids)

        side = np.where(k == 0, "H", "A")
        home_team = meta_rows["home_team"].to_numpy()
        away_team = meta_rows["away_team"].to_numpy()
        xg = (
            rng.choice(self.marks, size=len(m))
            if self.marks is not None and self.marks.size
            else np.full(len(m), 0.1)
        )
        out = pd.DataFrame(
            {
                "match_id": ids,
                "source": "simulated",
                "league": meta_rows["league"].to_numpy(),
                "season": meta_rows["season"].to_numpy(),
                "date": meta_rows["date"].to_numpy(),
                "side": side,
                "team": np.where(k == 0, home_team, away_team),
                "opponent": np.where(k == 0, away_team, home_team),
                "t": sim["t"],
                "minute": np.floor(sim["t"]).astype("int16"),
                "second": np.nan,
                "xg": xg,
                "situation": "open",
                "is_goal": False,
                "score_diff": design.score_levels[design.score_idx[rows]],
                "red_diff": design.red_levels[design.red_idx[rows]],
                "match_T": design.seg_T[seg],
                "n_merged": 1,
                "red_cards_known": meta_rows["red_cards_known"].to_numpy(),
            }
        )
        from src.ingest import schema

        return schema.validate(out)


def expected_counts(design: baseline.BackgroundDesign, rate_rows: np.ndarray) -> float:
    """int mu over the whole slate -- the eta = 0 validation target."""
    return float((rate_rows * design.exposure).sum())
