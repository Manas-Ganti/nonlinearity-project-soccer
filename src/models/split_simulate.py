"""Ogata thinning for the split-kernel model (docs/spec_spells.md section 6.3).

Game state **and the possession path (so Z)** are held at the observed path. A
simulated shot therefore cannot change later Z, and the power study does not
measure how much excitation the gamma term absorbs through that route. Stated in
the write-up.

Upper bound for thinning: the per-bin background rate times the largest possible
spell factor exp(max(gamma_own, 0) + max(gamma_opp, 0)), suffix-maxed over bins,
plus the current excitation (which only decays until the next event).
"""

from __future__ import annotations

import numpy as np

from src.config import BIN_MINUTES
from src.features.spells import SpellState
from src.models import baseline
from src.models.simulate import SlateSimulator

SCORE_CLASS = {0: 0, -1: 1, 1: 2}  # level, trailing, leading (sign of the score diff)


def score_class(score_diff: np.ndarray) -> np.ndarray:
    s = np.sign(np.asarray(score_diff)).astype(int)
    return np.where(s == 0, 0, np.where(s < 0, 1, 2)).astype(np.int64)


class SplitSimulator(SlateSimulator):
    def __init__(
        self,
        design: baseline.BackgroundDesign,
        rate_rows: np.ndarray,
        *,
        mode: str,
        spell: SpellState | None = None,
        gamma: tuple[float, float] = (0.0, 0.0),
        marks: np.ndarray | None = None,
    ):
        super().__init__(design, rate_rows, marks=marks)
        if mode not in ("spell", "score"):
            raise ValueError(f"unknown class mode {mode!r}")
        if mode == "spell" and spell is None:
            raise ValueError("spell mode needs a SpellState")
        self.mode = mode
        self.n_class = 2 if mode == "spell" else 3
        self.spell = spell
        self.gamma = np.asarray(gamma, dtype=np.float64)
        self.use_gamma = spell is not None and bool(np.any(self.gamma != 0.0))
        fmax = float(np.exp(max(self.gamma[0], 0.0) + max(self.gamma[1], 0.0))) if self.use_gamma else 1.0
        self.mu_suffix_max = self.mu_suffix_max * fmax
        self.spell_idx = spell.match_index(self.match_ids) if spell is not None else None
        # score class of each (match, team, bin), from the design's observed path
        N = self.mu.shape[2]
        self.score_cls = np.zeros((len(self.match_ids), 2, N), dtype=np.int64)
        for (mid, side), seg in design.seg_lookup.items():
            m = self.m_pos[mid]
            k = 0 if side == "H" else 1
            nb = int(design.seg_nbins[seg])
            start = int(design.seg_start[seg])
            lv = design.score_levels[design.score_idx[start : start + nb]]
            self.score_cls[m, k, :nb] = score_class(lv)

    def simulate_split(
        self,
        eta_own: np.ndarray,
        eta_cross: float,
        beta: float,
        rng: np.random.Generator,
        *,
        max_events_per_match: int = 400,
    ) -> dict:
        M = len(self.match_ids)
        C = self.n_class
        eta_own = np.asarray(eta_own, dtype=np.float64)
        if eta_own.size != C:
            raise ValueError(f"need {C} own-kernel values")
        contrib = eta_own + eta_cross  # one class-c event's total contribution to lambda_H + lambda_A

        t = np.zeros(M)
        E = np.zeros((M, 2, C))
        active = np.arange(M)
        out_m, out_k, out_t, out_c = [], [], [], []
        counts = np.zeros(M, dtype=np.int64)

        while active.size:
            ta = t[active]
            b = np.minimum((ta / BIN_MINUTES).astype(np.int64), self.n_bins[active] - 1)
            Ea = E[active]
            lam_bar = self.mu_suffix_max[active, b] + beta * (Ea.sum(axis=1) @ contrib)
            lam_bar = np.maximum(lam_bar, 1e-12)

            u = rng.exponential(1.0 / lam_bar)
            t_new = ta + u
            alive = t_new < self.T[active]
            Ea = Ea * np.exp(-beta * u)[:, None, None]
            E[active] = Ea
            t[active] = t_new

            b_new = np.minimum((t_new / BIN_MINUTES).astype(np.int64), self.n_bins[active] - 1)
            mu_h = self.mu[active, 0, b_new]
            mu_a = self.mu[active, 1, b_new]
            if self.use_gamma:
                t_eval = np.minimum(t_new, self.T[active])
                zh, za = self.spell.at_index(self.spell_idx[active], np.zeros(active.size), t_eval)
                mu_h = mu_h * np.exp(self.gamma[0] * zh + self.gamma[1] * za)
                mu_a = mu_a * np.exp(self.gamma[0] * za + self.gamma[1] * zh)
            own_h = Ea[:, 0, :] @ eta_own
            own_a = Ea[:, 1, :] @ eta_own
            lam_h = mu_h + beta * (own_h + eta_cross * Ea[:, 1, :].sum(axis=1))
            lam_a = mu_a + beta * (own_a + eta_cross * Ea[:, 0, :].sum(axis=1))
            lam_tot = lam_h + lam_a

            accept = alive & (rng.random(active.size) <= lam_tot / lam_bar)
            if accept.any():
                sel = np.flatnonzero(accept)
                p_home = np.where(lam_tot[sel] > 0, lam_h[sel] / lam_tot[sel], 0.5)
                k = (rng.random(sel.size) > p_home).astype(np.int64)
                m_sel = active[sel]
                if self.mode == "spell":
                    zo, _ = self.spell.at_index(self.spell_idx[m_sel], k, t_new[sel])
                    c = zo.astype(np.int64)
                else:
                    c = self.score_cls[m_sel, k, b_new[sel]]
                E[m_sel, k, c] += 1.0
                out_m.append(m_sel)
                out_k.append(k.astype(np.int8))
                out_t.append(t_new[sel])
                out_c.append(c)
                counts[m_sel] += 1

            keep = alive & (counts[active] < max_events_per_match)
            active = active[keep]

        if out_m:
            m_arr, k_arr = np.concatenate(out_m), np.concatenate(out_k)
            t_arr, c_arr = np.concatenate(out_t), np.concatenate(out_c)
            o = np.lexsort((t_arr, m_arr))
            return {"match": m_arr[o], "team": k_arr[o], "t": t_arr[o], "cls": c_arr[o]}
        z = np.zeros(0, np.int64)
        return {"match": z, "team": z.astype(np.int8), "t": np.zeros(0), "cls": z}
