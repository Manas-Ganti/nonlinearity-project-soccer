"""The spell indicator Z_k(t) (docs/spec_spells.md section 3).

    C_k(t) = dangerous possessions among team k's last `window` possessions that
             started before t's possession, within the current period
    Z_k(t) = 1[C_k(t) >= need]          (headline: need = 4, window = 5)

`Z` is **predictable**: the possession containing `t` is never counted, so a shot
cannot put its own team into a spell. It resets at each period and changes only at
possession boundaries, so it is stored as a step function per match:
breakpoints `t_b` with the state (Z_H, Z_A) holding on [t_b, t_{b+1}).

Two things are read off it:

- `at(match, side, t)`: (Z_own, Z_opp) at arbitrary times, vectorised, for shots
  (real or simulated).
- `row_weights(design)`: for every background row (match, side, minute bin), the
  exposure spent in each of the four (Z_own, Z_opp) states. That is what lets the
  background carry gamma_own * Z_own + gamma_opp * Z_opp exactly, even though Z
  switches inside a minute bin.

Bootstrap copies of a match are labelled `<id>#b<nnnnnn>`; the suffix is stripped
on lookup, so a resampled slate reads its original match's path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import BIN_MINUTES

# (Z_own, Z_opp) state s = 2 * Z_own + Z_opp
STATE_OWN = np.array([0.0, 0.0, 1.0, 1.0])
STATE_OPP = np.array([0.0, 1.0, 0.0, 1.0])
_SPACING = 10_000.0  # minutes between matches on the flattened lookup axis


def _base_id(mid) -> str:
    return str(mid).split("#b", 1)[0]


@dataclass
class SpellState:
    match_ids: np.ndarray  # (M,) str, sorted
    bp_t: list  # per match: breakpoint times, starting at 0
    bp_zh: list  # per match: Z_H on [t_b, t_{b+1})
    bp_za: list
    T: np.ndarray  # (M,) match length
    rule: dict

    def __post_init__(self):
        self.pos = {m: i for i, m in enumerate(self.match_ids)}
        keys = [i * _SPACING + np.asarray(t) for i, t in enumerate(self.bp_t)]
        self._key = np.concatenate(keys)
        self._zh = np.concatenate(self.bp_zh).astype(np.float64)
        self._za = np.concatenate(self.bp_za).astype(np.float64)

    # ------------------------------------------------------------ lookups

    def match_index(self, match_ids) -> np.ndarray:
        try:
            return np.array([self.pos[_base_id(m)] for m in match_ids], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"match {exc} has no possession path") from None

    def at_index(
        self, m_idx: np.ndarray, is_away: np.ndarray, t: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """(Z_own, Z_opp) for matches given by position, team 0 = home / 1 = away."""
        k = np.searchsorted(self._key, m_idx * _SPACING + np.asarray(t, dtype=np.float64), side="right") - 1
        zh, za = self._zh[k], self._za[k]
        away = np.asarray(is_away).astype(bool)
        return np.where(away, za, zh), np.where(away, zh, za)

    def at(self, match_ids, sides, t) -> tuple[np.ndarray, np.ndarray]:
        m = self.match_index(match_ids)
        return self.at_index(m, np.asarray(sides) == "A", np.asarray(t, dtype=np.float64))

    def annotate(self, events: pd.DataFrame) -> pd.DataFrame:
        """Add z_own / z_opp at every event."""
        out = events.copy()
        if len(out):
            zo, zp = self.at(out["match_id"].to_numpy(), out["side"].to_numpy(), out["t"].to_numpy())
        else:
            zo = zp = np.zeros(0)
        out["z_own"] = zo
        out["z_opp"] = zp
        return out

    # --------------------------------------------------- background exposure

    def row_weights(self, design) -> np.ndarray:
        """(n_rows, 4): minutes each background row spends in each (Z_own, Z_opp) state."""
        w = np.zeros((design.n_rows, 4))
        for (mid, side), seg in design.seg_lookup.items():
            i = self.pos[_base_id(mid)]
            n = int(design.seg_nbins[seg])
            start = int(design.seg_start[seg])
            T = float(design.seg_T[seg])
            bp = np.asarray(self.bp_t[i], dtype=np.float64)
            zh = np.asarray(self.bp_zh[i])
            za = np.asarray(self.bp_za[i])
            own, opp = (zh, za) if side == "H" else (za, zh)
            state = 2 * own.astype(int) + opp.astype(int)
            ends = np.append(bp[1:], T)
            lengths = np.clip(np.minimum(ends, T) - np.minimum(bp, T), 0.0, None)
            edges = np.minimum(np.arange(n + 1, dtype=np.float64) * BIN_MINUTES, T)
            knots = np.append(bp, T)
            for s in range(4):
                cum = np.concatenate([[0.0], np.cumsum(np.where(state == s, lengths, 0.0))])
                w[start : start + n, s] = np.diff(np.interp(edges, knots, cum))
        return w


def build(
    poss: pd.DataFrame,
    *,
    need: int = 4,
    window: int = 5,
    danger_col: str = "t_ft",
    match_T: dict | None = None,
) -> SpellState:
    """Possession table (one rule, already merged if wanted) -> step functions.

    `match_T` (match -> length) closes each path; defaults to the table's own column.
    """
    df = poss.sort_values(["match_id", "period", "t_start"], kind="stable").reset_index(drop=True)
    df["_d"] = df[danger_col].notna().to_numpy()
    ids, bps, zhs, zas, Ts = [], [], [], [], []
    for mid, g in df.groupby("match_id", sort=True):
        t_list, zh_list, za_list = [0.0], [0], [0]
        for _, h in g.groupby("period", sort=True):
            t0 = h["t_start"].to_numpy(dtype=np.float64)
            side = h["side"].to_numpy()
            d = h["_d"].to_numpy()
            z = {}
            for k in ("H", "A"):
                own = side == k
                n_before = np.concatenate([[0], np.cumsum(own)[:-1]])  # own possessions before j
                dcum = np.concatenate([[0], np.cumsum(d[own])])  # danger over own sequence
                lo = np.maximum(n_before - window, 0)
                count = dcum[n_before] - dcum[lo]
                z[k] = (count >= need).astype(np.int8)
            # period start: nothing before the first possession of the half counts
            t_list.extend(t0.tolist())
            zh_list.extend(z["H"].tolist())
            za_list.extend(z["A"].tolist())
            # after the period's last possession ends, reset until the next period starts
            t_end = float(h["t_next"].to_numpy(dtype=np.float64)[-1])
            t_list.append(t_end)
            zh_list.append(0)
            za_list.append(0)
        t_arr = np.asarray(t_list)
        order = np.argsort(t_arr, kind="stable")
        ids.append(str(mid))
        bps.append(t_arr[order])
        zhs.append(np.asarray(zh_list, dtype=np.int8)[order])
        zas.append(np.asarray(za_list, dtype=np.int8)[order])
        T = match_T.get(mid) if match_T is not None else None
        Ts.append(float(T) if T is not None else float(g["match_T"].iloc[0]))
    return SpellState(
        match_ids=np.asarray(ids, dtype=object),
        bp_t=bps,
        bp_zh=zhs,
        bp_za=zas,
        T=np.asarray(Ts),
        rule={"need": need, "window": window, "danger_col": danger_col},
    )


def prevalence(state: SpellState, events: pd.DataFrame | None = None) -> dict:
    """Share of team-time with Z = 1 (and share of shots), for the section 3 re-check."""
    tot = on = 0.0
    for i in range(len(state.match_ids)):
        bp = state.bp_t[i]
        T = state.T[i]
        ln = np.clip(np.minimum(np.append(bp[1:], T), T) - np.minimum(bp, T), 0.0, None)
        on += float((ln * state.bp_zh[i]).sum() + (ln * state.bp_za[i]).sum())
        tot += 2.0 * T
    out = {"share_time_z1": on / tot if tot else float("nan"), "rule": state.rule}
    if events is not None and len(events):
        zo, _ = state.at(events["match_id"].to_numpy(), events["side"].to_numpy(), events["t"].to_numpy())
        out["share_shots_z1"] = float(zo.mean())
    return out
