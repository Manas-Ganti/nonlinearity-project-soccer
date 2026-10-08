"""Natural cubic spline basis, intercept column dropped.

Natural (rather than plain B-spline) because f(minute) is evaluated near the
boundaries of [0, T] constantly and a cubic basis is free to wander there. The
natural constraint forces linearity beyond the outer knots, which is the right
prior for "chance creation drifts up towards the whistle".
"""

from __future__ import annotations

import numpy as np


def natural_cubic_knots(df: int, lo: float, hi: float) -> np.ndarray:
    """`df` basis columns (excluding intercept) needs `df + 1` knots."""
    if df < 2:
        raise ValueError("need df >= 2")
    return np.linspace(lo, hi, df + 1)


def natural_cubic_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Columns: [x, d_1 - d_{K-1}, ..., d_{K-2} - d_{K-1}]  (Hastie et al., ESL 5.2.1)."""
    x = np.asarray(x, dtype=np.float64)
    k = np.asarray(knots, dtype=np.float64)
    K = k.size
    if K < 3:
        raise ValueError("need at least 3 knots")

    def d(j: int) -> np.ndarray:
        num = np.clip(x - k[j], 0, None) ** 3 - np.clip(x - k[K - 1], 0, None) ** 3
        return num / (k[K - 1] - k[j])

    dK1 = d(K - 2)
    cols = [x] + [d(j) - dK1 for j in range(K - 2)]
    B = np.column_stack(cols)
    # Scale each column to unit sd so the optimiser sees a well-conditioned problem.
    sd = B.std(axis=0)
    sd[sd == 0] = 1.0
    return B / sd
