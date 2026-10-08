"""Fork-based parallel map for the replicate loops.

Every simulation study here is embarrassingly parallel: each replicate builds a
slate, refits mu and the kernel, and returns a few numbers. A forked worker
inherits the fitted pipeline for free, so nothing large is pickled -- only the
per-replicate arguments go in and a small dict comes out. Replicate seeds are
drawn in the parent, in order, so the result is identical whatever the worker
count (including 1, which runs inline and is what the tests use).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from collections.abc import Callable, Iterable

log = logging.getLogger(__name__)

_FN: Callable | None = None


def default_workers() -> int:
    env = os.environ.get("SOCCER_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, (os.cpu_count() or 2) - 1)


def _call(arg):
    return _FN(arg)


def pmap(
    fn: Callable, items: Iterable, *, workers: int | None = None, label: str = "", every: int = 25
) -> list:
    """`[fn(x) for x in items]`, over `workers` forked processes, in order."""
    global _FN
    items = list(items)
    workers = default_workers() if workers is None else max(1, int(workers))
    workers = min(workers, len(items)) if items else 1
    if workers <= 1:
        out = []
        for i, x in enumerate(items, start=1):
            out.append(fn(x))
            if every and label and i % every == 0:
                log.info("  %s %d/%d", label, i, len(items))
        return out
    _FN = fn
    try:
        ctx = mp.get_context("fork")
        out = []
        with ctx.Pool(processes=workers) as pool:
            for i, r in enumerate(pool.imap(_call, items, chunksize=1), start=1):
                out.append(r)
                if every and label and i % every == 0:
                    log.info("  %s %d/%d (%d workers)", label, i, len(items), workers)
        return out
    finally:
        _FN = None
