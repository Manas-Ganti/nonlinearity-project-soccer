import os
import signal
from concurrent.futures.process import BrokenProcessPool

import pytest

from src.inference.parallel import pmap


def _square(x):
    return x * x


def _die_on_two(x):
    if x == 2:
        os.kill(os.getpid(), signal.SIGKILL)
    return x


def test_pmap_keeps_order_across_workers():
    assert pmap(_square, range(10), workers=3) == [x * x for x in range(10)]


def test_pmap_accepts_closures_through_fork():
    k = 7
    assert pmap(lambda x: x + k, range(5), workers=2) == [7, 8, 9, 10, 11]


def test_pmap_fails_loudly_when_a_worker_dies():
    """multiprocessing.Pool hangs forever here; the job must fail instead."""
    with pytest.raises(BrokenProcessPool):
        pmap(_die_on_two, range(6), workers=2)
