import numpy as np
import pytest

from src.config import CONFIG
from src.inference.pipeline import Pipeline
from src.ingest import synthetic


@pytest.fixture(autouse=True)
def _private_audit_log(tmp_path_factory, monkeypatch):
    """Tests never write to the real logs/holdout_access.log.

    That file is the audit trail for holdout access; test entries in it (including
    synthetic split=holdout loads) make it unreadable as evidence.
    """
    from src import runlog

    monkeypatch.setattr(runlog, "LOGS", tmp_path_factory.mktemp("logs"))


@pytest.fixture(scope="session")
def small_slate():
    return synthetic.make_slate(n_matches=60, n_teams=12, seed=7)


@pytest.fixture(scope="session")
def tiny_slate():
    return synthetic.make_slate(n_matches=8, n_teams=6, seed=3)


@pytest.fixture(scope="session")
def fitted(small_slate):
    pipe = Pipeline(small_slate, CONFIG)
    res = pipe.run(small_slate.events, warm_start=False)
    return pipe, res


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
