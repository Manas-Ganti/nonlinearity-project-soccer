import pytest

from src.config import CONFIG
from src.ingest import loader, schema, synthetic


def test_validate_rejects_events_past_the_whistle(tiny_slate):
    bad = tiny_slate.events.copy()
    bad.loc[bad.index[0], "t"] = bad["match_T"].iloc[0] + 1.0
    with pytest.raises(schema.SchemaError, match="after the end"):
        schema.validate(bad)


def test_validate_rejects_impossible_xg(tiny_slate):
    bad = tiny_slate.events.copy()
    bad.loc[bad.index[0], "xg"] = 1.4
    with pytest.raises(schema.SchemaError, match="xg outside"):
        schema.validate(bad)


def test_validate_rejects_unknown_situations(tiny_slate):
    bad = tiny_slate.events.copy()
    bad["situation"] = bad["situation"].astype(object)
    bad.loc[bad.index[0], "situation"] = "freekick"
    with pytest.raises(schema.SchemaError, match="unknown situation"):
        schema.validate(bad)


def test_validate_rejects_varying_match_length(tiny_slate):
    bad = tiny_slate.events.copy()
    bad.loc[bad.index[0], "match_T"] = 90.0
    with pytest.raises(schema.SchemaError, match="match_T is not constant"):
        schema.validate(bad)


def test_match_frame_recovers_both_teams(tiny_slate):
    mf = schema.match_frame(tiny_slate.events)
    assert mf["home_team"].notna().all()
    assert mf["away_team"].notna().all()
    assert (mf["home_team"] != mf["away_team"]).all()


def test_slate_round_trips_through_disk(tmp_path, tiny_slate):
    tiny_slate.save(tmp_path)
    back = schema.Slate.load(tmp_path)
    assert len(back.events) == len(tiny_slate.events)
    assert len(back.goals) == len(tiny_slate.goals)
    assert back.red_cards_known == tiny_slate.red_cards_known


def test_filter_matches_keeps_all_three_streams(small_slate):
    keep = sorted(small_slate.events["match_id"].unique())[:5]
    sub = small_slate.filter_matches(keep)
    assert sub.events["match_id"].nunique() == 5
    assert set(sub.goals["match_id"]) <= set(keep)


def test_holdout_is_refused_without_the_flag(tmp_path, monkeypatch):
    slate = synthetic.make_slate(n_matches=20, n_teams=8, seasons=(2016, 2020), seed=5)
    monkeypatch.setattr(loader, "slate_dir", lambda source: tmp_path)
    slate.save(tmp_path)
    with pytest.raises(loader.HoldoutViolation, match="final-run"):
        loader.load_slate("x", split="all", caller="test")
    with pytest.raises(loader.HoldoutViolation):
        loader.load_slate("x", split="holdout", caller="test")


def test_dev_split_never_touches_holdout_seasons(tmp_path, monkeypatch):
    slate = synthetic.make_slate(n_matches=20, n_teams=8, seasons=(2016, 2020), seed=5)
    monkeypatch.setattr(loader, "slate_dir", lambda source: tmp_path)
    slate.save(tmp_path)
    dev = loader.load_slate("x", split="dev", caller="test")
    assert set(dev.events["season"].unique()) == {2016}


def test_holdout_access_is_logged(tmp_path, monkeypatch):
    from src.config import LOGS

    slate = synthetic.make_slate(n_matches=10, n_teams=6, seasons=(2016, 2020), seed=6)
    monkeypatch.setattr(loader, "slate_dir", lambda source: tmp_path)
    slate.save(tmp_path)
    logfile = LOGS / loader.HOLDOUT_LOG
    before = logfile.read_text() if logfile.exists() else ""
    loader.load_slate("x", split="all", final_run=True, caller="pytest-marker")
    after = logfile.read_text()
    assert "pytest-marker" in after[len(before) :]
    assert "touches_holdout=True" in after[len(before) :]


def test_prepare_model_slate_reports_what_it_removed(small_slate):
    _slate2, stats = loader.prepare_model_slate(small_slate, CONFIG, jitter_seed=1)
    assert stats["n_after"] <= stats["n_before"]
    assert 0.0 <= stats["removed_fraction"] <= CONFIG.dedup.max_removed_fraction
    assert stats["remaining_exact_ties"] == 0
    assert "jitter_seed" in stats
