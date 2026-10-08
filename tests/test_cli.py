"""The CLI is the interface the build order is run through, so it gets a smoke test."""

import pytest

from src.cli import main as cli
from src.config import CONFIG


def test_every_build_step_has_a_subcommand():
    parser = cli.build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    names = set(actions[0].choices)
    for step in ("ingest-understat", "ingest-statsbomb", "build", "calibrate-dedup",
                 "fit-null", "recovery", "power", "rounding-bias", "fit"):
        assert step in names


def test_holdout_needs_the_flag_at_the_command_line():
    args = cli.build_parser().parse_args(["fit", "--split", "holdout"])
    assert args.final_run is False
    args = cli.build_parser().parse_args(["fit", "--split", "holdout", "--final-run"])
    assert args.final_run is True


def test_config_is_serialisable_for_the_run_manifest():
    d = CONFIG.to_dict()
    assert d["hawkes"]["exclude_penalties"] is True
    assert d["hawkes"]["symmetric"] is True
    assert d["dedup"]["threshold_seconds"] > 0
    assert 0.0 in d["power"]["eta_grid"], "the sweep needs a null arm to calibrate against"


def test_stationarity_cap_is_below_one():
    assert CONFIG.hawkes.rho_max < 1.0


def test_dev_and_holdout_seasons_do_not_overlap():
    from src.config import DEV_SEASONS, FIRST_HOLDOUT_SEASON, HOLDOUT_SEASONS

    assert set(DEV_SEASONS).isdisjoint(HOLDOUT_SEASONS)
    assert max(DEV_SEASONS) < FIRST_HOLDOUT_SEASON
    assert min(HOLDOUT_SEASONS) == FIRST_HOLDOUT_SEASON


def test_verdict_calls_a_below_floor_estimate_an_upper_bound():
    payload = {"headline": {"hawkes": {"eta_self": 0.02}}, "detection_floor": 0.10}
    assert "upper bound" in cli._verdict(payload)


def test_verdict_needs_a_floor():
    payload = {"headline": {"hawkes": {"eta_self": 0.4}}}
    assert "cannot be interpreted" in cli._verdict(payload)


@pytest.mark.parametrize("bad", [["fit", "--split", "nonsense"], ["build", "--source", "espn"]])
def test_bad_arguments_are_rejected(bad):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(bad)
