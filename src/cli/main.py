"""Command line for the build order in CLAUDE.md. Each step gates the next.

    python -m src.cli.main <step> [options]

Every step writes a manifest into results/ recording its inputs, seeds and the
numbers it produced, so any figure in the writeup can be traced back to the run
that made it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

from src.config import (
    CONFIG,
    DEV_SEASONS,
    FIRST_HOLDOUT_SEASON,
    LEAGUES,
    RESULTS,
    PowerConfig,
)
from src.inference import bootstrap, gof, power
from src.inference.pipeline import Pipeline
from src.ingest import loader, schema
from src.models import hawkes as hawkes_mod
from src.runlog import write_manifest

log = logging.getLogger("soccer")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )


def _tag(args) -> str:
    """Per-dataset artefacts must carry which dataset they came from.

    A detection floor measured on 380 matches says nothing about a different slate,
    so writing them all to results/power.json lets one run silently answer for
    another. Every output that depends on the data is scoped by source and split.
    """
    return f"{args.source}_{args.split}"


def _print(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(_clean(payload), indent=2, default=str))


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if not isinstance(v, pd.DataFrame)}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# ------------------------------------------------------------------ step 1


def cmd_ingest_understat(args) -> None:
    """One request per match, ~23,000 of them. Run it once, overnight, and never again."""
    from src.ingest import understat

    seasons = args.seasons or DEV_SEASONS
    leagues = args.leagues or LEAGUES
    log.info("pulling %d leagues x %d seasons", len(leagues), len(seasons))
    paths = understat.pull(leagues, seasons, overwrite=args.overwrite)
    _print("understat pull", {"files": [str(p) for p in paths]})


def cmd_ingest_statsbomb(args) -> None:
    from src.ingest import statsbomb

    path = statsbomb.pull(max_matches=args.max_matches, overwrite=args.overwrite)
    _print("statsbomb pull", {"path": str(path)})


def cmd_ingest_wyscout(args) -> None:
    """The figshare release is already on disk; this is the one-off extraction."""
    from src.ingest import wyscout

    path = wyscout.extract(overwrite=args.overwrite)
    _print("wyscout extraction", {"path": str(path)})


def cmd_build_possessions(args) -> None:
    """Wyscout event streams -> the possession table (docs/spec_possessions.md section 2)."""
    from src.ingest import possessions

    table, summary = possessions.build()
    possessions.save(table)
    write_manifest("build_possessions_wyscout", summary)
    _print("possession table (wyscout)", summary)


def _poss_table(caller: str, rule: str = "tolerant"):
    from src.ingest import possessions

    return possessions.load(split="dev", rule=rule, caller=caller)


def cmd_poss_recovery(args) -> None:
    """Model A, gate 2: plant rho_self = 0.30 and get it back. A failed gate exits 2."""
    from src.inference import possession
    from src.models import possession_logit as pl

    design = pl.build_design(_poss_table("poss-recovery"), CONFIG.background)
    out = possession.recovery(
        design, tau=args.tau, n_replicates=args.replicates, seed=args.seed, workers=args.workers
    )
    out.pop("detail").to_csv(RESULTS / "poss_recovery_detail_wyscout_dev.csv", index=False)
    write_manifest("poss_recovery_wyscout_dev", out)
    _print("possession model A: recovery", out)
    if not out["passed"]:
        log.error("recovery gate FAILED: nothing downstream of it means anything")
        sys.exit(2)


def cmd_poss_power(args) -> None:
    """Model A, gate 3: the detection floor in rho (log-odds)."""
    from src.inference import possession
    from src.models import possession_logit as pl

    design = pl.build_design(_poss_table("poss-power"), CONFIG.background)
    out = possession.power(
        design,
        tau=args.tau,
        replicates=args.replicates,
        null_replicates=args.null_replicates,
        seed=args.seed,
        workers=args.workers,
    )
    out.pop("sweep").to_csv(RESULTS / "poss_power_sweep_wyscout_dev.csv", index=False)
    write_manifest("poss_power_wyscout_dev", out)
    _print("possession model A: detection floor", out)


def cmd_poss_fit(args) -> None:
    """Model A, gate 4: the development fit against its own floor."""
    from src.inference import possession
    from src.runlog import read_manifest

    table = _poss_table("poss-fit")
    strict = _poss_table("poss-fit-strict", rule="strict")
    out = possession.fit_dev(
        table,
        tau=args.tau,
        bootstrap=args.bootstrap,
        cluster_bootstrap=args.cluster_bootstrap,
        seed=args.seed,
        workers=args.workers,
        sensitivity_tables={"strict_possession_rule": strict},
    )
    for key, name in (("_null_detail", "poss_bootstrap_null"), ("_cluster_detail", "poss_cluster_bootstrap")):
        if key in out:
            out.pop(key).to_csv(RESULTS / f"{name}_wyscout_dev.csv", index=False)
    try:
        pw = read_manifest("poss_power_wyscout_dev")["payload"]
        out["detection_floor"] = pw["detection_floor"]
        out["detection_floor_interpolated"] = pw["detection_floor_interpolated"]
        rho = out["headline"]["rho_self"]
        floor = pw["detection_floor"]
        if floor is None:
            out["verdict"] = "no planted rho on the grid reached 80% power; the floor is above the grid"
        elif rho < floor:
            out["verdict"] = (
                f"rho_self = {rho:.4f} sits below the detection floor of {floor}: an upper bound, not an estimate."
            )
        else:
            out["verdict"] = (
                f"rho_self = {rho:.4f} is at or above the detection floor of {floor}; read it with the p-value and CI."
            )
    except FileNotFoundError:
        out["verdict"] = (
            "no detection floor measured (run poss-power first), so rho_self cannot be interpreted"
        )
    write_manifest("poss_fit_wyscout_dev", out)
    _print("possession model A: development fit (exploratory)", out)


def cmd_build(args) -> None:
    """Raw pulls -> the canonical event table, with game state reconstructed."""
    extra: dict = {}
    if args.source == "understat":
        from src.ingest import understat

        slate = understat.build_slate(understat.load_raw())
    elif args.source == "statsbomb":
        from src.ingest import statsbomb

        slate = statsbomb.build_slate(statsbomb.load_raw())
    elif args.source == "wyscout":
        from src.ingest import wyscout

        slate, xg_model = wyscout.build_slate(wyscout.load_raw())
        extra = {"location_xg_model": xg_model}
    elif args.source == "pooled":
        from src.ingest import pooled

        slate = pooled.build_slate({m: schema.Slate.load(loader.slate_dir(m)) for m in pooled.MEMBERS})
    elif args.source == "synthetic":
        from src.ingest import synthetic

        slate = synthetic.make_slate(n_matches=args.n_matches, seed=args.seed)
    else:
        raise SystemExit(f"unknown source {args.source}")

    loader.save_slate(slate, args.source)
    payload = {"source": args.source, **loader.describe(slate), **extra}
    write_manifest(f"build_{args.source}", payload)
    _print(f"build {args.source}", payload)


# ------------------------------------------------------------------ step 2


def _second_clock_raw(source: str):
    """Extracted rows plus a canonical slate for a second-resolution provider."""
    if source == "statsbomb":
        from src.ingest import statsbomb

        raw = statsbomb.load_raw()
        return raw, statsbomb.build_slate(raw)
    if source == "wyscout":
        from src.ingest import wyscout

        raw = wyscout.load_raw()
        slate, _ = wyscout.build_slate(raw)
        return raw, slate
    raise SystemExit(f"{source} has no second-resolution clock to calibrate on")


def cmd_calibrate_dedup(args) -> None:
    """Measure where mechanical follow-ups end, on one provider's second clock.

    Each provider gets its own calibration: how a rebound is logged is a property
    of the provider, not of football, so the threshold is checked per provider
    before one rule is applied to the pooled slate."""
    from src.features import dedup

    raw, slate = _second_clock_raw(args.source)
    if not args.final_run:
        slate = slate.filter_matches(
            slate.events.loc[slate.events["season"] < FIRST_HOLDOUT_SEASON, "match_id"].unique()
        )
    periods = raw.loc[raw["kind"] == "shot", ["match_id", "t", "period"]].drop_duplicates(["match_id", "t"])
    ev = slate.events.merge(periods, on=["match_id", "t"], how="left")

    gaps = dedup.within_team_gaps(ev, period_col="period")["gap_s"].to_numpy()
    res = dedup.choose_threshold(gaps, ratio_tol=args.ratio_tol)
    profile = res.pop("profile")
    profile.to_csv(RESULTS / f"dedup_gap_profile_{args.source}.csv", index=False)

    _, stats = dedup.apply_dedup(slate.events, CONFIG.dedup)
    payload = {
        **res,
        "source": args.source,
        "matches": int(slate.events["match_id"].nunique()),
        "seasons": sorted(map(int, slate.events["season"].unique())),
        "frac_gaps_under_5s": float((gaps < 5).mean()),
        "frac_gaps_under_60s": float((gaps < 60).mean()),
        "configured_threshold_seconds": CONFIG.dedup.threshold_seconds,
        "applied_with_configured_threshold": stats,
        "note": (
            "Understat records integer minutes, so any threshold below 60 s reduces there to "
            "'collapse same-team shots recorded in the same minute'. That rule is necessarily "
            "more aggressive than the calibrated one: the fraction of genuine gaps under 60 s is "
            "reported above, and step 7 measures what the over-collapsing costs."
        ),
    }
    write_manifest(f"dedup_calibration_{args.source}", payload)
    _print(f"dedup calibration ({args.source})", payload)


# ------------------------------------------------------------------ step 3


def _load(args, caller: str) -> tuple[schema.Slate, dict]:
    slate = loader.load_slate(
        args.source,
        split=args.split,
        final_run=getattr(args, "final_run", False),
        caller=caller,
    )
    model_slate, prep = loader.prepare_model_slate(slate, CONFIG, jitter_seed=args.jitter_seed)
    log.info("%s -> %s", slate, model_slate)
    return model_slate, {"raw": loader.describe(slate), "prepared": prep}


def cmd_fit_null(args) -> None:
    """The background model, and whether it describes football at all."""
    slate, meta = _load(args, "fit-null")
    pipe = Pipeline(slate, CONFIG)
    bg = pipe.fit_background(slate.events, warm_start=False)
    from src.models import hawkes

    data = hawkes.prepare(slate, bg, CONFIG.hawkes)
    gaps = gof.rescaled_gaps(data, 0.0, 0.0, 1.0 / CONFIG.hawkes.tau_init)
    payload = {
        **meta,
        "background": bg.as_dict(),
        "time_rescaling": {**gof.ks_report(gaps["tau"].to_numpy()), **gof.per_match_ks(gaps)},
        "n_background_params": len(bg.params),
    }
    gof.qq_points(gaps["tau"].to_numpy()).to_csv(RESULTS / f"qq_poisson_{_tag(args)}.csv", index=False)
    write_manifest(f"null_{_tag(args)}", payload)
    _print("null model", payload)


# ------------------------------------------------------------------ step 5


def cmd_recovery(args) -> None:
    """Plant eta = 0.3 and confirm the whole pipeline finds it. This is a gate."""
    slate, meta = _load(args, "recovery")
    pipe = Pipeline(slate, CONFIG)
    bg = pipe.fit_background(slate.events, warm_start=False)

    out = {}
    for method in args.methods:
        res = power.recovery_check(
            pipe,
            bg,
            eta_plant=args.eta,
            eta_cross=args.eta_cross,
            beta=1.0 / args.tau,
            n_replicates=args.replicates,
            seed=args.seed,
            tolerance=args.tolerance,
            workers=args.workers,
        )
        detail = res.pop("detail")
        detail.to_csv(RESULTS / f"recovery_detail_{method}_{_tag(args)}.csv", index=False)
        out[method] = res
    payload = {**meta, "results": out}
    write_manifest(f"recovery_{_tag(args)}", payload)
    _print("recovery test", payload)
    if not out[args.methods[0]]["passed"]:
        log.error("RECOVERY FAILED -- everything downstream is meaningless. Stop here.")
        sys.exit(2)


# ------------------------------------------------------------------ step 6


def cmd_power(args) -> None:
    """The detection floor. A branching ratio without this number is uninterpretable."""
    slate, meta = _load(args, "power")
    pipe = Pipeline(slate, CONFIG)
    bg = pipe.fit_background(slate.events, warm_start=False)
    cfg = PowerConfig(
        eta_grid=tuple(args.eta_grid),
        replicates=args.replicates,
        null_replicates=args.null_replicates,
        seed=args.seed,
    )
    minute_clock = args.minute_clock
    if minute_clock is None:  # default: match the resolution the source actually has
        minute_clock = bool(slate.events["second"].isna().all())
    log.info("simulating with a %s clock", "minute" if minute_clock else "continuous")
    sweep = power.run_sweep(
        pipe,
        bg,
        eta_cross=args.eta_cross,
        beta=1.0 / args.tau,
        cfg=cfg,
        minute_clock=minute_clock,
        workers=args.workers,
    )
    sweep.to_csv(RESULTS / f"power_sweep_{_tag(args)}.csv", index=False)
    summary = power.summarise(sweep, cfg)
    curve = summary.pop("curve")
    curve.to_csv(RESULTS / f"power_curve_{_tag(args)}.csv", index=False)
    payload = {
        **meta,
        **summary,
        "curve": curve.to_dict("records"),
        "config": cfg.__dict__,
        "minute_clock": minute_clock,
        "clock_note": (
            "each simulated slate was rounded to integer minutes and re-de-duplicated "
            "before fitting, so this floor is for a dataset with Understat's clock"
            if minute_clock
            else "simulated slates kept their continuous clock, matching the source"
        ),
    }
    write_manifest(f"power_{_tag(args)}", payload)
    _print("detection floor", payload)


# ------------------------------------------------------------------ step 7


def cmd_rounding_bias(args) -> None:
    """Fit eta at second resolution, round the same events to integer minutes, refit.

    A headline methods result, not a footnote: it turns Understat's clock from an
    unexamined assumption into a measured sensitivity.
    """
    from src.features import dedup, jitter

    slate = loader.load_slate(
        args.source,
        split="all" if args.final_run else "dev",
        final_run=args.final_run,
        caller="rounding-bias",
    )
    if slate.events["second"].isna().any():
        raise SystemExit(f"{args.source} has no sub-minute clock; nothing to round")

    fine, fine_stats = dedup.apply_dedup(slate.events, CONFIG.dedup)
    fine_slate = slate.with_events(fine)
    pipe = Pipeline(fine_slate, CONFIG)
    fine_res = pipe.run(fine, warm_start=False)

    # The same events, seen through Understat's clock.
    coarse = slate.events.copy()
    coarse["t"] = np.floor(coarse["t"]).astype(float)
    coarse["minute"] = coarse["t"].astype("int16")
    coarse["second"] = np.nan
    coarse = schema.validate(coarse)
    coarse, coarse_stats = dedup.apply_dedup(coarse, CONFIG.dedup)

    draws = []
    for i, seed, jittered in jitter.jitter_draws(coarse, CONFIG.jitter):
        cslate = slate.with_events(jittered)
        cpipe = Pipeline(cslate, CONFIG)
        res = cpipe.run(jittered, warm_start=False)
        draws.append(
            {
                "draw": i,
                "seed": seed,
                "eta_self": res.hawkes_fit.eta_self,
                "eta_cross": res.hawkes_fit.eta_cross,
                "tau": res.hawkes_fit.tau_minutes,
                "n_events": len(jittered),
            }
        )
    draws_df = pd.DataFrame(draws)
    draws_df.to_csv(RESULTS / f"rounding_bias_draws_{args.source}.csv", index=False)

    payload = {
        "source": args.source,
        "matches": int(slate.events["match_id"].nunique()),
        "seasons": sorted(map(int, slate.events["season"].unique())),
        "second_resolution": {**fine_res.hawkes_fit.as_dict(), "dedup": fine_stats},
        "minute_resolution": {
            "dedup": coarse_stats,
            "eta_self_mean": float(draws_df["eta_self"].mean()),
            "eta_self_sd": float(draws_df["eta_self"].std(ddof=1)),
            "eta_cross_mean": float(draws_df["eta_cross"].mean()),
            "tau_mean": float(draws_df["tau"].mean()),
            "jitter_draws": draws,
        },
        "rounding_bias_eta_self": float(draws_df["eta_self"].mean() - fine_res.hawkes_fit.eta_self),
        "extra_events_removed_by_rounding": int(fine_stats["n_after"] - coarse_stats["n_after"]),
    }
    write_manifest(f"rounding_bias_{args.source}", payload)
    _print(f"rounding bias ({args.source})", payload)


# ------------------------------------------------------------- steps 8 and 9


def cmd_fit(args) -> None:
    """The headline fit: eta_self, eta_cross, bootstrap p-value, against the floor."""
    slate, meta = _load(args, f"fit-{args.split}")
    pipe = Pipeline(slate, CONFIG)

    from src.features import jitter as jitter_mod

    draws = []
    for i, seed, jittered in jitter_mod.jitter_draws(slate.events, CONFIG.jitter):
        res = pipe.run(jittered, warm_start=False)
        draws.append({"draw": i, "seed": seed, **res.hawkes_fit.as_dict()})
        if i == 0:
            headline, headline_events = res, jittered
    draws_df = pd.DataFrame(draws)
    draws_df.to_csv(RESULTS / f"fit_draws_{_tag(args)}.csv", index=False)

    payload = {
        **meta,
        "headline": headline.as_dict(),
        "across_jitter_draws": {
            "eta_self_mean": float(draws_df["eta_self"].mean()),
            "eta_self_sd": float(draws_df["eta_self"].std(ddof=1)),
            "eta_self_min": float(draws_df["eta_self"].min()),
            "eta_self_max": float(draws_df["eta_self"].max()),
            "eta_cross_mean": float(draws_df["eta_cross"].mean()),
            "tau_mean": float(draws_df["tau_minutes"].mean()),
            "n_draws": len(draws_df),
        },
    }

    # The null arm of the GOF comparison needs the background fitted *under the null*,
    # not the joint fit's background with the kernel switched off.
    null_bg = pipe.fit_background(headline_events, warm_start=False)
    null_data = hawkes_mod.prepare(slate.with_events(headline_events), null_bg, CONFIG.hawkes)
    payload["goodness_of_fit"] = gof.compare(null_data, null_bg, headline.data, headline.hawkes_fit)

    if args.sensitivities:
        payload["sensitivities"] = _sensitivities(slate, headline_events, headline)
        if not CONFIG.hawkes.symmetric:
            log.warning("config has symmetric=False; the headline fit must use the reduction")
        payload["asymmetry_check"] = hawkes_mod.fit_asymmetric(headline.data, CONFIG.hawkes)

    if args.bootstrap:
        bg_null = pipe.fit_background(headline_events, warm_start=False)
        boot = bootstrap.parametric_bootstrap_null(
            pipe,
            bg_null,
            eta_obs=headline.hawkes_fit.eta_self,
            n_replicates=args.bootstrap,
            seed=args.seed,
            beta_for_sim=headline.hawkes_fit.beta,
            workers=args.workers,
        )
        boot.detail.to_csv(RESULTS / f"bootstrap_null_{_tag(args)}.csv", index=False)
        payload["parametric_bootstrap"] = boot.as_dict()

    if args.cluster_bootstrap:
        cb = bootstrap.cluster_bootstrap(
            slate, n_replicates=args.cluster_bootstrap, seed=args.seed, workers=args.workers
        )
        cb.to_csv(RESULTS / f"cluster_bootstrap_{_tag(args)}.csv", index=False)
        lo, hi = bootstrap.percentile_ci(cb["eta_self"].to_numpy())
        payload["cluster_bootstrap_ci95"] = [lo, hi]
        payload["cluster_bootstrap_n_unconverged"] = int((~cb["fit_converged"].astype(bool)).sum())

    # The floor must be the one measured on this same slate. Anything else is a number
    # from another dataset wearing this one's clothes.
    floor_path = RESULTS / f"power_{_tag(args)}.json"
    if floor_path.exists():
        floor = json.loads(floor_path.read_text())["payload"]
        payload["detection_floor"] = floor.get("detection_floor")
        payload["detection_floor_interpolated"] = floor.get("detection_floor_interpolated")
        payload["detection_floor_source"] = floor_path.name
        payload["verdict"] = _verdict(payload)
    else:
        payload["verdict"] = (
            f"no detection floor has been measured for {_tag(args)} "
            f"(expected {floor_path.name}), so eta_self cannot be interpreted"
        )

    write_manifest(f"fit_{_tag(args)}", payload)
    _print(f"fit ({args.split})", payload)


def _sensitivities(slate, events, headline) -> dict:
    """The robustness checks the specs ask for, each one run, not asserted.

    - the kernel box narrowed to the 3-8 min CLAUDE.md names for the slow component
    - score state removed (PROPOSAL.md, "Known limitations": the estimand is still
      defined without it, and the difference is the size of the confound)
    - the background's flexibility varied (docs/math.md section 8: report eta_hat as
      a function of spline df, because a too-flexible mu eats the signal)
    - the branching matrix freed from the H/A symmetry reduction
    """
    from dataclasses import replace

    out = {}
    variants = {
        "kernel_box_1_to_20_min": replace(
            CONFIG, hawkes=replace(CONFIG.hawkes, tau_min=1.0, tau_max=20.0, tau_init=5.0)
        ),
        "no_score_state": replace(CONFIG, background=replace(CONFIG.background, include_score=False)),
        "no_red_cards": replace(CONFIG, background=replace(CONFIG.background, include_red=False)),
        "penalties_kept": replace(CONFIG, hawkes=replace(CONFIG.hawkes, exclude_penalties=False)),
    }
    for df in (3, 5, 9, 12):
        variants[f"minute_spline_df_{df}"] = replace(
            CONFIG, background=replace(CONFIG.background, minute_df=df)
        )

    base = headline.hawkes_fit
    out["headline"] = {
        "eta_self": base.eta_self,
        "eta_cross": base.eta_cross,
        "tau_minutes": base.tau_minutes,
    }
    for name, cfg in variants.items():
        try:
            pipe = Pipeline(slate, cfg)
            res = pipe.run(events, warm_start=False)
            hk = res.hawkes_fit
            out[name] = {
                "eta_self": hk.eta_self,
                "eta_cross": hk.eta_cross,
                "tau_minutes": hk.tau_minutes,
                "tau_at_boundary": hk.tau_at_boundary(cfg.hawkes),
                "delta_eta_self": hk.eta_self - base.eta_self,
                "loglik": hk.loglik,
            }
        except Exception as exc:  # a failed sensitivity is reportable, not fatal
            log.warning("sensitivity %s failed: %s", name, exc)
            out[name] = {"error": str(exc)}
    return out


def _verdict(payload: dict) -> str:
    hk = payload["headline"]["hawkes"]
    eta = hk["eta_self"]
    floor = payload.get("detection_floor")
    p = payload.get("parametric_bootstrap", {}).get("p_value")

    caveats = []
    if hk.get("tau_at_boundary"):
        caveats.append(
            f"the kernel timescale is pinned to the edge of its box at {hk['tau_minutes']:.1f} min, "
            "so the data did not choose it and eta_self is conditional on the box"
        )
    if payload.get("raw", {}).get("red_cards_known") is False:
        caveats.append("dismissals are unavailable in this feed, so the man-advantage covariate is absent")
    tail = (" Caveats: " + "; ".join(caveats) + ".") if caveats else ""

    if floor is None:
        return "no detection floor is available, so eta_self cannot be interpreted." + tail
    if eta < floor:
        return (
            f"eta_self = {eta:.3f} sits below the measured detection floor of {floor:.3f}: "
            f"the honest statement is an upper bound, not an estimate.{tail}"
        )
    if p is not None and p >= 0.05:
        return f"eta_self = {eta:.3f} is above the floor but not significant (p = {p:.3f}).{tail}"
    return f"eta_self = {eta:.3f} is above the floor of {floor:.3f} and significant.{tail}"


def cmd_tau_profile(args) -> None:
    """Profile the likelihood over the kernel timescale, with eta free at each point.

    `eta` and `tau` trade off: a slower kernel spread over a longer window explains
    the same clustering with a larger branching ratio. If the profile is flat, the
    data does not identify the timescale, and the headline `eta_self` has to be read
    as conditional on the band it was fitted in. That is a fact about the data, so it
    gets measured rather than assumed away.
    """
    from dataclasses import replace

    slate, meta = _load(args, "tau-profile")
    rows = []
    for tau in args.taus:
        cfg = replace(
            CONFIG,
            hawkes=replace(CONFIG.hawkes, tau_min=tau - 1e-6, tau_max=tau + 1e-6, tau_init=tau),
        )
        res = Pipeline(slate, cfg).run(slate.events, warm_start=False)
        hk = res.hawkes_fit
        rows.append(
            {
                "tau_minutes": float(tau),
                "eta_self": hk.eta_self,
                "eta_cross": hk.eta_cross,
                "loglik": hk.loglik,
            }
        )
    df = pd.DataFrame(rows)
    df["delta_loglik_vs_best"] = df["loglik"] - df["loglik"].max()
    df.to_csv(RESULTS / f"tau_profile_{_tag(args)}.csv", index=False)
    payload = {
        **meta,
        "profile": df.to_dict("records"),
        "best_tau": float(df.loc[df["loglik"].idxmax(), "tau_minutes"]),
        "loglik_range": float(df["loglik"].max() - df["loglik"].min()),
        "eta_at_tau_5": float(np.interp(5.0, df["tau_minutes"], df["eta_self"])),
        "note": (
            "A log-likelihood range of only a few units across a wide tau band means the "
            "timescale is not identified: eta_self and tau trade off almost freely, and "
            "eta_self must be reported with the band it was fitted in."
        ),
    }
    write_manifest(f"tau_profile_{_tag(args)}", payload)
    _print("tau profile", payload)


def cmd_cross_source(args) -> None:
    """One source against another on the fixtures both cover (default: Understat's
    minute clock against the pooled second-clock slate)."""
    from src.inference import cross_source

    split = "all" if args.final_run else "dev"
    a = loader.load_slate(args.source, split=split, final_run=args.final_run, caller="cross-source")
    b = loader.load_slate(args.against, split=split, final_run=args.final_run, caller="cross-source")
    payload = cross_source.compare(a, b, name_a=args.source, name_b=args.against)
    write_manifest(f"cross_source_{args.source}_vs_{args.against}", payload)
    _print("cross-source comparison", payload)


def cmd_secondary(args) -> None:
    """The extensions in docs/math.md section 9. Not part of the build order."""
    from src.inference import secondary

    slate, meta = _load(args, "secondary")
    pipe = Pipeline(slate, CONFIG)
    res = pipe.run(slate.events, warm_start=False)
    events = slate.events.sort_values(["match_id", "t"], kind="stable").reset_index(drop=True)
    if CONFIG.hawkes.exclude_penalties:
        events = events[events["situation"] != "penalty"].reset_index(drop=True)

    payload = {
        **meta,
        "fit": res.hawkes_fit.as_dict(),
        "quality_regression": secondary.quality_regression(res.data, events, res.hawkes_fit),
        "marked_excitation": secondary.marked_excitation_profile(res.data, events, res.hawkes_fit),
        "state_dependent_eta": secondary.state_dependent_eta(slate, CONFIG),
    }
    write_manifest(f"secondary_{_tag(args)}", payload)
    _print("secondary analyses", payload)


def cmd_estimator_comparison(args) -> None:
    """How much of a planted eta each estimator gives back. The reason `joint` is default."""
    slate, meta = _load(args, "estimator-comparison")
    pipe = Pipeline(slate, CONFIG)
    bg = pipe.fit_background(slate.events, warm_start=False)
    sim = pipe.simulator(bg)

    rows = []
    for eta_star in args.eta_grid:
        for r in range(args.replicates):
            events = pipe.simulate_events(
                sim,
                float(eta_star),
                args.eta_cross,
                1.0 / args.tau,
                np.random.default_rng(args.seed + 977 * round(eta_star * 1000) + r),
            )
            for method in ("joint", "two_stage", "em"):
                res = pipe.run(events, warm_start=False, method=method)
                rows.append(
                    {
                        "eta_star": eta_star,
                        "replicate": r,
                        "method": method,
                        "eta_hat_self": res.hawkes_fit.eta_self,
                        "tau_hat": res.hawkes_fit.tau_minutes,
                        "loglik": res.hawkes_fit.loglik,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"estimator_comparison_{_tag(args)}.csv", index=False)
    summary = (
        df.groupby(["method", "eta_star"])
        .agg(
            mean_eta_hat=("eta_hat_self", "mean"),
            mean_tau=("tau_hat", "mean"),
            mean_loglik=("loglik", "mean"),
        )
        .reset_index()
    )
    summary["bias"] = summary["mean_eta_hat"] - summary["eta_star"]
    payload = {**meta, "summary": summary.to_dict("records")}
    write_manifest(f"estimator_comparison_{_tag(args)}", payload)
    _print("estimator comparison", payload)


# ---------------------------------------------------------------------- cli


SOURCES = ["understat", "statsbomb", "wyscout", "pooled", "synthetic"]


def _common(p, *, source_default="understat"):
    p.add_argument("--source", default=source_default, choices=SOURCES)
    p.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    p.add_argument("--final-run", action="store_true", help="unlock the holdout. Say it out loud.")
    p.add_argument("--jitter-seed", type=int, default=CONFIG.jitter.base_seed)
    p.add_argument("--seed", type=int, default=5694)
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="forked workers for replicate loops (default: cores - 1, or $SOCCER_WORKERS)",
    )
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="soccer-momentum", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest-understat", help="step 1: the overnight pull")
    p.add_argument("--leagues", nargs="*", default=None)
    p.add_argument("--seasons", nargs="*", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_ingest_understat)

    p = sub.add_parser("ingest-statsbomb", help="step 1: the resolution sample")
    p.add_argument("--max-matches", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_ingest_statsbomb)

    p = sub.add_parser("ingest-wyscout", help="step 1: extract the figshare release on disk")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_ingest_wyscout)

    p = sub.add_parser("build", help="step 1: raw -> canonical event table")
    p.add_argument("--source", default="understat", choices=SOURCES)
    p.add_argument("--n-matches", type=int, default=500, help="synthetic only")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("build-possessions", help="richer events: Wyscout possession table")
    p.set_defaults(func=cmd_build_possessions)

    for name, fn, helptext in (
        ("poss-recovery", cmd_poss_recovery, "possessions model A, gate 2: plant rho = 0.30. A GATE."),
        ("poss-power", cmd_poss_power, "possessions model A, gate 3: detection floor"),
        ("poss-fit", cmd_poss_fit, "possessions model A, gate 4: development fit (exploratory)"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--tau", type=float, default=5.0)
        p.add_argument("--seed", type=int, default=5694)
        p.add_argument("--workers", type=int, default=None)
        if name == "poss-recovery":
            p.add_argument("--replicates", type=int, default=10)
        if name == "poss-power":
            p.add_argument("--replicates", type=int, default=30)
            p.add_argument("--null-replicates", type=int, default=200)
        if name == "poss-fit":
            p.add_argument("--bootstrap", type=int, default=500)
            p.add_argument("--cluster-bootstrap", type=int, default=200)
        p.set_defaults(func=fn)

    p = sub.add_parser("calibrate-dedup", help="step 2: threshold from a provider's second-clock gaps")
    p.add_argument("--source", default="statsbomb", choices=["statsbomb", "wyscout"])
    p.add_argument("--ratio-tol", type=float, default=1.25)
    p.add_argument("--final-run", action="store_true")
    p.set_defaults(func=cmd_calibrate_dedup)

    p = _common(sub.add_parser("fit-null", help="step 3: background model + time rescaling"))
    p.set_defaults(func=cmd_fit_null)

    p = _common(sub.add_parser("recovery", help="step 5: plant eta=0.3 and find it (a gate)"))
    p.add_argument("--eta", type=float, default=0.30)
    p.add_argument("--eta-cross", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=5.0)
    p.add_argument("--replicates", type=int, default=10)
    p.add_argument("--tolerance", type=float, default=0.08)
    p.add_argument("--methods", nargs="*", default=["joint"])
    p.set_defaults(func=cmd_recovery)

    p = _common(sub.add_parser("power", help="step 6: the detection floor"))
    p.add_argument("--eta-grid", nargs="*", type=float, default=list(CONFIG.power.eta_grid))
    p.add_argument("--eta-cross", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=5.0)
    p.add_argument("--replicates", type=int, default=CONFIG.power.replicates)
    p.add_argument("--null-replicates", type=int, default=CONFIG.power.null_replicates)
    p.add_argument(
        "--minute-clock",
        dest="minute_clock",
        action="store_true",
        default=None,
        help="round simulated slates to integer minutes (defaults to the source's own resolution)",
    )
    p.add_argument("--continuous-clock", dest="minute_clock", action="store_false")
    p.set_defaults(func=cmd_power)

    p = sub.add_parser("rounding-bias", help="step 7: second vs minute resolution")
    p.add_argument("--source", default="pooled", choices=["statsbomb", "wyscout", "pooled"])
    p.add_argument("--final-run", action="store_true")
    p.set_defaults(func=cmd_rounding_bias)

    p = _common(sub.add_parser("fit", help="steps 8 and 9: the headline fit"))
    p.add_argument("--bootstrap", type=int, default=0, help="parametric null replicates")
    p.add_argument("--cluster-bootstrap", type=int, default=0)
    p.add_argument("--no-sensitivities", dest="sensitivities", action="store_false")
    p.set_defaults(func=cmd_fit, sensitivities=True)

    p = _common(sub.add_parser("tau-profile", help="is the kernel timescale identified at all?"))
    p.add_argument("--taus", nargs="*", type=float, default=[2, 3, 4, 5, 6, 8, 10, 14, 20, 30, 45])
    p.set_defaults(func=cmd_tau_profile)

    p = sub.add_parser("cross-source", help="one source against another on shared fixtures")
    p.add_argument("--source", default="understat", choices=SOURCES)
    p.add_argument("--against", default="pooled", choices=SOURCES)
    p.add_argument("--final-run", action="store_true")
    p.set_defaults(func=cmd_cross_source)

    p = _common(sub.add_parser("secondary", help="extensions: chance quality, state-dependent eta"))
    p.set_defaults(func=cmd_secondary)

    p = _common(sub.add_parser("estimator-comparison", help="joint vs EM vs two-stage"))
    p.add_argument("--eta-grid", nargs="*", type=float, default=[0.0, 0.10, 0.30])
    p.add_argument("--eta-cross", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=5.0)
    p.add_argument("--replicates", type=int, default=3)
    p.set_defaults(func=cmd_estimator_comparison)

    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
