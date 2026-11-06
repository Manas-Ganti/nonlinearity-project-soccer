"""CLI for Models S and G (docs/spec_spells.md). Registered by `src.cli.main`.

    spell-recovery  --model S|G --source wyscout|statsbomb     gate (exit 2 on failure)
    spell-power     --model S|G --source ...                   floors at alpha = 0.01
    spell-fit       --model S|G --source ...                   development fit (exploratory)
    spell-holm      --source ...                               Holm over the five tests

`--exclude-one-club` drops the one-club (Barcelona) matches, for the section 9
confirmation. Development seasons only; nothing here takes --final-run.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace

from src.config import CONFIG, RESULTS
from src.runlog import read_manifest, write_manifest

log = logging.getLogger("soccer")

MERGE_S = 10.0  # spec section 2: quick-regain merged possessions
RULE = {"need": 4, "window": 5, "danger_col": "t_ft"}  # spec section 3


def _tag(args) -> str:
    return f"{args.model}_{args.source}{'_no1club' if args.exclude_one_club else ''}_dev"


def _inputs(args, caller: str, *, rule: str = "tolerant", spell_rule: dict | None = None, cfg=CONFIG):
    """(model slate, spell path, prep stats, prevalence)."""
    from src.cli.main import _poss_table
    from src.features import spells
    from src.ingest import loader

    slate = loader.load_slate(args.source, split="dev", caller=caller)
    if args.exclude_one_club:
        keep = slate.events.loc[~slate.events["one_club_sample"].astype(bool), "match_id"].unique()
        slate = slate.filter_matches(keep)
    model_slate, prep = loader.prepare_model_slate(slate, cfg, jitter_seed=cfg.jitter.base_seed)
    poss = _poss_table(caller, rule=rule, source=args.source, merge_s=MERGE_S)
    merge = poss.attrs.get("merge")
    poss = poss[poss["match_id"].isin(set(model_slate.events["match_id"]))]
    r = {**RULE, **(spell_rule or {})}
    spell = spells.build(poss, need=r["need"], window=r["window"], danger_col=r["danger_col"])
    ev = model_slate.events
    prev = spells.prevalence(spell, ev[ev["situation"] != "penalty"])
    meta = {
        "source": args.source,
        "exclude_one_club": bool(args.exclude_one_club),
        "n_matches": int(ev["match_id"].nunique()),
        "n_shots_modelled": int((ev["situation"] != "penalty").sum()),
        "prep": prep,
        "possession_rule": rule,
        "merge": merge,
        "spell_rule": r,
        "prevalence": prev,
    }
    return model_slate, spell, meta


def _pipeline(args, slate, spell, *, gamma=True, cfg=CONFIG):
    from src.inference.spells import SpellPipeline

    return SpellPipeline(slate, args.model, spell=spell, gamma=gamma, cfg=cfg)


def cmd_spell_recovery(args) -> None:
    from src.inference import spells as inf

    slate, spell, meta = _inputs(args, "spell-recovery")
    pipe = _pipeline(args, slate, spell)
    null = pipe.run(slate.events, kernel=False).fit
    out = inf.recovery(
        pipe, null, tau=args.tau, n_replicates=args.replicates, seed=args.seed, workers=args.workers
    )
    out.pop("detail").to_csv(RESULTS / f"spell_recovery_detail_{_tag(args)}.csv", index=False)
    payload = {**meta, **out}
    write_manifest(f"spell_recovery_{_tag(args)}", payload)
    _show("spell recovery", payload)
    if not out["passed"]:
        log.error("SPELL RECOVERY FAILED (model %s): nothing downstream runs", args.model)
        sys.exit(2)


def _gate(args) -> None:
    try:
        rec = read_manifest(f"spell_recovery_{_tag(args)}")["payload"]
    except FileNotFoundError:
        log.error("no recovery manifest for %s: run spell-recovery first", _tag(args))
        sys.exit(2)
    if not rec["passed"]:
        log.error("recovery failed for %s: refusing to continue", _tag(args))
        sys.exit(2)


def cmd_spell_power(args) -> None:
    from src.inference import spells as inf

    _gate(args)
    slate, spell, meta = _inputs(args, "spell-power")
    pipe = _pipeline(args, slate, spell)
    null = pipe.run(slate.events, kernel=False).fit
    out = inf.power(
        pipe,
        null,
        tau=args.tau,
        replicates=args.replicates,
        null_replicates=args.null_replicates,
        seed=args.seed,
        workers=args.workers,
    )
    out.pop("sweep").to_csv(RESULTS / f"spell_power_sweep_{_tag(args)}.csv", index=False)
    payload = {**meta, **out}
    write_manifest(f"spell_power_{_tag(args)}", payload)
    _show("spell detection floors", payload)


def _sensitivities(args, slate, spell) -> dict:
    """Section 7, point fits only, run in parallel (one forked worker per fit)."""
    from src.inference import spells as inf
    from src.inference.parallel import pmap

    def fit_with(pipe, events=None, hawkes_cfg=None):
        return pipe.run(slate.events if events is None else events, hawkes_cfg=hawkes_cfg).as_dict()

    jobs = []
    box = replace(CONFIG.hawkes, tau_min=1.0, tau_max=20.0)
    jobs.append(("box_1_20", lambda: fit_with(_pipeline(args, slate, spell), hawkes_cfg=box)))
    for df in (9, 12):
        cfg = replace(CONFIG, background=replace(CONFIG.background, minute_df=df))
        jobs.append(
            (f"minute_spline_df_{df}", lambda cfg=cfg: fit_with(_pipeline(args, slate, spell, cfg=cfg)))
        )
    if args.model == "S":
        jobs.append(("no_gamma_confounded", lambda: fit_with(_pipeline(args, slate, spell, gamma=False))))
        for name, rule, sr in (
            ("rule_3_of_5", "tolerant", {"need": 3, "window": 5}),
            ("rule_5_of_5", "tolerant", {"need": 5, "window": 5}),
            ("rule_6_of_8", "tolerant", {"need": 6, "window": 8}),
            ("danger_x75", "tolerant", {"danger_col": "t_ft75"}),
            ("strict_possession_rule", "strict", None),
        ):

            def one(rule=rule, sr=sr, name=name):
                s2, sp2, meta2 = _inputs(args, f"spell-fit-{name}", rule=rule, spell_rule=sr)
                return {
                    **fit_with(_pipeline(args, s2, sp2), events=s2.events),
                    "prevalence": meta2["prevalence"],
                }

            jobs.append((name, one))
    for tau in inf.TAU_PROFILE:
        jobs.append(
            (
                f"tau_{tau:g}",
                lambda tau=tau: inf.tau_profile(_pipeline(args, slate, spell), slate.events, taus=(tau,))[0],
            )
        )

    def run(i):
        # only the index crosses the process boundary: the jobs are closures, which
        # cannot be pickled, and a forked worker inherits the list as it stands
        name, fn = jobs[i]
        try:
            return name, fn()
        except Exception as exc:  # a sensitivity failing must not lose the headline
            log.exception("sensitivity %s failed", name)
            return name, {"error": repr(exc)}

    done = dict(pmap(run, range(len(jobs)), workers=args.workers, label="spell sensitivities", every=1))
    out = {k: v for k, v in done.items() if not k.startswith("tau_")}
    out["tau_profile"] = [done[f"tau_{t:g}"] for t in inf.TAU_PROFILE]
    return out


def cmd_spell_fit(args) -> None:
    from src.inference import spells as inf

    _gate(args)
    slate, spell, meta = _inputs(args, "spell-fit")
    pipe = _pipeline(args, slate, spell)
    null = pipe.run(slate.events, kernel=False)
    head = pipe.run(slate.events)
    h = head.as_dict()
    log.info("headline %s", {k: round(v, 5) for k, v in h.items() if k.startswith("eta_")})
    payload = {**meta, "headline": h, "background_only": null.as_dict()}
    payload["lr_vs_background"] = 2.0 * (head.fit.loglik - null.fit.loglik)

    if args.bootstrap:
        pb = inf.parametric_bootstrap(
            pipe, null.fit, h, n_replicates=args.bootstrap, seed=args.seed, workers=args.workers
        )
        pb.pop("_detail").to_csv(RESULTS / f"spell_bootstrap_null_{_tag(args)}.csv", index=False)
        payload["parametric_bootstrap"] = pb
    if args.cluster_bootstrap:
        cb = inf.cluster_bootstrap(
            pipe, n_replicates=args.cluster_bootstrap, seed=args.seed, workers=args.workers
        )
        cb.pop("_detail").to_csv(RESULTS / f"spell_cluster_bootstrap_{_tag(args)}.csv", index=False)
        payload["cluster_bootstrap"] = cb
    _verdict(args, payload, h)
    if not args.no_sensitivities:
        # checkpoint first: the headline and bootstraps are hours of work
        write_manifest(f"spell_fit_{_tag(args)}", {**payload, "sensitivities": "pending"})
        payload["sensitivities"] = _sensitivities(args, slate, spell)
    write_manifest(f"spell_fit_{_tag(args)}", payload)
    _show("spell model development fit", payload)


def _verdict(args, payload, h) -> None:
    try:
        pw = read_manifest(f"spell_power_{_tag(args)}")["payload"]
        payload["floors"] = {k: v["detection_floor"] for k, v in pw["floors"].items()}
        verdict = {}
        for lab, floor in payload["floors"].items():
            est = h[f"eta_{lab}"]
            if floor is None:
                verdict[lab] = "no planted value on the grid reached 80% power: floor above 0.20"
            elif est < floor:
                verdict[lab] = f"{est:.4f} is below the floor of {floor}: an upper bound, not an estimate"
            else:
                verdict[lab] = f"{est:.4f} is at or above the floor of {floor}: read with the Holm-adjusted p"
        payload["verdict"] = verdict
    except FileNotFoundError:
        payload["verdict"] = (
            "no detection floor measured (run spell-power first): the estimates cannot be read"
        )
    payload["label"] = "exploratory (docs/spec_spells.md section 1)"


def cmd_spell_holm(args) -> None:
    from src.inference.spells import holm

    p, est = {}, {}
    for model in ("S", "G"):
        args.model = model
        fit = read_manifest(f"spell_fit_{_tag(args)}")["payload"]
        for lab, pv in fit["parametric_bootstrap"]["p_value"].items():
            p[f"eta_{lab}"] = pv
            est[f"eta_{lab}"] = fit["headline"][f"eta_{lab}"]
    out = {
        **holm(p),
        "estimates": est,
        "source": args.source,
        "exclude_one_club": bool(args.exclude_one_club),
    }
    args.model = "SG"
    write_manifest(f"spell_holm_{_tag(args)}", out)
    _show("Holm over the five primary tests", out)


def _show(title, payload) -> None:
    from src.cli.main import _print

    _print(title, payload)


def add_parsers(sub) -> None:
    for name, fn, helptext in (
        ("spell-recovery", cmd_spell_recovery, "spells S/G: plant and recover. A GATE."),
        ("spell-power", cmd_spell_power, "spells S/G: detection floor per parameter (alpha 0.01)"),
        ("spell-fit", cmd_spell_fit, "spells S/G: development fit (exploratory)"),
        ("spell-holm", cmd_spell_holm, "spells: Holm over eta_in, eta_out, eta_level, eta_trail, eta_lead"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--source", default="wyscout", choices=["wyscout", "statsbomb"])
        p.add_argument("--exclude-one-club", action="store_true")
        if name != "spell-holm":
            p.add_argument("--model", required=True, choices=["S", "G"])
            p.add_argument("--tau", type=float, default=5.0)
            p.add_argument("--seed", type=int, default=5694)
            p.add_argument("--workers", type=int, default=None)
        if name == "spell-recovery":
            p.add_argument("--replicates", type=int, default=10)
        if name == "spell-power":
            p.add_argument("--replicates", type=int, default=30)
            p.add_argument("--null-replicates", type=int, default=200)
        if name == "spell-fit":
            p.add_argument("--bootstrap", type=int, default=500)
            p.add_argument("--cluster-bootstrap", type=int, default=200)
            p.add_argument("--no-sensitivities", action="store_true")
        p.set_defaults(func=fn)
