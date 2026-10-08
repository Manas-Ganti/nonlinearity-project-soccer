# soccer-momentum

Is soccer momentum real, or just game state? A bivariate Hawkes test on shot
arrivals, with a measured detection floor.

`PROPOSAL.md` is the framing, `docs/math.md` the derivations, `CLAUDE.md` the
project's rules. `docs/RESULTS.md` is generated from the runs.

## Quick start

```bash
make venv                 # .venv + deps
make check                # ruff + the fast test suite (~75 tests, no network)
make demo                 # the whole pipeline on synthetic data, a few minutes
```

Real data, in build order (`SOURCE` defaults to `pooled`):

```bash
make ingest-statsbomb ingest-wyscout    # StatsBomb from GitHub (~10 min); Wyscout is extracted from data/raw/wyscout
make build-pooled                       # statsbomb + wyscout -> the headline slate
make calibrate-dedup                    # step 2, per provider
make fit-null                           # step 3
make recovery                           # step 5 -- a gate
make power                              # step 6
make rounding-bias                      # step 7
make fit-dev                            # step 8
make report                             # -> docs/RESULTS.md
```

The Wyscout release is not fetched by a target: it is a one-off figshare download
(`data/raw/wyscout/`, digests in `manifest.sha256`). `docs/PROCESS.md` is the
plain-language walkthrough of the whole design.

`make ingest-understat` is the 6-19 hour overnight pull. It is resumable (one
parquet per league-season, `soccerdata` caches each match underneath) and it is
deliberately not part of any other target. **Never re-scrape**: the cache is
immutable input, and re-pulling mid-project silently changes the dataset under
results already computed.

## Layout

```
src/
  config.py         every tunable, in one place, echoed into each run manifest
  runlog.py         run manifests: inputs, digests, seeds, environment
  ingest/
    schema.py       the canonical event table + Slate (events, goals, cards)
    understat.py    soccerdata loader -> canonical events (minute clock)
    statsbomb.py    open-data JSON -> canonical events, at second resolution
    wyscout.py      Pappalardo open data -> canonical events, at second resolution
    pooled.py       statsbomb + wyscout stacked: the headline slate
    synthetic.py    a slate with known ground truth, for tests and the demo
    loader.py       split handling, and the holdout guard
  features/
    dedup.py        rebound/corner collapse, threshold calibrated on StatsBomb
    state.py        score and man-advantage reconstruction
    jitter.py       order-preserving uniform jitter inside the recorded minute
  models/
    spline.py       natural cubic basis
    baseline.py     inhomogeneous Poisson mu_k(t)
    hawkes.py       O(n) recursion, likelihood, compensator
    joint.py        joint MLE over (mu, eta, beta)
    simulate.py     Ogata thinning, vectorised across the whole slate
  inference/
    pipeline.py     the full estimation pipeline as one callable
    bootstrap.py    parametric null, cluster bootstrap
    gof.py          time rescaling, KS, Q-Q
    power.py        recovery test and detection floor
  cli/main.py       one subcommand per build step
```

## The result that should shape the plan

`docs/RESULTS.md` has everything; this is the part that changes decisions.

**Understat's minute clock puts the detection floor at eta_self = 0.20.** The same
380 fixtures at StatsBomb's second resolution give a floor of 0.05. That is not a
4x difference in precision, it is a qualitative barrier:

| planted eta_self | recovered, second clock | recovered, minute clock |
| --- | --- | --- |
| 0.02 | 0.013 (53% power) | 0.000 -- no power |
| 0.05 | 0.038 (100% power) | 0.000 -- no power |
| 0.10 | 0.090 (100% power) | 0.000 -- no power |
| 0.15 | 0.138 (100% power) | 0.030 (70% power) |
| 0.20 | 0.192 (100% power) | 0.095 (100% power) |
| 0.30 | 0.293 (100% power) | 0.210 (100% power) |

Below a planted 0.15 the minute-clock estimator does not return a small noisy
number, it returns **numerically zero** -- the largest `eta_hat` across all 90
replicates at eta* <= 0.10 is 1.6e-4. The mechanism is visible in the same sweep:
rounding forces the de-duplication rule to collapse every same-minute pair, and the
fraction of events it removes climbs with the planted effect (6.4% at eta* = 0,
11.7% at eta* = 0.30), because more clustering means more same-minute collisions.
The rounding eats the signal preferentially.

**More matches will not fix this.** It is not a variance problem that `sqrt(n)`
shrinks: the likelihood has no gradient in `eta` at zero once the same-minute pairs
are gone, so 20,000 matches land on the same floor as 380. If the effect is where
the second-resolution data suggests (0.04 to 0.10), an Understat-only study is
designed to return 0.00 regardless of how much of it is collected.

Two independent routes agree on this. Step 7 takes StatsBomb's own events, rounds
them, and refits: eta_self moves 0.095 -> 0.000. And the two providers, on 367
shared fixtures, agree on the events (shot counts correlate 0.985, 0.67 shots per
match apart, total xG within 1.9%) while disagreeing completely on eta_self
(0.092 vs 0.000). They are looking at the same football through different clocks.

## What has been run, and what has not

Numbers live in `docs/RESULTS.md`, regenerated by `make report` from the manifests
in `results/`. Every per-dataset artefact is scoped by source and split
(`power_statsbomb_dev.json`, not `power.json`) because a detection floor measured
on one slate says nothing about another, and an unscoped filename lets one run
silently answer for a different one.

Run, on real data:

- **StatsBomb open data**, all 472 matches pulled; the development split is the
  Premier League 2015/16 season, 380 matches at second resolution.
- **Understat**, Premier League 2015/16 pulled and validated end to end (380
  matches, 9,781 shots). This was a loader-validation pull, not the full dataset.
- Build steps 1-8 on both, plus the estimator comparison, the kernel-timescale
  profile, and an Understat-vs-StatsBomb comparison on the fixtures both cover.

Not run, deliberately:

- **The full Understat pull.** `make ingest-understat` covers five leagues and
  every development season, and at the rate the validation pull ran it is roughly
  five to six hours. It is one command, it is resumable, and nothing else needs to
  change to use it.
- **Step 9, the holdout.** It is run once, at the end, after the development
  analysis is settled. It is not settled, the holdout seasons are not pulled, and
  running it now would spend the one look this project gets.
- **Red cards for Understat.** See below.

## Three places this departs from the specs, and why

**1. The estimator.** `docs/math.md` section 4 offers a two-stage fit -- fit `mu`
under the Poisson null, then hold it fixed -- and predicts it is biased *towards*
finding excitation. Measured on planted data, it is biased hard the *other* way:
a planted `eta_self = 0.30` comes back as 0.10. Fitting a Poisson background to
data that contains excitation inflates `mu` by roughly `1/(1 - rho)`, because
every offspring event has to be explained as background; freezing that inflated
`mu` then leaves nothing for `eta`. The default is a joint MLE over `mu`, `eta`
and `beta` (bias -0.012 at the same planted value). EM reaches the same fixed
point and is kept for comparison. See `results/estimator_comparison.csv` and
`docs/math.md` section 10.

**2. The kernel is single-component.** `PROPOSAL.md` section D specifies a
two-timescale kernel with a fast component absorbing rebounds. `CLAUDE.md`
overrides this: at Understat's minute resolution a 10-20 second kernel is not
identifiable, so the fast structure is removed in preprocessing instead. The
measurement backs the override -- the mechanical excess is confined to gaps under
4 seconds, which integer minutes cannot see. Where the two documents conflict,
`CLAUDE.md` governs method.

**3. The power study calibrates its null once, not 210 times.** `docs/math.md`
section 8 runs a full parametric bootstrap inside every replicate: 7 eta values x
30 replicates x 500 draws is ~105,000 pipeline runs. Instead the `eta* = 0` arm is
run at 200 replicates and its 95th percentile is the critical value for every arm.
Same Monte Carlo test, same level, ~1/500th the compute. The per-dataset bootstrap
still runs in full for the headline fit, where it is affordable and where it
matters.

## Two things the data cannot currently supply

**Red cards are not in Understat's shot feed.** `soccerdata`'s Understat reader
returns shots only, and dismissal times are not recoverable from them. The event
schema carries a `red_cards_known` flag, `state.py` sets the man-advantage
covariate to zero when it is False, and every fit records which case it was in.
StatsBomb has dismissals and uses them. Wiring FBref's match events in as a card
source for Understat is the obvious extension; `CLAUDE.md` marks FBref as an
extension that must never block a build step, so it does not.

**`soccerdata` 1.9.1 drops the penalty label.** Understat's `situation`
vocabulary includes `Penalty`, and the reader's translation table omits it, so
every penalty arrives as `<NA>`. Penalties are load-bearing here -- they excite
nothing and are excluded entirely -- so `ingest/understat.py` patches the table
before reading and cross-checks against Understat's constant penalty xG
(~0.7608). Without that, ~1% of shots would silently become open play at the
highest xG in the dataset.

## The discipline that is enforced in code, not just documented

- **Holdout.** `ingest/loader.py` refuses to return seasons from 2019-20 onward
  unless `--final-run` is passed, and appends every request -- granted or refused
  -- to `logs/holdout_access.log`.
- **De-duplication.** `apply_dedup` raises if the rule removes more than 15% of
  shots, rather than quietly eating real chances.
- **The recovery test is a gate.** `make recovery` exits non-zero if a planted
  `eta = 0.3` does not come back, because nothing downstream means anything if it
  does not.
- **Stationarity.** `eta_self + eta_cross < 1` holds by construction of the
  parameterisation, not by a penalty that an optimiser can trade away. There is a
  test that hits it with 500 extreme parameter vectors.
- **The recursion.** The O(n) likelihood is tested against the naive O(n^2) double
  sum on five parameter settings plus an asymmetric branching matrix, to 1e-8.
- **Reproducibility.** Every step writes a manifest to `results/` with its inputs'
  digests, its seeds and the environment.

## Reading a result

Never report `eta_self` alone. The pipeline prints it next to the detection floor
and, when `tau` lands on the edge of its box, says so -- a pinned timescale means
the optimiser is reaching for structure the data cannot resolve, and the point
estimate should not be read as though the data chose it.

A branching ratio below the floor is an upper bound, not a finding. That is a
designed outcome, not a failed project.
