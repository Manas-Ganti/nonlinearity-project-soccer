# soccer-momentum

Testing whether chance creation in football is self-exciting once game state is
accounted for. Read `PROPOSAL.md` for the framing and `docs/math.md` for full
derivations before writing model code.

## The one-sentence version

Fit an inhomogeneous Poisson null and a bivariate Hawkes alternative to
shot-arrival times, and report the branching ratio against a detection floor
measured by simulation.

## Core objects

- **Event**: a shot, at time `t` in minutes from kickoff, with mark `x` = xG,
  belonging to team `k ∈ {H, A}`, in situation `s ∈ {open, setpiece, penalty}`.
- **Match**: two point processes on `[0, T]`, plus a game-state trajectory
  (score differential, red-card differential) that is piecewise constant.
- **Background intensity** `μ_k(t)`: the null model. Carries all domain
  knowledge.
- **Branching ratio** `η`: the estimand. Expected fraction of chances triggered
  by earlier chances.

## Model specification

Bivariate Hawkes, single exponential kernel, fitted on de-duplicated events:

```
λ_k(t) = μ_k(t) + Σ_j Σ_{t_i^j < t} η_kj · β · e^(−β(t−t_i^j))
```

with `1/β` on the order of 3–8 minutes. The second, faster component that would
absorb rebounds and corner chains is deliberately absent: it is not identifiable
at Understat's resolution, so that structure is removed in preprocessing instead.
See "Data and resolution".

Kernel normalised so `η` **is** the branching ratio directly (`∫φ = η`). Do not
use the `α, β` parameterisation where the branching ratio is `α/β`; it makes the
stationarity constraint harder to enforce and the results harder to read.

Background, piecewise constant on one-minute bins:

```
log μ_k(t) = θ_0 + θ_home·1[k=H] + a_team(k) + d_opp(k)
           + f_minute(t) + g_score(D_k(t)) + h_red(R_k(t))
```

Symmetry reduction for the branching matrix: `η_self := η_HH = η_AA`,
`η_cross := η_HA = η_AH`. Stationarity requires `η_self + η_cross < 1`.
Relax the symmetry only as a robustness check, never in the headline fit.

**Report `η_self` as the momentum estimate.** Nothing else, and never without
the detection floor beside it.

## Non-negotiable domain constraints

These come from the sport, not from statistics, and getting them wrong produces
a confident wrong answer rather than a failure.

1. **Mechanical clustering must be removed before fitting, not modelled.**
   Rebounds and corner sequences cluster on a seconds timescale and have nothing
   to do with momentum. Understat's minute resolution cannot separate them from
   genuine excitation (see "Data and resolution" below), so a fast kernel
   component is **not identifiable and must not be fitted**. Handle them by
   de-duplication in preprocessing, then fit a single slow kernel with
   `1/β_s ≈ 3–8 min`. A single-component kernel fitted on raw data will lock onto
   mechanical structure and report a large branching ratio that means nothing.
2. **Penalties do not excite.** A penalty arrives from a foul, carries xG ≈ 0.76,
   and has no build-up. Exclude penalty shots from the excitation term. Whether
   they can *be* excited is a separate question; default to excluding them
   entirely.
3. **Score state is the dominant confound and it points the wrong way.** Teams
   that score drop deeper and create less. Omit `g_score` and the fit will report
   *negative* momentum, which looks like a finding and is a substitution pattern.
   `g_score` goes in before anything else is examined.
4. **Red cards are a step change, not a covariate to smooth.** Piecewise
   constant, applied from the minute of dismissal.

## Data and resolution

Do not write scrapers. Both sources have maintained packages.

**Primary — Understat via `soccerdata`** (`pip install soccerdata`, v1.9.1+).
Big 5 leagues plus RFPL, 2014/15 to present, roughly 20,000 matches. Caches to
`~/soccerdata/data/`.

```python
import soccerdata as sd
shots = sd.Understat('ENG-Premier League', '2024').read_shot_events()
```

One request per match, so ~23,000 requests at a polite 1–3s delay: 6–19 hours.
Run it once, overnight. Never re-scrape; the cache is the source of truth and
should be treated as immutable input.

**Resolution source — StatsBomb open data** (`statsbombpy`, or JSON directly from
the `statsbomb/open-data` GitHub repo). Free, no key, second-level timestamps and
its own xG. Far fewer matches, and volume is not what it is for.

**The resolution constraint, stated precisely.** Understat records integer
minutes. With ~25 shots across ~95 minute-slots, expected ties are
`C(25,2)/95 ≈ 3` pairs per match, and rebound pairs collide every time. A 10–20
second kernel is therefore unidentifiable on Understat. This is settled, not a
question to re-open.

StatsBomb exists in this project to do two specific jobs:

1. **Calibrate the de-duplication rule.** At second resolution, measure the
   empirical distribution of shot-to-shot gaps within a team and locate where
   mechanical follow-ups end and independent attacks begin. That threshold
   becomes the collapse rule applied to Understat.
2. **Quantify rounding bias.** Fit `η` on StatsBomb at true resolution, round the
   same events to integer minutes, refit, and report the difference. This
   converts the data limitation into a measured sensitivity rather than an
   unexamined assumption.

**Ties on Understat.** After de-duplication, remaining same-minute events need
uniform jitter within the minute to make the likelihood well defined. Seed it,
and report `η` across several jitter draws so the estimate does not depend on one
realisation.

**Cross-check — FBref via `soccerdata`.** Independent xG model. Requires a
headless Chrome path and proxy rotation for failed requests, so it is materially
harder than Understat. Extension only; never let it block a build step.

## Statistical discipline

- **Held-out split.** Develop on seasons 2014–15 to 2018–19. The 2019–20+ data is
  touched exactly once, at the end. Enforce this in code: the loader refuses to
  return holdout data unless an explicit `--final-run` flag is passed, and it
  logs every time it is.
- **No LRT for η = 0.** It sits on the parameter-space boundary, so the standard
  χ² asymptotics do not apply. Use the parametric bootstrap in
  `src/inference/bootstrap.py`.
- **Re-estimate μ inside every simulation.** Using the true background in the
  power study overstates power, badly. The whole point is to measure how much
  background estimation absorbs.
- **Goodness of fit before model comparison.** Time-rescaling KS on both models.
  If neither fits, the comparison between them is not meaningful and that is
  itself the result.

## Build order

Do not reorder this. Each step gates the next.

1. **Ingest.** Pull Understat via `soccerdata` (overnight) and a StatsBomb open
   sample. Build the event table and reconstruct score and card state at every
   event time. Write reconstruction tests against known scorelines.
2. **Calibrate de-duplication on StatsBomb.** Second-resolution gap distribution,
   pick the collapse threshold, apply it to Understat. Report how many events
   the rule removes; if it is removing more than ~15% of shots, the threshold is
   too aggressive and is eating real chances.
3. **Null model.** Fit `μ_k(t)` on de-duplicated events. Check with
   time-rescaling KS. This is most of the work and most of the value; a good null
   is what makes the rest credible.
4. **Simulator.** Ogata thinning from the fitted background with a settable `η`.
5. **Recovery test.** Plant `η = 0.3`, run the full pipeline, confirm it is
   recovered. **If this fails, everything downstream is meaningless.** Do not
   proceed past this step until it passes.
6. **Detection floor.** Sweep `η ∈ {0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30}`,
   30+ replicates each, full pipeline per replicate. Floor = smallest η at 80%
   power.
7. **Rounding-bias experiment.** On StatsBomb matches, fit `η` at second
   resolution, round the same events to integer minutes, refit, report the
   difference. This is a headline methods result, not a footnote.
8. **Fit development data.** Report `η̂_self`, `η̂_cross`, bootstrap p-value,
   against the floor.
9. **Holdout, once.** Report and stop.

## Repo layout

```
src/
  ingest/       soccerdata + statsbombpy loaders, event table construction
  features/
    dedup.py    rebound/corner collapse, threshold calibrated on StatsBomb
    state.py    game state reconstruction, covariate build
  models/
    baseline.py inhomogeneous Poisson, μ(t)
    hawkes.py   bivariate Hawkes: intensity, likelihood, fit
    simulate.py Ogata thinning
  inference/
    bootstrap.py parametric bootstrap null
    gof.py       time rescaling, KS, Q-Q
    power.py     detection floor sweep
data/raw|interim|processed/
docs/math.md
tests/
```

## Implementation notes

- Likelihood must use the **O(n) recursion** for exponential kernels (see
  `docs/math.md`). The naive O(n²) double sum will make the power study
  impossible at 20,000 matches × 7 η values × 30 replicates.
- Optimise in unconstrained space: `η = sigmoid(z)` scaled to respect the
  spectral radius bound, `β = exp(w)`. Report on the natural scale.
- Fit `μ` once per bootstrap replicate, not once per iteration of the Hawkes
  optimiser. Profile likelihood or a two-stage fit; document which.
- Set seeds everywhere and log them. The power study is worthless if it is not
  reproducible.
- `float64` throughout. The compensator involves differences of large sums.

## Do not

- Do not report a branching ratio without the detection floor next to it. The
  number alone is uninterpretable.
- Do not add covariates to `μ` after looking at `η̂`. That is fitting the null
  until the alternative disappears, and it is undetectable in a writeup.
- Do not switch to goals as the event type. The power calculation does not work
  and the project fails.
- Do not fit a fast kernel component on Understat. It is unidentifiable at
  minute resolution, and an optimiser will happily return a confident number for
  it anyway.
- Do not re-scrape. The cache is immutable input; re-pulling mid-project silently
  changes the dataset underneath results already computed.
- Do not touch holdout data outside step 9.
- Do not treat a null result as a failed project. A measured upper bound is the
  designed outcome, and forcing a positive finding out of the data is the one
  way to make the work worthless.
