# Spec: momentum in dangerous possessions

**Status: approved 2026-10-08, with the set-piece change in §2.** The possession table (§2) is
being built. It is ingest and fits nothing. No model in §3–§4 is fitted before
this spec is approved.

A follow-on analysis to the shot-based headline, which it does not revise. It
asks the same question with a denser event: is a team more likely to have a
dangerous possession shortly after it has had one, once game state is accounted
for?

## 1. Why

The shot headline is power-limited: ~12 shots per team-match, floor 0.02. The
feasibility pilot (`results/pilot_wyscout_possessions.json`) found ~42 dangerous
possessions per team-match on Wyscout, 3.5× the shot count. It also found a
**refractory gap**: same-team dangerous possessions within ~30 s run at 0–40%
of a renewal baseline, because the opponent has the ball in between. A model
that ignores possession reads that gap as negative excitation. §3 is built
around it.

Exploratory throughout. The idea of looking at possessions came after the shot
results, and there is no holdout yet (§7).

## 2. Event table (fixed now)

From the Wyscout event streams already on disk (1,941 matches, all development
seasons), `src/ingest/possessions.py`:

- **On-ball events:** Pass, Free Kick, Shot, Others on the ball. Duels, fouls,
  saves and offsides neither start nor end a possession.
- **Possession, tolerant rule.** A run of one team's on-ball events within a
  period. A single opponent event between two runs of the same team is a
  *touch* and does not end it, unless it is an accurate pass or a shot.
- **Own-possession time.** `[start of possession p, start of possession p+1)`. The
  last possession of a period runs to the period's end. So `P_H(t) + P_A(t) = 1`
  from each period's first on-ball event to its end.
- **Dangerous: final third.** An own event starting at `x ≥ 66.7`, or an accurate
  pass/free kick ending there. Event time `t_d` is the first such moment.
- **Set-piece flag** (amended 2026-10-08, approved). True if the possession
  has a set-piece restart (corner, free kick, throw-in or penalty; not a goal
  kick) at or before its first danger moment, or anywhere if it never becomes
  dangerous. "Start type" was dropped: under the tolerant rule a corner usually
  continues the attacking possession, so only 467 possessions started with one.
  The "at or before" keeps the flag from encoding the outcome: most corners are
  won by an attack that has already reached the final third.
- **Clock.** Offsets taken from the shot extraction, so possessions and shots
  share one clock. **Game state** comes from the shot slate's goal and card
  tables: goals strictly before `t`, dismissals at or before.

## 3. Model A, first: possession-sequence regression

Possessions `p = 1, 2, …` in a match, `D_p = 1` if possession `p` is dangerous:

```
logit P(D_p = 1) = θ₀ + θ_home + a_team + d_opp + f(minute) + g(score) + h(red)
                 + c · setpiece(p)
                 + ρ_self  · S_self(p)  + ρ_cross · S_cross(p)

S_self(p)  = Σ_{q<p, same team, D_q=1} e^{−(t_p − t_q)/τ}
S_cross(p) = Σ_{q<p, opponent,  D_q=1} e^{−(t_p − t_q)/τ}
```

- Background terms are the shot model's: team effects per (source, team, season)
  with ridge 1.0, a natural cubic spline in minute (df 6) at one-minute bin
  centres, and score and red bins clipped as for shots. The set-piece flag is the
  only addition. Nothing else is added, now or after a fit.
- `t_p` and `t_q` are possession **start** times, so simulated and real data use
  the same clock. The sums run over the whole match on the continuous clock, as
  the shot kernel does.
- `τ` is fixed at 5 min for the headline of this analysis. A τ profile over
  {2, 3, 5, 8, 14, 30} min is reported beside it.
- **Estimand:** `ρ_self`, the log-odds lift per recent dangerous possession,
  reported as an odds ratio `e^{ρ_self}`.
- The sequence is alternating by construction, so the refractory gap is handled
  by the unit of analysis.

## 4. Model B, only if A is run first: possession-aware Hawkes

```
λ_k(t) = P_k(t) · [ μ_k(t) + Σ_j Σ_{t_i^j < t} η_kj · β e^{−β(t − t_i^j)} ]
```

- `μ_k` is log-linear as in the shot model, piecewise constant on minute bins.
- The compensator integrates only over the team's own-possession time, so each
  past event contributes `η β ∫ e^{−β(s−t_i)} ds` over the later own intervals.
  That is closed form, O(events + switches).
- **Estimand:** `η_self`, read exactly as the shot `η` is.

## 5. Gates, for each model in turn

1. **Unit tests.**
   - A: on a constructed sequence, the `S` terms match a brute-force sum.
   - B: at `η = 0` the likelihood equals Poisson-on-own-time, and the recursion
     matches the O(n²) sum.
2. **Recovery.** Simulate along each match's observed possession path:
   - A: Bernoulli draws with planted `ρ_self = 0.3`.
   - B: thinning with planted `η = 0.3`.

   Refit everything, 10 replicates. Pass bands as for shots (±0.08 absolute).
   **Nothing downstream runs on a failure.**
3. **Detection floor.** Planted grid {0, 0.01, 0.02, 0.05, 0.10, 0.20}, 30
   replicates per cell, 200 for the null arm. The critical value comes from the
   pure null (no cross term planted).
4. **Development fit,** with a match-bootstrap CI (B = 200) and a parametric
   bootstrap p-value (B = 500).

## 6. Sensitivities, fixed now

- Strict possession rule.
- Box instead of final third.
- `x ≥ 75` for the final third.
- No set-piece term.
- No score state (expected to push the estimate down, as for shots).
- Spline df 3 and 9.

## 7. Caveats and how outcomes are read

- **Possession is conditioned on.** A team on top also has more of the ball.
  Conditioning on the possession path removes that part of momentum, just as
  conditioning on score removes momentum that became goals. This analysis
  measures momentum in *what a team does with the ball*, and it says so.
- **Not significant:** an upper bound on `e^{ρ_self}` (or on `η_self`) against
  its floor.
- **Significant, with τ profile flat or rising to 30 min:** slow swings, read
  with the hidden-state caveats (`docs/spec_hidden_state.md` §8), not as
  possession-to-possession momentum.
- **Holdout gap.** Wyscout is all 2016–18, so there is no holdout. Confirmation
  needs StatsBomb's full event files, which means a new download into a new
  cache, leaving the shot cache untouched. That is a separate decision.

## 8. Out of scope

- Expected threat (xT).
- Tracking data.
- Choosing the danger definition from fitted results.
- Any covariate added to the background after a fit.

## 9. Amendment 1, 2026-11-03: checks added after the Model A result

**Status: approved 2026-11-03 by the author.** Everything in this section was
designed **after** seeing the development fit (job 1372527). It is labelled
post hoc wherever it is reported. The Model A result in §3 stays the reported
headline of this analysis, exactly as fitted:

> `ρ_self = 0.194` (odds ratio 1.21), CI [0.180, 0.199], parametric p = 0.002,
> floor 0.02; `ρ_cross = −0.198`. The τ profile is best at the shortest τ (2 min,
> `ρ_self = 0.39`) and falls steadily to 30 min.

**Why amend.** The τ profile falls with τ, which §7 did not anticipate. A raw
descriptive (no model) shows two components:

- **Fast.** The lift in P(dangerous) after a dangerous own possession is +18 points
  when the team's previous possession started < 30 s earlier, and +12 at 30–60 s.
  Of those follow-ups, 32% start already in the final third: the ball was won back
  high.
- **Slow.** About +5 points persists at 2–5 min and beyond 5 min, without decaying.

Read as: ball-position carry-over (the possession analogue of rebounds) plus
match-level dominance. Neither is momentum. Model A, as specified, cannot tell
them apart.

### 9.1 Quick-regain merge (the possession analogue of the 4 s shot rule)

If team `k`'s possession `q` is followed by **one** opponent possession lasting
**< 10 s** (first opponent event to `k`'s next first event), and then `k`'s
possession `p` **starts in the final third** (its first own event is there), then
`q`, the opponent spell and `p` are **one attack** for `k`:

- merged start = `t_start(q)`
- dangerous if `q` or `p` was
- set-piece flag as §2, on the merged attack
- chains merge transitively

The threshold is fixed at 10 s now. 5 s and 20 s are sensitivities. Report the
fraction of possessions merged.

**Revision before any merged fit, 2026-11-03.** The first draft had no
"starts in the final third" condition. Its merge counts on Wyscout, computed
before any model was fitted on merged data, ruled it out:

| rule | 5 s | 10 s | 20 s |
| --- | --- | --- | --- |
| any short opponent spell | 16% of possessions removed | **42%** | 68% |
| + next possession starts in final third | 5.5% | **12%** | 18% |

The unconditioned rule collapses ordinary midfield turnovers, which is not
the mechanism in question. The conditioned rule targets the ball being won back
high, and at 10 s it removes about as much as the shot rebound rule's 15% guard.
It is the rule. The unconditioned counts are reported only as context.

Model A is then refitted unchanged on the merged table (same background, same
`τ = 5`, same τ profile, gates 2–4 rerun).

### 9.2 Within-half shuffle test (dominance vs order)

For each (match, half, team), permute the danger labels `D` across that team's
possessions. Possession times, sides and game-state covariates stay where they
are. Refit Model A on each shuffled table. B = 200, seeded.

- **Shuffled `ρ̂_self`** is what match-half-level dominance alone produces, because
  shuffling keeps each team's count of dangerous possessions per half.
- **The order component** = `ρ̂_self(observed) − mean ρ̂_self(shuffled)`, with a
  one-sided p-value from the shuffle distribution.
- Run on both the §3 table and the §9.1 merged table.
- Caveat: shuffling also scrambles `D` against minute and score within a half,
  which weakens those background terms slightly. That is reported, not corrected.

**Reading.**
- An order component that survives the merge (9.1) **and** the shuffle (9.2) is
  evidence of possession-to-possession dependence beyond ball position and
  dominance. It is still exploratory until §9.3.
- An order component that does not survive: Model A's `ρ` is ball position plus
  dominance, and the possession analysis reports that.

### 9.3 Replication on StatsBomb (approved 2026-11-03)

Wyscout has no holdout, so §9.1–§9.2 and Model A are repeated on an independent
source: StatsBomb full event files for the **2,112 development matches already in
the shot slate**. They are pulled into a new cache, `data/raw/statsbomb_events/`.
The shot cache is untouched, and no holdout match is downloaded.

Possession table, same rules as §2, mapped once and fixed now:

- **On-ball events:** Pass, Shot, Carry, Dribble, Clearance, Miscontrol.
- **Accurate pass:** a pass with no `outcome`.
- **Final third:** `x ≥ 80` on StatsBomb's 120 × 80 pitch.
- **Set pieces:** pass type Corner / Free Kick / Throw-in, shot type Free Kick /
  Penalty. Goal kicks excluded.
- **Clock and game state:** from the StatsBomb shot slate, as §2 does for Wyscout.
- StatsBomb's own `possession` id is a sensitivity, not the headline.

**Replicated** means: on StatsBomb, the same sign and parametric p < 0.05 for the
quantity in question (Model A `ρ_self` on the merged table, and the §9.2 order
component), each against its own floor. Reported with and without the one-club
(Barcelona) matches, which are an extreme of the dominance this amendment is
about.
