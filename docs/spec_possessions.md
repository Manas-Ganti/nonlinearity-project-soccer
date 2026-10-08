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
