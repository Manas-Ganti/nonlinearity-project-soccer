# Spec: shots inside final-third spells, and by game state

**Status: approved 2026-11-03 by the author, with the quick-regain merge (§2) and
StatsBomb confirmation (§9).** Nothing in §4–§6 is fitted before its gates pass.

A follow-on analysis. It does not revise the shot headline (`η_self = 3e-9`,
floor 0.02, job 1048269) or replace `docs/spec_possessions.md` Model A
(job 1372527), which runs first and independently.

## 1. Why

Feedback on the progress presentation: look for momentum in smaller, well-defined
situations rather than across all shots at once. Two situations are tested:

1. **Spells.** Does a shot cause further shots when the team is already pinning
   the opponent back, even though shots cause no further shots on average?
2. **Game state.** Does the answer differ when the shooting team is level,
   trailing or leading?

One development observation points at (1): on Wyscout alone, widening the
kernel box to 1–20 min gives `η_self = 0.058` at `τ ≈ 20` min (+14
log-likelihood), while StatsBomb alone gives `η ≈ 0` (job 1051636).

**This analysis was designed after seeing development results, and it runs
on the data that produced the hint.** It is exploratory throughout and is
labelled so wherever it is reported (§9).

## 2. Data (fixed now)

Wyscout, 1,941 matches, 2016–18, development seasons only. No holdout exists for
this source.

- **Shots:** `data/interim/wyscout`, prepared exactly as for the headline: penalties
  excluded, same-team shots within 4 s merged. 43,544 shots after preparation.
- **Possessions:** `data/interim/wyscout_possessions/possessions.parquet`, tolerant
  rule, danger = final third (`t_ft`), as in `docs/spec_possessions.md` §2, **then
  quick-regain merged** (spec_possessions §9.1: an opponent possession < 10 s between
  two of a team's possessions, the second starting in the final third, joins them
  into one attack). Without the merge, a
  ball won straight back high up the pitch would count as two dangerous
  possessions and push a team into a "spell" on ball position alone.
- **Game state:** score and red-card differential from the shot slate's goal and card
  tables, as everywhere else.
- Possessions and shots share one clock (spec_possessions §2).

## 3. The spell indicator (fixed now)

For team `k` at time `t`:

```
C_k(t) = number of dangerous possessions among team k's last 5 possessions
         that started before t's possession, within the current period
Z_k(t) = 1  if  C_k(t) ≥ 4,  else 0
```

- **Past possessions only.** The possession containing `t` is never counted, so
  `Z` is predictable: it depends only on the history before `t`. A shot cannot put
  its own team into a spell. This is what stops the test being circular.
- **Resets at each half.** Fewer than 4 earlier possessions in the half means
  `Z = 0`.
- `Z` changes only at possession boundaries, so it is piecewise constant.

**How 4-of-5 was chosen.** On 2026-11-03, from how often the rule fires
only. No spell model had been fitted, and no momentum quantity was looked at:

| rule | share of match time with `Z = 1` | share of shots with `Z = 1` |
| --- | --- | --- |
| 3 of 5 | 0.59 | 0.63 |
| **4 of 5** | **0.32** | **0.35** |
| 5 of 5 | 0.10 | 0.11 |

At 57% of possessions dangerous, 3-of-5 is the normal state of play, not a
spell. 4-of-5 marks about a third of the match.

**Re-check after the merge (mechanical, fixed now).** The table above is on
unmerged possessions. After the §2 merge, recompute the share of match time with
`Z = 1`. Keep 4-of-5 if it lies in [0.20, 0.40]. Otherwise use the `k`-of-5 whose
share is closest to one third. This uses prevalence only, before any fit, and the
choice is recorded with the build. The threshold is not revisited after any fit.
Neighbouring `k`, and 6-of-8, are sensitivities (§7).

**Re-check result, 2026-11-03** (prevalence only, no fit). On the 10 s merged tables,
4-of-5 gives `Z = 1` for **0.34** of match time on Wyscout (0.36 of shots) and **0.22**
on StatsBomb (0.24 of shots). Both are inside [0.20, 0.40], so **4-of-5 stands on
both sources**. StatsBomb has fewer dangerous possessions (0.49 against 0.60 after
the merge), which is why its share is lower.

**Measurement note, 2026-11-04 (prevalence only, before any Model S or G fit).**
The re-check above measured `Z` over each team's *own* possession time. Measured
over all match time, as `Z_k(t)` is defined (the opponent's possessions included,
and time before a half's first possessions counted as `Z = 0`), the shares are
**0.32** (Wyscout) and **0.19** (StatsBomb); shot shares are unchanged at 0.36 and
0.24. StatsBomb's 0.19 sits just under the 0.20 edge of the band. 4-of-5 is kept on
both sources: the recorded re-check passed, the shot share is inside the band, and
one rule on both sources is what makes section 9 a confirmation of the same model.
3-of-5 (0.44 of StatsBomb match time) runs as a sensitivity (section 7).
`src/features/spells.py` implements the all-match-time definition.

## 4. Model S: spell-modulated Hawkes on shots

```
log μ_k(t) = [headline background terms] + γ_own · Z_k(t) + γ_opp · Z_{−k}(t)

λ_k(t) = μ_k(t)
       + Σ_{i: k_i = k}  η_{Z_k(t_i)} · β e^{−β(t − t_i)}      own shots, split by spell at the shot
       + Σ_{i: k_i ≠ k}  η_cross      · β e^{−β(t − t_i)}      opponent shots, not split
```

- **Estimands:** `η_in` (own-team excitation from shots taken while `Z = 1`) and
  `η_out` (while `Z = 0`). The contrast `η_in − η_out` is reported beside them.
- **`γ` must be in the background.** Teams in a spell shoot more just by being in
  one. Without `γ_own`, that level effect would be counted as shots causing
  shots inside spells, which is the same trap as omitting score state. Fitting
  without `γ` is a sensitivity, labelled as confounded.
- Single exponential kernel, `1/β ∈ [3, 8]` min, as in the headline. The 1–20 min box
  and a τ profile over {3, 5, 8, 14, 20, 30} min are reported beside it, because the
  motivating hint is slow.
- Stationarity: `max(η_in, η_out) + η_cross < 1`.
- Joint MLE of the background and the kernel, as in the headline (`src/models/joint.py`),
  not two-stage.
- O(n) recursion with one running sum per (source team, `Z` at the shot), four sums
  in place of two.

## 5. Model G: game-state split

The headline Hawkes on Wyscout shots, background unchanged, with the own-team
kernel split by the shooting team's score state at the shot:

```
η_level (D = 0),   η_trail (D < 0),   η_lead (D > 0)
```

`g_score` stays in the background, so each `η` is excitation beyond the
score-state shift in rate. `η_cross` is not split.

## 6. Inference and gates, in order

1. **Unit tests.**
   - With `η_in = η_out` (or all three game-state `η` equal), the likelihood equals
     the existing Hawkes likelihood exactly.
   - `Z(t)` uses only possessions before `t`'s possession.
   - The recursion matches the O(n²) sum.
2. **Recovery (gate).** 10 replicates, full pipeline, `μ` re-estimated, pass band
   ±0.08:
   - Model S: plant `η_in = 0.30`, `η_out = 0.10`.
   - Model G: plant `(0.30, 0.10, 0.20)`.

   **Nothing downstream runs on a failure.**
3. **Simulation.** Ogata thinning with the split kernel. Possession paths, and so
   `Z`, are held at the observed path, as game state already is (math.md §10).
   **Consequence, stated in the write-up:** a simulated shot cannot change later
   `Z`. So the power study does not measure how much excitation the `γ` term
   absorbs through that route.
4. **Multiple testing.** Five primary tests: `η_in`, `η_out`, `η_level`, `η_trail`,
   `η_lead`. Holm correction at family-wise α = 0.05. Detection floors are computed
   at α = 0.01 (= 0.05 / 5), so a floor is valid whatever Holm step a test lands on.
5. **Detection floor, per parameter.** Plant the target `η ∈ {0, 0.02, 0.05, 0.10, 0.20}`
   with the others at 0. 30 replicates per cell, 200 in the null arm. The critical
   value comes from the pure null. Floor = smallest planted `η` at 80% power.
   Expect floors above the headline's 0.02: each subgroup has fewer shots.
6. **Development fit.** Match-level cluster bootstrap CI (B = 200), parametric
   bootstrap p-value under the fitted Poisson null with `γ` (B = 500).

## 7. Sensitivities, fixed now

- Spell rule 3-of-5, 5-of-5, 6-of-8.
- Danger at `x ≥ 75` (`t_ft75`) instead of the final third.
- No `γ` terms (expected to inflate `η_in`; labelled confounded).
- Kernel box 1–20 min, and the τ profile.
- **Minute spline df 9 and 12.** On both providers, df 9 and 12 drove `η_self` to
  exactly 0 with τ at the box edge. If a spell effect vanishes under the
  flexible spline, it is read as background misfit, not momentum.
- Strict possession rule.

## 8. Caveats and how outcomes are read

- **Conditioning on `Z` removes part of what "being on top" means.** `η_in`
  measures whether a shot adds further shots beyond the elevated spell rate. It
  does not measure whether spells exist. That question belongs to Model A and
  `docs/spec_hidden_state.md`.
- **Background misfit.** The Poisson null fails time-rescaling KS on both providers.
  A slow unmodelled drift in rate looks like a slow spell effect. §7's spline
  sensitivity is the check, and if the two cannot be told apart, the result says so.
- **Not significant:** upper bounds on `η_in` and each game-state `η`, against their
  own floors.
- **Significant `η_in`, null `η_out`:** "momentum appears only inside sustained
  pressure". That is exploratory on Wyscout and unconfirmed until §9.

## 9. Confirmation on StatsBomb (approved 2026-11-03)

Wyscout has no holdout. Models S and G are refitted, unchanged, on StatsBomb's
2,112 development matches:

- full event files in a new cache, `data/raw/statsbomb_events/`
- possessions built and merged as in spec_possessions §9.3
- the spell rule as fixed in §3, re-checked by the same prevalence rule

**Confirmed** means the same sign and Holm-adjusted p < 0.05 on StatsBomb, against
StatsBomb's own floors. Reported with and without the one-club (Barcelona) matches.
This is decided before Model S is fitted on either source, so it does not depend
on the Wyscout result.

## 10. Out of scope

- Choosing the spell threshold, window or danger definition from fitted results.
- Subgroups beyond the five in §6.4, added after any fit.
- Any background covariate beyond `γ_own`, `γ_opp`, added after a fit.
- Goals as the event type.
- Expected threat (xT), tracking data.
