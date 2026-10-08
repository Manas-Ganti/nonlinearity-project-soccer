# Mathematics

Everything needed to implement the models. Notation: time `t` in minutes from
kickoff, match on `[0, T]`, teams `k, j ∈ {H, A}`.

---

## 1. Point process basics

A simple point process is characterised by its **conditional intensity**

```
λ(t | H_t) = lim_{dt→0} (1/dt) · P(event in [t, t+dt) | H_t)
```

where `H_t` is the history up to `t`. Everything below is a choice of `λ`.

The **compensator** is `Λ(t) = ∫₀ᵗ λ(s) ds`. The log-likelihood of observing
events at `t_1 < … < t_n` on `[0, T]` is

```
ℓ = Σᵢ log λ(tᵢ) − ∫₀ᵀ λ(s) ds
```

The first term rewards high intensity where events occurred; the second
penalises high intensity everywhere else. Both terms are needed — maximising
only the first drives `λ → ∞`.

---

## 2. Null model: inhomogeneous Poisson

```
λ_k(t) = μ_k(t),     log μ_k(t) = θ₀ + θ_home·1[k=H] + a_{team(k)} + d_{opp(k)}
                                 + f(m(t)) + g(D_k(t)) + h(R_k(t))
```

- `m(t)` = match minute; `f` a natural cubic spline (5–7 df) or 5-minute bins
- `D_k(t)` = score differential from `k`'s perspective; `g` binned at
  `{≤−2, −1, 0, +1, ≥+2}`
- `R_k(t)` = red-card differential; `h` binned at `{−1, 0, +1}` (multi-card cases
  are rare; clip)
- `a`, `d` = team attack / opponent defence effects, fitted per season

**Make `μ` piecewise constant on one-minute bins.** Then the compensator is a
finite sum,

```
∫₀ᵀ μ_k(s) ds = Σ_{b=1}^{B} μ_k,b · Δ_b
```

with `Δ_b` the bin width (1 minute, except the final partial bin). No numerical
integration anywhere in the codebase.

Since `D_k` and `R_k` change only at goals and cards, and `f` is evaluated at bin
centres, `μ_k,b` is exactly constant within bins by construction.

---

## 3. Alternative: bivariate Hawkes

```
λ_k(t) = μ_k(t) + Σ_j Σ_{tᵢ^j < t} φ_kj(t − tᵢ^j)
```

with the two-component exponential kernel

```
φ_kj(s) = η^f_kj · β_f · e^(−β_f s)  +  η^s_kj · β_s · e^(−β_s s)
```

### Why this normalisation

```
∫₀^∞ β e^(−βs) ds = 1    ⟹    ∫₀^∞ φ_kj(s) ds = η^f_kj + η^s_kj
```

So `η` **is** the expected number of direct offspring, with no algebra needed.
The alternative parameterisation `φ(s) = α e^(−βs)` gives branching ratio `α/β`,
which couples the two parameters in the stationarity constraint and makes the
optimiser's job harder. Use the normalised form.

### Branching matrix and stability

```
N = [[η_HH, η_HA],
     [η_AH, η_AA]]     (summing fast and slow components)
```

`N_kj` is the expected number of team-`k` chances directly triggered by one
team-`j` chance. The process is stationary iff the spectral radius `ρ(N) < 1`.

Under the symmetry reduction `η_HH = η_AA = η_self`, `η_HA = η_AH = η_cross`:

```
N = [[η_self, η_cross],
     [η_cross, η_self]]        eigenvalues: η_self ± η_cross
     ρ(N) = η_self + η_cross
```

so the constraint is simply `η_self + η_cross < 1`. This is a genuine
bifurcation: as `ρ → 1⁻` the expected cluster size `1/(1−ρ)` diverges.

### Interpretation of the two off-diagonals

- `η_self`: my chances beget my chances. **This is momentum.**
- `η_cross`: my chances beget the opponent's. This is the open, end-to-end game
  — counter-attacks, a match stretching. Physically real, and it must be in the
  model or it leaks into `η_self`.

---

## 4. Log-likelihood

For one match:

```
ℓ = Σ_k [ Σᵢ log λ_k(tᵢ^k) − ∫₀ᵀ λ_k(s) ds ]
```

### Compensator in closed form

```
∫₀ᵀ λ_k(s) ds = ∫₀ᵀ μ_k(s) ds
              + Σ_j Σ_{tᵢ^j < T} [ η^f_kj (1 − e^(−β_f(T−tᵢ^j)))
                                 + η^s_kj (1 − e^(−β_s(T−tᵢ^j))) ]
```

Derivation: `∫₀^{T−tᵢ} ηβe^(−βs) ds = η(1 − e^(−β(T−tᵢ)))`. The `(1 − e^…)` factor
is the fraction of an event's offspring expected to arrive before the final
whistle — edge correction, and it matters because matches are short relative to
nothing but still finite.

### O(n) recursion for the intensity

Evaluating `λ_k(tᵢ^k)` naively is O(n²). For exponential kernels there is an
exact recursion (Ogata 1981). For each ordered pair `(k, j)` and each kernel
component with decay `β`, define

```
R_kj(i) = Σ_{t_m^j < tᵢ^k} e^(−β(tᵢ^k − t_m^j))
```

Then

```
R_kj(i) = e^(−β(tᵢ^k − t_{i−1}^k)) · R_kj(i−1)
        + Σ_{t_m^j ∈ [t_{i−1}^k, tᵢ^k)} e^(−β(tᵢ^k − t_m^j))
```

with `R_kj(0) = 0` and `t_0^k = 0`. Each event of team `j` is visited once, so
the whole pass is O(n_H + n_A). Then

```
λ_k(tᵢ^k) = μ_k(tᵢ^k) + Σ_j [ η^f_kj β_f R^f_kj(i) + η^s_kj β_s R^s_kj(i) ]
```

For the self term `k = j`, the second sum on the right of the recursion is
empty (no team-`k` events strictly between consecutive team-`k` events), so it
collapses to `R_kk(i) = e^(−βΔ)(1 + R_kk(i−1))`.

**Implement and unit-test the recursion against the naive O(n²) sum on small
inputs.** This is the single most likely place for a silent bug.

### Optimisation

Reparameterise to remove constraints:

```
β = exp(w)
(η_self, η_cross) = ρ_max · softmax-like map of (z₁, z₂) with ρ < 1
```

A workable choice: `ρ = sigmoid(z₀) · 0.99`, `p = sigmoid(z₁)`,
`η_self = ρ·p`, `η_cross = ρ·(1−p)`. Report on the natural scale with the
delta-method or bootstrap standard errors.

Fit `μ` and the Hawkes parameters in two stages (fit `μ` under the Poisson null,
then hold it fixed while fitting `η`), or by profile likelihood. Two-stage is
biased toward finding excitation, since `μ` was fitted without excitation in the
model. **Document which is used and check the direction of the bias in the
recovery test.**

---

## 5. Simulation: Ogata thinning

To generate from `λ(t)` on `[0, T]`:

```
t ← 0
while t < T:
    λ̄ ← upper bound on λ(s) for s ≥ t          # see below
    draw u ~ Exp(λ̄); t ← t + u
    if t ≥ T: stop
    draw v ~ Uniform(0,1)
    if v ≤ λ(t)/λ̄: accept t as an event; update history
    # else reject and continue
```

**Upper bound.** Immediately after any event, `λ` is at a local maximum and
decays until the next event. So `λ̄ = max_b μ_b + (current excitation at t)` is
valid and tight enough. Recompute it after every accepted event.

For the bivariate case, either thin each team's process with its own bound, or
thin the superposition `λ_H + λ_A` and assign an accepted event to team `k` with
probability `λ_k/(λ_H + λ_A)`. The superposition version is simpler and less
error-prone.

**Validation.** Simulate with `η = 0` and confirm the event count matches
`∫μ` within Monte Carlo error. Simulate with known `η` and confirm the empirical
branching ratio (from a cluster-size estimate) matches.

---

## 6. Goodness of fit: time rescaling

**Theorem (random time change).** If `λ` is the true conditional intensity, then
the transformed times `Λ(tᵢ)` form a unit-rate Poisson process, so the rescaled
gaps

```
τᵢ = Λ(tᵢ) − Λ(t_{i−1}) = ∫_{t_{i−1}}^{tᵢ} λ(s) ds
```

are i.i.d. `Exp(1)`.

Test by transforming to uniformity, `uᵢ = 1 − e^(−τᵢ) ~ U(0,1)`, then a KS test
plus a Q–Q plot. Pool `τᵢ` across matches within a team-process.

Apply to **both** models. If the null already passes, there is no residual
structure for excitation to explain, and that is a clean answer. If both fail,
report it — the comparison between two misspecified models is weak evidence and
should be presented as such.

---

## 7. Inference on η

### Why not a likelihood-ratio test

`H₀: η = 0` lies on the boundary of the parameter space `η ≥ 0`. The LR
statistic is not `χ²_p`; for a single boundary parameter it is a `½:½` mixture
of `χ²_0` and `χ²_1`. With several `η` components and nuisance parameters
(`β` unidentified under the null) the correct reference distribution is not
standard. Do not use it.

### Parametric bootstrap

```
1. Fit the null on the development data → μ̂₀
2. For b = 1..B (B ≥ 500):
     simulate a full slate of matches from inhomogeneous Poisson with μ̂₀
     re-estimate μ̂^(b) from the simulated data          ← do not skip
     fit the Hawkes model → η̂^(b)
3. p = (1 + #{η̂^(b) ≥ η̂_obs}) / (B + 1)
```

Re-estimating `μ` inside each replicate is what makes the null distribution
account for background-estimation noise. Skipping it gives an anticonservative
test.

---

## 8. Detection floor

```
for η* in {0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30}:
    for r in 1..R (R ≥ 30):
        simulate slate from Hawkes(μ̂, η_self = η*, η_cross = η̂_cross)
        run the ENTIRE pipeline: re-estimate μ, fit Hawkes, bootstrap test
        record: reject?  η̂_slow  CI coverage
    power(η*) = fraction rejected at α = 0.05
    bias(η*)  = mean(η̂_slow) − η*
```

**Detection floor** = smallest `η*` with `power ≥ 0.80`.

The `bias(η*)` curve is a second result in its own right: it measures how much
of a real effect the background model absorbs. If `bias` is large and negative,
`μ` is too flexible and is eating the signal, and the flexibility should be
reduced. Report `η̂` as a function of spline df as a sensitivity analysis.

Cost estimate: 7 η values × 30 replicates × 20k matches. With the O(n)
recursion this is feasible; without it, it is not.

---

## 9. Secondary analyses

**Marked excitation.** Weight each event's contribution by its xG:

```
φ_kj(s, x) = w(x) · [η^f β_f e^(−β_f s) + η^s β_s e^(−β_s s)],   w(x) = (x/x̄)^γ
```

`γ = 0` recovers the unmarked model, so this nests too. Tests whether big
chances generate more momentum than speculative ones.

**Momentum in quality rather than quantity.** Separately from arrival times,
regress a shot's xG on the recent excitation state:

```
E[xᵢ] = ψ₀ + ψ₁ · (excitation at tᵢ) + game-state controls
```

Momentum could mean teams create *better* chances during a spell rather than
merely more of them. `ψ₁` tests that directly and is cheap once the intensity
machinery exists.

**State-dependent η.** Let `η_self` vary with score differential. Tactical and
psychological readings make different predictions here: a tactical account
predicts excitation is suppressed when leading (the block goes deeper), a
psychological one predicts it is elevated.

---

## References

- Hawkes (1971), *Spectra of some self-exciting and mutually exciting point
  processes*, Biometrika — the original.
- Ogata (1981), *On Lewis' simulation method for point processes*, IEEE Trans.
  Inf. Theory — thinning algorithm and the recursion.
- Daley & Vere-Jones, *An Introduction to the Theory of Point Processes* —
  compensators, time rescaling, stability.
- Laub, Taimre & Pollett (2015), *Hawkes Processes*, arXiv:1507.02822 — the
  clearest short introduction; read this first.
- Self & Liang (1987) on boundary-constrained likelihood ratio asymptotics, for
  the reason §7 avoids the LRT.

---

## 10. Addendum: what the recovery test measured

*Added during implementation. Sections 1-9 above are the design; this section
records where the implementation had to depart from it, and why. It corrects one
claim in section 4.*

### The two-stage fit is biased downwards, not upwards

Section 4 says: *"Two-stage is biased toward finding excitation, since `μ` was
fitted without excitation in the model."* That reasoning is about the kernel's
shape, and it is true as far as it goes. It is dominated by a level effect that
points the other way.

Fitting an inhomogeneous Poisson `μ` to data generated by a Hawkes process with
spectral radius `ρ` inflates `μ̂` by roughly `1/(1 − ρ)`: every offspring event has
to be explained as background, because the model has nowhere else to put it.
Holding that inflated `μ̂` fixed and then fitting `η` on top would over-predict the
total rate, so the optimiser is forced to push `η̂` towards zero.

Measured on the StatsBomb Premier League 2015/16 slate, planting `η_self` and
running the whole pipeline (`results/estimator_comparison.csv`):

| planted `η_self` | two-stage `η̂` | EM `η̂` | joint `η̂` |
| --- | --- | --- | --- |
| 0.00 | 0.000 | 0.000 | 0.000 |
| 0.10 | 0.032 | 0.086 | 0.091 |
| 0.30 | 0.102 | 0.272 | 0.288 |

The two-stage estimator returns about a third of a real effect. It is kept in the
codebase as `method="two_stage"` because the size of that gap is itself a result,
but it is not the default and no reported number uses it.

### What replaces it

EM for a Hawkes process with a parametric background (Veen & Schoenberg 2008)
removes the bias but converges slowly — on the slate above it was still climbing
after ten iterations. The useful observation is that the gradient of the
*observed* log-likelihood with respect to the background parameters is exactly the
EM M-step gradient:

```
∂/∂θ [ Σᵢ log(μᵢ + excᵢ) − ∫μ ]  =  Σᵢ (μᵢ/λᵢ) ∂log μᵢ/∂θ  −  Σ_bins Δ_b μ_b ∂log μ_b/∂θ
```

which is a Poisson-regression gradient evaluated at the *fractional* counts
`p_bg = μ/λ`. So `μ`, `η` and `β` go into one L-BFGS pass with analytic gradients
for the background parameters and central differences for the three kernel
parameters. Same fixed point as EM, an order of magnitude fewer likelihood
evaluations. This is `src/models/joint.py`, and it is what "profile likelihood or
a two-stage fit; document which" resolves to here.

**Initialisation matters and is not incidental.** Warm-starting the joint fit from
the two-stage solution recovers `η̂ = 0.12` from a planted 0.30: the two-stage fit
lands on the short-`τ` boundary and the joint optimiser stays in that basin.
Started from the configured defaults it recovers 0.30 from any of
`τ₀ ∈ {2, 5, 10}` minutes. The pipeline therefore does *not* warm-start it.

### The mean of the rescaled gaps is below 1 by construction

Section 6 says the rescaled gaps are i.i.d. `Exp(1)`. The *complete* gaps are. The
final interval of each process — from its last chance to the whistle — is
censored, is not a gap, and is excluded. The pooled mean is therefore

```
mean(τ) ≈ (n_events − n_processes · E[censored tail]) / n_events
```

which at ~15 events per team-process sits near 0.92. A pooled mean of exactly 1.0
would mean the censored tails were being counted by mistake.

### The pooled KS p-value is not the number to read

At 10⁴–10⁶ pooled gaps the KS test rejects any model of football, and says nothing
about which of two models is better. `inference/gof.py` reports three things
instead: the KS *statistic* (a distance, comparable between models on the same
data), the fraction of individual team-processes rejecting at α = 0.05 (nominal is
5%), and whether those per-process p-values are themselves uniform. On the
Premier League 2015/16 slate the inhomogeneous Poisson null gives 5.5% and
uniform — that is, the null already fits, which section 6 anticipates as a clean
answer in its own right.

### Simulation holds the game-state path fixed

Section 8 does not say whether the score and card trajectories are regenerated
inside each simulation replicate. They are not: each simulated match reuses the
observed path of the real match it stands for, which keeps `μ` a known function
and lets every replicate reuse the same design matrix. The consequence, stated
plainly, is that the detection floor measures how much of a planted `η` the
*background estimation* absorbs — not how much the score-state feedback loop
absorbs. The second quantity needs goals to be resimulated from the marks and the
covariate path rebuilt per replicate; the hook for it is
`models/simulate.py`, and it is not currently exercised.
