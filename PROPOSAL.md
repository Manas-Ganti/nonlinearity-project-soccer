# Project 2 Proposal

**Is soccer momentum real, or just game state? A Hawkes-process test with a
measured detection floor.**

---

## A) The system

Association football, treated as a **bivariate point process of chance
creation**.

Each match is a stream of shot events for each of two teams. Every shot carries
an expected-goals (xG) value, the probability that the shot becomes a goal given
its location, body part, assist type, and situation. A match therefore reduces
to two marked point processes on [0, T], roughly 12 events per team.

Goals are deliberately **not** the unit of analysis. At roughly 2.7 goals per
match, any effect of realistic size is unmeasurable without decades of data.
Shots carry roughly ten times the event count and a continuous mark, which
recovers the statistical power that makes the question answerable inside a term.

The system is interesting for this course because the hypothesised mechanism is
**self-excitation**: an event raising the probability of further events. That is
a feedback loop, and it admits a stability threshold. A self-exciting process
with branching ratio approaching one becomes explosive. Where real football sits
relative to that threshold is a quantitative statement about how nonlinear the
sport actually is.

## B) The question

> After conditioning on observable game state, does a chance make the next
> chance more likely?

Chances certainly cluster. Almost all of that clustering has mundane
explanations: a team that takes the lead drops into a low block and creates
less, a red card reshapes both teams' rates, chance creation rises toward the
end of a half, and team quality varies. The question is whether **residual**
clustering survives once those are modelled.

### How I would answer it

Fit two nested models to the same data.

- **H₀**: an inhomogeneous Poisson process whose rate depends only on game state
  (score differential, minute, red cards, team strength, home advantage). No
  memory.
- **H₁**: the same background rate plus a self-exciting term. Each chance
  temporarily elevates the rate of subsequent chances, with the elevation
  decaying exponentially.

H₀ is H₁ with the excitation set to zero, so the comparison is a clean nested
test rather than a judgement between rival specifications.

The quantity of interest is the **branching ratio** η, the expected fraction of
chances triggered by earlier chances. η = 0 is memoryless. η → 1 is unstable.
Making the process bivariate separates two distinct effects that commentary
conflates: **self-excitation** (η_self, my chances beget my chances, which is
momentum as usually meant) and **cross-excitation** (η_cross, my chances beget
yours, which is the end-to-end open game).

### How I would evaluate the answer

Three safeguards, in order of importance.

**1. A measured detection floor.** Simulate matches from the fitted model with
momentum planted at known strength. Run the entire pipeline on them, including
re-estimating the background rate. Sweep the planted value and find the smallest
η recovered with 80% power. This converts a null result from "we found nothing"
into "the effect is below X or absent," which is the difference between a
non-finding and a finding.

This step also measures the project's central methodological risk directly. A
flexible background rate can absorb real momentum, and a rigid one can
manufacture it. Planting a known effect and seeing how much of it survives
background estimation quantifies that absorption instead of arguing about it.

**2. A parametric bootstrap null.** The likelihood-ratio test is non-standard
here because η = 0 sits on the boundary of the parameter space. Instead,
simulate from the fitted H₀, refit H₁ to each replicate, and compare the
observed η̂ against that null distribution.

**3. A held-out split.** Develop everything on 2014–15 through 2018–19, then run
once on 2019–20 onward and report that number. Momentum can be found in any
dataset given enough freedom to look.

Goodness of fit is assessed by the time-rescaling theorem: under a correct
model, rescaled inter-event times are i.i.d. Exp(1), testable by KS and Q–Q.
This checks whether either model describes football at all, separately from
which of the two is better.

## C) Data

**Primary: Understat.** Shot-level xG for the top five European leagues from
2014–15 onward, freely accessible. Roughly 20,000 matches and on the order of
500,000 shots. Each record carries minute, xG, team, situation (open play, set
piece, penalty), and result, which is enough to reconstruct the score and card
state at every instant.

**Cross-check: FBref.** Independent xG model over overlapping fixtures. Fitting
the headline model against both quantifies how much the answer depends on whose
xG model is used, which is a real source of uncertainty that most xG work leaves
unexamined.

**Fallback for timestamp resolution: StatsBomb open data.** Understat records
minute as an integer, which is coarse relative to the fastest clustering in the
data. If sub-minute structure turns out to matter, StatsBomb's second-level
events cover fewer matches but resolve it. Verifying the actual timestamp
resolution is the first task in the project, because it determines whether the
fast-timescale component below is identifiable at all.

No licensing barrier, no cost, no privileged access. Scraping and cleaning is
a weekend of work.

## D) Model classes

**1. Inhomogeneous Poisson (the null).** Log-linear intensity, piecewise
constant on one-minute bins so the compensator integral is a finite sum. Terms:
team attack and opponent defence strength, home advantage, a smooth function of
match minute, a function of score differential, and red-card state. This model
carries all the domain knowledge, and building it well is most of the work.

**2. Bivariate Hawkes with exponential kernel (the alternative).** The null's
background plus self- and cross-excitation terms. Fit by maximum likelihood
using the O(n) recursion available for exponential kernels. Stationarity is
enforced through the spectral radius of the branching matrix.

**Two-timescale kernel.** This is a design decision forced by the sport rather
than by statistics. Rebound shots and corner sequences produce clustering on a
timescale of seconds, and it is mechanical, not psychological. Momentum in the
sense people mean it operates over minutes. A single-component kernel will lock
onto the fast mechanical structure and report a large branching ratio that means
nothing. The kernel therefore has a fast component (seconds) absorbing rebounds
and set-piece chains and a slow component (minutes) carrying the hypothesis of
interest. Only the slow component is reported as momentum.

**3. A simulator.** Ogata thinning from the fitted bivariate Hawkes, with the
branching ratio under my control. This generates the ground truth for the
detection-floor experiment and is what makes the evaluation in (B) possible.

### Extensions if time allows

- **Marked excitation**: let a chance's excitation scale with its xG, testing
  whether big chances generate more momentum than speculative ones.
- **Momentum in quality rather than quantity**: regress a shot's xG on recent
  chance history, asking whether teams create *better* chances during a spell
  rather than merely more of them.
- **State dependence of η**: does self-excitation differ when level, ahead, or
  behind? This is where the psychological reading and the tactical reading make
  different predictions.

## Known limitations

**Score state is endogenous.** Goals are produced by the process being modelled,
so conditioning on score differential conditions on a downstream outcome. This is
standard practice in the football modelling literature and the resulting estimand
is still well defined (excess clustering given observable state), but it is not
a causal quantity. I will report a sensitivity fit with score state excluded.

**xG is a model, not a measurement.** Any result inherits the biases of the
upstream xG model, which is why the FBref cross-check is in the plan rather than
optional.

**A null result is a likely outcome.** The design anticipates this: the
detection floor is what makes that outcome publishable rather than empty.
