# How this project works, in plain terms

This is the whole pipeline, start to finish, without the maths. `docs/math.md`
has the derivations and `CLAUDE.md` has the rules; this page is the map.

## The question

When a team creates a chance, does that make their *next* chance more likely?
Commentators call it momentum. We want to know whether it exists once the boring
explanations are removed.

The boring explanations are:

- **Scoreline.** A team that is losing pushes forward and shoots more. A team
  that is winning sits back. This alone makes chances cluster in time.
- **Red cards.** Eleven against ten changes everything, from that minute on.
- **Time in the match.** Chances are rarer at kickoff and more common late.
- **Team quality.** Good teams shoot more, and against bad teams more still.

If chances *still* cluster after all four are accounted for, that leftover is
momentum. The number we report is the **branching ratio**: out of every 100
chances, how many were set off by an earlier chance. Zero means no momentum;
0.10 means one in ten.

## Why the clock matters more than the sample size

A chance can only "cause" the next one if we can see how far apart they are.
Momentum, if real, lives on a scale of a few minutes. Our first data source
(Understat) records shots to the nearest **minute**, and that turns out to be
fatal: we planted momentum into simulated matches, ran the whole method, and it
came back as **exactly zero** every time unless the effect was very large. More
matches did not help. It is not a noise problem, it is a blindness problem.

Sources that record shots to the **second** do not have this problem. So the
project now runs on second-resolution data, and uses the minute-resolution data
only to show the failure on real matches.

## The data

Three sets, three jobs.

**1. Development set (second resolution, before 2019).** Where the model is
built and the estimate is made. Two independent providers, pooled:

| Provider | What | Matches |
| --- | --- | --- |
| StatsBomb open data | Premier League, Serie A, La Liga, Ligue 1, 2015/16 | ~1,520 |
| StatsBomb open data | Barcelona's La Liga matches 2004–2019, Bundesliga 2015/16 (one club each) | ~470 |
| StatsBomb open data | FA WSL 2018/19, NWSL 2018 | ~140 |
| Wyscout (Pappalardo) open data | All five big leagues 2017/18, World Cup 2018, Euro 2016 | ~1,940 |

Roughly 4,000 matches. League and gender do not matter for the question — the
model's team-quality terms absorb them — but they are recorded, and the estimate
is reported per group as a check that pooling was fair.

**2. Holdout set (second resolution, 2019 onward).** StatsBomb's modern
releases: several women's leagues 2023/24, Ligue 1 2021–23, Bundesliga 2023/24,
Indian Super League, AFCON 2023, Women's World Cup 2023, and more. About 1,400
matches. It is pulled now and **locked**: the code refuses to read it without an
explicit `--final-run` flag, and logs every attempt. It is looked at exactly once,
at the very end. That is the only way to prove the answer was not tuned into
existence.

**3. Understat (minute resolution).** The same 2015/16 and 2017/18 seasons as
the two providers above, so the fixtures line up. Not used for estimation. Its
job is to show, on real matches, that the same shots on a coarser clock give a
different answer.

The raw pulls live in `data/raw/`. They are never re-pulled: re-scraping
mid-project would silently change the data under results already computed.

## The steps

Each step gates the next. They are run in this order and not reordered.

1. **Ingest.** Pull the data, put every provider into one common table (one row
   per shot: match, team, time, type, xG), and reconstruct the score and red-card
   state at every shot. Tested against known final scorelines.

2. **De-duplicate.** A rebound three seconds after a shot is not momentum, it is
   the same attack. Using the second-resolution data we measure where "same
   attack" ends (it is about 4 seconds) and merge anything closer than that. If
   the rule ever removes more than 15% of shots, it is eating real chances and
   the code stops.

3. **Fit the null model.** The "no momentum" model: shot rate as a function of
   scoreline, red cards, minute, and team. This is most of the work. We check
   it fits (a standard goodness-of-fit test) before comparing anything to it.

4. **Build a simulator.** Generate fake matches from the null model, with a
   momentum dial we can set.

5. **Recovery test — the gate.** Set the dial to 0.30, run the *entire* pipeline
   on the fake matches, and check it says 0.30. If this fails nothing downstream
   means anything, so nothing downstream runs.

6. **Detection floor.** Set the dial to 0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30;
   30 fake datasets each; run the pipeline on all of them. The **floor** is the
   smallest setting the method catches at least 80% of the time. Every estimate
   is reported next to this number, because an estimate below the floor is not
   "small momentum", it is "could not have seen it".

7. **Rounding experiment.** Take the second-resolution data, round it to whole
   minutes, refit. The estimate collapses. This is the resolution result stated
   in one table.

8. **Fit the real development data.** Report the branching ratio, its
   uncertainty (bootstrap over whole matches), and the floor beside it. Also
   report it per provider and per group, and with each confound switched off, so
   the reader can see what moves it.

9. **Holdout, once.** Unlock the 2019+ data, fit, report, stop.

## How to read the answer

- **Above the floor and significant:** momentum exists, at that size.
- **Below the floor:** "momentum, if it exists, is smaller than X." This is a
  real result and the one a bare zero cannot give.
- Either way the number is only meaningful with the floor next to it.

A null result is not a failed project. Forcing a positive finding out of the
data is the one way to make the work worthless.

## Rules we hold ourselves to

- Confounds go into the null model **before** the momentum estimate is looked
  at. Adding one afterwards is fitting the null until the result disappears.
- The holdout is read once.
- Every run writes a manifest: inputs, digests, seeds, versions. Every number in
  `docs/RESULTS.md` is read from a manifest, never retyped.
- Seeds everywhere. The simulations are worthless if they cannot be repeated.

## Decisions still open

- **Kernel timescale box.** How long a chance is allowed to keep raising the
  rate of the next. The spec says 3–8 minutes; the current fits let it run to 20
  and it goes to the edge. This is decided *before* the pooled data is fitted,
  so it is not a choice made after seeing the answer. Recommendation: headline
  at the spec's 3–8 minutes, wider box reported as a sensitivity.

## Where things stand

See `docs/RESULTS.md` for current numbers. All three data sets are pulled and
built: the pooled second-resolution slate is 5,439 matches (about 4,050
development, 1,390 holdout), and every Wyscout scoreline reconstructs exactly
against the official one. Both providers' de-duplication thresholds land at
4 seconds. The pipeline is rerun end to end on the pooled slate once the kernel
timescale box above is settled.
