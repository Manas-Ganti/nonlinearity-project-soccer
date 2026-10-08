# Is football momentum real, or is it just the scoreline?

When a team wins a corner, then another, then hits the bar, the commentary says a
goal is coming. The testable version of that claim is narrow: creating a chance
makes the next chance more likely.

Chances do cluster, but mostly for dull reasons. A team that leads drops deeper. A
red card reshapes both sides. Chances pile up before half time. The question is
whether any clustering is *left over* once those are accounted for.

I treat each match as two streams of shots and fit two versions of one model: a
no-memory version, where the rate of chances depends only on score, minute, red
cards and team quality, and a memory version, where each chance briefly lifts the
rate of the chances that follow it. The quantity of interest is the **branching
ratio** — the share of chances set off by an earlier chance.

**The uncertainty is the hard part.** A result of "no momentum" can mean the effect
is absent, or that the method could never have seen it. Nothing in the answer
itself tells you which. So before touching the real data, I plant momentum of a
known size into simulated matches, run the entire pipeline on them, and check
whether it comes back. Sweeping that size gives a **detection floor**: the smallest
effect the method can reliably detect.

**The goal is a number that can be interpreted**, not one that sounds impressive.
Every estimate is reported against that floor. Above it, the estimate stands. Below
it, the honest output is an upper bound — momentum is smaller than X, or absent —
which is a real finding, and the one thing a bare zero cannot give you.
