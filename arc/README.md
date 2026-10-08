# Running on VT ARC (OWL, CPU)

Only the replicate-heavy steps go to ARC. Every replicate is an independent refit,
and `src/inference/parallel.py` forks one worker per core, so one 96-core Genoa
node runs `fit-dev` in ~3–4 h instead of ~15 h on the Mac. No GPU is involved.

## Once

On ARC (OWL login node, `owl1.arc.vt.edu`):

```bash
git clone <repo> ~/soccer-momentum && cd ~/soccer-momentum
mkdir -p ~/.config/soccer-momentum
printf 'SM_ACCOUNT=<slurm account>\nSM_MAIL_USER=<pid>@vt.edu\n' > ~/.config/soccer-momentum/arc.env
arc/setup_env.sh          # conda env pinned to arc/requirements.lock.txt, then ruff + pytest
```

On the Mac (`data/interim` is gitignored, so git does not carry it):

```bash
arc/push_data.sh          # rsync + sha256 check against arc/data.sha256
```

## Each run

On ARC:

```bash
git pull --ff-only
arc/submit.sh --dry-run arc/fit_dev.slurm                                       # look at the sbatch line
arc/submit.sh --time 01:00:00 arc/fit_dev.slurm --bootstrap 50 --cluster-bootstrap 20   # smoke run first
arc/submit.sh arc/fit_dev.slurm                                                 # the real step-8 run
squeue -u $USER ; tail -f logs/slurm/sm-fit-dev-<jobid>.out
```

On the Mac, when it finishes:

```bash
arc/pull_results.sh       # results/, logs/slurm/, logs/holdout_access.arc.log
```

## Rules

- `fit_dev.slurm` runs only `--source pooled --split dev` and refuses `--final-run`.
  The holdout is step 9: run once, deliberately, never through this launcher.
- Every job checks `data/interim/pooled` against `arc/data.sha256` before it runs.
  If the slate is ever rebuilt, regenerate that file on purpose; never edit it to
  make a job start.
- Never `--mem=0` (whole node, never backfills). `--mail-type` is required for mail.
- QOS defaults to `owl_normal_base`; `owl_normal_short` bills 2×.
- Pulled numbers are checked on the Mac before they go into `docs/RESULTS.md`.
