#!/usr/bin/env bash
#SBATCH --job-name=sm-possessions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/slurm/%x-%j.out
#
# Possessions Model A (docs/spec_possessions.md), gates 2-4 in order:
#
#   poss-recovery (GATE) -> poss-power -> poss-fit
#
#   arc/submit.sh --time 01:00:00 arc/possessions.slurm
#
# Reads data/interim/wyscout_possessions/possessions.parquet, which arrives by
# arc/push_data.sh and is checked against arc/data.sha256 like every other slate.
# `make` stops at a failed recovery gate. Wyscout is development seasons only, and
# the loader logs every read; nothing here can reach a holdout.

source "${SLURM_SUBMIT_DIR:-.}/arc/arc_env.sh"

[[ $# -eq 0 ]] || { echo "[possessions] takes no arguments; got: $*" >&2; exit 2; }

export SOCCER_WORKERS="${SOCCER_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-2} - 1 ))}"

for step in poss-recovery poss-power poss-fit; do
  echo "[possessions] === $step ($(date -u +%H:%M:%S) UTC, workers=$SOCCER_WORKERS)"
  start=$(date +%s)
  make --no-print-directory PY="$PY" "$step"
  echo "[possessions] === $step done in $(( ($(date +%s) - start) / 60 )) min"
done
echo "[possessions] all steps done"
