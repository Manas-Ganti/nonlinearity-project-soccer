#!/usr/bin/env bash
#SBATCH --job-name=sm-providers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/slurm/%x-%j.out
#
# Pooling check: the headline pipeline on each provider's development slate alone.
#
#   arc/submit.sh --time 04:00:00 arc/providers.slurm
#
# Per provider, in order: recovery (GATE) -> power -> fit-dev. Each provider gets its
# own detection floor, because a floor measured on the pooled 4,053 matches says
# nothing about ~2,000. A failed recovery gate stops that provider only; the other
# still runs, and the job exits non-zero at the end so the failure is not missed.
#
# The provider slates are the pooled slate's rows, split by source (checked
# 2026-10-08), so these fits are slices of the headline data, not a different build.
# The holdout is never touched: no target here passes --final-run.

source "${SLURM_SUBMIT_DIR:-.}/arc/arc_env.sh"

[[ $# -eq 0 ]] || { echo "[providers] takes no arguments; got: $*" >&2; exit 2; }

export SOCCER_WORKERS="${SOCCER_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-2} - 1 ))}"

failed=()
for source in statsbomb wyscout; do
  for step in recovery power fit-dev; do
    echo "[providers] === $source $step ($(date -u +%H:%M:%S) UTC, workers=$SOCCER_WORKERS)"
    start=$(date +%s)
    if ! make --no-print-directory PY="$PY" SOURCE="$source" "$step"; then
      echo "[providers] === $source $step FAILED; skipping the rest of $source" >&2
      failed+=("$source:$step")
      break
    fi
    echo "[providers] === $source $step done in $(( ($(date +%s) - start) / 60 )) min"
  done
done

if (( ${#failed[@]} )); then
  echo "[providers] failed: ${failed[*]}" >&2
  exit 1
fi
echo "[providers] all steps done"
