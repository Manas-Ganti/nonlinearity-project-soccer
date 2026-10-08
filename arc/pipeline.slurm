#!/usr/bin/env bash
#SBATCH --job-name=sm-pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/slurm/%x-%j.out
#
# Build-order steps 3-8 on the pooled development slate, in order, through make:
#
#   fit-null -> estimator-comparison -> recovery (GATE) -> power -> tau-profile
#            -> rounding-bias -> fit-dev
#
#   arc/submit.sh --time 04:00:00 arc/pipeline.slurm
#
# Everything that runs the joint fit is here, so a change to that fit is rerun
# everywhere at once and no manifest is left computed under the old code. De-dup
# calibration (step 2) does not fit anything and is not rerun; `make report` is run
# on the Mac after the results are pulled and checked.
#
# `make` stops at the first failing step. The recovery gate exits non-zero when the
# planted eta is not recovered, so nothing downstream of a failed gate is computed.
# The holdout is never touched: no target here passes --final-run.

source "${SLURM_SUBMIT_DIR:-.}/arc/arc_env.sh"

[[ $# -eq 0 ]] || { echo "[pipeline] takes no arguments; got: $*" >&2; exit 2; }

export SOCCER_WORKERS="${SOCCER_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-2} - 1 ))}"

for step in fit-null estimator-comparison recovery power tau-profile rounding-bias fit-dev; do
  echo "[pipeline] === $step ($(date -u +%H:%M:%S) UTC, workers=$SOCCER_WORKERS)"
  start=$(date +%s)
  make --no-print-directory PY="$PY" SOURCE=pooled "$step"
  echo "[pipeline] === $step done in $(( ($(date +%s) - start) / 60 )) min"
done
echo "[pipeline] all steps done"
