#!/usr/bin/env bash
#SBATCH --job-name=sm-poss-followup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/slurm/%x-%j.out
#
# Possession Model A follow-up: docs/spec_possessions.md section 9 (amendment 1, post hoc).
#
#   arc/submit.sh --time 03:00:00 arc/possessions_followup.slurm
#
# Six arms, each recovery (GATE) -> power -> fit, then the within-half shuffle:
#
#   wyscout   merged 10 s   (9.1)        statsbomb unmerged   (9.3: Model A as specified)
#   wyscout   shuffle, unmerged and merged (9.2)
#   statsbomb merged 10 s   (9.1 + 9.3)  statsbomb shuffle, unmerged and merged
#
# The plain Wyscout Model A arm already ran (job 1372527) and is not repeated.
# A failed recovery gate stops that arm only; the job exits non-zero at the end.
# Reads data/interim/{wyscout,statsbomb}_possessions, both checked against
# arc/data.sha256. Development seasons only; nothing here passes --final-run.
source "${SLURM_SUBMIT_DIR:-.}/arc/arc_env.sh"
[[ $# -eq 0 ]] || { echo "[followup] takes no arguments; got: $*" >&2; exit 2; }
export SOCCER_WORKERS="${SOCCER_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-2} - 1 ))}"
CLI=("$PY" -m src.cli.main -v)
failed=()

run() {  # label, then CLI arguments
  local label="$1"; shift
  echo "[followup] === $label ($(date -u +%H:%M:%S) UTC, workers=$SOCCER_WORKERS)"
  local start=$(date +%s)
  if ! "${CLI[@]}" "$@"; then
    echo "[followup] === $label FAILED" >&2
    return 1
  fi
  echo "[followup] === $label done in $(( ($(date +%s) - start) / 60 )) min"
}

arm() {  # source, merge seconds
  local src="$1" merge="$2" tag="$1 merge=$2s"
  for step in poss-recovery poss-power poss-fit; do
    if ! run "$tag $step" "$step" --source "$src" --merge-regain "$merge"; then
      failed+=("$tag:$step"); return
    fi
  done
}

arm wyscout 10
arm statsbomb 0
arm statsbomb 10
for src in wyscout statsbomb; do
  for merge in 0 10; do
    run "$src merge=${merge}s poss-shuffle" poss-shuffle --source "$src" --merge-regain "$merge" \
      || failed+=("$src merge=${merge}s:poss-shuffle")
  done
done

if (( ${#failed[@]} )); then
  echo "[followup] failed: ${failed[*]}" >&2
  exit 1
fi
echo "[followup] all steps done"
