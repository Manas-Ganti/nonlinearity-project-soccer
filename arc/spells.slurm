#!/usr/bin/env bash
#SBATCH --job-name=sm-spells
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/slurm/%x-%j.out
#
# Spell models S and G: docs/spec_spells.md sections 4-6 and 9.
#
#   arc/submit.sh --time 06:00:00 arc/spells.slurm
#
# Arms, in order; each is spell-recovery (GATE) -> spell-power -> spell-fit:
#
#   wyscout    S, G   then spell-holm     (development, exploratory)
#   statsbomb  S, G   then spell-holm     (section 9 confirmation)
#   statsbomb without one-club matches, S, G, then spell-holm
#
# A failed recovery gate stops that arm only (and that source's Holm step);
# the job exits non-zero at the end. Reads the shot slates and the
# possession tables, all checked against arc/data.sha256. Development seasons
# only; nothing here passes --final-run.
source "${SLURM_SUBMIT_DIR:-.}/arc/arc_env.sh"
[[ $# -eq 0 ]] || { echo "[spells] takes no arguments; got: $*" >&2; exit 2; }
export SOCCER_WORKERS="${SOCCER_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-2} - 1 ))}"
CLI=("$PY" -m src.cli.main -v)
failed=()

run() {  # label, then CLI arguments
  local label="$1"; shift
  echo "[spells] === $label ($(date -u +%H:%M:%S) UTC, workers=$SOCCER_WORKERS)"
  local start=$(date +%s)
  if ! "${CLI[@]}" "$@"; then
    echo "[spells] === $label FAILED" >&2
    return 1
  fi
  echo "[spells] === $label done in $(( ($(date +%s) - start) / 60 )) min"
}

arm() {  # label, then source flags; runs S and G, then Holm if both fitted
  local label="$1"; shift
  local ok=1
  for model in S G; do
    for step in spell-recovery spell-power spell-fit; do
      if ! run "$label $model $step" "$step" --model "$model" "$@"; then
        failed+=("$label:$model:$step"); ok=0; break
      fi
    done
  done
  if (( ok )); then
    run "$label spell-holm" spell-holm "$@" || failed+=("$label:holm")
  fi
}

# Gate 0: the test suite in ARC's own environment. A failure stops the whole job.
echo "[spells] === pytest ($(date -u +%H:%M:%S) UTC)"
if ! "$PY" -m pytest -q; then
  echo "[spells] test suite FAILED on ARC: nothing runs" >&2
  exit 1
fi

arm wyscout --source wyscout
arm statsbomb --source statsbomb
arm statsbomb-no1club --source statsbomb --exclude-one-club

if (( ${#failed[@]} )); then
  echo "[spells] failed: ${failed[*]}" >&2
  exit 1
fi
echo "[spells] all steps done"
