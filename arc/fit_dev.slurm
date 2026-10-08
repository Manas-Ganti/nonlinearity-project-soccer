#!/usr/bin/env bash
#SBATCH --job-name=sm-fit-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/slurm/%x-%j.out
#
# Build-order step 8: the headline fit on the development seasons.
# Submit through arc/submit.sh, which supplies account/partition/QOS/cpus/mem:
#
#   arc/submit.sh arc/fit_dev.slurm
#   arc/submit.sh arc/fit_dev.slurm --bootstrap 50 --cluster-bootstrap 20   # smoke run
#
# Arguments after the .slurm file go straight to `src.cli.main fit`, where argparse
# rejects a typo. The split is fixed to dev here and --final-run is refused: the
# holdout (step 9) is run once, deliberately, never through this launcher.
#
# Wall time: about 15 serial fits come first (5 jitter draws, null background, 8
# sensitivities, the asymmetry check) at roughly 9 core-minutes each, then the 700
# bootstrap refits spread over the workers. Expect ~3-4 h on 96 cores.

source "${SLURM_SUBMIT_DIR:-.}/arc/arc_env.sh"

for a in "$@"; do
  case "$a" in
    --final-run|--split|--split=*|--source|--source=*)
      echo "[fit_dev] refusing '$a': this launcher is the pooled dev fit only" >&2; exit 2 ;;
  esac
done

# One core stays with the parent process.
export SOCCER_WORKERS="${SOCCER_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-2} - 1 ))}"

ARGS=(--source pooled --split dev)
# Defaults match `make fit-dev`; any --bootstrap/--cluster-bootstrap passed wins.
[[ " $* " == *" --bootstrap"* ]] || ARGS+=(--bootstrap 500)
[[ " $* " == *" --cluster-bootstrap"* ]] || ARGS+=(--cluster-bootstrap 200)

echo "[fit_dev] workers=$SOCCER_WORKERS  $PY -m src.cli.main -v fit ${ARGS[*]} $*"
start=$(date +%s)
"$PY" -m src.cli.main -v fit "${ARGS[@]}" "$@"
echo "[fit_dev] done in $(( ($(date +%s) - start) / 60 )) min -> results/fit_pooled_dev.json"
