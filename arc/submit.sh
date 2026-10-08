#!/usr/bin/env bash
# Submit a soccer-momentum job to VT ARC OWL (CPU). Run from an OWL login node.
#
#   arc/submit.sh [--time HH:MM:SS] [--cpus N] [--mem 64G] [--qos Q] [--dry-run]
#                 <arc/*.slurm> [passthrough args...]
#
#   arc/submit.sh --dry-run arc/fit_dev.slurm            # print the sbatch line only
#   arc/submit.sh --time 01:00:00 arc/fit_dev.slurm --bootstrap 50 --cluster-bootstrap 20
#   arc/submit.sh arc/fit_dev.slurm                      # the real step-8 run
#
# Account and mail come from ~/.config/soccer-momentum/arc.env, never from the repo:
#     SM_ACCOUNT=<slurm account>
#     SM_MAIL_USER=<pid>@vt.edu
#
# OWL QOS tiers on normal_q (sacctmgr, 2026-09-28):
#   owl_normal_short  prio 1500  1 day    bills 2x
#   owl_normal_base   prio 1000  7 days   bills 1x   <- default
#   owl_normal_long   prio  500  14 days  bills 1x
# Pass --qos owl_normal_short only when queue time matters more than allocation.

set -euo pipefail

TIME=08:00:00
CPUS=96
MEM=""
QOS=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --time) TIME="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n 2,20p "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) break ;;
  esac
done

SCRIPT="${1:?usage: arc/submit.sh [options] <arc/*.slurm> [args...]}"
shift
[[ -f "$SCRIPT" ]] || { echo "no such launcher: $SCRIPT" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONF="${SM_ARC_CONF:-$HOME/.config/soccer-momentum/arc.env}"
# shellcheck disable=SC1090
[[ -f "$CONF" ]] && source "$CONF"
ACCOUNT="${SM_ACCOUNT:?set SM_ACCOUNT in $CONF}"

if [[ "$(hostname)" != owl* ]]; then
  echo "warning: OWL jobs are submitted from an OWL login node, not $(hostname)" >&2
fi
(( CPUS <= 96 )) || { echo "OWL Genoa nodes have 96 cores; --cpus $CPUS cannot fit" >&2; exit 2; }

HOURS="$(awk -F'[-:]' '{ if (NF==4) print $1*24+$2; else print $1+0 }' <<<"$TIME")"
if [[ -z "$QOS" ]]; then
  if (( HOURS >= 168 )); then QOS=owl_normal_long; else QOS=owl_normal_base; fi
fi

# Never --mem=0: it means the whole node and the job never backfills. The forked
# workers share the parent's pages, and the slate is ~7 MB on disk; 1 GB per core
# is generous.
MEM="${MEM:-$(( CPUS ))G}"

ARGS=(
  --account="$ACCOUNT"
  --partition=normal_q
  --qos="$QOS"
  --constraint=genoa
  --cpus-per-task="$CPUS"
  --mem="$MEM"
  --time="$TIME"
)
# --mail-user alone sends nothing; --mail-type is required.
[[ -n "${SM_MAIL_USER:-}" ]] && ARGS+=(--mail-user="$SM_MAIL_USER" --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80)

# SLURM opens --output before the job starts; a missing dir fails the job.
mkdir -p logs/slurm

# .slurm files are snapshotted at submit time, Python is read at run time. Both come
# from this checkout, so warn if it is not what was pushed.
if [[ -n "$(git status --porcelain -- src arc Makefile pyproject.toml 2>/dev/null)" ]]; then
  echo "warning: uncommitted changes in this checkout -- is this the code you meant to run?" >&2
fi
sha256sum --quiet -c arc/data.sha256 || { echo "data/interim/pooled is missing or differs; run arc/push_data.sh on the Mac" >&2; exit 2; }

echo "sbatch ${ARGS[*]} $SCRIPT $*"
(( DRY_RUN )) && exit 0
sbatch "${ARGS[@]}" "$SCRIPT" "$@"
