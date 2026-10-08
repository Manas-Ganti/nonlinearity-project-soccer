#!/usr/bin/env bash
# Run on the Mac: copy a finished ARC run back into this checkout.
#
#   arc/pull_results.sh
#
# Brings back results/ (new manifests and bootstrap CSVs) and the SLURM logs. The
# ARC holdout-access log lands beside the local one, not on top of it, so the audit
# trail from both machines is kept. Nothing pulled here goes into docs/RESULTS.md
# until it has been read and checked on this machine.

set -euo pipefail

ARC_HOST="${ARC_HOST:-owl1.arc.vt.edu}"
ARC_USER="${ARC_USER:-$USER}"
ARC_REPO="${ARC_REPO:-/home/$ARC_USER/soccer-momentum}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ARC_USER}@${ARC_HOST}:${ARC_REPO}"

mkdir -p "$REPO_ROOT/logs/slurm"
rsync -av --exclude '_superseded/' "$SRC/results/" "$REPO_ROOT/results/"
rsync -av "$SRC/logs/slurm/" "$REPO_ROOT/logs/slurm/"
rsync -av "$SRC/logs/holdout_access.log" "$REPO_ROOT/logs/holdout_access.arc.log"
