#!/usr/bin/env bash
# Run on the Mac: copy the input slate to the ARC checkout.
#
#   arc/push_data.sh
#
# data/interim is gitignored (and holds the locked holdout matches too), so git
# never carries it. This copies exactly the files listed in arc/data.sha256 and
# checks them on arrival; every ARC job re-checks before it runs.

set -euo pipefail

ARC_HOST="${ARC_HOST:-owl1.arc.vt.edu}"
ARC_USER="${ARC_USER:-$USER}"
ARC_REPO="${ARC_REPO:-/home/$ARC_USER/ondemand/data/nonlinearity-project-soccer}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

shasum -a 256 -c arc/data.sha256   # never ship something other than what was recorded

# Every slate directory named in the checksum file, and only the files it names.
FILES=$(awk '{print $2}' arc/data.sha256)
DIRS=$(dirname $FILES | sort -u)
ssh "${ARC_USER}@${ARC_HOST}" "cd '${ARC_REPO}' && mkdir -p $(echo $DIRS)"
rsync -avR $FILES "${ARC_USER}@${ARC_HOST}:${ARC_REPO}/"
ssh "${ARC_USER}@${ARC_HOST}" "cd '${ARC_REPO}' && sha256sum -c arc/data.sha256"
