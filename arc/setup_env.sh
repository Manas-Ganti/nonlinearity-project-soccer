#!/usr/bin/env bash
# One-time install of the project env on an OWL login node.
#
#   arc/setup_env.sh            # create + install + verify
#   arc/setup_env.sh --verify   # verify only
#
# A dedicated conda env pinned to arc/requirements.lock.txt, which is a freeze of
# the Mac .venv that produced every result already in results/. Same numpy/scipy
# on both machines, so ARC numbers are comparable with the Mac ones.

set -euo pipefail

SM_ENV="${SM_ENV:-$HOME/miniconda3/envs/soccer-momentum}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
PY="$SM_ENV/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

verify() {
  echo "== verify $SM_ENV"
  "$PY" -V || { echo "FATAL: interpreter broken"; exit 3; }
  "$PY" -c 'import numpy, scipy, pandas, pyarrow; print("numpy", numpy.__version__, "| scipy", scipy.__version__, "| pandas", pandas.__version__)'
  "$PY" -m pip check || echo "(pip check reported conflicts above -- read them)"
  # `make check`, with this env's tools instead of .venv/bin.
  "$SM_ENV/bin/ruff" check .
  "$PY" -m pytest -q
  sha256sum -c arc/data.sha256 || echo "data not here yet: run arc/push_data.sh on the Mac"
}

if [[ "${1:-}" == "--verify" ]]; then verify; exit 0; fi

[[ -x "$CONDA" ]] || { echo "no conda at $CONDA; set CONDA=/path/to/bin/conda" >&2; exit 2; }

# Login-node /tmp is small; build in $HOME.
export TMPDIR="$HOME/tmp/soccer-momentum-install"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

[[ -x "$PY" ]] || "$CONDA" create -y -p "$SM_ENV" python=3.12.7
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r arc/requirements.lock.txt

verify
