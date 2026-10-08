# Sourced by arc/*.slurm. Not executable on its own.
#
# Rule throughout: fail loudly in the first second of the job, never silently an
# hour in. Each check below is a failure already seen on VT ARC on another project.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# --- Interpreter: absolute path, never `conda activate` -----------------------
# `module load Miniforge3` cannot see a personal ~/miniconda3, and `source activate`
# can report success without switching interpreters. So no activation at all.
SM_ENV="${SM_ENV:-$HOME/miniconda3/envs/soccer-momentum}"
export PY="$SM_ENV/bin/python"
export PATH="$SM_ENV/bin:$PATH"

# A zero-byte interpreter "runs" every script as an empty shell script and exits 0.
# A real CPython always prints its version.
PY_VERSION="$("$PY" -V 2>&1 || true)"
if [[ "$PY_VERSION" != Python\ 3.* ]]; then
  echo "[arc_env] FATAL: '$PY -V' printed '${PY_VERSION}'. Run arc/setup_env.sh on a login node." >&2
  exit 3
fi
"$PY" - <<'EOF'
import importlib.util, sys
missing = [m for m in ("numpy", "scipy", "pandas", "pyarrow", "src.cli.main")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"[arc_env] FATAL: {sys.executable} cannot import {missing}")
EOF

# --- Input data: byte-identical to what the Mac results were computed on -------
# data/interim is gitignored, so it arrives by arc/push_data.sh, not git. A missing
# or different file would silently change the dataset under every result already
# in results/ (CLAUDE.md: the cache is immutable input).
if ! sha256sum --quiet -c arc/data.sha256; then
  echo "[arc_env] FATAL: data/interim/pooled does not match arc/data.sha256. Run arc/push_data.sh from the Mac." >&2
  exit 4
fi

# --- Threads: the forked replicate workers own the cores -----------------------
# Without this every worker's BLAS spawns one thread per *node* core: 95 workers x
# 96 threads on a 96-core node.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg

mkdir -p logs/slurm results
echo "[arc_env] job=${SLURM_JOB_ID:-local} node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} python=$PY ($PY_VERSION) commit=$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
