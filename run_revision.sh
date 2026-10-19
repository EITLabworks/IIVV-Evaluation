#!/usr/bin/env bash
# Run from any directory. All dependent stages are controlled by one outer CV loop.
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/iivv-matplotlib}"

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

exec "${PYTHON:-python}" "$TASK_ROOT/run_revision.py" "$@"