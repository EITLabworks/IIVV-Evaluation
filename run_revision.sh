#!/usr/bin/env bash
# Run from any directory. All dependent stages are controlled by one outer CV loop.
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/iivv-matplotlib}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# run_all.sh defaults to CPUSET=224-255; direct invocation uses allowed CPUs if unset.
# Override affinity explicitly, e.g. CPUSET=0-7 ./run_all.sh ...
if [[ -n "${CPUSET:-}" ]]; then
    if ! command -v taskset >/dev/null 2>&1; then
        echo 'CPUSET requires the Linux taskset utility; unset CPUSET on this host.' >&2
        exit 2
    fi
    exec taskset -c "$CPUSET" "${PYTHON:-python}" "$TASK_ROOT/run_revision.py" "$@"
fi
exec "${PYTHON:-python}" "$TASK_ROOT/run_revision.py" "$@"
