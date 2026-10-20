#!/usr/bin/env bash
# Complete clinical workflow; each selection stage remains inside the outer loop.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
exec "$TASK_ROOT/run_revision.sh" --config "$TASK_ROOT/configs/revision.json" --prepare-source "$@"
