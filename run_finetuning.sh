#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
export CPUSET="${CPUSET:-224-255}"
exec "$TASK_ROOT/run_all.sh" --config "$TASK_ROOT/configs/revision-finetuning.json" \
    --output "${FINETUNING_OUTPUT:-results/revision-finetuning}" "$@"
