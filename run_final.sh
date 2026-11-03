#!/usr/bin/env bash
# One entry point: new nested selection, all reports, then structural/input checks.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"

export CPUSET="${CPUSET:-224-255}"
export PYTHONUNBUFFERED=1
output=""
resume=0
preflight=0
forward=()
while (($#)); do
    case "$1" in
        -h|--help)
            printf '%s\n' 'Usage: ./run_final.sh [--output PATH] [--resume | --preflight] [run_revision.py options]' 'Defaults: configs/revision-final.json; CPUs 224-255; a new timestamped result directory.'
            exit 0 ;;
        --output)
            [[ $# -ge 2 && -n "$2" ]] || { echo '--output requires a path' >&2; exit 2; }
            output="$2"; shift 2 ;;
        --output=*)
            output="${1#*=}"
            [[ -n "$output" ]] || { echo '--output requires a path' >&2; exit 2; }
            shift ;;
        --resume) resume=1; forward+=("$1"); shift ;;
        --preflight) preflight=1; forward+=("$1"); shift ;;
        --config|--config=*) echo 'Edit configs/revision-final.json; this launcher fixes its protocol config.' >&2; exit 2 ;;
        *) forward+=("$1"); shift ;;
    esac
done
if ((resume && preflight)); then
    echo 'Choose --resume or --preflight, not both.' >&2
    exit 2
fi
if ((resume)) && [[ -z "$output" ]]; then
    echo 'Resume requires --output results/revision-final-<original timestamp>' >&2
    exit 2
fi
if ((resume)) && [[ ! -f "$output/run_manifest.json" ]]; then
    echo 'Resume requires the original result directory containing run_manifest.json.' >&2
    exit 2
fi
output="${output:-results/revision-final-$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p logs
run_name="$(basename -- "$output")"
log="logs/${run_name}$(date +-%Y%m%d-%H%M%S)-$$.log"
: >> "$log"
ln -sfn -- "$(basename -- "$log")" logs/revision-final-latest.log
exec > >(tee -a "$log") 2>&1
printf 'PIPELINE: CPUs=%s; config=configs/revision-final.json\nOUTPUT=%s\nLOG=%s\n' "$CPUSET" "$output" "$log"
if ((!preflight)); then
    echo 'PIPELINE 0/6: regression checks on the server'
    taskset -c "$CPUSET" "${PYTHON:-python}" -m pytest -q tests/test_final_selection.py tests/test_revision.py tests/test_finetuning.py tests/test_parameter_explorer.py
fi
./run_revision.sh --config configs/revision-final.json --prepare-source --output "$output" "${forward[@]}"
if ((!preflight)); then
    echo 'PIPELINE 6/6: verify completed artifacts, patient separation, statistics and input hashes'
    taskset -c "$CPUSET" "${PYTHON:-python}" -m revision.report_cli check --results-dir "$output" --verify-inputs
    taskset -c "$CPUSET" "${PYTHON:-python}" scripts/create_parameter_explorer.py --results-dir "$output"
    printf 'FINAL RUN READY: %s\n' "$output"
fi
