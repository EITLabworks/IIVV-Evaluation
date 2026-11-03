#!/usr/bin/env bash
# Quality first; then one complete nested selection and reporting pipeline.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
export CPUSET="${CPUSET:-224-255}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/iivv-matplotlib}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
output=""
quality=""
quality_only=0
resume=0
forward=()
while (($#)); do
    case "$1" in
        -h|--help)
            printf '%s\n' 'Quality: ./run_all.sh --data-quality-only --output results/revision-quality' \
                'Tuning: ./run_all.sh --quality-results results/revision-quality [--output PATH] [--resume]' \
                'Config: configs/revision.json. Default CPUs: 224-255; override CPUSET=0-23.'
            exit 0 ;;
        --output|--quality-results)
            [[ $# -ge 2 && -n "$2" ]] || { echo "$1 requires a path" >&2; exit 2; }
            if [[ "$1" == --output ]]; then output="$2"; else quality="$2"; fi
            shift 2 ;;
        --output=*)
            output="${1#*=}"
            [[ -n "$output" ]] || { echo '--output requires a path' >&2; exit 2; }
            shift ;;
        --quality-results=*)
            quality="${1#*=}"
            [[ -n "$quality" ]] || { echo '--quality-results requires a path' >&2; exit 2; }
            shift ;;
        --data-quality-only|--preflight) quality_only=1; shift ;;
        --resume) resume=1; forward+=(--resume); shift ;;
        --config|--config=*) echo 'Edit configs/revision.json; this launcher fixes the study protocol.' >&2; exit 2 ;;
        *) forward+=("$1"); shift ;;
    esac
done
command -v taskset >/dev/null || { echo 'Linux taskset is required.' >&2; exit 2; }
if ((quality_only)); then
    ((resume == 0)) || { echo 'Quality analysis requires a fresh directory.' >&2; exit 2; }
    output="${output:-results/revision-quality-$(date +%Y%m%d-%H%M%S)-$$}"
    forward+=(--data-quality-only)
else
    [[ -n "$quality" && -f "$quality/quality_manifest.json" ]] || {
        echo 'First run --data-quality-only, then supply its directory with --quality-results.' >&2; exit 2;
    }
    if ((resume)); then
        [[ -n "$output" && -f "$output/run_manifest.json" ]] || {
            echo 'Resume requires --output pointing to the original interrupted/failed run.' >&2; exit 2;
        }
    fi
    output="${output:-results/revision-three-tasks-$(date +%Y%m%d-%H%M%S)-$$}"
    forward+=(--quality-results "$quality")
fi
if ((!resume)) && [[ -f "$output/run_manifest.json" ]]; then
    echo 'This output already contains a run; choose a fresh output directory.' >&2
    exit 2
fi
mkdir -p logs
log="logs/$(basename -- "$output")-$(date +%Y%m%d-%H%M%S)-$$.log"
: >> "$log"
ln -sfn -- "$(basename -- "$log")" logs/revision-latest.log
exec > >(tee -a "$log") 2>&1
printf 'PIPELINE: CPUs=%s; config=configs/revision.json\nOUTPUT=%s\nLOG=%s\n' "$CPUSET" "$output" "$log"
if ((!quality_only)); then
    echo 'PIPELINE 0/6: regression checks on the server'
    taskset -c "$CPUSET" "${PYTHON:-python}" -m pytest -q tests
fi
taskset -c "$CPUSET" "${PYTHON:-python}" run_revision.py --config configs/revision.json \
    --prepare-source --output "$output" "${forward[@]}"
if ((!quality_only)); then
    echo 'PIPELINE 6/6: verify complete results, patient separation, quality provenance and inputs'
    taskset -c "$CPUSET" "${PYTHON:-python}" -m revision.report_cli check --results-dir "$output" --verify-inputs
    taskset -c "$CPUSET" "${PYTHON:-python}" scripts/create_parameter_explorer.py --results-dir "$output"
    printf 'REVISION RESULTS READY: %s\n' "$output"
fi
