#!/usr/bin/env bash
# Source from Bash or Zsh to create/reuse and activate the environment here.
# Usage: source ./setup_environment.sh

_iivv_setup_environment() {
    local iivv_script iivv_root iivv_shell iivv_hook
    if [ -n "${BASH_VERSION:-}" ]; then
        if [ "${BASH_SOURCE[0]}" = "$0" ]; then
            printf '%s\n' 'Use: source ./setup_environment.sh (activation must affect your current shell).' >&2
            return 2
        fi
        iivv_script="${BASH_SOURCE[0]}"
        iivv_shell=bash
    elif [ -n "${ZSH_VERSION:-}" ]; then
        iivv_script="${(%):-%x}"
        iivv_shell=zsh
    else
        printf '%s\n' 'Source this helper from Bash or Zsh.' >&2
        return 2
    fi
    iivv_root="$(cd -- "$(dirname -- "$iivv_script")" && pwd)" || return 1
    if ! command -v conda >/dev/null 2>&1; then
        printf '%s\n' 'Conda is not on PATH. Initialize your Conda installation first.' >&2
        return 1
    fi
    iivv_hook="$(conda "shell.$iivv_shell" hook)" || return 1
    eval "$iivv_hook" || return 1
    if ! conda env list | awk '$1 == "iivv-revision" {found=1} END {exit !found}'; then
        conda env create --file "$iivv_root/environment.yml" || return 1
    fi
    conda activate iivv-revision || return 1
    printf '%s\n' 'Activated iivv-revision. Next: make revision-tests'
}

if _iivv_setup_environment "$@"; then
    unset -f _iivv_setup_environment
else
    unset -f _iivv_setup_environment
    return 1 2>/dev/null || exit 1
fi
