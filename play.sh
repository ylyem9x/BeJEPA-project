#!/usr/bin/env bash
set -euo pipefail

mode="${1:-goal}"
if [[ $# -gt 0 ]]; then
    shift
fi
case "$mode" in
    goal|tracking) ;;
    *) echo "Usage: $0 {goal|tracking} [options]" >&2; exit 2 ;;
esac
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
args=(--model-folder "${MODEL_FOLDER:-outputs/bejepa}")
if [[ -n "${DATA_PATH:-}" ]]; then
    args+=(--data-path "$DATA_PATH")
fi
exec .venv/bin/python -m "humanoidverse.play_${mode}" "${args[@]}" "$@"
