#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
uv sync --locked
.venv/bin/python -m humanoidverse.prepare_data
