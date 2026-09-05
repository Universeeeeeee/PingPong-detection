#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${D455_PYTHON:-/home/pc/anaconda3/envs/isaac/bin/python}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
for arg in "$@"; do
    if [[ "$arg" == "--display" && -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
        printf 'No graphical session. Omit --display over SSH.\n' >&2
        exit 2
    fi
done
exec "$PY" -u "$BASE/record_d455_motion.py" "$@"
