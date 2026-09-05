#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHON="${D455_PYTHON:-/home/pc/anaconda3/envs/isaac/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    printf 'Python not found: %s. Set D455_PYTHON.\n' "$PYTHON" >&2
    exit 1
fi
for arg in "$@"; do
    if [[ "$arg" == "--display" && -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
        printf 'Run --display on the pc graphical desktop, or omit --display for SSH.\n' >&2
        exit 2
    fi
done
exec 9>"$SCRIPT_DIR/.d455_tracker.lock"
if ! flock -n 9; then
    printf 'A D455 tracker is already running.\n' >&2
    exit 1
fi
exec "$PYTHON" "$SCRIPT_DIR/d455_rgb_2d.py" "$@"
