#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${D455_PYTHON:-}"
RECORDING="$BASE/recordings/20260903_203939_366112_rally"
DEADLINE=50
SCRIPT=replay_fast_rgbd.py
BACKEND=ncc
while (( $# )); do
    case "$1" in
        --recording) RECORDING="$2"; shift 2 ;;
        --deadline-ms) DEADLINE="$2"; shift 2 ;;
        --baseline) SCRIPT=replay_realtime_3d.py; shift ;;
        --stereo-backend) BACKEND="$2"; shift 2 ;;
        --help|-h)
            printf 'Usage: %s [--recording DIR] [--deadline-ms 50] [--stereo-backend ncc|ffs] [--baseline]\n' "$0"
            printf '1x wall-clock replay of recorded RGB + stereo arrays. No physical camera is opened.\n'
            printf 'Experimental measurement branch; the 95%% target has NOT been met.\n'
            exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
case "$BACKEND" in ncc|ffs) ;; *) printf 'Unknown stereo backend: %s\n' "$BACKEND" >&2; exit 2 ;; esac
if [[ "$SCRIPT" == replay_realtime_3d.py && "$BACKEND" != ncc ]]; then
    printf 'The original baseline cannot use the experimental FFS backend.\n' >&2; exit 2
fi
if [[ -z "$PY" ]]; then
    if [[ "$BACKEND" == ffs ]]; then PY="$BASE/.venv_ffs/bin/python"
    else PY=/home/pc/anaconda3/envs/isaac/bin/python; fi
fi
if [[ ! -x "$PY" || ! -f "$RECORDING/frames/camera.json" ]]; then
    printf 'Missing Python environment or exported RGB/stereo recording: %s\n' "$RECORDING" >&2
    exit 2
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$BASE/realtime_runs"
RUN="$BASE/realtime_runs/$(date +%Y%m%d_%H%M%S)_$$"
printf '[INFO] Experimental real-time replay; output: %s\n' "$RUN"
EXTRA=()
if [[ "$SCRIPT" == replay_fast_rgbd.py ]]; then EXTRA=(--stereo-backend "$BACKEND"); fi
exec "$PY" -u "$BASE/$SCRIPT" "$RECORDING" "$RUN" --deadline-ms "$DEADLINE" "${EXTRA[@]}"
