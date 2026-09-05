#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${D455_PYTHON:-/home/pc/anaconda3/envs/isaac/bin/python}"
if [[ ! -x "$PY" ]]; then
    printf 'Python not found: %s. Set D455_PYTHON to a compatible interpreter.\n' "$PY" >&2
    exit 1
fi
for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        exec "$PY" "$BASE/d455_table_tennis_tracker.py" --help
    fi
    if [[ "$arg" == "--visualize" || "$arg" == "--display" ]]; then
        if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
            printf 'Display requested but no graphical session is available. Run on the pc desktop or omit --display.\n' >&2
            exit 2
        fi
    fi
done
exec 9>"$BASE/.d455_tracker.lock"
if ! flock -n 9; then
    printf 'This tracker is already running. Stop its existing process before restarting.\n' >&2
    exit 1
fi
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "$BASE/runs"
RUN_DIR="$(mktemp -d "$BASE/runs/$(date +%Y%m%d_%H%M%S)_XXXXXX")"
extra=()
for arg in "$@"; do
    if [[ "$arg" == "--display" ]]; then extra+=(--visualize); else extra+=("$arg"); fi
done
command=("$PY" -u "$BASE/d455_table_tennis_tracker.py"
    --table-length 2.74 --table-width 1.525
    --table-hsv-low 85,80,40 --table-hsv-high 135,255,255
    --table-min-area 15000 --ball-color orange --ir-fps 90
    --color-exposure-us 0
    --table-pose-file "$BASE/table_pose_initial_guess.npz"
    --jsonl "$RUN_DIR/telemetry.jsonl" --debug-dir "$RUN_DIR"
    --table-video "$RUN_DIR/table_pose.mp4"
    --ball-video "$RUN_DIR/ball_overlay.mp4"
    "${extra[@]}")
"$PY" - "$RUN_DIR/run.json" "${command[@]}" <<'PY'
import datetime, hashlib, importlib.metadata, json, pathlib, sys
versions = {}
for package in ("numpy", "opencv-python", "pyzmq", "pyrealsense2"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = "unknown"
meta = {
    "started_at": datetime.datetime.now().astimezone().isoformat(),
    "command": sys.argv[2:],
    "python": sys.executable,
    "versions": versions,
    "script_sha256": hashlib.sha256(pathlib.Path(sys.argv[4]).read_bytes()).hexdigest(),
    "table_geometry_sha256": hashlib.sha256(pathlib.Path(sys.argv[4]).with_name("table_pose_geometry.py").read_bytes()).hexdigest(),
    "ball_validation_sha256": hashlib.sha256(pathlib.Path(sys.argv[4]).with_name("ball_validation.py").read_bytes()).hexdigest(),
    "ball_motion_sha256": hashlib.sha256(pathlib.Path(sys.argv[4]).with_name("ball_motion.py").read_bytes()).hexdigest(),
    "ball_image_sha256": hashlib.sha256(pathlib.Path(sys.argv[4]).with_name("ball_image.py").read_bytes()).hexdigest(),
}
try:
    import pyrealsense2 as rs
    meta["connected_devices"] = [
        {"name": d.get_info(rs.camera_info.name),
         "serial": d.get_info(rs.camera_info.serial_number)}
        for d in rs.context().query_devices()]
except Exception as exc:
    meta["device_query_error"] = str(exc)
pathlib.Path(sys.argv[1]).write_text(json.dumps(meta, indent=2) + "\n")
PY
printf '[INFO] Run directory: %s\n' "$RUN_DIR"
cd "$BASE"
exec "${command[@]}" > >(tee -a "$RUN_DIR/console.log") 2>&1
