#!/usr/bin/env python3
"""ZED Mini orange-ball stereo tracker with continuous optional preview."""

import argparse
from collections import Counter, deque
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np

from zed_mini_ball import ZedMiniStereoBallPipeline
from zed_recording import Mp4PreviewRecorder
from zed_source import ZedMiniSource, ZedSourceConfig, ZedSourceError
from zed_table_pose import ZedTablePoseTracker


def parse_args():
    parser = argparse.ArgumentParser(description="ZED Mini synchronized colour-stereo ball tracker")
    parser.add_argument("--serial", type=int)
    parser.add_argument("--resolution", default="VGA")
    parser.add_argument("--fps", type=int, default=100)
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="frames to process; 0 means run continuously until q/Esc/Ctrl-C",
    )
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--print-measurements", action="store_true")
    parser.add_argument(
        "--record-svo",
        help="record the native ZED stream to .svo/.svo2 (preserves stereo and SDK timestamps)",
    )
    parser.add_argument(
        "--record-compression",
        default="H264",
        help="native SVO compression enum (default: H264; SDKs commonly also support H265/LOSSLESS)",
    )
    parser.add_argument(
        "--record-mp4",
        help="also write a viewable side-by-side left/right MP4 preview",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="disable the live window (useful over SSH without a desktop display)",
    )
    parser.add_argument(
        "--table-pose",
        action="store_true",
        help="enable low-rate rule-based table pose from left colour plus ZED depth",
    )
    parser.add_argument("--table-length", type=float, default=2.74)
    parser.add_argument("--table-width", type=float, default=1.525)
    parser.add_argument("--table-hsv-low", default="85,80,40")
    parser.add_argument("--table-hsv-high", default="135,255,255")
    parser.add_argument("--table-min-area", type=int, default=15000)
    parser.add_argument("--table-confirm-frames", type=int, default=5)
    parser.add_argument(
        "--table-pose-file",
        help="write the current validated table pose here when s is pressed in the preview window",
    )
    parser.add_argument(
        "--table-hz", type=float, default=8.0,
        help="maximum table-pose update rate; ball processing still runs every frame",
    )
    parser.add_argument(
        "--debug-dir",
        help="write final annotated ball preview and table-pose debug images when the run stops",
    )
    return parser.parse_args()


def measurement_dict(measurement):
    return {
        "timestamp_s": measurement.timestamp_s,
        "position_camera_m": measurement.position_camera_m.tolist(),
        "covariance_m2": measurement.covariance_m2.tolist(),
        "confidence": measurement.confidence,
        "source": measurement.source,
        "camera_frame_id": measurement.camera_frame_id,
        "calibration_id": measurement.calibration_id,
    }


def parse_hsv(value: str) -> np.ndarray:
    try:
        parsed = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise ValueError("HSV values must be comma-separated integers") from error
    if len(parsed) != 3 or any(item < 0 or item > 255 for item in parsed):
        raise ValueError("HSV values must have exactly three entries in [0, 255]")
    return np.asarray(parsed, dtype=np.uint8)


def table_measurement_fields(measurement, snapshot, metadata):
    if snapshot is None:
        return {"position_table_m": None, "table_frame_id": metadata.get("table_frame_id", 0)}
    position = snapshot.T_camera_table[:3, :3].T @ (
        measurement.position_camera_m - snapshot.T_camera_table[:3, 3]
    )
    return {
        "position_table_m": position.tolist(),
        "table_frame_id": metadata["table_frame_id"],
    }


def _draw_preview(
    frames, result, frames_read: int, started: float, trajectory_uv=(),
    table_metadata=None, table_position=None,
):
    """Build a side-by-side debug preview without changing the measurement path."""
    left = frames.detection.image_bgr.copy()
    right = frames.right_detection.image_bgr.copy()
    candidate = result.left_result
    if candidate.get("valid") and candidate.get("bbox") is not None:
        x, y, width, height = map(int, candidate["bbox"])
        colour = (0, 255, 0) if result.measurement is not None else (255, 255, 0)
        cv2.rectangle(left, (x, y), (x + width, y + height), colour, 2)
        cv2.circle(left, tuple(map(int, candidate["uv"])), 4, colour, -1)
    if len(trajectory_uv) >= 2:
        cv2.polylines(left, [np.rint(np.asarray(trajectory_uv)).astype(np.int32)], False, (0, 255, 0), 2)
    right_candidates = result.right_result.get("candidates", [])
    for index, item in enumerate(right_candidates):
        if item.get("bbox") is None:
            continue
        x, y, width, height = map(int, item["bbox"])
        item_colour = (0, 255, 0) if index == 0 and result.measurement is not None else (0, 220, 255)
        cv2.rectangle(right, (x, y), (x + width, y + height), item_colour, 2)
        cv2.circle(right, tuple(map(int, item["uv"])), 4, item_colour, -1)
    status = result.diagnostics.get("reason", "unknown")
    colour = (0, 220, 0) if result.measurement is not None else (0, 180, 255)
    cv2.putText(left, "ZED Mini LEFT", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(left, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    cv2.putText(right, "ZED Mini RIGHT", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    right_status = "matched stereo" if result.measurement is not None else (
        "ROI candidates: %d" % len(right_candidates) if right_candidates else "ROI search"
    )
    cv2.putText(right, right_status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    if result.measurement is not None:
        xyz = result.measurement.position_camera_m
        text = "XYZ %.2f %.2f %.2f m" % tuple(float(value) for value in xyz)
        cv2.putText(left, text, (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    if table_metadata is not None:
        table_state = table_metadata.get("state", "DISABLED")
        table_colour = (0, 255, 0) if table_metadata.get("valid") else (0, 180, 255)
        table_text = "table {} #{}".format(table_state, table_metadata.get("table_frame_id", 0))
        cv2.putText(left, table_text, (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, table_colour, 2)
        if table_position is not None:
            cv2.putText(
                left, "table xyz %.2f %.2f %.2f m" % tuple(table_position), (10, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2,
            )
    if right.shape[:2] != left.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
    canvas = np.hstack((left, right))
    elapsed = max(time.monotonic() - started, 1e-6)
    cv2.putText(
        canvas,
        "frames %d | %.1f FPS | q/Esc quit" % (frames_read, frames_read / elapsed),
        (10, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        canvas, "cyan=2D candidate green=3D measured | r clear trail t re-table s save pose",
        (10, canvas.shape[0] - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
    )
    return canvas


def save_debug_images(directory: str, ball_preview, table_tracker) -> None:
    output = Path(directory).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    if ball_preview is not None and not cv2.imwrite(str(output / "ball_preview.jpg"), ball_preview):
        raise OSError("could not write {}".format(output / "ball_preview.jpg"))
    if table_tracker is not None:
        table_debug = table_tracker.get_debug_image()
        if table_debug is not None and not cv2.imwrite(str(output / "table_pose.jpg"), table_debug):
            raise OSError("could not write {}".format(output / "table_pose.jpg"))


def main() -> int:
    args = parse_args()
    if (args.frames < 0 or args.warmup_frames < 0 or args.table_hz <= 0.0
            or args.table_length <= 0.0 or args.table_width <= 0.0):
        print(
            json.dumps(
                {"ok": False, "error": "frame counts must be non-negative; table dimensions and --table-hz must be positive"}
            ),
            file=sys.stderr,
        )
        return 2
    config = ZedSourceConfig(
        serial_number=args.serial,
        resolution=args.resolution,
        fps=args.fps,
        enable_depth=args.table_pose,
        enable_right_color=True,
        record_path=args.record_svo,
        record_compression=args.record_compression,
    )
    pipeline = ZedMiniStereoBallPipeline()
    reasons = Counter()
    frames_read = 0
    measurements = 0
    first_drop_count = None
    last_drop_count = None
    last_measurement = None
    timestamps_ns = []
    read_durations_ms = []
    process_durations_ms = []
    started = None
    source_metadata = None
    stopped_by_key = False
    display_enabled = not args.headless
    mp4_recorder = None
    table_tracker = None
    table_metadata = {"valid": False, "state": "DISABLED", "table_frame_id": 0}
    table_snapshot = None
    last_table_update_s = None
    measured_trajectory = deque(maxlen=48)
    last_preview = None
    try:
        with ZedMiniSource(config) as source:
            source_metadata = dict(source.metadata)
            if args.table_pose:
                table_tracker = ZedTablePoseTracker(
                    source.normalizer.left_intrinsics,
                    table_length_m=args.table_length,
                    table_width_m=args.table_width,
                    hsv_ranges=[(parse_hsv(args.table_hsv_low), parse_hsv(args.table_hsv_high))],
                    min_area=args.table_min_area,
                    confirm_frames=args.table_confirm_frames,
                )
            if args.record_mp4:
                mp4_recorder = Mp4PreviewRecorder(
                    args.record_mp4,
                    fps=float(source_metadata.get("configured_fps") or args.fps or 30),
                )
            for _ in range(args.warmup_frames):
                warmup_result = source.read()
                if warmup_result is None:
                    raise ZedSourceError("input ended during warmup")
                if table_tracker is not None:
                    table_tracker.update(
                        warmup_result.frames.detection, warmup_result.frames.depth
                    )
            started = time.monotonic()
            while args.frames == 0 or frames_read < args.frames:
                read_started = time.monotonic_ns()
                read_result = source.read()
                read_durations_ms.append((time.monotonic_ns() - read_started) / 1e6)
                if read_result is None:
                    break
                frames_read += 1
                timestamps_ns.append(read_result.frames.stereo.sdk_image_timestamp_ns)
                timestamp_s = read_result.frames.stereo.sdk_image_timestamp_ns / 1e9
                if table_tracker is not None:
                    if (last_table_update_s is None
                            or timestamp_s - last_table_update_s >= 1.0 / args.table_hz):
                        table_tracker.update(read_result.frames.detection, read_result.frames.depth)
                        last_table_update_s = timestamp_s
                    table_snapshot, table_metadata = table_tracker.snapshot(timestamp_s)
                if first_drop_count is None:
                    first_drop_count = read_result.sdk_dropped_frame_count
                last_drop_count = read_result.sdk_dropped_frame_count
                process_started = time.monotonic_ns()
                result = pipeline.process(read_result.frames)
                process_durations_ms.append((time.monotonic_ns() - process_started) / 1e6)
                reasons[result.diagnostics["reason"]] += 1
                if result.measurement is not None:
                    measurements += 1
                    measured_trajectory.append(result.left_result["uv"])
                    last_measurement = measurement_dict(result.measurement)
                    last_measurement.update(
                        table_measurement_fields(result.measurement, table_snapshot, table_metadata)
                    )
                    if args.print_measurements:
                        print(json.dumps(last_measurement, ensure_ascii=False), flush=True)
                if display_enabled or mp4_recorder is not None or args.debug_dir:
                    table_position = (
                        table_measurement_fields(result.measurement, table_snapshot, table_metadata)["position_table_m"]
                        if result.measurement is not None else None
                    )
                    last_preview = _draw_preview(
                        read_result.frames, result, frames_read, started, measured_trajectory,
                        table_metadata, table_position,
                    )
                if mp4_recorder is not None:
                    mp4_recorder.write_canvas(last_preview)
                if display_enabled:
                    try:
                        cv2.imshow("ZED Mini ball detection", last_preview)
                        if table_tracker is not None:
                            table_debug = table_tracker.get_debug_image()
                            if table_debug is not None:
                                cv2.imshow("ZED Mini table pose", table_debug)
                        key = cv2.waitKey(1) & 0xFF
                    except cv2.error as error:
                        raise ZedSourceError(
                            "live preview unavailable; install GUI OpenCV or rerun with --headless: {}".format(error)
                        ) from error
                    if key in (ord("q"), 27):
                        stopped_by_key = True
                        break
                    if key == ord("r"):
                        measured_trajectory.clear()
                        print("[INFO] Cleared ZED measured-ball trajectory")
                    elif key == ord("t") and table_tracker is not None:
                        table_tracker.reset()
                        table_snapshot = None
                        table_metadata = {"valid": False, "state": "SEARCHING", "table_frame_id": 0}
                        print("[INFO] ZED table reinitialization requested")
                    elif key == ord("s") and table_tracker is not None:
                        if not args.table_pose_file:
                            print("[WARN] Set --table-pose-file before saving a table pose")
                        else:
                            try:
                                print("[INFO] Saved ZED table pose: {}".format(
                                    table_tracker.save_pose(args.table_pose_file)
                                ))
                            except RuntimeError as error:
                                print("[WARN] {}".format(error))
            ended = time.monotonic()
            if display_enabled:
                cv2.destroyAllWindows()
        if args.debug_dir:
            save_debug_images(args.debug_dir, last_preview, table_tracker)
        elapsed = ended - started
        timestamp_steps_ms = [
            (newer - older) / 1e6
            for older, newer in zip(timestamps_ns, timestamps_ns[1:])
        ]
        expected_step_ms = 1000.0 / source_metadata["configured_fps"]
        timestamp_drops = sum(
            max(0, round(step_ms / expected_step_ms) - 1)
            for step_ms in timestamp_steps_ms
        )
        report = {
            "ok": True,
            "camera": source_metadata,
            "frames_read": frames_read,
            "warmup_frames": args.warmup_frames,
            "elapsed_s": elapsed,
            "processing_fps": frames_read / elapsed if elapsed > 0.0 else None,
            "sdk_timestamp_fps": (
                (frames_read - 1) * 1e9 / (timestamps_ns[-1] - timestamps_ns[0])
                if len(timestamps_ns) > 1 and timestamps_ns[-1] > timestamps_ns[0]
                else None
            ),
            "timestamp_step_ms": {
                "min": min(timestamp_steps_ms) if timestamp_steps_ms else None,
                "median": statistics.median(timestamp_steps_ms) if timestamp_steps_ms else None,
                "max": max(timestamp_steps_ms) if timestamp_steps_ms else None,
            },
            "timestamp_estimated_dropped_frames": timestamp_drops,
            "read_ms_median": (
                statistics.median(read_durations_ms) if read_durations_ms else None
            ),
            "pipeline_ms_median": (
                statistics.median(process_durations_ms) if process_durations_ms else None
            ),
            "measurements": measurements,
            "measurement_fraction": measurements / frames_read if frames_read else 0.0,
            "reasons": dict(reasons),
            "sdk_dropped_frame_count_start": first_drop_count,
            "sdk_dropped_frame_count_end": last_drop_count,
            "sdk_dropped_frame_count_delta": (
                last_drop_count - first_drop_count
                if first_drop_count is not None and last_drop_count is not None
                else None
            ),
            "last_measurement": last_measurement,
            "stopped_by_key": stopped_by_key,
            "recording": {
                "svo_path": source_metadata.get("recording_path"),
                "svo_compression": source_metadata.get("recording_compression"),
                "mp4_path": str(mp4_recorder.path) if mp4_recorder is not None else None,
                "mp4_frames_written": mp4_recorder.frames_written if mp4_recorder is not None else 0,
            },
            "table": {
                **table_metadata,
                "T_camera_table": (
                    table_snapshot.T_camera_table.tolist() if table_snapshot is not None else None
                ),
            },
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print(json.dumps({"ok": False, "error": "interrupted", "frames_read": frames_read}))
        return 130
    except (ValueError, ZedSourceError, RuntimeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    finally:
        if mp4_recorder is not None:
            mp4_recorder.close()


if __name__ == "__main__":
    raise SystemExit(main())
