#!/usr/bin/env python3
"""Open a ZED input, read a small number of frames and print JSON diagnostics."""

import argparse
import json
import statistics
import sys

import numpy as np

from zed_capture import import_zed_sdk
from zed_source import ZedSdkSource, ZedSourceConfig, ZedSourceError


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", choices=("live", "svo", "stream"), default="live")
    parser.add_argument("--svo")
    parser.add_argument("--stream-host")
    parser.add_argument("--stream-port", type=int, default=0)
    parser.add_argument("--serial", type=int)
    parser.add_argument("--resolution", default="AUTO")
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--depth-mode", default="PERFORMANCE")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.list_devices:
            sl = import_zed_sdk()
            devices = [
                {
                    "serial_number": int(item.serial_number),
                    "camera_model": str(item.camera_model),
                    "camera_state": str(item.camera_state),
                }
                for item in sl.Camera.get_device_list()
            ]
            print(json.dumps({"devices": devices}, ensure_ascii=False, indent=2))
            return 0
        if args.frames <= 0:
            raise ValueError("--frames must be positive")
        config = ZedSourceConfig(
            input_mode=args.input,
            svo_path=args.svo,
            stream_host=args.stream_host,
            stream_port=args.stream_port,
            serial_number=args.serial,
            resolution=args.resolution,
            fps=args.fps,
            enable_depth=args.depth,
            depth_mode=args.depth_mode,
        )
        timestamps = []
        dropped_counts = []
        last_result = None
        with ZedSdkSource(config) as source:
            for _ in range(args.frames):
                result = source.read()
                if result is None:
                    break
                last_result = result
                timestamps.append(result.frames.stereo.sdk_image_timestamp_ns)
                dropped_counts.append(result.sdk_dropped_frame_count)
            timestamp_steps_ms = [
                (newer - older) / 1e6
                for older, newer in zip(timestamps, timestamps[1:])
            ]
            expected_step_ms = (
                1000.0 / source.metadata["configured_fps"]
                if source.metadata["configured_fps"] > 0
                else None
            )
            estimated_timestamp_drops = (
                sum(
                    max(0, round(step_ms / expected_step_ms) - 1)
                    for step_ms in timestamp_steps_ms
                )
                if expected_step_ms is not None
                else None
            )
            depth_report = None
            if last_result is not None and last_result.frames.depth is not None:
                depth = last_result.frames.depth
                valid_depth = depth.depth_m[depth.valid_mask]
                depth_report = {
                    "dtype": str(depth.depth_m.dtype),
                    "shape": list(depth.depth_m.shape),
                    "valid_fraction": float(depth.valid_mask.mean()),
                    "valid_min_m": (
                        float(valid_depth.min()) if valid_depth.size else None
                    ),
                    "valid_median_m": (
                        float(np.median(valid_depth))
                        if valid_depth.size
                        else None
                    ),
                    "valid_max_m": (
                        float(valid_depth.max()) if valid_depth.size else None
                    ),
                }
            report = dict(source.metadata)
            report.update(
                {
                    "frames_read": len(timestamps),
                    "timestamps_strictly_increasing": all(
                        newer > older for older, newer in zip(timestamps, timestamps[1:])
                    ),
                    "timestamp_step_ms": {
                        "min": min(timestamp_steps_ms) if timestamp_steps_ms else None,
                        "median": (
                            statistics.median(timestamp_steps_ms)
                            if timestamp_steps_ms
                            else None
                        ),
                        "max": max(timestamp_steps_ms) if timestamp_steps_ms else None,
                    },
                    "timestamp_estimated_dropped_frames": estimated_timestamp_drops,
                    "sdk_dropped_frame_count_start": (
                        dropped_counts[0] if dropped_counts else None
                    ),
                    "sdk_dropped_frame_count": (
                        last_result.sdk_dropped_frame_count if last_result else None
                    ),
                    "sdk_dropped_frame_count_delta": (
                        dropped_counts[-1] - dropped_counts[0]
                        if dropped_counts
                        else None
                    ),
                    "sdk_current_fps": last_result.sdk_current_fps if last_result else None,
                    "depth": depth_report,
                }
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, ZedSourceError, RuntimeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
