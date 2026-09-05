#!/usr/bin/env python3
"""
D455 table-tennis perception pipeline
=====================================

Purpose
-------
1. Use D455 RGB + HSV for low-cost global ball reacquisition.
2. Use the D455 left/right IR pair at the highest supported frame rate
   (90 FPS requested, with automatic 60/30 FPS fallback) for high-rate local tracking.
3. Fuse validated stereo triangulation and independent RGB-aligned depth with
   source-dependent uncertainty in a single camera-frame estimate.
4. Use trajectory prediction to constrain ROIs and reject false positives.
5. Estimate and track the 6-DoF table pose from table-colour segmentation,
   line features, and the depth plane. After initialization, projected model-edge
   tracking allows partial-table observations.
6. Publish both camera-frame and table-centred coordinates over ZMQ.

Coordinate conventions
----------------------
* D455 optical camera frame: +x right, +y down, +z forward.
* Table frame: origin at the table-top centre, +x from robot side toward the
  opponent, +y to the robot's left, +z upward.
* T_camera_table maps a point from table frame to the left-IR camera frame.
* A sphere has no visually observable orientation. T_camera_ball therefore uses
  a motion frame: x-axis follows ball velocity, z-axis is derived from the table
  normal. This is NOT a spin/orientation estimate.

Important limitations
---------------------
* A line-only method cannot recover a unique table centre from a single unknown
  edge without a prior. For reliable partial-view tracking, initialize once with
  enough table geometry visible or load a saved pose acquired at the same head pose.
* Table validity and camera-frame ball validity are reported separately.
* Short-horizon ball prediction uses constant velocity; automatic table-contact
  prediction is disabled until a contact detector is validated on flight data.

Dependencies
------------
    pip install numpy opencv-python pyzmq pyrealsense2

Typical run
-----------
    python d455_table_tennis_tracker.py \
        --ball-color orange \
        --visualize \
        --zmq-port 5555

Keyboard when visualization is enabled
--------------------------------------
    q / Esc : quit
    r       : reset ball tracker
    t       : force table re-initialization
    s       : save current table pose to --table-pose-file
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq
from table_pose_geometry import FixedTablePoseTracker as TablePoseTracker
from ball_validation import (BallValidationConfig, BallTrackGate, contour_features,
                             validate_rgb_candidate, size_check, depth_evidence)
from ball_image import ImageBallDetector
from ball_motion import TimestampedBallFilter, measurement_covariance


# =============================================================================
# Basic math utilities
# =============================================================================

EPS = 1e-9


def normalize(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < EPS:
        if fallback is None:
            return np.zeros_like(v)
        return normalize(np.asarray(fallback, dtype=np.float64))
    return v / n


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return T[:3, :3] @ np.asarray(p, dtype=np.float64).reshape(3) + T[:3, 3]


def rotation_angle(R: np.ndarray) -> float:
    value = clamp((float(np.trace(R)) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(value))


def rotation_interp(R0: np.ndarray, R1: np.ndarray, alpha: float) -> np.ndarray:
    """Geodesic interpolation on SO(3), implemented with Rodrigues."""
    alpha = clamp(alpha, 0.0, 1.0)
    dR = R0.T @ R1
    rvec, _ = cv2.Rodrigues(dR)
    Rstep, _ = cv2.Rodrigues(rvec * alpha)
    return R0 @ Rstep


def exp_rotation(omega_dt: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(omega_dt, dtype=np.float64).reshape(3, 1))
    return R


def rotation_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = normalize(a)
    b = normalize(b)
    cross = np.cross(a, b)
    s = float(np.linalg.norm(cross))
    c = clamp(float(np.dot(a, b)), -1.0, 1.0)
    if s < 1e-8:
        if c > 0:
            return np.eye(3, dtype=np.float64)
        # 180-degree rotation around any axis orthogonal to a.
        axis = normalize(np.cross(a, np.array([1.0, 0.0, 0.0])))
        if np.linalg.norm(axis) < 1e-6:
            axis = normalize(np.cross(a, np.array([0.0, 1.0, 0.0])))
        return exp_rotation(axis * math.pi)
    axis = cross / s
    angle = math.atan2(s, c)
    return exp_rotation(axis * angle)


def transform_to_list(T: Optional[np.ndarray]) -> Optional[List[List[float]]]:
    if T is None:
        return None
    return np.asarray(T, dtype=np.float64).tolist()


def matrix_to_quaternion_xyzw(R: np.ndarray) -> List[float]:
    """Convert rotation matrix to quaternion [x, y, z, w]."""
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), EPS)
    return q.tolist()


def ensure_rotation_matrix(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    Rout = U @ Vt
    if np.linalg.det(Rout) < 0:
        U[:, -1] *= -1
        Rout = U @ Vt
    return Rout


def intrinsics_to_cv(intr: rs.intrinsics) -> Tuple[np.ndarray, np.ndarray]:
    K = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    coeffs = np.asarray(intr.coeffs, dtype=np.float64).reshape(-1)
    # OpenCV accepts 4/5/8 coefficients. RealSense typically provides 5.
    if coeffs.size < 5:
        coeffs = np.pad(coeffs, (0, 5 - coeffs.size))
    return K, coeffs[:5]


def extrinsics_to_transform(ext: rs.extrinsics) -> np.ndarray:
    # rs2_extrinsics stores rotation column-major, matching SDK point transforms.
    R = np.asarray(ext.rotation, dtype=np.float64).reshape(3, 3, order="F")
    t = np.asarray(ext.translation, dtype=np.float64).reshape(3)
    return make_transform(R, t)


def project_rs(intr: rs.intrinsics, p: np.ndarray) -> Optional[np.ndarray]:
    p = np.asarray(p, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(p)) or p[2] <= 1e-4:
        return None
    uv = rs.rs2_project_point_to_pixel(intr, p.astype(float).tolist())
    return np.asarray(uv, dtype=np.float64)


def deproject_rs(intr: rs.intrinsics, uv: Sequence[float], depth: float) -> np.ndarray:
    p = rs.rs2_deproject_pixel_to_point(intr, [float(uv[0]), float(uv[1])], float(depth))
    return np.asarray(p, dtype=np.float64)


def transform_rs(ext: rs.extrinsics, p: np.ndarray) -> np.ndarray:
    out = rs.rs2_transform_point_to_point(ext, np.asarray(p, dtype=float).tolist())
    return np.asarray(out, dtype=np.float64)


def ball_motion_rotation(velocity_camera: np.ndarray, table_z_camera: Optional[np.ndarray]) -> np.ndarray:
    """Return a motion-aligned frame, not a physical/spin orientation."""
    v = np.asarray(velocity_camera, dtype=np.float64)
    if np.linalg.norm(v) < 1e-4:
        return np.eye(3, dtype=np.float64)
    x_axis = normalize(v)
    z_hint = normalize(
        table_z_camera if table_z_camera is not None else np.array([0.0, -1.0, 0.0]),
        fallback=np.array([0.0, -1.0, 0.0]),
    )
    # Make z orthogonal to x.
    z_axis = z_hint - float(np.dot(z_hint, x_axis)) * x_axis
    if np.linalg.norm(z_axis) < 1e-4:
        z_axis = np.array([0.0, 0.0, 1.0]) - x_axis[2] * x_axis
    z_axis = normalize(z_axis, fallback=np.array([0.0, -1.0, 0.0]))
    y_axis = normalize(np.cross(z_axis, x_axis), fallback=np.array([1.0, 0.0, 0.0]))
    z_axis = normalize(np.cross(x_axis, y_axis))
    return ensure_rotation_matrix(np.column_stack((x_axis, y_axis, z_axis)))


# =============================================================================
# Thread-safe latest-frame queues
# =============================================================================


class DropOldestQueue:
    """Small bounded queue that preserves low latency by dropping oldest items."""

    def __init__(self, maxsize: int = 3):
        self._q: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self.dropped = 0

    def put_latest(self, item: Any) -> None:
        while True:
            try:
                self._q.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def get(self, timeout: Optional[float] = None) -> Any:
        return self._q.get(timeout=timeout)

    def get_nowait(self) -> Any:
        return self._q.get_nowait()

    def drain_latest(self) -> Optional[Any]:
        latest = None
        while True:
            try:
                latest = self._q.get_nowait()
            except queue.Empty:
                break
        return latest


# =============================================================================
# RealSense capture
# =============================================================================


@dataclass
class StereoPacket:
    ir_left: np.ndarray
    ir_right: np.ndarray
    depth_raw: Optional[np.ndarray]
    timestamp_s: float
    host_timestamp_s: float
    frame_number: int


@dataclass
class ColorFramesetPacket:
    frameset: Any
    timestamp_s: float
    host_timestamp_s: float
    frame_number: int


@dataclass
class CameraModel:
    ir_left_intr: rs.intrinsics
    ir_right_intr: rs.intrinsics
    color_intr: rs.intrinsics
    left_to_right: rs.extrinsics
    right_to_left: rs.extrinsics
    color_to_left: rs.extrinsics
    left_to_color: rs.extrinsics
    T_left_right: np.ndarray
    T_right_left: np.ndarray
    T_left_color: np.ndarray
    T_color_left: np.ndarray
    depth_scale: float
    selected_ir_fps: int
    selected_color_size: Tuple[int, int]


class D455AsyncCapture:
    """Asynchronous D455 capture with minimal callback work and bounded queues."""

    def __init__(
        self,
        serial: Optional[str],
        requested_ir_fps: int,
        color_width: int,
        color_height: int,
        color_fps: int,
        queue_size: int,
        emitter_enabled: Optional[bool],
        ir_exposure_us: float,
        color_exposure_us: float,
        ir_gain: Optional[float] = None,
    ) -> None:
        self.serial = serial
        self.requested_ir_fps = requested_ir_fps
        self.color_width = color_width
        self.color_height = color_height
        self.color_fps = color_fps
        self.emitter_enabled = emitter_enabled
        self.ir_exposure_us = ir_exposure_us
        self.color_exposure_us = color_exposure_us
        self.ir_gain = ir_gain

        self.pipeline = rs.pipeline()
        self.profile: Optional[rs.pipeline_profile] = None
        self.model: Optional[CameraModel] = None
        self.stereo_queue = DropOldestQueue(maxsize=queue_size)
        self.color_queue = DropOldestQueue(maxsize=max(2, queue_size // 2))
        self._running = False
        self._callback_lock = threading.Lock()
        self._last_ir_frame_number: Optional[int] = None
        self._last_color_frame_number: Optional[int] = None
        self.hardware_ir_drops = 0
        self.callback_errors = 0

    @staticmethod
    def _candidate_ir_fps(requested: int) -> List[int]:
        values = [requested, 90, 60, 30]
        out: List[int] = []
        for value in values:
            if value not in out and value > 0:
                out.append(value)
        return out

    def _make_config(self, ir_fps: int, color_size: Tuple[int, int]) -> rs.config:
        cfg = rs.config()
        if self.serial:
            cfg.enable_device(self.serial)
        # The depth stream and both IR streams are requested at the same rate so
        # that the stereo-module frames are coherent.
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, ir_fps)
        cfg.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, ir_fps)
        cfg.enable_stream(rs.stream.infrared, 2, 848, 480, rs.format.y8, ir_fps)
        cfg.enable_stream(
            rs.stream.color,
            int(color_size[0]),
            int(color_size[1]),
            rs.format.bgr8,
            self.color_fps,
        )
        return cfg

    def start(self) -> CameraModel:
        if self._running:
            if self.model is None:
                raise RuntimeError("Capture marked running without camera model")
            return self.model

        devices = rs.context().query_devices()
        if not len(devices):
            raise RuntimeError("No RealSense device connected; check the D455 USB cable")
        if self.serial and not any(d.get_info(rs.camera_info.serial_number) == self.serial for d in devices):
            raise RuntimeError(f"Requested RealSense serial {self.serial} is not connected")

        color_sizes = [
            (self.color_width, self.color_height),
            (1280, 720),
            (848, 480),
            (640, 480),
        ]
        dedup_sizes: List[Tuple[int, int]] = []
        for size in color_sizes:
            if size not in dedup_sizes:
                dedup_sizes.append(size)

        last_error: Optional[BaseException] = None
        selected_fps = 0
        selected_color = dedup_sizes[0]
        for ir_fps in self._candidate_ir_fps(self.requested_ir_fps):
            for color_size in dedup_sizes:
                cancel = getattr(self, "cancel_event", None)
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("Camera startup cancelled")
                cfg = self._make_config(ir_fps, color_size)
                try:
                    self.profile = self.pipeline.start(cfg, self._frame_callback)
                    selected_fps = ir_fps
                    selected_color = color_size
                    last_error = None
                    break
                except Exception as exc:  # hardware/profile-dependent fallback
                    last_error = exc
                    try:
                        self.pipeline.stop()
                    except Exception:
                        pass
            if self.profile is not None and last_error is None:
                break

        if self.profile is None or last_error is not None:
            raise RuntimeError(
                "Unable to start D455 with requested/fallback stream profiles. "
                f"Last error: {last_error}"
            )

        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())

        self._set_sensor_options(device)

        ir_left_profile = self.profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        ir_right_profile = self.profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()

        ir_left_intr = ir_left_profile.get_intrinsics()
        ir_right_intr = ir_right_profile.get_intrinsics()
        color_intr = color_profile.get_intrinsics()

        left_to_right = ir_left_profile.get_extrinsics_to(ir_right_profile)
        right_to_left = ir_right_profile.get_extrinsics_to(ir_left_profile)
        color_to_left = color_profile.get_extrinsics_to(ir_left_profile)
        left_to_color = ir_left_profile.get_extrinsics_to(color_profile)

        self.model = CameraModel(
            ir_left_intr=ir_left_intr,
            ir_right_intr=ir_right_intr,
            color_intr=color_intr,
            left_to_right=left_to_right,
            right_to_left=right_to_left,
            color_to_left=color_to_left,
            left_to_color=left_to_color,
            T_left_right=extrinsics_to_transform(left_to_right),
            T_right_left=extrinsics_to_transform(right_to_left),
            T_left_color=extrinsics_to_transform(color_to_left),
            T_color_left=extrinsics_to_transform(left_to_color),
            depth_scale=depth_scale,
            selected_ir_fps=selected_fps,
            selected_color_size=selected_color,
        )
        self._running = True
        return self.model

    def _set_sensor_options(self, device: rs.device) -> None:
        for sensor in device.query_sensors():
            name = ""
            try:
                name = sensor.get_info(rs.camera_info.name)
            except Exception:
                pass
            try:
                if "Stereo" in name or "Depth" in name:
                    if self.emitter_enabled is not None and sensor.supports(rs.option.emitter_enabled):
                        sensor.set_option(rs.option.emitter_enabled, 1.0 if self.emitter_enabled else 0.0)
                    if self.ir_exposure_us <= 0 and sensor.supports(rs.option.enable_auto_exposure):
                        sensor.set_option(rs.option.enable_auto_exposure, 1.0)
                    elif self.ir_exposure_us > 0 and sensor.supports(rs.option.enable_auto_exposure):
                        sensor.set_option(rs.option.enable_auto_exposure, 0.0)
                        if sensor.supports(rs.option.exposure):
                            rng = sensor.get_option_range(rs.option.exposure)
                            sensor.set_option(
                                rs.option.exposure,
                                clamp(self.ir_exposure_us, rng.min, rng.max),
                            )
                    if self.ir_gain is not None and sensor.supports(rs.option.gain):
                        rng = sensor.get_option_range(rs.option.gain)
                        sensor.set_option(rs.option.gain, clamp(self.ir_gain, rng.min, rng.max))
                elif "RGB" in name and self.color_exposure_us <= 0:
                    if sensor.supports(rs.option.enable_auto_exposure):
                        sensor.set_option(rs.option.enable_auto_exposure, 1.0)
                elif "RGB" in name and self.color_exposure_us > 0:
                    if sensor.supports(rs.option.enable_auto_exposure):
                        sensor.set_option(rs.option.enable_auto_exposure, 0.0)
                    if sensor.supports(rs.option.exposure):
                        rng = sensor.get_option_range(rs.option.exposure)
                        sensor.set_option(
                            rs.option.exposure,
                            clamp(self.color_exposure_us, rng.min, rng.max),
                        )
            except Exception as exc:
                print(f"[WARN] Could not set options on sensor '{name}': {exc}")

    def _frame_callback(self, frame: Any) -> None:
        try:
            fs = frame.as_frameset()
            ir_left = fs.get_infrared_frame(1)
            ir_right = fs.get_infrared_frame(2)
            depth = fs.get_depth_frame()
            color = fs.get_color_frame()

            if ir_left and ir_right:
                frame_number = int(ir_left.get_frame_number())
                timestamp_s = float(ir_left.get_timestamp()) * 1e-3
                if self._last_ir_frame_number is not None and frame_number > self._last_ir_frame_number + 1:
                    self.hardware_ir_drops += frame_number - self._last_ir_frame_number - 1
                self._last_ir_frame_number = frame_number
                packet = StereoPacket(
                    ir_left=np.asanyarray(ir_left.get_data()).copy(),
                    ir_right=np.asanyarray(ir_right.get_data()).copy(),
                    depth_raw=np.asanyarray(depth.get_data()).copy() if depth else None,
                    timestamp_s=timestamp_s,
                    host_timestamp_s=time.monotonic(),
                    frame_number=frame_number,
                )
                self.stereo_queue.put_latest(packet)

            # Only enqueue framesets that contain an RGB frame and a matching depth
            # frame. Alignment is intentionally done outside the device callback.
            if color and depth:
                color_frame_number = int(color.get_frame_number())
                if color_frame_number != self._last_color_frame_number:
                    self._last_color_frame_number = color_frame_number
                    try:
                        fs.keep()
                    except Exception:
                        try:
                            frame.keep()
                        except Exception:
                            pass
                    self.color_queue.put_latest(
                        ColorFramesetPacket(
                            frameset=fs,
                            timestamp_s=float(color.get_timestamp()) * 1e-3,
                            host_timestamp_s=time.monotonic(),
                            frame_number=color_frame_number,
                        )
                    )
        except Exception:
            with self._callback_lock:
                self.callback_errors += 1

    def stop(self) -> None:
        if not self._running and self.profile is None:
            return
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self._running = False


# =============================================================================
# 2D detections
# =============================================================================


@dataclass
class BallCandidate2D:
    center: np.ndarray
    radius: float
    bbox: Tuple[int, int, int, int]
    score: float
    area: float
    solidity: float
    aspect: float
    source: str
    contour: Optional[np.ndarray] = None
    features: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HSVDetectorConfig:
    hsv_ranges: List[Tuple[np.ndarray, np.ndarray]]
    min_area: float = 3.0
    max_area: float = 8000.0
    min_solidity: float = 0.40
    max_aspect: float = 6.0
    morphology_size: int = 3
    max_candidates: int = 12


class HSVBallDetector:
    """Low-cost colour detector with contour scoring and prediction gating."""

    def __init__(self, cfg: HSVDetectorConfig) -> None:
        self.cfg = cfg
        self.prev_gray: Optional[np.ndarray] = None

    @staticmethod
    def _clip_roi(
        roi: Optional[Tuple[int, int, int, int]], width: int, height: int
    ) -> Tuple[int, int, int, int]:
        if roi is None:
            return 0, 0, width, height
        x0, y0, x1, y1 = roi
        x0 = int(max(0, min(width - 1, x0)))
        y0 = int(max(0, min(height - 1, y0)))
        x1 = int(max(x0 + 1, min(width, x1)))
        y1 = int(max(y0 + 1, min(height, y1)))
        return x0, y0, x1, y1

    def detect(
        self,
        bgr: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        predicted_uv: Optional[np.ndarray] = None,
    ) -> Tuple[List[BallCandidate2D], np.ndarray]:
        h, w = bgr.shape[:2]
        x0, y0, x1, y1 = self._clip_roi(roi, w, h)
        crop = bgr[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        for low, high in self.cfg.hsv_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))

        # A small closing operation preserves elongated motion blur better than
        # aggressive opening/erosion.
        k = max(1, int(self.cfg.morphology_size))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is not None and self.prev_gray.shape == bgr.shape[:2]:
            prev_crop = self.prev_gray[y0:y1, x0:x1]
            motion = cv2.absdiff(gray, prev_crop)
            _, motion = cv2.threshold(motion, 15, 255, cv2.THRESH_BINARY)
            # Keep colour pixels even when motion is weak; motion only increases
            # confidence, avoiding failure when the camera actively follows the ball.
        else:
            motion = np.zeros_like(mask)
        self.prev_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[BallCandidate2D] = []
        crop_diag = math.hypot(crop.shape[1], crop.shape[0])

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.cfg.min_area or area > self.cfg.max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = max(float(bw) / max(bh, 1), float(bh) / max(bw, 1))
            if aspect > self.cfg.max_aspect:
                continue
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            if hull_area < EPS:
                continue
            solidity = area / hull_area
            if solidity < self.cfg.min_solidity:
                continue

            M = cv2.moments(contour)
            if abs(M["m00"]) < EPS:
                continue
            cx = float(M["m10"] / M["m00"] + x0)
            cy = float(M["m01"] / M["m00"] + y0)
            (_, _), radius = cv2.minEnclosingCircle(contour)

            perimeter = float(cv2.arcLength(contour, True))
            circularity = 4.0 * math.pi * area / max(perimeter * perimeter, EPS)
            circularity = clamp(circularity, 0.0, 1.0)
            local_motion = motion[y : y + bh, x : x + bw]
            motion_ratio = float(np.mean(local_motion > 0)) if local_motion.size else 0.0
            if predicted_uv is not None:
                dist = float(np.linalg.norm(np.array([cx, cy]) - predicted_uv))
                pred_score = math.exp(-0.5 * (dist / max(30.0, 0.10 * crop_diag)) ** 2)
            else:
                pred_score = 0.5

            # Elongated contours are allowed because a fast ball can be a streak.
            shape_score = 0.45 * solidity + 0.35 * circularity + 0.20 * math.exp(-0.35 * (aspect - 1.0))
            score = 0.48 * shape_score + 0.22 * pred_score + 0.20 * motion_ratio + 0.10
            candidates.append(
                BallCandidate2D(
                    center=np.array([cx, cy], dtype=np.float64),
                    radius=float(radius),
                    bbox=(x + x0, y + y0, bw, bh),
                    score=float(score),
                    area=area,
                    solidity=solidity,
                    aspect=aspect,
                    source="rgb_hsv",
                    contour=contour + np.array([[[x0, y0]]], dtype=contour.dtype),
                    features={"motion_ratio": motion_ratio},
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        # Validate before selecting a winner; clutter must not displace the real
        # ball just because its legacy HSV rank is slightly higher.
        candidates = candidates[:64]
        for candidate in candidates:
            candidate.features.update(contour_features(bgr, candidate.contour, self.cfg.hsv_ranges))
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = mask
        return candidates, full_mask


@dataclass
class IRDetectorConfig:
    min_area: float = 2.0
    max_area: float = 1800.0
    min_solidity: float = 0.28
    max_aspect: float = 7.0
    max_candidates: int = 12
    min_threshold: int = 16
    percentile: float = 97.0
    local_contrast_weight: float = 0.55


class IRBallDetector:
    """High-rate IR detector using motion-compensated frame difference + local contrast."""

    def __init__(self, intr: rs.intrinsics, cfg: IRDetectorConfig, name: str) -> None:
        self.intr = intr
        self.cfg = cfg
        self.name = name
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_table_pose: Optional[np.ndarray] = None
        self.K, _ = intrinsics_to_cv(intr)

    @staticmethod
    def _clip_roi(
        roi: Optional[Tuple[int, int, int, int]], width: int, height: int
    ) -> Tuple[int, int, int, int]:
        if roi is None:
            return 0, 0, width, height
        x0, y0, x1, y1 = roi
        x0 = int(max(0, min(width - 1, x0)))
        y0 = int(max(0, min(height - 1, y0)))
        x1 = int(max(x0 + 1, min(width, x1)))
        y1 = int(max(y0 + 1, min(height, y1)))
        return x0, y0, x1, y1

    def _aligned_previous(
        self, current: np.ndarray, current_table_pose: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if self.prev_frame is None or self.prev_frame.shape != current.shape:
            return None
        previous = self.prev_frame
        if current_table_pose is not None and self.prev_table_pose is not None:
            # Ignore translation in the image warp; the rotation term removes the
            # dominant head-induced background motion at 90 Hz.
            R_cur_prev = current_table_pose[:3, :3] @ self.prev_table_pose[:3, :3].T
            H = self.K @ R_cur_prev @ np.linalg.inv(self.K)
            try:
                return cv2.warpPerspective(
                    previous,
                    H,
                    (current.shape[1], current.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            except cv2.error:
                pass

        # Lightweight fallback for small inter-frame image translation.
        try:
            small_cur = cv2.resize(current, None, fx=0.25, fy=0.25)
            small_prev = cv2.resize(previous, None, fx=0.25, fy=0.25)
            shift, response = cv2.phaseCorrelate(
                np.float32(small_prev), np.float32(small_cur)
            )
            if response > 0.04 and abs(shift[0]) < 30 and abs(shift[1]) < 30:
                M = np.array([[1.0, 0.0, shift[0] * 4.0], [0.0, 1.0, shift[1] * 4.0]])
                return cv2.warpAffine(
                    previous,
                    M,
                    (current.shape[1], current.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
        except cv2.error:
            pass
        return previous

    def reset(self) -> None:
        self.prev_frame = None
        self.prev_table_pose = None

    def detect(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]],
        predicted_uv: Optional[np.ndarray],
        current_table_pose: Optional[np.ndarray],
    ) -> Tuple[List[BallCandidate2D], np.ndarray]:
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self._clip_roi(roi, w, h)
        previous = self._aligned_previous(frame, current_table_pose)

        crop = frame[y0:y1, x0:x1]
        if previous is not None:
            prev_crop = previous[y0:y1, x0:x1]
            motion = cv2.absdiff(crop, prev_crop)
        else:
            motion = np.zeros_like(crop)

        # Difference-of-Gaussians-like local contrast helps when the ball and camera
        # move together and temporal events become weak.
        blur = cv2.GaussianBlur(crop, (0, 0), sigmaX=2.0, sigmaY=2.0)
        contrast = cv2.absdiff(crop, blur)
        combined = cv2.addWeighted(
            motion,
            1.0,
            contrast,
            float(self.cfg.local_contrast_weight),
            0.0,
        )

        if combined.size:
            threshold = max(
                int(self.cfg.min_threshold),
                int(np.percentile(combined, self.cfg.percentile)),
            )
        else:
            threshold = int(self.cfg.min_threshold)
        _, binary = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[BallCandidate2D] = []
        roi_diag = math.hypot(x1 - x0, y1 - y0)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.cfg.min_area or area > self.cfg.max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = max(float(bw) / max(1, bh), float(bh) / max(1, bw))
            if aspect > self.cfg.max_aspect:
                continue
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            if hull_area < EPS:
                continue
            solidity = area / hull_area
            if solidity < self.cfg.min_solidity:
                continue
            M = cv2.moments(contour)
            if abs(M["m00"]) < EPS:
                continue
            cx = float(M["m10"] / M["m00"] + x0)
            cy = float(M["m01"] / M["m00"] + y0)
            (_, _), radius = cv2.minEnclosingCircle(contour)
            perimeter = float(cv2.arcLength(contour, True))
            circularity = clamp(4.0 * math.pi * area / max(perimeter * perimeter, EPS), 0.0, 1.0)
            peak = float(np.mean(combined[y : y + bh, x : x + bw])) / 255.0
            if predicted_uv is not None:
                dist = float(np.linalg.norm(np.array([cx, cy]) - predicted_uv))
                pred_score = math.exp(-0.5 * (dist / max(16.0, 0.18 * roi_diag)) ** 2)
            else:
                pred_score = 0.45
            shape_score = 0.40 * solidity + 0.35 * circularity + 0.25 * math.exp(-0.35 * (aspect - 1.0))
            score = 0.40 * shape_score + 0.35 * pred_score + 0.25 * clamp(peak * 2.0, 0.0, 1.0)
            candidates.append(
                BallCandidate2D(
                    center=np.array([cx, cy], dtype=np.float64),
                    radius=float(radius),
                    bbox=(x + x0, y + y0, bw, bh),
                    score=float(score),
                    area=area,
                    solidity=solidity,
                    aspect=aspect,
                    source=f"ir_{self.name}",
                    features={"minor_px": min(cv2.minAreaRect(contour)[1]) + 1.0,
                              "major_px": max(cv2.minAreaRect(contour)[1]) + 1.0},
                )
            )

        self.prev_frame = frame.copy()
        self.prev_table_pose = None if current_table_pose is None else current_table_pose.copy()
        candidates.sort(key=lambda c: c.score, reverse=True)
        full_binary = np.zeros((h, w), dtype=np.uint8)
        full_binary[y0:y1, x0:x1] = binary
        return candidates[: self.cfg.max_candidates], full_binary


# =============================================================================
# Table pose estimation and tracking
# =============================================================================


@dataclass
class PlaneFit:
    normal: np.ndarray
    d: float
    centroid: np.ndarray
    inlier_ratio: float
    rms: float
    inlier_points: np.ndarray


@dataclass
class Line3D:
    mean: np.ndarray
    direction: np.ndarray
    length: float
    pixel_line: Tuple[int, int, int, int]


@dataclass
class TablePoseMeasurement:
    T_camera_table: np.ndarray
    confidence: float
    method: str
    reprojection_error_px: float = float("nan")
    plane_rms_m: float = float("nan")
    visible_line_count: int = 0


@dataclass
class PoseSnapshot:
    T: np.ndarray
    timestamp_s: float
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    confidence: float
    last_measurement_timestamp_s: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SE3ConstantVelocityFilter:
    """Small SE(3) predictor/update filter for smooth, low-latency table pose."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshot: Optional[PoseSnapshot] = None

    def reset(self) -> None:
        with self._lock:
            self._snapshot = None

    def initialize(self, T: np.ndarray, timestamp_s: float, confidence: float) -> None:
        with self._lock:
            self._snapshot = PoseSnapshot(
                T=np.asarray(T, dtype=np.float64).copy(),
                timestamp_s=float(timestamp_s),
                linear_velocity=np.zeros(3, dtype=np.float64),
                angular_velocity=np.zeros(3, dtype=np.float64),
                confidence=float(confidence),
                last_measurement_timestamp_s=float(timestamp_s),
            )

    @staticmethod
    def _predict_snapshot(snapshot: PoseSnapshot, timestamp_s: float) -> PoseSnapshot:
        dt = clamp(float(timestamp_s - snapshot.timestamp_s), -0.1, 0.25)
        T = snapshot.T.copy()
        T[:3, :3] = ensure_rotation_matrix(T[:3, :3] @ exp_rotation(snapshot.angular_velocity * dt))
        T[:3, 3] = T[:3, 3] + snapshot.linear_velocity * dt
        confidence = snapshot.confidence * math.exp(-max(0.0, dt) / 0.70)
        return PoseSnapshot(
            T=T,
            timestamp_s=float(timestamp_s),
            linear_velocity=snapshot.linear_velocity.copy(),
            angular_velocity=snapshot.angular_velocity.copy(),
            confidence=confidence,
            last_measurement_timestamp_s=snapshot.last_measurement_timestamp_s,
        )

    def predict(self, timestamp_s: float) -> Optional[PoseSnapshot]:
        with self._lock:
            if self._snapshot is None:
                return None
            return self._predict_snapshot(self._snapshot, timestamp_s)

    def update(self, T_meas: np.ndarray, timestamp_s: float, confidence: float) -> bool:
        confidence = clamp(confidence, 0.0, 1.0)
        T_meas = np.asarray(T_meas, dtype=np.float64)
        with self._lock:
            if self._snapshot is None:
                self.initialize(T_meas, timestamp_s, confidence)
                return True

            pred = self._predict_snapshot(self._snapshot, timestamp_s)
            trans_jump = float(np.linalg.norm(T_meas[:3, 3] - pred.T[:3, 3]))
            rot_jump = rotation_angle(pred.T[:3, :3].T @ T_meas[:3, :3])
            # Reject implausible one-frame jumps, but allow lower-confidence recovery
            # when the filter has been stale for a while.
            stale = timestamp_s - self._snapshot.last_measurement_timestamp_s
            trans_limit = 0.20 + 0.35 * max(0.0, stale)
            rot_limit = math.radians(15.0 + 45.0 * max(0.0, stale))
            if (trans_jump > trans_limit or rot_jump > rot_limit) and stale < 0.50:
                return False

            alpha = clamp(0.18 + 0.62 * confidence, 0.18, 0.80)
            T_new = pred.T.copy()
            T_new[:3, 3] = (1.0 - alpha) * pred.T[:3, 3] + alpha * T_meas[:3, 3]
            T_new[:3, :3] = rotation_interp(pred.T[:3, :3], T_meas[:3, :3], alpha)

            dt = max(float(timestamp_s - self._snapshot.timestamp_s), 1e-3)
            raw_v = (T_new[:3, 3] - self._snapshot.T[:3, 3]) / dt
            dR = self._snapshot.T[:3, :3].T @ T_new[:3, :3]
            rvec, _ = cv2.Rodrigues(dR)
            raw_w = rvec.reshape(3) / dt
            beta = 0.30
            v_new = (1.0 - beta) * self._snapshot.linear_velocity + beta * raw_v
            w_new = (1.0 - beta) * self._snapshot.angular_velocity + beta * raw_w

            self._snapshot = PoseSnapshot(
                T=T_new,
                timestamp_s=float(timestamp_s),
                linear_velocity=v_new,
                angular_velocity=w_new,
                confidence=0.65 * pred.confidence + 0.35 * confidence,
                last_measurement_timestamp_s=float(timestamp_s),
            )
            return True

    def save(self, path: Path) -> bool:
        with self._lock:
            if self._snapshot is None:
                return False
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                str(path),
                T_camera_table=self._snapshot.T,
                timestamp_s=self._snapshot.timestamp_s,
                confidence=self._snapshot.confidence,
            )
            return True

    def load(self, path: Path, current_timestamp_s: float) -> bool:
        if not path.exists():
            return False
        data = np.load(str(path))
        key = "T_camera_table" if "T_camera_table" in data else "T"
        T = np.asarray(data[key], dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"Invalid table pose shape in {path}: {T.shape}")
        self.initialize(T, current_timestamp_s, confidence=0.35)
        return True


def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, float]:
    centroid = np.mean(points, axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = normalize(Vt[-1])
    d = -float(np.dot(normal, centroid))
    residuals = np.abs(points @ normal + d)
    rms = float(np.sqrt(np.mean(residuals * residuals)))
    return normal, d, centroid, rms


def fit_plane_ransac(
    points: np.ndarray,
    threshold_m: float = 0.012,
    iterations: int = 80,
    rng: Optional[np.random.Generator] = None,
) -> Optional[PlaneFit]:
    if points.shape[0] < 80:
        return None
    rng = rng or np.random.default_rng(7)
    if points.shape[0] > 3500:
        idx = rng.choice(points.shape[0], 3500, replace=False)
        pts = points[idx]
    else:
        pts = points

    best_mask: Optional[np.ndarray] = None
    best_count = 0
    for _ in range(iterations):
        ids = rng.choice(pts.shape[0], 3, replace=False)
        p0, p1, p2 = pts[ids]
        n = np.cross(p1 - p0, p2 - p0)
        nn = float(np.linalg.norm(n))
        if nn < 1e-7:
            continue
        n /= nn
        d = -float(np.dot(n, p0))
        mask = np.abs(pts @ n + d) < threshold_m
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count = count
            best_mask = mask

    if best_mask is None or best_count < max(60, int(0.25 * pts.shape[0])):
        return None
    inliers = pts[best_mask]
    n, d, centroid, rms = fit_plane_svd(inliers)
    residuals_all = np.abs(pts @ n + d)
    refined_mask = residuals_all < threshold_m
    inliers = pts[refined_mask]
    n, d, centroid, rms = fit_plane_svd(inliers)
    return PlaneFit(
        normal=n,
        d=d,
        centroid=centroid,
        inlier_ratio=float(inliers.shape[0]) / float(pts.shape[0]),
        rms=rms,
        inlier_points=inliers,
    )


def robust_depth_patch(
    depth_raw: np.ndarray,
    center: Sequence[float],
    depth_scale: float,
    radius: int = 4,
    min_depth_m: float = 0.25,
    max_depth_m: float = 6.0,
) -> Optional[float]:
    u = int(round(float(center[0])))
    v = int(round(float(center[1])))
    h, w = depth_raw.shape[:2]
    x0, x1 = max(0, u - radius), min(w, u + radius + 1)
    y0, y1 = max(0, v - radius), min(h, v + radius + 1)
    patch = depth_raw[y0:y1, x0:x1].astype(np.float64) * depth_scale
    valid = patch[(patch >= min_depth_m) & (patch <= max_depth_m)]
    if valid.size < max(3, int(0.10 * patch.size)):
        return None
    lo, hi = np.percentile(valid, [15.0, 85.0])
    trimmed = valid[(valid >= lo) & (valid <= hi)]
    if trimmed.size == 0:
        return None
    return float(np.median(trimmed))


def deproject_pixels_with_depth(
    pixels_uv: np.ndarray,
    depths_m: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    pixels = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pixels, K, dist).reshape(-1, 2)
    z = np.asarray(depths_m, dtype=np.float64).reshape(-1)
    return np.column_stack((und[:, 0] * z, und[:, 1] * z, z))


def largest_component(mask: np.ndarray, min_area: int = 500) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    if stats[idx, cv2.CC_STAT_AREA] < min_area:
        return np.zeros_like(mask)
    return np.where(labels == idx, 255, 0).astype(np.uint8)


def sample_depth_points_from_mask(
    depth_raw: np.ndarray,
    mask: np.ndarray,
    depth_scale: float,
    K: np.ndarray,
    dist: np.ndarray,
    stride: int = 5,
    min_depth_m: float = 0.35,
    max_depth_m: float = 5.0,
    max_points: int = 6000,
) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask[::stride, ::stride] > 0)
    ys = ys * stride
    xs = xs * stride
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    z = depth_raw[ys, xs].astype(np.float64) * depth_scale
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if xs.size > max_points:
        choose = np.linspace(0, xs.size - 1, max_points).astype(int)
        xs, ys, z = xs[choose], ys[choose], z[choose]
    pixels = np.column_stack((xs, ys)).astype(np.float64)
    points = deproject_pixels_with_depth(pixels, z, K, dist)
    return points, pixels


def estimate_axis_center(
    observed: Sequence[float],
    weights: Sequence[float],
    model_offsets: Sequence[float],
    prior: Optional[float],
    residual_scale: float = 0.035,
) -> Tuple[Optional[float], float, int]:
    obs = np.asarray(observed, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    models = np.asarray(model_offsets, dtype=np.float64)
    if obs.size == 0:
        return prior, float("inf"), 0
    w = w / max(float(np.sum(w)), EPS)
    candidates = [float(o - m) for o in obs for m in models]
    if prior is not None:
        candidates.extend([float(prior), float(prior - 0.03), float(prior + 0.03)])

    best_center: Optional[float] = None
    best_cost = float("inf")
    best_unique = 0
    for c in candidates:
        residual_matrix = np.abs(obs[:, None] - (c + models[None, :]))
        nearest_idx = np.argmin(residual_matrix, axis=1)
        residual = residual_matrix[np.arange(obs.size), nearest_idx]
        # Huber-like bounded cost prevents a spurious Hough line from dominating.
        scaled = residual / residual_scale
        robust = np.where(scaled <= 1.0, 0.5 * scaled * scaled, scaled - 0.5)
        cost = float(np.sum(w * robust))
        if prior is not None:
            cost += 0.18 * abs(c - prior) / max(residual_scale, EPS)
        unique = int(np.unique(nearest_idx[residual < 0.08]).size)
        if cost < best_cost:
            best_cost = cost
            best_center = c
            best_unique = unique

    # Without a prior, one visible line cannot reveal whether it is an outer edge
    # or a centre/net line. Refuse to invent a unique centre.
    if prior is None and best_unique < 2:
        return None, best_cost, best_unique
    return best_center, best_cost, best_unique


class LegacyTablePoseTracker:
    def __init__(
        self,
        color_intr: rs.intrinsics,
        depth_scale: float,
        table_length_m: float,
        table_width_m: float,
        hsv_ranges: List[Tuple[np.ndarray, np.ndarray]],
        pose_file: Path,
        min_area: int = 0,
    ) -> None:
        self.color_intr = color_intr
        self.K, self.dist = intrinsics_to_cv(color_intr)
        self.depth_scale = depth_scale
        self.length = float(table_length_m)
        self.width = float(table_width_m)
        self.hsv_ranges = hsv_ranges
        self.pose_file = pose_file
        self.min_area = min_area
        self.filter = SE3ConstantVelocityFilter()
        self._force_reinitialize = False
        self._debug_lock = threading.Lock()
        self._debug_image: Optional[np.ndarray] = None
        self._last_measurement: Optional[TablePoseMeasurement] = None
        self._rng = np.random.default_rng(1234)
        self.model_points, self.model_tangents, self.model_edge_ids = self._build_model_edges()

    def _build_model_edges(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        points: List[np.ndarray] = []
        tangents: List[np.ndarray] = []
        edge_ids: List[int] = []
        edge_id = 0

        def add_segment(p0: Sequence[float], p1: Sequence[float], count: int) -> None:
            nonlocal edge_id
            p0a = np.asarray(p0, dtype=np.float64)
            p1a = np.asarray(p1, dtype=np.float64)
            tangent = normalize(p1a - p0a)
            for s in np.linspace(0.04, 0.96, count):
                points.append((1.0 - s) * p0a + s * p1a)
                tangents.append(tangent)
                edge_ids.append(edge_id)
            edge_id += 1

        L2, W2 = 0.5 * self.length, 0.5 * self.width
        add_segment((-L2, -W2, 0.0), (L2, -W2, 0.0), 22)
        add_segment((-L2, W2, 0.0), (L2, W2, 0.0), 22)
        add_segment((-L2, -W2, 0.0), (-L2, W2, 0.0), 14)
        add_segment((L2, -W2, 0.0), (L2, W2, 0.0), 14)
        # Net/centre line and doubles centre line provide extra partial-view features.
        add_segment((0.0, -W2, 0.0), (0.0, W2, 0.0), 14)
        add_segment((-L2, 0.0, 0.0), (L2, 0.0, 0.0), 22)
        return (
            np.asarray(points, dtype=np.float64),
            np.asarray(tangents, dtype=np.float64),
            np.asarray(edge_ids, dtype=np.int32),
        )

    def request_reinitialize(self) -> None:
        self._force_reinitialize = True

    def reset(self) -> None:
        self.filter.reset()
        self._force_reinitialize = False

    def load_initial_pose(self, timestamp_s: float) -> bool:
        try:
            return self.filter.load(self.pose_file, timestamp_s)
        except Exception as exc:
            print(f"[WARN] Failed to load table pose from {self.pose_file}: {exc}")
            return False

    def save_pose(self) -> bool:
        try:
            return self.filter.save(self.pose_file)
        except Exception as exc:
            print(f"[WARN] Failed to save table pose: {exc}")
            return False

    def predict(self, timestamp_s: float) -> Optional[PoseSnapshot]:
        return self.filter.predict(timestamp_s)

    def get_debug_image(self) -> Optional[np.ndarray]:
        with self._debug_lock:
            return None if self._debug_image is None else self._debug_image.copy()

    def _table_mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        for low, high in self.hsv_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
            iterations=2,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1,
        )
        return largest_component(mask, min_area=self.min_area or max(500, int(0.005 * mask.size)))

    def _extract_plane(
        self, depth_raw: np.ndarray, mask: np.ndarray
    ) -> Optional[PlaneFit]:
        points, _ = sample_depth_points_from_mask(
            depth_raw,
            mask,
            self.depth_scale,
            self.K,
            self.dist,
            stride=4,
        )
        if points.shape[0] < 150:
            return None
        plane = fit_plane_ransac(points, threshold_m=0.014, iterations=70, rng=self._rng)
        if plane is None:
            return None
        # Point the normal toward the camera; for a camera above the table this is
        # the table's physical upward direction.
        if float(np.dot(plane.normal, -plane.centroid)) < 0.0:
            plane.normal *= -1.0
            plane.d *= -1.0
        return plane

    def _line_points_3d(
        self,
        line: Tuple[int, int, int, int],
        depth_raw: np.ndarray,
        plane: PlaneFit,
    ) -> Optional[Line3D]:
        x1, y1, x2, y2 = line
        pixel_length = math.hypot(x2 - x1, y2 - y1)
        if pixel_length < 30.0:
            return None
        us = np.linspace(x1, x2, 18)
        vs = np.linspace(y1, y2, 18)
        pixels: List[Tuple[float, float]] = []
        depths: List[float] = []
        for u, v in zip(us, vs):
            z = robust_depth_patch(depth_raw, (u, v), self.depth_scale, radius=2)
            if z is None:
                continue
            pixels.append((u, v))
            depths.append(z)
        if len(pixels) < 5:
            return None
        pts = deproject_pixels_with_depth(
            np.asarray(pixels, dtype=np.float64),
            np.asarray(depths, dtype=np.float64),
            self.K,
            self.dist,
        )
        dist = np.abs(pts @ plane.normal + plane.d)
        pts = pts[dist < 0.035]
        if pts.shape[0] < 4:
            return None
        mean = np.mean(pts, axis=0)
        _, _, Vt = np.linalg.svd(pts - mean, full_matrices=False)
        direction = Vt[0]
        direction = direction - float(np.dot(direction, plane.normal)) * plane.normal
        if np.linalg.norm(direction) < 1e-5:
            return None
        direction = normalize(direction)
        projected = (pts - mean) @ direction
        length = float(np.max(projected) - np.min(projected))
        if length < 0.06:
            return None
        return Line3D(mean=mean, direction=direction, length=length, pixel_line=line)

    def _extract_lines_3d(
        self,
        bgr: np.ndarray,
        depth_raw: np.ndarray,
        mask: np.ndarray,
        plane: PlaneFit,
    ) -> Tuple[List[Line3D], np.ndarray]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 55, 150, apertureSize=3)
        support = cv2.dilate(mask, np.ones((11, 11), dtype=np.uint8), iterations=1)
        edges = cv2.bitwise_and(edges, support)
        raw = cv2.HoughLinesP(
            edges,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=45,
            minLineLength=35,
            maxLineGap=18,
        )
        lines3d: List[Line3D] = []
        if raw is not None:
            for item in raw[:100]:
                line = tuple(int(v) for v in item[0])
                line3d = self._line_points_3d(line, depth_raw, plane)
                if line3d is not None:
                    lines3d.append(line3d)
        return lines3d, edges

    def _dominant_axes(
        self,
        plane: PlaneFit,
        lines: List[Line3D],
        prior: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z_axis = normalize(plane.normal)
        camera_forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        q0 = camera_forward - float(np.dot(camera_forward, z_axis)) * z_axis
        if np.linalg.norm(q0) < 0.1:
            q0 = np.array([1.0, 0.0, 0.0]) - z_axis[0] * z_axis
        q0 = normalize(q0)
        q1 = normalize(np.cross(z_axis, q0))

        if lines:
            bins = 180
            hist = np.zeros(bins, dtype=np.float64)
            angles: List[float] = []
            for line in lines:
                angle = math.atan2(float(np.dot(line.direction, q1)), float(np.dot(line.direction, q0)))
                angle %= math.pi
                angles.append(angle)
                idx = int(round(angle / math.pi * (bins - 1)))
                for di, scale in [(0, 1.0), (-1, 0.55), (1, 0.55), (-2, 0.25), (2, 0.25)]:
                    hist[(idx + di) % bins] += line.length * scale
            theta = float(np.argmax(hist)) / float(bins - 1) * math.pi
            a = normalize(math.cos(theta) * q0 + math.sin(theta) * q1)
        else:
            a = q0
        b = normalize(np.cross(z_axis, a))

        # Pick the long table axis. Prior orientation is strongest; otherwise use
        # camera-forward alignment, which is valid for the usual robot-side view.
        if prior is not None:
            prior_x = prior[:3, 0]
            if abs(float(np.dot(a, prior_x))) >= abs(float(np.dot(b, prior_x))):
                x_axis = a
            else:
                x_axis = b
            if float(np.dot(x_axis, prior_x)) < 0:
                x_axis *= -1.0
            if float(np.dot(z_axis, prior[:3, 2])) < 0:
                z_axis *= -1.0
        else:
            x_axis = a if abs(float(np.dot(a, q0))) >= abs(float(np.dot(b, q0))) else b
            if float(np.dot(x_axis, q0)) < 0:
                x_axis *= -1.0

        y_axis = normalize(np.cross(z_axis, x_axis))
        # +y is robot-left. In the usual front-facing setup camera +x is robot-right,
        # so prefer y opposite camera +x when there is no prior.
        if prior is None and float(np.dot(y_axis, np.array([1.0, 0.0, 0.0]))) > 0:
            y_axis *= -1.0
            x_axis *= -1.0
        R = ensure_rotation_matrix(np.column_stack((x_axis, y_axis, z_axis)))
        return R[:, 0], R[:, 1], R[:, 2]

    def _full_detect(
        self,
        bgr: np.ndarray,
        depth_raw: np.ndarray,
        prior: Optional[np.ndarray],
    ) -> Tuple[Optional[TablePoseMeasurement], np.ndarray, np.ndarray]:
        mask = self._table_mask(bgr)
        if int(np.count_nonzero(mask)) < max(700, int(0.003 * mask.size)):
            return None, mask, np.zeros_like(mask)
        plane = self._extract_plane(depth_raw, mask)
        if plane is None:
            return None, mask, np.zeros_like(mask)
        lines, edges = self._extract_lines_3d(bgr, depth_raw, mask, plane)
        x_axis, y_axis, z_axis = self._dominant_axes(plane, lines, prior)

        pts = plane.inlier_points
        qx = pts @ x_axis
        qy = pts @ y_axis
        extent_x = float(np.percentile(qx, 97) - np.percentile(qx, 3))
        extent_y = float(np.percentile(qy, 97) - np.percentile(qy, 3))

        x_obs: List[float] = []
        x_w: List[float] = []
        y_obs: List[float] = []
        y_w: List[float] = []
        for line in lines:
            dx = abs(float(np.dot(line.direction, x_axis)))
            dy = abs(float(np.dot(line.direction, y_axis)))
            if dx >= dy:
                # Line parallel to table x => constant table y.
                y_obs.append(float(np.dot(line.mean, y_axis)))
                y_w.append(line.length)
            else:
                x_obs.append(float(np.dot(line.mean, x_axis)))
                x_w.append(line.length)

        prior_cx = float(np.dot(prior[:3, 3], x_axis)) if prior is not None else None
        prior_cy = float(np.dot(prior[:3, 3], y_axis)) if prior is not None else None
        cx, cost_x, unique_x = estimate_axis_center(
            x_obs,
            x_w,
            [-0.5 * self.length, 0.0, 0.5 * self.length],
            prior_cx,
        )
        cy, cost_y, unique_y = estimate_axis_center(
            y_obs,
            y_w,
            [-0.5 * self.width, 0.0, 0.5 * self.width],
            prior_cy,
        )

        # If most of a dimension is visible, the robust point-cloud extent supplies
        # an unambiguous centre even when Hough misses a painted line.
        if cx is None and extent_x > 0.62 * self.length:
            cx = 0.5 * (float(np.percentile(qx, 3)) + float(np.percentile(qx, 97)))
        if cy is None and extent_y > 0.62 * self.width:
            cy = 0.5 * (float(np.percentile(qy, 3)) + float(np.percentile(qy, 97)))
        if cx is None and prior_cx is not None:
            cx = prior_cx
        if cy is None and prior_cy is not None:
            cy = prior_cy
        if cx is None or cy is None:
            return None, mask, edges

        cz = -plane.d  # z-axis equals plane normal
        center = x_axis * cx + y_axis * cy + z_axis * cz
        R = ensure_rotation_matrix(np.column_stack((x_axis, y_axis, z_axis)))
        T = make_transform(R, center)

        line_support = min(1.0, len(lines) / 10.0)
        plane_score = clamp(plane.inlier_ratio * math.exp(-plane.rms / 0.018), 0.0, 1.0)
        center_cost = 0.0
        finite_costs = [c for c in (cost_x, cost_y) if np.isfinite(c)]
        if finite_costs:
            center_cost = float(np.mean(finite_costs))
        center_score = math.exp(-0.45 * center_cost)
        ambiguity_penalty = 1.0
        if prior is None and (unique_x < 2 or unique_y < 2):
            ambiguity_penalty = 0.72
        confidence = clamp(
            ambiguity_penalty * (0.50 * plane_score + 0.30 * line_support + 0.20 * center_score),
            0.0,
            1.0,
        )
        measurement = TablePoseMeasurement(
            T_camera_table=T,
            confidence=confidence,
            method="full_plane_lines",
            plane_rms_m=plane.rms,
            visible_line_count=len(lines),
        )
        return measurement, mask, edges

    def _projected_edge_refine(
        self,
        bgr: np.ndarray,
        depth_raw: np.ndarray,
        prior: np.ndarray,
        table_mask: np.ndarray,
    ) -> Tuple[Optional[TablePoseMeasurement], np.ndarray]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        h, w = gray.shape

        rvec0, _ = cv2.Rodrigues(prior[:3, :3])
        tvec0 = prior[:3, 3].reshape(3, 1).copy()
        projected, _ = cv2.projectPoints(self.model_points, rvec0, tvec0, self.K, self.dist)
        projected2, _ = cv2.projectPoints(
            self.model_points + 0.035 * self.model_tangents,
            rvec0,
            tvec0,
            self.K,
            self.dist,
        )
        projected = projected.reshape(-1, 2)
        projected2 = projected2.reshape(-1, 2)

        object_points: List[np.ndarray] = []
        image_points: List[np.ndarray] = []
        used_edges: List[int] = []
        search_radius = 10
        support_mask = cv2.dilate(table_mask, np.ones((21, 21), dtype=np.uint8), iterations=1)
        if np.count_nonzero(support_mask) == 0:
            support_mask[:] = 255

        for i, (uv, uv2) in enumerate(zip(projected, projected2)):
            if not np.all(np.isfinite(uv)):
                continue
            u, v = float(uv[0]), float(uv[1])
            if u < 12 or u >= w - 12 or v < 12 or v >= h - 12:
                continue
            tangent = uv2 - uv
            nt = float(np.linalg.norm(tangent))
            if nt < 0.5:
                continue
            tangent /= nt
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
            best_score = 0.0
            best_uv: Optional[np.ndarray] = None
            for offset in range(-search_radius, search_radius + 1):
                sample = uv + normal * float(offset)
                su, sv = int(round(sample[0])), int(round(sample[1]))
                if su < 1 or su >= w - 1 or sv < 1 or sv >= h - 1:
                    continue
                if support_mask[sv, su] == 0:
                    continue
                score = float(magnitude[sv, su])
                if score > best_score:
                    best_score = score
                    best_uv = np.array([su, sv], dtype=np.float64)
            if best_uv is not None and best_score > 45.0:
                object_points.append(self.model_points[i])
                image_points.append(best_uv)
                used_edges.append(int(self.model_edge_ids[i]))

        debug_edges = cv2.convertScaleAbs(magnitude)
        if len(object_points) < 10 or len(set(used_edges)) < 2:
            return None, debug_edges

        obj = np.asarray(object_points, dtype=np.float64)
        img = np.asarray(image_points, dtype=np.float64)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj,
                img,
                self.K,
                self.dist,
                rvec0.copy(),
                tvec0.copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                return None, debug_edges
            if hasattr(cv2, "solvePnPRefineLM"):
                rvec, tvec = cv2.solvePnPRefineLM(obj, img, self.K, self.dist, rvec, tvec)
            R, _ = cv2.Rodrigues(rvec)
            R = ensure_rotation_matrix(R)
            t = tvec.reshape(3)
            T = make_transform(R, t)
            reproj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
            errors = np.linalg.norm(reproj.reshape(-1, 2) - img, axis=1)
            median_error = float(np.median(errors))
            if median_error > 5.5:
                return None, debug_edges
            if np.linalg.norm(T[:3, 3] - prior[:3, 3]) > 0.16:
                return None, debug_edges
            if rotation_angle(prior[:3, :3].T @ T[:3, :3]) > math.radians(12.0):
                return None, debug_edges

            # Use the depth plane to correct planar normal/distance without changing
            # the along-plane centre estimated by PnP.
            plane = self._extract_plane(depth_raw, table_mask)
            plane_rms = float("nan")
            if plane is not None:
                n = plane.normal.copy()
                if float(np.dot(n, T[:3, 2])) < 0:
                    n *= -1.0
                angle = math.acos(clamp(float(np.dot(normalize(T[:3, 2]), normalize(n))), -1.0, 1.0))
                if angle < math.radians(15.0):
                    R_align = rotation_between_vectors(T[:3, 2], n)
                    T[:3, :3] = ensure_rotation_matrix(R_align @ T[:3, :3])
                    T[:3, 3] -= (float(np.dot(n, T[:3, 3])) + plane.d) * n
                    plane_rms = plane.rms

            confidence = clamp(
                0.35
                + 0.035 * min(len(object_points), 20)
                + 0.20 * math.exp(-median_error / 3.0),
                0.0,
                0.95,
            )
            return (
                TablePoseMeasurement(
                    T_camera_table=T,
                    confidence=confidence,
                    method="projected_line_pnp",
                    reprojection_error_px=median_error,
                    plane_rms_m=plane_rms,
                    visible_line_count=len(set(used_edges)),
                ),
                debug_edges,
            )
        except cv2.error:
            return None, debug_edges

    def update(
        self,
        bgr: np.ndarray,
        aligned_depth_raw: np.ndarray,
        timestamp_s: float,
    ) -> Optional[TablePoseMeasurement]:
        prior_snapshot = self.filter.predict(timestamp_s)
        prior = prior_snapshot.T if prior_snapshot is not None else None
        mask = self._table_mask(bgr)

        measurement: Optional[TablePoseMeasurement] = None
        edge_debug = np.zeros(bgr.shape[:2], dtype=np.uint8)
        if prior is not None and not self._force_reinitialize:
            measurement, edge_debug = self._projected_edge_refine(
                bgr, aligned_depth_raw, prior, mask
            )

        if measurement is None:
            measurement, mask, edge_debug = self._full_detect(
                bgr, aligned_depth_raw, prior
            )

        if measurement is not None:
            accepted = self.filter.update(
                measurement.T_camera_table,
                timestamp_s,
                measurement.confidence,
            )
            if accepted:
                self._last_measurement = measurement
                self._force_reinitialize = False
            else:
                measurement = None

        debug = bgr.copy()
        if np.count_nonzero(mask) > 0:
            tint = np.zeros_like(debug)
            tint[:] = (100, 40, 0)
            blended = cv2.addWeighted(debug, 0.55, tint, 0.45, 0.0)
            debug[mask > 0] = blended[mask > 0]
        snap = self.filter.predict(timestamp_s)
        if snap is not None:
            self._draw_table_model(debug, snap.T)
            stale = timestamp_s - snap.last_measurement_timestamp_s
            cv2.putText(
                debug,
                f"table conf={snap.confidence:.2f} stale={stale:.2f}s",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
        if measurement is not None:
            cv2.putText(
                debug,
                measurement.method,
                (15, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 0),
                2,
            )
        with self._debug_lock:
            self._debug_image = debug
        return measurement

    def _draw_table_model(self, image: np.ndarray, T: np.ndarray) -> None:
        L2, W2 = 0.5 * self.length, 0.5 * self.width
        corners = np.array(
            [
                [-L2, -W2, 0.0],
                [L2, -W2, 0.0],
                [L2, W2, 0.0],
                [-L2, W2, 0.0],
            ],
            dtype=np.float64,
        )
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        uv, _ = cv2.projectPoints(corners, rvec, T[:3, 3], self.K, self.dist)
        uv = np.round(uv.reshape(-1, 2)).astype(int)
        h, w = image.shape[:2]
        for i in range(4):
            p0 = tuple(uv[i])
            p1 = tuple(uv[(i + 1) % 4])
            if (
                -2 * w < p0[0] < 3 * w
                and -2 * h < p0[1] < 3 * h
                and -2 * w < p1[0] < 3 * w
                and -2 * h < p1[1] < 3 * h
            ):
                cv2.line(image, p0, p1, (0, 255, 0), 2)
        centre = T[:3, 3]
        axes = np.array(
            [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]],
            dtype=np.float64,
        )
        origin_uv, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, T[:3, 3], self.K, self.dist)
        axes_uv, _ = cv2.projectPoints(axes, rvec, T[:3, 3], self.K, self.dist)
        origin = origin_uv.reshape(2)
        if not np.all(np.isfinite(origin)) or np.any(np.abs(origin) > 1000000):
            return
        o = tuple(int(v) for v in np.round(origin))
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for p, color in zip(axes_uv.reshape(-1, 2), colors):
            if not np.all(np.isfinite(p)) or np.any(np.abs(p) > 1000000):
                continue
            endpoint = tuple(int(v) for v in np.round(p))
            visible, start, end = cv2.clipLine((0, 0, w, h), o, endpoint)
            if visible:
                cv2.line(image, start, end, color, 3)


# =============================================================================
# Stereo triangulation and ball filter
# =============================================================================


@dataclass
class StereoBallMeasurement:
    position_left_camera: np.ndarray
    left_candidate: BallCandidate2D
    right_candidate: BallCandidate2D
    confidence: float
    reprojection_error_px: float
    source: str = "ir_stereo"
    identity_timestamp_s: Optional[float] = None
    validation: Dict[str, Any] = field(default_factory=dict)


def triangulate_rs_pair(
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    model: CameraModel,
) -> Optional[Tuple[np.ndarray, float]]:
    ray_l = deproject_rs(model.ir_left_intr, left_uv, 1.0)
    ray_r = deproject_rs(model.ir_right_intr, right_uv, 1.0)
    if abs(ray_l[2]) < EPS or abs(ray_r[2]) < EPS:
        return None
    nl = np.array([ray_l[0] / ray_l[2], ray_l[1] / ray_l[2]], dtype=np.float64)
    nr = np.array([ray_r[0] / ray_r[2], ray_r[1] / ray_r[2]], dtype=np.float64)
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((model.T_left_right[:3, :3], model.T_left_right[:3, 3:4]))
    p4 = cv2.triangulatePoints(P1, P2, nl.reshape(2, 1), nr.reshape(2, 1))
    if abs(float(p4[3, 0])) < EPS:
        return None
    p_left = (p4[:3, 0] / p4[3, 0]).astype(np.float64)
    p_right = transform_point(model.T_left_right, p_left)
    if p_left[2] <= 0.08 or p_right[2] <= 0.08:
        return None
    uv_l_hat = project_rs(model.ir_left_intr, p_left)
    uv_r_hat = project_rs(model.ir_right_intr, p_right)
    if uv_l_hat is None or uv_r_hat is None:
        return None
    reproj = 0.5 * (
        float(np.linalg.norm(uv_l_hat - left_uv))
        + float(np.linalg.norm(uv_r_hat - right_uv))
    )
    return p_left, reproj


def rgb_guided_ir_candidates(frame, rgb_candidates, model, right=False):
    """Find current-frame IR blobs inside calibrated RGB viewing cones.

    The entire 0.2--6 m depth interval is searched, without deriving depth from
    the assumed diameter. RGB only supplies a search region; the selector must
    still establish independent stereo geometry, size, appearance and identity.
    Positive spatial contrast avoids ghosts from alternating bright/dark frames.
    """
    intr = model.ir_right_intr if right else model.ir_left_intr
    transform = model.T_left_right @ model.T_left_color if right else model.T_left_color
    h, w = frame.shape[:2]
    found = []
    for rgb in rgb_candidates:
        if not rgb.validation.get('identity_ok'):
            continue
        x, y, bw, bh = rgb.bbox
        projected = []
        for uv in ((x,y),(x+bw,y),(x,y+bh),(x+bw,y+bh)):
            for z in (.2, .35, .7, 1.5, 3., 6.):
                p = transform_point(transform, deproject_rs(model.color_intr, np.array(uv), z))
                q = project_rs(intr, p)
                if q is not None and np.isfinite(q).all(): projected.append(q)
        if not projected: continue
        points = np.array(projected)
        # A bounded timing margin; it is not a relaxed identity/size gate.
        x0,y0 = np.maximum(0, np.floor(points.min(axis=0)-8)).astype(int)
        x1,y1 = np.minimum([w,h], np.ceil(points.max(axis=0)+9)).astype(int)
        if x1-x0 < 5 or y1-y0 < 5: continue
        crop = frame[y0:y1,x0:x1]
        smooth = cv2.GaussianBlur(crop.astype(np.float32), (0,0), 5.)
        contrast = crop.astype(np.float32)-smooth
        # Keep weak but resolved balls; raw-patch contrast and cross-view NCC
        # are checked independently downstream before any depth is admitted.
        binary = np.uint8(contrast > 3.)*255
        contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        local = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if not 5. <= area <= 1200.: continue
            bx,by,cw,ch = cv2.boundingRect(c)
            if bx <= 0 or by <= 0 or bx+cw >= crop.shape[1] or by+ch >= crop.shape[0]: continue
            minor,major = sorted(float(v)+1. for v in cv2.minAreaRect(c)[1])
            solidity = area/max(1.,float(cv2.contourArea(cv2.convexHull(c))))
            if minor < 3. or major/minor > 7. or solidity < .80: continue
            m = cv2.moments(c)
            center = np.array([m['m10']/m['m00']+x0,m['m01']/m['m00']+y0])
            mask = np.zeros(crop.shape,np.uint8);cv2.drawContours(mask,[c],-1,255,-1)
            peak = float(np.percentile(contrast[mask>0],90))
            score = float(np.clip(.5*solidity+.5*min(1.,peak/40.),0.,1.))
            global_c = c+np.array([[[x0,y0]]],dtype=c.dtype)
            local.append(BallCandidate2D(center,major/2.,(bx+x0,by+y0,cw,ch),score,
                area,solidity,major/minor,'ir_guided',global_c,
                dict(minor_px=minor,major_px=major,spatial_contrast=peak)))
        for c in sorted(local,key=lambda c:c.score,reverse=True)[:6]:
            if all(np.linalg.norm(c.center-old.center)>2. for old in found):found.append(c)
    return sorted(found,key=lambda c:c.score,reverse=True)[:24]


def merge_ir_candidates(guided, temporal):
    return guided+[c for c in temporal if all(np.linalg.norm(c.center-g.center)>2. for g in guided)]


def roi_around(
    uv: Optional[np.ndarray], radius: int, width: int, height: int
) -> Optional[Tuple[int, int, int, int]]:
    if uv is None or not np.all(np.isfinite(uv)):
        return None
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    return (
        max(0, u - radius),
        max(0, v - radius),
        min(width, u + radius + 1),
        min(height, v + radius + 1),
    )


class BallKalmanFilter:
    """6D [position, velocity] ballistic filter in table coordinates."""

    def __init__(
        self,
        ball_radius_m: float = 0.020,
        restitution_z: float = 0.86,
        horizontal_restitution: float = 0.93,
    ) -> None:
        self.ball_radius = float(ball_radius_m)
        self.restitution_z = float(restitution_z)
        self.horizontal_restitution = float(horizontal_restitution)
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64)
        self.timestamp_s: Optional[float] = None
        self.initialized = False
        self.last_measurement_timestamp_s: Optional[float] = None
        self.last_confidence = 0.0
        self.reject_count = 0

    def reset(self) -> None:
        self.__init__(self.ball_radius, self.restitution_z, self.horizontal_restitution)

    def initialize(self, position: np.ndarray, timestamp_s: float, confidence: float) -> None:
        self.x[:] = 0.0
        self.x[:3] = np.asarray(position, dtype=np.float64)
        self.P = np.diag([0.03**2] * 3 + [3.0**2] * 3).astype(np.float64)
        self.timestamp_s = float(timestamp_s)
        self.last_measurement_timestamp_s = float(timestamp_s)
        self.initialized = True
        self.last_confidence = float(confidence)
        self.reject_count = 0

    def _predict_once(
        self, x: np.ndarray, P: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        F = np.eye(6, dtype=np.float64)
        F[:3, 3:] = np.eye(3) * dt
        x_new = x.copy()
        x_new[:3] = x[:3] + x[3:] * dt + 0.5 * self.gravity * dt * dt
        x_new[3:] = x[3:] + self.gravity * dt
        accel_noise = 5.0
        q_pos = 0.25 * dt**4 * accel_noise**2
        q_cross = 0.5 * dt**3 * accel_noise**2
        q_vel = dt**2 * accel_noise**2
        Q = np.block(
            [
                [np.eye(3) * q_pos, np.eye(3) * q_cross],
                [np.eye(3) * q_cross, np.eye(3) * q_vel],
            ]
        )
        P_new = F @ P @ F.T + Q

        # A below-plane measurement is not evidence of a physical collision.
        # Contact prediction stays disabled pending validated impact detection.
        return x_new, P_new

    def predict_state(self, timestamp_s: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.initialized or self.timestamp_s is None:
            return None
        x = self.x.copy()
        P = self.P.copy()
        remaining = max(0.0, float(timestamp_s - self.timestamp_s))
        while remaining > 1e-6:
            dt = min(0.02, remaining)
            x, P = self._predict_once(x, P, dt)
            remaining -= dt
        return x, P

    def advance(self, timestamp_s: float) -> None:
        predicted = self.predict_state(timestamp_s)
        if predicted is None:
            return
        self.x, self.P = predicted
        self.timestamp_s = float(timestamp_s)

    def update(
        self,
        position: np.ndarray,
        timestamp_s: float,
        confidence: float,
        measurement_sigma_m: float,
        allow_reinitialize: bool = False,
    ) -> bool:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        confidence = clamp(confidence, 0.05, 1.0)
        if not self.initialized:
            self.initialize(position, timestamp_s, confidence)
            return True
        self.advance(timestamp_s)
        H = np.zeros((3, 6), dtype=np.float64)
        H[:, :3] = np.eye(3)
        sigma = float(measurement_sigma_m) / math.sqrt(confidence)
        Rm = np.eye(3, dtype=np.float64) * sigma**2
        innovation = position - H @ self.x
        S = H @ self.P @ H.T + Rm
        try:
            mahal = float(innovation.T @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            return False
        stale = (
            float(timestamp_s - self.last_measurement_timestamp_s)
            if self.last_measurement_timestamp_s is not None
            else 999.0
        )
        gate = 16.3 if stale < 0.12 else 35.0
        if mahal > gate:
            self.reject_count += 1
            if allow_reinitialize and stale > 0.25 and confidence > 0.65:
                self.initialize(position, timestamp_s, confidence)
                return True
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ Rm @ K.T
        self.last_measurement_timestamp_s = float(timestamp_s)
        self.last_confidence = confidence
        self.reject_count = 0
        return True

    def measurement_age(self, timestamp_s: float) -> float:
        if self.last_measurement_timestamp_s is None:
            return float("inf")
        return max(0.0, float(timestamp_s - self.last_measurement_timestamp_s))

    def confidence(self, timestamp_s: float) -> float:
        if not self.initialized:
            return 0.0
        age = self.measurement_age(timestamp_s)
        covariance_term = math.exp(-float(np.trace(self.P[:3, :3])) / 0.08)
        return clamp(self.last_confidence * math.exp(-age / 0.18) * covariance_term, 0.0, 1.0)


class LegacyCameraBallTracker:
    """Low-latency alpha-beta tracker in the left-IR camera frame.

    This is used before the table frame is initialized and as a camera-frame
    fallback. It is not physically invariant to camera motion, so the table-frame
    ballistic filter remains the authoritative world-state estimator.
    """

    def __init__(self) -> None:
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.timestamp_s: Optional[float] = None
        self.last_measurement_timestamp_s: Optional[float] = None
        self.last_confidence = 0.0
        self.initialized = False

    def reset(self) -> None:
        self.__init__()

    def predict(self, timestamp_s: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.initialized or self.timestamp_s is None:
            return None
        dt = clamp(float(timestamp_s - self.timestamp_s), 0.0, 0.20)
        return self.position + self.velocity * dt, self.velocity.copy()

    def update(self, position: np.ndarray, timestamp_s: float, confidence: float) -> bool:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        confidence = clamp(confidence, 0.05, 1.0)
        if not self.initialized or self.timestamp_s is None:
            self.position = position.copy()
            self.velocity[:] = 0.0
            self.timestamp_s = float(timestamp_s)
            self.last_measurement_timestamp_s = float(timestamp_s)
            self.last_confidence = confidence
            self.initialized = True
            return True
        dt = max(float(timestamp_s - self.timestamp_s), 1e-3)
        predicted = self.position + self.velocity * dt
        innovation = position - predicted
        age = (
            float(timestamp_s - self.last_measurement_timestamp_s)
            if self.last_measurement_timestamp_s is not None
            else 999.0
        )
        gate = 0.55 + 3.0 * max(0.0, age)
        if np.linalg.norm(innovation) > gate and age < 0.20:
            return False
        alpha = clamp(0.35 + 0.45 * confidence, 0.35, 0.82)
        beta = clamp(0.08 + 0.24 * confidence, 0.08, 0.32)
        self.position = predicted + alpha * innovation
        self.velocity = self.velocity + (beta / dt) * innovation
        self.timestamp_s = float(timestamp_s)
        self.last_measurement_timestamp_s = float(timestamp_s)
        self.last_confidence = confidence
        return True

    def age(self, timestamp_s: float) -> float:
        if self.last_measurement_timestamp_s is None:
            return float("inf")
        return max(0.0, float(timestamp_s - self.last_measurement_timestamp_s))

    def confidence(self, timestamp_s: float) -> float:
        if not self.initialized:
            return 0.0
        return clamp(self.last_confidence * math.exp(-self.age(timestamp_s) / 0.14), 0.0, 1.0)


class CameraBallTracker:
    """Noise-aware camera-frame KF; asynchronous sensors must not create velocity
    spikes by dividing their different depth errors by a tiny inter-frame dt.
    """
    def __init__(self):
        self.filter=TimestampedBallFilter()

    @property
    def initialized(self):return self.filter.initialized

    @property
    def velocity(self):return self.filter.x[3:]

    @velocity.setter
    def velocity(self,value):self.filter.x[3:]=value

    def reset(self):
        counts=self.filter.counters.copy()
        self.__init__()
        self.filter.counters=counts

    def predict(self,t):
        predicted=self.filter.predict_state(t)
        return None if predicted is None else (predicted[0][:3],predicted[0][3:])

    def update(self,p,t,confidence,covariance=None,key=None,source='unknown'):
        return self.filter.update(p,t,confidence,covariance,key,source)

    def age(self,t):return self.filter.measurement_age(t)

    def confidence(self,t):return self.filter.confidence(t)


class LegacyStereoCandidateSelector:
    def __init__(
        self,
        model: CameraModel,
        ball_radius_m: float,
        max_pairs: int = 64,
    ) -> None:
        self.model = model
        self.ball_radius = float(ball_radius_m)
        self.max_pairs = int(max_pairs)

    def select(
        self,
        left: Sequence[BallCandidate2D],
        right: Sequence[BallCandidate2D],
        predicted_position_left: Optional[np.ndarray],
        predicted_position_table: Optional[np.ndarray],
        T_left_table: Optional[np.ndarray],
        position_cov_table: Optional[np.ndarray],
        recent_rgb_position_left: Optional[np.ndarray],
    ) -> Optional[StereoBallMeasurement]:
        if not left or not right:
            return None
        best: Optional[StereoBallMeasurement] = None
        best_cost = float("inf")
        count = 0
        for lc in left:
            for rc in right:
                count += 1
                if count > self.max_pairs:
                    break
                tri = triangulate_rs_pair(lc.center, rc.center, self.model)
                if tri is None:
                    continue
                p_left, reproj = tri
                if reproj > 3.2 or not (0.20 <= p_left[2] <= 6.0):
                    continue

                # Physical-size consistency is deliberately weak because a fast ball
                # is an elongated streak rather than a clean circle.
                expected_radius = self.model.ir_left_intr.fx * self.ball_radius / max(p_left[2], 0.1)
                observed_radius = max(0.8, 0.5 * (lc.radius + rc.radius))
                size_log_error = abs(math.log(max(observed_radius, 0.8) / max(expected_radius, 0.8)))
                if size_log_error > 2.0:
                    continue

                table_gate = 0.0
                p_table: Optional[np.ndarray] = None
                if T_left_table is not None:
                    p_table = transform_point(invert_transform(T_left_table), p_left)
                    # Broad playing-volume gate. Ball may be before/after the table.
                    if not (
                        -2.2 <= p_table[0] <= 2.2
                        and -1.6 <= p_table[1] <= 1.6
                        and -0.15 <= p_table[2] <= 2.2
                    ):
                        continue
                    if predicted_position_table is not None:
                        diff = p_table - predicted_position_table
                        if position_cov_table is not None:
                            S = position_cov_table + np.eye(3) * 0.035**2
                            try:
                                table_gate = float(diff.T @ np.linalg.solve(S, diff))
                            except np.linalg.LinAlgError:
                                table_gate = float(np.dot(diff, diff) / 0.12**2)
                        else:
                            table_gate = float(np.dot(diff, diff) / 0.12**2)
                        if table_gate > 45.0:
                            continue

                pred_dist = (
                    float(np.linalg.norm(p_left - predicted_position_left))
                    if predicted_position_left is not None
                    else 0.25
                )
                rgb_dist = (
                    float(np.linalg.norm(p_left - recent_rgb_position_left))
                    if recent_rgb_position_left is not None
                    else 0.20
                )
                score2d = 0.5 * (lc.score + rc.score)
                cost = (
                    0.85 * reproj
                    + 0.80 * size_log_error
                    + 6.0 * pred_dist
                    + 3.0 * rgb_dist
                    + 0.055 * table_gate
                    - 1.5 * score2d
                )
                if cost < best_cost:
                    confidence = clamp(
                        0.48 * score2d
                        + 0.27 * math.exp(-reproj / 2.0)
                        + 0.15 * math.exp(-pred_dist / 0.15)
                        + 0.10 * math.exp(-size_log_error),
                        0.0,
                        1.0,
                    )
                    best = StereoBallMeasurement(
                        position_left_camera=p_left,
                        left_candidate=lc,
                        right_candidate=rc,
                        confidence=confidence,
                        reprojection_error_px=reproj,
                    )
                    best_cost = cost
            if count > self.max_pairs:
                break
        return best


class StereoCandidateSelector:
    """Reject geometrically invalid/ambiguous pairs before ranking survivors."""
    def __init__(self, model, ball_radius_m, cfg=None):
        self.model=model;self.ball_radius=ball_radius_m
        self.cfg=cfg or BallValidationConfig()
        self.diagnostics={}

    def select(self,left,right,predicted_position_left=None,predicted_position_table=None,
               T_left_table=None,position_cov_table=None,recent_rgb_position_left=None,
               *,timestamp_s=0.,rgb_observation=None,tracking=False,velocity_left=None,
               left_image=None,right_image=None):
        rejected={};eligible=[]
        def reject(reason): rejected[reason]=rejected.get(reason,0)+1
        rgb_time,rgb_candidates = rgb_observation or (None,[])
        age=timestamp_s-rgb_time if rgb_time is not None else float('inf')
        # Do not equate a past RGB pixel with the current 90 Hz IR position.
        fresh=abs(age)<=self.cfg.rgb_match_seconds and (tracking or abs(age)<=.012)
        for lc in left:
            for rc in right:
                tri=triangulate_rs_pair(lc.center,rc.center,self.model)
                if tri is None: reject('triangulation');continue
                p,reproj=tri
                if reproj>1.5 or not .2<=p[2]<=6.:
                    reject('stereo_geometry');continue
                widths=[lc.features.get('minor_px',2.*lc.radius),rc.features.get('minor_px',2.*rc.radius)]
                p_right=transform_point(self.model.T_left_right,p)
                expected=[self.model.ir_left_intr.fx*2*self.ball_radius/p[2],
                          self.model.ir_right_intr.fx*2*self.ball_radius/p_right[2]]
                errors=[size_check(o,e,self.cfg)[1] for o,e in zip(widths,expected)]
                if max(errors)>1. or max(widths)/max(min(widths),1.)>1.8:
                    reject('physical_size');continue
                if T_left_table is not None:
                    q=transform_point(invert_transform(T_left_table),p)
                    # A broad volume, not the table image silhouette.
                    if not (-2.8<q[0]<2.8 and -2.<q[1]<2. and -.12<q[2]<2.5):
                        reject('playing_volume');continue
                ncc=1.
                if left_image is not None and right_image is not None:
                    radius=int(np.clip(max(expected)*.75,4,16))
                    patches=[cv2.getRectSubPix(im,(2*radius+1,2*radius+1),tuple(map(float,c.center))).astype(float)
                             for im,c in ((left_image,lc),(right_image,rc))]
                    if any(np.std(x)<1.5 or np.ptp(x)<8. for x in patches):
                        reject('insufficient_ir_contrast');continue
                    if not fresh:
                        # Absolute temporal differences also outline shadows and
                        # the previous ball location. Without contemporaneous
                        # RGB identity, require a positive current-frame object
                        # in both eyes; trajectory proximity is not identity.
                        yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
                        rr=xx*xx+yy*yy
                        positive=[]
                        for patch,width in zip(patches,expected):
                            core=rr<=max(1.,.25*width)**2
                            ring=(rr>=max(2.,.60*width)**2)&(rr<=radius**2)
                            positive.append(bool(ring.any() and
                                np.median(patch[core])-np.median(patch[ring])>=3.))
                        if not all(positive):reject('stale_rgb_requires_positive_ir_object');continue
                    a,b=[x-x.mean() for x in patches]
                    denom=np.linalg.norm(a)*np.linalg.norm(b)
                    ncc=float(np.sum(a*b)/denom) if denom>1.e-6 else 0.
                    minimum_ncc=.55 if lc.source=='ir_guided' or rc.source=='ir_guided' else .30
                    if ncc<minimum_ncc: reject('stereo_appearance');continue
                identity_time=None;rgb_error=0.
                if fresh:
                    past=p.copy()
                    if velocity_left is not None: past=p-np.asarray(velocity_left)*age
                    color_point=transform_point(self.model.T_color_left,past)
                    uv=project_rs(self.model.color_intr,color_point)
                    matches=[]
                    if uv is not None:
                        for c in rgb_candidates:
                            if not c.validation.get('identity_ok'): continue
                            if not tracking and not c.validation.get('can_initialize_3d',True): continue
                            allowed=max(5.,.65*c.features['minor_px'])
                            # Residual timing/model uncertainty, not an arbitrary
                            # huge association radius for old RGB measurements.
                            allowed+=self.model.color_intr.fx*(2.*age*age)/max(color_point[2],.2)
                            dist=float(np.linalg.norm(c.center-uv))
                            size_ok,err=size_check(c.features['minor_px'],
                                self.model.color_intr.fx*2*self.ball_radius/color_point[2],self.cfg)
                            if c.validation.get('depth_source')=='aligned_depth':
                                if abs(c.validation['z_m']-color_point[2])>max(.06,.04*color_point[2]):
                                    continue
                            if dist<=allowed and size_ok: matches.append((dist/allowed,err))
                    if not matches: reject('rgb_identity_mismatch');continue
                    rgb_error=min(matches)[0];identity_time=rgb_time
                elif not tracking:
                    reject('fresh_rgb_identity_required');continue
                distance=float(np.linalg.norm(p-predicted_position_left)) if predicted_position_left is not None else 0.
                association_limit=.12
                if position_cov_table is not None:
                    association_limit=float(np.clip(.04+3.*math.sqrt(max(0.,np.trace(position_cov_table)/3.)),.07,.18))
                if tracking and distance>association_limit:
                    reject('prediction_mismatch');continue
                score=.5*(lc.score+rc.score)
                cost=.7*reproj+.5*max(errors)+.6*(1.-ncc)+.6*rgb_error+2.*distance-.4*score
                if cost>1.9: reject('insufficient_pair_evidence');continue
                confidence=float(np.clip(.5*score+.25*ncc+.25*(1.-min(1.,max(errors))),.1,.95))
                eligible.append((cost,StereoBallMeasurement(p,lc,rc,confidence,reproj,
                    identity_timestamp_s=identity_time,
                    validation=dict(size_error=max(errors),patch_ncc=ncc,rgb_match=fresh))))
        eligible.sort(key=lambda x:x[0])
        reason=None
        if len(eligible)>1:
            # Different hypotheses at nearly equal cost are not a confirmed ball.
            first=eligible[0][1]
            alternatives=[e for e in eligible[1:] if np.linalg.norm(e[1].position_left_camera-first.position_left_camera)>.035]
            if alternatives and alternatives[0][0]-eligible[0][0]<self.cfg.ambiguity_margin:
                reason='ambiguous_stereo';reject(reason)
        self.diagnostics=dict(rejected=rejected,eligible_pairs=len(eligible),reason=reason)
        return None if reason or not eligible else eligible[0][1]


# =============================================================================
# RGB/table worker shared state
# =============================================================================


@dataclass
class RGBMeasurement:
    timestamp_s: float
    position_left_camera: np.ndarray
    candidate: BallCandidate2D
    confidence: float
    depth_source: str


class SharedRGBState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.measurement: Optional[RGBMeasurement] = None
        self.debug_image: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.timestamp_s: Optional[float] = None
        self.candidates: List[BallCandidate2D] = []
        self.image_observation = None

    def set_candidates(self, timestamp_s, candidates, measurement, image_observation=None):
        # Candidates are immutable after publication. Publish all RGB evidence
        # atomically, including the selected independent-depth measurement.
        with self._lock:
            self.timestamp_s=timestamp_s
            self.candidates=list(candidates)
            self.measurement=measurement
            self.image_observation=copy.deepcopy(image_observation)

    def get_image_observation(self):
        with self._lock:
            return copy.deepcopy(self.image_observation)

    def get_candidates(self):
        with self._lock:
            return self.timestamp_s,list(self.candidates)

    def get_observation(self):
        with self._lock:
            return self.timestamp_s,list(self.candidates),self.measurement

    def set_measurement(self, measurement: Optional[RGBMeasurement]) -> None:
        with self._lock:
            self.measurement = measurement

    def get_measurement(self) -> Optional[RGBMeasurement]:
        with self._lock:
            if self.measurement is None:
                return None
            return RGBMeasurement(
                timestamp_s=self.measurement.timestamp_s,
                position_left_camera=self.measurement.position_left_camera.copy(),
                candidate=self.measurement.candidate,
                confidence=self.measurement.confidence,
                depth_source=self.measurement.depth_source,
            )

    def set_debug(self, image: np.ndarray, mask: np.ndarray) -> None:
        with self._lock:
            self.debug_image = image.copy()
            self.mask = mask.copy()

    def get_debug(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self.debug_image is None else self.debug_image.copy()


# =============================================================================
# Main system
# =============================================================================


@dataclass
class RuntimeStats:
    frame_count: int = 0
    measured_count: int = 0
    predicted_count: int = 0
    start_host_s: float = field(default_factory=time.monotonic)
    last_print_host_s: float = field(default_factory=time.monotonic)
    latest_fps: float = 0.0
    last_print_frame_count: int = 0

    def tick(self, measured: bool) -> None:
        self.frame_count += 1
        if measured:
            self.measured_count += 1
        else:
            self.predicted_count += 1
        now = time.monotonic()
        elapsed = now - self.last_print_host_s
        if elapsed >= 0.5:
            self.latest_fps = (self.frame_count - self.last_print_frame_count) / elapsed
            self.last_print_frame_count = self.frame_count
            self.last_print_host_s = now


class TableTennisPerceptionSystem:
    def __init__(self, args: argparse.Namespace) -> None:
        # RGB, table and IR already have independent workers. Nested OpenCV
        # pools (32 threads on this host) cause latency spikes on small images.
        cv2.setNumThreads(1)
        self.args = args
        self.stop_event = threading.Event()
        self.capture = D455AsyncCapture(
            serial=args.serial,
            requested_ir_fps=args.ir_fps,
            color_width=args.color_width,
            color_height=args.color_height,
            color_fps=args.color_fps,
            queue_size=args.capture_queue,
            emitter_enabled=args.emitter,
            ir_exposure_us=args.ir_exposure_us,
            color_exposure_us=args.color_exposure_us,
            ir_gain=args.ir_gain,
        )
        self.model: Optional[CameraModel] = None
        self.align: Optional[rs.align] = None
        self.capture.cancel_event = self.stop_event
        self.table_tracker: Optional[TablePoseTracker] = None
        self.rgb_detector: Optional[HSVBallDetector] = None
        self.left_ir_detector: Optional[IRBallDetector] = None
        self.right_ir_detector: Optional[IRBallDetector] = None
        self.selector: Optional[StereoCandidateSelector] = None
        self.camera_ball_tracker = CameraBallTracker()
        self.ball_validation = BallValidationConfig(
            confirm_observations=args.ball_confirm_observations,
            predict_seconds=args.ball_predict_seconds,
            identity_hold_seconds=args.ball_identity_hold_seconds)
        self.ball_gate = BallTrackGate(self.ball_validation)
        self.last_ball_payload = None
        self.ball_video_writer = None
        self._ball_video_start = None
        self._ball_video_frames = 0
        self.ball_trail = []
        self.last_admission_reason = 'no_measurement'
        self.ir_evidence_buffer = []
        self.last_rgb_evidence_time = None
        self.last_rgb_depth_time = None
        self.rgb_processed_frames = 0
        self.last_rgb_processed_time = None
        self.last_published_rgb_time = None

        self.rgb_state = SharedRGBState()
        self.color_thread: Optional[threading.Thread] = None
        self.table_thread: Optional[threading.Thread] = None
        self.table_queue = DropOldestQueue(maxsize=1)
        self.stats = RuntimeStats()
        self.last_left_candidates: List[BallCandidate2D] = []
        self.last_right_candidates: List[BallCandidate2D] = []
        self.last_left_binary: Optional[np.ndarray] = None
        self.last_right_binary: Optional[np.ndarray] = None
        self.last_table_pose_left: Optional[np.ndarray] = None
        self.last_packet: Optional[StereoPacket] = None
        self._ball_table_frame_id = 0
        self.table_video_writer = None
        self._table_video_start = None
        self._table_video_frames = 0

        self.zmq_context = zmq.Context.instance()
        self.pub_socket = self.zmq_context.socket(zmq.PUB)
        self.pub_socket.setsockopt(zmq.SNDHWM, 2)
        self.pub_socket.bind(f"tcp://*:{args.zmq_port}")

        self.jsonl_file = None
        if args.jsonl:
            path = Path(args.jsonl)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_file = path.open("a", encoding="utf-8")

    def start(self) -> None:
        self.model = self.capture.start()
        print(
            "[INFO] D455 started: "
            f"IR/depth {848}x{480}@{self.model.selected_ir_fps}, "
            f"RGB {self.model.selected_color_size[0]}x{self.model.selected_color_size[1]}@{self.args.color_fps}"
        )
        print(f"[INFO] ZMQ publisher: tcp://*:{self.args.zmq_port}")

        self.align = rs.align(rs.stream.color)
        self._initialize_rgb_detector()
        self.left_ir_detector = IRBallDetector(
            self.model.ir_left_intr, IRDetectorConfig(), "left"
        )
        self.right_ir_detector = IRBallDetector(
            self.model.ir_right_intr, IRDetectorConfig(), "right"
        )
        self.selector = StereoCandidateSelector(self.model, self.args.ball_radius, self.ball_validation)
        self.table_tracker = TablePoseTracker(
            color_intr=self.model.color_intr,
            depth_scale=self.model.depth_scale,
            table_length_m=self.args.table_length,
            table_width_m=self.args.table_width,
            hsv_ranges=table_hsv_ranges(self.args),
            pose_file=Path(self.args.table_pose_file),
            min_area=self.args.table_min_area,
            confirm_frames=self.args.table_confirm_frames,
            hold_seconds=self.args.table_hold_seconds,
            edge_tolerance_px=self.args.table_edge_tolerance,
            validation_hz=self.args.table_validation_hz,
        )

        # A saved pose is only an initial guess. The line/depth tracker must validate
        # and update it. Timestamp zero is acceptable because the next update resets
        # the prediction clock.
        if self.args.load_table_pose:
            loaded = self.table_tracker.load_initial_pose(0.0)
            print(f"[INFO] Loaded table-pose initial guess: {loaded}")

        self.table_thread = threading.Thread(target=self._table_worker,
            name="d455-table-worker", daemon=True)
        self.table_thread.start()
        self.color_thread = threading.Thread(
            target=self._color_worker,
            name="d455-color-table-worker",
            daemon=True,
        )
        self.color_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.capture.stop()
        if self.color_thread is not None and self.color_thread.is_alive():
            self.color_thread.join(timeout=5.0)
        if self.table_thread is not None and self.table_thread.is_alive():
            self.table_thread.join(timeout=5.0)
        if self.table_video_writer is not None:
            self.table_video_writer.release()
        if self.ball_video_writer is not None:
            self.ball_video_writer.release()
        if self.args.debug_dir:
            try:
                directory = Path(self.args.debug_dir)
                directory.mkdir(parents=True, exist_ok=True)
                images = {"rgb_detection.jpg": self.rgb_state.get_debug()}
                if self.table_tracker is not None:
                    images["table_pose.jpg"] = self.table_tracker.get_debug_image()
                if self.last_packet is not None:
                    images["ir_left.jpg"] = self.last_packet.ir_left
                    images["ir_right.jpg"] = self.last_packet.ir_right
                for name, image in images.items():
                    if image is not None and not cv2.imwrite(str(directory / name), image):
                        raise OSError(f"Could not write {directory / name}")
                print(f"\n[INFO] Final debug images: {directory}")
            except Exception as exc:
                print(f"[WARN] Could not save debug images: {exc}", file=sys.stderr)
        if self.jsonl_file is not None:
            self.jsonl_file.close()
        try:
            self.pub_socket.close(linger=0)
        except Exception:
            pass
        cv2.destroyAllWindows()

    def _table_snapshot_left(self, timestamp_s: float):
        if self.table_tracker is None or self.model is None:
            return None, {"valid": False, "state": "SEARCHING", "table_frame_id": 0}
        color_snapshot, metadata = self.table_tracker.snapshot(timestamp_s)
        if color_snapshot is None:
            return None, metadata
        T_left_table = self.model.T_left_color @ color_snapshot.T
        return PoseSnapshot(
            T=T_left_table,
            timestamp_s=color_snapshot.timestamp_s,
            linear_velocity=self.model.T_left_color[:3, :3] @ color_snapshot.linear_velocity,
            angular_velocity=self.model.T_left_color[:3, :3] @ color_snapshot.angular_velocity,
            confidence=color_snapshot.confidence,
            last_measurement_timestamp_s=color_snapshot.last_measurement_timestamp_s,
            metadata=metadata,
        ), metadata

    def _predict_table_left(self, timestamp_s: float) -> Optional[PoseSnapshot]:
        return self._table_snapshot_left(timestamp_s)[0]

    def _sync_table_frame(self, table_snapshot: Optional[PoseSnapshot]) -> None:
        if table_snapshot is not None:
            frame_id = table_snapshot.metadata.get("table_frame_id", 0)
            if frame_id != self._ball_table_frame_id:
                self._ball_table_frame_id = frame_id

    def _predict_ball_in_left(self, timestamp_s, table_snapshot_left):
        if not self.ball_gate.can_associate(timestamp_s):
            return None, None, None, None, None
        prediction=self.camera_ball_tracker.filter.predict_state(timestamp_s)
        if prediction is None:
            return None, None, None, None, None
        x,P=prediction
        if table_snapshot_left is None:
            return x[:3],None,None,P[:3,:3],None
        T=table_snapshot_left.T;R=T[:3,:3]
        position=transform_point(invert_transform(T),x[:3])
        velocity=R.T@x[3:]
        return x[:3],position,velocity,R.T@P[:3,:3]@R,np.r_[position,velocity]

    def _make_ir_rois(
        self,
        predicted_left: Optional[np.ndarray],
        timestamp_s: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
        if self.model is None:
            return None, None, None, None
        uv_l: Optional[np.ndarray] = None
        uv_r: Optional[np.ndarray] = None
        if predicted_left is not None:
            uv_l = project_rs(self.model.ir_left_intr, predicted_left)
            p_right = transform_point(self.model.T_left_right, predicted_left)
            uv_r = project_rs(self.model.ir_right_intr, p_right)

        tracking_age = self.camera_ball_tracker.age(timestamp_s)
        if not self.camera_ball_tracker.initialized or tracking_age > self.args.full_search_after:
            return uv_l, uv_r, None, None
        base = self.args.roi_radius
        grow = int(450.0 * max(0.0, tracking_age))
        radius = int(clamp(base + grow, base, self.args.max_roi_radius))
        roi_l = roi_around(uv_l, radius, self.model.ir_left_intr.width, self.model.ir_left_intr.height)
        roi_r = roi_around(uv_r, radius, self.model.ir_right_intr.width, self.model.ir_right_intr.height)
        return uv_l, uv_r, roi_l, roi_r

    def _recent_rgb_position(self, timestamp_s: float) -> Optional[np.ndarray]:
        measurement = self.rgb_state.get_measurement()
        if measurement is None:
            return None
        if abs(timestamp_s - measurement.timestamp_s) > 0.075:
            return None
        return measurement.position_left_camera

    def _color_worker(self) -> None:
        assert self.align is not None
        assert self.model is not None
        assert self.rgb_detector is not None
        assert self.table_tracker is not None
        table_counter = 0

        while not self.stop_event.is_set():
            try:
                packet: ColorFramesetPacket = self.capture.color_queue.get(timeout=0.20)
                # A slow table re-detection must not force ball verification to
                # process old queued RGB frames before the newest observation.
                latest_packet = self.capture.color_queue.drain_latest()
                if latest_packet is not None:
                    packet = latest_packet
            except queue.Empty:
                continue
            try:
                aligned = self.align.process(packet.frameset)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame:
                    continue
                bgr = np.asanyarray(color_frame.get_data()).copy()
                depth_raw = np.asanyarray(depth_frame.get_data()).copy() if depth_frame else None
                timestamp_s = float(color_frame.get_timestamp()) * 1e-3

                # Publish ball evidence before expensive table re-detection.
                self._process_rgb_frame(bgr, depth_raw, timestamp_s,
                                        float(depth_frame.get_timestamp()) * 1e-3 if depth_frame else None)
                table_counter += 1
                if depth_raw is not None and table_counter % max(1, self.args.table_update_stride) == 0:
                    self.table_queue.put_latest((bgr, depth_raw, timestamp_s))

            except Exception as exc:
                print(f"[WARN] RGB/table worker error: {exc}")
                time.sleep(0.01)

    def _table_worker(self):
        while not self.stop_event.is_set():
            try:
                bgr,depth,timestamp_s=self.table_queue.get(timeout=.2)
            except queue.Empty:
                continue
            try:
                self.table_tracker.update(bgr,depth,timestamp_s)
                if not self.args.table_video:
                    continue
                debug=self.table_tracker.get_debug_image()
                if debug is None:
                    continue
                if self.table_video_writer is None:
                    path=Path(self.args.table_video);path.parent.mkdir(parents=True,exist_ok=True)
                    self.table_video_writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),15.,(bgr.shape[1],bgr.shape[0]))
                    if not self.table_video_writer.isOpened():raise RuntimeError(f'Could not open table video: {path}')
                    self._table_video_start=timestamp_s
                target=int(max(0.,timestamp_s-self._table_video_start)*15)+1
                while self._table_video_frames<target:
                    self.table_video_writer.write(debug);self._table_video_frames+=1
            except Exception as exc:
                print(f'[WARN] Table worker error: {exc}')

    def _initialize_rgb_detector(self):
        mode=self.args.rgb_detector
        use_image=mode=='image' or (mode=='auto' and self.args.ball_color=='orange')
        ranges=ball_hsv_ranges(self.args.ball_color,self.args)
        if use_image:
            if self.args.ball_color=='orange' and self.args.ball_hsv_low is None and self.args.ball_hsv_high is None:
                ranges=[(np.array([2,75,65],np.uint8),np.array([35,255,255],np.uint8))]
            self.rgb_detector=ImageBallDetector(ranges,self.args.ball_color=='orange',retain_contours=True)
        else:
            self.rgb_detector=HSVBallDetector(HSVDetectorConfig(hsv_ranges=ranges))

    def _image_rgb_evidence(self,bgr,depth,timestamp_s):
        result,_,mask=self.rgb_detector.detect(bgr,timestamp_s)
        if not result['valid']:return [],mask,result
        c=self.rgb_detector.selected_candidate
        contour=c['contour']
        features=dict(minor_px=c['minor_px'],major_px=c['major_px'],aspect=c['aspect'],
            area_px=float(cv2.contourArea(contour)),circularity=c['circularity'],
            solidity=c['solidity'],surrounding_color=c['context'],clipped=c['clipped'],
            motion_ratio=c['motion'],mode='streak' if c['aspect']>1.7 else 'compact')
        candidate=BallCandidate2D(np.array(c['uv']),c['major_px']/2,tuple(c['bbox']),
            c['quality'],features['area_px'],c['solidity'],c['aspect'],'rgb_image',contour,features)
        evidence=depth_evidence(depth,self.model.depth_scale,contour)
        validation=dict(identity_ok=True,can_initialize_3d=bool(c['strong'] and not c['clipped']),
            rejection_reason=None,depth=evidence,depth_source='stereo_required',size_error=None,
            image_track_id=result['track_id'],image_strong=c['strong'])
        if evidence['reliable']:
            z=evidence['z_m']+self.args.ball_radius
            ok,error=size_check(c['minor_px'],self.model.color_intr.fx*2*self.args.ball_radius/z,
                self.ball_validation,evidence['mad_m'],z)
            validation.update(size_error=error,z_m=z)
            if ok:validation['depth_source']='aligned_depth'
            else:
                # Keep the independent 2D observation; this depth does not pass
                # the physical-size check and must not establish a 3D position.
                validation.update(identity_ok=False,rejection_reason='physical_size')
        candidate.validation=validation
        return [candidate],mask,result

    def _process_rgb_frame(self, bgr, depth_raw, timestamp_s, depth_timestamp_s=None):
        if self.last_rgb_processed_time is not None and timestamp_s<=self.last_rgb_processed_time:
            return []
        self.last_rgb_processed_time=timestamp_s
        self.rgb_processed_frames += 1
        depth_fresh = depth_raw is not None and (depth_timestamp_s is None or abs(timestamp_s-depth_timestamp_s) <= .012)
        validation_depth = depth_raw if depth_fresh else np.zeros(bgr.shape[:2],np.uint16)
        image_result=None
        if isinstance(self.rgb_detector,ImageBallDetector):
            candidates,mask,image_result=self._image_rgb_evidence(bgr,validation_depth,timestamp_s)
        else:
            candidates,mask=self.rgb_detector.detect(bgr)
        possible = []
        for c in candidates:
            if not isinstance(self.rgb_detector,ImageBallDetector):
                c.validation = validate_rgb_candidate(c, bgr, validation_depth, self.model.depth_scale,
                    self.model.color_intr.fx, self.args.ball_radius, self.ball_validation,
                    self.rgb_detector.cfg.hsv_ranges)
            v = c.validation
            if not depth_fresh:
                v['depth_source'] = 'stereo_required' if v['identity_ok'] else 'unavailable'
                v['depth']['reliable'] = False
                v['depth']['reason'] = 'depth_time_mismatch'
            if not v['identity_ok'] or v['depth_source'] != 'aligned_depth':
                continue
            # A streak is not a new RGB-only 3D identity; demand stereo evidence.
            if c.features['mode'] != 'compact' or c.features['circularity'] < .78 or not v.get('can_initialize_3d',True):
                c.validation['fallback_rejection'] = 'stereo_confirmation_required'
                continue
            z = v['z_m']
            point = transform_point(self.model.T_left_color,
                deproject_rs(self.model.color_intr, c.center, z))
            cost = .5 * v['size_error'] + .3 * c.features['surrounding_color'] - .5 * c.score
            possible.append((cost, RGBMeasurement(timestamp_s, point, c,
                min(.9, .5 + .4*c.score), 'aligned_depth')))
        possible.sort(key=lambda item: item[0])
        best = possible[0][1] if possible else None
        if len(possible) > 1 and possible[1][0]-possible[0][0] < self.ball_validation.ambiguity_margin:
            best = None
            for _, m in possible:
                m.candidate.validation['fallback_rejection'] = 'ambiguous_rgb'
        self.rgb_state.set_candidates(timestamp_s, candidates, best, image_result)
        debug = bgr.copy()
        if self.table_tracker is not None:
            snap, meta = self.table_tracker.snapshot(timestamp_s)
            if snap is not None:
                corners = self.table_tracker.corners @ snap.T[:3,:3].T + snap.T[:3,3]
                uv = self.table_tracker.camera.project(corners)
                if uv is not None and np.isfinite(uv).all():
                    uv = np.rint(np.clip(uv,-10000,10000)).astype(int)
                    for k in range(4):
                        ok,a,b = cv2.clipLine((0,0,bgr.shape[1],bgr.shape[0]),tuple(uv[k]),tuple(uv[(k+1)%4]))
                        if ok: cv2.line(debug,a,b,(255,180,0),2)
        # 2D identity and 3D depth validity are displayed separately.
        for c in candidates:
            if not c.validation['identity_ok'] and c.score < .58:
                continue
            color = (0,180,255) if c.validation['identity_ok'] else (100,100,180)
            center = tuple(np.rint(c.center).astype(int))
            cv2.circle(debug,center,max(3,int(c.radius)),color,1)
            reason = c.validation['rejection_reason'] or c.validation['depth_source']
            cv2.putText(debug,reason,(center[0]+5,center[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.36,color,1)
        latest = self.last_ball_payload
        state = 'SEARCHING'
        if latest is not None and abs(timestamp_s-latest['timestamp_s']) <= self.ball_validation.predict_seconds:
            ball = latest['ball'];state = ball['state']
            point = ball.get('position_camera_m')
            if ball['valid'] and point is not None:
                history=latest.get('render_history',[])
                eligible=[e for e in history if e['timestamp_s']<=timestamp_s]
                if eligible:
                    e=eligible[-1];dt=timestamp_s-e['timestamp_s']
                    xyz=np.asarray(e['position'])+dt*np.asarray(e['velocity'])
                else:
                    xyz=np.asarray(point)
                uv = project_rs(self.model.color_intr,transform_point(self.model.T_color_left,xyz))
                if uv is not None and np.isfinite(uv).all() and np.max(np.abs(uv)) < 10000:
                    center = tuple(np.rint(uv).astype(int))
                    color = (0,255,0) if state=='CONFIRMED' else (255,0,255)
                    cv2.circle(debug,center,12,color,2)
                    self.ball_trail.append((timestamp_s,center,ball['track_id']))
        self.ball_trail = [item for item in self.ball_trail if timestamp_s-item[0] < .6]
        for a,b in zip(self.ball_trail,self.ball_trail[1:]):
            if a[2]==b[2]: cv2.line(debug,a[1],b[1],(0,255,255),1)
        if image_result is not None and image_result['valid']:
            uv=tuple(np.rint(image_result['uv']).astype(int))
            cv2.drawMarker(debug,uv,(255,255,0),cv2.MARKER_CROSS,18,2)
        cv2.rectangle(debug,(0,0),(min(850,bgr.shape[1]-1),48),(20,20,20),-1)
        image_state='DETECTED' if image_result and image_result['valid'] else 'MISSING'
        cv2.putText(debug,'2D: '+image_state+' | 3D: '+state+' | '+self.last_admission_reason,(12,20),
                    cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        cv2.putText(debug,'cyan=2D measured  green=3D confirmed  purple=3D predicted',(12,40),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,(230,230,230),1)
        self.rgb_state.set_debug(debug,mask)
        if self.args.ball_video:
            if self.ball_video_writer is None:
                path=Path(self.args.ball_video);path.parent.mkdir(parents=True,exist_ok=True)
                self.ball_video_writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),15.,(bgr.shape[1],bgr.shape[0]))
                if not self.ball_video_writer.isOpened(): raise RuntimeError('Cannot open ball overlay video')
                self._ball_video_start=timestamp_s
            target=int(max(0.,timestamp_s-self._ball_video_start)*15)+1
            while self._ball_video_frames<target:
                self.ball_video_writer.write(debug);self._ball_video_frames+=1
        return candidates

    def _admit_ball_measurement(self, point, timestamp_s, key, confidence, identity_time, sigma, table_left):
        tracker=self.camera_ball_tracker
        late=(self.ball_gate.last_measurement is not None and timestamp_s<=self.ball_gate.last_measurement)
        prediction=tracker.filter.predict_state(timestamp_s)
        if late:
            if prediction is None or not self.ball_gate.can_associate(tracker.filter.timestamp_s):
                self.last_admission_reason='outside_history';return False
            if np.linalg.norm(np.asarray(point)-prediction[0][:3])>.18:
                self.last_admission_reason='historical_position_mismatch';return False
            new_track=False
        else:
            decision=self.ball_gate.propose(point,timestamp_s,key,identity_time,sigma,
                None if prediction is None else prediction[0][:3],
                None if prediction is None else prediction[1][:3,:3])
            self.last_admission_reason=decision.reason
            if not decision.accepted:return False
            new_track=decision.new_track
        before=copy.deepcopy(tracker.__dict__)
        source='rgb_depth_verified' if key[0]=='rgb' else 'ir_stereo'
        baseline=float(np.linalg.norm(self.model.T_left_right[:3,3]))
        covariance=measurement_covariance(point,source,self.model.ir_left_intr.fx,baseline,sigma)
        if new_track:
            tracker.reset()
            for n,(t,p) in enumerate(list(self.ball_gate.pending)[:-1]):
                tracker.update(p,t,confidence,covariance,('seed',n),'seed')
        if not tracker.update(point,timestamp_s,confidence,covariance,key,source):
            reason=tracker.filter.last_reason
            counters=tracker.filter.counters.copy()
            tracker.__dict__.clear();tracker.__dict__.update(before)
            tracker.filter.counters=counters
            self.last_admission_reason=reason
            return False
        reason=tracker.filter.last_reason
        if late:
            # The current clock never moves back to the late frame. The filter
            # has replayed it and all subsequent measurements in sensor order.
            latest=tracker.filter.events[-1]
            self.ball_gate.position=latest.p.copy()
            self.ball_gate.last_measurement=latest.t
            self.ball_gate.velocity=tracker.velocity.copy()
            self.ball_gate.refresh_identity(identity_time) if identity_time is not None else None
            self.ball_gate.state='CONFIRMED';self.ball_gate.reason='accepted_late_measurement'
        else:
            self.ball_gate.commit(point,timestamp_s,identity_time,new_track,tracker.velocity)
        self.last_admission_reason='independently_confirmed' if new_track else reason
        return True

    def _refresh_rgb_identity(self, rgb_t, candidates, now):
        if not self.ball_gate.can_associate(now) or not 0<=now-rgb_t<=.10:
            return False
        pred=self.camera_ball_tracker.filter.predict_state(rgb_t)
        if pred is None:return False
        point=transform_point(self.model.T_color_left,pred[0][:3])
        uv=project_rs(self.model.color_intr,point)
        if uv is None:return False
        matches=[]
        for c in candidates:
            if not c.validation.get('identity_ok'):continue
            if not size_check(c.features.get('minor_px',2*c.radius),
                    self.model.color_intr.fx*2*self.args.ball_radius/point[2],self.ball_validation)[0]:continue
            if np.linalg.norm(c.center-uv)<=max(5.,.65*c.features.get('minor_px',2*c.radius)):
                matches.append(c)
        if len(matches)!=1:return False
        # This renews identity only: it neither adds a 3D measurement nor a birth
        # confirmation, and 3D output still expires after predict_seconds.
        self.ball_gate.refresh_identity(rgb_t)
        return True

    def _process_ball_packet(self, packet, table_left):
        timestamp_s=packet.timestamp_s
        pred_left,_,_,_,_=self._predict_ball_in_left(timestamp_s,table_left)
        uv_l,uv_r,roi_l,roi_r=self._make_ir_rois(pred_left,timestamp_s)
        left,left_binary=self.left_ir_detector.detect(packet.ir_left,roi_l,uv_l,None if table_left is None else table_left.T)
        right,right_binary=self.right_ir_detector.detect(packet.ir_right,roi_r,uv_r,None if table_left is None else table_left.T)
        self.last_left_candidates=left;self.last_right_candidates=right
        self.last_left_binary=left_binary;self.last_right_binary=right_binary
        rgb_t,rgb_candidates,rgb=self.rgb_state.get_observation()
        rgb_observation=(rgb_t,rgb_candidates)
        if rgb_t is not None and abs(timestamp_s-rgb_t)<=.012:
            left=merge_ir_candidates(rgb_guided_ir_candidates(packet.ir_left,rgb_candidates,self.model),left)
            right=merge_ir_candidates(rgb_guided_ir_candidates(packet.ir_right,rgb_candidates,self.model,True),right)
            self.last_left_candidates=left;self.last_right_candidates=right
        self.ir_evidence_buffer.append(dict(packet=packet,left=left,right=right,
            prediction=None if pred_left is None else pred_left.copy(),
            track_id=self.ball_gate.track_id if self.ball_gate.can_associate(timestamp_s) else None))
        self.ir_evidence_buffer=[e for e in self.ir_evidence_buffer if timestamp_s-e['packet'].timestamp_s<=.20][-24:]
        measured=False;source='prediction_only';measurement=None
        new_rgb=rgb_t is not None and rgb_t!=self.last_rgb_evidence_time and 0<=timestamp_s-rgb_t<=.10
        if new_rgb:
            self.last_rgb_evidence_time=rgb_t
            self._refresh_rgb_identity(rgb_t,rgb_candidates,timestamp_s)
            if rgb is not None:
                measured=self._admit_ball_measurement(rgb.position_left_camera,rgb.timestamp_s,
                    ('rgb',rgb.timestamp_s),rgb.confidence,rgb.timestamp_s,.035,table_left)
                if measured:source='rgb_depth_verified'
            if not measured:
                pairs=[e for e in self.ir_evidence_buffer if abs(e['packet'].timestamp_s-rgb_t)<=.012]
                for past in sorted(pairs,key=lambda e:abs(e['packet'].timestamp_s-rgb_t))[:3]:
                    old=past['packet']
                    if old.frame_number==packet.frame_number:continue
                    # RGB often arrives after the corresponding IR pair. Search
                    # its saved raw images too, not only the old tracking ROI.
                    old_left=merge_ir_candidates(rgb_guided_ir_candidates(old.ir_left,rgb_candidates,self.model),past['left'])
                    old_right=merge_ir_candidates(rgb_guided_ir_candidates(old.ir_right,rgb_candidates,self.model,True),past['right'])
                    historical=self.selector.select(old_left,old_right,timestamp_s=old.timestamp_s,
                        T_left_table=None if table_left is None else table_left.T,
                        rgb_observation=rgb_observation,tracking=False,
                        left_image=old.ir_left,right_image=old.ir_right)
                    if historical is None:continue
                    if self._admit_ball_measurement(historical.position_left_camera,old.timestamp_s,
                        ('ir',old.frame_number),historical.confidence,rgb_t,
                        max(.025,.008+.012*historical.position_left_camera[2]**2),table_left):
                        measured=True;source='ir_stereo_delayed';measurement=historical
                    break
        # First use newly arrived evidence, then expire the 3D output clock.
        if self.ball_gate.tick(timestamp_s):self.camera_ball_tracker.reset()
        pred_left,pred_table,_,pred_cov,_=self._predict_ball_in_left(timestamp_s,table_left)
        camera_pred=self.camera_ball_tracker.predict(timestamp_s)
        tracking=(self.ball_gate.can_associate(timestamp_s) and self.ball_gate.last_identity is not None
                  and 0<=timestamp_s-self.ball_gate.last_identity<=self.ball_validation.identity_hold_seconds)
        current=self.selector.select(left,right,pred_left,pred_table,
            None if table_left is None else table_left.T,pred_cov,
            timestamp_s=timestamp_s,rgb_observation=rgb_observation,tracking=tracking,
            velocity_left=None if camera_pred is None else camera_pred[1],
            left_image=packet.ir_left,right_image=packet.ir_right)
        if current is not None:
            if self._admit_ball_measurement(current.position_left_camera,timestamp_s,('ir',packet.frame_number),
                current.confidence,current.identity_timestamp_s,
                max(.012,.008+.012*current.position_left_camera[2]**2),table_left):
                measured=True;source='ir_stereo';measurement=current
        elif not measured and not new_rgb:
            self.last_admission_reason=self.selector.diagnostics.get('reason') or 'no_verified_measurement'
        if measured and self.ball_gate.valid(timestamp_s):self.ball_gate.state='CONFIRMED'
        return measured,source,measurement,(uv_l,uv_r,roi_l,roi_r)

    def _publish(
        self,
        packet: StereoPacket,
        table_left: Optional[PoseSnapshot],
        measured: bool,
        measurement_source: str,
        stereo_measurement: Optional[StereoBallMeasurement],
        table_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        timestamp_s = packet.timestamp_s
        track_valid=self.ball_gate.valid(timestamp_s)
        camera_valid=bool(track_valid and self.camera_ball_tracker.initialized)
        world_valid=bool(camera_valid and table_left is not None)
        ball_payload: Dict[str, Any] = {
            **self.ball_gate.metadata(timestamp_s),
            "admission_reason": self.last_admission_reason,
            "position_camera_m": None,
            "velocity_camera_mps": None,
            "T_camera_ball": None,
            "valid": bool(world_valid or camera_valid),
            "world_valid": world_valid,
            "camera_valid": camera_valid,
            "measured_this_frame": measured,
            "measurement_source": measurement_source,
            "table_frame_id": (table_metadata or {}).get("table_frame_id", 0),
            "position_table_m": None,
            "velocity_table_mps": None,
            "T_table_ball": None,
            "measurement_age_s": (
                self.camera_ball_tracker.age(timestamp_s)
            ),
            "confidence": (
                self.camera_ball_tracker.confidence(timestamp_s)
            ),
        }
        table_payload: Dict[str, Any] = {
            **(table_metadata or {}),
            "valid": table_left is not None,
            "T_camera_table": None,
            "T_table_camera": None,
            "position_camera_m": None,
            "frame_definition": {
                "origin": "table-top centre",
                "x": "robot side toward opponent",
                "y": "robot left",
                "z": "up",
            },
        }

        if table_left is not None:
            stale = max(0.0, timestamp_s - table_left.last_measurement_timestamp_s)
            table_payload.update(
                {
                    "confidence": table_left.confidence,
                    "stale_s": stale,
                    "T_camera_table": transform_to_list(table_left.T),
                    "T_table_camera": transform_to_list(invert_transform(table_left.T)),
                    "position_camera_m": table_left.T[:3, 3].tolist(),
                    "rotation_camera_table": table_left.T[:3, :3].tolist(),
                    "quaternion_camera_table_xyzw": matrix_to_quaternion_xyzw(table_left.T[:3, :3]),
                }
            )

        camera_position: Optional[np.ndarray] = None
        camera_velocity: Optional[np.ndarray] = None

        if camera_valid:
            predicted_x,predicted_P=self.camera_ball_tracker.filter.predict_state(timestamp_s)
            camera_position=predicted_x[:3].copy();camera_velocity=predicted_x[3:].copy()
            ball_payload['position_covariance_camera']=predicted_P[:3,:3].tolist()
            ball_payload['motion_model']='camera_constant_velocity_no_contact'
            if world_valid:
                R=table_left.T[:3,:3]
                p_table=transform_point(invert_transform(table_left.T),camera_position)
                ball_payload['position_table_m']=p_table.tolist()
                ball_payload['velocity_table_mps']=(R.T@camera_velocity).tolist()
                ball_payload['T_table_ball']=transform_to_list(make_transform(np.eye(3),p_table))
                ball_payload['position_covariance_table']=(R.T@predicted_P[:3,:3]@R).tolist()

        if camera_position is not None and camera_velocity is not None:
            table_z = table_left.T[:3, 2] if table_left is not None else None
            R_motion = ball_motion_rotation(camera_velocity, table_z)
            T_camera_ball = make_transform(R_motion, camera_position)
            ball_payload.update(
                {
                    "position_camera_m": camera_position.tolist(),
                    "velocity_camera_mps": camera_velocity.tolist(),
                    "T_camera_ball": transform_to_list(T_camera_ball),
                    "rotation_camera_ball_motion": R_motion.tolist(),
                    "quaternion_camera_ball_motion_xyzw": matrix_to_quaternion_xyzw(R_motion),
                    "orientation_semantics": "motion frame; x follows velocity; not ball spin",
                }
            )

        if stereo_measurement is not None:
            ball_payload["stereo_reprojection_error_px"] = stereo_measurement.reprojection_error_px
            ball_payload["left_uv"] = stereo_measurement.left_candidate.center.tolist()
            ball_payload["right_uv"] = stereo_measurement.right_candidate.center.tolist()

        if not math.isfinite(ball_payload["measurement_age_s"]):
            ball_payload["measurement_age_s"] = None
        image_observation=self.rgb_state.get_image_observation()
        image_payload=dict(valid=False,measured=False,uv=None,timestamp_s=None,
            camera_frame='d455_color_optical_frame',depth_valid=False,reason='no_rgb_observation',
            new_observation=False)
        if image_observation is not None:
            age=timestamp_s-image_observation['timestamp_s']
            fresh=0<=age<=.10
            image_payload.update(image_observation,age_s=age,fresh=fresh,
                camera_frame='d455_color_optical_frame',
                new_observation=bool(fresh and image_observation['timestamp_s']!=self.last_published_rgb_time))
            # Coordinates belong to their RGB timestamp, never the current IR
            # timestamp. Repeated publication is not a new image measurement.
            if not fresh:
                image_payload.update(valid=False,measured=False,uv=None,reason='stale_rgb_observation')
            else:self.last_published_rgb_time=image_observation['timestamp_s']
        payload: Dict[str, Any] = {
            "timestamp_s": timestamp_s,
            "host_timestamp_s": packet.host_timestamp_s,
            "frame_number": packet.frame_number,
            "camera_frame": "d455_left_ir_optical_frame",
            "ball": ball_payload,
            "ball_2d": image_payload,
            "table": table_payload,
            "diagnostics": {
                "processing_fps": self.stats.latest_fps,
                "selected_ir_fps": self.model.selected_ir_fps if self.model else None,
                "hardware_ir_frame_gaps": self.capture.hardware_ir_drops,
                "stereo_queue_overwrites": self.capture.stereo_queue.dropped,
                "color_queue_overwrites": self.capture.color_queue.dropped,
                "capture_callback_errors": self.capture.callback_errors,
                "rgb_processed_frames": getattr(self,'rgb_processed_frames',0),
                "table_queue_overwrites": self.table_queue.dropped if hasattr(self,'table_queue') else 0,
                "rgb_evidence_age_s": (None if self.rgb_state.get_candidates()[0] is None else
                    max(0.,timestamp_s-self.rgb_state.get_candidates()[0])),
                "ball_validation": self.selector.diagnostics,
                "measurement_fusion": self.camera_ball_tracker.filter.counters,
                "rgb_candidates": [{"uv":c.center.tolist(),
                    "identity_ok":c.validation.get('identity_ok',False),
                    "reason":c.validation.get('rejection_reason'),
                    "depth_source":c.validation.get('depth_source')}
                    for c in self.rgb_state.get_candidates()[1]],
            },
        }
        self.last_ball_payload={**payload,'render_history':[
            dict(timestamp_s=t,position=x[:3].tolist(),velocity=x[3:].tolist())
            for x,P,t in self.camera_ball_tracker.filter.states]}
        self.pub_socket.send_pyobj(payload, flags=zmq.NOBLOCK)
        if self.jsonl_file is not None:
            self.jsonl_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.jsonl_file.flush()
        return payload

    def _visualize(
        self,
        packet: StereoPacket,
        uv_l_pred: Optional[np.ndarray],
        uv_r_pred: Optional[np.ndarray],
        roi_l: Optional[Tuple[int, int, int, int]],
        roi_r: Optional[Tuple[int, int, int, int]],
        measurement: Optional[StereoBallMeasurement],
        payload: Dict[str, Any],
    ) -> bool:
        left = cv2.cvtColor(packet.ir_left, cv2.COLOR_GRAY2BGR)
        right = cv2.cvtColor(packet.ir_right, cv2.COLOR_GRAY2BGR)
        if roi_l is not None:
            cv2.rectangle(left, (roi_l[0], roi_l[1]), (roi_l[2], roi_l[3]), (255, 255, 0), 1)
        if roi_r is not None:
            cv2.rectangle(right, (roi_r[0], roi_r[1]), (roi_r[2], roi_r[3]), (255, 255, 0), 1)
        if uv_l_pred is not None:
            cv2.drawMarker(left, tuple(np.round(uv_l_pred).astype(int)), (255, 0, 255), cv2.MARKER_CROSS, 12, 1)
        if uv_r_pred is not None:
            cv2.drawMarker(right, tuple(np.round(uv_r_pred).astype(int)), (255, 0, 255), cv2.MARKER_CROSS, 12, 1)
        for c in self.last_left_candidates[:6]:
            cv2.circle(left, tuple(np.round(c.center).astype(int)), max(2, int(round(c.radius))), (0, 180, 255), 1)
        for c in self.last_right_candidates[:6]:
            cv2.circle(right, tuple(np.round(c.center).astype(int)), max(2, int(round(c.radius))), (0, 180, 255), 1)
        if measurement is not None and payload['ball']['measured_this_frame'] and payload['ball']['measurement_source']=='ir_stereo':
            cv2.circle(left, tuple(np.round(measurement.left_candidate.center).astype(int)), max(3, int(round(measurement.left_candidate.radius))), (0, 255, 0), 2)
            cv2.circle(right, tuple(np.round(measurement.right_candidate.center).astype(int)), max(3, int(round(measurement.right_candidate.radius))), (0, 255, 0), 2)

        ball = payload["ball"]
        text = f"FPS {self.stats.latest_fps:.1f} | {ball.get('state','SEARCHING')} | {ball.get('measurement_source','none')}"
        cv2.putText(left, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
        if ball.get("position_table_m") is not None:
            p = ball["position_table_m"]
            cv2.putText(left, f"table xyz {p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1)

        scale = self.args.vis_scale
        cv2.imshow("IR left", cv2.resize(left, None, fx=scale, fy=scale))
        cv2.imshow("IR right", cv2.resize(right, None, fx=scale, fy=scale))
        rgb_debug = self.rgb_state.get_debug()
        if rgb_debug is not None:
            cv2.imshow("RGB HSV", cv2.resize(rgb_debug, None, fx=scale, fy=scale))
        if self.table_tracker is not None:
            table_debug = self.table_tracker.get_debug_image()
            if table_debug is not None:
                cv2.imshow("Table pose", cv2.resize(table_debug, None, fx=scale, fy=scale))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key == ord("r"):
            self.ball_gate.reset()
            self.ir_evidence_buffer.clear()
            self.last_rgb_evidence_time = self.rgb_state.get_candidates()[0]
            self.camera_ball_tracker.reset()
            if self.left_ir_detector:
                self.left_ir_detector.reset()
            if self.right_ir_detector:
                self.right_ir_detector.reset()
            print("[INFO] Ball tracker reset")
        elif key == ord("t") and self.table_tracker is not None:
            self.table_tracker.request_reinitialize()
            print("[INFO] Forced table re-initialization")
        elif key == ord("s") and self.table_tracker is not None:
            print(f"[INFO] Saved table pose: {self.table_tracker.save_pose()}")
        return True

    def run(self) -> None:
        assert self.model is not None
        assert self.left_ir_detector is not None
        assert self.right_ir_detector is not None
        assert self.selector is not None

        started = time.monotonic()
        last_frame_host_s = started
        while not self.stop_event.is_set():
            if self.args.duration > 0 and time.monotonic() - started >= self.args.duration:
                if self.stats.frame_count == 0:
                    raise RuntimeError("Duration elapsed without receiving any D455 stereo frames")
                print(f"\n[INFO] Duration reached: {self.args.duration:g} seconds")
                break
            try:
                packet: StereoPacket = self.capture.stereo_queue.get(timeout=0.30)
            except queue.Empty:
                if time.monotonic() - last_frame_host_s >= self.args.frame_timeout:
                    raise RuntimeError(
                        f"No D455 stereo frames for {self.args.frame_timeout:g}s; "
                        "check camera/USB connection and competing camera processes"
                    )
                continue

            last_frame_host_s = time.monotonic()
            self.last_packet = packet
            timestamp_s = packet.timestamp_s
            table_left, table_metadata = self._table_snapshot_left(timestamp_s)
            self._sync_table_frame(table_left)
            self.last_table_pose_left = None if table_left is None else table_left.T.copy()
            measured,source,measurement,visuals=self._process_ball_packet(packet,table_left)
            uv_l_pred,uv_r_pred,roi_l,roi_r=visuals
            self.stats.tick(measured)

            try:
                payload = self._publish(
                    packet,
                    table_left,
                    measured,
                    source,
                    measurement if measured and source == "ir_stereo" else None,
                    table_metadata,
                )
            except zmq.Again:
                # A slow subscriber must never block the perception loop.
                payload = {
                    "ball": {
                        "measurement_source": source,
                        "confidence": self.camera_ball_tracker.confidence(timestamp_s),
                    }
                }

            if self.args.visualize:
                if not self._visualize(
                    packet,
                    uv_l_pred,
                    uv_r_pred,
                    roi_l,
                    roi_r,
                    measurement,
                    payload,
                ):
                    self.stop_event.set()
                    break

            now = time.monotonic()
            if now - self.stats.last_print_host_s < 0.05 and self.stats.frame_count % 5 == 0:
                ball_conf = self.camera_ball_tracker.confidence(timestamp_s)
                table_conf = table_left.confidence if table_left is not None else 0.0
                sys.stdout.write(
                    "\r"
                    f"[TRACK] {self.stats.latest_fps:5.1f} FPS | "
                    f"ball={source:18s} conf={ball_conf:.2f} | "
                    f"table={table_conf:.2f} | "
                    f"hw_gaps={self.capture.hardware_ir_drops} "
                    f"queue_drop={self.capture.stereo_queue.dropped}     "
                )
                sys.stdout.flush()


# =============================================================================
# CLI configuration
# =============================================================================


def parse_hsv_triplet(text: str) -> np.ndarray:
    values = [int(v.strip()) for v in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("HSV triplet must be H,S,V")
    return np.asarray(values, dtype=np.uint8)


def ball_hsv_ranges(
    preset: str, args: argparse.Namespace
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if args.ball_hsv_low is not None and args.ball_hsv_high is not None:
        return [(args.ball_hsv_low, args.ball_hsv_high)]
    if preset == "orange":
        return [
            (np.array([2, 75, 65], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8))
        ]
    if preset == "white":
        return [
            (np.array([0, 0, 155], dtype=np.uint8), np.array([179, 95, 255], dtype=np.uint8))
        ]
    if preset == "yellow":
        return [
            (np.array([18, 65, 70], dtype=np.uint8), np.array([42, 255, 255], dtype=np.uint8))
        ]
    raise ValueError(f"Unknown ball colour preset: {preset}")


def table_hsv_ranges(args: argparse.Namespace) -> List[Tuple[np.ndarray, np.ndarray]]:
    if args.table_hsv_low is not None and args.table_hsv_high is not None:
        return [(args.table_hsv_low, args.table_hsv_high)]
    # Broad blue/cyan range; tune from actual D455 RGB images.
    return [
        (np.array([78, 28, 22], dtype=np.uint8), np.array([145, 255, 255], dtype=np.uint8))
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="D455 high-rate HSV/IR stereo table-tennis tracker"
    )
    parser.add_argument("--serial", default=None, help="D455 serial number")
    parser.add_argument("--ir-fps", type=int, default=90, help="Requested IR/depth FPS; falls back to 60/30")
    parser.add_argument("--color-width", type=int, default=1280)
    parser.add_argument("--color-height", type=int, default=720)
    parser.add_argument("--color-fps", type=int, default=30)
    parser.add_argument("--capture-queue", type=int, default=4)
    parser.add_argument("--emitter", type=int, choices=[0, 1], default=1, help="D455 IR emitter; 1 improves table depth, 0 may give cleaner passive IR motion images")
    parser.add_argument("--ir-exposure-us", type=float, default=800.0, help="0 enables auto exposure; shorter reduces ball blur")
    parser.add_argument("--ir-gain", type=float, default=None, help="Manual IR gain; omitted preserves sensor setting. Compare signal and noise before choosing a default.")
    parser.add_argument("--color-exposure-us", type=float, default=2500.0, help="0 enables auto exposure")

    parser.add_argument("--ball-color", choices=["orange", "white", "yellow"], default="orange")
    parser.add_argument("--rgb-detector", choices=["auto","image","hsv"], default="auto",
                        help="auto uses the image detector for orange, legacy HSV for other colours")
    parser.add_argument("--ball-hsv-low", type=parse_hsv_triplet, default=None)
    parser.add_argument("--ball-hsv-high", type=parse_hsv_triplet, default=None)
    parser.add_argument("--ball-radius", type=float, default=0.020)
    parser.add_argument("--ball-confirm-observations", type=int, default=3)
    parser.add_argument("--ball-predict-seconds", type=float, default=.12)
    parser.add_argument("--ball-identity-hold-seconds", type=float, default=.15)
    parser.add_argument("--ball-video", default=None, help="Combined table, ball identity and track-state overlay MP4")
    parser.add_argument("--roi-radius", type=int, default=85, help="IR tracking ROI half-size in pixels")
    parser.add_argument("--max-roi-radius", type=int, default=230)
    parser.add_argument("--rgb-roi-radius", type=int, default=180)
    parser.add_argument("--full-search-after", type=float, default=0.10, help="Seconds without accepted measurement before full-frame search")

    parser.add_argument("--table-length", type=float, default=2.74)
    parser.add_argument("--table-width", type=float, default=1.525)
    parser.add_argument("--table-min-area", type=int, default=0, help="Minimum blue component area in pixels; 0 uses the original adaptive threshold")
    parser.add_argument("--table-hsv-low", type=parse_hsv_triplet, default=None)
    parser.add_argument("--table-hsv-high", type=parse_hsv_triplet, default=None)
    parser.add_argument("--table-update-stride", type=int, default=1, help="Run table tracker every N RGB frames")
    parser.add_argument("--table-confirm-frames", type=int, default=5, help="Independent consistent observations before locking a fixed table pose")
    parser.add_argument("--table-hold-seconds", type=float, default=0.6, help="Historical display grace period; unverified world coordinates are invalid")
    parser.add_argument("--table-edge-tolerance", type=float, default=5.0, help="Maximum boundary support distance in pixels")
    parser.add_argument("--table-validation-hz", type=float, default=12.0, help="Maximum independent table validation frequency")
    parser.add_argument("--table-video", default=None, help="Record table contour/status overlay MP4")
    parser.add_argument("--table-pose-file", default="table_pose_initial_guess.npz")
    parser.add_argument("--load-table-pose", action="store_true")

    parser.add_argument("--table-restitution", type=float, default=0.86, help="Reserved; automatic contact prediction is disabled")
    parser.add_argument("--horizontal-restitution", type=float, default=0.93, help="Reserved; automatic contact prediction is disabled")
    parser.add_argument("--zmq-port", type=int, default=5555)
    parser.add_argument("--jsonl", default=None, help="Optional output JSONL log")
    parser.add_argument("--duration", type=float, default=0.0, help="Run for this many seconds after camera startup; 0 runs until stopped")
    parser.add_argument("--frame-timeout", type=float, default=5.0, help="Fail if no stereo frames arrive for this many seconds")
    parser.add_argument("--debug-dir", default=None, help="Save final RGB/table/IR debug images on shutdown")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis-scale", type=float, default=0.65)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.ir_gain is not None and (not math.isfinite(args.ir_gain) or args.ir_gain<=0):
        parser.error("ir-gain must be finite and positive")
    if not math.isfinite(args.duration) or args.duration < 0 or args.table_min_area < 0:
        parser.error("duration must be finite and nonnegative; table-min-area must be nonnegative")
    if not math.isfinite(args.frame_timeout) or args.frame_timeout <= 0:
        parser.error("frame-timeout must be finite and positive")
    if args.table_confirm_frames < 2 or any(not math.isfinite(v) or v <= 0 for v in (args.table_hold_seconds, args.table_edge_tolerance, args.table_validation_hz)):
        parser.error("table-confirm-frames must be >=2 and table timing/tolerance options must be finite and positive")
    if args.ball_confirm_observations < 2 or any(not math.isfinite(v) or v<=0 for v in
        (args.ball_radius,args.ball_predict_seconds,args.ball_identity_hold_seconds)):
        parser.error("ball confirmation count must be >=2 and ball radius/timing must be finite and positive")
    args.emitter = None if args.emitter is None else bool(args.emitter)
    system = TableTennisPerceptionSystem(args)

    def handle_signal(_signum: int, _frame: Any) -> None:
        system.stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        system.start()
        system.run()
        return 0
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        system.stop()
        print("\n[INFO] Tracker stopped")


if __name__ == "__main__":
    raise SystemExit(main())
