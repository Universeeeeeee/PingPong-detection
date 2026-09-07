"""Rule-based fixed-table registration for rectified ZED colour and depth frames.

This deliberately requires a complete, visible outer table boundary during
initialization.  It uses table colour only to propose pixels; metric ZED depth
and the known physical table dimensions decide whether the proposal is valid.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from camera_types import CameraIntrinsics, DepthFrame, DetectionFrame


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(first.T @ second) - 1.0) / 2.0, -1.0, 1.0)))


@dataclass(frozen=True)
class ZedTablePoseSnapshot:
    """A table-frame-to-left-camera transform, valid only at this timestamp."""

    T_camera_table: np.ndarray
    timestamp_s: float
    confidence: float
    metadata: dict


class ZedTablePoseTracker:
    """Validate a coloured, metric rectangle against ZED's aligned depth map."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        table_length_m: float = 2.74,
        table_width_m: float = 1.525,
        hsv_ranges: Optional[Iterable[Tuple[Sequence[int], Sequence[int]]]] = None,
        min_area: int = 15000,
        confirm_frames: int = 5,
        hold_seconds: float = 0.6,
    ) -> None:
        if table_length_m <= 0.0 or table_width_m <= 0.0:
            raise ValueError("table dimensions must be positive")
        if min_area <= 0 or confirm_frames <= 0 or hold_seconds < 0.0:
            raise ValueError("table tracker thresholds must be positive")
        self.intrinsics = intrinsics
        self.length = float(table_length_m)
        self.width = float(table_width_m)
        self.hsv_ranges = tuple(
            (np.asarray(low, np.uint8), np.asarray(high, np.uint8))
            for low, high in (hsv_ranges or [([85, 80, 40], [135, 255, 255])])
        )
        self.min_area = int(min_area)
        self.confirm_frames = int(confirm_frames)
        self.hold_seconds = float(hold_seconds)
        self._pending = []
        self._snapshot: Optional[ZedTablePoseSnapshot] = None
        self._epoch = 0
        self._state = "SEARCHING"
        self._reason = "not_initialized"
        self._debug = None
        self._new_epoch_on_next_valid = False

    def _rays(self, uv: np.ndarray) -> np.ndarray:
        pixels = np.asarray(uv, dtype=float).reshape(-1, 2)
        return np.column_stack(
            ((pixels[:, 0] - self.intrinsics.cx) / self.intrinsics.fx,
             (pixels[:, 1] - self.intrinsics.cy) / self.intrinsics.fy,
             np.ones(len(pixels)))
        )

    def project(self, xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(xyz, dtype=float).reshape(-1, 3)
        z = points[:, 2]
        uv = np.full((len(points), 2), np.nan)
        good = z > 1e-6
        uv[good, 0] = self.intrinsics.fx * points[good, 0] / z[good] + self.intrinsics.cx
        uv[good, 1] = self.intrinsics.fy * points[good, 1] / z[good] + self.intrinsics.cy
        return uv

    def _mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        raw = np.zeros(bgr.shape[:2], np.uint8)
        for low, high in self.hsv_ranges:
            raw |= cv2.inRange(hsv, low, high)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(1200, int(self.min_area * 0.08))
        return (keep[labels] * 255).astype(np.uint8)

    def _plane(self, depth_m: np.ndarray, mask: np.ndarray):
        interior = cv2.erode(mask, np.ones((11, 11), np.uint8))
        yy, xx = np.nonzero(interior[::5, ::5])
        xx, yy = xx * 5, yy * 5
        z = depth_m[yy, xx]
        valid = np.isfinite(z) & (z > 0.3) & (z < 6.0)
        if valid.sum() < 200:
            return None, "insufficient_depth"
        points = self._rays(np.column_stack((xx[valid], yy[valid]))) * z[valid, None]
        rng = np.random.default_rng(741)
        points = points[rng.permutation(len(points))[:2400]]
        train, held = points[::2], points[1::2]
        if len(train) < 3:
            return None, "insufficient_depth"
        best = None
        best_count = 0
        for _ in range(60):
            first, second, third = train[rng.choice(len(train), 3, replace=False)]
            normal = np.cross(second - first, third - first)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal /= norm
            inliers = np.abs((train - first) @ normal) < 0.018
            if int(inliers.sum()) > best_count:
                best, best_count = inliers, int(inliers.sum())
        if best is None or best_count < max(120, int(0.5 * len(train))):
            return None, "no_dominant_plane"
        fit = train[best]
        center = fit.mean(axis=0)
        _, _, vectors = np.linalg.svd(fit - center, full_matrices=False)
        normal = vectors[-1]
        if normal @ (-center) < 0.0:
            normal = -normal
        d = -float(normal @ center)
        residual = np.abs(held @ normal + d)
        valid_held = residual < 0.025
        if not len(held) or float(valid_held.mean()) < 0.60:
            return None, "depth_validation_failed"
        inliers = points[np.abs(points @ normal + d) < 0.025]
        spread = np.linalg.svd(inliers - inliers.mean(axis=0), compute_uv=False) / math.sqrt(len(inliers))
        if len(spread) < 2 or spread[1] < 0.12:
            return None, "depth_coverage_too_small"
        basis_x = np.array([0.0, 0.0, 1.0]) - normal[2] * normal
        if np.linalg.norm(basis_x) < 0.1:
            basis_x = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
        basis_x /= np.linalg.norm(basis_x)
        return {
            "normal": normal,
            "d": d,
            "origin": -d * normal,
            "basis": np.column_stack((basis_x, np.cross(normal, basis_x))),
            "inlier_ratio": float(valid_held.mean()),
            "error_m": float(np.median(residual[valid_held])),
        }, None

    def _candidate(self, image: DetectionFrame, depth: DepthFrame):
        expected_shape = (self.intrinsics.width, self.intrinsics.height)
        if ((image.intrinsics.width, image.intrinsics.height) != expected_shape
                or (depth.intrinsics.width, depth.intrinsics.height) != expected_shape):
            raise ValueError("table colour and depth must use this same rectified ZED intrinsics")
        mask = self._mask(image.image_bgr)
        if int(np.count_nonzero(mask)) < self.min_area:
            return None, mask, "insufficient_table_colour_area"
        plane, reason = self._plane(depth.depth_m, mask)
        if plane is None:
            return None, mask, reason
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea, default=None)
        if contour is None or cv2.contourArea(contour) < self.min_area:
            return None, mask, "no_table_outer_contour"
        pixels = contour.reshape(-1, 2)
        sample = pixels[::max(1, len(pixels) // 600)]
        xyz = self._rays(sample) * depth.depth_m[sample[:, 1], sample[:, 0], None]
        finite = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.3)
        on_plane = np.abs(xyz @ plane["normal"] + plane["d"]) < 0.030
        xy = (xyz[finite & on_plane] - plane["origin"]) @ plane["basis"]
        if len(xy) < 30:
            return None, mask, "insufficient_outer_boundary_depth"
        rectangle = cv2.minAreaRect(xy.astype(np.float32))
        center, size, angle_deg = rectangle
        side_a, side_b = map(float, size)
        observed_length, observed_width = max(side_a, side_b), min(side_a, side_b)
        if abs(observed_length - self.length) > 0.20 or abs(observed_width - self.width) > 0.16:
            return None, mask, "table_dimensions_mismatch"
        angle = math.radians(float(angle_deg))
        axis = np.array([math.cos(angle), math.sin(angle)])
        if side_a < side_b:
            axis = np.array([-axis[1], axis[0]])
        x_axis = plane["basis"] @ axis
        y_axis = np.cross(plane["normal"], x_axis)
        # Keep a stable orientation for a visually symmetric rectangle.
        if x_axis @ plane["basis"][:, 0] < 0.0:
            x_axis, y_axis = -x_axis, -y_axis
        transform = np.eye(4)
        transform[:3, :3] = np.column_stack((x_axis, y_axis, plane["normal"]))
        transform[:3, 3] = plane["origin"] + plane["basis"] @ np.asarray(center)
        confidence = float(np.clip(
            plane["inlier_ratio"] * (1.0 - plane["error_m"] / 0.025)
            * (1.0 - abs(observed_length - self.length) / 0.20)
            * (1.0 - abs(observed_width - self.width) / 0.16), 0.0, 1.0
        ))
        return (transform, confidence, {
            "plane_error_m": plane["error_m"],
            "plane_inlier_ratio": plane["inlier_ratio"],
            "observed_length_m": observed_length,
            "observed_width_m": observed_width,
        }), mask, None

    def update(self, image: DetectionFrame, depth: DepthFrame) -> None:
        if image.sdk_image_timestamp_ns != depth.sdk_image_timestamp_ns:
            raise ValueError("table colour and depth must come from the same ZED IMAGE timestamp")
        timestamp_s = image.sdk_image_timestamp_ns / 1e9
        candidate, mask, reason = self._candidate(image, depth)
        debug = image.image_bgr.copy()
        debug[mask == 0] = (debug[mask == 0] * 0.25).astype(np.uint8)
        if candidate is None:
            self._pending.clear()
            self._reason = reason
            if self._snapshot is None or timestamp_s - self._snapshot.timestamp_s > self.hold_seconds:
                self._state = "LOST" if self._snapshot is not None else "SEARCHING"
                self._new_epoch_on_next_valid = self._snapshot is not None
            cv2.putText(
                debug, "table {}: {}".format(self._state, self._reason), (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2,
            )
            self._debug = debug
            return
        transform, confidence, metrics = candidate
        self._pending.append((transform, confidence, metrics, timestamp_s))
        self._pending = self._pending[-self.confirm_frames:]
        consistent = len(self._pending) == self.confirm_frames and all(
            np.linalg.norm(item[0][:3, 3] - transform[:3, 3]) < 0.08
            and _rotation_distance(item[0][:3, :3], transform[:3, :3]) < math.radians(4.0)
            for item in self._pending
        )
        if consistent:
            if self._snapshot is None or self._new_epoch_on_next_valid:
                self._epoch += 1
                self._new_epoch_on_next_valid = False
            self._snapshot = ZedTablePoseSnapshot(
                transform, timestamp_s, confidence,
                {**metrics, "state": "VALID", "table_frame_id": self._epoch},
            )
            self._state, self._reason = "VALID", "measurement"
        else:
            self._state, self._reason = "CANDIDATE", "awaiting_consistent_measurements"
        corners = np.array([
            [-self.length / 2.0, -self.width / 2.0, 0.0],
            [self.length / 2.0, -self.width / 2.0, 0.0],
            [self.length / 2.0, self.width / 2.0, 0.0],
            [-self.length / 2.0, self.width / 2.0, 0.0],
        ])
        projected = self.project(corners @ transform[:3, :3].T + transform[:3, 3])
        colour = (0, 255, 0) if self._state == "VALID" else (0, 180, 255)
        if np.isfinite(projected).all():
            projected_int = np.rint(projected).astype(np.int32)
            cv2.polylines(debug, [projected_int], True, colour, 2)
            for index, point in enumerate(projected_int):
                cv2.circle(debug, tuple(point), 4, colour, -1)
                cv2.putText(debug, str(index), tuple(point + (6, -6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        cv2.putText(debug, "table {}: {}".format(self._state, self._reason), (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        cv2.putText(
            debug, "%.3f x %.3f m | plane %.1f mm" % (
                metrics["observed_length_m"], metrics["observed_width_m"], 1000.0 * metrics["plane_error_m"],
            ), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
        )
        self._debug = debug

    def snapshot(self, timestamp_s: float):
        if self._snapshot is None:
            return None, {"valid": False, "state": self._state, "reason": self._reason, "table_frame_id": self._epoch}
        stale = max(0.0, timestamp_s - self._snapshot.timestamp_s)
        valid = stale <= self.hold_seconds and self._state == "VALID"
        metadata = {**self._snapshot.metadata, "valid": valid, "state": self._state,
                    "reason": self._reason, "stale_s": stale}
        return (self._snapshot if valid else None), metadata

    def reset(self) -> None:
        """Forget the current fixed-table solution and require fresh evidence."""
        self._pending.clear()
        self._snapshot = None
        self._state = "SEARCHING"
        self._reason = "manual_reinitialize"
        self._new_epoch_on_next_valid = False

    def save_pose(self, path: str) -> Path:
        """Persist the last valid metric table pose for external inspection."""
        if self._snapshot is None or self._state != "VALID":
            raise RuntimeError("cannot save a table pose before it is valid")
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            T_camera_table=self._snapshot.T_camera_table,
            table_length=self.length,
            table_width=self.width,
            table_frame_id=self._epoch,
            timestamp_s=self._snapshot.timestamp_s,
        )
        return output

    def get_debug_image(self):
        return None if self._debug is None else self._debug.copy()
