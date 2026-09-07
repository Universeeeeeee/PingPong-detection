"""Strict first-pass ZED Mini colour-stereo ball frontend."""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ball_image import ImageBallDetector
from camera_types import BallMeasurement3D
from projection import deproject_pixels, project_points, transform_points, triangulate_point
from zed_capture import NormalizedZedFrames
from zed_frontend import ZedMiniBallFrontend


def orange_hsv_ranges() -> list:
    return [
        (
            np.array([2, 75, 65], dtype=np.uint8),
            np.array([35, 255, 255], dtype=np.uint8),
        )
    ]


@dataclass(frozen=True)
class ZedMiniBallConfig:
    ball_diameter_m: float = 0.04
    min_depth_m: float = 0.25
    max_depth_m: float = 6.0
    max_epipolar_error_px: float = 2.0
    max_reprojection_error_px: float = 1.5
    max_relative_size_error: float = 0.75
    max_left_right_size_ratio: float = 2.0

    def __post_init__(self) -> None:
        positive = (
            self.ball_diameter_m,
            self.min_depth_m,
            self.max_depth_m,
            self.max_epipolar_error_px,
            self.max_reprojection_error_px,
            self.max_relative_size_error,
            self.max_left_right_size_ratio,
        )
        if not np.isfinite(positive).all() or min(positive) <= 0.0:
            raise ValueError("ZED Mini ball configuration values must be positive")
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_depth_m must be greater than min_depth_m")


@dataclass(frozen=True)
class ZedMiniBallResult:
    measurement: Optional[BallMeasurement3D]
    left_result: Dict
    right_result: Dict
    diagnostics: Dict


class ZedMiniStereoBallPipeline:
    """Require independent colour identity in both rectified Mini views."""

    def __init__(self, config: Optional[ZedMiniBallConfig] = None) -> None:
        self.config = config or ZedMiniBallConfig()
        ranges = orange_hsv_ranges()
        self.left_detector = ImageBallDetector(ranges, orange=True)
        self.frontend = ZedMiniBallFrontend()

    def _empty(self, left: Dict, right: Dict, reason: str, **details) -> ZedMiniBallResult:
        diagnostics = {"reason": reason}
        diagnostics.update(details)
        return ZedMiniBallResult(None, left, right, diagnostics)

    def _right_candidates(self, frames: NormalizedZedFrames, left: Dict) -> list:
        calibration = frames.stereo.calibration
        left_uv = np.asarray(left["uv"], dtype=np.float64)
        projected = []
        for depth_m in (self.config.min_depth_m, self.config.max_depth_m):
            point_left = deproject_pixels(left_uv, depth_m, calibration.left)
            point_right = transform_points(point_left, calibration.T_right_from_left)
            if point_right[2] > 0.0:
                projected.append(project_points(point_right, calibration.right))
        if not projected:
            return []

        image = frames.right_detection.image_bgr
        height, width = image.shape[:2]
        x, y, box_width, box_height = left["bbox"]
        margin = max(3, int(math.ceil(left["minor_px"] * 0.5)))
        right_u = np.asarray(projected)[:, 0]
        x0 = max(0, int(math.floor(float(right_u.min()))) - margin)
        x1 = min(width, int(math.ceil(float(right_u.max()))) + margin + 1)
        y0 = max(0, y - margin - int(math.ceil(self.config.max_epipolar_error_px)))
        y1 = min(
            height,
            y + box_height + margin + int(math.ceil(self.config.max_epipolar_error_px)),
        )
        if x1 - x0 < 3 or y1 - y0 < 3:
            return []

        crop = image[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        strict = cv2.inRange(
            hsv, np.array([2, 75, 65], np.uint8), np.array([35, 255, 255], np.uint8)
        )
        mask = cv2.inRange(
            hsv, np.array([0, 35, 55], np.uint8), np.array([40, 255, 255], np.uint8)
        )
        mask |= cv2.inRange(
            hsv, np.array([150, 35, 55], np.uint8), np.array([179, 255, 255], np.uint8)
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not 5.0 <= area <= 5000.0:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bx <= 0 or by <= 0 or bx + bw >= crop.shape[1] or by + bh >= crop.shape[0]:
                continue
            minor, major = sorted(float(value) + 1.0 for value in cv2.minAreaRect(contour)[1])
            if minor < 3.0 or minor > 100.0 or major / minor > 9.0:
                continue
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = area / max(hull_area, 1.0)
            fill = area / max(minor * major, 1.0)
            if solidity < 0.75 or fill < 0.35:
                continue
            inside = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(inside, [contour], -1, 255, -1)
            region = inside > 0
            seed_fraction = float(np.mean(strict[region] > 0))
            pixels = crop[region].astype(np.float32)
            chroma = float(np.percentile(pixels[:, 2] - pixels[:, 1], 75))
            if seed_fraction < 0.10 or chroma < 25.0:
                continue
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-9:
                continue
            center = np.array(
                [moments["m10"] / moments["m00"] + x0, moments["m01"] / moments["m00"] + y0],
                dtype=np.float64,
            )
            candidates.append(
                {
                    "uv": center,
                    "bbox": [bx + x0, by + y0, bw, bh],
                    "minor_px": minor,
                    "major_px": major,
                    "quality": float(
                        0.4 * solidity
                        + 0.25 * min(1.0, fill / 0.7)
                        + 0.2 * min(1.0, seed_fraction / 0.6)
                        + 0.15 * min(1.0, chroma / 80.0)
                    ),
                }
            )
        return candidates

    @staticmethod
    def _patch_ncc(left_gray, right_gray, left_uv, right_uv, radius: int) -> float:
        size = (2 * radius + 1, 2 * radius + 1)
        patches = [
            cv2.getRectSubPix(image, size, tuple(map(float, center))).astype(np.float32)
            for image, center in ((left_gray, left_uv), (right_gray, right_uv))
        ]
        if any(float(np.std(patch)) < 1.5 for patch in patches):
            return 0.0
        normalized = [patch - float(patch.mean()) for patch in patches]
        denominator = float(np.linalg.norm(normalized[0]) * np.linalg.norm(normalized[1]))
        return float(np.sum(normalized[0] * normalized[1]) / denominator) if denominator > 1e-6 else 0.0

    def process(self, frames: NormalizedZedFrames) -> ZedMiniBallResult:
        if frames.right_detection is None:
            raise ValueError("ZED Mini ball pipeline requires synchronized right colour")
        timestamps = {
            frames.detection.sdk_image_timestamp_ns,
            frames.right_detection.sdk_image_timestamp_ns,
            frames.stereo.sdk_image_timestamp_ns,
        }
        if len(timestamps) != 1:
            raise ValueError("left, right and stereo frames must share one SDK timestamp")

        timestamp_s = frames.stereo.sdk_image_timestamp_ns * 1e-9
        left_result, left_candidates, _ = self.left_detector.detect(
            frames.detection.image_bgr, timestamp_s
        )
        counts = {"left_candidates": len(left_candidates), "right_candidates": 0}
        if not left_result["valid"]:
            right_result = {"valid": False, "reason": "not_searched", "candidates": []}
            return self._empty(
                left_result, right_result, "left_" + left_result["reason"], **counts
            )

        left = self.left_detector.selected_candidate
        if left is None:
            right_result = {"valid": False, "reason": "not_searched", "candidates": []}
            return self._empty(left_result, right_result, "detector_contract_error", **counts)
        right_candidates = self._right_candidates(frames, left)
        counts["right_candidates"] = len(right_candidates)
        right_result = {
            "valid": bool(right_candidates),
            "reason": "roi_candidates" if right_candidates else "no_roi_candidate",
            "candidates": right_candidates,
        }
        if not right_candidates:
            return self._empty(left_result, right_result, "right_no_roi_candidate", **counts)
        left_uv = np.asarray(left["uv"], dtype=np.float64)
        calibration = frames.stereo.calibration
        eligible = []
        rejected = {}
        for right in right_candidates:
            right_uv = np.asarray(right["uv"], dtype=np.float64)
            epipolar_error = abs(float(left_uv[1] - right_uv[1]))
            if epipolar_error > self.config.max_epipolar_error_px:
                rejected["epipolar_error"] = rejected.get("epipolar_error", 0) + 1
                continue
            try:
                point_left = triangulate_point(left_uv, right_uv, calibration)
            except ValueError:
                rejected["triangulation"] = rejected.get("triangulation", 0) + 1
                continue
            point_right = transform_points(point_left, calibration.T_right_from_left)
            if not (
                self.config.min_depth_m <= point_left[2] <= self.config.max_depth_m
                and self.config.min_depth_m <= point_right[2] <= self.config.max_depth_m
            ):
                rejected["depth_range"] = rejected.get("depth_range", 0) + 1
                continue
            projected_left = project_points(point_left, calibration.left)
            projected_right = project_points(point_right, calibration.right)
            reprojection_error = 0.5 * (
                float(np.linalg.norm(projected_left - left_uv))
                + float(np.linalg.norm(projected_right - right_uv))
            )
            if reprojection_error > self.config.max_reprojection_error_px:
                rejected["reprojection_error"] = rejected.get("reprojection_error", 0) + 1
                continue
            observed = np.array([left["minor_px"], right["minor_px"]], dtype=np.float64)
            expected = np.array(
                [
                    calibration.left.fx * self.config.ball_diameter_m / point_left[2],
                    calibration.right.fx * self.config.ball_diameter_m / point_right[2],
                ]
            )
            relative_size_error = float(np.max(np.abs(observed - expected) / expected))
            size_ratio = float(np.max(observed) / np.min(observed))
            if (
                relative_size_error > self.config.max_relative_size_error
                or size_ratio > self.config.max_left_right_size_ratio
            ):
                rejected["physical_size"] = rejected.get("physical_size", 0) + 1
                continue
            patch_radius = int(np.clip(math.ceil(float(np.max(observed)) * 0.7), 4, 16))
            ncc = self._patch_ncc(
                frames.stereo.left_gray,
                frames.stereo.right_gray,
                left_uv,
                right_uv,
                patch_radius,
            )
            if ncc < 0.25:
                rejected["stereo_appearance"] = rejected.get("stereo_appearance", 0) + 1
                continue
            cost = (
                reprojection_error
                + 0.7 * relative_size_error
                + 0.5 * (1.0 - ncc)
                - 0.25 * float(right["quality"])
            )
            eligible.append(
                (cost, right, right_uv, point_left, observed, expected, relative_size_error, epipolar_error, reprojection_error, ncc)
            )
        eligible.sort(key=lambda item: item[0])
        if not eligible:
            return self._empty(
                left_result, right_result, "no_valid_stereo_pair", rejected=rejected, **counts
            )
        if len(eligible) > 1 and eligible[1][0] - eligible[0][0] < 0.12:
            return self._empty(
                left_result, right_result, "ambiguous_stereo_pair", rejected=rejected, **counts
            )
        (
            _, right, right_uv, point_left, observed, expected, relative_size_error,
            epipolar_error, reprojection_error, ncc,
        ) = eligible[0]

        identity_quality = 0.6 * float(left["quality"]) + 0.4 * float(right["quality"])
        geometry_quality = math.exp(-reprojection_error)
        size_quality = max(0.0, 1.0 - relative_size_error / self.config.max_relative_size_error)
        confidence = float(
            np.clip(
                0.45 * identity_quality
                + 0.20 * geometry_quality
                + 0.20 * size_quality
                + 0.15 * max(0.0, ncc),
                0.05,
                0.95,
            )
        )
        pixel_sigma = max(0.5, 0.08 * float(np.mean(observed)))
        measurement = self.frontend.measure_from_centers(
            frames.stereo,
            left_uv,
            right_uv,
            confidence=confidence,
            pixel_sigma=pixel_sigma,
            identity_timestamp_s=timestamp_s,
        )
        diagnostics = {
            "reason": "measurement",
            "left_candidates": len(left_candidates),
            "right_candidates": len(right_candidates),
            "left_uv": left_uv.tolist(),
            "right_uv": right_uv.tolist(),
            "epipolar_error_px": epipolar_error,
            "reprojection_error_px": reprojection_error,
            "observed_minor_px": observed.tolist(),
            "expected_diameter_px": expected.tolist(),
            "relative_size_error": relative_size_error,
            "patch_ncc": ncc,
            "rejected": rejected,
        }
        return ZedMiniBallResult(measurement, left_result, right_result, diagnostics)
