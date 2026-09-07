"""Camera-SDK-independent data contracts for the ZED migration.

All image timestamps are the integer timestamps reported by the camera SDK.
They are not asserted to be physical exposure start or midpoint timestamps.
"""

from dataclasses import dataclass
from typing import Hashable, Optional, Tuple

import numpy as np


def _readonly_array(value: np.ndarray, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    array = np.asarray(value).copy()
    if shape is not None and array.shape != shape:
        raise ValueError("expected array shape {}, got {}".format(shape, array.shape))
    array.setflags(write=False)
    return array


def _require_image_timestamp(timestamp_ns: int) -> None:
    if not isinstance(timestamp_ns, (int, np.integer)) or timestamp_ns < 0:
        raise ValueError("sdk_image_timestamp_ns must be a non-negative integer")


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray
    distortion_model: str = "none"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all():
            raise ValueError("intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")
        object.__setattr__(self, "distortion", _readonly_array(self.distortion).reshape(-1))


@dataclass(frozen=True)
class StereoCalibration:
    left: CameraIntrinsics
    right: CameraIntrinsics
    T_right_from_left: np.ndarray
    image_geometry: str
    calibration_variant: str
    calibration_id: str

    def __post_init__(self) -> None:
        transform = _readonly_array(self.T_right_from_left, (4, 4))
        if not np.isfinite(transform).all():
            raise ValueError("stereo transform must be finite")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("stereo transform must be homogeneous")
        expected = {
            "rectified": "calibration_parameters",
            "unrectified": "calibration_parameters_raw",
        }
        if self.image_geometry not in expected:
            raise ValueError("image_geometry must be rectified or unrectified")
        if self.calibration_variant != expected[self.image_geometry]:
            raise ValueError(
                "{} images require {}".format(
                    self.image_geometry, expected[self.image_geometry]
                )
            )
        if not self.calibration_id:
            raise ValueError("calibration_id must be non-empty")
        object.__setattr__(self, "T_right_from_left", transform)


@dataclass(frozen=True)
class CameraRigCalibration:
    reference_frame_id: str
    detection_intrinsics: CameraIntrinsics
    stereo: StereoCalibration
    T_reference_from_detection: np.ndarray

    def __post_init__(self) -> None:
        if not self.reference_frame_id:
            raise ValueError("reference_frame_id must be non-empty")
        transform = _readonly_array(self.T_reference_from_detection, (4, 4))
        if not np.isfinite(transform).all():
            raise ValueError("rig transform must be finite")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("rig transform must be homogeneous")
        object.__setattr__(self, "T_reference_from_detection", transform)


@dataclass(frozen=True)
class DetectionFrame:
    image_bgr: np.ndarray
    sdk_image_timestamp_ns: int
    frame_number: int
    capture_session_id: str
    intrinsics: CameraIntrinsics
    camera_frame_id: str

    def __post_init__(self) -> None:
        _require_image_timestamp(self.sdk_image_timestamp_ns)
        image = _readonly_array(self.image_bgr)
        expected = (self.intrinsics.height, self.intrinsics.width, 3)
        if image.dtype != np.uint8 or image.shape != expected:
            raise ValueError("image_bgr must be uint8 with shape {}".format(expected))
        if self.frame_number < 0 or not self.capture_session_id or not self.camera_frame_id:
            raise ValueError("frame number, capture session and camera frame ID must be valid")
        object.__setattr__(self, "image_bgr", image)


@dataclass(frozen=True)
class StereoFrame:
    left_gray: np.ndarray
    right_gray: np.ndarray
    sdk_image_timestamp_ns: int
    frame_number: int
    capture_session_id: str
    calibration: StereoCalibration

    def __post_init__(self) -> None:
        _require_image_timestamp(self.sdk_image_timestamp_ns)
        left = _readonly_array(self.left_gray)
        right = _readonly_array(self.right_gray)
        left_shape = (self.calibration.left.height, self.calibration.left.width)
        right_shape = (self.calibration.right.height, self.calibration.right.width)
        if left.dtype != np.uint8 or left.shape != left_shape:
            raise ValueError("left_gray must be uint8 with shape {}".format(left_shape))
        if right.dtype != np.uint8 or right.shape != right_shape:
            raise ValueError("right_gray must be uint8 with shape {}".format(right_shape))
        if self.frame_number < 0 or not self.capture_session_id:
            raise ValueError("frame number and capture session must be valid")
        object.__setattr__(self, "left_gray", left)
        object.__setattr__(self, "right_gray", right)


@dataclass(frozen=True)
class DepthFrame:
    depth_m: np.ndarray
    valid_mask: np.ndarray
    confidence: Optional[np.ndarray]
    sdk_image_timestamp_ns: int
    frame_number: int
    capture_session_id: str
    intrinsics: CameraIntrinsics
    aligned_to_frame_id: str

    def __post_init__(self) -> None:
        _require_image_timestamp(self.sdk_image_timestamp_ns)
        shape = (self.intrinsics.height, self.intrinsics.width)
        depth = _readonly_array(self.depth_m, shape)
        valid = _readonly_array(self.valid_mask, shape)
        confidence = None
        if self.confidence is not None:
            confidence = _readonly_array(self.confidence, shape)
        if depth.dtype not in (np.float32, np.float64):
            raise ValueError("depth_m must use floating-point metres")
        if valid.dtype != np.bool_:
            raise ValueError("valid_mask must be boolean")
        if np.any(valid & (~np.isfinite(depth) | (depth <= 0.0))):
            raise ValueError("valid depth samples must be finite and positive")
        if self.frame_number < 0 or not self.capture_session_id or not self.aligned_to_frame_id:
            raise ValueError("frame number, capture session and aligned frame ID must be valid")
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class BallMeasurement3D:
    timestamp_s: float
    position_camera_m: np.ndarray
    covariance_m2: np.ndarray
    confidence: float
    source: str
    observation_id: Hashable
    correlation_group: Hashable
    identity_timestamp_s: Optional[float]
    camera_frame_id: str
    calibration_id: str

    def __post_init__(self) -> None:
        position = _readonly_array(self.position_camera_m, (3,))
        covariance = _readonly_array(self.covariance_m2, (3, 3))
        if not np.isfinite(position).all() or not np.isfinite(covariance).all():
            raise ValueError("position and covariance must be finite")
        if not np.allclose(covariance, covariance.T, atol=1e-12):
            raise ValueError("covariance must be symmetric")
        if np.linalg.eigvalsh(covariance).min() < -1e-12:
            raise ValueError("covariance must be positive semidefinite")
        if not np.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.identity_timestamp_s is not None:
            if not np.isfinite(self.identity_timestamp_s):
                raise ValueError("identity_timestamp_s must be finite")
            if self.identity_timestamp_s > self.timestamp_s:
                raise ValueError("identity confirmation cannot be newer than the measurement")
        if not self.source or not self.camera_frame_id or not self.calibration_id:
            raise ValueError("source, camera_frame_id and calibration_id must be non-empty")
        hash(self.observation_id)
        hash(self.correlation_group)
        object.__setattr__(self, "position_camera_m", position)
        object.__setattr__(self, "covariance_m2", covariance)
