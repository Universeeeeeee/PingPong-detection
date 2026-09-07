"""ZED SDK boundary helpers that remain testable without a connected camera."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from camera_types import (
    DepthFrame,
    DetectionFrame,
    StereoCalibration,
    StereoFrame,
)


class ZedSdkUnavailable(RuntimeError):
    pass


def import_zed_sdk():
    """Import pyzed only when a real capture source is constructed."""
    try:
        import pyzed.sl as sl
    except ImportError as error:
        raise ZedSdkUnavailable(
            "pyzed.sl is unavailable; install a ZED SDK version compatible with this host"
        ) from error
    return sl


def bgra_to_bgr(image_bgra: np.ndarray) -> np.ndarray:
    image = np.asarray(image_bgra)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("ZED VIEW.LEFT must be a uint8 BGRA image")
    return image[:, :, :3].copy()


@dataclass(frozen=True)
class NormalizedZedFrames:
    detection: DetectionFrame
    stereo: StereoFrame
    depth: Optional[DepthFrame]
    right_detection: Optional[DetectionFrame] = None


class ZedFrameNormalizer:
    """Convert arrays from one successful ZED acquisition into typed frames."""

    def __init__(
        self,
        stereo_calibration: StereoCalibration,
        left_frame_id: str = "zed_left_optical_frame",
        capture_session_id: str = "offline-mock",
    ) -> None:
        if stereo_calibration.image_geometry != "rectified":
            raise ValueError("the first ZED path requires rectified calibration")
        self.left_intrinsics = stereo_calibration.left
        self.stereo_calibration = stereo_calibration
        self.left_frame_id = left_frame_id
        if not capture_session_id:
            raise ValueError("capture_session_id must be non-empty")
        self.capture_session_id = capture_session_id

    def normalize(
        self,
        left_bgra: np.ndarray,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        sdk_image_timestamp_ns: int,
        frame_number: int,
        depth_m: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None,
        right_bgra: Optional[np.ndarray] = None,
    ) -> NormalizedZedFrames:
        detection = DetectionFrame(
            image_bgr=bgra_to_bgr(left_bgra),
            sdk_image_timestamp_ns=sdk_image_timestamp_ns,
            frame_number=frame_number,
            capture_session_id=self.capture_session_id,
            intrinsics=self.left_intrinsics,
            camera_frame_id=self.left_frame_id,
        )
        stereo = StereoFrame(
            left_gray=left_gray,
            right_gray=right_gray,
            sdk_image_timestamp_ns=sdk_image_timestamp_ns,
            frame_number=frame_number,
            capture_session_id=self.capture_session_id,
            calibration=self.stereo_calibration,
        )
        right_detection = None
        if right_bgra is not None:
            right_detection = DetectionFrame(
                image_bgr=bgra_to_bgr(right_bgra),
                sdk_image_timestamp_ns=sdk_image_timestamp_ns,
                frame_number=frame_number,
                capture_session_id=self.capture_session_id,
                intrinsics=self.stereo_calibration.right,
                camera_frame_id="zed_right_optical_frame",
            )
        normalized_depth = None
        if depth_m is not None:
            depth = np.asarray(depth_m, dtype=np.float32)
            valid = np.isfinite(depth) & (depth > 0.0)
            normalized_depth = DepthFrame(
                depth_m=depth,
                valid_mask=valid,
                confidence=confidence,
                sdk_image_timestamp_ns=sdk_image_timestamp_ns,
                frame_number=frame_number,
                capture_session_id=self.capture_session_id,
                intrinsics=self.left_intrinsics,
                aligned_to_frame_id=self.left_frame_id,
            )
        elif confidence is not None:
            raise ValueError("confidence cannot be supplied without depth")
        return NormalizedZedFrames(
            detection=detection,
            stereo=stereo,
            depth=normalized_depth,
            right_detection=right_detection,
        )


def zed_measurement_keys(
    frame_number: int, evidence_kind: str, capture_session_id: str
) -> Tuple[tuple, tuple]:
    """Create a unique observation key and a shared same-frame correlation key."""
    if frame_number < 0 or not evidence_kind or not capture_session_id:
        raise ValueError("frame number, evidence kind and capture session must be valid")
    return (
        ("zed", capture_session_id, frame_number, evidence_kind),
        ("zed_stereo_pair", capture_session_id, frame_number),
    )


def zed_mini_measurement_keys(
    frame_number: int, evidence_kind: str, capture_session_id: str
) -> Tuple[tuple, tuple]:
    if frame_number < 0 or not evidence_kind or not capture_session_id:
        raise ValueError("frame number, evidence kind and capture session must be valid")
    return (
        ("zed_mini", capture_session_id, frame_number, evidence_kind),
        ("zed_mini_stereo_pair", capture_session_id, frame_number),
    )
