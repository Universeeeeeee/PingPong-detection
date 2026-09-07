"""Camera-specific ZED stereo frontend boundary."""

from typing import Optional

import numpy as np

from camera_types import BallMeasurement3D, StereoFrame
from projection import triangulation_covariance
from zed_capture import zed_mini_measurement_keys


class ZedMiniBallFrontend:
    """Turn ZED Mini rectified ball centres into a common 3D measurement."""

    def __init__(self, camera_frame_id: str = "zed_left_optical_frame") -> None:
        if not camera_frame_id:
            raise ValueError("camera_frame_id must be non-empty")
        self.camera_frame_id = camera_frame_id

    def measure_from_centers(
        self,
        frame: StereoFrame,
        left_uv: np.ndarray,
        right_uv: np.ndarray,
        confidence: float,
        pixel_sigma: float = 0.5,
        identity_timestamp_s: Optional[float] = None,
    ) -> BallMeasurement3D:
        point, covariance = triangulation_covariance(
            left_uv, right_uv, frame.calibration, pixel_sigma
        )
        observation_id, correlation_group = zed_mini_measurement_keys(
            frame.frame_number, "rectified_stereo", frame.capture_session_id
        )
        return BallMeasurement3D(
            timestamp_s=frame.sdk_image_timestamp_ns * 1e-9,
            position_camera_m=point,
            covariance_m2=covariance,
            confidence=confidence,
            source="zed_mini_rectified_stereo",
            observation_id=observation_id,
            correlation_group=correlation_group,
            identity_timestamp_s=identity_timestamp_s,
            camera_frame_id=self.camera_frame_id,
            calibration_id=frame.calibration.calibration_id,
        )
