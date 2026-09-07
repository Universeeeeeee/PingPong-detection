"""Conversion from ZED SDK objects to SDK-independent calibration types."""

import hashlib
import json
from typing import Any, Dict

import numpy as np

from camera_types import CameraIntrinsics, StereoCalibration


def camera_intrinsics_from_zed(camera_parameters: Any) -> CameraIntrinsics:
    """Convert rectified ZED CameraParameters to a pinhole model."""
    size = camera_parameters.image_size
    distortion = np.asarray(camera_parameters.disto, dtype=np.float64).reshape(-1)
    # Rectified VIEW.LEFT/RIGHT images must not be distorted again, even if an
    # older SDK exposes small residual numbers in the CameraParameters object.
    distortion = np.zeros_like(distortion)
    return CameraIntrinsics(
        width=int(size.width),
        height=int(size.height),
        fx=float(camera_parameters.fx),
        fy=float(camera_parameters.fy),
        cx=float(camera_parameters.cx),
        cy=float(camera_parameters.cy),
        distortion=distortion,
        distortion_model="pinhole",
    )


def stereo_calibration_from_zed(camera_information: Any) -> StereoCalibration:
    """Convert the active rectified calibration returned after Camera.open()."""
    configuration = camera_information.camera_configuration
    calibration = configuration.calibration_parameters
    left = camera_intrinsics_from_zed(calibration.left_cam)
    right = camera_intrinsics_from_zed(calibration.right_cam)
    if (left.width, left.height) != (right.width, right.height):
        raise ValueError("rectified left/right resolutions differ")

    baseline_m = float(calibration.get_camera_baseline())
    if not np.isfinite(baseline_m) or baseline_m <= 0.0:
        raise ValueError("ZED baseline must be positive and expressed in metres")

    # Our transform name describes point-coordinate conversion. For a right
    # camera physically +baseline on the left camera X axis, p_right.x is
    # p_left.x - baseline. This does not rely on ambiguous transform wording.
    T_right_from_left = np.eye(4, dtype=np.float64)
    T_right_from_left[0, 3] = -baseline_m

    identity_payload = {
        "serial_number": int(camera_information.serial_number),
        "width": left.width,
        "height": left.height,
        "left": [left.fx, left.fy, left.cx, left.cy],
        "right": [right.fx, right.fy, right.cx, right.cy],
        "baseline_m": baseline_m,
        "variant": "calibration_parameters",
    }
    digest = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return StereoCalibration(
        left=left,
        right=right,
        T_right_from_left=T_right_from_left,
        image_geometry="rectified",
        calibration_variant="calibration_parameters",
        calibration_id="zed-{}-{}".format(camera_information.serial_number, digest),
    )


def camera_metadata_from_zed(camera: Any) -> Dict[str, Any]:
    information = camera.get_camera_information()
    configuration = information.camera_configuration
    return {
        "sdk_version": str(camera.get_sdk_version()),
        "serial_number": int(information.serial_number),
        "camera_model": str(information.camera_model),
        "input_type": str(information.input_type),
        "firmware_version": int(configuration.firmware_version),
        "width": int(configuration.resolution.width),
        "height": int(configuration.resolution.height),
        "configured_fps": float(configuration.fps),
    }
