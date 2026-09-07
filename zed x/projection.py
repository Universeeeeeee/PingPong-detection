"""Pinhole geometry with explicit T_destination_from_source transforms."""

from typing import Tuple

import numpy as np

from camera_types import CameraIntrinsics, StereoCalibration


def camera_matrix(intrinsics: CameraIntrinsics) -> np.ndarray:
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _require_pinhole(intrinsics: CameraIntrinsics) -> None:
    if intrinsics.distortion_model not in ("none", "pinhole"):
        raise ValueError("this path only supports undistorted pinhole images")
    if intrinsics.distortion.size and not np.allclose(intrinsics.distortion, 0.0):
        raise ValueError("non-zero distortion is not valid on the rectified pinhole path")


def project_points(points_camera_m: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    _require_pinhole(intrinsics)
    points = np.asarray(points_camera_m, dtype=np.float64)
    single = points.ndim == 1
    points = points.reshape(-1, 3)
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0.0):
        raise ValueError("points must be finite and in front of the camera")
    uv = np.column_stack(
        (
            intrinsics.fx * points[:, 0] / points[:, 2] + intrinsics.cx,
            intrinsics.fy * points[:, 1] / points[:, 2] + intrinsics.cy,
        )
    )
    return uv[0] if single else uv


def deproject_pixels(
    pixels_uv: np.ndarray, depth_m: np.ndarray, intrinsics: CameraIntrinsics
) -> np.ndarray:
    _require_pinhole(intrinsics)
    pixels = np.asarray(pixels_uv, dtype=np.float64)
    single = pixels.ndim == 1
    pixels = pixels.reshape(-1, 2)
    depth = np.asarray(depth_m, dtype=np.float64).reshape(-1)
    if depth.size == 1 and len(pixels) != 1:
        depth = np.full(len(pixels), depth.item())
    if len(depth) != len(pixels):
        raise ValueError("one depth value is required per pixel")
    if not np.isfinite(pixels).all() or not np.isfinite(depth).all() or np.any(depth <= 0.0):
        raise ValueError("pixels and positive metre depths must be finite")
    points = np.column_stack(
        (
            (pixels[:, 0] - intrinsics.cx) * depth / intrinsics.fx,
            (pixels[:, 1] - intrinsics.cy) * depth / intrinsics.fy,
            depth,
        )
    )
    return points[0] if single else points


def transform_points(points_src: np.ndarray, T_dst_from_src: np.ndarray) -> np.ndarray:
    points = np.asarray(points_src, dtype=np.float64)
    single = points.ndim == 1
    points = points.reshape(-1, 3)
    transform = np.asarray(T_dst_from_src, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("T_dst_from_src must be a finite 4x4 matrix")
    homogeneous = np.column_stack((points, np.ones(len(points))))
    transformed = homogeneous.dot(transform.T)
    if np.any(np.abs(transformed[:, 3]) < 1e-12):
        raise ValueError("transform produced a point at infinity")
    result = transformed[:, :3] / transformed[:, 3:4]
    return result[0] if single else result


def triangulate_point(
    left_uv: np.ndarray, right_uv: np.ndarray, calibration: StereoCalibration
) -> np.ndarray:
    """Triangulate one correspondence into the left camera frame using DLT."""
    if calibration.image_geometry != "rectified":
        raise ValueError("the first implementation only accepts rectified stereo images")
    _require_pinhole(calibration.left)
    _require_pinhole(calibration.right)
    left_uv = np.asarray(left_uv, dtype=np.float64).reshape(2)
    right_uv = np.asarray(right_uv, dtype=np.float64).reshape(2)
    if not np.isfinite(left_uv).all() or not np.isfinite(right_uv).all():
        raise ValueError("image coordinates must be finite")

    P_left = camera_matrix(calibration.left).dot(
        np.column_stack((np.eye(3), np.zeros(3)))
    )
    P_right = camera_matrix(calibration.right).dot(
        calibration.T_right_from_left[:3, :]
    )
    A = np.vstack(
        (
            left_uv[0] * P_left[2] - P_left[0],
            left_uv[1] * P_left[2] - P_left[1],
            right_uv[0] * P_right[2] - P_right[0],
            right_uv[1] * P_right[2] - P_right[1],
        )
    )
    _, _, vh = np.linalg.svd(A)
    homogeneous = vh[-1]
    if abs(homogeneous[3]) < 1e-12:
        raise ValueError("stereo rays do not produce a finite point")
    point_left = homogeneous[:3] / homogeneous[3]
    point_right = transform_points(point_left, calibration.T_right_from_left)
    if point_left[2] <= 0.0 or point_right[2] <= 0.0:
        raise ValueError("triangulated point is behind a camera")
    return point_left


def triangulation_covariance(
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    calibration: StereoCalibration,
    pixel_sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the triangulated point and first-order covariance from pixel noise."""
    if not np.isfinite(pixel_sigma) or pixel_sigma <= 0.0:
        raise ValueError("pixel_sigma must be positive")
    observation = np.concatenate(
        (np.asarray(left_uv, dtype=np.float64), np.asarray(right_uv, dtype=np.float64))
    )
    point = triangulate_point(observation[:2], observation[2:], calibration)
    jacobian = np.empty((3, 4), dtype=np.float64)
    step = max(1e-4, pixel_sigma * 1e-3)
    for index in range(4):
        plus = observation.copy()
        minus = observation.copy()
        plus[index] += step
        minus[index] -= step
        jacobian[:, index] = (
            triangulate_point(plus[:2], plus[2:], calibration)
            - triangulate_point(minus[:2], minus[2:], calibration)
        ) / (2.0 * step)
    covariance = (pixel_sigma ** 2) * jacobian.dot(jacobian.T)
    return point, 0.5 * (covariance + covariance.T)
