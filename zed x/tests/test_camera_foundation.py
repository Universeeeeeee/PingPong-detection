import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera_types import CameraIntrinsics, StereoCalibration
from projection import deproject_pixels, project_points, transform_points, triangulate_point
from zed_capture import ZedFrameNormalizer, bgra_to_bgr, zed_measurement_keys
from zed_frontend import ZedMiniBallFrontend
if cv2 is not None:
    from zed_mini_ball import ZedMiniStereoBallPipeline


def intrinsics(width=8, height=6):
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=400.0,
        fy=400.0,
        cx=width / 2.0,
        cy=height / 2.0,
        distortion=np.zeros(5),
    )


def stereo_calibration():
    transform = np.eye(4)
    transform[0, 3] = -0.063
    return StereoCalibration(
        left=intrinsics(),
        right=intrinsics(),
        T_right_from_left=transform,
        image_geometry="rectified",
        calibration_variant="calibration_parameters",
        calibration_id="mock-zed-mini-rectified",
    )


class CameraContractTests(unittest.TestCase):
    def test_rectified_image_rejects_raw_calibration(self):
        with self.assertRaisesRegex(ValueError, "calibration_parameters"):
            StereoCalibration(
                left=intrinsics(),
                right=intrinsics(),
                T_right_from_left=np.eye(4),
                image_geometry="rectified",
                calibration_variant="calibration_parameters_raw",
                calibration_id="wrong-pair",
            )

    def test_bgra_conversion_is_explicit_and_copies(self):
        bgra = np.array([[[1, 2, 3, 4]]], dtype=np.uint8)
        bgr = bgra_to_bgr(bgra)
        np.testing.assert_array_equal(bgr, [[[1, 2, 3]]])
        bgra[0, 0, 0] = 99
        self.assertEqual(bgr[0, 0, 0], 1)
        with self.assertRaises(ValueError):
            bgra_to_bgr(np.zeros((2, 2, 3), dtype=np.uint8))

    def test_normalizer_preserves_one_timestamp_and_marks_invalid_depth(self):
        calibration = stereo_calibration()
        normalizer = ZedFrameNormalizer(calibration)
        shape = (calibration.left.height, calibration.left.width)
        depth = np.full(shape, 2.0, dtype=np.float32)
        depth[0, 0] = np.nan
        frames = normalizer.normalize(
            np.zeros(shape + (4,), dtype=np.uint8),
            np.zeros(shape, dtype=np.uint8),
            np.zeros(shape, dtype=np.uint8),
            sdk_image_timestamp_ns=1_234_000_000,
            frame_number=7,
            depth_m=depth,
        )
        self.assertEqual(frames.detection.sdk_image_timestamp_ns, 1_234_000_000)
        self.assertEqual(frames.stereo.frame_number, 7)
        self.assertFalse(frames.depth.valid_mask[0, 0])
        self.assertTrue(frames.depth.valid_mask[1, 1])


class GeometryTests(unittest.TestCase):
    def test_project_deproject_round_trip(self):
        camera = intrinsics(640, 480)
        point = np.array([0.12, -0.08, 2.0])
        pixel = project_points(point, camera)
        recovered = deproject_pixels(pixel, point[2], camera)
        np.testing.assert_allclose(recovered, point, atol=1e-12)

    def test_transform_name_encodes_direction(self):
        T_right_from_left = np.eye(4)
        T_right_from_left[0, 3] = -0.063
        point_left = np.array([0.2, 0.0, 2.0])
        point_right = transform_points(point_left, T_right_from_left)
        np.testing.assert_allclose(point_right, [0.137, 0.0, 2.0], atol=1e-12)

    def test_stereo_triangulation_recovers_left_camera_point(self):
        calibration = stereo_calibration()
        point_left = np.array([0.03, -0.01, 2.0])
        point_right = transform_points(point_left, calibration.T_right_from_left)
        left_uv = project_points(point_left, calibration.left)
        right_uv = project_points(point_right, calibration.right)
        recovered = triangulate_point(left_uv, right_uv, calibration)
        np.testing.assert_allclose(recovered, point_left, atol=1e-10)


class FrontendBoundaryTests(unittest.TestCase):
    def test_mock_stereo_frontend_outputs_common_measurement(self):
        calibration = stereo_calibration()
        shape = (calibration.left.height, calibration.left.width)
        frames = ZedFrameNormalizer(calibration).normalize(
            np.zeros(shape + (4,), dtype=np.uint8),
            np.zeros(shape, dtype=np.uint8),
            np.zeros(shape, dtype=np.uint8),
            sdk_image_timestamp_ns=2_000_000_000,
            frame_number=11,
        )
        point_left = np.array([0.02, 0.01, 1.5])
        point_right = transform_points(point_left, calibration.T_right_from_left)
        measurement = ZedMiniBallFrontend().measure_from_centers(
            frames.stereo,
            project_points(point_left, calibration.left),
            project_points(point_right, calibration.right),
            confidence=0.9,
            identity_timestamp_s=1.99,
        )
        np.testing.assert_allclose(measurement.position_camera_m, point_left, atol=1e-10)
        self.assertEqual(measurement.timestamp_s, 2.0)
        self.assertEqual(
            measurement.observation_id,
            ("zed_mini", "offline-mock", 11, "rectified_stereo"),
        )
        self.assertEqual(
            measurement.correlation_group,
            ("zed_mini_stereo_pair", "offline-mock", 11),
        )

    def test_same_pair_evidence_is_unique_but_correlated(self):
        stereo_id, stereo_group = zed_measurement_keys(12, "rectified_stereo", "run-a")
        depth_id, depth_group = zed_measurement_keys(12, "sdk_depth", "run-a")
        self.assertNotEqual(stereo_id, depth_id)
        self.assertEqual(stereo_group, depth_group)
        self.assertEqual(
            stereo_id, zed_measurement_keys(12, "rectified_stereo", "run-a")[0]
        )
        self.assertNotEqual(
            stereo_id, zed_measurement_keys(12, "rectified_stereo", "run-b")[0]
        )


@unittest.skipIf(cv2 is None, "OpenCV is not installed in this Python environment")
class ZedMiniPipelineTests(unittest.TestCase):
    @staticmethod
    def scene(center=None):
        image = np.full((160, 240, 3), (100, 35, 20), dtype=np.uint8)
        if center is not None:
            cv2.circle(image, center, 5, (15, 110, 245), -1)
        return image

    @staticmethod
    def bgra(image):
        return np.dstack((image, np.full(image.shape[:2], 255, dtype=np.uint8)))

    def frames(self, left_center=None, right_center=None):
        calibration = StereoCalibration(
            left=intrinsics(240, 160),
            right=intrinsics(240, 160),
            T_right_from_left=np.array(
                [[1, 0, 0, -0.063], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=float,
            ),
            image_geometry="rectified",
            calibration_variant="calibration_parameters",
            calibration_id="mock-zed-mini-240x160",
        )
        left = self.scene(left_center)
        right = self.scene(right_center)
        return ZedFrameNormalizer(calibration).normalize(
            self.bgra(left),
            cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(right, cv2.COLOR_BGR2GRAY),
            sdk_image_timestamp_ns=1_000_000_000,
            frame_number=1,
            right_bgra=self.bgra(right),
        )

    def test_synchronized_orange_pair_outputs_mini_measurement(self):
        result = ZedMiniStereoBallPipeline().process(
            self.frames((120, 80), (103, 80))
        )
        self.assertIsNotNone(result.measurement)
        self.assertEqual(result.measurement.source, "zed_mini_rectified_stereo")
        self.assertAlmostEqual(result.measurement.position_camera_m[2], 1.482, places=2)
        self.assertEqual(result.diagnostics["reason"], "measurement")

    def test_missing_right_identity_never_outputs_3d(self):
        result = ZedMiniStereoBallPipeline().process(self.frames((120, 80), None))
        self.assertIsNone(result.measurement)
        self.assertTrue(result.diagnostics["reason"].startswith("right_"))

    def test_non_epipolar_pair_is_rejected(self):
        result = ZedMiniStereoBallPipeline().process(
            self.frames((120, 70), (103, 74))
        )
        self.assertIsNone(result.measurement)
        self.assertEqual(result.diagnostics["reason"], "no_valid_stereo_pair")
        self.assertGreater(result.diagnostics["rejected"].get("epipolar_error", 0), 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
