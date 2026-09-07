import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera_types import CameraIntrinsics, DepthFrame, DetectionFrame

if cv2 is not None:
    from zed_table_pose import ZedTablePoseTracker


@unittest.skipIf(cv2 is None, "OpenCV is not installed in this Python environment")
class ZedTablePoseTests(unittest.TestCase):
    def setUp(self):
        self.intrinsics = CameraIntrinsics(
            width=640, height=480, fx=400.0, fy=400.0, cx=320.0, cy=240.0,
            distortion=np.zeros(5), distortion_model="pinhole",
        )

    def frame(self, timestamp_ns, table=True, depth_m=2.0):
        bgr = np.full((480, 640, 3), 20, dtype=np.uint8)
        if table:
            cv2.rectangle(bgr, (46, 87), (594, 392), (255, 0, 0), -1)
        depth = np.full((480, 640), np.nan, dtype=np.float32)
        if table:
            depth[87:393, 46:595] = depth_m
        valid = np.isfinite(depth) & (depth > 0.0)
        image = DetectionFrame(
            image_bgr=bgr, sdk_image_timestamp_ns=timestamp_ns, frame_number=1,
            capture_session_id="test", intrinsics=self.intrinsics,
            camera_frame_id="zed_left_optical_frame",
        )
        depth_frame = DepthFrame(
            depth_m=depth, valid_mask=valid, confidence=None,
            sdk_image_timestamp_ns=timestamp_ns, frame_number=1, capture_session_id="test",
            intrinsics=self.intrinsics, aligned_to_frame_id="zed_left_optical_frame",
        )
        return image, depth_frame

    def test_metric_rectangle_becomes_valid_after_confirmation(self):
        tracker = ZedTablePoseTracker(
            self.intrinsics, min_area=5000, confirm_frames=2, hold_seconds=0.5,
        )
        for timestamp_ns in (1_000_000_000, 1_016_000_000):
            tracker.update(*self.frame(timestamp_ns))
        snapshot, metadata = tracker.snapshot(1.016)
        self.assertIsNotNone(snapshot)
        self.assertTrue(metadata["valid"])
        self.assertEqual(metadata["state"], "VALID")
        self.assertAlmostEqual(snapshot.T_camera_table[2, 3], 2.0, places=2)
        self.assertAlmostEqual(metadata["observed_length_m"], 2.74, places=2)
        self.assertAlmostEqual(metadata["observed_width_m"], 1.525, places=2)

    def test_missing_table_never_produces_a_pose(self):
        tracker = ZedTablePoseTracker(self.intrinsics, min_area=5000, confirm_frames=1)
        tracker.update(*self.frame(1_000_000_000, table=False))
        snapshot, metadata = tracker.snapshot(1.0)
        self.assertIsNone(snapshot)
        self.assertEqual(metadata["reason"], "insufficient_table_colour_area")

    def test_mismatched_image_and_depth_timestamps_are_rejected(self):
        tracker = ZedTablePoseTracker(self.intrinsics, min_area=5000, confirm_frames=1)
        image, depth = self.frame(1_000_000_000)
        bad_depth = DepthFrame(
            depth_m=depth.depth_m, valid_mask=depth.valid_mask, confidence=None,
            sdk_image_timestamp_ns=1_001_000_000, frame_number=1, capture_session_id="test",
            intrinsics=self.intrinsics, aligned_to_frame_id="zed_left_optical_frame",
        )
        with self.assertRaisesRegex(ValueError, "same ZED IMAGE timestamp"):
            tracker.update(image, bad_depth)

