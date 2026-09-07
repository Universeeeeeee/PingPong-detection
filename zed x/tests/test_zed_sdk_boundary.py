import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zed_sdk_adapter import stereo_calibration_from_zed
from zed_source import ZedMiniSource, ZedSdkSource, ZedSourceConfig, ZedSourceError


def fake_camera_information():
    size = SimpleNamespace(width=8, height=6)
    left = SimpleNamespace(
        image_size=size, fx=400.0, fy=400.0, cx=4.0, cy=3.0, disto=[1.0] * 12
    )
    right = SimpleNamespace(
        image_size=size, fx=400.0, fy=400.0, cx=4.0, cy=3.0, disto=[2.0] * 12
    )
    calibration = SimpleNamespace(
        left_cam=left,
        right_cam=right,
        get_camera_baseline=lambda: 0.063,
    )
    configuration = SimpleNamespace(
        calibration_parameters=calibration,
        resolution=size,
        firmware_version=123,
        fps=100.0,
    )
    return SimpleNamespace(
        camera_configuration=configuration,
        serial_number=456,
        camera_model="ZED_M",
        input_type="USB",
    )


class FakeMat:
    def __init__(self):
        self.data = None

    def get_data(self):
        return self.data


class FakeCamera:
    last_init = None
    recording_parameters = None
    recording_disabled = False

    def __init__(self):
        self.info = fake_camera_information()
        self.closed = False

    def open(self, init):
        FakeCamera.last_init = init
        return 0

    def close(self):
        self.closed = True

    def enable_recording(self, parameters):
        FakeCamera.recording_parameters = parameters
        return 0

    def disable_recording(self):
        FakeCamera.recording_disabled = True

    def get_camera_information(self):
        return self.info

    def get_sdk_version(self):
        return "fake-5.4.1"

    def grab(self, runtime):
        return 0

    def retrieve_image(self, mat, view):
        if view == "LEFT":
            mat.data = np.zeros((6, 8, 4), dtype=np.uint8)
        elif view == "RIGHT":
            mat.data = np.zeros((6, 8, 4), dtype=np.uint8)
        else:
            mat.data = np.zeros((6, 8), dtype=np.uint8)
        return 0

    def retrieve_measure(self, mat, measure):
        mat.data = np.full((6, 8), 2.0 if measure == "DEPTH" else 10.0, np.float32)
        return 0

    def get_timestamp(self, reference):
        return SimpleNamespace(get_nanoseconds=lambda: 1_500_000_000)

    def get_frame_dropped_count(self):
        return 0

    def get_current_fps(self):
        return 99.5


class FakeInitParameters:
    def set_from_svo_file(self, path):
        self.svo_path = path

    def set_from_stream(self, host, port):
        self.stream = (host, port)

    def set_from_serial_number(self, serial):
        self.serial = serial


class FakeSL:
    ERROR_CODE = SimpleNamespace(SUCCESS=0, END_OF_SVOFILE_REACHED=1)
    UNIT = SimpleNamespace(METER="METER")
    COORDINATE_SYSTEM = SimpleNamespace(IMAGE="IMAGE")
    RESOLUTION = SimpleNamespace(AUTO="AUTO", VGA="VGA")
    DEPTH_MODE = SimpleNamespace(NONE="NONE", PERFORMANCE="PERFORMANCE")
    SVO_COMPRESSION_MODE = SimpleNamespace(H264="H264", H265="H265", LOSSLESS="LOSSLESS")
    VIEW = SimpleNamespace(
        LEFT="LEFT", RIGHT="RIGHT", LEFT_GRAY="LEFT_GRAY", RIGHT_GRAY="RIGHT_GRAY"
    )
    MEASURE = SimpleNamespace(DEPTH="DEPTH", CONFIDENCE="CONFIDENCE")
    TIME_REFERENCE = SimpleNamespace(IMAGE="IMAGE")
    InitParameters = FakeInitParameters
    RecordingParameters = lambda path, compression: SimpleNamespace(
        video_filename=path, compression_mode=compression
    )
    RuntimeParameters = type("RuntimeParameters", (), {})
    Camera = FakeCamera
    Mat = FakeMat


class CalibrationAdapterTests(unittest.TestCase):
    def test_rectified_calibration_uses_coordinate_transform_direction(self):
        calibration = stereo_calibration_from_zed(fake_camera_information())
        self.assertEqual(calibration.T_right_from_left[0, 3], -0.063)
        np.testing.assert_array_equal(calibration.left.distortion, np.zeros(12))
        self.assertTrue(calibration.calibration_id.startswith("zed-456-"))

    def test_calibration_id_is_deterministic(self):
        first = stereo_calibration_from_zed(fake_camera_information()).calibration_id
        second = stereo_calibration_from_zed(fake_camera_information()).calibration_id
        self.assertEqual(first, second)


class SourceConfigTests(unittest.TestCase):
    def test_input_modes_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            ZedSourceConfig(input_mode="live", svo_path="capture.svo2")
        with self.assertRaises(ValueError):
            ZedSourceConfig(input_mode="stream")

    def test_mock_sdk_open_and_read(self):
        source = ZedSdkSource(
            ZedSourceConfig(resolution="VGA", fps=100, enable_depth=True),
            sl_module=FakeSL,
        )
        with source:
            result = source.read()
            self.assertEqual(result.frames.stereo.sdk_image_timestamp_ns, 1_500_000_000)
            self.assertEqual(result.frames.stereo.capture_session_id, source.capture_session_id)
            self.assertEqual(result.frames.detection.image_bgr.shape, (6, 8, 3))
            self.assertEqual(result.frames.depth.depth_m.dtype, np.float32)
            self.assertEqual(result.sdk_current_fps, 99.5)
            self.assertEqual(source.metadata["sdk_image_timestamp_semantics"],
                             "entire_image_available_in_pc_memory")
        self.assertTrue(source.camera is None)

    def test_zed_mini_source_retrieves_same_timestamp_right_color(self):
        source = ZedMiniSource(
            ZedSourceConfig(resolution="VGA", fps=100, enable_right_color=True),
            sl_module=FakeSL,
        )
        with source:
            result = source.read()
            self.assertEqual(result.frames.right_detection.image_bgr.shape, (6, 8, 3))
            self.assertEqual(
                result.frames.right_detection.sdk_image_timestamp_ns,
                result.frames.detection.sdk_image_timestamp_ns,
            )

    def test_native_recording_is_enabled_and_disabled(self):
        FakeCamera.recording_parameters = None
        FakeCamera.recording_disabled = False
        source = ZedSdkSource(
            ZedSourceConfig(resolution="VGA", fps=100, record_path="capture.svo2"),
            sl_module=FakeSL,
        )
        with source:
            self.assertTrue(source.recording_enabled)
            self.assertEqual(FakeCamera.recording_parameters.video_filename, "capture.svo2")
            self.assertEqual(FakeCamera.recording_parameters.compression_mode, "H264")
        self.assertTrue(FakeCamera.recording_disabled)

    def test_zed_mini_source_rejects_other_model(self):
        source = ZedMiniSource(ZedSourceConfig(), sl_module=FakeSL)
        source._MODEL_NAMES = {"NOT-ZED-M"}
        with self.assertRaisesRegex(ZedSourceError, "requires camera model ZED-M"):
            source.open()
        self.assertIsNone(source.camera)


if __name__ == "__main__":
    unittest.main(verbosity=2)
