"""Live, SVO2 and local-stream ZED input with one typed-frame output contract."""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, Optional
import uuid

import numpy as np

from zed_capture import NormalizedZedFrames, ZedFrameNormalizer, import_zed_sdk
from zed_sdk_adapter import camera_metadata_from_zed, stereo_calibration_from_zed


class ZedSourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ZedSourceConfig:
    input_mode: str = "live"
    svo_path: Optional[str] = None
    stream_host: Optional[str] = None
    stream_port: int = 0
    serial_number: Optional[int] = None
    resolution: str = "AUTO"
    fps: int = 0
    enable_depth: bool = False
    depth_mode: str = "PERFORMANCE"
    svo_real_time_mode: bool = False
    enable_right_color: bool = False
    record_path: Optional[str] = None
    record_compression: str = "H264"

    def __post_init__(self) -> None:
        if self.input_mode not in ("live", "svo", "stream"):
            raise ValueError("input_mode must be live, svo or stream")
        if self.input_mode == "svo":
            if not self.svo_path:
                raise ValueError("svo input requires svo_path")
            if self.stream_host is not None or self.serial_number is not None:
                raise ValueError("svo input cannot also select stream or serial")
        elif self.input_mode == "stream":
            if not self.stream_host:
                raise ValueError("stream input requires stream_host")
            if self.svo_path is not None or self.serial_number is not None:
                raise ValueError("stream input cannot also select SVO or serial")
            if not 0 <= self.stream_port <= 65535:
                raise ValueError("stream_port must be in [0, 65535]")
        elif self.svo_path is not None or self.stream_host is not None:
            raise ValueError("live input cannot also select SVO or stream")
        if self.serial_number is not None and self.serial_number <= 0:
            raise ValueError("serial_number must be positive")
        if self.fps < 0:
            raise ValueError("fps must be non-negative")
        if not self.resolution or not self.depth_mode:
            raise ValueError("resolution and depth_mode must be non-empty")
        if self.record_path is not None and not str(self.record_path).strip():
            raise ValueError("record_path must be non-empty when supplied")
        if not self.record_compression:
            raise ValueError("record_compression must be non-empty")


@dataclass(frozen=True)
class ZedReadResult:
    frames: NormalizedZedFrames
    host_arrival_monotonic_ns: int
    sdk_dropped_frame_count: int
    sdk_current_fps: float


class ZedSdkSource:
    def __init__(self, config: ZedSourceConfig, sl_module: Optional[Any] = None) -> None:
        self.config = config
        self.sl = sl_module
        self.camera = None
        self.runtime = None
        self.normalizer = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.frame_number = 0
        self.capture_session_id = uuid.uuid4().hex
        self._mats = {}
        self.recording_enabled = False

    def _enum(self, enum_group: Any, name: str, option: str) -> Any:
        try:
            return getattr(enum_group, name)
        except AttributeError as error:
            raise ZedSourceError(
                "ZED SDK does not support {} {}; inspect installed SDK/version first".format(
                    option, name
                )
            ) from error

    def _require_success(self, status: Any, operation: str) -> None:
        if status != self.sl.ERROR_CODE.SUCCESS:
            raise ZedSourceError("{} failed: {}".format(operation, status))

    def _enable_recording(self, camera: Any) -> None:
        path = self.config.record_path
        if path is None:
            return
        try:
            recording_path = Path(path).expanduser()
            recording_path.parent.mkdir(parents=True, exist_ok=True)
            recording_parameters_type = self.sl.RecordingParameters
            compression_group = self.sl.SVO_COMPRESSION_MODE
            compression = self._enum(
                compression_group,
                self.config.record_compression,
                "recording compression",
            )
            recording_parameters = recording_parameters_type(str(recording_path), compression)
            self._require_success(
                camera.enable_recording(recording_parameters), "Camera.enable_recording"
            )
        except AttributeError as error:
            raise ZedSourceError(
                "installed ZED SDK does not expose native recording support"
            ) from error
        self.recording_enabled = True

    def open(self) -> None:
        if self.camera is not None:
            raise ZedSourceError("source is already open")
        if self.sl is None:
            self.sl = import_zed_sdk()
        sl = self.sl
        init = sl.InitParameters()
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        init.camera_resolution = self._enum(sl.RESOLUTION, self.config.resolution, "resolution")
        init.camera_fps = self.config.fps
        init.depth_mode = self._enum(
            sl.DEPTH_MODE,
            self.config.depth_mode if self.config.enable_depth else "NONE",
            "depth mode",
        )
        init.depth_stabilization = 0
        if self.config.input_mode == "svo":
            path = Path(self.config.svo_path).expanduser()
            if not path.is_file():
                raise ZedSourceError("SVO/SVO2 file does not exist: {}".format(path))
            init.set_from_svo_file(str(path))
            init.svo_real_time_mode = self.config.svo_real_time_mode
        elif self.config.input_mode == "stream":
            init.set_from_stream(self.config.stream_host, self.config.stream_port)
        elif self.config.serial_number is not None:
            init.set_from_serial_number(self.config.serial_number)

        camera = sl.Camera()
        status = camera.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            camera.close()
            raise ZedSourceError("Camera.open failed: {}".format(status))
        try:
            self._enable_recording(camera)
            calibration = stereo_calibration_from_zed(camera.get_camera_information())
            self.normalizer = ZedFrameNormalizer(
                calibration,
                capture_session_id=self.capture_session_id,
            )
            self.runtime = sl.RuntimeParameters()
            self.runtime.enable_depth = self.config.enable_depth
            self._mats = {
                "left": sl.Mat(),
                "left_gray": sl.Mat(),
                "right_gray": sl.Mat(),
            }
            if self.config.enable_right_color:
                self._mats["right"] = sl.Mat()
            if self.config.enable_depth:
                self._mats["depth"] = sl.Mat()
                self._mats["confidence"] = sl.Mat()
            self.metadata = camera_metadata_from_zed(camera)
            self.metadata.update(
                {
                    "capture_session_id": self.capture_session_id,
                    "calibration_id": calibration.calibration_id,
                    "baseline_m": float(-calibration.T_right_from_left[0, 3]),
                    "sdk_image_timestamp_semantics": "entire_image_available_in_pc_memory",
                    "recording_path": (
                        str(Path(self.config.record_path).expanduser())
                        if self.config.record_path is not None
                        else None
                    ),
                    "recording_compression": (
                        self.config.record_compression if self.config.record_path is not None else None
                    ),
                }
            )
            self.camera = camera
        except Exception:
            if self.recording_enabled:
                try:
                    camera.disable_recording()
                finally:
                    self.recording_enabled = False
            camera.close()
            raise

    def close(self) -> None:
        camera = self.camera
        self.camera = None
        if camera is not None:
            try:
                if self.recording_enabled:
                    camera.disable_recording()
            finally:
                self.recording_enabled = False
                camera.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _array(self, key: str) -> np.ndarray:
        return np.asarray(self._mats[key].get_data())

    def read(self) -> Optional[ZedReadResult]:
        if self.camera is None or self.normalizer is None:
            raise ZedSourceError("source is not open")
        status = self.camera.grab(self.runtime)
        if status == getattr(self.sl.ERROR_CODE, "END_OF_SVOFILE_REACHED", None):
            return None
        self._require_success(status, "Camera.grab")
        host_arrival_ns = time.monotonic_ns()

        views = [
            ("left", self.sl.VIEW.LEFT),
            ("left_gray", self.sl.VIEW.LEFT_GRAY),
            ("right_gray", self.sl.VIEW.RIGHT_GRAY),
        ]
        if self.config.enable_right_color:
            views.append(("right", self.sl.VIEW.RIGHT))
        for key, view in views:
            self._require_success(
                self.camera.retrieve_image(self._mats[key], view),
                "retrieve_image({})".format(key),
            )
        timestamp_ns = int(
            self.camera.get_timestamp(self.sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        )
        if timestamp_ns <= 0:
            raise ZedSourceError("SDK returned a zero/invalid IMAGE timestamp")

        depth = None
        confidence = None
        if self.config.enable_depth:
            self._require_success(
                self.camera.retrieve_measure(self._mats["depth"], self.sl.MEASURE.DEPTH),
                "retrieve_measure(DEPTH)",
            )
            self._require_success(
                self.camera.retrieve_measure(
                    self._mats["confidence"], self.sl.MEASURE.CONFIDENCE
                ),
                "retrieve_measure(CONFIDENCE)",
            )
            depth = self._array("depth")
            confidence = self._array("confidence")

        left_gray = np.squeeze(self._array("left_gray"))
        right_gray = np.squeeze(self._array("right_gray"))
        frames = self.normalizer.normalize(
            self._array("left"),
            left_gray,
            right_gray,
            sdk_image_timestamp_ns=timestamp_ns,
            frame_number=self.frame_number,
            depth_m=depth,
            confidence=confidence,
            right_bgra=(self._array("right") if self.config.enable_right_color else None),
        )
        self.frame_number += 1
        return ZedReadResult(
            frames=frames,
            host_arrival_monotonic_ns=host_arrival_ns,
            sdk_dropped_frame_count=int(self.camera.get_frame_dropped_count()),
            sdk_current_fps=float(self.camera.get_current_fps()),
        )


class ZedMiniSource(ZedSdkSource):
    """ZED SDK source that fails closed unless the recording/device is a ZED Mini."""

    _MODEL_NAMES = {"ZED-M", "ZED_M", "MODEL.ZED_M"}

    def open(self) -> None:
        super().open()
        model = str(self.metadata["camera_model"]).upper()
        if model not in self._MODEL_NAMES:
            self.close()
            raise ZedSourceError(
                "ZedMiniSource requires camera model ZED-M, got {}".format(model)
            )
        if self.config.input_mode == "live" and "USB" not in str(
            self.metadata["input_type"]
        ).upper():
            self.close()
            raise ZedSourceError("live ZED Mini input must use the USB camera path")
