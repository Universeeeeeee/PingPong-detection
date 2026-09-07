"""Convenient MP4 preview recording for ZED frames.

The native SDK recorder in :mod:`zed_source` is the experiment capture path.
This small writer is intentionally only a viewable side-by-side MP4;
it is not used as an input to stereo processing.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class Mp4PreviewRecorder:
    """Write left/right BGR frames as one side-by-side MP4 stream."""

    def __init__(self, path: str, fps: float = 30.0) -> None:
        if not str(path).strip():
            raise ValueError("MP4 path must be non-empty")
        if fps <= 0.0:
            raise ValueError("MP4 fps must be positive")
        self.path = Path(path).expanduser()
        self.fps = float(fps)
        self._writer: Optional[cv2.VideoWriter] = None
        self.frames_written = 0

    def _open(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("MP4 frames must be uint8 BGR images")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        self._writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if not self._writer.isOpened():
            self._writer.release()
            self._writer = None
            raise RuntimeError("OpenCV could not open MP4 writer: {}".format(self.path))

    def write(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> None:
        left = np.asarray(left_bgr)
        right = np.asarray(right_bgr)
        if left.ndim != 3 or right.ndim != 3 or left.shape[2] != 3 or right.shape[2] != 3:
            raise ValueError("MP4 preview inputs must be BGR images")
        if left.dtype != np.uint8 or right.dtype != np.uint8:
            raise ValueError("MP4 preview inputs must be uint8 BGR images")
        if right.shape[:2] != left.shape[:2]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
        canvas = np.hstack((left, right))
        if self._writer is None:
            self._open(canvas)
        self._writer.write(canvas)
        self.frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
