"""AdaptiveEye Camera Manager.

Provides robust, production-grade video capture lifecycle management, DirectShow/MSMF
backend support on Windows, horizontal mirroring, warmup stabilization, and exception handling.
"""

import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np
import yaml

from app.core.logger import get_logger

logger = get_logger("CameraManager")


class CameraError(Exception):
    """Base exception for camera-related failures."""


class CameraNotFoundError(CameraError):
    """Raised when the specified camera device cannot be opened."""


class CameraReadError(CameraError):
    """Raised when frame capture fails or camera disconnects."""


@dataclass
class CameraConfig:
    """Strongly-typed camera configuration."""
    device_id: int = 0
    video_source: Optional[str] = None  # Optional video file path for testing/playback
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    auto_exposure: bool = True
    warmup_frames: int = 10
    buffer_size: int = 1

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "CameraConfig":
        """Load configuration from a YAML file with fallback to defaults."""
        if not yaml_path.exists():
            logger.warning("Config %s not found. Using defaults.", yaml_path)
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cam_data = data.get("camera", {})
        return cls(
            device_id=int(cam_data.get("device_id", 0)),
            video_source=cam_data.get("video_source", None),
            width=int(cam_data.get("width", 1280)),
            height=int(cam_data.get("height", 720)),
            fps=int(cam_data.get("fps", 30)),
            mirror=bool(cam_data.get("mirror", True)),
            auto_exposure=bool(cam_data.get("auto_exposure", True)),
            warmup_frames=int(cam_data.get("warmup_frames", 10)),
            buffer_size=int(cam_data.get("buffer_size", 1)),
        )


class CameraManager:
    """Manages OpenCV VideoCapture lifecycle, buffer configuration, and frame acquisition."""

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        self.config = config or CameraConfig()
        self.cap: Optional[cv2.VideoCapture] = None
        self._actual_width: int = 0
        self._actual_height: int = 0
        self._actual_fps: float = 0.0
        self._frame_count: int = 0
        self._start_time: float = 0.0

    @property
    def is_opened(self) -> bool:
        """Return True if the underlying capture device is actively opened."""
        return self.cap is not None and self.cap.isOpened()

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return (width, height) of the active camera stream."""
        return self._actual_width, self._actual_height

    @property
    def fps(self) -> float:
        """Return measured running FPS."""
        elapsed = time.perf_counter() - self._start_time
        if elapsed > 0 and self._frame_count > 0:
            return self._frame_count / elapsed
        return self._actual_fps

    def start(self) -> None:
        """Initialize and open the camera device or video source."""
        if self.is_opened:
            logger.warning("Camera already open.")
            return

        # Check if streaming from a pre-recorded video file for benchmarking
        if self.config.video_source:
            logger.info("Initializing capture from video file: %s", self.config.video_source)
            cap = cv2.VideoCapture(str(self.config.video_source))
            if not cap.isOpened():
                raise CameraNotFoundError(f"Failed to open video source: {self.config.video_source}")
        else:
            logger.info("Initializing hardware camera (device_id=%d)...", self.config.device_id)
            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.config.device_id, backend)

            if not cap.isOpened() and backend != cv2.CAP_ANY:
                logger.warning("DirectShow backend failed. Retrying with default backend...")
                cap = cv2.VideoCapture(self.config.device_id)

            if not cap.isOpened():
                raise CameraNotFoundError(
                    f"Failed to open camera device index {self.config.device_id}. "
                    "Ensure the webcam is connected, not occupied by another app (e.g. Teams/Zoom), "
                    "and webcam privacy permissions are enabled in Windows Settings."
                )

        # Configure stream properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

        self._actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._actual_fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(self.config.fps)

        self.cap = cap
        self._frame_count = 0
        self._start_time = time.perf_counter()

        logger.info(
            "Capture started: %dx%d @ %.1f FPS (Requested: %dx%d @ %d FPS)",
            self._actual_width,
            self._actual_height,
            self._actual_fps,
            self.config.width,
            self.config.height,
            self.config.fps,
        )

        # Warmup loop to stabilize exposure and auto-gain (only for hardware cameras)
        if not self.config.video_source:
            self._warmup()

    def _warmup(self) -> None:
        """Discard initial sensor frames while auto-exposure/white-balance stabilizes."""
        warmup_n = max(0, self.config.warmup_frames)
        if warmup_n > 0:
            logger.debug("Warming up sensor with %d frame reads...", warmup_n)
            for _ in range(warmup_n):
                if self.cap is not None:
                    ret, _ = self.cap.read()
                    if not ret:
                        break

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera.
        
        Returns:
            Tuple of (success_flag, bgr_frame_or_none).
            If mirror=True in config, the returned frame is horizontally mirrored.
        """
        if not self.is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        if self.config.mirror:
            frame = cv2.flip(frame, 1)

        self._frame_count += 1
        return True, frame

    def read_frame_or_raise(self) -> np.ndarray:
        """Read frame, raising CameraReadError on capture failure."""
        ret, frame = self.read_frame()
        if not ret or frame is None:
            raise CameraReadError("Failed to read frame from active camera stream.")
        return frame

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Generator yielding consecutive BGR frames until stopped or stream fails."""
        while self.is_opened:
            ret, frame = self.read_frame()
            if not ret or frame is None:
                logger.warning("Frame stream interrupted or camera disconnected.")
                break
            yield frame

    def stop(self) -> None:
        """Release the camera hardware and clean up OpenCV resources."""
        if self.cap is not None:
            logger.info("Releasing camera device %d...", self.config.device_id)
            try:
                self.cap.release()
            except Exception as e:
                logger.warning("Error releasing camera: %s", e)
            finally:
                self.cap = None
        self._actual_width = 0
        self._actual_height = 0

    def __enter__(self) -> "CameraManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
