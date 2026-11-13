"""Module 2 Camera Manager & Preview Unit and Integration Tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from app.camera.preview import draw_hud
from app.camera.webcam import (
    CameraConfig,
    CameraError,
    CameraManager,
    CameraNotFoundError,
    CameraReadError,
)


def test_camera_config_defaults():
    """Verify default camera configurations."""
    cfg = CameraConfig()
    assert cfg.device_id == 0
    assert cfg.width == 1280
    assert cfg.height == 720
    assert cfg.fps == 30
    assert cfg.mirror is True
    assert cfg.warmup_frames == 10
    assert cfg.video_source is None


def test_camera_config_from_yaml(tmp_path):
    """Test loading configuration from a YAML file."""
    yaml_content = """
camera:
  device_id: 1
  width: 1920
  height: 1080
  fps: 60
  mirror: false
  warmup_frames: 5
"""
    cfg_file = tmp_path / "custom_cam.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    cfg = CameraConfig.from_yaml(cfg_file)
    assert cfg.device_id == 1
    assert cfg.width == 1920
    assert cfg.height == 1080
    assert cfg.fps == 60
    assert cfg.mirror is False
    assert cfg.warmup_frames == 5


def test_camera_manager_mirroring_logic():
    """Verify that horizontal mirroring properly flips asymmetric pixel coordinates."""
    cfg = CameraConfig(mirror=True, warmup_frames=0)
    manager = CameraManager(cfg)

    # Create synthetic frame with unique left and right pixels
    test_frame = np.zeros((100, 200, 3), dtype=np.uint8)
    test_frame[:, 0:10] = [255, 0, 0]  # Left edge is Blue
    test_frame[:, 190:200] = [0, 255, 0]  # Right edge is Green

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, test_frame.copy())
    mock_cap.get.side_effect = lambda prop: 200 if prop == cv2.CAP_PROP_FRAME_WIDTH else 100

    manager.cap = mock_cap

    ret, flipped = manager.read_frame()
    assert ret is True
    assert flipped is not None

    # After horizontal flip, the blue bar (originally on left) must be on the right edge
    assert np.array_equal(flipped[:, 190:200, 0], np.full((100, 10), 255, dtype=np.uint8))
    # Green bar (originally on right) must now be on the left edge
    assert np.array_equal(flipped[:, 0:10, 1], np.full((100, 10), 255, dtype=np.uint8))


def test_camera_manager_error_on_invalid_device():
    """Ensure opening a non-existent device index raises CameraNotFoundError."""
    with patch("cv2.VideoCapture") as mock_cv:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_cv.return_value = mock_instance

        manager = CameraManager(CameraConfig(device_id=999))
        with pytest.raises(CameraNotFoundError) as exc_info:
            manager.start()
        assert "Failed to open camera device index 999" in str(exc_info.value)


def test_camera_manager_video_source_playback(tmp_path):
    """Test reading from a synthetic video file."""
    video_path = tmp_path / "test_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    for i in range(5):
        frame = np.full((480, 640, 3), i * 40, dtype=np.uint8)
        out.write(frame)
    out.release()

    cfg = CameraConfig(video_source=str(video_path), mirror=False)
    with CameraManager(cfg) as cam:
        assert cam.is_opened is True
        frames = list(cam.stream())
        assert len(frames) == 5
        assert frames[0].shape == (480, 640, 3)


def test_camera_manager_context_lifecycle():
    """Verify context manager calls start() and stop() appropriately."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: 1280,
        cv2.CAP_PROP_FRAME_HEIGHT: 720,
        cv2.CAP_PROP_FPS: 30,
    }.get(prop, 0)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        cfg = CameraConfig(warmup_frames=2)
        with CameraManager(cfg) as cam:
            assert cam.is_opened is True
            assert cam.resolution == (1280, 720)
            ret, frame = cam.read_frame()
            assert ret is True
            assert frame.shape == (720, 1280, 3)

        mock_cap.release.assert_called_once()
        assert cam.is_opened is False


def test_draw_hud_overlay():
    """Verify telemetry HUD rendering keeps frame dimensions and renders cleanly."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    annotated = draw_hud(
        frame=frame,
        fps=30.0,
        resolution=(1280, 720),
        is_mirrored=True,
        is_fullscreen=False,
    )
    assert annotated.shape == (720, 1280, 3)
    assert np.any(annotated[10:30, 10:300] > 0)
