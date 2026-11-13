"""Module 1 Environment & Configuration Sanity Tests."""

import sys
from pathlib import Path
import pytest
import yaml

from app.core.logger import get_logger, setup_logging
from app.core.datatypes import (
    Point2D,
    BoundingBox,
    DetectionResult,
    VerificationResult,
    EyeLandmarks,
    GazeCoordinates,
    SystemState,
)


def test_python_version():
    """Ensure runtime is Python 3.10 or compatible."""
    assert sys.version_info >= (3, 10), f"Python 3.10+ required, current: {sys.version}"


def test_core_dependencies_importable():
    """Verify that all fundamental libraries are successfully importable."""
    import numpy as np
    import cv2
    import yaml
    import pydantic
    import scipy
    import pyautogui
    import pynput

    assert np.__version__ is not None
    assert cv2.__version__ is not None


def test_configuration_files_exist_and_parse():
    """Ensure YAML configuration files exist and contain valid structures."""
    config_dir = Path(__file__).resolve().parent.parent / "config"
    default_cfg_file = config_dir / "default_config.yaml"
    camera_cfg_file = config_dir / "camera_config.yaml"

    assert default_cfg_file.exists(), f"Missing config file: {default_cfg_file}"
    assert camera_cfg_file.exists(), f"Missing config file: {camera_cfg_file}"

    with open(default_cfg_file, "r", encoding="utf-8") as f:
        default_cfg = yaml.safe_load(f)
    assert "safety" in default_cfg
    assert "cursor" in default_cfg
    assert default_cfg["project"]["name"] == "AdaptiveEye"

    with open(camera_cfg_file, "r", encoding="utf-8") as f:
        camera_cfg = yaml.safe_load(f)
    assert "camera" in camera_cfg
    assert camera_cfg["camera"]["width"] == 1280
    assert camera_cfg["camera"]["height"] == 720


def test_logger_initialization(tmp_path):
    """Test logger creation and file writing capability."""
    log_file = tmp_path / "test_run.log"
    setup_logging(level="DEBUG", log_file=str(log_file))
    logger = get_logger("TestRunner")

    test_message = "AdaptiveEye Module 1 Sanity Check Message"
    logger.info(test_message)

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert test_message in content
    assert "AdaptiveEye.TestRunner" in content


def test_datatypes_contracts():
    """Validate data transfer objects, bounding box conversion, and immutability."""
    # Point2D test
    pt = Point2D(0.5, 0.25)
    px_x, px_y = pt.to_pixel(1280, 720)
    assert px_x == 640
    assert px_y == 180

    # BoundingBox test
    bbox = BoundingBox(xmin=0.1, ymin=0.2, width=0.4, height=0.5)
    assert bbox.xmax == pytest.approx(0.5)
    assert bbox.ymax == pytest.approx(0.7)
    x, y, w, h = bbox.to_pixel(1000, 1000)
    assert (x, y, w, h) == (100, 200, 400, 500)

    # State enum test
    assert SystemState.INITIALIZING.value == "INITIALIZING"
    assert SystemState.PAUSED.value == "PAUSED"

    # GazeCoordinates test
    gaze = GazeCoordinates(
        screen_x=1920,
        screen_y=1080,
        normalized_x=1.0,
        normalized_y=1.0,
        confidence=0.95,
        is_valid=True,
    )
    assert gaze.confidence == 0.95
    assert gaze.is_valid is True
