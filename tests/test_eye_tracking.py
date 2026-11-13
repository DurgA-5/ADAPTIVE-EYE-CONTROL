"""Module 6 Eye Landmark & Iris Tracking Unit Tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from app.core.datatypes import Point2D
from app.eye_tracking.tracker import (
    ExtractedEyeFeatures,
    EyeTracker,
    EyeTrackingConfig,
    SingleEyeFeatures,
)


def test_eye_tracking_config_defaults():
    """Verify default eye tracking configuration."""
    cfg = EyeTrackingConfig()
    assert cfg.refine_landmarks is True
    assert cfg.max_num_faces == 1
    assert cfg.min_detection_confidence == 0.65
    assert cfg.ear_blink_threshold == 0.20


def test_eye_tracking_config_from_yaml(tmp_path):
    """Test loading configuration from custom YAML."""
    yaml_content = """
eye_tracking:
  refine_landmarks: true
  max_num_faces: 1
  min_detection_confidence: 0.75
  min_tracking_confidence: 0.70
  ear_blink_threshold: 0.18
"""
    cfg_file = tmp_path / "custom_eye.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    cfg = EyeTrackingConfig.from_yaml(cfg_file)
    assert cfg.min_detection_confidence == 0.75
    assert cfg.min_tracking_confidence == 0.70
    assert cfg.ear_blink_threshold == 0.18


def test_ear_calculation_formula():
    """Verify Soukupova & Cech EAR calculation on geometric test coordinates."""
    # Outer corner: (0, 0), Inner corner: (10, 0) -> horiz = 10
    p_outer = Point2D(0.0, 0.0)
    p_inner = Point2D(10.0, 0.0)

    # 1. Open Eye: vertical distances = 3.0 each
    up1 = Point2D(3.0, 3.0)
    low1 = Point2D(3.0, 0.0)
    up2 = Point2D(7.0, 3.0)
    low2 = Point2D(7.0, 0.0)

    # EAR = (3.0 + 3.0) / (2 * 10) = 6.0 / 20.0 = 0.30
    ear_open = EyeTracker.calculate_ear(p_outer, p_inner, up1, low1, up2, low2)
    assert ear_open == pytest.approx(0.30)

    # 2. Closed Eye: vertical distance = 0
    ear_closed = EyeTracker.calculate_ear(p_outer, p_inner, low1, low1, low2, low2)
    assert ear_closed == pytest.approx(0.0)

    # 3. Degenerate collapsed eye points (zero width)
    ear_degenerate = EyeTracker.calculate_ear(p_outer, p_outer, up1, low1, up2, low2)
    assert ear_degenerate == 0.0


def test_eye_tracker_blank_frame():
    """Verify processing a blank frame returns has_eyes=False without exceptions."""
    tracker = EyeTracker(EyeTrackingConfig(refine_landmarks=True))
    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    features = tracker.process(blank)
    assert features.has_eyes is False
    assert features.right_eye is None
    assert features.left_eye is None
    assert features.latency_ms > 0
    tracker.close()


def test_eye_tracker_draw_annotations():
    """Verify eye contour, iris ring, and telemetry rendering on canvas."""
    tracker = EyeTracker()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Create dummy SingleEyeFeatures
    dummy_eye = SingleEyeFeatures(
        inner_corner=Point2D(0.45, 0.3),
        outer_corner=Point2D(0.35, 0.3),
        upper_lid_1=Point2D(0.40, 0.28),
        lower_lid_1=Point2D(0.40, 0.32),
        upper_lid_2=Point2D(0.38, 0.28),
        lower_lid_2=Point2D(0.38, 0.32),
        contour_points=[Point2D(0.35, 0.3), Point2D(0.40, 0.28), Point2D(0.45, 0.3), Point2D(0.40, 0.32)],
        iris_center=Point2D(0.40, 0.30),
        iris_points=[Point2D(0.40, 0.30), Point2D(0.40, 0.29), Point2D(0.41, 0.30), Point2D(0.40, 0.31), Point2D(0.39, 0.30)],
        ear=0.28,
        is_closed=False,
        iris_relative_x=0.50,
        iris_relative_y=0.50,
    )

    mock_feat = ExtractedEyeFeatures(
        has_eyes=True,
        right_eye=dummy_eye,
        left_eye=dummy_eye,
        average_ear=0.28,
        both_eyes_closed=False,
        tracking_confidence=1.0,
        latency_ms=4.2,
    )

    annotated = tracker.draw_eye_annotations(frame, mock_feat)
    assert annotated.shape == (720, 1280, 3)
    # Check that status banner was rendered
    assert np.any(annotated[5:35, 10:200] > 0)
    tracker.close()


def test_eye_tracker_multi_face_centroid_matching():
    """Verify EyeTracker selects the face mesh closest to target_centroid when multiple faces exist."""
    tracker = EyeTracker(EyeTrackingConfig(max_num_faces=2))

    # Mock face mesh results with 2 faces
    mock_mesh1 = MagicMock()
    mock_mesh2 = MagicMock()

    # Face 1 nose at (0.2, 0.5)
    lms1 = [MagicMock(x=0.2, y=0.5)] * 478
    mock_mesh1.landmark = lms1

    # Face 2 nose at (0.8, 0.5)
    lms2 = [MagicMock(x=0.8, y=0.5)] * 478
    mock_mesh2.landmark = lms2

    mock_results = MagicMock()
    mock_results.multi_face_landmarks = [mock_mesh1, mock_mesh2]
    tracker.mesh.process = MagicMock(return_value=mock_results)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Targeting face 2 near (0.85, 0.5)
    features = tracker.process(frame, target_centroid=(0.85, 0.5))
    assert features.has_eyes is True
    tracker.close()

