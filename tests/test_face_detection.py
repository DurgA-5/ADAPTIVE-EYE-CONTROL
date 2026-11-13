"""Module 3 Face Detection Unit and Integration Tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from app.core.datatypes import BoundingBox, Point2D
from app.face_detection.detector import (
    FaceDetectionResult,
    FaceDetector,
    FaceDetectorConfig,
)


def test_face_detector_config_defaults():
    """Verify face detector default configurations."""
    cfg = FaceDetectorConfig()
    assert cfg.model_selection == 0
    assert cfg.min_detection_confidence == 0.65
    assert cfg.min_face_size_pixels == 60
    assert cfg.allow_multiple_faces is False
    assert cfg.max_num_faces == 5


def test_face_detector_config_from_yaml(tmp_path):
    """Test loading face detector config from custom YAML."""
    yaml_content = """
face_detection:
  model_selection: 1
  min_detection_confidence: 0.80
  min_face_size_pixels: 80
  allow_multiple_faces: true
  max_num_faces: 3
"""
    cfg_file = tmp_path / "custom_face.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    cfg = FaceDetectorConfig.from_yaml(cfg_file)
    assert cfg.model_selection == 1
    assert cfg.min_detection_confidence == 0.80
    assert cfg.min_face_size_pixels == 80
    assert cfg.allow_multiple_faces is True
    assert cfg.max_num_faces == 3


def test_face_detector_blank_image():
    """Verify behavior on a blank black image (no face detected)."""
    detector = FaceDetector(FaceDetectorConfig(min_detection_confidence=0.5))
    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    result = detector.detect(blank)
    assert result.count == 0
    assert result.primary_face is None
    assert result.is_safe is False
    assert result.multi_face_hazard is False
    assert result.latency_ms > 0
    detector.close()


def test_face_detector_multi_face_safety_logic():
    """Verify that multiple faces trigger safety hold (is_safe=False, multi_face_hazard=True)."""
    detector = FaceDetector(FaceDetectorConfig(min_detection_confidence=0.5, allow_multiple_faces=False))

    # Mock MediaPipe detection returning 2 faces
    mock_det1 = MagicMock()
    mock_det1.score = [0.95]
    mock_det1.location_data.relative_bounding_box.xmin = 0.1
    mock_det1.location_data.relative_bounding_box.ymin = 0.1
    mock_det1.location_data.relative_bounding_box.width = 0.3
    mock_det1.location_data.relative_bounding_box.height = 0.4
    mock_det1.location_data.relative_keypoints = []

    mock_det2 = MagicMock()
    mock_det2.score = [0.88]
    mock_det2.location_data.relative_bounding_box.xmin = 0.6
    mock_det2.location_data.relative_bounding_box.ymin = 0.2
    mock_det2.location_data.relative_bounding_box.width = 0.2
    mock_det2.location_data.relative_bounding_box.height = 0.3
    mock_det2.location_data.relative_keypoints = []

    mock_mp_results = MagicMock()
    mock_mp_results.detections = [mock_det1, mock_det2]

    with patch.object(detector.detector, "process", return_value=mock_mp_results):
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = detector.detect(dummy_frame)

        assert result.count == 2
        assert result.multi_face_hazard is True
        assert result.is_safe is False
        assert result.primary_face is not None
        # Largest face by area should be primary (0.3*0.4 = 0.12 > 0.2*0.3 = 0.06)
        assert result.primary_face.confidence == 0.95

    detector.close()


def test_face_detector_single_safe_face():
    """Verify single confident face marks system state as safe."""
    detector = FaceDetector(FaceDetectorConfig(min_detection_confidence=0.6))

    mock_det = MagicMock()
    mock_det.score = [0.92]
    mock_det.location_data.relative_bounding_box.xmin = 0.3
    mock_det.location_data.relative_bounding_box.ymin = 0.2
    mock_det.location_data.relative_bounding_box.width = 0.4
    mock_det.location_data.relative_bounding_box.height = 0.5

    mock_kp = MagicMock()
    mock_kp.x = 0.4
    mock_kp.y = 0.35
    mock_det.location_data.relative_keypoints = [mock_kp]

    mock_mp_results = MagicMock()
    mock_mp_results.detections = [mock_det]

    with patch.object(detector.detector, "process", return_value=mock_mp_results):
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = detector.detect(dummy_frame)

        assert result.count == 1
        assert result.is_safe is True
        assert result.multi_face_hazard is False
        assert result.primary_face is not None
        assert result.primary_face.confidence == 0.92
        assert len(result.primary_face.landmarks) == 1

    detector.close()


def test_face_detector_low_confidence_filtering():
    """Ensure detections below confidence threshold are discarded."""
    detector = FaceDetector(FaceDetectorConfig(min_detection_confidence=0.70))

    mock_det = MagicMock()
    mock_det.score = [0.55]  # Below 0.70 threshold
    mock_mp_results = MagicMock()
    mock_mp_results.detections = [mock_det]

    with patch.object(detector.detector, "process", return_value=mock_mp_results):
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = detector.detect(dummy_frame)

        assert result.count == 0
        assert result.primary_face is None
        assert result.is_safe is False

    detector.close()


def test_face_detector_draw_annotations():
    """Ensure annotations render properly and preserve image dimensions."""
    detector = FaceDetector()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Mock detection result with 1 face
    bbox = BoundingBox(xmin=0.2, ymin=0.2, width=0.3, height=0.3)
    kps = [Point2D(0.3, 0.3), Point2D(0.4, 0.3)]
    det = MagicMock(bbox=bbox, confidence=0.91, landmarks=kps)

    mock_res = FaceDetectionResult(
        faces=[det],
        count=1,
        primary_face=det,
        is_safe=True,
        multi_face_hazard=False,
        latency_ms=8.5,
    )

    annotated = detector.draw_annotations(frame, mock_res)
    assert annotated.shape == (720, 1280, 3)
    # Check that status banner was rendered
    assert np.any(annotated[5:30, 10:200] > 0)
    detector.close()
