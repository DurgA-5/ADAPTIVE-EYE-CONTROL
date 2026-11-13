"""Module 5 Person Tracking & Identity Locking Unit and Safety Tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from app.core.datatypes import BoundingBox, DetectionResult, Point2D
from app.face_detection.detector import FaceDetectionResult
from app.face_verification.verifier import AuthState, FaceVerifier, LiveVerificationOutcome
from app.person_tracking.tracker import (
    IdentityLockTracker,
    PersonTrackingConfig,
    TrackedPerson,
    TrackingState,
)


@pytest.fixture
def mock_verifier():
    """Mock FaceVerifier for controlled testing."""
    verifier = MagicMock(spec=FaceVerifier)
    verifier.is_enrolled = True
    verifier.high_threshold = 0.88
    verifier.low_threshold = 0.72
    return verifier


def test_person_tracking_config_defaults():
    """Verify default tracking configuration."""
    cfg = PersonTrackingConfig()
    assert cfg.min_iou_continuity == 0.35
    assert cfg.max_centroid_distance_normalized == 0.25
    assert cfg.max_lost_frames_tolerance == 15
    assert cfg.reverification_interval_frames == 45


def test_person_tracking_config_from_yaml(tmp_path):
    """Test loading configuration from YAML."""
    yaml_content = """
person_tracking:
  min_iou_continuity: 0.40
  max_centroid_distance_normalized: 0.20
  max_lost_frames_tolerance: 10
  reverification_interval_frames: 30
"""
    cfg_file = tmp_path / "custom_tracker.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    cfg = PersonTrackingConfig.from_yaml(cfg_file)
    assert cfg.min_iou_continuity == 0.40
    assert cfg.max_centroid_distance_normalized == 0.20
    assert cfg.max_lost_frames_tolerance == 10
    assert cfg.reverification_interval_frames == 30


def test_iou_and_distance_math(mock_verifier):
    """Verify Intersection over Union (IoU) and centroid distance formulas."""
    tracker = IdentityLockTracker(verifier=mock_verifier)

    box_a = BoundingBox(xmin=0.1, ymin=0.1, width=0.2, height=0.2)
    # Identical box -> IoU = 1.0, distance = 0.0
    assert tracker.compute_iou(box_a, box_a) == pytest.approx(1.0)
    assert tracker.compute_centroid_distance(box_a, box_a) == pytest.approx(0.0)

    # Disjoint box -> IoU = 0.0, distance > 0.0
    box_b = BoundingBox(xmin=0.5, ymin=0.5, width=0.2, height=0.2)
    assert tracker.compute_iou(box_a, box_b) == pytest.approx(0.0)
    assert tracker.compute_centroid_distance(box_a, box_b) > 0.4

    # 50% coordinate overlap
    box_c = BoundingBox(xmin=0.2, ymin=0.1, width=0.2, height=0.2)
    # Intersection: w=0.1, h=0.2 -> area = 0.02
    # Area A = 0.04, Area C = 0.04 -> union = 0.06 -> IoU = 0.02 / 0.06 = 0.3333
    assert tracker.compute_iou(box_a, box_c) == pytest.approx(1.0 / 3.0, abs=1e-3)


def test_identity_lock_acquisition(mock_verifier):
    """Verify successful authentication transitions tracker to LOCKED_AUTHORIZED."""
    # Configure mock verifier to authorize candidate
    mock_verifier.verify.return_value = LiveVerificationOutcome(
        auth_state=AuthState.AUTHORIZED,
        is_authorized=True,
        instant_similarity=0.92,
        smoothed_similarity=0.92,
        high_threshold=0.88,
        low_threshold=0.72,
        user_name="Durga Prasad",
    )

    tracker = IdentityLockTracker(verifier=mock_verifier)
    assert tracker.current_state == TrackingState.SEARCHING

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    face_det = DetectionResult(
        bbox=BoundingBox(0.3, 0.3, 0.3, 0.3),
        confidence=0.95,
        landmarks=[Point2D(0.4, 0.4)],
    )
    det_result = FaceDetectionResult(faces=[face_det], count=1, primary_face=face_det)

    result = tracker.update(frame, det_result)

    assert result.tracking_state == TrackingState.LOCKED_AUTHORIZED
    assert result.is_safe_for_control is True
    assert result.user_name == "Durga Prasad"
    assert result.consecutive_locked_frames == 1


def test_anti_takeover_multi_person_scene(mock_verifier):
    """Ensure tracker maintains lock on primary user and ignores intruder face."""
    # First establish lock
    mock_verifier.verify.return_value = LiveVerificationOutcome(
        auth_state=AuthState.AUTHORIZED,
        is_authorized=True,
        instant_similarity=0.91,
        smoothed_similarity=0.91,
        high_threshold=0.88,
        low_threshold=0.72,
        user_name="Authorized User",
    )
    tracker = IdentityLockTracker(verifier=mock_verifier)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Initial frame: single authorized user at center
    primary_box = BoundingBox(0.4, 0.3, 0.2, 0.3)
    det1 = DetectionResult(bbox=primary_box, confidence=0.92)
    tracker.update(frame, FaceDetectionResult(faces=[det1], count=1, primary_face=det1))
    assert tracker.is_locked is True

    # Next frame: Second person (intruder) enters at x=0.05
    intruder_box = BoundingBox(0.05, 0.3, 0.2, 0.3)
    det_intruder = DetectionResult(bbox=intruder_box, confidence=0.89)

    # Primary person shifted slightly to x=0.42 (high IoU continuity)
    primary_shifted = BoundingBox(0.42, 0.3, 0.2, 0.3)
    det_primary_shifted = DetectionResult(bbox=primary_shifted, confidence=0.93)

    two_faces = [det_intruder, det_primary_shifted]
    result2 = tracker.update(frame, FaceDetectionResult(faces=two_faces, count=2, primary_face=det_primary_shifted))

    # Anti-takeover check: Lock MUST stay on primary user (x=0.42), NOT jump to intruder (x=0.05)
    assert result2.tracking_state == TrackingState.LOCKED_AUTHORIZED
    assert result2.is_safe_for_control is True
    assert result2.bbox == primary_shifted
    assert result2.bbox != intruder_box


def test_occlusion_and_reverification_on_reacquire(mock_verifier):
    """Ensure face disappearance revokes lock and mandates re-authentication before control resumes."""
    mock_verifier.verify.return_value = LiveVerificationOutcome(
        auth_state=AuthState.AUTHORIZED,
        is_authorized=True,
        instant_similarity=0.93,
        smoothed_similarity=0.93,
        high_threshold=0.88,
        low_threshold=0.72,
        user_name="Primary User",
    )
    cfg = PersonTrackingConfig(max_lost_frames_tolerance=3)
    tracker = IdentityLockTracker(config=cfg, verifier=mock_verifier)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Lock face
    box = BoundingBox(0.3, 0.3, 0.3, 0.3)
    det = DetectionResult(bbox=box, confidence=0.9)
    tracker.update(frame, FaceDetectionResult(faces=[det], count=1, primary_face=det))
    assert tracker.is_locked is True

    # Frames 1-3: Face missing (coasting window, control paused)
    empty_det = FaceDetectionResult(faces=[], count=0)
    for _ in range(3):
        res = tracker.update(frame, empty_det)
        assert res.is_safe_for_control is False

    # Frame 4: Exceeds tolerance of 3 -> lock revoked
    res4 = tracker.update(frame, empty_det)
    assert res4.tracking_state == TrackingState.TRACKING_LOST
    assert tracker.is_locked is False

    # Frame 5: Face returns! But verifier fails or is uncertain
    mock_verifier.verify.return_value = LiveVerificationOutcome(
        auth_state=AuthState.UNCERTAIN,
        is_authorized=False,
        instant_similarity=0.75,
        smoothed_similarity=0.75,
        high_threshold=0.88,
        low_threshold=0.72,
    )
    res5 = tracker.update(frame, FaceDetectionResult(faces=[det], count=1, primary_face=det))
    # Safety Rule 6: Control MUST NOT be granted!
    assert res5.is_safe_for_control is False
    assert res5.tracking_state == TrackingState.REVERIFY_REQUIRED


def test_draw_tracking_overlay(mock_verifier):
    """Ensure visual tracking HUD overlays preserve frame dimensions."""
    tracker = IdentityLockTracker(verifier=mock_verifier)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    person = TrackedPerson(
        tracking_state=TrackingState.LOCKED_AUTHORIZED,
        is_safe_for_control=True,
        bbox=BoundingBox(0.3, 0.3, 0.3, 0.3),
        user_name="Alice",
        similarity_score=0.94,
        consecutive_locked_frames=12,
        status_message="Tracking active",
    )
    annotated = tracker.draw_tracking_overlay(frame, person)
    assert annotated.shape == (720, 1280, 3)
    assert np.any(annotated[10:35, 10:200] > 0)
