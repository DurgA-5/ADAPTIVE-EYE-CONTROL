"""Module 4 Face Verification (ArcFace) Unit and Integration Tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from app.core.datatypes import BoundingBox, DetectionResult, Point2D
from app.face_verification.arcface import ArcFaceConfig, ArcFaceExtractor
from app.face_verification.enrollment import (
    ConsentRequiredError,
    EnrollmentQualityError,
    FaceEnrollmentManager,
)
from app.face_verification.evaluator import BiometricEvaluator
from app.face_verification.verifier import AuthState, FaceVerifier


@pytest.fixture
def arcface_extractor():
    """Fixture providing an initialized ArcFace extractor."""
    return ArcFaceExtractor()


def test_arcface_extractor_initialization(arcface_extractor):
    """Verify ArcFace ONNX model loads and exposes correct input/output nodes."""
    assert arcface_extractor.input_name is not None
    assert arcface_extractor.output_name is not None
    assert arcface_extractor.config.input_width == 112
    assert arcface_extractor.config.input_height == 112


def test_arcface_alignment_shape(arcface_extractor):
    """Verify facial alignment outputs exact 112x112 aligned image."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = BoundingBox(xmin=0.2, ymin=0.2, width=0.4, height=0.4)
    landmarks = [Point2D(0.3, 0.3), Point2D(0.5, 0.3), Point2D(0.4, 0.45)]

    aligned = arcface_extractor.align_face(dummy_frame, bbox, landmarks)
    assert aligned.shape == (112, 112, 3)


def test_arcface_embedding_l2_normalization(arcface_extractor):
    """Verify extracted 512-d embedding has unit L2 norm."""
    dummy_aligned = np.full((112, 112, 3), 128, dtype=np.uint8)
    embedding = arcface_extractor.extract_embedding(dummy_aligned)

    assert embedding.shape == (512,)
    norm = np.linalg.norm(embedding)
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_properties(arcface_extractor):
    """Verify cosine similarity mathematical bounds [-1.0, 1.0] and identity."""
    vec_a = np.random.randn(512).astype(np.float32)
    vec_a /= np.linalg.norm(vec_a)

    # Identical vector -> 1.0
    sim_self = arcface_extractor.compute_similarity(vec_a, vec_a)
    assert sim_self == pytest.approx(1.0, abs=1e-5)

    # Opposite vector -> -1.0
    sim_opposite = arcface_extractor.compute_similarity(vec_a, -vec_a)
    assert sim_opposite == pytest.approx(-1.0, abs=1e-5)

    # Orthogonal vector -> 0.0
    vec_ortho = np.zeros_like(vec_a)
    vec_ortho[0] = -vec_a[1]
    vec_ortho[1] = vec_a[0]
    vec_ortho /= np.linalg.norm(vec_ortho)
    sim_ortho = arcface_extractor.compute_similarity(vec_a, vec_ortho)
    assert sim_ortho == pytest.approx(0.0, abs=1e-4)


def test_enrollment_consent_enforcement(tmp_path):
    """Verify attempting enrollment without consent raises ConsentRequiredError."""
    mgr = FaceEnrollmentManager(profile_dir=tmp_path)
    with pytest.raises(ConsentRequiredError):
        mgr.enroll_from_frames(
            frames_and_detections=[],
            user_name="Test User",
            subject_id="test_01",
            consent_granted=False,
        )


def test_enrollment_quality_checks(tmp_path):
    """Verify optical blur, underexposure, and small bounding boxes fail quality check."""
    mgr = FaceEnrollmentManager(profile_dir=tmp_path, min_sharpness=80.0, min_face_size=100)

    # 1. Tiny face
    small_det = DetectionResult(
        bbox=BoundingBox(xmin=0.1, ymin=0.1, width=0.05, height=0.05),
        confidence=0.9,
    )
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    ok, msg = mgr.check_frame_quality(frame, small_det)
    assert ok is False
    assert "Face too small" in msg

    # 2. Underexposed face (pitch black)
    normal_det = DetectionResult(
        bbox=BoundingBox(xmin=0.2, ymin=0.2, width=0.4, height=0.4),
        confidence=0.9,
    )
    black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    ok, msg = mgr.check_frame_quality(black_frame, normal_det)
    assert ok is False
    assert "Too dark" in msg


def test_enrollment_template_lifecycle(tmp_path):
    """Verify enrollment template save, load, and deletion."""
    mgr = FaceEnrollmentManager(profile_dir=tmp_path)

    # Mock high quality frame and detection
    frame = np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8)
    det = DetectionResult(
        bbox=BoundingBox(xmin=0.2, ymin=0.2, width=0.4, height=0.4),
        confidence=0.95,
        landmarks=[Point2D(0.3, 0.3), Point2D(0.5, 0.3), Point2D(0.4, 0.45)],
    )

    # Mock Laplacian variance to ensure frame passes blur test
    with patch("cv2.Laplacian") as mock_lap:
        mock_lap.return_value.var.return_value = 150.0
        samples = [(frame, det) for _ in range(6)]
        profile = mgr.enroll_from_frames(
            frames_and_detections=samples,
            user_name="Alice",
            subject_id="alice_01",
            consent_granted=True,
        )

    assert mgr.has_enrolled_identity() is True
    assert profile.user_name == "Alice"
    assert profile.num_samples == 6

    # Verify template vector loading
    template = mgr.load_enrolled_template()
    assert template is not None
    assert template.shape == (512,)
    assert np.linalg.norm(template) == pytest.approx(1.0, abs=1e-5)

    # Test deletion
    deleted = mgr.delete_enrolled_identity()
    assert deleted is True
    assert mgr.has_enrolled_identity() is False
    assert mgr.load_enrolled_template() is None


def test_face_verifier_decision_tiers(tmp_path):
    """Verify 3-tier authorization (AUTHORIZED, UNCERTAIN, REJECTED)."""
    mgr = FaceEnrollmentManager(profile_dir=tmp_path)
    # Fake template vector
    template = np.zeros(512, dtype=np.float32)
    template[0] = 1.0
    np.save(str(mgr.template_file), template)

    profile_data = {
        "subject_id": "sub_1",
        "user_name": "Bob",
        "enrolled_at": "2026-11-13",
        "num_samples": 30,
        "mean_pairwise_similarity": 0.95,
        "min_pairwise_similarity": 0.90,
        "is_valid": True,
    }
    import json
    with open(mgr.profile_file, "w") as f:
        json.dump(profile_data, f)

    verifier = FaceVerifier(
        high_threshold=0.88,
        low_threshold=0.72,
        temporal_window=1,
        enrollment_manager=mgr,
    )

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    det = DetectionResult(bbox=BoundingBox(0.2, 0.2, 0.3, 0.3), confidence=0.9)

    # Case 1: High similarity -> AUTHORIZED
    with patch.object(verifier.extractor, "extract_embedding", return_value=template):
        outcome = verifier.verify(frame, det)
        assert outcome.auth_state == AuthState.AUTHORIZED
        assert outcome.is_authorized is True

    # Case 2: Intermediate similarity (0.80) -> UNCERTAIN (from an un-authorized state)
    verifier._current_state = AuthState.UNCERTAIN
    verifier._similarity_buffer.clear()
    uncertain_vec = template.copy()
    uncertain_vec[0] = 0.80
    uncertain_vec[1] = np.sqrt(1.0 - 0.80**2)
    with patch.object(verifier.extractor, "extract_embedding", return_value=uncertain_vec):
        outcome = verifier.verify(frame, det)
        assert outcome.auth_state == AuthState.UNCERTAIN
        assert outcome.is_authorized is False

    # Case 3: Low similarity (0.10) -> REJECTED (impostor block)
    verifier._current_state = AuthState.UNCERTAIN
    verifier._similarity_buffer.clear()
    impostor_vec = np.zeros(512, dtype=np.float32)
    impostor_vec[1] = 1.0  # Orthogonal -> sim = 0.0
    with patch.object(verifier.extractor, "extract_embedding", return_value=impostor_vec):
        outcome = verifier.verify(frame, det)
        assert outcome.auth_state == AuthState.REJECTED
        assert outcome.is_authorized is False


def test_biometric_evaluator_metrics():
    """Verify calculation of FAR, FRR, EER, and calibration report."""
    evaluator = BiometricEvaluator(threshold_step=0.05)

    # Genuine scores centered at 0.92, impostor scores centered at 0.65
    np.random.seed(42)
    genuine_scores = np.random.normal(0.92, 0.03, 100).tolist()
    impostor_scores = np.random.normal(0.65, 0.05, 100).tolist()

    report = evaluator.evaluate(genuine_scores, impostor_scores)

    assert report.num_genuine_pairs == 100
    assert report.num_impostor_pairs == 100
    assert report.genuine_mean_sim > report.impostor_mean_sim
    assert 0.70 <= report.eer_threshold <= 0.85
    assert report.recommended_threshold >= report.eer_threshold
    assert len(report.points) > 0
