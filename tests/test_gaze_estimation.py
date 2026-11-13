"""Module 7 Gaze Estimation Unit Tests."""

from pathlib import Path
import numpy as np
import pytest

from app.core.datatypes import Point2D
from app.eye_tracking.tracker import ExtractedEyeFeatures, SingleEyeFeatures
from app.gaze_estimation.estimator import GazeConfig, GazeEstimator
from app.gaze_estimation.head_pose import HeadPoseAngles, HeadPoseEstimator


def _create_mock_single_eye(
    iris_rel_x: float = 0.50,
    iris_rel_y: float = 0.50,
    is_closed: bool = False,
    ear: float = 0.30,
) -> SingleEyeFeatures:
    """Helper to synthesize eye features for testing."""
    inner = Point2D(0.40, 0.40)
    outer = Point2D(0.45, 0.40)
    iris = Point2D(0.425, 0.40)
    return SingleEyeFeatures(
        inner_corner=inner,
        outer_corner=outer,
        upper_lid_1=Point2D(0.42, 0.38),
        lower_lid_1=Point2D(0.42, 0.42),
        upper_lid_2=Point2D(0.43, 0.38),
        lower_lid_2=Point2D(0.43, 0.42),
        contour_points=[outer, inner],
        iris_center=iris,
        iris_points=[iris],
        ear=ear,
        is_closed=is_closed,
        iris_relative_x=iris_rel_x,
        iris_relative_y=iris_rel_y,
    )


def _create_mock_landmarks(count: int = 478) -> list:
    """Generate mock 478 MediaPipe landmarks around center of frame."""
    landmarks = []
    for i in range(count):
        # Create subtle geometric separation for canonical solvePnP points
        landmarks.append(Point2D(x=0.50 + (i % 20) * 0.005, y=0.50 + (i // 20) * 0.005))
    # Explicit landmarks for solvePnP points [4, 152, 263, 33, 291, 61]
    landmarks[4] = Point2D(0.50, 0.50)    # Nose
    landmarks[152] = Point2D(0.50, 0.70)  # Chin
    landmarks[263] = Point2D(0.60, 0.45)  # Left eye outer
    landmarks[33] = Point2D(0.40, 0.45)   # Right eye outer
    landmarks[291] = Point2D(0.55, 0.60)  # Left mouth
    landmarks[61] = Point2D(0.45, 0.60)   # Right mouth
    return landmarks


def test_gaze_config_defaults():
    """Verify default GazeConfig values."""
    cfg = GazeConfig()
    assert cfg.estimation_approach == "compound"
    assert cfg.iris_weight == 0.65
    assert cfg.head_pose_weight == 0.35
    assert cfg.sensitivity_x == 2.8
    assert cfg.sensitivity_y == 3.2
    assert cfg.center_bias_x == 0.50
    assert cfg.center_bias_y == 0.50
    assert cfg.screen_width == 1920
    assert cfg.screen_height == 1080


def test_gaze_config_from_yaml(tmp_path: Path):
    """Test loading GazeConfig from YAML."""
    yaml_content = """
gaze_estimation:
  estimation_approach: "pure_iris"
  iris_weight: 0.80
  head_pose_weight: 0.20
  sensitivity_x: 2.5
  sensitivity_y: 2.1
  center_bias_x: 0.48
  center_bias_y: 0.52
"""
    cfg_file = tmp_path / "custom_gaze.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    cfg = GazeConfig.from_yaml(cfg_file)
    assert cfg.estimation_approach == "pure_iris"
    assert cfg.iris_weight == 0.80
    assert cfg.head_pose_weight == 0.20
    assert cfg.sensitivity_x == 2.5
    assert cfg.sensitivity_y == 2.1
    assert cfg.center_bias_x == 0.48
    assert cfg.center_bias_y == 0.52


def test_head_pose_estimator_insufficient_landmarks():
    """HeadPoseEstimator should return None when landmarks list is insufficient."""
    estimator = HeadPoseEstimator()
    assert estimator.estimate([], 1280, 720) is None
    assert estimator.estimate([Point2D(0.5, 0.5)] * 50, 1280, 720) is None


def test_head_pose_estimator_valid_solvepnp():
    """HeadPoseEstimator should return valid Euler angles for canonical face points."""
    estimator = HeadPoseEstimator()
    landmarks = _create_mock_landmarks(478)
    angles = estimator.estimate(landmarks, 1280, 720)

    assert angles is not None
    assert isinstance(angles.pitch, float)
    assert isinstance(angles.yaw, float)
    assert isinstance(angles.roll, float)
    assert angles.rvec is not None
    assert angles.tvec is not None


def test_head_pose_draw_axes():
    """HeadPoseEstimator.draw_axes should render coordinate lines without exceptions."""
    estimator = HeadPoseEstimator()
    landmarks = _create_mock_landmarks(478)
    angles = estimator.estimate(landmarks, 1280, 720)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    estimator.draw_axes(canvas, angles, landmarks, 1280, 720, length=50.0)
    assert np.any(canvas > 0)


def test_gaze_estimator_pure_iris_neutral():
    """GazeEstimator in pure iris mode centered at (0.50, 0.50) should yield screen center."""
    cfg = GazeConfig(estimation_approach="pure_iris", center_bias_x=0.50, center_bias_y=0.50)
    estimator = GazeEstimator(cfg)

    eye_features = ExtractedEyeFeatures(
        has_eyes=True,
        right_eye=_create_mock_single_eye(iris_rel_x=0.50, iris_rel_y=0.50),
        left_eye=_create_mock_single_eye(iris_rel_x=0.50, iris_rel_y=0.50),
        average_ear=0.30,
        both_eyes_closed=False,
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    res = estimator.estimate(frame, eye_features)
    assert res.is_valid is True
    assert res.normalized_x == pytest.approx(0.50, abs=0.01)
    assert res.normalized_y == pytest.approx(0.50, abs=0.01)
    assert res.screen_x in (959, 960)
    assert res.screen_y in (539, 540)
    assert res.confidence >= 0.70


def test_gaze_estimator_compound_mode():
    """Compound mode fuses iris shift and head pose without crashing."""
    cfg = GazeConfig(estimation_approach="compound")
    estimator = GazeEstimator(cfg)

    eye_features = ExtractedEyeFeatures(
        has_eyes=True,
        right_eye=_create_mock_single_eye(iris_rel_x=0.60, iris_rel_y=0.55),
        left_eye=_create_mock_single_eye(iris_rel_x=0.60, iris_rel_y=0.55),
        average_ear=0.30,
        both_eyes_closed=False,
    )
    landmarks = _create_mock_landmarks(478)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    res = estimator.estimate(frame, eye_features, landmarks)
    assert res.is_valid is True
    assert 0.0 <= res.normalized_x <= 1.0
    assert 0.0 <= res.normalized_y <= 1.0
    assert res.head_angles is not None


def test_gaze_invalidation_when_eyes_closed():
    """Safety interlock: gaze must be marked invalid immediately when eyes are closed or lost."""
    estimator = GazeEstimator()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # 1. both_eyes_closed is True
    closed_features = ExtractedEyeFeatures(
        has_eyes=True,
        right_eye=_create_mock_single_eye(is_closed=True, ear=0.05),
        left_eye=_create_mock_single_eye(is_closed=True, ear=0.05),
        average_ear=0.05,
        both_eyes_closed=True,
    )
    res_closed = estimator.estimate(frame, closed_features)
    assert res_closed.is_valid is False
    assert res_closed.confidence == 0.0
    assert "CLOSED OR OCCLUDED" in res_closed.status_message

    # 2. has_eyes is False
    no_eyes_features = ExtractedEyeFeatures(has_eyes=False)
    res_no_eyes = estimator.estimate(frame, no_eyes_features)
    assert res_no_eyes.is_valid is False
    assert res_no_eyes.confidence == 0.0


def test_gaze_estimator_clamping():
    """Extreme iris or head values must be strictly clamped to [0.0, 1.0]."""
    estimator = GazeEstimator()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    extreme_features = ExtractedEyeFeatures(
        has_eyes=True,
        right_eye=_create_mock_single_eye(iris_rel_x=1.5, iris_rel_y=2.0),
        left_eye=_create_mock_single_eye(iris_rel_x=1.5, iris_rel_y=2.0),
        average_ear=0.35,
        both_eyes_closed=False,
    )
    res = estimator.estimate(frame, extreme_features)
    assert res.is_valid is True
    assert res.normalized_x in (0.0, 1.0)
    assert res.normalized_y == 1.0
    assert res.screen_x in (0, estimator.config.screen_width - 1)
    assert res.screen_y == estimator.config.screen_height - 1


def test_draw_gaze_overlay_rendering():
    """Overlay drawing should run cleanly for both valid and invalid gaze results."""
    estimator = GazeEstimator()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    eye_features = ExtractedEyeFeatures(
        has_eyes=True,
        right_eye=_create_mock_single_eye(),
        left_eye=_create_mock_single_eye(),
        average_ear=0.30,
        both_eyes_closed=False,
    )
    landmarks = _create_mock_landmarks(478)

    # Valid result
    valid_res = estimator.estimate(frame, eye_features, landmarks)
    out_valid = estimator.draw_gaze_overlay(frame, valid_res, eye_features, landmarks)
    assert out_valid.shape == frame.shape
    assert np.any(out_valid > 0)

    # Invalid result (eyes closed)
    closed_features = ExtractedEyeFeatures(has_eyes=False, both_eyes_closed=True)
    invalid_res = estimator.estimate(frame, closed_features)
    out_invalid = estimator.draw_gaze_overlay(frame, invalid_res, closed_features)
    assert out_invalid.shape == frame.shape
    assert np.any(out_invalid > 0)


def test_horizontal_gaze_direction():
    """Verify looking left moves gaze to screen left and looking right moves gaze to screen right."""
    estimator = GazeEstimator(GazeConfig(invert_x=True))
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # In mirrored image, looking left shifts iris relative coordinate > neutral center
    left_features = ExtractedEyeFeatures(
        has_eyes=True,
        right_eye=_create_mock_single_eye(iris_rel_x=0.55),
        left_eye=_create_mock_single_eye(iris_rel_x=0.55),
        average_ear=0.30,
        both_eyes_closed=False,
    )
    res_left = estimator.estimate(frame, left_features)
    assert res_left.normalized_x < 0.50
    assert res_left.screen_x < estimator.config.screen_width // 2

    estimator.reset_smoothing()

    # In mirrored image, looking right shifts iris relative coordinate < neutral center
    right_features = ExtractedEyeFeatures(
        has_eyes=True,
        right_eye=_create_mock_single_eye(iris_rel_x=0.45),
        left_eye=_create_mock_single_eye(iris_rel_x=0.45),
        average_ear=0.30,
        both_eyes_closed=False,
    )
    res_right = estimator.estimate(frame, right_features)
    assert res_right.normalized_x > 0.50
    assert res_right.screen_x > estimator.config.screen_width // 2
