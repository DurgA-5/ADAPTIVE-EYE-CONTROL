"""Module 9 Adaptive Calibration & Drift Compensation Unit Tests."""

from pathlib import Path
import numpy as np
import pytest

from app.adaptive_calibration.drift_compensator import (
    AdaptiveCalibrator,
    AdaptiveConfig,
    DriftState,
    HeadPostureAnchor,
)
from app.core.datatypes import Point2D
from app.eye_tracking.tracker import ExtractedEyeFeatures, SingleEyeFeatures
from app.gaze_estimation.head_pose import HeadPoseAngles


def _create_mock_eye_features(
    r_iris_px: tuple = (560, 320),
    l_iris_px: tuple = (720, 320),
    img_w: int = 1280,
    img_h: int = 720,
) -> ExtractedEyeFeatures:
    """Synthesize mock eye features with specific pixel iris locations."""
    r_norm = Point2D(x=r_iris_px[0] / img_w, y=r_iris_px[1] / img_h)
    l_norm = Point2D(x=l_iris_px[0] / img_w, y=l_iris_px[1] / img_h)

    r_eye = SingleEyeFeatures(
        inner_corner=Point2D(0.48, 0.44),
        outer_corner=Point2D(0.40, 0.44),
        upper_lid_1=Point2D(0.44, 0.42),
        lower_lid_1=Point2D(0.44, 0.46),
        upper_lid_2=Point2D(0.45, 0.42),
        lower_lid_2=Point2D(0.45, 0.46),
        contour_points=[Point2D(0.40, 0.44), Point2D(0.48, 0.44)],
        iris_center=r_norm,
        iris_points=[r_norm],
        ear=0.30,
        is_closed=False,
    )
    l_eye = SingleEyeFeatures(
        inner_corner=Point2D(0.52, 0.44),
        outer_corner=Point2D(0.60, 0.44),
        upper_lid_1=Point2D(0.56, 0.42),
        lower_lid_1=Point2D(0.56, 0.46),
        upper_lid_2=Point2D(0.57, 0.42),
        lower_lid_2=Point2D(0.57, 0.46),
        contour_points=[Point2D(0.52, 0.44), Point2D(0.60, 0.44)],
        iris_center=l_norm,
        iris_points=[l_norm],
        ear=0.30,
        is_closed=False,
    )

    return ExtractedEyeFeatures(
        has_eyes=True,
        right_eye=r_eye,
        left_eye=l_eye,
        average_ear=0.30,
        both_eyes_closed=False,
    )


def test_adaptive_config_defaults():
    """Verify default AdaptiveConfig parameters."""
    cfg = AdaptiveConfig()
    assert cfg.enable_drift_compensation is True
    assert cfg.head_drift_gain_x == 10.0
    assert cfg.head_drift_gain_y == 8.5
    assert cfg.enable_distance_scaling is True
    assert cfg.max_drift_offset_px == 160.0
    assert cfg.drift_ema_smoothing == 0.15
    assert cfg.baseline_auto_anchor is True
    assert cfg.posture_alert_threshold_deg == 25.0


def test_adaptive_config_from_yaml(tmp_path: Path):
    """Verify loading AdaptiveConfig from custom YAML."""
    yaml_content = """
adaptive_calibration:
  enable_drift_compensation: false
  head_drift_gain_x: 12.0
  head_drift_gain_y: 9.0
  enable_distance_scaling: false
  max_drift_offset_px: 120.0
  drift_ema_smoothing: 0.20
  baseline_auto_anchor: false
  posture_alert_threshold_deg: 20.0
"""
    cfg_file = tmp_path / "custom_adaptive.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    cfg = AdaptiveConfig.from_yaml(cfg_file)
    assert cfg.enable_drift_compensation is False
    assert cfg.head_drift_gain_x == 12.0
    assert cfg.head_drift_gain_y == 9.0
    assert cfg.enable_distance_scaling is False
    assert cfg.max_drift_offset_px == 120.0
    assert cfg.drift_ema_smoothing == 0.20
    assert cfg.baseline_auto_anchor is False
    assert cfg.posture_alert_threshold_deg == 20.0


def test_compute_eye_distance():
    """Calculate inter-pupillary pixel distance between iris centers."""
    eye_feats = _create_mock_eye_features(r_iris_px=(500, 300), l_iris_px=(650, 300))
    dist = AdaptiveCalibrator.compute_eye_distance(eye_feats, 1280, 720)
    assert dist == pytest.approx(150.0, abs=1.0)


def test_set_and_reset_baseline_anchor():
    """Test locking and resetting physiological posture baseline."""
    calibrator = AdaptiveCalibrator()
    assert calibrator.anchor.is_anchored is False

    head_angles = HeadPoseAngles(pitch=5.0, yaw=-2.0, roll=0.5, rvec=np.zeros(3), tvec=np.zeros(3))
    eye_feats = _create_mock_eye_features()

    calibrator.set_baseline(head_angles, eye_feats, 1280, 720)
    assert calibrator.anchor.is_anchored is True
    assert calibrator.anchor.baseline_pitch == 5.0
    assert calibrator.anchor.baseline_yaw == -2.0
    assert calibrator.anchor.baseline_eye_distance_px > 0

    calibrator.reset_baseline()
    assert calibrator.anchor.is_anchored is False


def test_compensate_inactive_when_disabled():
    """When disabled, screen coordinates should pass through unmodified."""
    cfg = AdaptiveConfig(enable_drift_compensation=False)
    calibrator = AdaptiveCalibrator(cfg)
    eye_feats = _create_mock_eye_features()
    head_angles = HeadPoseAngles(pitch=10.0, yaw=15.0, roll=0.0, rvec=np.zeros(3), tvec=np.zeros(3))

    x, y, state = calibrator.compensate(500, 400, head_angles, eye_feats, 1280, 720)
    assert x == 500
    assert y == 400
    assert "Inactive" in state.status_message


def test_compensate_auto_anchor_first_frame():
    """On first frame with auto_anchor=True, calibrator should lock baseline."""
    calibrator = AdaptiveCalibrator(AdaptiveConfig(baseline_auto_anchor=True))
    eye_feats = _create_mock_eye_features()
    head_angles = HeadPoseAngles(pitch=2.0, yaw=3.0, roll=0.0, rvec=np.zeros(3), tvec=np.zeros(3))

    x, y, state = calibrator.compensate(960, 540, head_angles, eye_feats, 1280, 720)
    assert calibrator.anchor.is_anchored is True
    assert calibrator.anchor.baseline_yaw == 3.0
    assert calibrator.anchor.baseline_pitch == 2.0


def test_compensate_posture_shift_proportional():
    """Turning head to the right (+yaw) should apply proportional positive X offset."""
    cfg = AdaptiveConfig(head_drift_gain_x=10.0, drift_ema_smoothing=1.0)  # instantaneous for test
    calibrator = AdaptiveCalibrator(cfg)
    eye_feats = _create_mock_eye_features()

    # Anchor at yaw = 0.0
    anchor_angles = HeadPoseAngles(pitch=0.0, yaw=0.0, roll=0.0, rvec=np.zeros(3), tvec=np.zeros(3))
    calibrator.set_baseline(anchor_angles, eye_feats, 1280, 720)

    # Shift head 5 degrees right
    shifted_angles = HeadPoseAngles(pitch=0.0, yaw=5.0, roll=0.0, rvec=np.zeros(3), tvec=np.zeros(3))
    out_x, out_y, state = calibrator.compensate(
        screen_x=960,
        screen_y=540,
        head_angles=shifted_angles,
        eye_features=eye_feats,
        img_w=1280,
        img_h=720,
        monitor_w=1920,
        monitor_h=1080,
    )

    # Expected: 960 + 10.0 * 5.0 = 1010
    assert out_x == pytest.approx(1010, abs=2)
    assert out_y == 540
    assert state.yaw_shift_deg == 5.0
    assert state.delta_x_px == pytest.approx(50.0, abs=2.0)


def test_compensate_safety_clamping():
    """Extreme posture shift must be strictly clamped to max_drift_offset_px."""
    cfg = AdaptiveConfig(
        head_drift_gain_x=10.0,
        max_drift_offset_px=100.0,
        drift_ema_smoothing=1.0,
    )
    calibrator = AdaptiveCalibrator(cfg)
    eye_feats = _create_mock_eye_features()

    # Baseline at 0.0
    anchor_angles = HeadPoseAngles(pitch=0.0, yaw=0.0, roll=0.0, rvec=np.zeros(3), tvec=np.zeros(3))
    calibrator.set_baseline(anchor_angles, eye_feats, 1280, 720)

    # Extreme head turn of 40 degrees -> raw would be 400px
    extreme_angles = HeadPoseAngles(pitch=0.0, yaw=40.0, roll=0.0, rvec=np.zeros(3), tvec=np.zeros(3))
    out_x, out_y, state = calibrator.compensate(
        screen_x=960,
        screen_y=540,
        head_angles=extreme_angles,
        eye_features=eye_feats,
        img_w=1280,
        img_h=720,
        monitor_w=1920,
        monitor_h=1080,
    )

    assert state.drift_magnitude_px <= 100.0
    assert out_x <= 960 + 100
    assert state.is_posture_alert is True  # Alert on 40 deg shift


def test_distance_scaling_lean_in():
    """Leaning in increases eye distance and scales gaze expansion around screen center."""
    cfg = AdaptiveConfig(
        enable_drift_compensation=True,
        enable_distance_scaling=True,
        head_drift_gain_x=0.0,  # Isolate distance scaling
        head_drift_gain_y=0.0,
    )
    calibrator = AdaptiveCalibrator(cfg)

    # Baseline eye distance = 100px
    base_eye_feats = _create_mock_eye_features(r_iris_px=(590, 360), l_iris_px=(690, 360))
    anchor_angles = HeadPoseAngles(pitch=0.0, yaw=0.0, roll=0.0, rvec=np.zeros(3), tvec=np.zeros(3))
    calibrator.set_baseline(anchor_angles, base_eye_feats, 1280, 720)

    # User leans closer: eye distance increases to 120px (scale = 1.2x)
    closer_eye_feats = _create_mock_eye_features(r_iris_px=(580, 360), l_iris_px=(700, 360))

    # Point at screen_x = 1160 (200px to right of center 960)
    # Expected scaled: 960 + 200 * 1.2 = 1200
    out_x, out_y, state = calibrator.compensate(
        screen_x=1160,
        screen_y=540,
        head_angles=anchor_angles,
        eye_features=closer_eye_feats,
        img_w=1280,
        img_h=720,
        monitor_w=1920,
        monitor_h=1080,
    )

    assert state.distance_scale == pytest.approx(1.20, abs=0.05)
    assert out_x == pytest.approx(1200, abs=5)


def test_draw_telemetry_hud_rendering():
    """Telemetry HUD drawing should run cleanly without exceptions."""
    calibrator = AdaptiveCalibrator()
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    state = DriftState(
        delta_x_px=35.0,
        delta_y_px=-20.0,
        drift_magnitude_px=40.3,
        yaw_shift_deg=3.5,
        pitch_shift_deg=-2.0,
        distance_scale=1.05,
        is_posture_alert=False,
    )

    out = calibrator.draw_telemetry_hud(canvas, state, (960, 540), (995, 520))
    assert out.shape == (1080, 1920, 3)
    assert np.any(out > 0)
