"""Module 10 Cursor Stabilization & Jitter Suppression Unit Tests."""

from pathlib import Path
import numpy as np
import pytest

from app.stabilization.filters import AdaptiveEMAFilter, ConfidenceKalmanFilter
from app.stabilization.stabilizer import (
    AdaptiveCursorStabilizer,
    StabilizedCursorState,
    StabilizerConfig,
)


def test_stabilizer_config_defaults():
    """Verify default StabilizerConfig parameters."""
    cfg = StabilizerConfig()
    assert cfg.algorithm == "adaptive_ema"
    assert cfg.deadzone_radius_px == 6.0
    assert cfg.min_alpha == 0.08
    assert cfg.max_alpha == 0.85
    assert cfg.saccade_velocity_threshold_px_s == 480.0
    assert cfg.kalman_process_noise == 0.08
    assert cfg.kalman_measurement_noise_base == 1.2
    assert cfg.confidence_noise_multiplier == 12.0
    assert cfg.hold_position_on_loss is True


def test_stabilizer_config_from_yaml(tmp_path: Path):
    """Verify loading StabilizerConfig from custom YAML."""
    yaml_content = """
stabilization:
  algorithm: "hybrid"
  deadzone_radius_px: 8.0
  min_alpha: 0.05
  max_alpha: 0.90
  saccade_velocity_threshold_px_s: 500.0
  kalman_process_noise: 0.10
  kalman_measurement_noise_base: 1.5
  confidence_noise_multiplier: 15.0
  hold_position_on_loss: false
"""
    cfg_file = tmp_path / "custom_stab.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    cfg = StabilizerConfig.from_yaml(cfg_file)
    assert cfg.algorithm == "hybrid"
    assert cfg.deadzone_radius_px == 8.0
    assert cfg.min_alpha == 0.05
    assert cfg.max_alpha == 0.90
    assert cfg.saccade_velocity_threshold_px_s == 500.0
    assert cfg.kalman_process_noise == 0.10
    assert cfg.kalman_measurement_noise_base == 1.5
    assert cfg.confidence_noise_multiplier == 15.0
    assert cfg.hold_position_on_loss is False


def test_adaptive_ema_filter_initialization():
    """First sample should initialize position without delay."""
    f = AdaptiveEMAFilter()
    res = f.filter(500.0, 300.0, timestamp=10.0)
    assert res.x == 500.0
    assert res.y == 300.0
    assert res.velocity_px_s == 0.0
    assert res.is_deadzone_active is False


def test_adaptive_ema_filter_deadzone_tremor_suppression():
    """Small displacements below deadzone_radius must be frozen as zero movement."""
    f = AdaptiveEMAFilter(deadzone_radius_px=6.0)
    f.filter(500.0, 300.0, timestamp=1.0)

    # Involuntary ocular tremor of 3px (below 6px threshold)
    res = f.filter(503.0, 301.0, timestamp=1.033)
    assert res.x == 500.0
    assert res.y == 300.0
    assert res.is_deadzone_active is True
    assert res.velocity_px_s == 0.0


def test_adaptive_ema_filter_saccadic_speed_snap():
    """Rapid gaze jump (saccade) must trigger high alpha for instant response."""
    f = AdaptiveEMAFilter(
        min_alpha=0.08,
        max_alpha=0.85,
        saccade_velocity_threshold=400.0,
        deadzone_radius_px=5.0,
    )
    f.filter(100.0, 100.0, timestamp=1.0)

    # Big jump: 400px jump in 33ms -> velocity ~ 12,000 px/s
    res = f.filter(500.0, 100.0, timestamp=1.033)
    assert res.alpha >= 0.80  # Snaps to max_alpha
    assert res.x > 400.0      # Follows immediately with almost zero lag
    assert res.is_deadzone_active is False


def test_confidence_kalman_filter_initialization():
    """Kalman filter should track initial point on first update."""
    kf = ConfidenceKalmanFilter()
    kx, ky, speed = kf.update(960.0, 540.0, confidence=1.0, timestamp=1.0)
    assert kx == 960.0
    assert ky == 540.0
    assert speed == 0.0


def test_confidence_kalman_filter_low_confidence_damping():
    """Low tracking confidence must increase measurement noise R and dampen erratic jumps."""
    # High confidence filter
    kf_high = ConfidenceKalmanFilter(measurement_noise_base=1.0, confidence_multiplier=20.0)
    kf_high.update(100.0, 100.0, confidence=1.0, timestamp=1.0)
    high_x, _, _ = kf_high.update(200.0, 100.0, confidence=1.0, timestamp=1.033)

    # Low confidence filter with identical initial state
    kf_low = ConfidenceKalmanFilter(measurement_noise_base=1.0, confidence_multiplier=20.0)
    kf_low.update(100.0, 100.0, confidence=1.0, timestamp=1.0)
    low_x, _, _ = kf_low.update(200.0, 100.0, confidence=0.2, timestamp=1.033)

    # Low confidence update should reject the noisy jump more heavily (smaller shift)
    assert abs(low_x - 100.0) < abs(high_x - 100.0)


def test_stabilizer_modes():
    """Verify stabilizer works across 'adaptive_ema', 'kalman', and 'hybrid' modes."""
    for mode in ("adaptive_ema", "kalman", "hybrid"):
        cfg = StabilizerConfig(algorithm=mode)
        stabilizer = AdaptiveCursorStabilizer(cfg)

        state1 = stabilizer.stabilize(500, 400, confidence=0.9, is_valid=True)
        assert state1.screen_x == 500
        assert state1.screen_y == 400

        state2 = stabilizer.stabilize(508, 404, confidence=0.9, is_valid=True)
        assert 0 <= state2.screen_x <= 1920
        assert 0 <= state2.screen_y <= 1080
        assert state2.is_valid is True


def test_stabilizer_hold_position_during_blink():
    """During blinks or occlusions (is_valid=False), cursor must hold last stable coordinate."""
    cfg = StabilizerConfig(hold_position_on_loss=True)
    stabilizer = AdaptiveCursorStabilizer(cfg)

    # 1. Establish stable point
    stabilizer.stabilize(640, 480, confidence=1.0, is_valid=True)
    state = stabilizer.stabilize(650, 485, confidence=1.0, is_valid=True)
    last_x, last_y = state.screen_x, state.screen_y

    # 2. Eye closure / blink event (is_valid=False, raw_x/raw_y might be 0,0)
    blink_state = stabilizer.stabilize(0, 0, confidence=0.0, is_valid=False)

    assert blink_state.is_valid is False
    assert blink_state.screen_x == last_x  # Cursor did not jump to 0,0!
    assert blink_state.screen_y == last_y
    assert "Paused" in blink_state.status_message


def test_stabilizer_boundary_clamping():
    """Coordinates must never exceed monitor boundaries [0, W-1], [0, H-1]."""
    stabilizer = AdaptiveCursorStabilizer()

    # Extreme negative
    state_neg = stabilizer.stabilize(-200, -150, confidence=1.0, is_valid=True, monitor_w=1920, monitor_h=1080)
    assert state_neg.screen_x == 0
    assert state_neg.screen_y == 0

    # Extreme overflow
    state_over = stabilizer.stabilize(2500, 1500, confidence=1.0, is_valid=True, monitor_w=1920, monitor_h=1080)
    assert state_over.screen_x == 1919
    assert state_over.screen_y == 1079


def test_draw_telemetry_hud_rendering():
    """HUD overlay drawing must run cleanly on both valid and invalid states."""
    stabilizer = AdaptiveCursorStabilizer()
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)

    valid_state = StabilizedCursorState(
        screen_x=960,
        screen_y=540,
        raw_x=975,
        raw_y=548,
        jitter_reduced_px=17.0,
        is_fixating=True,
        velocity_px_s=42.0,
        alpha=0.08,
        confidence=0.95,
        is_valid=True,
        status_message="FIXATION [ROCK-SOLID]",
    )
    out = stabilizer.draw_telemetry_hud(canvas, valid_state)
    assert out.shape == (1080, 1920, 3)
    assert np.any(out > 0)

    # Invalid state
    invalid_state = StabilizedCursorState(
        screen_x=960,
        screen_y=540,
        raw_x=0,
        raw_y=0,
        jitter_reduced_px=0.0,
        is_fixating=True,
        velocity_px_s=0.0,
        alpha=0.0,
        confidence=0.0,
        is_valid=False,
        status_message="Tracking Paused",
    )
    out_inv = stabilizer.draw_telemetry_hud(canvas, invalid_state)
    assert out_inv.shape == (1080, 1920, 3)
