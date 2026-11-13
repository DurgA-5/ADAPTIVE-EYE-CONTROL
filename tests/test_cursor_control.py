"""Module 11 OS Cursor Control & Safety Interlock Unit Tests."""

from pathlib import Path
import numpy as np
import pytest

from app.cursor_control.controller import (
    CursorControlConfig,
    MouseState,
    OSCursorController,
    VK_F8,
)


def test_cursor_control_config_defaults():
    """Verify default CursorControlConfig parameters."""
    cfg = CursorControlConfig()
    assert cfg.enable_mouse_dispatch is True
    assert cfg.mouse_backend == "win32"
    assert cfg.smoothing_factor == 0.35
    assert cfg.acceleration_curve == "sigmoid"
    assert cfg.toggle_hotkey_vkey == VK_F8
    assert cfg.emergency_corner_failsafe is True
    assert cfg.safety_interlock_require_verified is True
    assert cfg.safety_interlock_pause_on_blink is True
    assert cfg.cursor_update_hz == 60.0


def test_cursor_control_config_from_yaml(tmp_path: Path):
    """Verify loading CursorControlConfig from custom YAML."""
    yaml_content = """
cursor_control:
  enable_mouse_dispatch: false
  mouse_backend: "pyautogui"
  smoothing_factor: 0.40
  acceleration_curve: "power"
  toggle_hotkey_vkey: 119
  emergency_corner_failsafe: false
  safety_interlock_require_verified: false
  safety_interlock_pause_on_blink: false
  cursor_update_hz: 30.0
"""
    cfg_file = tmp_path / "custom_cursor.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    cfg = CursorControlConfig.from_yaml(cfg_file)
    assert cfg.enable_mouse_dispatch is False
    assert cfg.mouse_backend == "pyautogui"
    assert cfg.smoothing_factor == 0.40
    assert cfg.acceleration_curve == "power"
    assert cfg.toggle_hotkey_vkey == 119
    assert cfg.emergency_corner_failsafe is False
    assert cfg.safety_interlock_require_verified is False
    assert cfg.safety_interlock_pause_on_blink is False
    assert cfg.cursor_update_hz == 30.0


def test_controller_initialization():
    """Verify controller initialization and initial screen bounds."""
    ctrl = OSCursorController(CursorControlConfig(enable_mouse_dispatch=False))
    assert ctrl.screen_w > 0
    assert ctrl.screen_h > 0
    assert ctrl.is_engaged is False  # Derived from enable_mouse_dispatch


def test_ballistic_step_deadband():
    """Small movements under 4px should be frozen in deadband."""
    ctrl = OSCursorController(CursorControlConfig(enable_mouse_dispatch=False))
    ctrl._curr_x = 500.0
    ctrl._curr_y = 500.0

    step_x, step_y = ctrl.compute_ballistic_step(502, 501, dt=0.016)
    assert step_x == 500
    assert step_y == 500


def test_ballistic_step_precision_mode():
    """Small targeting steps (< 45px) should interpolate smoothly with fine precision."""
    ctrl = OSCursorController(CursorControlConfig(enable_mouse_dispatch=False))
    ctrl._curr_x = 500.0
    ctrl._curr_y = 500.0

    # Target 20px away: interp = 0.22 -> step moves 4.4px
    step_x, step_y = ctrl.compute_ballistic_step(520, 500, dt=0.016)
    assert step_x == pytest.approx(504, abs=1)
    assert step_y == 500


def test_ballistic_step_rapid_saccade_mode():
    """Large gaze sweeps (>= 200px) should interpolate aggressively with ballistic speed."""
    ctrl = OSCursorController(CursorControlConfig(enable_mouse_dispatch=False))
    ctrl._curr_x = 200.0
    ctrl._curr_y = 200.0

    # Target 400px away: interp = 0.65 -> moves 260px in single step
    step_x, step_y = ctrl.compute_ballistic_step(600, 200, dt=0.016)
    assert step_x == pytest.approx(460, abs=5)
    assert step_y == 200


def test_safety_interlock_unauthorized_user():
    """Zero-trust interlock must freeze cursor when user is unverified."""
    cfg = CursorControlConfig(enable_mouse_dispatch=False, safety_interlock_require_verified=True)
    ctrl = OSCursorController(cfg)
    ctrl.is_engaged = True

    state = ctrl.dispatch(target_x=800, target_y=600, is_gaze_valid=True, is_user_authorized=False)
    assert state.is_interlocked is True
    assert "IDENTITY NOT AUTHORIZED" in state.interlock_reason


def test_safety_interlock_blink():
    """Safety interlock must freeze cursor during eye closure or blinks."""
    cfg = CursorControlConfig(enable_mouse_dispatch=False, safety_interlock_pause_on_blink=True)
    ctrl = OSCursorController(cfg)
    ctrl.is_engaged = True

    state = ctrl.dispatch(target_x=800, target_y=600, is_gaze_valid=False, is_user_authorized=True)
    assert state.is_interlocked is True
    assert "EYES CLOSED / BLINK DETECTED" in state.interlock_reason


def test_safety_interlock_disengaged_hotkey():
    """When user pauses control (is_engaged=False), dispatch must hold position."""
    cfg = CursorControlConfig(enable_mouse_dispatch=False)
    ctrl = OSCursorController(cfg)
    ctrl.is_engaged = False

    state = ctrl.dispatch(target_x=800, target_y=600, is_gaze_valid=True, is_user_authorized=True)
    assert state.is_engaged is False
    assert state.is_interlocked is True
    assert "PAUSED BY USER" in state.interlock_reason


def test_emergency_corner_failsafe():
    """Physical mouse moved to (0, 0) top-left corner must trigger failsafe shutdown."""
    cfg = CursorControlConfig(enable_mouse_dispatch=False, emergency_corner_failsafe=True)
    ctrl = OSCursorController(cfg)
    ctrl.is_engaged = True

    # Mock get_physical_cursor_pos to top-left corner (0, 0)
    ctrl.get_physical_cursor_pos = lambda: (0, 0)

    state = ctrl.dispatch(target_x=800, target_y=600, is_gaze_valid=True, is_user_authorized=True)
    assert ctrl.is_engaged is False
    assert state.is_interlocked is True
    assert "EMERGENCY CORNER FAILSAFE" in state.interlock_reason


def test_boundary_clamping():
    """Target positions outside screen bounds must be clamped strictly within monitor."""
    cfg = CursorControlConfig(enable_mouse_dispatch=False)
    ctrl = OSCursorController(cfg)
    ctrl.is_engaged = True
    ctrl._curr_x = 100.0
    ctrl._curr_y = 100.0

    step_x, step_y = ctrl.compute_ballistic_step(5000, 3000, dt=0.016)
    assert step_x <= ctrl.screen_w - 1
    assert step_y <= ctrl.screen_h - 1

    ctrl._curr_x = 50.0
    ctrl._curr_y = 50.0
    step_neg_x, step_neg_y = ctrl.compute_ballistic_step(-500, -500, dt=0.016)
    assert step_neg_x >= 0
    assert step_neg_y >= 0
