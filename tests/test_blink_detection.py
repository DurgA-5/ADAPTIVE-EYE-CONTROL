"""Module 12 Blink Detection & Eye Clicking Engine Unit Tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.blink_detection.detector import (
    BlinkClickDetector,
    BlinkDetectorConfig,
    BlinkState,
    BlinkType,
    ClickDispatcher,
    ClickEvent,
    ClickType,
)
from app.core.datatypes import Point2D
from app.eye_tracking.tracker import ExtractedEyeFeatures, SingleEyeFeatures


def _create_mock_eye(ear: float = 0.30, is_closed: bool = False) -> SingleEyeFeatures:
    """Helper creating mock SingleEyeFeatures with specified EAR."""
    pt = Point2D(0.5, 0.5)
    return SingleEyeFeatures(
        inner_corner=pt,
        outer_corner=pt,
        upper_lid_1=pt,
        lower_lid_1=pt,
        upper_lid_2=pt,
        lower_lid_2=pt,
        contour_points=[pt] * 6,
        iris_center=pt,
        iris_points=[pt] * 5,
        ear=ear,
        is_closed=is_closed,
        iris_relative_x=0.5,
        iris_relative_y=0.5,
    )


def test_blink_config_defaults():
    """Verify default BlinkDetectorConfig values."""
    cfg = BlinkDetectorConfig()
    assert cfg.ear_threshold == 0.20
    assert cfg.min_blink_duration_ms == 220.0
    assert cfg.max_click_blink_duration_ms == 600.0
    assert cfg.double_click_window_ms == 650.0
    assert cfg.long_blink_right_click_ms == 700.0
    assert cfg.enable_wink_right_click is True
    assert cfg.wink_min_duration_ms == 280.0
    assert cfg.click_debounce_ms == 450.0
    assert cfg.enable_sound_feedback is True
    assert cfg.hold_position_during_blink is True
    assert cfg.mirror_mode is True
    assert cfg.enable_wink_clicking is True


def test_blink_config_from_yaml(tmp_path: Path):
    """Verify loading BlinkDetectorConfig from YAML."""
    yaml_content = """
blink_detection:
  ear_threshold: 0.18
  min_blink_duration_ms: 250.0
  max_click_blink_duration_ms: 550.0
  double_click_window_ms: 700.0
  long_blink_right_click_ms: 800.0
  enable_wink_right_click: false
  wink_min_duration_ms: 300.0
  click_debounce_ms: 500.0
  enable_sound_feedback: false
  hold_position_during_blink: false
"""
    cfg_file = tmp_path / "custom_blink.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    cfg = BlinkDetectorConfig.from_yaml(cfg_file)
    assert cfg.ear_threshold == 0.18
    assert cfg.min_blink_duration_ms == 250.0
    assert cfg.max_click_blink_duration_ms == 550.0
    assert cfg.double_click_window_ms == 700.0
    assert cfg.long_blink_right_click_ms == 800.0
    assert cfg.enable_wink_right_click is False
    assert cfg.wink_min_duration_ms == 300.0
    assert cfg.click_debounce_ms == 500.0
    assert cfg.enable_sound_feedback is False
    assert cfg.hold_position_during_blink is False


def test_involuntary_blink_ignored():
    """Fast involuntary blinks (< 220ms) must not produce any click events."""
    detector = BlinkClickDetector(BlinkDetectorConfig(enable_sound_feedback=False))
    
    open_eyes = ExtractedEyeFeatures(has_eyes=True, left_eye=_create_mock_eye(0.32), right_eye=_create_mock_eye(0.32))
    closed_eyes = ExtractedEyeFeatures(has_eyes=True, left_eye=_create_mock_eye(0.12), right_eye=_create_mock_eye(0.12))

    # T=0.0s: Open
    evt, state = detector.process(open_eyes, 500, 500, timestamp=0.0)
    assert evt is None

    # T=0.05s: Eyes close
    evt, state = detector.process(closed_eyes, 500, 500, timestamp=0.05)
    assert evt is None
    assert state.is_blinking is True

    # T=0.18s: Eyes reopen (130ms duration = involuntary)
    evt, state = detector.process(open_eyes, 500, 500, timestamp=0.18)
    assert evt is None
    assert "Involuntary Blink Ignored" in state.status_message

    # T=1.0s: Well past double-click window, still no click
    evt, state = detector.process(open_eyes, 500, 500, timestamp=1.0)
    assert evt is None


def test_intentional_left_click():
    """Single deliberate blink (350ms) must trigger Left Click after expiration window."""
    detector = BlinkClickDetector(BlinkDetectorConfig(enable_sound_feedback=False, double_click_window_ms=300.0))
    detector.dispatcher.dispatch_left_click = MagicMock()

    open_eyes = ExtractedEyeFeatures(has_eyes=True, left_eye=_create_mock_eye(0.32), right_eye=_create_mock_eye(0.32))
    closed_eyes = ExtractedEyeFeatures(has_eyes=True, left_eye=_create_mock_eye(0.12), right_eye=_create_mock_eye(0.12))

    # T=0.0s: Open
    detector.process(open_eyes, 600, 400, timestamp=0.0)

    # T=0.1s: Eyes close
    detector.process(closed_eyes, 600, 400, timestamp=0.1)

    # T=0.45s: Eyes open (350ms blink duration)
    evt, state = detector.process(open_eyes, 600, 400, timestamp=0.45)
    assert evt is None  # Pending double click window
    assert detector._pending_single_click is True

    # T=0.80s: Double click window expires (350ms elapsed > 300ms window)
    evt, state = detector.process(open_eyes, 600, 400, timestamp=0.80)
    assert evt is not None
    assert evt.click_type == ClickType.LEFT_CLICK
    assert evt.x == 600
    assert evt.y == 400
    detector.dispatcher.dispatch_left_click.assert_called_once_with(600, 400)


def test_double_click_detection():
    """Two rapid intentional blinks within double click window must trigger Double Click."""
    detector = BlinkClickDetector(BlinkDetectorConfig(enable_sound_feedback=False, double_click_window_ms=650.0))
    detector.dispatcher.dispatch_double_click = MagicMock()

    open_eyes = ExtractedEyeFeatures(has_eyes=True, left_eye=_create_mock_eye(0.32), right_eye=_create_mock_eye(0.32))
    closed_eyes = ExtractedEyeFeatures(has_eyes=True, left_eye=_create_mock_eye(0.12), right_eye=_create_mock_eye(0.12))

    # 1st Blink: 0.1s to 0.4s (300ms)
    detector.process(open_eyes, 700, 300, timestamp=0.0)
    detector.process(closed_eyes, 700, 300, timestamp=0.1)
    detector.process(open_eyes, 700, 300, timestamp=0.4)
    assert detector._pending_single_click is True

    # Brief open pause between blinks: 0.4s to 0.6s (200ms)
    detector.process(open_eyes, 700, 300, timestamp=0.6)

    # 2nd Blink: 0.65s to 0.95s (300ms)
    detector.process(closed_eyes, 700, 300, timestamp=0.65)
    evt, state = detector.process(open_eyes, 700, 300, timestamp=0.95)

    assert evt is not None
    assert evt.click_type == ClickType.DOUBLE_CLICK
    assert evt.x == 700
    assert evt.y == 300
    detector.dispatcher.dispatch_double_click.assert_called_once_with(700, 300)


def test_long_blink_right_click():
    """Prolonged deliberate bilateral eye closure (> 700ms) must trigger Right Click upon reopen."""
    detector = BlinkClickDetector(BlinkDetectorConfig(enable_sound_feedback=False, long_blink_right_click_ms=700.0))
    detector.dispatcher.dispatch_right_click = MagicMock()

    open_eyes = ExtractedEyeFeatures(has_eyes=True, left_eye=_create_mock_eye(0.32), right_eye=_create_mock_eye(0.32))
    closed_eyes = ExtractedEyeFeatures(has_eyes=True, left_eye=_create_mock_eye(0.12), right_eye=_create_mock_eye(0.12))

    # T=0.0s: Close
    detector.process(closed_eyes, 800, 250, timestamp=0.0)

    # T=0.75s: Still closed (750ms held)
    evt, state = detector.process(closed_eyes, 800, 250, timestamp=0.75)
    assert evt is None
    assert "Hold for Right Click" in state.status_message

    # T=0.85s: Reopen
    evt, state = detector.process(open_eyes, 800, 250, timestamp=0.85)
    assert evt is not None
    assert evt.click_type == ClickType.RIGHT_CLICK
    assert evt.x == 800
    assert evt.y == 250
    detector.dispatcher.dispatch_right_click.assert_called_once_with(800, 250)


def test_unilateral_wink_right_click():
    """Right eye wink (screen-right closed, screen-left open in mirrored view) held for > 280ms triggers Right Click."""
    detector = BlinkClickDetector(BlinkDetectorConfig(enable_sound_feedback=False, wink_min_duration_ms=280.0, mirror_mode=True))
    detector.dispatcher.dispatch_right_click = MagicMock()

    open_eyes = ExtractedEyeFeatures(has_eyes=True, left_eye=_create_mock_eye(0.32), right_eye=_create_mock_eye(0.32))
    right_wink = ExtractedEyeFeatures(has_eyes=True, left_eye=_create_mock_eye(0.10), right_eye=_create_mock_eye(0.32))

    # T=0.0s: User's right eye winks
    detector.process(right_wink, 450, 650, timestamp=0.0)

    # T=0.20s: Holding wink (200ms)
    evt, state = detector.process(right_wink, 450, 650, timestamp=0.20)
    assert evt is None
    assert "Right Eye Wink" in state.status_message

    # T=0.35s: Right eye reopens (350ms wink duration > 280ms)
    evt, state = detector.process(open_eyes, 450, 650, timestamp=0.35)
    assert evt is not None
    assert evt.click_type == ClickType.RIGHT_CLICK
    assert evt.x == 450
    assert evt.y == 650
    detector.dispatcher.dispatch_right_click.assert_called_once_with(450, 650)


def test_unilateral_wink_left_click():
    """Left eye wink (screen-left closed, screen-right open in mirrored view) held for > 280ms triggers Left Click."""
    detector = BlinkClickDetector(BlinkDetectorConfig(enable_sound_feedback=False, wink_min_duration_ms=280.0, mirror_mode=True))
    detector.dispatcher.dispatch_left_click = MagicMock()

    open_eyes = ExtractedEyeFeatures(has_eyes=True, left_eye=_create_mock_eye(0.32), right_eye=_create_mock_eye(0.32))
    left_wink = ExtractedEyeFeatures(has_eyes=True, left_eye=_create_mock_eye(0.32), right_eye=_create_mock_eye(0.10))

    # T=0.0s: User's left eye winks
    detector.process(left_wink, 300, 400, timestamp=0.0)

    # T=0.20s: Holding wink (200ms)
    evt, state = detector.process(left_wink, 300, 400, timestamp=0.20)
    assert evt is None
    assert "Left Eye Wink" in state.status_message

    # T=0.35s: Left eye reopens (350ms wink duration > 280ms)
    evt, state = detector.process(open_eyes, 300, 400, timestamp=0.35)
    assert evt is not None
    assert evt.click_type == ClickType.LEFT_CLICK
    assert evt.x == 300
    assert evt.y == 400
    detector.dispatcher.dispatch_left_click.assert_called_once_with(300, 400)


def test_click_debounce_refractory_period():
    """After any click dispatch, new clicks must be locked out during debounce refractory period."""
    detector = BlinkClickDetector(BlinkDetectorConfig(enable_sound_feedback=False, click_debounce_ms=500.0))
    detector._last_dispatched_click_time = 1.0

    open_eyes = ExtractedEyeFeatures(has_eyes=True, left_eye=_create_mock_eye(0.32), right_eye=_create_mock_eye(0.32))
    closed_eyes = ExtractedEyeFeatures(has_eyes=True, left_eye=_create_mock_eye(0.12), right_eye=_create_mock_eye(0.12))

    # T=1.2s: Within 500ms debounce (only 200ms elapsed)
    evt, state = detector.process(closed_eyes, 500, 500, timestamp=1.2)
    assert state.is_debouncing is True

    # Reopen at T=1.45s (250ms blink): Still within 500ms debounce (450ms elapsed since click)
    evt, state = detector.process(open_eyes, 500, 500, timestamp=1.45)
    assert evt is None
    assert detector._pending_single_click is False  # Refractory period prevented new click


def test_click_dispatcher_methods():
    """Verify ClickDispatcher methods execute Win32 or fallback without exceptions."""
    dispatcher = ClickDispatcher(enable_sound=False)
    assert dispatcher.dispatch_left_click(100, 100) is True
    assert dispatcher.dispatch_right_click(100, 100) is True
    assert dispatcher.dispatch_double_click(100, 100) is True
