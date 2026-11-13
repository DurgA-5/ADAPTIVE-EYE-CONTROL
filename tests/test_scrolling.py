"""Module 13 Eye Scrolling & Dwell Control Engine Unit Tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.blink_detection.detector import ClickType
from app.scrolling.scroller import (
    DwellClickState,
    DwellClicker,
    EyeScroller,
    ScrollDirection,
    ScrollDispatcher,
    ScrollEvent,
    ScrollingConfig,
    ScrollState,
    draw_scrolling_overlay,
)


def test_scrolling_config_defaults():
    """Verify default ScrollingConfig initialization."""
    cfg = ScrollingConfig()
    assert cfg.enable_edge_scrolling is True
    assert cfg.top_zone_ratio == 0.12
    assert cfg.bottom_zone_ratio == 0.12
    assert cfg.activation_delay_ms == 300.0
    assert cfg.scroll_interval_ms == 70.0
    assert cfg.base_wheel_delta == 120
    assert cfg.dynamic_speed_scaling is True
    assert cfg.max_speed_multiplier == 3.0
    assert cfg.enable_dwell_clicking is True
    assert cfg.dwell_time_ms == 1200.0
    assert cfg.dwell_radius_px == 35.0
    assert cfg.dwell_debounce_ms == 800.0
    assert cfg.dwell_click_action == "LEFT"
    assert cfg.enable_sound_feedback is True


def test_scrolling_config_from_yaml(tmp_path: Path):
    """Verify loading ScrollingConfig from a custom YAML file."""
    yaml_text = """
scrolling:
  enable_edge_scrolling: false
  top_zone_ratio: 0.15
  bottom_zone_ratio: 0.18
  activation_delay_ms: 250.0
  scroll_interval_ms: 60.0
  base_wheel_delta: 240
  dynamic_speed_scaling: false
  max_speed_multiplier: 2.5
  enable_dwell_clicking: false
  dwell_time_ms: 1500.0
  dwell_radius_px: 40.0
  dwell_debounce_ms: 1000.0
  dwell_click_action: "DOUBLE"
  enable_sound_feedback: false
"""
    cfg_file = tmp_path / "custom_scroll.yaml"
    cfg_file.write_text(yaml_text, encoding="utf-8")

    cfg = ScrollingConfig.from_yaml(cfg_file)
    assert cfg.enable_edge_scrolling is False
    assert cfg.top_zone_ratio == 0.15
    assert cfg.bottom_zone_ratio == 0.18
    assert cfg.activation_delay_ms == 250.0
    assert cfg.scroll_interval_ms == 60.0
    assert cfg.base_wheel_delta == 240
    assert cfg.dynamic_speed_scaling is False
    assert cfg.max_speed_multiplier == 2.5
    assert cfg.enable_dwell_clicking is False
    assert cfg.dwell_time_ms == 1500.0
    assert cfg.dwell_radius_px == 40.0
    assert cfg.dwell_debounce_ms == 1000.0
    assert cfg.dwell_click_action == "DOUBLE"
    assert cfg.enable_sound_feedback is False


def test_scroll_dispatcher_win32():
    """Verify Win32 scroll dispatch call."""
    dispatcher = ScrollDispatcher()
    mock_user32 = MagicMock()
    dispatcher._user32 = mock_user32
    dispatcher._is_win32 = True

    success = dispatcher.dispatch_scroll(wheel_delta=120)
    assert success is True
    mock_user32.mouse_event.assert_called_once_with(0x0800, 0, 0, 120, 0)


def test_scroll_dispatcher_fallback():
    """Verify PyAutoGUI fallback when Win32 is unavailable."""
    dispatcher = ScrollDispatcher()
    dispatcher._is_win32 = False
    dispatcher._user32 = None

    with patch("pyautogui.scroll") as mock_pyautogui_scroll:
        success = dispatcher.dispatch_scroll(wheel_delta=-240, x=500, y=500)
        assert success is True
        mock_pyautogui_scroll.assert_called_once_with(-2, x=500, y=500)


def test_eye_scroller_neutral_zone():
    """Verify neutral screen region produces no scrolling."""
    scroller = EyeScroller()
    screen_h = 1080
    cursor_y = 540  # Center screen

    event, state = scroller.process(cursor_y, screen_h, timestamp=1.0)
    assert event is None
    assert state.direction == ScrollDirection.NONE
    assert state.is_scrolling is False
    assert state.zone_dwell_progress == 0.0


def test_eye_scroller_top_activation_delay():
    """Verify gaze entering top zone respects activation delay before scrolling."""
    cfg = ScrollingConfig(activation_delay_ms=300.0, top_zone_ratio=0.10)
    scroller = EyeScroller(cfg)
    screen_h = 1000
    cursor_y = 50  # Inside top zone (<= 100px)

    # Initial frame at t=1.000: entry into zone
    event, state = scroller.process(cursor_y, screen_h, timestamp=1.000)
    assert event is None
    assert state.direction == ScrollDirection.UP
    assert state.is_scrolling is False
    assert state.zone_dwell_progress == 0.0

    # Intermediate frame at t=1.150 (150ms elapsed, 50% progress)
    event, state = scroller.process(cursor_y, screen_h, timestamp=1.150)
    assert event is None
    assert state.direction == ScrollDirection.UP
    assert state.is_scrolling is False
    assert 0.49 <= state.zone_dwell_progress <= 0.51


def test_eye_scroller_active_scrolling_up_and_down():
    """Verify active scrolling fires after activation delay and dispatches proper wheel direction."""
    cfg = ScrollingConfig(
        activation_delay_ms=300.0,
        scroll_interval_ms=50.0,
        base_wheel_delta=120,
        top_zone_ratio=0.10,
        bottom_zone_ratio=0.10,
    )
    scroller = EyeScroller(cfg)
    scroller.dispatcher = MagicMock()
    screen_h = 1000

    # 1. Top Zone Scroll Up (t=1.000 to t=1.350s = 350ms elapsed)
    scroller.process(cursor_y=40, screen_h=screen_h, timestamp=1.000)
    evt_up, state_up = scroller.process(cursor_y=40, screen_h=screen_h, timestamp=1.350)

    assert evt_up is not None
    assert evt_up.direction == ScrollDirection.UP
    assert evt_up.wheel_delta > 0
    assert state_up.is_scrolling is True
    scroller.dispatcher.dispatch_scroll.assert_called()

    # Reset
    scroller.reset()
    scroller.dispatcher.reset_mock()

    # 2. Bottom Zone Scroll Down (t=2.000 to t=2.350s)
    scroller.process(cursor_y=950, screen_h=screen_h, timestamp=2.000)
    evt_down, state_down = scroller.process(cursor_y=950, screen_h=screen_h, timestamp=2.350)

    assert evt_down is not None
    assert evt_down.direction == ScrollDirection.DOWN
    assert evt_down.wheel_delta < 0
    assert state_down.is_scrolling is True
    scroller.dispatcher.dispatch_scroll.assert_called()


def test_eye_scroller_dynamic_speed_scaling():
    """Verify deeper penetration into margin yields higher speed multiplier."""
    cfg = ScrollingConfig(
        activation_delay_ms=100.0,
        top_zone_ratio=0.10,
        dynamic_speed_scaling=True,
        max_speed_multiplier=3.0,
    )
    scroller = EyeScroller(cfg)
    scroller.dispatcher = MagicMock()
    screen_h = 1000  # top zone <= 100px

    # Border entry: cursor_y = 100 (0% penetration)
    scroller.process(cursor_y=100, screen_h=screen_h, timestamp=1.0)
    evt_shallow, _ = scroller.process(cursor_y=100, screen_h=screen_h, timestamp=1.2)

    # Deep penetration: cursor_y = 0 (100% penetration)
    scroller.reset()
    scroller.process(cursor_y=0, screen_h=screen_h, timestamp=2.0)
    evt_deep, _ = scroller.process(cursor_y=0, screen_h=screen_h, timestamp=2.2)

    assert evt_shallow is not None and evt_deep is not None
    assert evt_deep.speed_multiplier > evt_shallow.speed_multiplier
    assert evt_deep.wheel_delta > evt_shallow.wheel_delta


def test_dwell_clicker_fixation_and_fire():
    """Verify stationary cursor triggers dwell click upon completion."""
    cfg = ScrollingConfig(
        enable_dwell_clicking=True,
        dwell_time_ms=1000.0,
        dwell_radius_px=30.0,
        dwell_debounce_ms=500.0,
        dwell_click_action="LEFT",
    )
    dwell = DwellClicker(cfg)
    dwell.dispatcher = MagicMock()

    # Initial frame at t=1.000
    evt1, st1 = dwell.process(cursor_x=500, cursor_y=500, timestamp=1.000)
    assert evt1 is None
    assert st1.is_dwelling is True
    assert st1.dwell_progress == 0.0

    # Intermediate frame at t=1.500 (500ms elapsed -> 50% progress)
    evt2, st2 = dwell.process(cursor_x=510, cursor_y=510, timestamp=1.500)
    assert evt2 is None
    assert 0.49 <= st2.dwell_progress <= 0.51

    # Dwell completed at t=2.050 (1050ms elapsed >= 1000ms)
    evt3, st3 = dwell.process(cursor_x=515, cursor_y=505, timestamp=2.050)
    assert evt3 is not None
    assert evt3.click_type == ClickType.LEFT_CLICK
    assert evt3.x == 515
    assert evt3.y == 505
    assert st3.dwell_progress == 1.0
    dwell.dispatcher.dispatch_left_click.assert_called_once_with(515, 505)


def test_dwell_clicker_drift_resets_progress():
    """Verify cursor movement outside dwell radius cancels and re-anchors dwell progress."""
    cfg = ScrollingConfig(dwell_time_ms=1000.0, dwell_radius_px=25.0)
    dwell = DwellClicker(cfg)

    # Frame 1: anchor at (100, 100)
    dwell.process(cursor_x=100, cursor_y=100, timestamp=1.000)

    # Frame 2: halfway through (100, 105)
    _, st = dwell.process(cursor_x=100, cursor_y=105, timestamp=1.500)
    assert st.dwell_progress > 0.40

    # Frame 3: sudden saccadic jump to (300, 300) (> 25px drift)
    evt, st_jump = dwell.process(cursor_x=300, cursor_y=300, timestamp=1.600)
    assert evt is None
    assert st_jump.dwell_progress == 0.0
    assert st_jump.anchor_x == 300
    assert st_jump.anchor_y == 300


def test_dwell_clicker_debounce_cooldown():
    """Verify click execution is followed by debounce refractory cooldown."""
    cfg = ScrollingConfig(dwell_time_ms=500.0, dwell_debounce_ms=600.0)
    dwell = DwellClicker(cfg)
    dwell.dispatcher = MagicMock()

    # Trigger click at t=1.600
    dwell.process(cursor_x=200, cursor_y=200, timestamp=1.000)
    evt, _ = dwell.process(cursor_x=200, cursor_y=200, timestamp=1.600)
    assert evt is not None

    # Immediate next frame at t=1.700 (100ms into 600ms cooldown)
    evt_cool, st_cool = dwell.process(cursor_x=200, cursor_y=200, timestamp=1.700)
    assert evt_cool is None
    assert st_cool.is_cooling_down is True
    assert st_cool.dwell_progress == 0.0


def test_draw_scrolling_overlay():
    """Verify visual scrolling and dwell overlay draws on numpy frame without error."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    scroll_state = ScrollState(
        direction=ScrollDirection.UP,
        is_scrolling=True,
        speed_multiplier=2.0,
        zone_dwell_progress=1.0,
        status_message="Scrolling UP",
    )
    dwell_state = DwellClickState(
        is_dwelling=True,
        dwell_progress=0.75,
        anchor_x=320,
        anchor_y=240,
        current_x=320,
        current_y=240,
        elapsed_ms=750.0,
        is_cooling_down=False,
        status_message="Fixating... 75%",
    )

    annotated = draw_scrolling_overlay(frame, scroll_state, dwell_state)
    assert annotated is not None
    assert annotated.shape == (480, 640, 3)
