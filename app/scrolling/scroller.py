"""AdaptiveEye Eye Scrolling & Dwell Control Engine.

Provides hands-free screen scrolling triggered by gaze border dwell zones and hands-free
fixation dwell clicking with circular radial progress visualization.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np
import yaml

from app.blink_detection.detector import ClickDispatcher, ClickEvent, ClickType
from app.core.logger import get_logger

logger = get_logger("EyeScroller")

# Win32 Mouse Wheel Event Flags
MOUSEEVENTF_WHEEL = 0x0800
WHEEL_DELTA = 120


class ScrollDirection(str, Enum):
    """Direction of screen scrolling."""
    NONE = "NONE"
    UP = "UP"
    DOWN = "DOWN"


@dataclass
class ScrollingConfig:
    """Strongly-typed parameters for Gaze Border Scrolling and Fixation Dwell Clicking."""
    # Border / Edge Scrolling
    enable_edge_scrolling: bool = True
    top_zone_ratio: float = 0.12
    bottom_zone_ratio: float = 0.12
    activation_delay_ms: float = 300.0
    scroll_interval_ms: float = 70.0
    base_wheel_delta: int = WHEEL_DELTA
    dynamic_speed_scaling: bool = True
    max_speed_multiplier: float = 3.0

    # Fixation Dwell Clicking
    enable_dwell_clicking: bool = True
    dwell_time_ms: float = 1200.0
    dwell_radius_px: float = 35.0
    dwell_debounce_ms: float = 800.0
    dwell_click_action: str = "LEFT"
    enable_sound_feedback: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ScrollingConfig":
        """Load scrolling configuration from YAML file."""
        if not yaml_path.exists():
            logger.warning("Scrolling config file %s not found. Using defaults.", yaml_path)
            return cls()

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        s_data = data.get("scrolling", {})
        return cls(
            enable_edge_scrolling=bool(s_data.get("enable_edge_scrolling", True)),
            top_zone_ratio=float(s_data.get("top_zone_ratio", 0.12)),
            bottom_zone_ratio=float(s_data.get("bottom_zone_ratio", 0.12)),
            activation_delay_ms=float(s_data.get("activation_delay_ms", 300.0)),
            scroll_interval_ms=float(s_data.get("scroll_interval_ms", 70.0)),
            base_wheel_delta=int(s_data.get("base_wheel_delta", WHEEL_DELTA)),
            dynamic_speed_scaling=bool(s_data.get("dynamic_speed_scaling", True)),
            max_speed_multiplier=float(s_data.get("max_speed_multiplier", 3.0)),
            enable_dwell_clicking=bool(s_data.get("enable_dwell_clicking", True)),
            dwell_time_ms=float(s_data.get("dwell_time_ms", 1200.0)),
            dwell_radius_px=float(s_data.get("dwell_radius_px", 35.0)),
            dwell_debounce_ms=float(s_data.get("dwell_debounce_ms", 800.0)),
            dwell_click_action=str(s_data.get("dwell_click_action", "LEFT")),
            enable_sound_feedback=bool(s_data.get("enable_sound_feedback", True)),
        )


@dataclass
class ScrollEvent:
    """Outcome of an active eye scrolling tick."""
    direction: ScrollDirection
    wheel_delta: int
    speed_multiplier: float
    timestamp: float


@dataclass
class ScrollState:
    """Real-time diagnostic telemetry of border gaze scrolling."""
    direction: ScrollDirection
    is_scrolling: bool
    speed_multiplier: float
    zone_dwell_progress: float
    status_message: str


@dataclass
class DwellClickState:
    """Real-time diagnostic telemetry of fixation dwell clicking."""
    is_dwelling: bool
    dwell_progress: float  # 0.0 to 1.0
    anchor_x: int
    anchor_y: int
    current_x: int
    current_y: int
    elapsed_ms: float
    is_cooling_down: bool
    status_message: str


class ScrollDispatcher:
    """Dispatches physical mouse wheel scroll events via Win32 or fallback."""

    def __init__(self) -> None:
        self._user32 = None
        self._is_win32 = False
        try:
            import ctypes
            self._user32 = ctypes.windll.user32
            self._is_win32 = True
        except Exception as exc:
            logger.warning("Win32 scroll dispatcher unavailable (%s). Using fallback.", exc)

    def dispatch_scroll(self, wheel_delta: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Dispatch mouse wheel scroll.

        Args:
            wheel_delta: Signed integer delta (+120 for up, -120 for down).
            x: Optional screen coordinate X.
            y: Optional screen coordinate Y.

        Returns:
            True if dispatched successfully.
        """
        if self._is_win32 and self._user32:
            self._user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(wheel_delta), 0)
            return True

        try:
            import pyautogui
            clicks = int(wheel_delta / WHEEL_DELTA)
            if clicks == 0:
                clicks = 1 if wheel_delta > 0 else -1
            pyautogui.scroll(clicks, x=x, y=y)
            return True
        except Exception as exc:
            logger.error("Scroll dispatch failed: %s", exc)
            return False


class EyeScroller:
    """Border-zone gaze tracking for continuous screen scrolling."""

    def __init__(self, config: Optional[ScrollingConfig] = None) -> None:
        self.config = config or ScrollingConfig()
        self.dispatcher = ScrollDispatcher()

        self._current_zone: ScrollDirection = ScrollDirection.NONE
        self._zone_entry_time: Optional[float] = None
        self._last_scroll_dispatch_time: float = -999.0

    def reset(self) -> None:
        """Reset internal zone and timing trackers."""
        self._current_zone = ScrollDirection.NONE
        self._zone_entry_time = None
        self._last_scroll_dispatch_time = -999.0

    def process(
        self,
        cursor_y: int,
        screen_h: int,
        cursor_x: Optional[int] = None,
        is_active: bool = True,
        timestamp: Optional[float] = None,
    ) -> Tuple[Optional[ScrollEvent], ScrollState]:
        """Evaluate cursor position against screen edge margins and dispatch scroll events.

        Args:
            cursor_y: Target Y screen coordinate.
            screen_h: Monitor screen height in pixels.
            cursor_x: Optional target X coordinate.
            is_active: True if system is operational and authorized.
            timestamp: Optional mockable timestamp (seconds).

        Returns:
            Tuple of (ScrollEvent if dispatched else None, ScrollState diagnostics).
        """
        now = time.perf_counter() if timestamp is None else timestamp

        if not self.config.enable_edge_scrolling or not is_active:
            self.reset()
            return None, ScrollState(
                direction=ScrollDirection.NONE,
                is_scrolling=False,
                speed_multiplier=1.0,
                zone_dwell_progress=0.0,
                status_message="Scrolling Disabled / Inactive",
            )

        top_limit = int(screen_h * self.config.top_zone_ratio)
        bottom_limit = int(screen_h * (1.0 - self.config.bottom_zone_ratio))

        # Detect active zone
        detected_zone = ScrollDirection.NONE
        if cursor_y <= top_limit:
            detected_zone = ScrollDirection.UP
        elif cursor_y >= bottom_limit:
            detected_zone = ScrollDirection.DOWN

        # Handle zone transition
        if detected_zone != self._current_zone:
            self._current_zone = detected_zone
            if detected_zone != ScrollDirection.NONE:
                self._zone_entry_time = now
            else:
                self._zone_entry_time = None

        if self._current_zone == ScrollDirection.NONE or self._zone_entry_time is None:
            return None, ScrollState(
                direction=ScrollDirection.NONE,
                is_scrolling=False,
                speed_multiplier=1.0,
                zone_dwell_progress=0.0,
                status_message="Neutral Screen Area",
            )

        elapsed_ms = (now - self._zone_entry_time) * 1000.0
        activation_ms = max(1.0, self.config.activation_delay_ms)
        dwell_progress = min(1.0, max(0.0, elapsed_ms / activation_ms))

        if dwell_progress < 1.0:
            return None, ScrollState(
                direction=self._current_zone,
                is_scrolling=False,
                speed_multiplier=1.0,
                zone_dwell_progress=round(dwell_progress, 3),
                status_message=f"Entering {self._current_zone.value} Zone ({int(dwell_progress * 100)}%)",
            )

        # In active scrolling mode: compute dynamic speed multiplier
        speed_mult = 1.0
        if self.config.dynamic_speed_scaling:
            if self._current_zone == ScrollDirection.UP:
                penetration = max(0.0, min(1.0, (top_limit - cursor_y) / max(1, top_limit)))
            else:
                penetration = max(0.0, min(1.0, (cursor_y - bottom_limit) / max(1, screen_h - bottom_limit)))
            speed_mult = 1.0 + penetration * (self.config.max_speed_multiplier - 1.0)

        # Check dispatch rate interval
        time_since_dispatch_ms = (now - self._last_scroll_dispatch_time) * 1000.0
        scroll_event = None

        if time_since_dispatch_ms >= self.config.scroll_interval_ms:
            sign = 1 if self._current_zone == ScrollDirection.UP else -1
            delta = int(sign * self.config.base_wheel_delta * speed_mult)
            self.dispatcher.dispatch_scroll(wheel_delta=delta, x=cursor_x, y=cursor_y)
            self._last_scroll_dispatch_time = now
            scroll_event = ScrollEvent(
                direction=self._current_zone,
                wheel_delta=delta,
                speed_multiplier=round(speed_mult, 2),
                timestamp=now,
            )

        return scroll_event, ScrollState(
            direction=self._current_zone,
            is_scrolling=True,
            speed_multiplier=round(speed_mult, 2),
            zone_dwell_progress=1.0,
            status_message=f"Scrolling {self._current_zone.value} ({speed_mult:.1f}x)",
        )


class DwellClicker:
    """Hands-free fixation clicker triggering OS mouse clicks via stationary gaze dwell."""

    def __init__(self, config: Optional[ScrollingConfig] = None) -> None:
        self.config = config or ScrollingConfig()
        self.dispatcher = ClickDispatcher(enable_sound=self.config.enable_sound_feedback)

        self._anchor_x: Optional[int] = None
        self._anchor_y: Optional[int] = None
        self._anchor_start_time: Optional[float] = None
        self._last_click_time: float = -999.0

    def reset(self) -> None:
        """Reset fixation anchor and timers."""
        self._anchor_x = None
        self._anchor_y = None
        self._anchor_start_time = None

    def process(
        self,
        cursor_x: int,
        cursor_y: int,
        is_active: bool = True,
        timestamp: Optional[float] = None,
    ) -> Tuple[Optional[ClickEvent], DwellClickState]:
        """Evaluate cursor fixation within dwell radius and fire click upon completion.

        Args:
            cursor_x: Current cursor X coordinate.
            cursor_y: Current cursor Y coordinate.
            is_active: True if system is operational and authorized.
            timestamp: Optional mockable timestamp (seconds).

        Returns:
            Tuple of (ClickEvent if fired else None, DwellClickState diagnostics).
        """
        now = time.perf_counter() if timestamp is None else timestamp

        if not self.config.enable_dwell_clicking or not is_active:
            self.reset()
            return None, DwellClickState(
                is_dwelling=False,
                dwell_progress=0.0,
                anchor_x=cursor_x,
                anchor_y=cursor_y,
                current_x=cursor_x,
                current_y=cursor_y,
                elapsed_ms=0.0,
                is_cooling_down=False,
                status_message="Dwell Click Inactive",
            )

        # Check post-click refractory cooldown
        cooldown_elapsed_ms = (now - self._last_click_time) * 1000.0
        if cooldown_elapsed_ms < self.config.dwell_debounce_ms:
            self.reset()
            return None, DwellClickState(
                is_dwelling=False,
                dwell_progress=0.0,
                anchor_x=cursor_x,
                anchor_y=cursor_y,
                current_x=cursor_x,
                current_y=cursor_y,
                elapsed_ms=0.0,
                is_cooling_down=True,
                status_message=f"Dwell Cooldown ({int(self.config.dwell_debounce_ms - cooldown_elapsed_ms)}ms)",
            )

        # Initialize or evaluate fixation anchor
        if self._anchor_x is None or self._anchor_y is None or self._anchor_start_time is None:
            self._anchor_x = cursor_x
            self._anchor_y = cursor_y
            self._anchor_start_time = now
            return None, DwellClickState(
                is_dwelling=True,
                dwell_progress=0.0,
                anchor_x=cursor_x,
                anchor_y=cursor_y,
                current_x=cursor_x,
                current_y=cursor_y,
                elapsed_ms=0.0,
                is_cooling_down=False,
                status_message="Fixation Anchored",
            )

        # Compute distance to anchor
        dist = float(np.hypot(cursor_x - self._anchor_x, cursor_y - self._anchor_y))

        # If gaze drifted beyond dwell radius, reset anchor to current point
        if dist > self.config.dwell_radius_px:
            self._anchor_x = cursor_x
            self._anchor_y = cursor_y
            self._anchor_start_time = now
            return None, DwellClickState(
                is_dwelling=True,
                dwell_progress=0.0,
                anchor_x=cursor_x,
                anchor_y=cursor_y,
                current_x=cursor_x,
                current_y=cursor_y,
                elapsed_ms=0.0,
                is_cooling_down=False,
                status_message="Anchor Re-centered",
            )

        # Gaze is steady within dwell radius: accumulate dwell progress
        elapsed_ms = (now - self._anchor_start_time) * 1000.0
        dwell_target_ms = max(100.0, self.config.dwell_time_ms)
        progress = min(1.0, max(0.0, elapsed_ms / dwell_target_ms))

        # Check if dwell timer reached 100%
        if progress >= 1.0:
            click_type = ClickType.LEFT_CLICK
            action_upper = self.config.dwell_click_action.upper()
            if "RIGHT" in action_upper:
                click_type = ClickType.RIGHT_CLICK
                self.dispatcher.dispatch_right_click(cursor_x, cursor_y)
            elif "DOUBLE" in action_upper:
                click_type = ClickType.DOUBLE_CLICK
                self.dispatcher.dispatch_double_click(cursor_x, cursor_y)
            else:
                click_type = ClickType.LEFT_CLICK
                self.dispatcher.dispatch_left_click(cursor_x, cursor_y)

            self._last_click_time = now
            fired_event = ClickEvent(
                click_type=click_type,
                x=cursor_x,
                y=cursor_y,
                duration_ms=round(elapsed_ms, 1),
                timestamp=now,
            )
            self.reset()
            return fired_event, DwellClickState(
                is_dwelling=False,
                dwell_progress=1.0,
                anchor_x=cursor_x,
                anchor_y=cursor_y,
                current_x=cursor_x,
                current_y=cursor_y,
                elapsed_ms=elapsed_ms,
                is_cooling_down=True,
                status_message=f"Dwell Click Executed ({click_type.value})",
            )

        return None, DwellClickState(
            is_dwelling=True,
            dwell_progress=round(progress, 3),
            anchor_x=self._anchor_x,
            anchor_y=self._anchor_y,
            current_x=cursor_x,
            current_y=cursor_y,
            elapsed_ms=round(elapsed_ms, 1),
            is_cooling_down=False,
            status_message=f"Fixating... {int(progress * 100)}%",
        )


def draw_scrolling_overlay(
    frame: np.ndarray,
    scroll_state: ScrollState,
    dwell_state: Optional[DwellClickState] = None,
) -> np.ndarray:
    """Render on-screen visual banners and radial dwell gauges onto video preview frame.

    Args:
        frame: OpenCV BGR image frame.
        scroll_state: Diagnostic state of edge scrolling.
        dwell_state: Optional diagnostic state of fixation dwell clicking.

    Returns:
        Annotated BGR image frame.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 1. Edge Scrolling Visual Banners
    if scroll_state.is_scrolling or scroll_state.zone_dwell_progress > 0.0:
        is_up = scroll_state.direction == ScrollDirection.UP
        banner_y1 = 0 if is_up else (h - 32)
        banner_y2 = 32 if is_up else h
        color = (0, 200, 255) if scroll_state.is_scrolling else (100, 150, 200)

        # Translucent banner overlay
        sub_img = annotated[banner_y1:banner_y2, 0:w]
        rect = np.full(sub_img.shape, (20, 40, 20) if is_up else (20, 20, 50), dtype=np.uint8)
        cv2.addWeighted(sub_img, 0.4, rect, 0.6, 1.0, sub_img)

        arrow_txt = "▲▲▲ SCROLLING UP ▲▲▲" if is_up else "▼▼▼ SCROLLING DOWN ▼▼▼"
        if not scroll_state.is_scrolling:
            arrow_txt = f"Entering {scroll_state.direction.value} Scroll ({int(scroll_state.zone_dwell_progress * 100)}%)"

        cv2.putText(
            annotated,
            arrow_txt,
            (w // 2 - 140, banner_y2 - 10 if is_up else banner_y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    # 2. Fixation Dwell Circular Ring Progress Overlay
    if dwell_state and dwell_state.is_dwelling and dwell_state.dwell_progress > 0.05:
        # Scale screen coordinate to frame preview coordinates
        cx = int(w * 0.5)
        cy = int(h * 0.5)
        radius = 28

        # Background anchor circle
        cv2.circle(annotated, (cx, cy), radius, (80, 80, 80), 2, cv2.LINE_AA)

        # Progress Arc (0 to 360 degrees)
        end_angle = int(dwell_state.dwell_progress * 360.0)
        cv2.ellipse(
            annotated,
            (cx, cy),
            (radius, radius),
            -90,  # Start at top (12 o'clock)
            0,
            end_angle,
            (0, 240, 255),
            3,
            cv2.LINE_AA,
        )
        pct_txt = f"{int(dwell_state.dwell_progress * 100)}%"
        cv2.putText(
            annotated,
            pct_txt,
            (cx - 14, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 240, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated
