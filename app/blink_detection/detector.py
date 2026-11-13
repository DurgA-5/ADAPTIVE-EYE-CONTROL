"""AdaptiveEye Blink Detection & Intentional Eye Clicking Engine.

Processes real-time Eye Aspect Ratio (EAR) telemetry from MediaPipe FaceMesh to classify
blinks into:
  1. Involuntary Blinks (100ms - 220ms): Natural lubrication, completely ignored.
  2. Intentional Left Click (220ms - 600ms): Deliberate conscious single blink.
  3. Double Click: Two intentional blinks within a rapid window (< 650ms).
  4. Right Click: Prolonged bilateral closure (> 700ms) or unilateral wink (> 280ms).
Dispatches native OS click events via Win32 API with anti-fatigue debounce refractory periods.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import threading
import time
from typing import Optional, Tuple
import yaml

from app.core.logger import get_logger
from app.eye_tracking.tracker import ExtractedEyeFeatures

logger = get_logger("BlinkDetector")

# Windows Win32 Mouse Event Flags
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010


class ClickType(str, Enum):
    """Categorization of dispatched mouse click actions."""
    NONE = "NONE"
    LEFT_CLICK = "LEFT_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"


class BlinkType(str, Enum):
    """Classification of biological ocular closure events."""
    NONE = "NONE"
    INVOLUNTARY = "INVOLUNTARY"
    LEFT_CLICK_BLINK = "LEFT_CLICK_BLINK"
    RIGHT_CLICK_WINK = "RIGHT_CLICK_WINK"
    RIGHT_CLICK_LONG = "RIGHT_CLICK_LONG"
    DOUBLE_CLICK_BLINK = "DOUBLE_CLICK_BLINK"


@dataclass
class BlinkDetectorConfig:
    """Strongly-typed parameters for blink discrimination and click dispatch."""
    ear_threshold: float = 0.20
    min_blink_duration_ms: float = 220.0
    max_click_blink_duration_ms: float = 600.0
    double_click_window_ms: float = 650.0
    long_blink_right_click_ms: float = 700.0
    enable_wink_clicking: bool = True
    enable_wink_right_click: bool = True
    wink_min_duration_ms: float = 280.0
    click_debounce_ms: float = 450.0
    enable_sound_feedback: bool = True
    hold_position_during_blink: bool = True
    mirror_mode: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "BlinkDetectorConfig":
        """Load configuration from a YAML file."""
        if not yaml_path.exists():
            logger.warning("Config %s not found. Using defaults.", yaml_path)
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        b_data = data.get("blink_detection", {})
        return cls(
            ear_threshold=float(b_data.get("ear_threshold", 0.20)),
            min_blink_duration_ms=float(b_data.get("min_blink_duration_ms", 220.0)),
            max_click_blink_duration_ms=float(b_data.get("max_click_blink_duration_ms", 600.0)),
            double_click_window_ms=float(b_data.get("double_click_window_ms", 650.0)),
            long_blink_right_click_ms=float(b_data.get("long_blink_right_click_ms", 700.0)),
            enable_wink_clicking=bool(b_data.get("enable_wink_clicking", b_data.get("enable_wink_right_click", True))),
            enable_wink_right_click=bool(b_data.get("enable_wink_right_click", True)),
            wink_min_duration_ms=float(b_data.get("wink_min_duration_ms", 280.0)),
            click_debounce_ms=float(b_data.get("click_debounce_ms", 450.0)),
            enable_sound_feedback=bool(b_data.get("enable_sound_feedback", True)),
            hold_position_during_blink=bool(b_data.get("hold_position_during_blink", True)),
            mirror_mode=bool(b_data.get("mirror_mode", True)),
        )


@dataclass
class ClickEvent:
    """Outcome of an intentional click gesture."""
    click_type: ClickType
    x: int
    y: int
    duration_ms: float
    timestamp: float


@dataclass
class BlinkState:
    """Real-time diagnostic telemetry of ocular closure and clicking."""
    is_blinking: bool
    blink_duration_ms: float
    left_ear: float
    right_ear: float
    left_closed: bool
    right_closed: bool
    is_debouncing: bool
    last_click_type: ClickType = ClickType.NONE
    last_click_time: float = 0.0
    status_message: str = "Eyes Open"


class ClickDispatcher:
    """Dispatches physical mouse clicks to Windows OS via Win32 or fallback."""

    def __init__(self, enable_sound: bool = True) -> None:
        self.enable_sound = enable_sound
        self._user32 = None
        self._is_win32 = False
        try:
            import ctypes
            self._user32 = ctypes.windll.user32
            self._is_win32 = True
        except Exception as exc:
            logger.warning("Win32 click dispatcher unavailable (%s). Using fallback.", exc)

    def _play_click_sound(self, frequency: int = 1200, duration_ms: int = 40) -> None:
        """Play quick non-blocking audio feedback click."""
        if not self.enable_sound:
            return
        def _beep():
            try:
                import winsound
                winsound.Beep(frequency, duration_ms)
            except Exception:
                pass
        threading.Thread(target=_beep, daemon=True).start()

    def dispatch_left_click(self, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Trigger physical Left Click."""
        self._play_click_sound(frequency=1400, duration_ms=45)
        if self._is_win32 and self._user32:
            self._user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            self._user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            return True
        try:
            import pyautogui
            pyautogui.click(x, y)
            return True
        except Exception as exc:
            logger.error("Click dispatch failed: %s", exc)
            return False

    def dispatch_right_click(self, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Trigger physical Right Click."""
        self._play_click_sound(frequency=900, duration_ms=65)
        if self._is_win32 and self._user32:
            self._user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            self._user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            return True
        try:
            import pyautogui
            pyautogui.rightClick(x, y)
            return True
        except Exception as exc:
            logger.error("Right click dispatch failed: %s", exc)
            return False

    def dispatch_double_click(self, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Trigger physical Double Click."""
        self._play_click_sound(frequency=1800, duration_ms=40)
        if self._is_win32 and self._user32:
            self._user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            self._user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            time.sleep(0.04)
            self._user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            self._user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            return True
        try:
            import pyautogui
            pyautogui.doubleClick(x, y)
            return True
        except Exception as exc:
            logger.error("Double click dispatch failed: %s", exc)
            return False


class BlinkClickDetector:
    """Temporal finite-state machine classifying intentional blinks and winks into click actions."""

    def __init__(self, config: Optional[BlinkDetectorConfig] = None) -> None:
        self.config = config or BlinkDetectorConfig()
        self.dispatcher = ClickDispatcher(enable_sound=self.config.enable_sound_feedback)

        # Bilateral state tracking
        self._bilateral_closed: bool = False
        self._blink_start_time: Optional[float] = None
        self._last_blink_duration_ms: float = 0.0

        # Unilateral wink tracking
        self._left_wink_start: Optional[float] = None
        self._right_wink_start: Optional[float] = None

        # Double click & debounce state
        self._pending_single_click: bool = False
        self._pending_single_click_time: float = 0.0
        self._pending_single_click_pos: Tuple[int, int] = (0, 0)
        self._last_dispatched_click_time: float = -999.0
        self._last_click_type: ClickType = ClickType.NONE

        logger.info(
            "BlinkClickDetector initialized (min_blink=%.0fms, max_blink=%.0fms, wink=%s, debounce=%.0fms)",
            self.config.min_blink_duration_ms,
            self.config.max_click_blink_duration_ms,
            self.config.enable_wink_right_click,
            self.config.click_debounce_ms,
        )

    def reset(self) -> None:
        """Reset internal temporal counters and pending click buffers."""
        self._bilateral_closed = False
        self._blink_start_time = None
        self._last_blink_duration_ms = 0.0
        self._left_wink_start = None
        self._right_wink_start = None
        self._pending_single_click = False
        self._pending_single_click_time = 0.0
        self._last_dispatched_click_time = -999.0
        self._last_click_type = ClickType.NONE

    def process(
        self,
        eye_features: ExtractedEyeFeatures,
        cursor_x: int,
        cursor_y: int,
        timestamp: Optional[float] = None,
    ) -> Tuple[Optional[ClickEvent], BlinkState]:
        """Evaluate current eye features and return any triggered ClickEvent and live telemetry.
        
        Args:
            eye_features: Extracted EAR and eye perimeters from EyeTracker.
            cursor_x: Current screen cursor X coordinate.
            cursor_y: Current screen cursor Y coordinate.
            timestamp: Optional mockable timestamp for unit tests.
            
        Returns:
            Tuple of (Optional[ClickEvent], BlinkState).
        """
        now = timestamp if timestamp is not None else time.perf_counter()
        triggered_event: Optional[ClickEvent] = None
        status_msg = "Eyes Open"

        if not eye_features.has_eyes or not eye_features.left_eye or not eye_features.right_eye:
            return None, BlinkState(
                is_blinking=False,
                blink_duration_ms=0.0,
                left_ear=0.0,
                right_ear=0.0,
                left_closed=True,
                right_closed=True,
                is_debouncing=False,
                status_message="No Eyes Detected",
            )

        # Map physiological left/right eyes based on mirror orientation
        # In mirrored webcam video (standard selfie mode):
        # - Screen-left eye (x < 0.5) corresponds to user's physical LEFT EYE.
        # - Screen-right eye (x > 0.5) corresponds to user's physical RIGHT EYE.
        # MediaPipe FaceMesh labels the screen-left eye as `right_eye` and screen-right eye as `left_eye`.
        if self.config.mirror_mode:
            user_left_eye = eye_features.right_eye
            user_right_eye = eye_features.left_eye
        else:
            user_left_eye = eye_features.left_eye
            user_right_eye = eye_features.right_eye

        left_ear = user_left_eye.ear
        right_ear = user_right_eye.ear
        avg_ear = (left_ear + right_ear) / 2.0
        left_closed = left_ear < self.config.ear_threshold
        right_closed = right_ear < self.config.ear_threshold

        # Robust bilateral closure detection:
        # Both eyes must be closed; avoids false bilateral triggers on single-eye winks
        relaxed_ceiling = self.config.ear_threshold + 0.04
        both_closed = (left_closed and right_closed) or (
            avg_ear < self.config.ear_threshold
            and left_ear < relaxed_ceiling
            and right_ear < relaxed_ceiling
        )
        both_open = not both_closed and (not left_closed and not right_closed)

        is_debouncing = (self._last_dispatched_click_time >= 0.0) and (
            (now - self._last_dispatched_click_time) * 1000.0 < self.config.click_debounce_ms
        )

        # -------------------------------------------------------------
        # 1. Bilateral Closure (Both Eyes Closed)
        # -------------------------------------------------------------
        if both_closed:
            if not self._bilateral_closed:
                # Transition: Eyes just closed
                self._bilateral_closed = True
                self._blink_start_time = now
                self._left_wink_start = None
                self._right_wink_start = None

            current_duration_ms = (now - self._blink_start_time) * 1000.0
            if current_duration_ms > self.config.long_blink_right_click_ms:
                status_msg = f"Hold for Right Click ({int(current_duration_ms)}ms)"
            elif current_duration_ms >= self.config.min_blink_duration_ms:
                status_msg = f"Intentional Blink ({int(current_duration_ms)}ms)"
            else:
                status_msg = f"Blinking ({int(current_duration_ms)}ms)"

        elif self._bilateral_closed and both_open:
            # Transition: Eyes just reopened after bilateral closure
            self._bilateral_closed = False
            duration_ms = (now - self._blink_start_time) * 1000.0 if self._blink_start_time is not None else 0.0
            self._last_blink_duration_ms = duration_ms
            self._blink_start_time = None

            if not is_debouncing:
                # Classify duration:
                if duration_ms >= self.config.long_blink_right_click_ms:
                    # Prolonged deliberate closure -> Right Click
                    triggered_event = ClickEvent(
                        click_type=ClickType.RIGHT_CLICK,
                        x=cursor_x,
                        y=cursor_y,
                        duration_ms=round(duration_ms, 1),
                        timestamp=now,
                    )
                    self.dispatcher.dispatch_right_click(cursor_x, cursor_y)
                    self._last_dispatched_click_time = now
                    self._last_click_type = ClickType.RIGHT_CLICK
                    self._pending_single_click = False
                    status_msg = f"RIGHT CLICK [Long Blink {int(duration_ms)}ms]"

                elif duration_ms >= self.config.min_blink_duration_ms and duration_ms <= self.config.max_click_blink_duration_ms:
                    # Intentional blink in Left Click window
                    if self._pending_single_click:
                        time_since_first = (now - self._pending_single_click_time) * 1000.0
                        if time_since_first <= self.config.double_click_window_ms:
                            # Second blink arrived within double click window -> DOUBLE CLICK!
                            triggered_event = ClickEvent(
                                click_type=ClickType.DOUBLE_CLICK,
                                x=cursor_x,
                                y=cursor_y,
                                duration_ms=round(duration_ms, 1),
                                timestamp=now,
                            )
                            self.dispatcher.dispatch_double_click(cursor_x, cursor_y)
                            self._last_dispatched_click_time = now
                            self._last_click_type = ClickType.DOUBLE_CLICK
                            self._pending_single_click = False
                            status_msg = "DOUBLE CLICK TRIGGERED!"
                        else:
                            # Previous pending expired, restart pending with current
                            self._pending_single_click = True
                            self._pending_single_click_time = now
                            self._pending_single_click_pos = (cursor_x, cursor_y)
                    else:
                        # First intentional blink: mark as pending double-click opportunity
                        self._pending_single_click = True
                        self._pending_single_click_time = now
                        self._pending_single_click_pos = (cursor_x, cursor_y)
                else:
                    status_msg = f"Involuntary Blink Ignored ({int(duration_ms)}ms)"

        # -------------------------------------------------------------
        # 2. Pending Single Click Expiration Check (Resolves Left Click)
        # -------------------------------------------------------------
        if self._pending_single_click and not both_closed:
            elapsed = (now - self._pending_single_click_time) * 1000.0
            if elapsed > self.config.double_click_window_ms:
                # No second blink arrived: Confirm single LEFT CLICK
                click_x, click_y = self._pending_single_click_pos
                triggered_event = ClickEvent(
                    click_type=ClickType.LEFT_CLICK,
                    x=click_x,
                    y=click_y,
                    duration_ms=round(self._last_blink_duration_ms, 1),
                    timestamp=now,
                )
                self.dispatcher.dispatch_left_click(click_x, click_y)
                self._last_dispatched_click_time = now
                self._last_click_type = ClickType.LEFT_CLICK
                self._pending_single_click = False
                status_msg = f"LEFT CLICK DISPATCHED ({int(self._last_blink_duration_ms)}ms blink)"

        # -------------------------------------------------------------
        # 3. Unilateral Wink Detection: Left Wink = Left Click, Right Wink = Right Click
        # -------------------------------------------------------------
        if (self.config.enable_wink_clicking or self.config.enable_wink_right_click) and not both_closed:
            # User Left Eye Wink (Left Closed, Right Open) -> LEFT CLICK
            if left_closed and not right_closed:
                if self._left_wink_start is None:
                    self._left_wink_start = now
                wink_dur = (now - self._left_wink_start) * 1000.0
                status_msg = f"Left Eye Wink ({int(wink_dur)}ms) [Hold to Left Click]"
            elif self._left_wink_start is not None and not left_closed:
                wink_dur = (now - self._left_wink_start) * 1000.0
                self._left_wink_start = None
                if wink_dur >= self.config.wink_min_duration_ms and not is_debouncing:
                    triggered_event = ClickEvent(
                        click_type=ClickType.LEFT_CLICK,
                        x=cursor_x,
                        y=cursor_y,
                        duration_ms=round(wink_dur, 1),
                        timestamp=now,
                    )
                    self.dispatcher.dispatch_left_click(cursor_x, cursor_y)
                    self._last_dispatched_click_time = now
                    self._last_click_type = ClickType.LEFT_CLICK
                    self._pending_single_click = False
                    status_msg = f"LEFT CLICK [Left Eye Wink {int(wink_dur)}ms]"

            # User Right Eye Wink (Right Closed, Left Open) -> RIGHT CLICK
            if right_closed and not left_closed:
                if self._right_wink_start is None:
                    self._right_wink_start = now
                wink_dur = (now - self._right_wink_start) * 1000.0
                status_msg = f"Right Eye Wink ({int(wink_dur)}ms) [Hold to Right Click]"
            elif self._right_wink_start is not None and not right_closed:
                wink_dur = (now - self._right_wink_start) * 1000.0
                self._right_wink_start = None
                if wink_dur >= self.config.wink_min_duration_ms and not is_debouncing:
                    triggered_event = ClickEvent(
                        click_type=ClickType.RIGHT_CLICK,
                        x=cursor_x,
                        y=cursor_y,
                        duration_ms=round(wink_dur, 1),
                        timestamp=now,
                    )
                    self.dispatcher.dispatch_right_click(cursor_x, cursor_y)
                    self._last_dispatched_click_time = now
                    self._last_click_type = ClickType.RIGHT_CLICK
                    self._pending_single_click = False
                    status_msg = f"RIGHT CLICK [Right Eye Wink {int(wink_dur)}ms]"

        active_dur = 0.0
        if self._bilateral_closed and self._blink_start_time is not None:
            active_dur = (now - self._blink_start_time) * 1000.0

        state = BlinkState(
            is_blinking=self._bilateral_closed,
            blink_duration_ms=round(active_dur, 1),
            left_ear=round(left_ear, 3),
            right_ear=round(right_ear, 3),
            left_closed=left_closed,
            right_closed=right_closed,
            is_debouncing=is_debouncing,
            last_click_type=self._last_click_type,
            last_click_time=self._last_dispatched_click_time,
            status_message=status_msg,
        )

        return triggered_event, state
