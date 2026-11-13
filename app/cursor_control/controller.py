"""AdaptiveEye OS Cursor Controller & Zero-Trust Safety Interlock.

Translates stabilized, drift-compensated gaze targeting coordinates into native Windows OS
mouse cursor movements via sub-millisecond Win32 API dispatch with ballistic acceleration curves,
user-toggle hotkeys (F8), emergency corner failsafes, and zero-trust biometric interlocks.
"""

import ctypes
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Optional, Tuple
import cv2
import numpy as np
import yaml

from app.camera.webcam import CameraConfig, CameraManager
from app.calibration.collector import get_primary_monitor_resolution
from app.calibration.mapper import PolynomialGazeMapper
from app.core.datatypes import Point2D
from app.core.logger import get_logger
from app.eye_tracking.tracker import ExtractedEyeFeatures, EyeTracker, EyeTrackingConfig
from app.face_detection.detector import FaceDetector, FaceDetectorConfig
from app.face_verification.verifier import AuthState, FaceVerifier
from app.gaze_estimation.estimator import GazeConfig, GazeEstimator
from app.adaptive_calibration.drift_compensator import AdaptiveCalibrator, AdaptiveConfig
from app.blink_detection.detector import BlinkClickDetector, BlinkDetectorConfig, ClickEvent, ClickType
from app.scrolling.scroller import DwellClicker, EyeScroller, ScrollingConfig, draw_scrolling_overlay
from app.stabilization.stabilizer import AdaptiveCursorStabilizer, StabilizerConfig

logger = get_logger("CursorController")

# Windows Virtual Key Codes
VK_F8 = 0x77
VK_ESCAPE = 0x1B


class POINT(ctypes.Structure):
    """Win32 POINT structure."""
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


@dataclass
class CursorControlConfig:
    """Strongly-typed parameters for OS cursor control."""
    enable_mouse_dispatch: bool = True
    mouse_backend: str = "win32"
    smoothing_factor: float = 0.35
    acceleration_curve: str = "sigmoid"
    toggle_hotkey_vkey: int = VK_F8
    emergency_corner_failsafe: bool = True
    corner_failsafe_frames: int = 1
    safety_interlock_require_verified: bool = True
    safety_interlock_pause_on_blink: bool = True
    cursor_update_hz: float = 60.0

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "CursorControlConfig":
        if not yaml_path.exists():
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        c_data = data.get("cursor_control", {})
        return cls(
            enable_mouse_dispatch=bool(c_data.get("enable_mouse_dispatch", True)),
            mouse_backend=str(c_data.get("mouse_backend", "win32")),
            smoothing_factor=float(c_data.get("smoothing_factor", 0.35)),
            acceleration_curve=str(c_data.get("acceleration_curve", "sigmoid")),
            toggle_hotkey_vkey=int(c_data.get("toggle_hotkey_vkey", VK_F8)),
            emergency_corner_failsafe=bool(c_data.get("emergency_corner_failsafe", True)),
            corner_failsafe_frames=int(c_data.get("corner_failsafe_frames", 1)),
            safety_interlock_require_verified=bool(c_data.get("safety_interlock_require_verified", True)),
            safety_interlock_pause_on_blink=bool(c_data.get("safety_interlock_pause_on_blink", True)),
            cursor_update_hz=float(c_data.get("cursor_update_hz", 60.0)),
        )


@dataclass
class MouseState:
    """Real-time cursor dispatch state."""
    current_x: int
    current_y: int
    target_x: int
    target_y: int
    is_engaged: bool
    is_interlocked: bool
    interlock_reason: str
    velocity_px_s: float
    dispatch_latency_ms: float


class OSCursorController:
    """Low-latency native Windows OS cursor dispatch engine with safety interlocks."""

    def __init__(self, config: Optional[CursorControlConfig] = None) -> None:
        self.config = config or CursorControlConfig()
        self.screen_w, self.screen_h = get_primary_monitor_resolution()

        self._user32 = None
        self._is_win32_available = False
        if sys.platform == "win32":
            try:
                self._user32 = ctypes.windll.user32
                self._user32.SetProcessDPIAware()
                self._is_win32_available = True
            except Exception as exc:
                logger.warning("Could not initialize Win32 user32 (%s); falling back to mock mode.", exc)

        # Engagement state
        self.is_engaged: bool = self.config.enable_mouse_dispatch
        self._last_hotkey_press: float = 0.0

        # Motion smoothing
        curr_x, curr_y = self.get_physical_cursor_pos()
        self._curr_x: float = float(curr_x)
        self._curr_y: float = float(curr_y)
        self._last_dispatch_time: float = time.perf_counter()
        self._min_dispatch_interval: float = 1.0 / max(1.0, self.config.cursor_update_hz)
        self._corner_dwell_frames: int = 0

        logger.info(
            "OSCursorController initialized (dispatch=%s, backend=%s, screen=%dx%d)",
            self.config.enable_mouse_dispatch,
            self.config.mouse_backend,
            self.screen_w,
            self.screen_h,
        )

    def get_physical_cursor_pos(self) -> Tuple[int, int]:
        """Query current OS physical mouse cursor location."""
        if self._is_win32_available and self._user32:
            pt = POINT()
            if self._user32.GetCursorPos(ctypes.byref(pt)):
                return int(pt.x), int(pt.y)
        return int(self.screen_w // 2), int(self.screen_h // 2)

    def check_hotkey_toggle(self) -> bool:
        """Poll asynchronous keyboard state to toggle eye mouse on/off."""
        if not self._is_win32_available or not self._user32:
            return self.is_engaged

        now = time.perf_counter()
        # Debounce hotkey (300ms window)
        if now - self._last_hotkey_press < 0.30:
            return self.is_engaged

        # Check if F8 key is pressed (high bit set)
        state = self._user32.GetAsyncKeyState(self.config.toggle_hotkey_vkey)
        if state & 0x8000:
            self.is_engaged = not self.is_engaged
            self._last_hotkey_press = now
            status_str = "ENABLED" if self.is_engaged else "PAUSED"
            logger.info("Eye Mouse Control toggled via hotkey: %s", status_str)
            print(f"\n[F8 Hotkey] Eye Mouse Control: {status_str}")

        return self.is_engaged

    def compute_ballistic_step(self, target_x: int, target_y: int, dt: float) -> Tuple[int, int]:
        """Calculate smooth, non-linear acceleration step toward target coordinates."""
        dx = target_x - self._curr_x
        dy = target_y - self._curr_y
        dist = float(np.hypot(dx, dy))

        # Precision Deadband: If within 4 pixels, lock to prevent icon-click jitter
        if dist < 4.0:
            return int(round(self._curr_x)), int(round(self._curr_y))

        # Non-linear Ballistic Acceleration Curve:
        # Small distances (< 45px): Fine precision pointing (slow, smooth, pixel-accurate)
        # Large distances (>= 45px): Rapid ballistic jump across the monitor
        if dist < 45.0:
            # Fine adjustment interpolation
            interp = 0.22
        elif dist < 200.0:
            # Medium approach
            interp = 0.40
        else:
            # Ballistic saccade sweep
            interp = 0.65

        # Move current coordinate toward target
        self._curr_x += dx * interp
        self._curr_y += dy * interp

        clamped_x = int(np.clip(round(self._curr_x), 0, self.screen_w - 1))
        clamped_y = int(np.clip(round(self._curr_y), 0, self.screen_h - 1))
        return clamped_x, clamped_y

    def dispatch(
        self,
        target_x: int,
        target_y: int,
        is_gaze_valid: bool = True,
        is_user_authorized: bool = True,
    ) -> MouseState:
        """Evaluate safety interlocks and dispatch mouse pointer to operating system."""
        start_time = time.perf_counter()
        now = start_time
        dt = max(1e-4, now - self._last_dispatch_time)

        # Check F8 hotkey toggle
        self.check_hotkey_toggle()

        # Query where the physical mouse currently is
        phys_x, phys_y = self.get_physical_cursor_pos()

        # 1. Emergency Corner Failsafe Check:
        # If the user grabs their physical mouse and slams it into the top-left corner (0,0), disengage eye control
        if self.config.emergency_corner_failsafe and phys_x <= 5 and phys_y <= 5:
            self._corner_dwell_frames += 1
            if self._corner_dwell_frames >= self.config.corner_failsafe_frames:
                self.is_engaged = False
                return MouseState(
                    current_x=phys_x,
                    current_y=phys_y,
                    target_x=target_x,
                    target_y=target_y,
                    is_engaged=False,
                    is_interlocked=True,
                    interlock_reason="EMERGENCY CORNER FAILSAFE TRIGGERED (PRESS F8 TO RESUME)",
                    velocity_px_s=0.0,
                    dispatch_latency_ms=(time.perf_counter() - start_time) * 1000.0,
                )
        else:
            self._corner_dwell_frames = 0

        # 2. Check User Engagement (Hotkey F8)
        if not self.is_engaged:
            return MouseState(
                current_x=phys_x,
                current_y=phys_y,
                target_x=target_x,
                target_y=target_y,
                is_engaged=False,
                is_interlocked=True,
                interlock_reason="EYE MOUSE PAUSED BY USER [PRESS F8 TO RESUME]",
                velocity_px_s=0.0,
                dispatch_latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        # 3. Zero-Trust Biometric Security Interlock:
        if self.config.safety_interlock_require_verified and not is_user_authorized:
            return MouseState(
                current_x=phys_x,
                current_y=phys_y,
                target_x=target_x,
                target_y=target_y,
                is_engaged=True,
                is_interlocked=True,
                interlock_reason="ZERO-TRUST SAFETY INTERLOCK: IDENTITY NOT AUTHORIZED",
                velocity_px_s=0.0,
                dispatch_latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        # 4. Blink / Eye-Loss Safety Interlock:
        if self.config.safety_interlock_pause_on_blink and not is_gaze_valid:
            return MouseState(
                current_x=phys_x,
                current_y=phys_y,
                target_x=target_x,
                target_y=target_y,
                is_engaged=True,
                is_interlocked=True,
                interlock_reason="SAFETY INTERLOCK: EYES CLOSED / BLINK DETECTED",
                velocity_px_s=0.0,
                dispatch_latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        # Rate-limiting check
        if now - self._last_dispatch_time < self._min_dispatch_interval:
            return MouseState(
                current_x=int(self._curr_x),
                current_y=int(self._curr_y),
                target_x=target_x,
                target_y=target_y,
                is_engaged=True,
                is_interlocked=False,
                interlock_reason="Active",
                velocity_px_s=0.0,
                dispatch_latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        self._last_dispatch_time = now

        # Compute ballistic motion
        new_x, new_y = self.compute_ballistic_step(target_x, target_y, dt)

        # Dispatch to Windows OS User32 API with multi-layer fallback
        if self.config.enable_mouse_dispatch:
            dispatched = False
            if self._is_win32_available and self._user32:
                res = self._user32.SetCursorPos(new_x, new_y)
                if res != 0:
                    dispatched = True
                else:
                    # Fallback to mouse_event (MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE = 0x8001)
                    norm_x = int(new_x * 65535 / max(1, self.screen_w - 1))
                    norm_y = int(new_y * 65535 / max(1, self.screen_h - 1))
                    self._user32.mouse_event(0x8001, norm_x, norm_y, 0, 0)
                    dispatched = True
            if not dispatched:
                try:
                    import pyautogui
                    pyautogui.moveTo(new_x, new_y)
                except Exception:
                    pass

        velocity = float(np.hypot(new_x - phys_x, new_y - phys_y)) / dt
        latency = (time.perf_counter() - start_time) * 1000.0

        return MouseState(
            current_x=new_x,
            current_y=new_y,
            target_x=target_x,
            target_y=target_y,
            is_engaged=True,
            is_interlocked=False,
            interlock_reason="Operating Nominally",
            velocity_px_s=round(velocity, 1),
            dispatch_latency_ms=round(latency, 2),
        )


def run_os_cursor_control_live() -> None:
    """Launch full live AdaptiveEye eye-mouse system driving Windows OS cursor."""
    screen_w, screen_h = get_primary_monitor_resolution()

    # Load Subsystems
    cam = CameraManager(CameraConfig(width=1280, height=720, mirror=True))
    face_detector = FaceDetector(FaceDetectorConfig(max_num_faces=4, min_detection_confidence=0.55, allow_multiple_faces=True))
    face_verifier = FaceVerifier()
    eye_tracker = EyeTracker(EyeTrackingConfig(refine_landmarks=True, max_num_faces=4))
    gaze_cfg_path = Path("config/gaze_estimation_config.yaml")
    gaze_config = GazeConfig.from_yaml(gaze_cfg_path)
    gaze_config.screen_width = screen_w
    gaze_config.screen_height = screen_h
    gaze_estimator = GazeEstimator(gaze_config)
    calibrator = AdaptiveCalibrator(AdaptiveConfig.from_yaml(Path("config/adaptive_calibration_config.yaml")))
    stabilizer = AdaptiveCursorStabilizer(StabilizerConfig.from_yaml(Path("config/stabilization_config.yaml")))
    mouse_controller = OSCursorController(CursorControlConfig.from_yaml(Path("config/cursor_control_config.yaml")))
    mouse_controller.config.corner_failsafe_frames = 15  # Require ~0.5s dwell in corner during live session
    blink_detector = BlinkClickDetector(BlinkDetectorConfig.from_yaml(Path("config/blink_detection_config.yaml")))
    scroll_config = ScrollingConfig.from_yaml(Path("config/scrolling_config.yaml"))
    eye_scroller = EyeScroller(scroll_config)
    dwell_clicker = DwellClicker(scroll_config)

    window_name = "AdaptiveEye - Live OS Cursor Control"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 360)

    print("\n=================================================================")
    print("   AdaptiveEye: Real-Time OS Cursor, Blink & Scroll Engine Running")
    print("=================================================================")
    print("Controls:")
    print("  [C] or [A] - Calibrate Center: Look at screen center and press [C] to lock center gaze!")
    print("  [M] - Toggle Mode: Switch between Pure Iris (100% Eye) and Compound (Head-assisted)")
    print("  [F8] - Toggle Eye Mouse On / Off (Pause & Resume)")
    print("  [+] / [-] - Increase / Decrease gaze reach sensitivity")
    print("  [S] - Toggle Scrolling On / Off")
    print("  [Q] or [ESC] - Exit application\n")
    print("Active Gestures & Features:")
    print("  * MULTI-PERSON TOLERANCE: Multiple faces allowed; only authorized person controls cursor")
    print("  * SINGLE CLICK: One deliberate blink (~150-550ms) -> Left Click")
    print("  * DOUBLE CLICK: Two rapid intentional blinks (<380ms) -> Double Click")
    print("  * HEAD-TILT SCROLL: Tilt head UP to scroll UP, tilt head DOWN to scroll DOWN")
    print("  * EDGE SCROLL: Look towards top or bottom border to scroll")
    print("Safety Feature: Hold physical mouse in top-left corner (0,0) to disengage.")
    print("=================================================================\n")

    frame_count = 0
    is_authorized = True  # Defaults to True; verified periodically
    auth_label = "Authorized User"
    auth_centroid: Optional[Tuple[float, float]] = None
    auth_box: Optional[Tuple[int, int, int, int]] = None
    bystander_boxes: List[Tuple[int, int, int, int]] = []
    unauth_counter = 0
    click_flash_counter = 0
    last_click_name = ""
    last_valid_x = screen_w // 2
    last_valid_y = screen_h // 2
    calib_countdown_frames = 75  # ~2.5 seconds at 30 FPS for graceful startup calibration
    initial_calib_samples: List[Tuple[float, float, float, float]] = []
    is_center_calibrated = False

    try:
        cam.start()
        for frame in cam.stream():
            h, w = frame.shape[:2]
            frame_count += 1

            # 1. Multi-Face Detection & ArcFace Biometrics (Every 6 frames)
            # Multiple faces are permitted in the frame; system locks strictly onto the authorized user!
            if frame_count % 6 == 0:
                det_res = face_detector.detect(frame)
                current_bystanders: List[Tuple[int, int, int, int]] = []

                if det_res.count > 0:
                    best_face = None
                    best_sim = -1.0

                    # Find the face with the highest biometric match to the enrolled authorized template
                    if face_verifier.is_enrolled and face_verifier._template is not None:
                        for face in det_res.faces:
                            try:
                                aligned = face_verifier.extractor.align_face(frame, face.bbox, face.landmarks)
                                emb = face_verifier.extractor.extract_embedding(aligned)
                                sim = float(face_verifier.extractor.compute_similarity(emb, face_verifier._template))
                                if sim > best_sim:
                                    best_sim = sim
                                    best_face = face
                            except Exception:
                                pass

                    # If candidate matches enrolled user, verify with state hysteresis
                    target_candidate = best_face if (best_face is not None and best_sim >= face_verifier.low_threshold) else det_res.primary_face

                    if target_candidate is not None:
                        ver_res = face_verifier.verify(frame, target_candidate)
                        if ver_res.is_authorized:
                            unauth_counter = 0
                            is_authorized = True
                            auth_label = ver_res.user_name or "Authorized User"
                            cx = target_candidate.bbox.xmin + target_candidate.bbox.width / 2.0
                            cy = target_candidate.bbox.ymin + target_candidate.bbox.height / 2.0
                            auth_centroid = (cx, cy)
                            auth_box = (
                                int(target_candidate.bbox.xmin * w),
                                int(target_candidate.bbox.ymin * h),
                                int(target_candidate.bbox.width * w),
                                int(target_candidate.bbox.height * h),
                            )

                            # Record bystander faces for display
                            for face in det_res.faces:
                                if face is not target_candidate:
                                    current_bystanders.append((
                                        int(face.bbox.xmin * w),
                                        int(face.bbox.ymin * h),
                                        int(face.bbox.width * w),
                                        int(face.bbox.height * h),
                                    ))
                        else:
                            unauth_counter += 1
                            if unauth_counter >= 4:
                                is_authorized = False
                                auth_label = "Unverified Identity"
                    else:
                        unauth_counter += 1
                        if unauth_counter >= 4:
                            is_authorized = False
                            auth_label = "Unverified Identity"

                    bystander_boxes = current_bystanders
                else:
                    unauth_counter += 1
                    if unauth_counter >= 8:
                        is_authorized = False
                        auth_label = "No Face Detected"
                        bystander_boxes = []
                        auth_box = None

            # 2. Eye & Iris Feature Tracking (Targeted at Authorized Person)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            mesh_res = eye_tracker.mesh.process(rgb)

            all_lms = None
            if mesh_res.multi_face_landmarks:
                if auth_centroid is not None and len(mesh_res.multi_face_landmarks) > 1:
                    best_idx = 0
                    min_dist = float("inf")
                    for idx, f_lms in enumerate(mesh_res.multi_face_landmarks):
                        nose = f_lms.landmark[4]
                        dist = float(np.hypot(nose.x - auth_centroid[0], nose.y - auth_centroid[1]))
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = idx
                    target_mesh = mesh_res.multi_face_landmarks[best_idx]
                else:
                    target_mesh = mesh_res.multi_face_landmarks[0]

                all_lms = [
                    Point2D(x=float(lm.x), y=float(lm.y))
                    for lm in target_mesh.landmark
                ]

            eye_feats = eye_tracker.process(frame, target_centroid=auth_centroid)
            raw_gaze = gaze_estimator.estimate(frame, eye_feats, all_lms)

            # 2.5s Center Calibration Countdown & Sampling
            if not is_center_calibrated:
                if eye_feats.has_eyes and eye_feats.right_eye and eye_feats.left_eye:
                    avg_ix = (eye_feats.right_eye.iris_relative_x + eye_feats.left_eye.iris_relative_x) / 2.0
                    avg_iy = (eye_feats.right_eye.iris_relative_y + eye_feats.left_eye.iris_relative_y) / 2.0
                    yaw = raw_gaze.head_angles.yaw if raw_gaze.head_angles else 0.0
                    pitch = raw_gaze.head_angles.pitch if raw_gaze.head_angles else 0.0
                    if frame_count >= 25:
                        initial_calib_samples.append((avg_ix, avg_iy, yaw, pitch))

                if frame_count >= calib_countdown_frames:
                    if initial_calib_samples:
                        med_ix = float(np.median([s[0] for s in initial_calib_samples]))
                        med_iy = float(np.median([s[1] for s in initial_calib_samples]))
                        med_yaw = float(np.median([s[2] for s in initial_calib_samples]))
                        med_pitch = float(np.median([s[3] for s in initial_calib_samples]))
                    else:
                        med_ix, med_iy = 0.50, 0.50
                        med_yaw, med_pitch = 0.0, 0.0

                    if raw_gaze.head_angles and eye_feats.has_eyes:
                        calibrator.set_baseline(raw_gaze.head_angles, eye_feats, w, h)
                    gaze_estimator.set_neutral_posture(med_yaw, med_pitch, med_ix, med_iy)
                    mouse_controller._curr_x = float(screen_w // 2)
                    mouse_controller._curr_y = float(screen_h // 2)
                    if mouse_controller._is_win32_available and mouse_controller._user32:
                        mouse_controller._user32.SetCursorPos(screen_w // 2, screen_h // 2)
                    is_center_calibrated = True
                    print(f"\n[AdaptiveEye Auto-Center] Done! Neutral resting gaze locked to ({med_ix:.3f}, {med_iy:.3f})")
                    print("Cursor is now actively following your eye movements! Press [C] anytime to re-center.\n")

            # 3. Adaptive Posture Drift Compensation & Stabilization
            if raw_gaze.is_valid and is_center_calibrated:
                adapted_x, adapted_y, _ = calibrator.compensate(
                    raw_gaze.screen_x,
                    raw_gaze.screen_y,
                    raw_gaze.head_angles,
                    eye_feats,
                    w,
                    h,
                    screen_w,
                    screen_h,
                )
                cursor_state = stabilizer.stabilize(
                    adapted_x,
                    adapted_y,
                    confidence=raw_gaze.confidence,
                    is_valid=True,
                    monitor_w=screen_w,
                    monitor_h=screen_h,
                )
            elif not is_center_calibrated:
                # During calibration countdown, hold cursor rock-solid at center
                cursor_state = stabilizer.stabilize(
                    screen_w // 2,
                    screen_h // 2,
                    confidence=1.0,
                    is_valid=True,
                    monitor_w=screen_w,
                    monitor_h=screen_h,
                )
            else:
                cursor_state = stabilizer.stabilize(
                    0, 0, confidence=0.0, is_valid=False, monitor_w=screen_w, monitor_h=screen_h
                )

            # 4. OS Mouse Dispatch with Zero-Trust Safety Interlocks
            mouse_state = mouse_controller.dispatch(
                target_x=cursor_state.screen_x,
                target_y=cursor_state.screen_y,
                is_gaze_valid=cursor_state.is_valid,
                is_user_authorized=is_authorized,
            )

            if cursor_state.is_valid:
                last_valid_x = mouse_state.current_x
                last_valid_y = mouse_state.current_y

            # 5. Process Intentional Eye Blink Clicking & Edge Scrolling
            click_evt = None
            blink_state = None
            dwell_state = None
            scroll_state = None

            # Always process blinks while eye mouse is engaged and user is authorized,
            # so eye closures are reliably tracked across frames even when cursor is interlocked/frozen!
            if mouse_state.is_engaged and is_authorized:
                click_evt, blink_state = blink_detector.process(
                    eye_feats,
                    cursor_x=last_valid_x,
                    cursor_y=last_valid_y,
                )

                if click_evt:
                    click_flash_counter = 18
                    last_click_name = click_evt.click_type.value
                    print(f"\n[AdaptiveEye Click] >>> {last_click_name} triggered at ({click_evt.x}, {click_evt.y}) <<<")

                # Continuous Scrolling (Head-tilt look up/down or edge margins)
                if not mouse_state.is_interlocked:
                    curr_pitch = raw_gaze.head_angles.pitch if raw_gaze.head_angles else None
                    neut_pitch = gaze_estimator._neutral_pitch
                    scroll_evt, scroll_state = eye_scroller.process(
                        cursor_y=mouse_state.current_y,
                        screen_h=screen_h,
                        cursor_x=mouse_state.current_x,
                        is_active=True,
                        head_pitch=curr_pitch,
                        neutral_pitch=neut_pitch,
                    )

            # 6. Render Compact Video HUD and Overlays
            annotated = eye_tracker.draw_eye_annotations(frame, eye_feats)

            # Render yellow gaze vector arrows, 3D head pose axes & mini screen target
            if raw_gaze.is_valid:
                annotated = gaze_estimator.draw_gaze_overlay(
                    annotated,
                    raw_gaze,
                    eye_feats,
                    all_lms,
                    draw_banner=False,
                )

            # Render bystander indicators & authorized user tag
            for (bx, by, bw, bh) in bystander_boxes:
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (60, 160, 220), 1)
                cv2.putText(
                    annotated,
                    "Bystander (Ignored)",
                    (bx, max(15, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (60, 160, 220),
                    1,
                    cv2.LINE_AA,
                )
            if auth_box and is_authorized:
                (ax, ay, aw, ah) = auth_box
                cv2.rectangle(annotated, (ax, ay), (ax + aw, ay + ah), (0, 255, 120), 2)
                cv2.putText(
                    annotated,
                    f"Authorized: {auth_label}",
                    (ax, max(15, ay - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 255, 120),
                    1,
                    cv2.LINE_AA,
                )

            if scroll_state:
                annotated = draw_scrolling_overlay(annotated, scroll_state, dwell_state)

            # Status Banner
            status_bg = (20, 20, 24)
            cv2.rectangle(annotated, (0, 0), (w, 54), status_bg, -1)

            if mouse_state.is_engaged and not mouse_state.is_interlocked:
                status_color = (0, 255, 120)
                blink_info = f" | EAR: L:{blink_state.left_ear:.2f}, R:{blink_state.right_ear:.2f}" if blink_state else ""
                status_txt = f"EYE MOUSE: ACTIVE | Cursor: ({mouse_state.current_x}, {mouse_state.current_y}){blink_info}"
            elif mouse_state.is_interlocked and "BLINK" in mouse_state.interlock_reason:
                status_color = (0, 240, 255)
                blink_info = f" | {blink_state.status_message} (L:{blink_state.left_ear:.2f}, R:{blink_state.right_ear:.2f})" if blink_state else ""
                status_txt = f"BLINK DETECTED{blink_info}"
            elif not mouse_state.is_engaged:
                status_color = (0, 220, 255)
                status_txt = "EYE MOUSE: PAUSED (Press F8 to Resume)"
            else:
                status_color = (0, 0, 255)
                status_txt = f"INTERLOCK: {mouse_state.interlock_reason}"

            auth_col = (0, 255, 120) if is_authorized else (0, 0, 255)
            cv2.putText(annotated, status_txt, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 2, cv2.LINE_AA)
            approach_tag = "PURE IRIS" if gaze_estimator.config.estimation_approach == "pure_iris" else "COMPOUND"
            cv2.putText(annotated, f"Mode: {approach_tag} | Gain: {gaze_estimator.range_tracker.gain:.2f}x | [C] Center | [M] Mode | [F8] Toggle | [Q] Exit", (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.40, auth_col, 1, cv2.LINE_AA)

            # Startup Center Calibration Visual Target
            if not is_center_calibrated:
                secs_left = max(1, (calib_countdown_frames - frame_count + 29) // 30)
                box_w, box_h = 420, 50
                cv2.rectangle(annotated, (w // 2 - box_w // 2, h // 2 - box_h // 2), (w // 2 + box_w // 2, h // 2 + box_h // 2), (15, 15, 20), -1)
                cv2.rectangle(annotated, (w // 2 - box_w // 2, h // 2 - box_h // 2), (w // 2 + box_w // 2, h // 2 + box_h // 2), (0, 240, 255), 2)
                cv2.putText(annotated, f"LOOK AT SCREEN CENTER ({secs_left}s)", (w // 2 - 180, h // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 240, 255), 2, cv2.LINE_AA)
                cv2.drawMarker(annotated, (w // 2, h // 2), (0, 255, 120), cv2.MARKER_CROSS, 30, 2, cv2.LINE_AA)

            # Click Event Visual Feedback Overlay
            if click_flash_counter > 0:
                click_flash_counter -= 1
                badge_bg = (0, 180, 50) if "LEFT" in last_click_name else ((0, 100, 230) if "RIGHT" in last_click_name else (230, 140, 0))
                cv2.rectangle(annotated, (w // 2 - 150, 62), (w // 2 + 150, 102), badge_bg, -1)
                cv2.rectangle(annotated, (w // 2 - 150, 62), (w // 2 + 150, 102), (255, 255, 255), 2)
                cv2.putText(
                    annotated,
                    f">> {last_click_name} <<",
                    (w // 2 - 130, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("+"), ord("=")):
                gaze_estimator.range_tracker.gain = round(gaze_estimator.range_tracker.gain + 0.15, 2)
                print(f"Increased gaze sensitivity gain to: {gaze_estimator.range_tracker.gain:.2f}x")
            elif key in (ord("-"), ord("_")):
                gaze_estimator.range_tracker.gain = max(0.50, round(gaze_estimator.range_tracker.gain - 0.15, 2))
                print(f"Decreased gaze sensitivity gain to: {gaze_estimator.range_tracker.gain:.2f}x")
            elif key in (ord("c"), ord("C"), ord("a"), ord("A")):
                if eye_feats.has_eyes and eye_feats.right_eye and eye_feats.left_eye:
                    avg_ix = (eye_feats.right_eye.iris_relative_x + eye_feats.left_eye.iris_relative_x) / 2.0
                    avg_iy = (eye_feats.right_eye.iris_relative_y + eye_feats.left_eye.iris_relative_y) / 2.0
                    yaw = raw_gaze.head_angles.yaw if raw_gaze.head_angles else 0.0
                    pitch = raw_gaze.head_angles.pitch if raw_gaze.head_angles else 0.0
                    if raw_gaze.head_angles:
                        calibrator.set_baseline(raw_gaze.head_angles, eye_feats, w, h)
                    gaze_estimator.set_neutral_posture(yaw, pitch, avg_ix, avg_iy)
                    mouse_controller._curr_x = float(screen_w // 2)
                    mouse_controller._curr_y = float(screen_h // 2)
                    if mouse_controller._is_win32_available and mouse_controller._user32:
                        mouse_controller._user32.SetCursorPos(screen_w // 2, screen_h // 2)
                    stabilizer.reset()
                    is_center_calibrated = True
                    print(f"\n[AdaptiveEye Center Calibrated] Neutral eye center locked to ({avg_ix:.3f}, {avg_iy:.3f}) | Cursor centered.\n")
            elif key in (ord("m"), ord("M")):
                curr_app = gaze_estimator.config.estimation_approach
                new_app = "compound" if curr_app == "pure_iris" else "pure_iris"
                gaze_estimator.config.estimation_approach = new_app
                mode_lbl = "PURE IRIS (100% Eye Control)" if new_app == "pure_iris" else "COMPOUND (Iris + Head Assistance)"
                print(f"\n[AdaptiveEye Mode Toggle] Gaze tracking mode: {mode_lbl}\n")
            elif key in (ord("s"), ord("S")):
                eye_scroller.config.enable_edge_scrolling = not eye_scroller.config.enable_edge_scrolling
                state_str = "ENABLED" if eye_scroller.config.enable_edge_scrolling else "DISABLED"
                print(f"[AdaptiveEye] Gaze border scrolling {state_str}")
            elif key in (ord("d"), ord("D")):
                dwell_clicker.config.enable_dwell_clicking = not dwell_clicker.config.enable_dwell_clicking
                state_str = "ENABLED" if dwell_clicker.config.enable_dwell_clicking else "DISABLED"
                print(f"[AdaptiveEye] Fixation dwell clicking {state_str}")

    except Exception as exc:
        logger.error("Error in live cursor control session: %s", exc, exc_info=True)
        print(f"\n[AdaptiveEye Error] Session encountered an exception: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        cam.stop()
        eye_tracker.close()
        face_detector.close()
        cv2.destroyAllWindows()
