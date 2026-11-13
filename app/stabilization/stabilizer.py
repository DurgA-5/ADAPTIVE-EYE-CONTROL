"""AdaptiveEye Confidence-Aware Cursor Stabilization Engine.

Fuses speed-adaptive filtering, confidence-weighted Kalman estimation, and deadzone suppression
to deliver a rock-solid, jitter-free mouse cursor during ocular fixations while preserving
instantaneous saccadic responsiveness.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional, Tuple
import cv2
import numpy as np
import yaml

from app.core.logger import get_logger
from app.core.datatypes import Point2D
from app.camera.webcam import CameraConfig, CameraManager
from app.eye_tracking.tracker import EyeTracker, EyeTrackingConfig
from app.gaze_estimation.estimator import GazeConfig, GazeEstimator
from app.calibration.mapper import PolynomialGazeMapper
from app.calibration.collector import get_primary_monitor_resolution
from app.adaptive_calibration.drift_compensator import AdaptiveCalibrator, AdaptiveConfig
from app.stabilization.filters import AdaptiveEMAFilter, ConfidenceKalmanFilter

logger = get_logger("CursorStabilizer")


@dataclass
class StabilizerConfig:
    """Strongly-typed cursor stabilization parameters."""
    algorithm: str = "adaptive_ema"  # "adaptive_ema", "kalman", or "hybrid"
    deadzone_radius_px: float = 6.0
    min_alpha: float = 0.08
    max_alpha: float = 0.85
    saccade_velocity_threshold_px_s: float = 480.0
    kalman_process_noise: float = 0.08
    kalman_measurement_noise_base: float = 1.2
    confidence_noise_multiplier: float = 12.0
    hold_position_on_loss: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "StabilizerConfig":
        if not yaml_path.exists():
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        s_data = data.get("stabilization", {})
        return cls(
            algorithm=str(s_data.get("algorithm", "adaptive_ema")),
            deadzone_radius_px=float(s_data.get("deadzone_radius_px", 6.0)),
            min_alpha=float(s_data.get("min_alpha", 0.08)),
            max_alpha=float(s_data.get("max_alpha", 0.85)),
            saccade_velocity_threshold_px_s=float(s_data.get("saccade_velocity_threshold_px_s", 480.0)),
            kalman_process_noise=float(s_data.get("kalman_process_noise", 0.08)),
            kalman_measurement_noise_base=float(s_data.get("kalman_measurement_noise_base", 1.2)),
            confidence_noise_multiplier=float(s_data.get("confidence_noise_multiplier", 12.0)),
            hold_position_on_loss=bool(s_data.get("hold_position_on_loss", True)),
        )


@dataclass
class StabilizedCursorState:
    """Complete telemetry state of stabilized cursor."""
    screen_x: int
    screen_y: int
    raw_x: int
    raw_y: int
    jitter_reduced_px: float
    is_fixating: bool
    velocity_px_s: float
    alpha: float
    confidence: float
    is_valid: bool
    status_message: str


class AdaptiveCursorStabilizer:
    """High-performance confidence-aware cursor stabilization engine."""

    def __init__(self, config: Optional[StabilizerConfig] = None) -> None:
        self.config = config or StabilizerConfig()

        self.ema_filter = AdaptiveEMAFilter(
            min_alpha=self.config.min_alpha,
            max_alpha=self.config.max_alpha,
            saccade_velocity_threshold=self.config.saccade_velocity_threshold_px_s,
            deadzone_radius_px=self.config.deadzone_radius_px,
        )

        self.kalman_filter = ConfidenceKalmanFilter(
            process_noise=self.config.kalman_process_noise,
            measurement_noise_base=self.config.kalman_measurement_noise_base,
            confidence_multiplier=self.config.confidence_noise_multiplier,
        )

        self._last_stable_x: int = 960
        self._last_stable_y: int = 540
        self._has_stable_point: bool = False
        self._fixation_anchor_x: int = 960
        self._fixation_anchor_y: int = 540
        self._fixation_counter: int = 0

        logger.info(
            "AdaptiveCursorStabilizer initialized (algo=%s, deadzone=%.1f px, min_alpha=%.2f, max_alpha=%.2f)",
            self.config.algorithm,
            self.config.deadzone_radius_px,
            self.config.min_alpha,
            self.config.max_alpha,
        )

    def reset(self) -> None:
        """Reset internal filters and historical coordinates."""
        self.ema_filter.reset()
        self.kalman_filter.reset()
        self._has_stable_point = False
        self._fixation_counter = 0

    def stabilize(
        self,
        raw_x: int,
        raw_y: int,
        confidence: float = 1.0,
        is_valid: bool = True,
        monitor_w: int = 1920,
        monitor_h: int = 1080,
    ) -> StabilizedCursorState:
        """Process incoming gaze coordinates and output rock-solid stabilized screen position."""
        # Safety Interlock: During blinks or occlusion, hold last known position
        if not is_valid:
            held_x = self._last_stable_x if (self.config.hold_position_on_loss and self._has_stable_point) else raw_x
            held_y = self._last_stable_y if (self.config.hold_position_on_loss and self._has_stable_point) else raw_y
            return StabilizedCursorState(
                screen_x=held_x,
                screen_y=held_y,
                raw_x=raw_x,
                raw_y=raw_y,
                jitter_reduced_px=0.0,
                is_fixating=True,
                velocity_px_s=0.0,
                alpha=0.0,
                confidence=0.0,
                is_valid=False,
                status_message="Tracking Paused (Eyes Closed / Blink)",
            )

        algo = self.config.algorithm
        velocity = 0.0
        alpha = self.config.min_alpha
        is_deadzone = False

        if algo == "kalman":
            kx, ky, velocity = self.kalman_filter.update(float(raw_x), float(raw_y), confidence=confidence)
            smooth_x, smooth_y = kx, ky
        elif algo == "hybrid":
            # Pass through confidence-weighted Kalman first, then apply deadzone EMA
            kx, ky, _ = self.kalman_filter.update(float(raw_x), float(raw_y), confidence=confidence)
            ema_res = self.ema_filter.filter(kx, ky)
            smooth_x, smooth_y = ema_res.x, ema_res.y
            velocity = ema_res.velocity_px_s
            alpha = ema_res.alpha
            is_deadzone = ema_res.is_deadzone_active
        else:  # "adaptive_ema" (default ultra-responsive mode)
            ema_res = self.ema_filter.filter(float(raw_x), float(raw_y))
            smooth_x, smooth_y = ema_res.x, ema_res.y
            velocity = ema_res.velocity_px_s
            alpha = ema_res.alpha
            is_deadzone = ema_res.is_deadzone_active

        # Strictly clamp within monitor pixel bounds
        clamped_x = int(np.clip(round(smooth_x), 0, monitor_w - 1))
        clamped_y = int(np.clip(round(smooth_y), 0, monitor_h - 1))

        # Fixation Dwell Anchor Engine:
        # When user looks steadily at one location, FREEZE the cursor completely stationary!
        fixation_dist = float(np.hypot(clamped_x - self._fixation_anchor_x, clamped_y - self._fixation_anchor_y))
        if fixation_dist < 18.0 or is_deadzone:
            self._fixation_counter += 1
            if self._fixation_counter >= 3:
                clamped_x = self._fixation_anchor_x
                clamped_y = self._fixation_anchor_y
                is_deadzone = True
        else:
            self._fixation_anchor_x = clamped_x
            self._fixation_anchor_y = clamped_y
            self._fixation_counter = 0

        self._last_stable_x = clamped_x
        self._last_stable_y = clamped_y
        self._has_stable_point = True

        jitter_dist = float(np.hypot(raw_x - clamped_x, raw_y - clamped_y))
        is_fixating = is_deadzone or (velocity < 120.0) or (self._fixation_counter >= 3)
        mode_str = "FIXATION [ROCK-SOLID]" if is_fixating else "SACCADE [FAST-TRACK]"

        return StabilizedCursorState(
            screen_x=clamped_x,
            screen_y=clamped_y,
            raw_x=raw_x,
            raw_y=raw_y,
            jitter_reduced_px=round(jitter_dist, 1),
            is_fixating=is_fixating,
            velocity_px_s=round(velocity, 1),
            alpha=round(alpha, 3),
            confidence=round(confidence, 2),
            is_valid=True,
            status_message=mode_str,
        )

    def draw_telemetry_hud(
        self,
        canvas: np.ndarray,
        state: StabilizedCursorState,
    ) -> np.ndarray:
        """Render side-by-side gaze stabilization HUD and jitter suppression metrics."""
        h, w = canvas.shape[:2]

        if state.is_valid:
            # 1. Draw raw gaze indicator in subdued orange/red
            cv2.circle(canvas, (state.raw_x, state.raw_y), 8, (0, 100, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, (state.raw_x, state.raw_y), 3, (0, 100, 255), -1, cv2.LINE_AA)

            # 2. Draw connector line between raw and stabilized cursor
            if state.jitter_reduced_px > 3.0:
                cv2.line(canvas, (state.raw_x, state.raw_y), (state.screen_x, state.screen_y), (60, 80, 100), 1, cv2.LINE_AA)

            # 3. Draw rock-solid stabilized cursor in emerald cyan
            cursor_col = (0, 255, 120) if state.is_fixating else (0, 220, 255)
            cv2.circle(canvas, (state.screen_x, state.screen_y), 20, cursor_col, 2, cv2.LINE_AA)
            cv2.circle(canvas, (state.screen_x, state.screen_y), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.drawMarker(canvas, (state.screen_x, state.screen_y), cursor_col, cv2.MARKER_CROSS, 28, 1, cv2.LINE_AA)

        # Telemetry panel in top-right
        panel_w, panel_h = 320, 120
        x0, y0 = w - panel_w - 20, 20
        cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h), (25, 28, 34), -1)
        border_col = (0, 255, 120) if state.is_fixating else (0, 200, 255)
        cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h), border_col, 1)

        lines = [
            f"Mode: {state.status_message}",
            f"Cursor: ({state.screen_x}, {state.screen_y}) | Raw: ({state.raw_x}, {state.raw_y})",
            f"Jitter Suppressed: {state.jitter_reduced_px:.1f} px",
            f"Speed: {state.velocity_px_s:.1f} px/s | Alpha: {state.alpha:.2f}",
            f"Tracking Conf: {state.confidence:.2f}",
        ]

        for i, text in enumerate(lines):
            col = (0, 255, 120) if i == 0 and state.is_fixating else (220, 220, 220)
            cv2.putText(
                canvas,
                text,
                (x0 + 10, y0 + 22 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                col,
                1,
                cv2.LINE_AA,
            )

        return canvas


def run_stabilized_cursor_live() -> None:
    """Launch full end-to-end stabilized cursor tracking sandbox."""
    stab_cfg_file = Path("config/stabilization_config.yaml")
    stab_config = StabilizerConfig.from_yaml(stab_cfg_file)
    stabilizer = AdaptiveCursorStabilizer(stab_config)

    adapt_cfg_file = Path("config/adaptive_calibration_config.yaml")
    adapt_config = AdaptiveConfig.from_yaml(adapt_cfg_file)
    calibrator = AdaptiveCalibrator(adapt_config)

    screen_w, screen_h = get_primary_monitor_resolution()
    calib_file = Path("data/calibration/screen_calibration.json")

    mapper: Optional[PolynomialGazeMapper] = None
    use_calibration: bool = False
    if calib_file.exists():
        try:
            mapper = PolynomialGazeMapper.load_from_file(calib_file)
            if mapper.quality and mapper.quality.grade in ("EXCELLENT", "GOOD"):
                use_calibration = True
                print(f"Loaded calibration profile (Grade: {mapper.quality.grade}, RMSE: {mapper.quality.rmse_px:.1f}px)")
            else:
                print(f"Calibration profile grade is {mapper.quality.grade if mapper.quality else 'N/A'}. Using Direct 100% Adaptive Iris Mapping.")
        except Exception as exc:
            logger.warning("Could not load calibration profile (%s); using direct adaptive mapping.", exc)

    if mapper is None:
        mapper = PolynomialGazeMapper(screen_width=screen_w, screen_height=screen_h)

    cam = CameraManager(CameraConfig(width=1280, height=720, mirror=True))
    eye_tracker = EyeTracker(EyeTrackingConfig(refine_landmarks=True))
    gaze_estimator = GazeEstimator(GazeConfig(screen_width=screen_w, screen_height=screen_h))

    window_name = "AdaptiveEye - Confidence-Aware Cursor Stabilization"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\nLive Cursor Stabilization sandbox running (100% Full-Screen Reach).")
    print("Controls:")
    print("  [1] - Switch to Adaptive EMA mode (Ultra-low latency)")
    print("  [2] - Switch to Confidence Kalman mode")
    print("  [3] - Switch to Hybrid mode (Kalman + Deadzone EMA)")
    print("  [+] / [-] - Increase / Decrease sensitivity gain")
    print("  [C] - Toggle Polynomial Calibration vs Direct Adaptive Iris Mapping")
    print("  [A] - Re-anchor posture baseline")
    print("  [R] - Reset iris boundaries & baseline")
    print("  [Q] or [ESC] - Exit\n")

    try:
        cam.start()
        for frame in cam.stream():
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            mesh_res = eye_tracker.mesh.process(rgb)

            all_lms = None
            if mesh_res.multi_face_landmarks:
                all_lms = [
                    Point2D(x=float(lm.x), y=float(lm.y))
                    for lm in mesh_res.multi_face_landmarks[0].landmark
                ]

            eye_feats = eye_tracker.process(frame)
            raw_gaze = gaze_estimator.estimate(frame, eye_feats, all_lms)

            # Fullscreen canvas
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            canvas[:] = (20, 22, 26)

            # Draw center crosshair reference guide
            cx, cy = screen_w // 2, screen_h // 2
            cv2.drawMarker(canvas, (cx, cy), (40, 45, 50), cv2.MARKER_CROSS, 40, 1, cv2.LINE_AA)

            if raw_gaze.is_valid:
                # 1. Gaze coordinate source: Calibrated Polynomial or Direct 100% Adaptive Iris Mapping
                if use_calibration and mapper and mapper.is_fitted:
                    static_x, static_y = mapper.predict(raw_gaze.raw_x, raw_gaze.raw_y)
                else:
                    static_x, static_y = raw_gaze.screen_x, raw_gaze.screen_y

                # 2. Adaptive posture drift compensation
                adapted_x, adapted_y, drift_state = calibrator.compensate(
                    static_x,
                    static_y,
                    raw_gaze.head_angles,
                    eye_feats,
                    w,
                    h,
                    screen_w,
                    screen_h,
                )

                # 3. Confidence-aware cursor stabilization
                cursor_state = stabilizer.stabilize(
                    adapted_x,
                    adapted_y,
                    confidence=raw_gaze.confidence,
                    is_valid=True,
                    monitor_w=screen_w,
                    monitor_h=screen_h,
                )

                stabilizer.draw_telemetry_hud(canvas, cursor_state)
                mode_label = "CALIB" if use_calibration else "DIRECT 100%"
                status_text = f"Stabilized Cursor: ({cursor_state.screen_x}, {cursor_state.screen_y}) | Mode: {mode_label} | Gain: {gaze_estimator.range_tracker.gain:.2f}x"
                status_color = (0, 255, 120) if cursor_state.is_fixating else (0, 220, 255)
            else:
                cursor_state = stabilizer.stabilize(
                    0, 0, confidence=0.0, is_valid=False, monitor_w=screen_w, monitor_h=screen_h
                )
                stabilizer.draw_telemetry_hud(canvas, cursor_state)
                status_text = "CURSOR PAUSED (EYES CLOSED / BLINK)"
                status_color = (0, 0, 255)

            # Header info
            cv2.putText(canvas, status_text, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"Keys: [1] EMA | [2] Kalman | [3] Hybrid | [+] / [-] Gain ({gaze_estimator.range_tracker.gain:.2f}x) | [C] Calib Toggle | [R] Reset | [Q] Exit",
                (40, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == ord("1"):
                stabilizer.config.algorithm = "adaptive_ema"
                print("Switched stabilization algorithm to: ADAPTIVE EMA")
            elif key == ord("2"):
                stabilizer.config.algorithm = "kalman"
                print("Switched stabilization algorithm to: KALMAN")
            elif key == ord("3"):
                stabilizer.config.algorithm = "hybrid"
                print("Switched stabilization algorithm to: HYBRID")
            elif key in (ord("+"), ord("=")):
                gaze_estimator.range_tracker.gain = round(gaze_estimator.range_tracker.gain + 0.15, 2)
                print(f"Increased gaze sensitivity gain to: {gaze_estimator.range_tracker.gain:.2f}x")
            elif key in (ord("-"), ord("_")):
                gaze_estimator.range_tracker.gain = max(0.50, round(gaze_estimator.range_tracker.gain - 0.15, 2))
                print(f"Decreased gaze sensitivity gain to: {gaze_estimator.range_tracker.gain:.2f}x")
            elif key in (ord("c"), ord("C")):
                use_calibration = not use_calibration
                print(f"Switched mapping mode: {'CALIBRATED POLYNOMIAL' if use_calibration else 'DIRECT 100% ADAPTIVE IRIS'}")
            elif key in (ord("r"), ord("R")):
                gaze_estimator.range_tracker.reset()
                calibrator.reset_baseline()
                print("Reset iris boundaries and posture baseline.")
            elif key in (ord("a"), ord("A")):
                if raw_gaze.head_angles and eye_feats.has_eyes:
                    calibrator.set_baseline(raw_gaze.head_angles, eye_feats, w, h)
                    print("Re-anchored posture baseline.")

    finally:
        cam.stop()
        eye_tracker.close()
        cv2.destroyAllWindows()
