"""AdaptiveEye Dynamic Adaptive Calibration & Posture Drift Compensator.

Monitors 3D head pose displacement, user-to-screen distance variations, and natural posture shifts
to apply real-time geometric correction deltas, preventing gaze targeting drift as the user slouches,
shifts, or leans over extended interaction sessions.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Optional, Tuple
import cv2
import numpy as np
import yaml

from app.core.logger import get_logger
from app.core.datatypes import Point2D
from app.camera.webcam import CameraConfig, CameraManager
from app.eye_tracking.tracker import ExtractedEyeFeatures, EyeTracker, EyeTrackingConfig
from app.gaze_estimation.estimator import GazeConfig, GazeEstimator
from app.gaze_estimation.head_pose import HeadPoseAngles
from app.calibration.mapper import PolynomialGazeMapper
from app.calibration.collector import get_primary_monitor_resolution

logger = get_logger("AdaptiveCalibrator")


@dataclass
class HeadPostureAnchor:
    """Baseline physical posture recorded at calibration time."""
    baseline_yaw: float = 0.0
    baseline_pitch: float = 0.0
    baseline_roll: float = 0.0
    baseline_eye_distance_px: float = 0.0
    is_anchored: bool = False
    anchored_at: str = ""


@dataclass
class DriftState:
    """Real-time posture drift measurements and applied correction."""
    delta_x_px: float = 0.0
    delta_y_px: float = 0.0
    drift_magnitude_px: float = 0.0
    yaw_shift_deg: float = 0.0
    pitch_shift_deg: float = 0.0
    distance_scale: float = 1.0
    is_posture_alert: bool = False
    status_message: str = "Nominal Posture"


@dataclass
class AdaptiveConfig:
    """Parameters governing dynamic posture adaptation."""
    enable_drift_compensation: bool = True
    head_drift_gain_x: float = 10.0
    head_drift_gain_y: float = 8.5
    enable_distance_scaling: bool = True
    max_drift_offset_px: float = 160.0
    drift_ema_smoothing: float = 0.15
    baseline_auto_anchor: bool = True
    posture_alert_threshold_deg: float = 25.0

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "AdaptiveConfig":
        if not yaml_path.exists():
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        a_data = data.get("adaptive_calibration", {})
        return cls(
            enable_drift_compensation=bool(a_data.get("enable_drift_compensation", True)),
            head_drift_gain_x=float(a_data.get("head_drift_gain_x", 10.0)),
            head_drift_gain_y=float(a_data.get("head_drift_gain_y", 8.5)),
            enable_distance_scaling=bool(a_data.get("enable_distance_scaling", True)),
            max_drift_offset_px=float(a_data.get("max_drift_offset_px", 160.0)),
            drift_ema_smoothing=float(a_data.get("drift_ema_smoothing", 0.15)),
            baseline_auto_anchor=bool(a_data.get("baseline_auto_anchor", True)),
            posture_alert_threshold_deg=float(a_data.get("posture_alert_threshold_deg", 25.0)),
        )


class AdaptiveCalibrator:
    """Online dynamic drift detection and geometric compensation engine."""

    def __init__(self, config: Optional[AdaptiveConfig] = None) -> None:
        self.config = config or AdaptiveConfig()
        self.anchor = HeadPostureAnchor()

        # Internal state for smooth low-pass filtered offsets
        self._current_delta_x: float = 0.0
        self._current_delta_y: float = 0.0
        self._last_state: DriftState = DriftState()

        logger.info(
            "AdaptiveCalibrator initialized (gain_x=%.1f, gain_y=%.1f, max_drift=%.1f px)",
            self.config.head_drift_gain_x,
            self.config.head_drift_gain_y,
            self.config.max_drift_offset_px,
        )

    def set_baseline(
        self,
        head_angles: HeadPoseAngles,
        eye_features: ExtractedEyeFeatures,
        img_w: int,
        img_h: int,
    ) -> None:
        """Establish baseline physiological posture anchor."""
        eye_dist = self.compute_eye_distance(eye_features, img_w, img_h)
        self.anchor = HeadPostureAnchor(
            baseline_yaw=head_angles.yaw,
            baseline_pitch=head_angles.pitch,
            baseline_roll=head_angles.roll,
            baseline_eye_distance_px=eye_dist,
            is_anchored=True,
            anchored_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._current_delta_x = 0.0
        self._current_delta_y = 0.0
        logger.info(
            "Posture anchor locked: Yaw=%.1f deg, Pitch=%.1f deg, Interpupillary=%.1f px",
            head_angles.yaw,
            head_angles.pitch,
            eye_dist,
        )

    def reset_baseline(self) -> None:
        """Clear posture anchor to re-anchor on next stable frame."""
        self.anchor = HeadPostureAnchor()
        self._current_delta_x = 0.0
        self._current_delta_y = 0.0
        self._last_state = DriftState()
        logger.info("Posture anchor reset.")

    @staticmethod
    def compute_eye_distance(
        eye_features: ExtractedEyeFeatures,
        img_w: int,
        img_h: int,
    ) -> float:
        """Calculate inter-pupillary Euclidean distance in camera pixel space."""
        if (
            not eye_features.has_eyes
            or not eye_features.right_eye
            or not eye_features.left_eye
            or not eye_features.right_eye.iris_center
            or not eye_features.left_eye.iris_center
        ):
            return 120.0  # Nominal baseline fallback

        r_pt = eye_features.right_eye.iris_center.to_pixel(img_w, img_h)
        l_pt = eye_features.left_eye.iris_center.to_pixel(img_w, img_h)
        dist = float(np.hypot(r_pt[0] - l_pt[0], r_pt[1] - l_pt[1]))
        return max(20.0, dist)

    def compensate(
        self,
        screen_x: int,
        screen_y: int,
        head_angles: Optional[HeadPoseAngles],
        eye_features: ExtractedEyeFeatures,
        img_w: int,
        img_h: int,
        monitor_w: int = 1920,
        monitor_h: int = 1080,
    ) -> Tuple[int, int, DriftState]:
        """Apply dynamic posture and distance drift compensation to screen coordinates.
        
        Args:
            screen_x: Static calibrated screen X coordinate.
            screen_y: Static calibrated screen Y coordinate.
            head_angles: Current 3D head pose angles.
            eye_features: Eye landmarks and iris features.
            img_w: Camera frame width.
            img_h: Camera frame height.
            monitor_w: Monitor width in pixels.
            monitor_h: Monitor height in pixels.
            
        Returns:
            Tuple of (adapted_screen_x, adapted_screen_y, DriftState).
        """
        # If disabled or no head angles available, pass through unchanged
        if not self.config.enable_drift_compensation or head_angles is None:
            state = DriftState(status_message="Compensation Inactive")
            return screen_x, screen_y, state

        # Auto-anchor on first valid frame if not yet anchored
        if not self.anchor.is_anchored:
            if self.config.baseline_auto_anchor and eye_features.has_eyes:
                self.set_baseline(head_angles, eye_features, img_w, img_h)
            else:
                state = DriftState(status_message="Waiting for Anchor")
                return screen_x, screen_y, state

        # 1. Posture Shifts
        yaw_shift = float(head_angles.yaw - self.anchor.baseline_yaw)
        pitch_shift = float(head_angles.pitch - self.anchor.baseline_pitch)

        # Check posture warning limit
        posture_alert = (
            abs(yaw_shift) > self.config.posture_alert_threshold_deg
            or abs(pitch_shift) > self.config.posture_alert_threshold_deg
        )

        # 2. Raw Proportional Drift Corrections
        target_dx = self.config.head_drift_gain_x * yaw_shift
        target_dy = self.config.head_drift_gain_y * pitch_shift

        # 3. Low-Pass Exponential Moving Average (EMA) Smoothing
        alpha = self.config.drift_ema_smoothing
        self._current_delta_x = (1.0 - alpha) * self._current_delta_x + alpha * target_dx
        self._current_delta_y = (1.0 - alpha) * self._current_delta_y + alpha * target_dy

        # 4. Strict Safety Clamping
        drift_mag = float(np.hypot(self._current_delta_x, self._current_delta_y))
        if drift_mag > self.config.max_drift_offset_px and drift_mag > 1e-6:
            scale = self.config.max_drift_offset_px / drift_mag
            self._current_delta_x *= scale
            self._current_delta_y *= scale
            drift_mag = self.config.max_drift_offset_px

        # 5. Distance Scaling (lean-in / lean-out compensation)
        distance_scale = 1.0
        if self.config.enable_distance_scaling and self.anchor.baseline_eye_distance_px > 1e-3:
            curr_dist = self.compute_eye_distance(eye_features, img_w, img_h)
            # Closer to screen -> larger eye distance -> wider gaze angle needed
            ratio = curr_dist / self.anchor.baseline_eye_distance_px
            distance_scale = float(np.clip(ratio, 0.75, 1.35))

        # 6. Apply Adaptation Around Screen Center
        center_x = monitor_w / 2.0
        center_y = monitor_h / 2.0

        # Scale relative to screen center, then add posture drift offset
        scaled_x = center_x + (screen_x - center_x) * distance_scale
        scaled_y = center_y + (screen_y - center_y) * distance_scale

        final_x = int(np.clip(round(scaled_x + self._current_delta_x), 0, monitor_w - 1))
        final_y = int(np.clip(round(scaled_y + self._current_delta_y), 0, monitor_h - 1))

        status_msg = "Excessive Posture Drift" if posture_alert else "Drift Compensation Active"

        state = DriftState(
            delta_x_px=round(self._current_delta_x, 1),
            delta_y_px=round(self._current_delta_y, 1),
            drift_magnitude_px=round(drift_mag, 1),
            yaw_shift_deg=round(yaw_shift, 1),
            pitch_shift_deg=round(pitch_shift, 1),
            distance_scale=round(distance_scale, 2),
            is_posture_alert=posture_alert,
            status_message=status_msg,
        )
        self._last_state = state
        return final_x, final_y, state

    def draw_telemetry_hud(
        self,
        canvas: np.ndarray,
        drift_state: DriftState,
        raw_screen_pt: Tuple[int, int],
        adapted_screen_pt: Tuple[int, int],
    ) -> np.ndarray:
        """Render drift offset telemetry vectors and posture status indicator."""
        h, w = canvas.shape[:2]

        # Top banner for drift info
        alert_col = (0, 0, 255) if drift_state.is_posture_alert else (0, 255, 200)

        # Draw drift vector from static point to adapted point
        if abs(drift_state.drift_magnitude_px) > 2.0:
            cv2.arrowedLine(canvas, raw_screen_pt, adapted_screen_pt, (0, 165, 255), 2, tipLength=0.25)
            cv2.circle(canvas, raw_screen_pt, 4, (120, 120, 120), -1, cv2.LINE_AA)

        # Telemetry panel in top-right corner
        card_w, card_h = 280, 100
        x0, y0 = w - card_w - 20, 20
        cv2.rectangle(canvas, (x0, y0), (x0 + card_w, y0 + card_h), (25, 28, 32), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + card_w, y0 + card_h), alert_col, 1)

        lines = [
            f"Adaptive Drift: {drift_state.drift_magnitude_px:.1f} px",
            f"Offset: (dX: {drift_state.delta_x_px:+.1f}, dY: {drift_state.delta_y_px:+.1f})",
            f"Head: Yaw {drift_state.yaw_shift_deg:+.1f} | Pitch {drift_state.pitch_shift_deg:+.1f}",
            f"Dist Scale: {drift_state.distance_scale:.2f}x | {drift_state.status_message}",
        ]

        for i, text in enumerate(lines):
            col = alert_col if i == 0 and drift_state.is_posture_alert else (220, 220, 220)
            cv2.putText(
                canvas,
                text,
                (x0 + 10, y0 + 20 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                col,
                1,
                cv2.LINE_AA,
            )

        return canvas


def run_adaptive_gaze_live() -> None:
    """Launch real-time calibrated gaze tracking with dynamic posture drift compensation."""
    cfg_file = Path("config/adaptive_calibration_config.yaml")
    config = AdaptiveConfig.from_yaml(cfg_file)
    calibrator = AdaptiveCalibrator(config)

    screen_w, screen_h = get_primary_monitor_resolution()
    calib_file = Path("data/calibration/screen_calibration.json")

    mapper: Optional[PolynomialGazeMapper] = None
    if calib_file.exists():
        try:
            mapper = PolynomialGazeMapper.load_from_file(calib_file)
            print(f"Loaded screen calibration: Grade={mapper.quality.grade if mapper.quality else 'N/A'}")
        except Exception as exc:
            logger.warning("Could not load calibration profile (%s); using fallback mapping.", exc)

    if mapper is None:
        mapper = PolynomialGazeMapper(screen_width=screen_w, screen_height=screen_h)

    cam = CameraManager(CameraConfig(width=1280, height=720, mirror=True))
    eye_tracker = EyeTracker(EyeTrackingConfig(refine_landmarks=True))
    gaze_estimator = GazeEstimator(GazeConfig(screen_width=screen_w, screen_height=screen_h))

    window_name = "AdaptiveEye - Adaptive Calibration & Drift Compensation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\nLive Adaptive Gaze Tracking running.")
    print("Controls:")
    print("  [A] - Re-anchor baseline head posture to your current position")
    print("  [R] - Reset drift compensation offsets")
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
                # 1. Static calibrated prediction
                static_x, static_y = mapper.predict(raw_gaze.raw_x, raw_gaze.raw_y)

                # 2. Dynamic posture drift compensation
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

                # Draw static uncompensated target in subdued grey
                cv2.circle(canvas, (static_x, static_y), 8, (100, 100, 100), 1, cv2.LINE_AA)

                # Draw dynamically adapted cursor target in bright cyan/emerald
                cv2.circle(canvas, (adapted_x, adapted_y), 24, (0, 255, 200), 2, cv2.LINE_AA)
                cv2.circle(canvas, (adapted_x, adapted_y), 6, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.drawMarker(canvas, (adapted_x, adapted_y), (0, 255, 200), cv2.MARKER_CROSS, 32, 1, cv2.LINE_AA)

                # Telemetry HUD
                calibrator.draw_telemetry_hud(canvas, drift_state, (static_x, static_y), (adapted_x, adapted_y))

                status_text = f"Adapted Screen Target: ({adapted_x}, {adapted_y}) | Static: ({static_x}, {static_y})"
                status_color = (0, 255, 200)
            else:
                status_text = "GAZE PAUSED (EYES CLOSED OR BLINKING)"
                status_color = (0, 0, 255)

            # Header info
            cv2.putText(canvas, status_text, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)
            anchor_info = f"Anchor: Yaw {calibrator.anchor.baseline_yaw:.1f} Pitch {calibrator.anchor.baseline_pitch:.1f} | [A] Re-Anchor | [R] Reset Drift | [Q] Exit"
            cv2.putText(canvas, anchor_info, (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("a"), ord("A")):
                if raw_gaze.head_angles and eye_feats.has_eyes:
                    calibrator.set_baseline(raw_gaze.head_angles, eye_feats, w, h)
            elif key in (ord("r"), ord("R")):
                calibrator.reset_baseline()

    finally:
        cam.stop()
        eye_tracker.close()
        cv2.destroyAllWindows()
