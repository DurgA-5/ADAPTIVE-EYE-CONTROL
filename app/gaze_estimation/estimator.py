"""AdaptiveEye Gaze Estimation Engine.

Fuses relative iris center displacements with 3D head pose Euler angles
into normalized screen gaze coordinates (gx, gy) and pixel coordinates.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np
import yaml

from app.camera.webcam import CameraConfig, CameraManager
from app.core.datatypes import GazeCoordinates, Point2D
from app.core.logger import get_logger
from app.eye_tracking.tracker import ExtractedEyeFeatures, EyeTracker, EyeTrackingConfig
from app.gaze_estimation.head_pose import HeadPoseAngles, HeadPoseEstimator

logger = get_logger("GazeEstimator")


@dataclass
class GazeConfig:
    """Strongly-typed gaze estimation parameters."""
    estimation_approach: str = "compound"  # "compound" or "pure_iris"
    iris_weight: float = 0.65
    head_pose_weight: float = 0.35
    sensitivity_x: float = 2.8
    sensitivity_y: float = 3.2
    center_bias_x: float = 0.50
    center_bias_y: float = 0.50
    screen_width: int = 1920
    screen_height: int = 1080

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "GazeConfig":
        if not yaml_path.exists():
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        g_data = data.get("gaze_estimation", {})
        return cls(
            estimation_approach=str(g_data.get("estimation_approach", "compound")),
            iris_weight=float(g_data.get("iris_weight", 0.65)),
            head_pose_weight=float(g_data.get("head_pose_weight", 0.35)),
            sensitivity_x=float(g_data.get("sensitivity_x", 2.8)),
            sensitivity_y=float(g_data.get("sensitivity_y", 3.2)),
            center_bias_x=float(g_data.get("center_bias_x", 0.50)),
            center_bias_y=float(g_data.get("center_bias_y", 0.50)),
        )


@dataclass
class IrisRangeBounds:
    """Dynamic physiological limits of iris displacement."""
    min_x: float = 0.44
    max_x: float = 0.56
    min_y: float = 0.44
    max_y: float = 0.56


class AdaptiveIrisRangeTracker:
    """Dynamically maps physiological iris motion across 100% of the display."""

    def __init__(
        self,
        bounds: Optional[IrisRangeBounds] = None,
        gain_x: float = 1.05,
        gain_y: float = 1.15,
        neutral_x: float = 0.50,
        neutral_y: float = 0.50,
    ) -> None:
        self.bounds = bounds or IrisRangeBounds()
        self.gain_x: float = gain_x
        self.gain_y: float = gain_y
        self._gain: float = gain_x
        self.neutral_iris_x: float = neutral_x
        self.neutral_iris_y: float = neutral_y

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, val: float) -> None:
        self._gain = val
        self.gain_x = val
        self.gain_y = round(val * 1.08, 2)

    def set_neutral(self, iris_x: float, iris_y: float) -> None:
        """Calibrate user's resting forward gaze center."""
        self.neutral_iris_x = float(iris_x)
        self.neutral_iris_y = float(iris_y)

    def reset(self) -> None:
        self.bounds = IrisRangeBounds()
        self.neutral_iris_x = 0.50
        self.neutral_iris_y = 0.50

    def update_and_map(self, iris_x: float, iris_y: float) -> Tuple[float, float]:
        """Map raw relative iris coordinate to full-screen range [0.0, 1.0] with precision acceleration."""
        # Centered displacement from user-calibrated neutral eye socket center
        disp_x = iris_x - self.neutral_iris_x
        disp_y = iris_y - self.neutral_iris_y

        # Physiologically accurate ocular limits for smooth, natural full-screen sweep:
        half_span_x = 0.085
        half_span_y_up = 0.055
        half_span_y_down = 0.045

        # Sub-pixel Pointer Precision with Smooth Corner Reach Extension:
        # Dampens micro-tremors near center while providing smooth progressive edge reach
        def _precision_curve(disp: float, span: float, gain: float) -> float:
            normalized = disp / span
            sign = 1.0 if normalized >= 0.0 else -1.0
            mag = abs(normalized)

            # Precision Center Zone: dampens tremor for rock-solid button targeting
            if mag < 0.40:
                curved = mag ** 1.15
            # Edge Acceleration Zone: smooth progressive boost for effortless corner reach
            else:
                curved = (0.40 ** 1.15) + (mag - 0.40) * 1.25

            return 0.50 + sign * curved * 0.50 * gain

        nx = _precision_curve(disp_x, half_span_x, self.gain_x)
        if disp_y >= 0.0:
            ny = _precision_curve(disp_y, half_span_y_down, self.gain_y)
        else:
            ny = _precision_curve(disp_y, half_span_y_up, self.gain_y)

        return float(np.clip(nx, 0.0, 1.0)), float(np.clip(ny, 0.0, 1.0))


@dataclass
class GazeEstimateResult:
    """Outcome of gaze vector estimation."""
    normalized_x: float
    normalized_y: float
    screen_x: int
    screen_y: int
    confidence: float
    is_valid: bool
    approach: str
    raw_x: float = 0.50
    raw_y: float = 0.50
    head_angles: Optional[HeadPoseAngles] = None
    iris_delta_x: float = 0.0
    iris_delta_y: float = 0.0
    latency_ms: float = 0.0
    status_message: str = "Active"


class GazeEstimator:
    """Estimates screen gaze coordinates by fusing pupil displacement and head pose."""

    def __init__(self, config: Optional[GazeConfig] = None) -> None:
        self.config = config or GazeConfig()
        self.head_pose_estimator = HeadPoseEstimator()
        self.range_tracker = AdaptiveIrisRangeTracker(
            neutral_x=self.config.center_bias_x,
            neutral_y=self.config.center_bias_y,
        )
        self._neutral_pitch: Optional[float] = None
        self._neutral_yaw: Optional[float] = None
        self._smooth_iris_x: Optional[float] = None
        self._smooth_iris_y: Optional[float] = None
        logger.info(
            "GazeEstimator initialized (approach=%s, iris_w=%.2f, head_w=%.2f)",
            self.config.estimation_approach,
            self.config.iris_weight,
            self.config.head_pose_weight,
        )

    def reset_smoothing(self) -> None:
        """Reset temporal iris smoothing."""
        self._smooth_iris_x = None
        self._smooth_iris_y = None

    def set_neutral_posture(
        self,
        yaw: float,
        pitch: float,
        neutral_iris_x: Optional[float] = None,
        neutral_iris_y: Optional[float] = None,
    ) -> None:
        """Lock current head pose and resting iris coordinates as neutral baseline."""
        self._neutral_yaw = yaw
        self._neutral_pitch = pitch
        if neutral_iris_x is not None and neutral_iris_y is not None:
            self.range_tracker.set_neutral(neutral_iris_x, neutral_iris_y)
            self._smooth_iris_x = neutral_iris_x
            self._smooth_iris_y = neutral_iris_y
        logger.info("GazeEstimator baseline locked: Yaw=%.1f deg, Pitch=%.1f deg", yaw, pitch)

    def reset_neutral_posture(self) -> None:
        """Reset posture baseline."""
        self._neutral_yaw = None
        self._neutral_pitch = None
        self.range_tracker.reset()

    def estimate(
        self,
        bgr_frame: np.ndarray,
        eye_features: ExtractedEyeFeatures,
        all_landmarks: Optional[List[Point2D]] = None,
    ) -> GazeEstimateResult:
        """Estimate gaze vector on screen coordinates.
        
        Args:
            bgr_frame: Camera image.
            eye_features: Extracted eye contours, irises, and EAR.
            all_landmarks: Optional full 478 Face Mesh landmark list for head pose.
            
        Returns:
            GazeEstimateResult with normalized and screen coordinates.
        """
        start_time = time.perf_counter()
        h, w = bgr_frame.shape[:2]

        # Safety Check: If eyes are lost or closed (blinking), invalidate gaze immediately
        if (
            not eye_features.has_eyes
            or eye_features.both_eyes_closed
            or not eye_features.right_eye
            or not eye_features.left_eye
        ):
            return GazeEstimateResult(
                normalized_x=0.50,
                normalized_y=0.50,
                screen_x=self.config.screen_width // 2,
                screen_y=self.config.screen_height // 2,
                confidence=0.0,
                is_valid=False,
                approach=self.config.estimation_approach,
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                status_message="EYES CLOSED OR OCCLUDED - GAZE PAUSED",
            )

        r_eye = eye_features.right_eye
        l_eye = eye_features.left_eye

        # Average relative iris positions across both eyes
        avg_iris_x = (r_eye.iris_relative_x + l_eye.iris_relative_x) / 2.0
        avg_iris_y = (r_eye.iris_relative_y + l_eye.iris_relative_y) / 2.0

        # Fast, responsive temporal tracking (eliminates sluggish dragging lag)
        if self._smooth_iris_x is None or self._smooth_iris_y is None:
            self._smooth_iris_x = avg_iris_x
            self._smooth_iris_y = avg_iris_y
        else:
            iris_alpha = 0.82  # Snappy, low-latency ocular tracking
            self._smooth_iris_x = (1.0 - iris_alpha) * self._smooth_iris_x + iris_alpha * avg_iris_x
            self._smooth_iris_y = (1.0 - iris_alpha) * self._smooth_iris_y + iris_alpha * avg_iris_y

        # Full-range adaptive iris mapping across [0.0, 1.0]
        iris_norm_x, iris_norm_y = self.range_tracker.update_and_map(self._smooth_iris_x, self._smooth_iris_y)

        # Centered displacement from full-range mapping
        iris_delta_x = (iris_norm_x - 0.50) * 2.0
        iris_delta_y = (iris_norm_y - 0.50) * 2.0

        head_angles: Optional[HeadPoseAngles] = None
        head_delta_x = 0.0
        head_delta_y = 0.0

        # Estimate head pose using solvePnP for telemetry & posture drift tracking
        if all_landmarks:
            head_angles = self.head_pose_estimator.estimate(all_landmarks, w, h)
            if head_angles:
                # Auto-initialize neutral posture on first head pose detection
                if self._neutral_pitch is None:
                    self._neutral_pitch = head_angles.pitch
                    self._neutral_yaw = head_angles.yaw
                
                rel_yaw = head_angles.yaw - self._neutral_yaw
                rel_pitch = head_angles.pitch - self._neutral_pitch
                # Normalized head motion (+/- 15 degrees nominal span)
                head_delta_x = (rel_yaw / 15.0)
                head_delta_y = (rel_pitch / 15.0)

        # In pure_iris mode, iris displacement provides 100% full-screen targeting.
        # In compound mode, iris provides primary targeting with subtle head pose assistance.
        if self.config.estimation_approach == "compound" and head_angles:
            raw_x = iris_norm_x + (self.config.head_pose_weight * head_delta_x)
            raw_y = iris_norm_y + (self.config.head_pose_weight * head_delta_y)
        else:
            raw_x = iris_norm_x
            raw_y = iris_norm_y

        # Clamp strictly to screen bounds [0.0, 1.0] for normalized display
        norm_x = float(np.clip(raw_x, 0.0, 1.0))
        norm_y = float(np.clip(raw_y, 0.0, 1.0))

        # Project to target screen resolution
        screen_x = int(round(norm_x * (self.config.screen_width - 1)))
        screen_y = int(round(norm_y * (self.config.screen_height - 1)))

        confidence = 0.95 if (r_eye.iris_center and l_eye.iris_center) else 0.70
        latency = (time.perf_counter() - start_time) * 1000.0

        return GazeEstimateResult(
            normalized_x=round(norm_x, 4),
            normalized_y=round(norm_y, 4),
            screen_x=screen_x,
            screen_y=screen_y,
            confidence=confidence,
            is_valid=True,
            approach=self.config.estimation_approach,
            raw_x=round(raw_x, 4),
            raw_y=round(raw_y, 4),
            head_angles=head_angles,
            iris_delta_x=round(iris_delta_x, 4),
            iris_delta_y=round(iris_delta_y, 4),
            latency_ms=latency,
            status_message="Tracking Gaze Active",
        )

    def draw_gaze_overlay(
        self,
        bgr_image: np.ndarray,
        result: GazeEstimateResult,
        eye_features: ExtractedEyeFeatures,
        all_landmarks: Optional[List[Point2D]] = None,
        draw_banner: bool = True,
    ) -> np.ndarray:
        """Render gaze vector rays, target crosshairs, and head pose orientation axes."""
        canvas = bgr_image.copy()
        h, w = canvas.shape[:2]

        if draw_banner:
            # Top banner
            status_color = (0, 255, 0) if result.is_valid else (0, 0, 255)
            cv2.rectangle(canvas, (0, 0), (w, 42), (20, 20, 20), -1)

            yaw_str = f"Yaw: {result.head_angles.yaw:+.1f} deg" if result.head_angles else "Head: N/A"
            pitch_str = f"Pitch: {result.head_angles.pitch:+.1f} deg" if result.head_angles else ""

            banner_text = (
                f"Gaze [{result.approach.upper()}]: ({result.normalized_x:.2f}, {result.normalized_y:.2f}) -> "
                f"Screen: ({result.screen_x}, {result.screen_y}) | {yaw_str} {pitch_str} | "
                f"Conf: {result.confidence:.2f}"
            )
            cv2.putText(canvas, banner_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 2, cv2.LINE_AA)

            if not result.is_valid:
                # Draw bold pause warning
                cv2.rectangle(canvas, (0, h - 36), (w, h), (0, 0, 180), -1)
                cv2.putText(
                    canvas,
                    result.status_message,
                    (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                return canvas
        elif not result.is_valid:
            return canvas

        # Draw 3D Head Pose Bounding Box, Facial Mesh Points & Orientation Axes
        if result.head_angles and all_landmarks:
            self.head_pose_estimator.draw_3d_head_box(canvas, result.head_angles, all_landmarks, w, h)
            self.head_pose_estimator.draw_axes(canvas, result.head_angles, all_landmarks, w, h, length=45.0)

        # Draw Gaze Vector Rays originating from pupils
        for eye in (eye_features.right_eye, eye_features.left_eye):
            if eye and eye.iris_center:
                origin = eye.iris_center.to_pixel(w, h)
                # Compute ray endpoint proportional to gaze direction
                ray_dx = int((result.normalized_x - 0.50) * 120.0)
                ray_dy = int((result.normalized_y - 0.50) * 120.0)
                target = (origin[0] + ray_dx, origin[1] + ray_dy)
                cv2.arrowedLine(canvas, origin, target, (0, 255, 255), 2, tipLength=0.25)

        # Draw mini screen preview target crosshair in bottom right corner
        mini_w, mini_h = 160, 90
        x0, y0 = w - mini_w - 15, h - mini_h - 15
        cv2.rectangle(canvas, (x0, y0), (x0 + mini_w, y0 + mini_h), (40, 40, 40), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + mini_w, y0 + mini_h), (0, 255, 200), 1)

        # Target point on mini screen
        target_x = int(x0 + result.normalized_x * mini_w)
        target_y = int(y0 + result.normalized_y * mini_h)
        cv2.circle(canvas, (target_x, target_y), 4, (0, 0, 255), -1)
        cv2.drawMarker(canvas, (target_x, target_y), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)

        return canvas


def run_gaze_estimation_live() -> None:
    """Launch interactive live gaze vector estimation display."""
    cam = CameraManager(CameraConfig(width=1280, height=720, mirror=True))
    eye_tracker = EyeTracker(EyeTrackingConfig(refine_landmarks=True))
    gaze_estimator = GazeEstimator()

    window_name = "AdaptiveEye - Real-Time Gaze Estimation"
    print("\nLive Gaze Estimation running. Look at different parts of the screen. Press [Q] to exit.")

    try:
        cam.start()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        for frame in cam.stream():
            # Extract mesh and eye features
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            mesh_results = eye_tracker.mesh.process(rgb_frame)

            all_lms = None
            if mesh_results.multi_face_landmarks:
                raw_lms = mesh_results.multi_face_landmarks[0].landmark
                all_lms = [Point2D(x=float(lm.x), y=float(lm.y)) for lm in raw_lms]

            eye_features = eye_tracker.process(frame)
            gaze_res = gaze_estimator.estimate(frame, eye_features, all_lms)

            annotated = eye_tracker.draw_eye_annotations(frame, eye_features)
            annotated = gaze_estimator.draw_gaze_overlay(annotated, gaze_res, eye_features, all_lms)

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    finally:
        cam.stop()
        eye_tracker.close()
        cv2.destroyAllWindows()
