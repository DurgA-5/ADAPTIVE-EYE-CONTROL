"""AdaptiveEye Eye Landmark Detection & Iris Localization Engine.

Extracts 478 3D landmarks from MediaPipe Face Mesh, isolates left/right eye perimeters,
computes Eye Aspect Ratio (EAR) for blink detection, and tracks sub-pixel iris centers.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
import yaml

from app.camera.webcam import CameraConfig, CameraManager
from app.core.datatypes import BoundingBox, Point2D
from app.core.logger import get_logger

logger = get_logger("EyeTracker")

# MediaPipe 478 Landmark Indices for Eyes & Irises
# Subject Right Eye (Viewer Left in mirrored view)
RIGHT_EYE_INNER = 133
RIGHT_EYE_OUTER = 33
RIGHT_EYE_UPPER_1 = 160
RIGHT_EYE_LOWER_1 = 144
RIGHT_EYE_UPPER_2 = 158
RIGHT_EYE_LOWER_2 = 153
RIGHT_EYE_CONTOUR = [33, 160, 158, 133, 153, 144]
RIGHT_IRIS_CENTER = 468
RIGHT_IRIS_POINTS = [468, 469, 470, 471, 472]

# Subject Left Eye (Viewer Right in mirrored view)
LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263
LEFT_EYE_UPPER_1 = 385
LEFT_EYE_LOWER_1 = 380
LEFT_EYE_UPPER_2 = 387
LEFT_EYE_LOWER_2 = 373
LEFT_EYE_CONTOUR = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_CENTER = 473
LEFT_IRIS_POINTS = [473, 474, 475, 476, 477]


@dataclass
class SingleEyeFeatures:
    """Detailed geometric measurements for a single eye."""
    inner_corner: Point2D
    outer_corner: Point2D
    upper_lid_1: Point2D
    lower_lid_1: Point2D
    upper_lid_2: Point2D
    lower_lid_2: Point2D
    contour_points: List[Point2D]
    iris_center: Optional[Point2D]
    iris_points: List[Point2D]
    ear: float                           # Eye Aspect Ratio (openness)
    is_closed: bool                      # EAR < threshold
    iris_relative_x: float = 0.5         # Horizontal displacement [0.0 = inner/outer, 1.0 = opposite]
    iris_relative_y: float = 0.5         # Vertical displacement [0.0 = top lid, 1.0 = bottom lid]


@dataclass
class ExtractedEyeFeatures:
    """Aggregate eye and iris tracking output for both eyes."""
    has_eyes: bool
    right_eye: Optional[SingleEyeFeatures] = None
    left_eye: Optional[SingleEyeFeatures] = None
    average_ear: float = 0.0
    both_eyes_closed: bool = False
    tracking_confidence: float = 0.0
    latency_ms: float = 0.0


@dataclass
class EyeTrackingConfig:
    """Configuration for Face Mesh and Iris localization."""
    refine_landmarks: bool = True
    max_num_faces: int = 1
    min_detection_confidence: float = 0.65
    min_tracking_confidence: float = 0.65
    ear_blink_threshold: float = 0.20

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "EyeTrackingConfig":
        if not yaml_path.exists():
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        et_data = data.get("eye_tracking", {})
        return cls(
            refine_landmarks=bool(et_data.get("refine_landmarks", True)),
            max_num_faces=int(et_data.get("max_num_faces", 1)),
            min_detection_confidence=float(et_data.get("min_detection_confidence", 0.65)),
            min_tracking_confidence=float(et_data.get("min_tracking_confidence", 0.65)),
            ear_blink_threshold=float(et_data.get("ear_blink_threshold", 0.20)),
        )


class EyeTracker:
    """Extracts sub-pixel eye landmarks and iris centers using MediaPipe Face Mesh."""

    def __init__(self, config: Optional[EyeTrackingConfig] = None) -> None:
        self.config = config or EyeTrackingConfig()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=self.config.max_num_faces,
            refine_landmarks=self.config.refine_landmarks,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        logger.info(
            "EyeTracker initialized (refine_landmarks=%s, ear_threshold=%.2f)",
            self.config.refine_landmarks,
            self.config.ear_blink_threshold,
        )

    @staticmethod
    def calculate_ear(
        p_outer: Point2D,
        p_inner: Point2D,
        p_up1: Point2D,
        p_low1: Point2D,
        p_up2: Point2D,
        p_low2: Point2D,
        img_w: Optional[int] = None,
        img_h: Optional[int] = None,
    ) -> float:
        """Calculate Eye Aspect Ratio (EAR) using Soukupova & Cech formulation.
        
        When img_w and img_h are provided, converts normalized coordinates to
        isotropic pixel space, eliminating widescreen (16:9) aspect-ratio distortion.
        """
        scale_x = float(img_w) if img_w is not None else 1.0
        scale_y = float(img_h) if img_h is not None else 1.0

        v1 = np.hypot((p_up1.x - p_low1.x) * scale_x, (p_up1.y - p_low1.y) * scale_y)
        v2 = np.hypot((p_up2.x - p_low2.x) * scale_x, (p_up2.y - p_low2.y) * scale_y)
        horiz = np.hypot((p_outer.x - p_inner.x) * scale_x, (p_outer.y - p_inner.y) * scale_y)
        if horiz <= 1e-6:
            return 0.0
        return float((v1 + v2) / (2.0 * horiz))

    def process(self, bgr_image: np.ndarray) -> ExtractedEyeFeatures:
        """Extract eye contours, irises, and EAR metrics from a BGR frame."""
        if bgr_image is None or bgr_image.size == 0:
            return ExtractedEyeFeatures(has_eyes=False)

        h, w = bgr_image.shape[:2]
        start_time = time.perf_counter()

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = self.mesh.process(rgb_image)

        latency = (time.perf_counter() - start_time) * 1000.0

        if not results.multi_face_landmarks:
            return ExtractedEyeFeatures(has_eyes=False, latency_ms=latency)

        # Single authorized locked face (face 0)
        landmarks = results.multi_face_landmarks[0].landmark
        num_landmarks = len(landmarks)

        # Check if iris points (468-477) are present
        has_iris = (num_landmarks >= 478) and self.config.refine_landmarks

        # Helper to convert normalized landmark to Point2D
        def get_pt(idx: int) -> Point2D:
            lm = landmarks[idx]
            return Point2D(x=float(lm.x), y=float(lm.y))

        # --- Right Eye Processing ---
        r_outer = get_pt(RIGHT_EYE_OUTER)
        r_inner = get_pt(RIGHT_EYE_INNER)
        r_up1 = get_pt(RIGHT_EYE_UPPER_1)
        r_low1 = get_pt(RIGHT_EYE_LOWER_1)
        r_up2 = get_pt(RIGHT_EYE_UPPER_2)
        r_low2 = get_pt(RIGHT_EYE_LOWER_2)

        r_ear = self.calculate_ear(r_outer, r_inner, r_up1, r_low1, r_up2, r_low2, img_w=w, img_h=h)
        r_contour = [get_pt(i) for i in RIGHT_EYE_CONTOUR]

        r_iris_center: Optional[Point2D] = None
        r_iris_pts: List[Point2D] = []
        r_rel_x = 0.5
        r_rel_y = 0.5

        if has_iris:
            r_iris_center = get_pt(RIGHT_IRIS_CENTER)
            r_iris_pts = [get_pt(i) for i in RIGHT_IRIS_POINTS]

            # Rigid Skeletal Canthus Normalization:
            # Uses rigid eye corners (inner and outer canthi) as fixed skeletal reference points.
            # Normalizing vertical displacement by corner width prevents downward gaze collapse
            # caused by physiological eyelid depression when looking downward.
            corner_w_px = float(np.hypot((r_outer.x - r_inner.x) * w, (r_outer.y - r_inner.y) * h))
            corner_w_px = max(10.0, corner_w_px)

            mid_x = (r_outer.x + r_inner.x) / 2.0
            mid_y = (r_outer.y + r_inner.y) / 2.0

            delta_x = ((r_iris_center.x - mid_x) * w) / corner_w_px
            delta_y = ((r_iris_center.y - mid_y) * h) / corner_w_px

            r_rel_x = float(np.clip(0.50 + delta_x, 0.0, 1.0))
            r_rel_y = float(np.clip(0.50 + delta_y, 0.0, 1.0))

        right_eye_feat = SingleEyeFeatures(
            inner_corner=r_inner,
            outer_corner=r_outer,
            upper_lid_1=r_up1,
            lower_lid_1=r_low1,
            upper_lid_2=r_up2,
            lower_lid_2=r_low2,
            contour_points=r_contour,
            iris_center=r_iris_center,
            iris_points=r_iris_pts,
            ear=round(r_ear, 4),
            is_closed=(r_ear < self.config.ear_blink_threshold),
            iris_relative_x=round(r_rel_x, 4),
            iris_relative_y=round(r_rel_y, 4),
        )

        # --- Left Eye Processing ---
        l_outer = get_pt(LEFT_EYE_OUTER)
        l_inner = get_pt(LEFT_EYE_INNER)
        l_up1 = get_pt(LEFT_EYE_UPPER_1)
        l_low1 = get_pt(LEFT_EYE_LOWER_1)
        l_up2 = get_pt(LEFT_EYE_UPPER_2)
        l_low2 = get_pt(LEFT_EYE_LOWER_2)

        l_ear = self.calculate_ear(l_outer, l_inner, l_up1, l_low1, l_up2, l_low2, img_w=w, img_h=h)
        l_contour = [get_pt(i) for i in LEFT_EYE_CONTOUR]

        l_iris_center: Optional[Point2D] = None
        l_iris_pts: List[Point2D] = []
        l_rel_x = 0.5
        l_rel_y = 0.5

        if has_iris:
            l_iris_center = get_pt(LEFT_IRIS_CENTER)
            l_iris_pts = [get_pt(i) for i in LEFT_IRIS_POINTS]

            corner_w_px = float(np.hypot((l_outer.x - l_inner.x) * w, (l_outer.y - l_inner.y) * h))
            corner_w_px = max(10.0, corner_w_px)

            mid_x = (l_outer.x + l_inner.x) / 2.0
            mid_y = (l_outer.y + l_inner.y) / 2.0

            delta_x = ((l_iris_center.x - mid_x) * w) / corner_w_px
            delta_y = ((l_iris_center.y - mid_y) * h) / corner_w_px

            l_rel_x = float(np.clip(0.50 + delta_x, 0.0, 1.0))
            l_rel_y = float(np.clip(0.50 + delta_y, 0.0, 1.0))

        left_eye_feat = SingleEyeFeatures(
            inner_corner=l_inner,
            outer_corner=l_outer,
            upper_lid_1=l_up1,
            lower_lid_1=l_low1,
            upper_lid_2=l_up2,
            lower_lid_2=l_low2,
            contour_points=l_contour,
            iris_center=l_iris_center,
            iris_points=l_iris_pts,
            ear=round(l_ear, 4),
            is_closed=(l_ear < self.config.ear_blink_threshold),
            iris_relative_x=round(l_rel_x, 4),
            iris_relative_y=round(l_rel_y, 4),
        )

        avg_ear = round((r_ear + l_ear) / 2.0, 4)
        both_closed = (avg_ear < self.config.ear_blink_threshold) or (right_eye_feat.is_closed and left_eye_feat.is_closed)

        return ExtractedEyeFeatures(
            has_eyes=True,
            right_eye=right_eye_feat,
            left_eye=left_eye_feat,
            average_ear=avg_ear,
            both_eyes_closed=both_closed,
            tracking_confidence=1.0 if has_iris else 0.8,
            latency_ms=latency,
        )

    def draw_eye_annotations(
        self,
        bgr_image: np.ndarray,
        features: ExtractedEyeFeatures,
    ) -> np.ndarray:
        """Render eye contours, iris circles, EAR telemetry, and eye zoom insets."""
        canvas = bgr_image.copy()
        h, w = canvas.shape[:2]

        if not features.has_eyes or not features.right_eye or not features.left_eye:
            cv2.rectangle(canvas, (0, 0), (w, 40), (20, 20, 20), -1)
            cv2.putText(
                canvas,
                "Eye Tracking: No eyes localized",
                (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return canvas

        # Top telemetry banner
        blink_status = "BLINK DETECTED" if features.both_eyes_closed else "EYES OPEN"
        status_color = (0, 0, 255) if features.both_eyes_closed else (0, 255, 0)

        cv2.rectangle(canvas, (0, 0), (w, 42), (20, 20, 20), -1)
        banner_text = (
            f"Eye Tracking Active | Latency: {features.latency_ms:.1f}ms | "
            f"Avg EAR: {features.average_ear:.3f} (R: {features.right_eye.ear:.3f}, L: {features.left_eye.ear:.3f}) | "
            f"Status: {blink_status}"
        )
        cv2.putText(canvas, banner_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2, cv2.LINE_AA)

        # Draw Right Eye (Viewer Left)
        self._draw_single_eye(canvas, features.right_eye, w, h, "Right Eye")
        # Draw Left Eye (Viewer Right)
        self._draw_single_eye(canvas, features.left_eye, w, h, "Left Eye")

        return canvas

    def _draw_single_eye(
        self,
        canvas: np.ndarray,
        eye: SingleEyeFeatures,
        w: int,
        h: int,
        label: str,
    ) -> None:
        """Draw contours, pupil center, and iris boundaries on canvas."""
        pts = np.array([pt.to_pixel(w, h) for pt in eye.contour_points], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 200), thickness=1, lineType=cv2.LINE_AA)

        # Draw Iris Center & Ring
        if eye.iris_center:
            center_px = eye.iris_center.to_pixel(w, h)
            # Red pupil center dot
            cv2.circle(canvas, center_px, 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            # Iris boundary circle
            if eye.iris_points and len(eye.iris_points) >= 4:
                iris_top = eye.iris_points[1].to_pixel(w, h)
                iris_radius = max(3, int(np.hypot(center_px[0] - iris_top[0], center_px[1] - iris_top[1])))
                cv2.circle(canvas, center_px, iris_radius, (0, 255, 255), 1, lineType=cv2.LINE_AA)

            # Small telemetry text above eye
            text = f"rel: ({eye.iris_relative_x:.2f}, {eye.iris_relative_y:.2f})"
            cv2.putText(
                canvas,
                text,
                (center_px[0] - 30, center_px[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    def close(self) -> None:
        """Release Face Mesh resources."""
        if hasattr(self, "mesh") and self.mesh is not None:
            self.mesh.close()
            self.mesh = None

    def __enter__(self) -> "EyeTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def run_eye_tracking_live() -> None:
    """Launch interactive real-time eye and iris tracking with live webcam."""
    cam = CameraManager(CameraConfig(width=1280, height=720, mirror=True))
    tracker = EyeTracker(EyeTrackingConfig(refine_landmarks=True))

    window_name = "AdaptiveEye - Eye Landmark & Iris Tracking"
    print("\nLive Eye Tracking running. Press [Q] to exit.")

    try:
        cam.start()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        for frame in cam.stream():
            features = tracker.process(frame)
            annotated = tracker.draw_eye_annotations(frame, features)

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    finally:
        cam.stop()
        tracker.close()
        cv2.destroyAllWindows()
