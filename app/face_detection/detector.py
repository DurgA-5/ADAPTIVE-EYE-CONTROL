"""AdaptiveEye Face Detection Module.

Utilizes MediaPipe BlazeFace to deliver sub-10ms real-time face detection,
bounding-box extraction, 6-point landmark keypoints, and multi-face safety monitoring.
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
from app.core.datatypes import BoundingBox, DetectionResult, Point2D
from app.core.logger import get_logger

logger = get_logger("FaceDetector")


@dataclass
class FaceDetectorConfig:
    """Strongly-typed face detector configuration."""
    model_selection: int = 0  # 0: short-range (<2m), 1: full-range (<5m)
    min_detection_confidence: float = 0.65
    min_face_size_pixels: int = 60
    allow_multiple_faces: bool = False
    max_num_faces: int = 5

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "FaceDetectorConfig":
        """Load configuration from a YAML file."""
        if not yaml_path.exists():
            logger.warning("Config %s not found. Using defaults.", yaml_path)
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        fd_data = data.get("face_detection", {})
        return cls(
            model_selection=int(fd_data.get("model_selection", 0)),
            min_detection_confidence=float(fd_data.get("min_detection_confidence", 0.65)),
            min_face_size_pixels=int(fd_data.get("min_face_size_pixels", 60)),
            allow_multiple_faces=bool(fd_data.get("allow_multiple_faces", False)),
            max_num_faces=int(fd_data.get("max_num_faces", 5)),
        )


@dataclass
class FaceDetectionResult:
    """Aggregate output from face detection."""
    faces: List[DetectionResult] = field(default_factory=list)
    count: int = 0
    primary_face: Optional[DetectionResult] = None
    is_safe: bool = False
    multi_face_hazard: bool = False
    latency_ms: float = 0.0


class FaceDetector:
    """Encapsulates MediaPipe Face Detection with safety checks and quality filtering."""

    def __init__(self, config: Optional[FaceDetectorConfig] = None) -> None:
        self.config = config or FaceDetectorConfig()
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=self.config.model_selection,
            min_detection_confidence=self.config.min_detection_confidence,
        )
        logger.info(
            "FaceDetector initialized (model_selection=%d, min_conf=%.2f)",
            self.config.model_selection,
            self.config.min_detection_confidence,
        )

    def detect(self, bgr_image: np.ndarray) -> FaceDetectionResult:
        """Perform face detection on a BGR image frame.
        
        Args:
            bgr_image: OpenCV BGR image array (H, W, 3).
            
        Returns:
            FaceDetectionResult containing valid detected faces, primary face,
            and safety hazard indicators.
        """
        if bgr_image is None or bgr_image.size == 0:
            return FaceDetectionResult()

        h, w = bgr_image.shape[:2]
        start_time = time.perf_counter()

        # MediaPipe expects RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        mp_results = self.detector.process(rgb_image)

        latency = (time.perf_counter() - start_time) * 1000.0
        detected_faces: List[DetectionResult] = []

        if mp_results.detections:
            for det in mp_results.detections:
                score = float(det.score[0]) if det.score else 0.0
                if score < self.config.min_detection_confidence:
                    continue

                bbox_rel = det.location_data.relative_bounding_box
                # Clamp normalized bounds to [0.0, 1.0]
                xmin = max(0.0, min(1.0, float(bbox_rel.xmin)))
                ymin = max(0.0, min(1.0, float(bbox_rel.ymin)))
                width = max(0.0, min(1.0 - xmin, float(bbox_rel.width)))
                height = max(0.0, min(1.0 - ymin, float(bbox_rel.height)))

                # Quality check: min pixel size
                if (width * w < self.config.min_face_size_pixels) or (
                    height * h < self.config.min_face_size_pixels
                ):
                    continue

                bbox = BoundingBox(xmin=xmin, ymin=ymin, width=width, height=height)

                # Extract 6 standard facial keypoints
                keypoints: List[Point2D] = []
                for kp in det.location_data.relative_keypoints:
                    kp_x = max(0.0, min(1.0, float(kp.x)))
                    kp_y = max(0.0, min(1.0, float(kp.y)))
                    keypoints.append(Point2D(x=kp_x, y=kp_y))

                detected_faces.append(
                    DetectionResult(
                        bbox=bbox,
                        confidence=score,
                        landmarks=keypoints,
                    )
                )

        # Sort faces by bounding box area (largest face first = primary subject)
        detected_faces.sort(
            key=lambda f: f.bbox.width * f.bbox.height,
            reverse=True,
        )

        face_count = len(detected_faces)
        primary = detected_faces[0] if face_count > 0 else None
        multi_hazard = face_count > 1 and not self.config.allow_multiple_faces
        is_safe = face_count == 1 and not multi_hazard

        return FaceDetectionResult(
            faces=detected_faces,
            count=face_count,
            primary_face=primary,
            is_safe=is_safe,
            multi_face_hazard=multi_hazard,
            latency_ms=latency,
        )

    def draw_annotations(
        self,
        bgr_image: np.ndarray,
        result: FaceDetectionResult,
    ) -> np.ndarray:
        """Render bounding boxes, keypoints, and safety status badges onto the frame."""
        canvas = bgr_image.copy()
        h, w = canvas.shape[:2]

        # Top status bar
        status_color = (0, 200, 0) if result.is_safe else (0, 0, 255)
        primary_conf = result.primary_face.confidence if result.primary_face else 0.0
        status_text = (
            f"Faces: {result.count} | Primary Conf: {primary_conf:.2f} | "
            f"Latency: {result.latency_ms:.1f}ms | "
            f"Status: {'SECURE' if result.is_safe else 'SECURITY HOLD'}"
        )

        cv2.rectangle(canvas, (0, 0), (w, 36), (25, 25, 25), -1)
        cv2.putText(
            canvas,
            status_text,
            (15, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color,
            1,
            cv2.LINE_AA,
        )

        # Render warning if multiple faces present (Safety Rule 1)
        if result.multi_face_hazard:
            cv2.rectangle(canvas, (0, 36), (w, 70), (0, 0, 180), -1)
            cv2.putText(
                canvas,
                "SAFETY WARNING: MULTIPLE FACES DETECTED - SYSTEM INTERLOCKED",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        for i, face in enumerate(result.faces):
            px_x, px_y, px_w, px_h = face.bbox.to_pixel(w, h)
            # Primary face green if safe, secondary faces yellow/orange
            box_color = (0, 255, 0) if (i == 0 and not result.multi_face_hazard) else (0, 140, 255)

            # Bounding box
            cv2.rectangle(canvas, (px_x, px_y), (px_x + px_w, px_y + px_h), box_color, 2)

            # Label badge
            label = f"Face #{i+1} ({face.confidence:.0%})"
            cv2.rectangle(canvas, (px_x, max(0, px_y - 22)), (px_x + 130, px_y), box_color, -1)
            cv2.putText(
                canvas,
                label,
                (px_x + 4, max(12, px_y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            # Keypoints (eyes, nose, mouth, ears)
            for kp in face.landmarks:
                kp_px = kp.to_pixel(w, h)
                cv2.circle(canvas, kp_px, 3, (0, 255, 255), -1)

        return canvas

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()
            self.detector = None

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def run_face_detection_live(
    camera_cfg_path: Optional[Path] = None,
    detector_cfg_path: Optional[Path] = None,
) -> None:
    """Launch interactive real-time face detection with camera stream."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    c_path = camera_cfg_path or (root_dir / "config" / "camera_config.yaml")
    d_path = detector_cfg_path or (root_dir / "config" / "face_detection_config.yaml")

    cam_cfg = CameraConfig.from_yaml(c_path)
    det_cfg = FaceDetectorConfig.from_yaml(d_path)

    window_name = "AdaptiveEye - Face Detection"

    with CameraManager(cam_cfg) as cam, FaceDetector(det_cfg) as detector:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        logger.info("Face detection running. Press [Q] or [ESC] to exit.")

        for frame in cam.stream():
            result = detector.detect(frame)
            annotated = detector.draw_annotations(frame, result)

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cv2.destroyAllWindows()
