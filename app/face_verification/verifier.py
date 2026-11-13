"""AdaptiveEye Real-Time Face Verification Engine.

Performs confidence-aware matching between live camera faces and enrolled ArcFace
identity templates, incorporating multi-frame temporal smoothing and three-tier decisioning.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time
from typing import Optional, Tuple
import cv2
import numpy as np
import yaml

from app.camera.webcam import CameraConfig, CameraManager
from app.core.datatypes import DetectionResult, VerificationResult
from app.core.logger import get_logger
from app.face_detection.detector import FaceDetector, FaceDetectorConfig
from app.face_verification.arcface import ArcFaceConfig, ArcFaceExtractor
from app.face_verification.enrollment import FaceEnrollmentManager

logger = get_logger("FaceVerifier")


def load_verification_thresholds() -> Tuple[float, float, float, int, int]:
    """Load calibrated thresholds, hysteresis floor, and cadence from config."""
    cfg_path = Path(__file__).resolve().parent.parent.parent / "config" / "face_verification_config.yaml"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            fv = data.get("face_verification", {})
            return (
                float(fv.get("high_confidence_threshold", 0.78)),
                float(fv.get("release_threshold", 0.68)),
                float(fv.get("low_confidence_threshold", 0.60)),
                int(fv.get("temporal_smoothing_frames", 5)),
                int(fv.get("verify_every_n_frames", 4)),
            )
        except Exception as e:
            logger.warning("Could not read verification config: %s", e)
    return 0.78, 0.68, 0.60, 5, 4


class AuthState(str, Enum):
    """Three-tier biometric authorization state."""
    AUTHORIZED = "AUTHORIZED"       # Similarity >= high threshold: full access
    UNCERTAIN = "UNCERTAIN"         # Low threshold <= Similarity < high threshold: safety hold
    REJECTED = "REJECTED"           # Similarity < low threshold: impostor block
    UNENROLLED = "UNENROLLED"       # No enrolled template on disk


@dataclass
class LiveVerificationOutcome:
    """Detailed verification outcome for live pipeline consumption."""
    auth_state: AuthState
    is_authorized: bool
    instant_similarity: float
    smoothed_similarity: float
    high_threshold: float
    low_threshold: float
    subject_id: Optional[str] = None
    user_name: Optional[str] = None


class FaceVerifier:
    """Performs real-time biometric identity verification with temporal smoothing and hysteresis."""

    def __init__(
        self,
        high_threshold: Optional[float] = None,
        release_threshold: Optional[float] = None,
        low_threshold: Optional[float] = None,
        temporal_window: Optional[int] = None,
        verify_every_n_frames: Optional[int] = None,
        enrollment_manager: Optional[FaceEnrollmentManager] = None,
        extractor: Optional[ArcFaceExtractor] = None,
    ) -> None:
        cfg_high, cfg_rel, cfg_low, cfg_win, cfg_cadence = load_verification_thresholds()
        self.high_threshold = high_threshold if high_threshold is not None else cfg_high
        self.release_threshold = release_threshold if release_threshold is not None else cfg_rel
        self.low_threshold = low_threshold if low_threshold is not None else cfg_low
        self.temporal_window = temporal_window if temporal_window is not None else cfg_win
        self.verify_every_n_frames = verify_every_n_frames if verify_every_n_frames is not None else cfg_cadence

        self.enrollment_mgr = enrollment_manager or FaceEnrollmentManager()
        self.extractor = extractor or ArcFaceExtractor()

        self._template: Optional[np.ndarray] = None
        self._profile = None
        self._similarity_buffer: deque[float] = deque(maxlen=self.temporal_window)

        # Hysteresis state tracking
        self._current_state = AuthState.UNENROLLED
        self._sub_threshold_counter: int = 0
        self._last_instant_sim: float = 0.0
        self._last_smoothed_sim: float = 0.0

        self.reload_template()

    def reload_template(self) -> bool:
        """Load or reload enrolled template and metadata from disk."""
        self._template = self.enrollment_mgr.load_enrolled_template()
        self._profile = self.enrollment_mgr.get_enrolled_profile()
        self._similarity_buffer.clear()
        self._current_state = AuthState.UNCERTAIN if self._template is not None else AuthState.UNENROLLED
        self._sub_threshold_counter = 0

        if self._template is not None and self._profile is not None:
            logger.info(
                "Enrolled identity loaded: '%s' (ID: %s, Samples: %d)",
                self._profile.user_name,
                self._profile.subject_id,
                self._profile.num_samples,
            )
            return True
        else:
            logger.warning("No valid enrolled template found on disk.")
            return False

    @property
    def is_enrolled(self) -> bool:
        """Return True if an authorized template is actively loaded."""
        return self._template is not None

    def verify(
        self,
        bgr_image: np.ndarray,
        detection: DetectionResult,
        force_extract: bool = True,
    ) -> LiveVerificationOutcome:
        """Verify detected face against the enrolled identity template with state hysteresis."""
        if not self.is_enrolled or self._template is None:
            return LiveVerificationOutcome(
                auth_state=AuthState.UNENROLLED,
                is_authorized=False,
                instant_similarity=0.0,
                smoothed_similarity=0.0,
                high_threshold=self.high_threshold,
                low_threshold=self.low_threshold,
            )

        if force_extract:
            aligned = self.extractor.align_face(bgr_image, detection.bbox, detection.landmarks)
            live_embedding = self.extractor.extract_embedding(aligned)
            instant_sim = self.extractor.compute_similarity(live_embedding, self._template)
            self._similarity_buffer.append(instant_sim)
            self._last_instant_sim = instant_sim

        if self._similarity_buffer:
            smoothed_sim = float(np.median(self._similarity_buffer))
        else:
            smoothed_sim = self._last_instant_sim
        self._last_smoothed_sim = smoothed_sim

        # Biometric Hysteresis Decision Logic:
        # If currently AUTHORIZED, stay AUTHORIZED unless score drops below the release floor
        if self._current_state == AuthState.AUTHORIZED:
            if smoothed_sim < self.release_threshold:
                self._sub_threshold_counter += 1
                if self._sub_threshold_counter >= 3:
                    self._current_state = AuthState.UNCERTAIN if smoothed_sim >= self.low_threshold else AuthState.REJECTED
            else:
                self._sub_threshold_counter = 0
        else:
            # Not yet authorized: must meet high threshold to enter AUTHORIZED state
            if smoothed_sim >= self.high_threshold:
                self._current_state = AuthState.AUTHORIZED
                self._sub_threshold_counter = 0
            elif smoothed_sim >= self.low_threshold:
                self._current_state = AuthState.UNCERTAIN
            else:
                self._current_state = AuthState.REJECTED

        is_auth = (self._current_state == AuthState.AUTHORIZED)
        subject_id = self._profile.subject_id if self._profile else None
        user_name = self._profile.user_name if self._profile else None

        return LiveVerificationOutcome(
            auth_state=self._current_state,
            is_authorized=is_auth,
            instant_similarity=round(self._last_instant_sim, 4),
            smoothed_similarity=round(smoothed_sim, 4),
            high_threshold=self.high_threshold,
            low_threshold=self.low_threshold,
            subject_id=subject_id,
            user_name=user_name,
        )

    def draw_verification_overlay(
        self,
        bgr_image: np.ndarray,
        outcome: LiveVerificationOutcome,
        detection: Optional[DetectionResult] = None,
    ) -> np.ndarray:
        """Render verification status, similarity gauge, and user identity onto frame."""
        canvas = bgr_image.copy()
        h, w = canvas.shape[:2]

        # Determine status colors
        if outcome.auth_state == AuthState.AUTHORIZED:
            status_color = (0, 220, 0)      # Green
            status_label = f"AUTHORIZED: {outcome.user_name or 'Enrolled User'}"
        elif outcome.auth_state == AuthState.UNCERTAIN:
            status_color = (0, 200, 255)    # Amber/Yellow
            status_label = "IDENTITY UNCERTAIN - SESSION ON HOLD"
        elif outcome.auth_state == AuthState.REJECTED:
            status_color = (0, 0, 255)      # Red
            status_label = "ACCESS DENIED: UNKNOWN PERSON"
        else:
            status_color = (180, 180, 180)  # Gray
            status_label = "NO IDENTITY ENROLLED (Run: python main.py enroll)"

        # Top verification banner
        cv2.rectangle(canvas, (0, 0), (w, 42), (20, 20, 20), -1)
        cv2.putText(canvas, status_label, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

        # Telemetry info
        metric_text = (
            f"Sim: {outcome.smoothed_similarity:.3f} (Instant: {outcome.instant_similarity:.3f}) | "
            f"Thresholds: [{outcome.low_threshold:.2f}, {outcome.high_threshold:.2f}]"
        )
        cv2.putText(canvas, metric_text, (w - 480, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw bounding box with authorization color
        if detection:
            px_x, px_y, px_w, px_h = detection.bbox.to_pixel(w, h)
            cv2.rectangle(canvas, (px_x, px_y), (px_x + px_w, px_y + px_h), status_color, 2)

        return canvas


def run_live_verification() -> None:
    """Launch interactive webcam verification loop with cadence decoupling for smooth 30 FPS."""
    cam = CameraManager(CameraConfig(width=1280, height=720, mirror=True))
    detector = FaceDetector(FaceDetectorConfig(min_detection_confidence=0.70))
    verifier = FaceVerifier()

    if not verifier.is_enrolled:
        print("\nERROR: No enrolled identity found.")
        print("Please run enrollment first: python main.py enroll\n")
        return

    window_name = "AdaptiveEye - Biometric Face Verification"
    print(f"\nLive verification running for '{verifier._profile.user_name}'. Press [Q] to exit.")

    try:
        cam.start()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_idx = 0
        last_outcome: Optional[LiveVerificationOutcome] = None
        fps_timer = time.perf_counter()
        fps_counter = 0
        live_fps = 30.0

        for frame in cam.stream():
            frame_idx += 1
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 0.5:
                live_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            det_result = detector.detect(frame)

            if det_result.count == 1 and det_result.primary_face:
                # Decoupled cadence: extract deep features on entry, when not authorized, or every N frames
                should_extract = (
                    (last_outcome is None)
                    or (not last_outcome.is_authorized)
                    or (frame_idx % verifier.verify_every_n_frames == 0)
                )
                last_outcome = verifier.verify(frame, det_result.primary_face, force_extract=should_extract)
                annotated = verifier.draw_verification_overlay(frame, last_outcome, det_result.primary_face)
            elif det_result.count > 1:
                # Multi-face safety interlock
                last_outcome = LiveVerificationOutcome(
                    auth_state=AuthState.REJECTED,
                    is_authorized=False,
                    instant_similarity=0.0,
                    smoothed_similarity=0.0,
                    high_threshold=verifier.high_threshold,
                    low_threshold=verifier.low_threshold,
                )
                annotated = verifier.draw_verification_overlay(frame, last_outcome)
                cv2.putText(
                    annotated,
                    "SAFETY LOCK: MULTIPLE FACES IN CAMERA VIEW",
                    (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                last_outcome = LiveVerificationOutcome(
                    auth_state=AuthState.UNCERTAIN,
                    is_authorized=False,
                    instant_similarity=0.0,
                    smoothed_similarity=0.0,
                    high_threshold=verifier.high_threshold,
                    low_threshold=verifier.low_threshold,
                )
                annotated = verifier.draw_verification_overlay(frame, last_outcome)

            # Draw live FPS indicator on right
            cv2.putText(
                annotated,
                f"FPS: {live_fps:.1f}",
                (annotated.shape[1] - 120, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    finally:
        cam.stop()
        detector.close()
        cv2.destroyAllWindows()
