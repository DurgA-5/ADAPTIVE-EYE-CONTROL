"""AdaptiveEye Person Tracking and Identity Locking Engine.

Maintains spatial trajectory continuity for the authenticated user, enforces zero-trust
takeover protection, handles multi-person scenes, and requires re-verification upon loss.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np
import yaml

from app.camera.webcam import CameraConfig, CameraManager
from app.core.datatypes import BoundingBox, DetectionResult, Point2D
from app.core.logger import get_logger
from app.face_detection.detector import FaceDetectionResult, FaceDetector, FaceDetectorConfig
from app.face_verification.verifier import AuthState, FaceVerifier, LiveVerificationOutcome

logger = get_logger("PersonTracker")


class TrackingState(str, Enum):
    """Lifecycle state of identity locking and person tracking."""
    SEARCHING = "SEARCHING"                   # Scanning camera view for human face
    VERIFYING = "VERIFYING"                   # Performing ArcFace verification
    LOCKED_AUTHORIZED = "LOCKED_AUTHORIZED"   # Identity verified & locked; safe for cursor control
    IDENTITY_AMBIGUOUS = "IDENTITY_AMBIGUOUS" # Conflicting or overlapping faces; session paused
    TRACKING_LOST = "TRACKING_LOST"           # Subject departed or face occluded; session paused
    REVERIFY_REQUIRED = "REVERIFY_REQUIRED"   # Face reappeared after loss; must re-authenticate


@dataclass
class TrackedPerson:
    """State vector of the actively tracked authorized subject."""
    tracking_state: TrackingState
    is_safe_for_control: bool
    bbox: Optional[BoundingBox] = None
    landmarks: List[Point2D] = None
    confidence: float = 0.0
    user_name: Optional[str] = None
    similarity_score: float = 0.0
    consecutive_locked_frames: int = 0
    lost_frames_count: int = 0
    status_message: str = "Initializing tracker"


@dataclass
class PersonTrackingConfig:
    """Configuration for spatial tracking and identity continuity."""
    min_iou_continuity: float = 0.35
    max_centroid_distance_normalized: float = 0.25
    max_lost_frames_tolerance: int = 15
    reverification_interval_frames: int = 45
    require_reverification_on_reacquire: bool = True
    min_face_confidence_tracking: float = 0.60

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "PersonTrackingConfig":
        if not yaml_path.exists():
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        pt_data = data.get("person_tracking", {})
        return cls(
            min_iou_continuity=float(pt_data.get("min_iou_continuity", 0.35)),
            max_centroid_distance_normalized=float(pt_data.get("max_centroid_distance_normalized", 0.25)),
            max_lost_frames_tolerance=int(pt_data.get("max_lost_frames_tolerance", 15)),
            reverification_interval_frames=int(pt_data.get("reverification_interval_frames", 45)),
            require_reverification_on_reacquire=bool(pt_data.get("require_reverification_on_reacquire", True)),
            min_face_confidence_tracking=float(pt_data.get("min_face_confidence_tracking", 0.60)),
        )


class IdentityLockTracker:
    """Enforces spatial identity locking and anti-takeover safety rules."""

    def __init__(
        self,
        config: Optional[PersonTrackingConfig] = None,
        verifier: Optional[FaceVerifier] = None,
    ) -> None:
        self.config = config or PersonTrackingConfig()
        self.verifier = verifier or FaceVerifier()

        self._state = TrackingState.SEARCHING
        self._last_locked_bbox: Optional[BoundingBox] = None
        self._last_locked_landmarks: List[Point2D] = []
        self._locked_user_name: Optional[str] = None
        self._locked_similarity: float = 0.0

        self._consecutive_locked_frames: int = 0
        self._lost_frames_count: int = 0
        self._frames_since_reverification: int = 0

    @property
    def current_state(self) -> TrackingState:
        return self._state

    @property
    def is_locked(self) -> bool:
        return self._state == TrackingState.LOCKED_AUTHORIZED

    def compute_iou(self, box_a: BoundingBox, box_b: BoundingBox) -> float:
        """Compute Intersection over Union (IoU) between two normalized boxes."""
        x_left = max(box_a.xmin, box_b.xmin)
        y_top = max(box_a.ymin, box_b.ymin)
        x_right = min(box_a.xmax, box_b.xmax)
        y_bottom = min(box_a.ymax, box_b.ymax)

        intersection_w = max(0.0, x_right - x_left)
        intersection_h = max(0.0, y_bottom - y_top)
        intersection_area = intersection_w * intersection_h

        area_a = box_a.width * box_a.height
        area_b = box_b.width * box_b.height
        union_area = area_a + area_b - intersection_area

        if union_area <= 1e-12:
            return 0.0
        return float(intersection_area / union_area)

    def compute_centroid_distance(self, box_a: BoundingBox, box_b: BoundingBox) -> float:
        """Compute Euclidean distance between normalized centroids."""
        c_a_x = box_a.xmin + box_a.width / 2.0
        c_a_y = box_a.ymin + box_a.height / 2.0
        c_b_x = box_b.xmin + box_b.width / 2.0
        c_b_y = box_b.ymin + box_b.height / 2.0
        return float(np.hypot(c_a_x - c_b_x, c_a_y - c_b_y))

    def update(
        self,
        bgr_frame: np.ndarray,
        detection_result: FaceDetectionResult,
    ) -> TrackedPerson:
        """Process incoming frame and face detections, updating tracking state and locking.
        
        Args:
            bgr_frame: Raw camera frame.
            detection_result: Detections from FaceDetector.
            
        Returns:
            TrackedPerson with safety flag and active bounding box.
        """
        # Case 1: No faces detected in scene
        if detection_result.count == 0:
            return self._handle_no_faces()

        # Case 2: In SEARCHING or REVERIFY_REQUIRED state -> Must verify candidate face
        if self._state in (TrackingState.SEARCHING, TrackingState.TRACKING_LOST, TrackingState.REVERIFY_REQUIRED):
            return self._handle_verification_attempt(bgr_frame, detection_result)

        # Case 3: Currently LOCKED_AUTHORIZED -> Match candidate faces against spatial lock
        if self._state == TrackingState.LOCKED_AUTHORIZED:
            return self._handle_locked_tracking(bgr_frame, detection_result)

        # Case 4: IDENTITY_AMBIGUOUS -> Try to resolve or force re-verification
        if self._state == TrackingState.IDENTITY_AMBIGUOUS:
            return self._handle_ambiguous_scene(bgr_frame, detection_result)

        return self._make_result(self._state, False, "Unknown tracking condition")

    def _handle_no_faces(self) -> TrackedPerson:
        """Handle frame with zero detected faces."""
        if self._state == TrackingState.LOCKED_AUTHORIZED:
            self._lost_frames_count += 1
            if self._lost_frames_count <= self.config.max_lost_frames_tolerance:
                # Brief tolerance window (e.g. blink or swift head turn)
                return self._make_result(
                    TrackingState.LOCKED_AUTHORIZED,
                    False,  # Pause cursor control while face is momentarily lost
                    f"Brief tracking occlusion ({self._lost_frames_count}/{self.config.max_lost_frames_tolerance})",
                    bbox=self._last_locked_bbox,
                    landmarks=self._last_locked_landmarks,
                )
            else:
                # Lock revoked!
                logger.info("Tracking lost: maximum tolerance exceeded (%d frames).", self._lost_frames_count)
                self._state = TrackingState.TRACKING_LOST
                self._last_locked_bbox = None
                return self._make_result(
                    TrackingState.TRACKING_LOST,
                    False,
                    "Tracking lost: user departed or camera obscured",
                )
        else:
            self._state = TrackingState.SEARCHING
            return self._make_result(TrackingState.SEARCHING, False, "Searching for user face...")

    def _handle_verification_attempt(
        self,
        bgr_frame: np.ndarray,
        detection_result: FaceDetectionResult,
    ) -> TrackedPerson:
        """Attempt ArcFace authentication on candidate face."""
        # Safety rule: if multiple faces are present during verification, reject immediately
        if detection_result.count > 1:
            self._state = TrackingState.IDENTITY_AMBIGUOUS
            return self._make_result(
                TrackingState.IDENTITY_AMBIGUOUS,
                False,
                "Multi-person scene: authorization blocked until single face present",
            )

        candidate = detection_result.primary_face
        if candidate.confidence < self.config.min_face_confidence_tracking:
            return self._make_result(self._state, False, "Candidate face confidence low")

        # Run ArcFace authentication
        self._state = TrackingState.VERIFYING
        outcome = self.verifier.verify(bgr_frame, candidate)

        if outcome.auth_state == AuthState.AUTHORIZED:
            # Identity confirmed! Lock onto this spatial trajectory
            self._state = TrackingState.LOCKED_AUTHORIZED
            self._last_locked_bbox = candidate.bbox
            self._last_locked_landmarks = candidate.landmarks
            self._locked_user_name = outcome.user_name
            self._locked_similarity = outcome.smoothed_similarity
            self._consecutive_locked_frames = 1
            self._lost_frames_count = 0
            self._frames_since_reverification = 0

            logger.info("Identity lock established on '%s' (Sim: %.3f)", outcome.user_name, outcome.smoothed_similarity)
            return self._make_result(
                TrackingState.LOCKED_AUTHORIZED,
                True,
                f"Identity Locked: {outcome.user_name} (Sim: {outcome.smoothed_similarity:.2f})",
                bbox=candidate.bbox,
                landmarks=candidate.landmarks,
                similarity=outcome.smoothed_similarity,
            )
        elif outcome.auth_state == AuthState.UNCERTAIN:
            self._state = TrackingState.REVERIFY_REQUIRED
            return self._make_result(
                TrackingState.REVERIFY_REQUIRED,
                False,
                f"Identity uncertain (Sim: {outcome.smoothed_similarity:.2f} < {outcome.high_threshold:.2f})",
                bbox=candidate.bbox,
                landmarks=candidate.landmarks,
                similarity=outcome.smoothed_similarity,
            )
        else:
            self._state = TrackingState.SEARCHING
            return self._make_result(
                TrackingState.SEARCHING,
                False,
                f"Unauthorized person rejected (Sim: {outcome.smoothed_similarity:.2f})",
                bbox=candidate.bbox,
                similarity=outcome.smoothed_similarity,
            )

    def _handle_locked_tracking(
        self,
        bgr_frame: np.ndarray,
        detection_result: FaceDetectionResult,
    ) -> TrackedPerson:
        """Spatially match detected candidate faces to existing locked trajectory."""
        best_match: Optional[DetectionResult] = None
        best_iou: float = 0.0

        # Find candidate with highest spatial continuity (IoU)
        for face in detection_result.faces:
            iou = self.compute_iou(self._last_locked_bbox, face.bbox)
            dist = self.compute_centroid_distance(self._last_locked_bbox, face.bbox)
            if iou >= self.config.min_iou_continuity or dist <= self.config.max_centroid_distance_normalized:
                if iou > best_iou:
                    best_iou = iou
                    best_match = face

        # If multiple faces detected, ensure intruder is not confounding our locked subject
        if detection_result.count > 1:
            if best_match is None:
                # Intruder stepped in, original face lost
                self._state = TrackingState.IDENTITY_AMBIGUOUS
                return self._make_result(
                    TrackingState.IDENTITY_AMBIGUOUS,
                    False,
                    "Multi-face intrusion: locked subject trajectory lost",
                )
            else:
                # Check for spatial collision with other faces
                for other in detection_result.faces:
                    if other is not best_match:
                        collision_iou = self.compute_iou(best_match.bbox, other.bbox)
                        if collision_iou > 0.20:
                            # Faces overlapping -> Pause cursor immediately (Safety Rule 1)
                            self._state = TrackingState.IDENTITY_AMBIGUOUS
                            return self._make_result(
                                TrackingState.IDENTITY_AMBIGUOUS,
                                False,
                                "Face occlusion hazard: overlapping subjects detected",
                                bbox=best_match.bbox,
                            )

        if best_match is not None:
            # Trajectory matched successfully!
            self._last_locked_bbox = best_match.bbox
            self._last_locked_landmarks = best_match.landmarks
            self._consecutive_locked_frames += 1
            self._lost_frames_count = 0
            self._frames_since_reverification += 1

            # Heartbeat periodic verification
            if self._frames_since_reverification >= self.config.reverification_interval_frames:
                self._frames_since_reverification = 0
                outcome = self.verifier.verify(bgr_frame, best_match)
                self._locked_similarity = outcome.smoothed_similarity
                if outcome.auth_state != AuthState.AUTHORIZED:
                    logger.warning("Periodic verification failed. Downgrading to REVERIFY_REQUIRED.")
                    self._state = TrackingState.REVERIFY_REQUIRED
                    return self._make_result(
                        TrackingState.REVERIFY_REQUIRED,
                        False,
                        f"Periodic auth check failed (Sim: {outcome.smoothed_similarity:.2f})",
                        bbox=best_match.bbox,
                        similarity=outcome.smoothed_similarity,
                    )

            return self._make_result(
                TrackingState.LOCKED_AUTHORIZED,
                True,  # SAFE FOR CURSOR DISPATCH
                f"Locked: {self._locked_user_name} | Continuity IoU: {best_iou:.2f}",
                bbox=best_match.bbox,
                landmarks=best_match.landmarks,
                similarity=self._locked_similarity,
            )
        else:
            # Face momentarily missed spatial matching
            self._lost_frames_count += 1
            if self._lost_frames_count > self.config.max_lost_frames_tolerance:
                self._state = TrackingState.REVERIFY_REQUIRED
                return self._make_result(
                    TrackingState.REVERIFY_REQUIRED,
                    False,
                    "Lock broken by motion jump: re-verification required",
                )
            return self._make_result(
                TrackingState.LOCKED_AUTHORIZED,
                False,
                f"Trajectory coasting ({self._lost_frames_count}/{self.config.max_lost_frames_tolerance})",
                bbox=self._last_locked_bbox,
            )

    def _handle_ambiguous_scene(
        self,
        bgr_frame: np.ndarray,
        detection_result: FaceDetectionResult,
    ) -> TrackedPerson:
        """Handle resolving multi-person ambiguity back to single verified face."""
        if detection_result.count == 1:
            # Back to single face -> require clean re-verification
            self._state = TrackingState.REVERIFY_REQUIRED
            return self._handle_verification_attempt(bgr_frame, detection_result)
        return self._make_result(
            TrackingState.IDENTITY_AMBIGUOUS,
            False,
            f"Multiple people present ({detection_result.count} faces) - Session Paused",
        )

    def _make_result(
        self,
        state: TrackingState,
        is_safe: bool,
        msg: str,
        bbox: Optional[BoundingBox] = None,
        landmarks: Optional[List[Point2D]] = None,
        similarity: float = 0.0,
    ) -> TrackedPerson:
        return TrackedPerson(
            tracking_state=state,
            is_safe_for_control=is_safe,
            bbox=bbox,
            landmarks=landmarks or [],
            confidence=similarity or self._locked_similarity,
            user_name=self._locked_user_name,
            similarity_score=similarity or self._locked_similarity,
            consecutive_locked_frames=self._consecutive_locked_frames,
            lost_frames_count=self._lost_frames_count,
            status_message=msg,
        )

    def draw_tracking_overlay(
        self,
        bgr_image: np.ndarray,
        person: TrackedPerson,
    ) -> np.ndarray:
        """Render tracking status badges, locked boundary bounding box, and telemetry."""
        canvas = bgr_image.copy()
        h, w = canvas.shape[:2]

        # Color coding by state
        if person.tracking_state == TrackingState.LOCKED_AUTHORIZED:
            status_color = (0, 255, 0) if person.is_safe_for_control else (0, 200, 255)
        elif person.tracking_state in (TrackingState.VERIFYING, TrackingState.REVERIFY_REQUIRED):
            status_color = (0, 215, 255)  # Gold
        elif person.tracking_state == TrackingState.IDENTITY_AMBIGUOUS:
            status_color = (0, 0, 255)    # Red alert
        else:
            status_color = (160, 160, 160)

        # Top banner
        cv2.rectangle(canvas, (0, 0), (w, 42), (20, 20, 20), -1)
        headline = f"Tracking: {person.tracking_state.value} | Control: {'ACTIVE' if person.is_safe_for_control else 'PAUSED'}"
        cv2.putText(canvas, headline, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

        # Subtitle banner
        cv2.rectangle(canvas, (0, 42), (w, 68), (30, 30, 30), -1)
        sub_text = f"Msg: {person.status_message} | Locked: {person.consecutive_locked_frames} frames"
        cv2.putText(canvas, sub_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

        # Draw locked bounding box with crosshair/target corners
        if person.bbox:
            px_x, px_y, px_w, px_h = person.bbox.to_pixel(w, h)
            # Main box
            cv2.rectangle(canvas, (px_x, px_y), (px_x + px_w, px_y + px_h), status_color, 2)
            # Locked identity tag
            tag = f"[ID-LOCK] {person.user_name or 'Subject'} ({person.similarity_score:.2f})"
            cv2.rectangle(canvas, (px_x, max(0, px_y - 20)), (px_x + 220, px_y), status_color, -1)
            cv2.putText(canvas, tag, (px_x + 5, max(12, px_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        return canvas


def run_person_tracking_live() -> None:
    """Launch interactive person tracking with identity locking."""
    cam = CameraManager(CameraConfig(width=1280, height=720, mirror=True))
    detector = FaceDetector(FaceDetectorConfig(min_detection_confidence=0.65))
    verifier = FaceVerifier()
    tracker = IdentityLockTracker(verifier=verifier)

    if not verifier.is_enrolled:
        print("\nERROR: No enrolled identity found on disk.")
        print("Please enroll first: python main.py enroll\n")
        return

    window_name = "AdaptiveEye - Authorized Person Tracking & Identity Lock"
    print(f"\nPerson tracking running for '{verifier._profile.user_name}'. Press [Q] to exit.")

    try:
        cam.start()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        for frame in cam.stream():
            det_result = detector.detect(frame)
            tracked_person = tracker.update(frame, det_result)
            annotated = tracker.draw_tracking_overlay(frame, tracked_person)

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    finally:
        cam.stop()
        detector.close()
        cv2.destroyAllWindows()
