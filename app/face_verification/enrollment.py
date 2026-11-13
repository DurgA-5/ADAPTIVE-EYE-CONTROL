"""AdaptiveEye Quality-Controlled Face Enrollment Engine.

Enforces explicit user consent, strict optical quality filtering (sharpness, lighting, scale),
computes normalized centroid identity templates, and provides secure biometric storage and deletion.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from app.camera.webcam import CameraConfig, CameraManager
from app.core.datatypes import DetectionResult
from app.core.logger import get_logger
from app.face_detection.detector import FaceDetector, FaceDetectorConfig
from app.face_verification.arcface import ArcFaceConfig, ArcFaceExtractor

logger = get_logger("FaceEnrollment")


class ConsentRequiredError(Exception):
    """Raised when enrollment is attempted without explicit user consent."""


class EnrollmentQualityError(Exception):
    """Raised when an enrolled frame fails optical quality criteria."""


@dataclass
class EnrollmentProfile:
    """Metadata describing an authorized user's biometric identity profile."""
    subject_id: str
    user_name: str
    enrolled_at: str
    num_samples: int
    mean_pairwise_similarity: float
    min_pairwise_similarity: float
    is_valid: bool = True


class FaceEnrollmentManager:
    """Manages consent verification, quality-controlled sample collection, and template creation."""

    def __init__(
        self,
        profile_dir: Optional[Path] = None,
        samples_required: int = 30,
        min_sharpness: float = 80.0,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        min_face_size: int = 100,
    ) -> None:
        self.profile_dir = profile_dir or (
            Path(__file__).resolve().parent.parent.parent / "data" / "enrollment"
        )
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        self.samples_required = samples_required
        self.min_sharpness = min_sharpness
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_face_size = min_face_size

        self.profile_file = self.profile_dir / "identity_profile.json"
        self.template_file = self.profile_dir / "identity_template.npy"

        self.extractor = ArcFaceExtractor()

    def has_enrolled_identity(self) -> bool:
        """Return True if an enrolled identity template and profile exist on disk."""
        return self.profile_file.exists() and self.template_file.exists()

    def get_enrolled_profile(self) -> Optional[EnrollmentProfile]:
        """Load enrolled identity metadata from disk if available."""
        if not self.has_enrolled_identity():
            return None
        try:
            with open(self.profile_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return EnrollmentProfile(**data)
        except Exception as e:
            logger.error("Failed to read enrolled profile: %s", e)
            return None

    def load_enrolled_template(self) -> Optional[np.ndarray]:
        """Load the 512-dimensional centroid template vector from disk."""
        if not self.template_file.exists():
            return None
        try:
            template = np.load(str(self.template_file))
            if template.shape == (512,):
                # Ensure unit norm
                return template / np.linalg.norm(template)
            logger.error("Invalid template shape: %s", template.shape)
            return None
        except Exception as e:
            logger.error("Failed to load identity template: %s", e)
            return None

    def delete_enrolled_identity(self) -> bool:
        """Permanently delete local biometric identity templates and profile records."""
        deleted = False
        if self.profile_file.exists():
            self.profile_file.unlink()
            deleted = True
        if self.template_file.exists():
            self.template_file.unlink()
            deleted = True
        logger.info("Enrolled identity records successfully deleted: %s", deleted)
        return deleted

    def check_frame_quality(
        self,
        bgr_image: np.ndarray,
        detection: DetectionResult,
    ) -> Tuple[bool, str]:
        """Validate optical quality: blur, illumination, and scale.
        
        Returns:
            Tuple of (passes_quality: bool, reason_message: str).
        """
        h, w = bgr_image.shape[:2]
        px_x, px_y, px_w, px_h = detection.bbox.to_pixel(w, h)

        # Scale check
        if px_w < self.min_face_size or px_h < self.min_face_size:
            return False, f"Face too small ({px_w}x{px_h}px < {self.min_face_size}px). Move closer."

        # Crop face for blur and brightness analysis
        x1 = max(0, px_x)
        y1 = max(0, px_y)
        x2 = min(w, px_x + px_w)
        y2 = min(h, px_y + px_h)
        crop = bgr_image[y1:y2, x1:x2]

        if crop.size == 0:
            return False, "Empty crop."

        # Lighting check: Mean brightness
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        if mean_val < self.min_brightness:
            return False, f"Too dark ({mean_val:.1f} < {self.min_brightness:.1f}). Increase lighting."
        if mean_val > self.max_brightness:
            return False, f"Too bright ({mean_val:.1f} > {self.max_brightness:.1f}). Reduce glare."

        # Blur check: Laplacian variance
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < self.min_sharpness:
            return (
                False,
                f"Motion blur detected (sharpness: {laplacian_var:.1f} < {self.min_sharpness:.1f}). Hold still.",
            )

        return True, "Quality acceptable"

    def enroll_from_frames(
        self,
        frames_and_detections: List[Tuple[np.ndarray, DetectionResult]],
        user_name: str,
        subject_id: str,
        consent_granted: bool,
    ) -> EnrollmentProfile:
        """Process pre-collected face frames into a verified identity template."""
        if not consent_granted:
            raise ConsentRequiredError("User consent is strictly required for biometric enrollment.")

        embeddings: List[np.ndarray] = []

        for frame, det in frames_and_detections:
            passes, reason = self.check_frame_quality(frame, det)
            if not passes:
                continue

            aligned = self.extractor.align_face(frame, det.bbox, det.landmarks)
            emb = self.extractor.extract_embedding(aligned)
            embeddings.append(emb)

        if len(embeddings) < 5:
            raise EnrollmentQualityError(
                f"Insufficient high-quality samples: collected {len(embeddings)} of {self.samples_required} required."
            )

        emb_matrix = np.array(embeddings)  # Shape (N, 512)

        # Compute centroid vector and normalize to unit hypersphere
        centroid = np.mean(emb_matrix, axis=0)
        centroid_norm = centroid / np.linalg.norm(centroid)

        # Compute intra-subject pairwise cosine similarities
        pairwise_sims = np.dot(emb_matrix, emb_matrix.T)
        n = len(embeddings)
        upper_tri_indices = np.triu_indices(n, k=1)
        sim_scores = pairwise_sims[upper_tri_indices]

        mean_sim = float(np.mean(sim_scores)) if len(sim_scores) > 0 else 1.0
        min_sim = float(np.min(sim_scores)) if len(sim_scores) > 0 else 1.0

        # Save template vector
        np.save(str(self.template_file), centroid_norm)

        # Save profile metadata
        profile = EnrollmentProfile(
            subject_id=subject_id,
            user_name=user_name,
            enrolled_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            num_samples=len(embeddings),
            mean_pairwise_similarity=round(mean_sim, 4),
            min_pairwise_similarity=round(min_sim, 4),
            is_valid=True,
        )

        with open(self.profile_file, "w", encoding="utf-8") as f:
            json.dump(asdict(profile), f, indent=2)

        logger.info(
            "Enrollment complete: %d samples, Mean Sim: %.4f, Min Sim: %.4f",
            profile.num_samples,
            profile.mean_pairwise_similarity,
            profile.min_pairwise_similarity,
        )
        return profile


def run_interactive_enrollment(user_name: str, subject_id: Optional[str] = None) -> None:
    """Run interactive webcam enrollment workflow with visual feedback."""
    print("=" * 60)
    print("      ADAPTIVEEYE BIOMETRIC ENROLLMENT CONSENT")
    print("=" * 60)
    print("This system will capture face samples to generate a local 512-d")
    print("ArcFace biometric template stored strictly on this machine.")
    print("You may delete your biometric data at any time via: python main.py delete-identity\n")

    consent = input("Do you consent to enrollment for research evaluation? (yes/no): ").strip().lower()
    if consent not in ("yes", "y"):
        print("Enrollment cancelled: Consent was not provided.")
        return

    sid = subject_id or f"user_{int(time.time())}"
    manager = FaceEnrollmentManager()
    detector = FaceDetector(FaceDetectorConfig(min_detection_confidence=0.70))
    cam = CameraManager(CameraConfig(width=1280, height=720, mirror=True))

    captured: List[Tuple[np.ndarray, DetectionResult]] = []
    window_name = "AdaptiveEye - Face Enrollment"

    print(f"\nStarting enrollment for '{user_name}'. Please look at the camera.")
    print("Slightly shift your head position (left, right, center) as samples are collected.\n")

    try:
        cam.start()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        last_sample_time = time.perf_counter()

        for frame in cam.stream():
            det_result = detector.detect(frame)
            canvas = frame.copy()
            h, w = canvas.shape[:2]

            quality_msg = "No face detected"
            quality_ok = False

            if det_result.count == 1 and det_result.primary_face:
                primary = det_result.primary_face
                quality_ok, quality_msg = manager.check_frame_quality(frame, primary)

                # Draw bounding box
                px_x, px_y, px_w, px_h = primary.bbox.to_pixel(w, h)
                box_color = (0, 255, 0) if quality_ok else (0, 165, 255)
                cv2.rectangle(canvas, (px_x, px_y), (px_x + px_w, px_y + px_h), box_color, 2)

                now = time.perf_counter()
                # Sample rate limit: collect 1 sample every 150ms
                if quality_ok and (now - last_sample_time >= 0.15):
                    captured.append((frame.copy(), primary))
                    last_sample_time = now

            # Progress overlay
            progress_pct = min(1.0, len(captured) / manager.samples_required)
            cv2.rectangle(canvas, (0, 0), (w, 50), (20, 20, 20), -1)

            prog_text = (
                f"Enrolling '{user_name}': {len(captured)}/{manager.samples_required} samples "
                f"({progress_pct:.0%}) | {quality_msg}"
            )
            text_color = (0, 255, 0) if quality_ok else (0, 200, 255)
            cv2.putText(canvas, prog_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)

            # Progress bar
            bar_w = int((w - 40) * progress_pct)
            cv2.rectangle(canvas, (20, 42), (20 + bar_w, 46), (0, 255, 0), -1)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or len(captured) >= manager.samples_required:
                break

    finally:
        cam.stop()
        detector.close()
        cv2.destroyAllWindows()

    if len(captured) >= manager.samples_required:
        print("\nProcessing captured face samples into ArcFace identity template...")
        profile = manager.enroll_from_frames(
            frames_and_detections=captured,
            user_name=user_name,
            subject_id=sid,
            consent_granted=True,
        )
        print("Enrollment successfully completed!")
        print(f"  Subject ID: {profile.subject_id}")
        print(f"  User Name: {profile.user_name}")
        print(f"  Samples: {profile.num_samples}")
        print(f"  Mean Pairwise Similarity: {profile.mean_pairwise_similarity:.4f}")
        print(f"  Min Pairwise Similarity: {profile.min_pairwise_similarity:.4f}")
    else:
        print(f"\nEnrollment incomplete: only {len(captured)} samples gathered. Profile not created.")
