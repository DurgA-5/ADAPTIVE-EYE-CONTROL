"""AdaptiveEye Core Data Transfer Objects and Contracts."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class SystemState(str, Enum):
    """Global operational state of the AdaptiveEye pipeline."""
    INITIALIZING = "INITIALIZING"
    UNAUTHORIZED = "UNAUTHORIZED"
    VERIFYING = "VERIFYING"
    TRACKING_ACTIVE = "TRACKING_ACTIVE"
    TRACKING_UNSTABLE = "TRACKING_UNSTABLE"
    TRACKING_LOST = "TRACKING_LOST"
    PAUSED = "PAUSED"
    CALIBRATING = "CALIBRATING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Point2D:
    """Normalized or pixel 2D point representation."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return self.x, self.y

    def to_pixel(self, width: int, height: int) -> Tuple[int, int]:
        return int(round(self.x * width)), int(round(self.y * height))


@dataclass(frozen=True)
class BoundingBox:
    """Normalized bounding box coordinates (0.0 to 1.0)."""
    xmin: float
    ymin: float
    width: float
    height: float

    @property
    def xmax(self) -> float:
        return self.xmin + self.width

    @property
    def ymax(self) -> float:
        return self.ymin + self.height

    def to_pixel(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to absolute integer pixel bounds."""
        px_x = int(round(self.xmin * img_w))
        px_y = int(round(self.ymin * img_h))
        px_w = int(round(self.width * img_w))
        px_h = int(round(self.height * img_h))
        return px_x, px_y, px_w, px_h


@dataclass
class DetectionResult:
    """Face detection bounding box, score, and keypoint landmarks."""
    bbox: BoundingBox
    confidence: float
    landmarks: List[Point2D] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Biometric identity verification outcome."""
    is_authorized: bool
    similarity_score: float
    threshold: float
    subject_id: Optional[str] = None
    confidence_margin: float = 0.0


@dataclass
class EyeLandmarks:
    """Extracted eye contours and iris center positions."""
    left_eye_points: List[Point2D]
    right_eye_points: List[Point2D]
    left_iris_center: Optional[Point2D] = None
    right_iris_center: Optional[Point2D] = None
    left_ear: float = 0.0
    right_ear: float = 0.0


@dataclass
class GazeCoordinates:
    """Estimated gaze screen coordinates with confidence metric."""
    screen_x: int
    screen_y: int
    normalized_x: float
    normalized_y: float
    confidence: float
    is_valid: bool = True
