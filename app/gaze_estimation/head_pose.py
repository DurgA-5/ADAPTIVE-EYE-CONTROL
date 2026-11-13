"""AdaptiveEye Head Pose Estimation via Perspective-n-Point (solvePnP).

Estimates 3D head orientation angles (Pitch, Yaw, Roll) to compensate for head posture shifts
and prevent gaze drift when the user leans or turns slightly.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from app.core.datatypes import Point2D
from app.core.logger import get_logger

logger = get_logger("HeadPose")

# 3D Canonical Anthropometric Face Model (in mm) in OpenCV coordinate system (+X Right, +Y Down, +Z Forward)
CANONICAL_FACE_3D = np.array(
    [
        [0.0, 0.0, 0.0],          # Nose tip (Landmark 4)
        [0.0, 330.0, 65.0],       # Chin (Landmark 152, below nose -> +Y)
        [225.0, -170.0, 135.0],   # Subject Left Eye / Image Right (Landmark 263, above nose -> -Y, +X)
        [-225.0, -170.0, 135.0],  # Subject Right Eye / Image Left (Landmark 33, above nose -> -Y, -X)
        [150.0, 150.0, 125.0],    # Subject Left Mouth Corner (Landmark 291, below nose -> +Y, +X)
        [-150.0, 150.0, 125.0],   # Subject Right Mouth Corner (Landmark 61, below nose -> +Y, -X)
    ],
    dtype=np.float64,
)

# Corresponding MediaPipe landmark indices
HEAD_POSE_LANDMARK_INDICES = [4, 152, 263, 33, 291, 61]


@dataclass
class HeadPoseAngles:
    """Estimated 3D head orientation angles in degrees."""
    pitch: float  # Up (-) / Down (+)
    yaw: float    # Left (-) / Right (+)
    roll: float   # Tilt counter-clockwise (-) / clockwise (+)
    rvec: np.ndarray
    tvec: np.ndarray


class HeadPoseEstimator:
    """Estimates rigid 3D head transformation using Perspective-n-Point."""

    def __init__(self) -> None:
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def get_camera_matrix(self, img_w: int, img_h: int) -> np.ndarray:
        """Construct synthetic pinhole camera calibration matrix."""
        focal_length = float(img_w)
        center_x = img_w / 2.0
        center_y = img_h / 2.0
        return np.array(
            [
                [focal_length, 0.0, center_x],
                [0.0, focal_length, center_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def estimate(
        self,
        landmarks: List[Point2D],
        img_w: int,
        img_h: int,
    ) -> Optional[HeadPoseAngles]:
        """Estimate Pitch, Yaw, Roll angles from 2D facial landmark points.
        
        Args:
            landmarks: Full MediaPipe landmarks list (at least 300 points).
            img_w: Image width in pixels.
            img_h: Image height in pixels.
            
        Returns:
            HeadPoseAngles or None if landmark points are insufficient.
        """
        if not landmarks or len(landmarks) < max(HEAD_POSE_LANDMARK_INDICES) + 1:
            return None

        # Extract 2D image coordinates for the 6 canonical points
        image_points = np.array(
            [
                [landmarks[idx].x * img_w, landmarks[idx].y * img_h]
                for idx in HEAD_POSE_LANDMARK_INDICES
            ],
            dtype=np.float64,
        )

        camera_matrix = self.get_camera_matrix(img_w, img_h)

        success, rvec, tvec = cv2.solvePnP(
            CANONICAL_FACE_3D,
            image_points,
            camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        # Convert rotation vector to 3x3 rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Decompose rotation matrix into Euler angles
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        return HeadPoseAngles(
            pitch=round(pitch, 2),
            yaw=round(yaw, 2),
            roll=round(roll, 2),
            rvec=rvec,
            tvec=tvec,
        )

    def draw_axes(
        self,
        canvas: np.ndarray,
        angles: HeadPoseAngles,
        landmarks: List[Point2D],
        img_w: int,
        img_h: int,
        length: float = 60.0,
    ) -> None:
        """Render 3D XYZ coordinate axes radiating from the nose tip."""
        if not landmarks or len(landmarks) <= 4:
            return

        nose_pt = landmarks[4].to_pixel(img_w, img_h)
        camera_matrix = self.get_camera_matrix(img_w, img_h)

        # 3D axis points: X (Red), Y (Green), Z (Blue pointing out toward viewer)
        axis_3d = np.array(
            [
                [length, 0.0, 0.0],
                [0.0, length, 0.0],
                [0.0, 0.0, -length],
            ],
            dtype=np.float64,
        )

        proj_points, _ = cv2.projectPoints(
            axis_3d,
            angles.rvec,
            angles.tvec,
            camera_matrix,
            self.dist_coeffs,
        )

        p_x = (int(proj_points[0][0][0]), int(proj_points[0][0][1]))
        p_y = (int(proj_points[1][0][0]), int(proj_points[1][0][1]))
        p_z = (int(proj_points[2][0][0]), int(proj_points[2][0][1]))

        # X axis (Right/Red), Y axis (Down/Green), Z axis (Forward/Blue)
        cv2.line(canvas, nose_pt, p_x, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, nose_pt, p_y, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(canvas, nose_pt, p_z, (255, 0, 0), 2, cv2.LINE_AA)
