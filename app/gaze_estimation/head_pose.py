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

    def draw_3d_head_box(
        self,
        canvas: np.ndarray,
        angles: HeadPoseAngles,
        landmarks: List[Point2D],
        img_w: int,
        img_h: int,
        box_color: Tuple[int, int, int] = (255, 160, 40),
        draw_mesh_dots: bool = True,
    ) -> None:
        """Render 3D Wireframe Bounding Box and facial surface mesh points."""
        if not landmarks or len(landmarks) < 468 or angles.rvec is None or angles.tvec is None:
            return

        camera_matrix = self.get_camera_matrix(img_w, img_h)

        # 1. Draw Dense Facial Surface Mesh Points (white/cyan dots)
        if draw_mesh_dots:
            for i in range(0, min(len(landmarks), 468), 2):
                pt = landmarks[i].to_pixel(img_w, img_h)
                cv2.circle(canvas, pt, 1, (240, 240, 240), -1, lineType=cv2.LINE_AA)

        # 2. 3D Head Bounding Box Cuboid in Canonical Space
        head_box_3d = np.array(
            [
                [-220.0, -210.0, -40.0],  # 0: Front top left
                [220.0, -210.0, -40.0],   # 1: Front top right
                [220.0, 310.0, -40.0],    # 2: Front bottom right
                [-220.0, 310.0, -40.0],   # 3: Front bottom left
                [-220.0, -210.0, 190.0],  # 4: Back top left
                [220.0, -210.0, 190.0],   # 5: Back top right
                [220.0, 310.0, 190.0],    # 6: Back bottom right
                [-220.0, 310.0, 190.0],   # 7: Back bottom left
            ],
            dtype=np.float64,
        )

        proj_box, _ = cv2.projectPoints(
            head_box_3d,
            angles.rvec,
            angles.tvec,
            camera_matrix,
            self.dist_coeffs,
        )
        pts = [(int(p[0][0]), int(p[0][1])) for p in proj_box]

        # Draw 12 connecting edges of the 3D cuboid
        # Front face
        for i in range(4):
            cv2.line(canvas, pts[i], pts[(i + 1) % 4], box_color, 2, cv2.LINE_AA)
        # Back face
        for i in range(4):
            cv2.line(canvas, pts[i + 4], pts[((i + 1) % 4) + 4], (180, 100, 20), 1, cv2.LINE_AA)
        # 4 connecting pillars
        for i in range(4):
            cv2.line(canvas, pts[i], pts[i + 4], box_color, 2, cv2.LINE_AA)

        # 3. Bright Green Nose Orientation Line
        if len(landmarks) > 168:
            bridge_pt = landmarks[168].to_pixel(img_w, img_h)
            nose_pt = landmarks[4].to_pixel(img_w, img_h)
            cv2.line(canvas, bridge_pt, nose_pt, (0, 255, 60), 3, cv2.LINE_AA)
            cv2.circle(canvas, bridge_pt, 3, (0, 255, 60), -1, cv2.LINE_AA)
            cv2.circle(canvas, nose_pt, 4, (0, 255, 60), -1, cv2.LINE_AA)
