"""AdaptiveEye ArcFace Feature Embedding Extractor.

Performs facial alignment, preprocessing, and 512-dimensional L2-normalized
feature representation generation using ONNX Runtime.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort
import yaml

from app.core.datatypes import BoundingBox, Point2D
from app.core.logger import get_logger

logger = get_logger("ArcFace")

# Canonical 112x112 ArcFace landmark reference points (left-eye, right-eye, nose-tip)
ARCFACE_REFERENCE_PTS_3 = np.array(
    [
        [38.2946, 51.6963],  # Viewer left eye
        [73.5318, 51.6963],  # Viewer right eye
        [56.0252, 71.7366],  # Nose tip
    ],
    dtype=np.float32,
)


@dataclass
class ArcFaceConfig:
    """Configuration for ArcFace feature extractor."""
    model_path: str = "models/weights/w600k_r50.onnx"
    input_width: int = 112
    input_height: int = 112
    intra_op_threads: int = 2

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ArcFaceConfig":
        if not yaml_path.exists():
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        fv_data = data.get("face_verification", {})
        inp_size = fv_data.get("input_size", [112, 112])
        return cls(
            model_path=str(fv_data.get("model_path", "models/weights/w600k_r50.onnx")),
            input_width=int(inp_size[0]),
            input_height=int(inp_size[1]),
        )


class ArcFaceExtractor:
    """Extracts 512-dimensional L2-normalized identity embeddings using ONNX Runtime."""

    def __init__(self, config: Optional[ArcFaceConfig] = None) -> None:
        self.config = config or ArcFaceConfig()
        model_path = Path(self.config.model_path)

        if not model_path.is_absolute():
            # Resolve relative to project root
            root_dir = Path(__file__).resolve().parent.parent.parent
            model_path = root_dir / model_path

        if not model_path.exists():
            raise FileNotFoundError(
                f"ArcFace model weights not found at: {model_path}. "
                "Ensure w600k_r50.onnx is located in models/weights/."
            )

        # Configure optimized ONNX Runtime CPU session
        session_opts = ort.SessionOptions()
        session_opts.intra_op_num_threads = self.config.intra_op_threads
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info(
            "ArcFaceExtractor initialized from %s (input: %s, output: %s)",
            model_path.name,
            self.input_name,
            self.output_name,
        )

    def align_face(
        self,
        bgr_image: np.ndarray,
        bbox: BoundingBox,
        landmarks: Optional[List[Point2D]] = None,
    ) -> np.ndarray:
        """Align and crop face to 112x112 using similarity transform or aspect-preserved crop.
        
        Args:
            bgr_image: Source BGR image.
            bbox: Normalized face bounding box.
            landmarks: Optional facial keypoints (right eye, left eye, nose tip).
            
        Returns:
            Aligned BGR image of shape (112, 112, 3).
        """
        h, w = bgr_image.shape[:2]
        target_size = (self.config.input_width, self.config.input_height)

        # Attempt affine similarity transformation if at least 3 landmarks exist
        if landmarks and len(landmarks) >= 3:
            # MediaPipe order: 0: right eye (viewer left), 1: left eye (viewer right), 2: nose
            src_pts = np.array(
                [
                    landmarks[0].to_pixel(w, h),
                    landmarks[1].to_pixel(w, h),
                    landmarks[2].to_pixel(w, h),
                ],
                dtype=np.float32,
            )

            # Estimate similarity transform matrix (rotation, uniform scale, translation)
            matrix, inliers = cv2.estimateAffinePartial2D(
                src_pts,
                ARCFACE_REFERENCE_PTS_3,
                method=cv2.LMEDS,
            )

            if matrix is not None:
                aligned = cv2.warpAffine(
                    bgr_image,
                    matrix,
                    target_size,
                    borderMode=cv2.BORDER_REFLECT101,
                )
                return aligned

        # Fallback: Square-aspect padded crop around bounding box
        px_x, px_y, px_w, px_h = bbox.to_pixel(w, h)
        center_x = px_x + px_w / 2.0
        center_y = px_y + px_h / 2.0
        crop_size = max(px_w, px_h) * 1.2  # 20% margin around face

        x1 = int(round(center_x - crop_size / 2.0))
        y1 = int(round(center_y - crop_size / 2.0))
        x2 = int(round(center_x + crop_size / 2.0))
        y2 = int(round(center_y + crop_size / 2.0))

        # Clamp and pad if out of image bounds
        pad_x1 = max(0, -x1)
        pad_y1 = max(0, -y1)
        pad_x2 = max(0, x2 - w)
        pad_y2 = max(0, y2 - h)

        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(w, x2)
        src_y2 = min(h, y2)

        cropped = bgr_image[src_y1:src_y2, src_x1:src_x2]
        if pad_x1 > 0 or pad_y1 > 0 or pad_x2 > 0 or pad_y2 > 0:
            cropped = cv2.copyMakeBorder(
                cropped,
                pad_y1,
                pad_y2,
                pad_x1,
                pad_x2,
                borderType=cv2.BORDER_REFLECT101,
            )

        if cropped.size == 0:
            return np.zeros((self.config.input_height, self.config.input_width, 3), dtype=np.uint8)

        return cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)

    def extract_embedding(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Extract a 512-dimensional L2-normalized feature vector from an aligned 112x112 face.
        
        Args:
            aligned_bgr: Aligned face crop (112, 112, 3), BGR format.
            
        Returns:
            L2-normalized 1D float32 numpy array of shape (512,).
        """
        # Ensure correct dimensions
        if aligned_bgr.shape[:2] != (self.config.input_height, self.config.input_width):
            aligned_bgr = cv2.resize(
                aligned_bgr,
                (self.config.input_width, self.config.input_height),
                interpolation=cv2.INTER_LINEAR,
            )

        # Preprocessing: BGR -> RGB, normalize to [-1.0, 1.0]
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) - 127.5) / 127.5
        # HWC -> CHW -> NCHW (1, 3, 112, 112)
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        # ONNX Runtime forward pass
        raw_output = self.session.run(
            [self.output_name],
            {self.input_name: blob},
        )[0][0]

        # L2 normalization: e = v / ||v||_2
        norm = np.linalg.norm(raw_output)
        if norm > 1e-12:
            normalized_embedding = raw_output / norm
        else:
            normalized_embedding = np.zeros_like(raw_output)

        return normalized_embedding.astype(np.float32)

    @staticmethod
    def compute_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Compute cosine similarity between two L2-normalized embeddings.
        
        For unit vectors, cosine similarity equals the dot product.
        
        Returns:
            Cosine similarity score in range [-1.0, 1.0].
        """
        if embedding_a is None or embedding_b is None:
            return 0.0
        # Dot product of unit vectors
        similarity = float(np.dot(embedding_a, embedding_b))
        return max(-1.0, min(1.0, similarity))
