"""AdaptiveEye Polynomial Screen Calibration Mapper.

Maps raw gaze feature estimates (iris displacement + head pose) to true physical screen coordinates
using 2nd-degree polynomial ridge regression to compensate for ocular curvature, geometric distortions,
and user-monitor angle variations.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from app.core.logger import get_logger

logger = get_logger("CalibrationMapper")


@dataclass
class CalibrationQuality:
    """Quantitative performance metrics for fitted calibration model."""
    rmse_px: float          # Root Mean Square Error in pixels
    mean_px: float          # Mean absolute error in pixels
    max_px: float           # Worst-case error in pixels
    samples_count: int      # Total calibration observation pairs
    grade: str              # "EXCELLENT", "GOOD", "ACCEPTABLE", "POOR"
    trained_at: str         # ISO timestamp


class PolynomialGazeMapper:
    """Non-linear 2nd-degree polynomial mapping engine for gaze-to-screen transformation."""

    def __init__(
        self,
        degree: int = 2,
        regularization_alpha: float = 1e-3,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> None:
        self.degree = degree
        self.alpha = regularization_alpha
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.weights_x: Optional[np.ndarray] = None
        self.weights_y: Optional[np.ndarray] = None
        self.mean_x: float = 0.0
        self.std_x: float = 1.0
        self.mean_y: float = 0.0
        self.std_y: float = 1.0
        self.quality: Optional[CalibrationQuality] = None
        self.is_fitted: bool = False

    @staticmethod
    def expand_features(x: np.ndarray, y: np.ndarray, degree: int = 2) -> np.ndarray:
        """Construct polynomial feature design matrix for 2D inputs.
        
        Degree 1: [1, x, y]
        Degree 2: [1, x, y, x^2, y^2, x*y]
        Degree 3: [1, x, y, x^2, y^2, x*y, x^3, y^3, x^2*y, x*y^2]
        """
        x_flat = np.asarray(x, dtype=np.float64).flatten()
        y_flat = np.asarray(y, dtype=np.float64).flatten()
        n = len(x_flat)

        if degree == 1:
            return np.column_stack([np.ones(n), x_flat, y_flat])
        elif degree == 2:
            return np.column_stack([
                np.ones(n),
                x_flat,
                y_flat,
                x_flat ** 2,
                y_flat ** 2,
                x_flat * y_flat,
            ])
        elif degree == 3:
            return np.column_stack([
                np.ones(n),
                x_flat,
                y_flat,
                x_flat ** 2,
                y_flat ** 2,
                x_flat * y_flat,
                x_flat ** 3,
                y_flat ** 3,
                (x_flat ** 2) * y_flat,
                x_flat * (y_flat ** 2),
            ])
        else:
            raise ValueError(f"Unsupported polynomial degree: {degree}")

    def fit(
        self,
        raw_x: Union[List[float], np.ndarray],
        raw_y: Union[List[float], np.ndarray],
        target_x: Union[List[int], List[float], np.ndarray],
        target_y: Union[List[int], List[float], np.ndarray],
    ) -> CalibrationQuality:
        """Fit regularized polynomial ridge regression models for X and Y screen mappings.
        
        Solves: W = (Phi^T * Phi + alpha * I)^(-1) * Phi^T * Target
        """
        rx = np.asarray(raw_x, dtype=np.float64)
        ry = np.asarray(raw_y, dtype=np.float64)
        tx = np.asarray(target_x, dtype=np.float64)
        ty = np.asarray(target_y, dtype=np.float64)

        if len(rx) != len(ry) or len(rx) != len(tx) or len(rx) != len(ty):
            raise ValueError("All input arrays must have the same length.")

        min_samples = 6 if self.degree == 2 else 3
        if len(rx) < min_samples:
            raise ValueError(f"At least {min_samples} samples required to fit degree {self.degree} polynomial.")

        # Standardize features (Z-score normalization) to condition design matrix Gram kernel
        self.mean_x = float(np.mean(rx))
        std_x = float(np.std(rx))
        self.std_x = std_x if std_x > 1e-6 else 1.0

        self.mean_y = float(np.mean(ry))
        std_y = float(np.std(ry))
        self.std_y = std_y if std_y > 1e-6 else 1.0

        std_rx = (rx - self.mean_x) / self.std_x
        std_ry = (ry - self.mean_y) / self.std_y

        phi = self.expand_features(std_rx, std_ry, degree=self.degree)
        num_features = phi.shape[1]

        # Regularized Normal Equations (Ridge Regression)
        reg_matrix = self.alpha * np.eye(num_features)
        reg_matrix[0, 0] = 0.0  # Do not penalize constant bias term

        phi_t = phi.T
        kernel = phi_t @ phi + reg_matrix
        rhs_x = phi_t @ tx
        rhs_y = phi_t @ ty

        self.weights_x = np.linalg.solve(kernel, rhs_x)
        self.weights_y = np.linalg.solve(kernel, rhs_y)
        self.is_fitted = True

        # Calculate Residual Quality
        pred_x = phi @ self.weights_x
        pred_y = phi @ self.weights_y
        dist_errors = np.sqrt((pred_x - tx) ** 2 + (pred_y - ty) ** 2)

        rmse = float(np.sqrt(np.mean(dist_errors ** 2)))
        mean_err = float(np.mean(dist_errors))
        max_err = float(np.max(dist_errors))

        # Grade calibration accuracy against desktop ergonomics standard:
        # Standard screen: ~1920x1080 @ 60cm viewer distance (~30 px per degree of visual angle)
        if rmse < 45.0:
            grade = "EXCELLENT"      # < 1.5 degrees visual angle
        elif rmse < 80.0:
            grade = "GOOD"           # < 2.5 degrees visual angle
        elif rmse < 130.0:
            grade = "ACCEPTABLE"     # Suitable for coarse UI targeting
        else:
            grade = "POOR"

        self.quality = CalibrationQuality(
            rmse_px=round(rmse, 2),
            mean_px=round(mean_err, 2),
            max_px=round(max_err, 2),
            samples_count=len(rx),
            grade=grade,
            trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        logger.info(
            "Screen calibration fitted successfully: RMSE=%.2f px, Mean=%.2f px, Grade=%s",
            rmse,
            mean_err,
            grade,
        )
        return self.quality

    def predict(self, raw_x: float, raw_y: float) -> Tuple[int, int]:
        """Transform raw gaze observation into calibrated pixel screen coordinates."""
        if not self.is_fitted or self.weights_x is None or self.weights_y is None:
            # Fallback to direct linear scaling when uncalibrated
            sx = int(np.clip(round(raw_x * (self.screen_width - 1)), 0, self.screen_width - 1))
            sy = int(np.clip(round(raw_y * (self.screen_height - 1)), 0, self.screen_height - 1))
            return sx, sy

        norm_x = (raw_x - self.mean_x) / self.std_x
        norm_y = (raw_y - self.mean_y) / self.std_y

        phi = self.expand_features(np.array([norm_x]), np.array([norm_y]), degree=self.degree)
        px = float((phi @ self.weights_x)[0])
        py = float((phi @ self.weights_y)[0])

        # Strictly clamp within monitor pixel bounds
        clamped_x = int(np.clip(round(px), 0, self.screen_width - 1))
        clamped_y = int(np.clip(round(py), 0, self.screen_height - 1))
        return clamped_x, clamped_y

    def predict_normalized(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        """Transform raw gaze into normalized [0.0, 1.0] calibrated screen coordinates."""
        px, py = self.predict(raw_x, raw_y)
        nx = float(px / (self.screen_width - 1))
        ny = float(py / (self.screen_height - 1))
        return nx, ny

    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Persist calibration parameters and performance metadata to JSON."""
        if not self.is_fitted or self.weights_x is None or self.weights_y is None:
            raise RuntimeError("Cannot save uncalibrated model.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": "1.0",
            "degree": self.degree,
            "alpha": self.alpha,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "mean_x": self.mean_x,
            "std_x": self.std_x,
            "mean_y": self.mean_y,
            "std_y": self.std_y,
            "weights_x": self.weights_x.tolist(),
            "weights_y": self.weights_y.tolist(),
            "quality": asdict(self.quality) if self.quality else None,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("Saved screen calibration parameters to %s", path)

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> "PolynomialGazeMapper":
        """Restore fitted calibration model from JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Calibration profile not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mapper = cls(
            degree=int(data.get("degree", 2)),
            regularization_alpha=float(data.get("alpha", 1e-3)),
            screen_width=int(data.get("screen_width", 1920)),
            screen_height=int(data.get("screen_height", 1080)),
        )

        mapper.weights_x = np.array(data["weights_x"], dtype=np.float64)
        mapper.weights_y = np.array(data["weights_y"], dtype=np.float64)
        mapper.mean_x = float(data.get("mean_x", 0.0))
        mapper.std_x = float(data.get("std_x", 1.0))
        mapper.mean_y = float(data.get("mean_y", 0.0))
        mapper.std_y = float(data.get("std_y", 1.0))
        mapper.is_fitted = True

        q_data = data.get("quality")
        if q_data:
            mapper.quality = CalibrationQuality(
                rmse_px=float(q_data.get("rmse_px", 0.0)),
                mean_px=float(q_data.get("mean_px", 0.0)),
                max_px=float(q_data.get("max_px", 0.0)),
                samples_count=int(q_data.get("samples_count", 0)),
                grade=str(q_data.get("grade", "UNKNOWN")),
                trained_at=str(q_data.get("trained_at", "")),
            )

        logger.info(
            "Loaded screen calibration: degree=%d, RMSE=%.2f px, grade=%s",
            mapper.degree,
            mapper.quality.rmse_px if mapper.quality else 0.0,
            mapper.quality.grade if mapper.quality else "N/A",
        )
        return mapper
