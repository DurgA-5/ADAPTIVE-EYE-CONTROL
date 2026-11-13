"""AdaptiveEye Stabilization Filters.

Implements speed-adaptive Exponential Moving Average (1€-style filter) and confidence-weighted
2D Kalman filter for suppressing involuntary ocular tremors and micro-saccades during fixations
while preserving instantaneous responsiveness during intentional saccades.
"""

from dataclasses import dataclass
import time
from typing import Optional, Tuple
import numpy as np

from app.core.logger import get_logger

logger = get_logger("StabilizationFilters")


@dataclass
class FilterResult:
    """Output state of a 2D smoothing filter."""
    x: float
    y: float
    velocity_px_s: float
    alpha: float
    is_deadzone_active: bool


class AdaptiveEMAFilter:
    """Speed-adaptive Exponential Moving Average filter with deadzone suppression."""

    def __init__(
        self,
        min_alpha: float = 0.08,
        max_alpha: float = 0.85,
        saccade_velocity_threshold: float = 480.0,
        deadzone_radius_px: float = 6.0,
    ) -> None:
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.saccade_velocity_threshold = max(1.0, saccade_velocity_threshold)
        self.deadzone_radius_px = deadzone_radius_px

        self._smoothed_x: Optional[float] = None
        self._smoothed_y: Optional[float] = None
        self._last_time: Optional[float] = None
        self._last_velocity: float = 0.0

    def reset(self) -> None:
        """Reset internal filter state."""
        self._smoothed_x = None
        self._smoothed_y = None
        self._last_time = None
        self._last_velocity = 0.0

    def filter(
        self,
        raw_x: float,
        raw_y: float,
        timestamp: Optional[float] = None,
    ) -> FilterResult:
        """Process incoming raw coordinate and return stabilized coordinate."""
        now = timestamp if timestamp is not None else time.perf_counter()

        # First sample initialization
        if self._smoothed_x is None or self._smoothed_y is None or self._last_time is None:
            self._smoothed_x = float(raw_x)
            self._smoothed_y = float(raw_y)
            self._last_time = now
            self._last_velocity = 0.0
            return FilterResult(
                x=float(raw_x),
                y=float(raw_y),
                velocity_px_s=0.0,
                alpha=self.min_alpha,
                is_deadzone_active=False,
            )

        dt = max(1e-4, now - self._last_time)
        self._last_time = now

        # Displacement from current smoothed position
        dx = raw_x - self._smoothed_x
        dy = raw_y - self._smoothed_y
        dist = float(np.hypot(dx, dy))

        # Deadzone check: If movement is tiny (< deadzone_radius), lock output to eliminate tremor
        if dist < self.deadzone_radius_px:
            return FilterResult(
                x=self._smoothed_x,
                y=self._smoothed_y,
                velocity_px_s=0.0,
                alpha=self.min_alpha,
                is_deadzone_active=True,
            )

        # Instantaneous velocity in pixels per second
        inst_velocity = dist / dt
        # Smooth velocity estimate to avoid instantaneous spikes
        velocity = 0.7 * self._last_velocity + 0.3 * inst_velocity
        self._last_velocity = velocity

        # Non-linear quadratic speed modulation between min_alpha and max_alpha
        velocity_ratio = min(max(velocity / self.saccade_velocity_threshold, 0.0), 1.0)
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * (velocity_ratio ** 2)

        # Update smoothed position
        self._smoothed_x = (1.0 - alpha) * self._smoothed_x + alpha * raw_x
        self._smoothed_y = (1.0 - alpha) * self._smoothed_y + alpha * raw_y

        return FilterResult(
            x=self._smoothed_x,
            y=self._smoothed_y,
            velocity_px_s=round(velocity, 1),
            alpha=round(alpha, 3),
            is_deadzone_active=False,
        )


class ConfidenceKalmanFilter:
    """Discrete 2D Constant-Velocity Kalman Filter with confidence-adaptive measurement covariance."""

    def __init__(
        self,
        process_noise: float = 0.08,
        measurement_noise_base: float = 1.2,
        confidence_multiplier: float = 12.0,
    ) -> None:
        self.process_noise = process_noise
        self.r_base = measurement_noise_base
        self.conf_multiplier = confidence_multiplier

        # State vector: [x, y, vx, vy]^T
        self.x = np.zeros(4, dtype=np.float64)
        # Covariance matrix P
        self.P = np.eye(4, dtype=np.float64) * 50.0
        # Measurement matrix H
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

        self._is_initialized = False
        self._last_time: Optional[float] = None

    def reset(self) -> None:
        """Reset Kalman state and covariance."""
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 50.0
        self._is_initialized = False
        self._last_time = None

    def update(
        self,
        meas_x: float,
        meas_y: float,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Perform Time Prediction and Confidence-Weighted Measurement Update.
        
        Returns:
            Tuple of (estimated_x, estimated_y, estimated_speed_px_s).
        """
        now = timestamp if timestamp is not None else time.perf_counter()

        if not self._is_initialized:
            self.x = np.array([meas_x, meas_y, 0.0, 0.0], dtype=np.float64)
            self._is_initialized = True
            self._last_time = now
            return meas_x, meas_y, 0.0

        dt = max(1e-4, now - (self._last_time or now))
        self._last_time = now

        # 1. State Transition Matrix F
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # 2. Process Noise Covariance Q
        q_pos = 0.5 * (dt ** 2) * self.process_noise
        q_vel = dt * self.process_noise
        Q = np.diag([q_pos, q_pos, q_vel, q_vel]) ** 2

        # 3. Time Update (Predict)
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        # 4. Confidence-Weighted Measurement Noise Covariance R
        # Low confidence -> High measurement uncertainty R -> Trust model velocity
        conf_clamped = min(max(confidence, 0.0), 1.0)
        r_scale = 1.0 + self.conf_multiplier * ((1.0 - conf_clamped) ** 2)
        R = np.eye(2, dtype=np.float64) * (self.r_base * r_scale)

        # 5. Measurement Update (Correct)
        z = np.array([meas_x, meas_y], dtype=np.float64)
        y_residual = z - (self.H @ x_pred)
        S = self.H @ P_pred @ self.H.T + R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + (K @ y_residual)
        self.P = (np.eye(4) - K @ self.H) @ P_pred

        speed = float(np.hypot(self.x[2], self.x[3]))
        return float(self.x[0]), float(self.x[1]), speed
