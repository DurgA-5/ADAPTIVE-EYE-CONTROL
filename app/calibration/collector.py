"""AdaptiveEye Interactive Screen Calibration Session & Target Collector.

Executes a 9-point full-screen calibration routine with animated fixation rings,
saccadic latency discard filtering, least-squares polynomial surface fitting,
and live visual accuracy verification.
"""

import ctypes
from dataclasses import dataclass
from pathlib import Path
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np
import yaml

from app.camera.webcam import CameraConfig, CameraManager
from app.calibration.mapper import CalibrationQuality, PolynomialGazeMapper
from app.core.datatypes import Point2D
from app.core.logger import get_logger
from app.eye_tracking.tracker import EyeTracker, EyeTrackingConfig
from app.gaze_estimation.estimator import GazeConfig, GazeEstimator

logger = get_logger("ScreenCalibration")


def get_primary_monitor_resolution() -> Tuple[int, int]:
    """Retrieve primary monitor resolution with high-DPI awareness on Windows."""
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        w = int(user32.GetSystemMetrics(0))
        h = int(user32.GetSystemMetrics(1))
        if w > 0 and h > 0:
            return w, h
    except Exception as exc:
        logger.warning("Could not query Windows user32 metrics (%s); using default 1920x1080", exc)
    return 1920, 1080


@dataclass
class CalibrationSessionConfig:
    """Configurable parameters for interactive calibration."""
    grid_points: int = 9
    dwell_time_sec: float = 2.0
    settle_time_sec: float = 0.6
    samples_per_point: int = 30
    polynomial_degree: int = 2
    regularization_alpha: float = 0.001
    margin_x: float = 0.10
    margin_y: float = 0.10
    target_radius: int = 28
    calibration_file: Path = Path("data/calibration/screen_calibration.json")

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "CalibrationSessionConfig":
        if not yaml_path.exists():
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        c_data = data.get("calibration", {})
        return cls(
            grid_points=int(c_data.get("grid_points", 9)),
            dwell_time_sec=float(c_data.get("dwell_time_sec", 2.0)),
            settle_time_sec=float(c_data.get("settle_time_sec", 0.6)),
            samples_per_point=int(c_data.get("samples_per_point", 30)),
            polynomial_degree=int(c_data.get("polynomial_degree", 2)),
            regularization_alpha=float(c_data.get("regularization_alpha", 0.001)),
            margin_x=float(c_data.get("margin_x", 0.10)),
            margin_y=float(c_data.get("margin_y", 0.10)),
            target_radius=int(c_data.get("target_radius", 28)),
            calibration_file=Path(c_data.get("calibration_file", "data/calibration/screen_calibration.json")),
        )


class ScreenCalibrationSession:
    """Interactive full-screen 9-point gaze calibration engine."""

    def __init__(self, config: Optional[CalibrationSessionConfig] = None) -> None:
        self.config = config or CalibrationSessionConfig()
        self.screen_width, self.screen_height = get_primary_monitor_resolution()
        self.window_name = "AdaptiveEye - Fullscreen Calibration"

        # Hardware & inference subsystems
        self.camera = CameraManager(CameraConfig(width=1280, height=720, mirror=True))
        self.eye_tracker = EyeTracker(EyeTrackingConfig(refine_landmarks=True))
        self.gaze_estimator = GazeEstimator(
            GazeConfig(
                screen_width=self.screen_width,
                screen_height=self.screen_height,
            )
        )

        self.targets: List[Tuple[int, int]] = self._generate_target_grid()
        self.raw_samples_x: List[float] = []
        self.raw_samples_y: List[float] = []
        self.target_samples_x: List[int] = []
        self.target_samples_y: List[int] = []

    def _generate_target_grid(self) -> List[Tuple[int, int]]:
        """Generate smooth serpentine (boustrophedon) 9-point grid to minimize eye fatigue."""
        mx = self.config.margin_x
        my = self.config.margin_y

        xs = [int(mx * self.screen_width), int(0.50 * self.screen_width), int((1.0 - mx) * self.screen_width)]
        ys = [int(my * self.screen_height), int(0.50 * self.screen_height), int((1.0 - my) * self.screen_height)]

        # Row 1: Left -> Right
        row1 = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[2], ys[0])]
        # Row 2: Right -> Left (serpentine return)
        row2 = [(xs[2], ys[1]), (xs[1], ys[1]), (xs[0], ys[1])]
        # Row 3: Left -> Right
        row3 = [(xs[0], ys[2]), (xs[1], ys[2]), (xs[2], ys[2])]

        return row1 + row2 + row3

    def _draw_instruction_screen(self) -> np.ndarray:
        """Render welcoming instruction screen."""
        canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        canvas[:] = (20, 22, 26)  # Dark slate background

        cx = self.screen_width // 2
        cy = self.screen_height // 2

        cv2.putText(
            canvas,
            "AdaptiveEye 9-Point Screen Calibration",
            (cx - 380, cy - 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )

        instructions = [
            "1. Sit at a comfortable distance (~50-65 cm) directly facing your screen.",
            "2. Keep your head relatively still during calibration.",
            "3. Look steadily at each target dot as the outer ring shrinks into it.",
            "4. Follow the dot as it moves across all 9 screen positions.",
            "",
            "Press [SPACE] or [ENTER] to Begin Calibration",
            "Press [Q] or [ESC] to Exit",
        ]

        for i, text in enumerate(instructions):
            color = (255, 255, 255) if i < 4 else (0, 220, 255)
            scale = 0.65 if i < 4 else 0.75
            thickness = 1 if i < 4 else 2
            cv2.putText(
                canvas,
                text,
                (cx - 360, cy - 60 + i * 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        return canvas

    def _draw_target_canvas(
        self,
        target_idx: int,
        target_pt: Tuple[int, int],
        elapsed_sec: float,
        is_settling: bool,
    ) -> np.ndarray:
        """Render animated calibration stimulus canvas."""
        canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        canvas[:] = (18, 20, 24)

        tx, ty = target_pt
        total_time = self.config.dwell_time_sec
        progress = min(max(elapsed_sec / total_time, 0.0), 1.0)

        # Shrinking animated ring radius
        max_r = int(self.config.target_radius * 2.2)
        min_r = int(self.config.target_radius * 0.4)
        ring_r = int(max_r - progress * (max_r - min_r))

        if is_settling:
            ring_color = (200, 200, 200)
            status_text = "Fixating..."
            status_color = (180, 180, 180)
        else:
            ring_color = (0, 255, 255)  # Bright cyan/yellow during recording
            status_text = "Recording Eye Data..."
            status_color = (0, 255, 0)

        # Draw outer subtle guideline ring
        cv2.circle(canvas, (tx, ty), max_r, (45, 50, 55), 1, cv2.LINE_AA)

        # Draw animated shrinking fixation ring
        cv2.circle(canvas, (tx, ty), ring_r, ring_color, 2, cv2.LINE_AA)

        # Center target dot with white core
        cv2.circle(canvas, (tx, ty), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty), 2, (255, 255, 255), -1, cv2.LINE_AA)

        # Header status bar
        header_text = f"Point {target_idx + 1} of {len(self.targets)} | {status_text}"
        cv2.putText(
            canvas,
            header_text,
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA,
        )

        # Footer progress bar
        bar_w = 400
        bar_h = 10
        bar_x = (self.screen_width - bar_w) // 2
        bar_y = self.screen_height - 50
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * ((target_idx + progress) / len(self.targets)))
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 200), -1)

        return canvas

    def _draw_summary_screen(self, quality: CalibrationQuality) -> np.ndarray:
        """Render calibration results and accuracy grade screen."""
        canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        canvas[:] = (20, 22, 26)

        cx = self.screen_width // 2
        cy = self.screen_height // 2

        cv2.putText(
            canvas,
            "Calibration Complete",
            (cx - 200, cy - 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )

        grade_colors = {
            "EXCELLENT": (0, 255, 0),
            "GOOD": (0, 255, 200),
            "ACCEPTABLE": (0, 220, 255),
            "POOR": (0, 0, 255),
        }
        g_color = grade_colors.get(quality.grade, (255, 255, 255))

        stats = [
            ("Accuracy Grade:", f"{quality.grade}", g_color),
            ("Root Mean Square Error (RMSE):", f"{quality.rmse_px:.1f} pixels", (255, 255, 255)),
            ("Mean Absolute Error:", f"{quality.mean_px:.1f} pixels", (255, 255, 255)),
            ("Peak Deviation:", f"{quality.max_px:.1f} pixels", (255, 255, 255)),
            ("Total Samples Collected:", f"{quality.samples_count} frames", (255, 255, 255)),
        ]

        for i, (label, val, col) in enumerate(stats):
            y = cy - 60 + i * 38
            cv2.putText(canvas, label, (cx - 260, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(canvas, val, (cx + 80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)

        actions = [
            "Press [ENTER] to Save & Apply Calibration",
            "Press [R] to Recalibrate",
            "Press [ESC] to Discard & Exit",
        ]
        for i, action in enumerate(actions):
            cv2.putText(
                canvas,
                action,
                (cx - 220, cy + 160 + i * 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )

        return canvas

    def run(self) -> Optional[PolynomialGazeMapper]:
        """Execute full calibration workflow."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.camera.start()
        try:
            # 1. Instruction Phase
            while True:
                instruction_frame = self._draw_instruction_screen()
                cv2.imshow(self.window_name, instruction_frame)
                key = cv2.waitKey(20) & 0xFF
                if key in (32, 13):  # SPACE or ENTER
                    break
                elif key in (27, ord("q"), ord("Q")):
                    logger.info("Calibration cancelled by user.")
                    return None

            # 2. Target Collection Loop
            self.raw_samples_x.clear()
            self.raw_samples_y.clear()
            self.target_samples_x.clear()
            self.target_samples_y.clear()

            stream_iter = self.camera.stream()

            for target_idx, target_pt in enumerate(self.targets):
                target_start = time.perf_counter()
                point_samples_count = 0

                while True:
                    frame = next(stream_iter)
                    now = time.perf_counter()
                    elapsed = now - target_start
                    is_settling = elapsed < self.config.settle_time_sec

                    # Inference pass
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame.flags.writeable = False
                    mesh_res = self.eye_tracker.mesh.process(rgb_frame)

                    all_lms = None
                    if mesh_res.multi_face_landmarks:
                        all_lms = [
                            Point2D(x=float(lm.x), y=float(lm.y))
                            for lm in mesh_res.multi_face_landmarks[0].landmark
                        ]

                    eye_feats = self.eye_tracker.process(frame)
                    gaze_res = self.gaze_estimator.estimate(frame, eye_feats, all_lms)

                    # During recording phase, record valid gaze features
                    if (
                        not is_settling
                        and gaze_res.is_valid
                        and point_samples_count < self.config.samples_per_point
                    ):
                        self.raw_samples_x.append(gaze_res.raw_x)
                        self.raw_samples_y.append(gaze_res.raw_y)
                        self.target_samples_x.append(target_pt[0])
                        self.target_samples_y.append(target_pt[1])
                        point_samples_count += 1

                    # Render stimulus
                    target_canvas = self._draw_target_canvas(
                        target_idx, target_pt, elapsed, is_settling
                    )
                    cv2.imshow(self.window_name, target_canvas)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        logger.info("Calibration aborted mid-session by user.")
                        return None

                    if elapsed >= self.config.dwell_time_sec:
                        break

            # 3. Model Fitting
            if len(self.raw_samples_x) < 9:
                logger.error("Insufficient samples collected during calibration (%d).", len(self.raw_samples_x))
                return None

            mapper = PolynomialGazeMapper(
                degree=self.config.polynomial_degree,
                regularization_alpha=self.config.regularization_alpha,
                screen_width=self.screen_width,
                screen_height=self.screen_height,
            )

            quality = mapper.fit(
                self.raw_samples_x,
                self.raw_samples_y,
                self.target_samples_x,
                self.target_samples_y,
            )

            # 4. Summary & Confirmation
            while True:
                summary_canvas = self._draw_summary_screen(quality)
                cv2.imshow(self.window_name, summary_canvas)
                key = cv2.waitKey(20) & 0xFF
                if key == 13:  # ENTER -> Save
                    mapper.save_to_file(self.config.calibration_file)
                    print(f"\nScreen calibration successfully saved to {self.config.calibration_file}")
                    return mapper
                elif key in (ord("r"), ord("R")):
                    logger.info("User requested recalibration. Restarting...")
                    return self.run()
                elif key in (27, ord("q"), ord("Q")):
                    logger.info("User discarded calibration results.")
                    return None

        finally:
            self.camera.stop()
            self.eye_tracker.close()
            cv2.destroyAllWindows()


def run_screen_calibration() -> Optional[PolynomialGazeMapper]:
    """CLI helper to launch interactive screen calibration session."""
    cfg_file = Path("config/calibration_config.yaml")
    config = CalibrationSessionConfig.from_yaml(cfg_file)
    session = ScreenCalibrationSession(config)
    return session.run()


def run_screen_calibration_verification() -> None:
    """Launch real-time full-screen visual calibration verification sandbox."""
    cfg_file = Path("config/calibration_config.yaml")
    config = CalibrationSessionConfig.from_yaml(cfg_file)
    cal_file = config.calibration_file

    if not cal_file.exists():
        print(f"\n[!] No calibration file found at {cal_file}.")
        print("Please run calibration first via: python main.py calibrate\n")
        return

    mapper = PolynomialGazeMapper.load_from_file(cal_file)
    screen_w, screen_h = get_primary_monitor_resolution()

    cam = CameraManager(CameraConfig(width=1280, height=720, mirror=True))
    eye_tracker = EyeTracker(EyeTrackingConfig(refine_landmarks=True))
    gaze_estimator = GazeEstimator(GazeConfig(screen_width=screen_w, screen_height=screen_h))

    window_name = "AdaptiveEye - Calibrated Gaze Verification (Look Around Screen - Press Q to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\nStarting Calibrated Gaze Verification sandbox.")
    print("Move your eyes across the screen. The cyan reticle should track your gaze accurately.")
    print("Press [Q] or [ESC] to exit.\n")

    try:
        cam.start()
        for frame in cam.stream():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            mesh_res = eye_tracker.mesh.process(rgb)

            all_lms = None
            if mesh_res.multi_face_landmarks:
                all_lms = [
                    Point2D(x=float(lm.x), y=float(lm.y))
                    for lm in mesh_res.multi_face_landmarks[0].landmark
                ]

            eye_feats = eye_tracker.process(frame)
            raw_gaze = gaze_estimator.estimate(frame, eye_feats, all_lms)

            # Canvas for fullscreen display
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            canvas[:] = (20, 22, 26)

            # Draw reference 9 target points as subtle guides
            mx, my = config.margin_x, config.margin_y
            xs = [int(mx * screen_w), int(0.50 * screen_w), int((1.0 - mx) * screen_w)]
            ys = [int(my * screen_h), int(0.50 * screen_h), int((1.0 - my) * screen_h)]
            for gx in xs:
                for gy in ys:
                    cv2.circle(canvas, (gx, gy), 10, (50, 55, 60), 1, cv2.LINE_AA)

            if raw_gaze.is_valid:
                # Transform through calibrated polynomial surface
                cal_x, cal_y = mapper.predict(raw_gaze.raw_x, raw_gaze.raw_y)

                # Render calibrated gaze cursor reticle
                cv2.circle(canvas, (cal_x, cal_y), 24, (0, 255, 200), 2, cv2.LINE_AA)
                cv2.circle(canvas, (cal_x, cal_y), 6, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.drawMarker(canvas, (cal_x, cal_y), (0, 255, 200), cv2.MARKER_CROSS, 32, 1, cv2.LINE_AA)

                coord_text = f"Calibrated Point: ({cal_x}, {cal_y}) | Raw Gaze: ({raw_gaze.normalized_x:.2f}, {raw_gaze.normalized_y:.2f})"
                status_color = (0, 255, 200)
            else:
                coord_text = "GAZE PAUSED (EYES CLOSED OR OCCLUDED)"
                status_color = (0, 0, 255)

            # Top overlay
            cv2.putText(canvas, coord_text, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"Model: Degree {mapper.degree} Polynomial | Quality: {mapper.quality.grade if mapper.quality else 'Loaded'} (RMSE: {mapper.quality.rmse_px if mapper.quality else 0:.1f}px)",
                (40, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

    finally:
        cam.stop()
        eye_tracker.close()
        cv2.destroyAllWindows()
