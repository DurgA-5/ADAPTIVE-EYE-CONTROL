"""AdaptiveEye Camera Live Preview Utility.

Supports full-screen and windowed mirrored previews with live telemetry
HUD (FPS, resolution, timestamp, instructions) and safe keyboard exits.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional
import cv2

from app.camera.webcam import CameraConfig, CameraManager, CameraError
from app.core.logger import get_logger, setup_logging

logger = get_logger("CameraPreview")


def draw_hud(
    frame: cv2.Mat,
    fps: float,
    resolution: tuple[int, int],
    is_mirrored: bool,
    is_fullscreen: bool,
) -> cv2.Mat:
    """Render telemetry and control shortcuts onto the preview frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Top semi-transparent banner
    cv2.rectangle(canvas, (0, 0), (w, 40), (20, 20, 20), -1)
    cv2.addWeighted(canvas, 0.75, frame, 0.25, 0, canvas)

    # Info text
    hud_text = (
        f"AdaptiveEye Camera Preview | {resolution[0]}x{resolution[1]} | "
        f"FPS: {fps:5.1f} | Mirror: {'ON' if is_mirrored else 'OFF'} | "
        f"Mode: {'FULLSCREEN' if is_fullscreen else 'WINDOWED'}"
    )
    cv2.putText(
        canvas,
        hud_text,
        (15, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 200),
        1,
        cv2.LINE_AA,
    )

    # Bottom helper instructions banner
    cv2.rectangle(canvas, (0, h - 30), (w, h), (20, 20, 20), -1)
    help_text = "[Q / ESC] Quit  |  [F] Toggle Fullscreen  |  [M] Toggle Mirror"
    cv2.putText(
        canvas,
        help_text,
        (15, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    return canvas


def run_camera_preview(
    config_path: Optional[Path] = None,
    fullscreen: bool = False,
    device_id: Optional[int] = None,
) -> None:
    """Launch the interactive camera preview loop.
    
    Args:
        config_path: Path to camera YAML config file.
        fullscreen: Start preview in full-screen mode if True.
        device_id: Optional camera index override.
    """
    cfg_file = config_path or (
        Path(__file__).resolve().parent.parent.parent / "config" / "camera_config.yaml"
    )
    config = CameraConfig.from_yaml(cfg_file)
    if device_id is not None:
        config.device_id = device_id

    window_name = "AdaptiveEye - Camera Preview"
    is_fullscreen = fullscreen
    is_mirrored = config.mirror

    logger.info("Starting live preview on device %d...", config.device_id)

    try:
        with CameraManager(config) as cam:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            if is_fullscreen:
                cv2.setWindowProperty(
                    window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                )

            fps_timer = time.perf_counter()
            fps_counter = 0
            live_fps = float(config.fps)

            for raw_frame in cam.stream():
                fps_counter += 1
                now = time.perf_counter()
                elapsed = now - fps_timer
                if elapsed >= 0.5:
                    live_fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_timer = now

                # Display frame (CameraManager mirrors by default if configured)
                display_frame = raw_frame if is_mirrored == config.mirror else cv2.flip(raw_frame, 1)

                annotated = draw_hud(
                    display_frame,
                    live_fps,
                    cam.resolution,
                    is_mirrored,
                    is_fullscreen,
                )

                cv2.imshow(window_name, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                    logger.info("Exit requested via keyboard shortcut.")
                    break
                elif key in (ord("f"), ord("F")):
                    is_fullscreen = not is_fullscreen
                    prop = (
                        cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                    )
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
                elif key in (ord("m"), ord("M")):
                    is_mirrored = not is_mirrored
                    logger.info("Mirroring set to: %s", is_mirrored)

    except CameraError as e:
        logger.error("Camera preview failed: %s", e)
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        logger.info("Preview closed and display windows destroyed.")


if __name__ == "__main__":
    setup_logging(level="INFO")
    parser = argparse.ArgumentParser(description="AdaptiveEye Camera Preview")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode")
    parser.add_argument("--device", type=int, default=None, help="Camera device index")
    args = parser.parse_args()

    run_camera_preview(fullscreen=args.fullscreen, device_id=args.device)
