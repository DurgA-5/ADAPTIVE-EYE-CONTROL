"""Module 8 Screen Calibration Unit Tests."""

from pathlib import Path
import numpy as np
import pytest

from app.calibration.collector import (
    CalibrationQuality,
    CalibrationSessionConfig,
    ScreenCalibrationSession,
)
from app.calibration.mapper import PolynomialGazeMapper


def test_calibration_config_defaults():
    """Verify default calibration configuration parameters."""
    cfg = CalibrationSessionConfig()
    assert cfg.grid_points == 9
    assert cfg.dwell_time_sec == 2.0
    assert cfg.settle_time_sec == 0.6
    assert cfg.samples_per_point == 30
    assert cfg.polynomial_degree == 2
    assert cfg.regularization_alpha == 0.001
    assert cfg.margin_x == 0.10
    assert cfg.margin_y == 0.10


def test_calibration_config_from_yaml(tmp_path: Path):
    """Verify loading calibration config from YAML file."""
    yaml_content = """
calibration:
  grid_points: 5
  dwell_time_sec: 1.8
  settle_time_sec: 0.5
  samples_per_point: 25
  polynomial_degree: 2
  regularization_alpha: 0.002
  margin_x: 0.15
  margin_y: 0.12
  target_radius: 30
  calibration_file: "data/calibration/test_calib.json"
"""
    cfg_file = tmp_path / "custom_calib.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    cfg = CalibrationSessionConfig.from_yaml(cfg_file)
    assert cfg.grid_points == 5
    assert cfg.dwell_time_sec == 1.8
    assert cfg.settle_time_sec == 0.5
    assert cfg.samples_per_point == 25
    assert cfg.margin_x == 0.15
    assert cfg.margin_y == 0.12
    assert cfg.target_radius == 30
    assert cfg.calibration_file == Path("data/calibration/test_calib.json")


def test_polynomial_feature_expansion():
    """Verify polynomial design matrix shapes for degrees 1, 2, and 3."""
    x = np.array([0.2, 0.5])
    y = np.array([0.3, 0.7])

    # Degree 1: [1, x, y] -> 3 features
    phi_1 = PolynomialGazeMapper.expand_features(x, y, degree=1)
    assert phi_1.shape == (2, 3)
    np.testing.assert_allclose(phi_1[0], [1.0, 0.2, 0.3])

    # Degree 2: [1, x, y, x^2, y^2, xy] -> 6 features
    phi_2 = PolynomialGazeMapper.expand_features(x, y, degree=2)
    assert phi_2.shape == (2, 6)
    np.testing.assert_allclose(phi_2[0], [1.0, 0.2, 0.3, 0.04, 0.09, 0.06])

    # Degree 3: 10 features
    phi_3 = PolynomialGazeMapper.expand_features(x, y, degree=3)
    assert phi_3.shape == (2, 10)


def test_polynomial_fit_exact_linear_synthetic():
    """Verify ridge regression polynomial solver on ideal linear screen transformation."""
    mapper = PolynomialGazeMapper(degree=2, regularization_alpha=1e-6, screen_width=1920, screen_height=1080)

    # 9 synthetic grid points in normalized gaze space [0.1 .. 0.9]
    gx = np.array([0.1, 0.5, 0.9, 0.1, 0.5, 0.9, 0.1, 0.5, 0.9])
    gy = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9])
    tx = gx * 1919.0
    ty = gy * 1079.0

    quality = mapper.fit(gx, gy, tx, ty)

    assert mapper.is_fitted is True
    assert quality.rmse_px < 1.0  # Perfect linear mapping should have < 1px error
    assert quality.grade == "EXCELLENT"

    # Test prediction on midpoint
    pred_x, pred_y = mapper.predict(0.5, 0.5)
    assert pred_x == pytest.approx(int(0.5 * 1919.0), abs=2)
    assert pred_y == pytest.approx(int(0.5 * 1079.0), abs=2)


def test_polynomial_fit_quadratic_curvature():
    """Verify quadratic terms capture non-linear distortion."""
    mapper = PolynomialGazeMapper(degree=2, regularization_alpha=1e-5, screen_width=1920, screen_height=1080)

    # Grid with synthetic non-linear barrel distortion
    gx = np.linspace(0.1, 0.9, 5)
    gy = np.linspace(0.1, 0.9, 5)
    GX, GY = np.meshgrid(gx, gy)
    gx_flat = GX.flatten()
    gy_flat = GY.flatten()

    tx = gx_flat * 1900.0 + 30.0 * (gx_flat ** 2)
    ty = gy_flat * 1050.0 + 20.0 * (gy_flat ** 2)

    quality = mapper.fit(gx_flat, gy_flat, tx, ty)
    assert quality.rmse_px < 5.0
    assert quality.grade == "EXCELLENT"


def test_polynomial_mapper_insufficient_samples():
    """Mapper should raise ValueError when fewer than minimum required points are provided."""
    mapper = PolynomialGazeMapper(degree=2)
    with pytest.raises(ValueError, match="At least 6 samples required"):
        mapper.fit([0.1, 0.2], [0.1, 0.2], [100, 200], [100, 200])


def test_polynomial_mapper_prediction_clamping():
    """Mapper prediction must never exceed screen pixel boundaries [0, W-1], [0, H-1]."""
    mapper = PolynomialGazeMapper(degree=2, screen_width=1920, screen_height=1080)
    gx = np.linspace(0.2, 0.8, 9)
    gy = np.linspace(0.2, 0.8, 9)
    tx = gx * 1919.0
    ty = gy * 1079.0
    mapper.fit(gx, gy, tx, ty)

    # Extreme out-of-screen raw gaze inputs
    pred_low = mapper.predict(-2.0, -2.0)
    assert pred_low[0] >= 0
    assert pred_low[1] >= 0

    pred_high = mapper.predict(5.0, 5.0)
    assert pred_high[0] <= 1919
    assert pred_high[1] <= 1079

    # Normalized prediction bounds
    norm_x, norm_y = mapper.predict_normalized(5.0, 5.0)
    assert 0.0 <= norm_x <= 1.0
    assert 0.0 <= norm_y <= 1.0


def test_polynomial_mapper_unfitted_fallback():
    """Unfitted mapper should gracefully fall back to linear scaling without crashing."""
    mapper = PolynomialGazeMapper(screen_width=1920, screen_height=1080)
    assert mapper.is_fitted is False

    pred_x, pred_y = mapper.predict(0.5, 0.5)
    assert pred_x == int(round(0.5 * (1920 - 1)))
    assert pred_y == int(round(0.5 * (1080 - 1)))


def test_calibration_serialization_roundtrip(tmp_path: Path):
    """Test saving and restoring calibration model to/from JSON."""
    mapper = PolynomialGazeMapper(degree=2, regularization_alpha=0.005, screen_width=1920, screen_height=1080)
    gx = np.array([0.1, 0.5, 0.9, 0.1, 0.5, 0.9, 0.1, 0.5, 0.9])
    gy = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9])
    tx = gx * 1919.0
    ty = gy * 1079.0
    mapper.fit(gx, gy, tx, ty)

    save_path = tmp_path / "screen_calib.json"
    mapper.save_to_file(save_path)
    assert save_path.exists()

    # Restore from file
    loaded = PolynomialGazeMapper.load_from_file(save_path)
    assert loaded.is_fitted is True
    assert loaded.degree == mapper.degree
    assert loaded.alpha == mapper.alpha
    assert loaded.screen_width == mapper.screen_width
    assert loaded.screen_height == mapper.screen_height
    np.testing.assert_allclose(loaded.weights_x, mapper.weights_x)
    np.testing.assert_allclose(loaded.weights_y, mapper.weights_y)
    assert loaded.quality.grade == mapper.quality.grade

    # Verify predictions match exactly
    orig_p = mapper.predict(0.42, 0.68)
    load_p = loaded.predict(0.42, 0.68)
    assert orig_p == load_p


def test_calibration_session_target_grid_generation():
    """Verify 9-point grid generates 9 distinct target positions within screen margins."""
    cfg = CalibrationSessionConfig(grid_points=9, margin_x=0.10, margin_y=0.10)
    session = ScreenCalibrationSession(cfg)

    targets = session.targets
    assert len(targets) == 9
    assert len(set(targets)) == 9  # All unique

    # Check margin bounding
    for tx, ty in targets:
        assert tx >= int(0.10 * session.screen_width)
        assert tx <= int(0.90 * session.screen_width)
        assert ty >= int(0.10 * session.screen_height)
        assert ty <= int(0.90 * session.screen_height)


def test_calibration_session_canvas_rendering():
    """Verify drawing session canvases generates valid non-empty OpenCV frames."""
    session = ScreenCalibrationSession()

    # 1. Instruction screen
    instr = session._draw_instruction_screen()
    assert instr.shape == (session.screen_height, session.screen_width, 3)
    assert np.any(instr > 0)

    # 2. Target canvas
    target_canvas = session._draw_target_canvas(
        target_idx=0,
        target_pt=(session.screen_width // 2, session.screen_height // 2),
        elapsed_sec=0.8,
        is_settling=False,
    )
    assert target_canvas.shape == (session.screen_height, session.screen_width, 3)
    assert np.any(target_canvas > 0)

    # 3. Summary screen
    dummy_quality = CalibrationQuality(
        rmse_px=32.4,
        mean_px=26.1,
        max_px=45.0,
        samples_count=270,
        grade="EXCELLENT",
        trained_at="2026-11-13 15:00:00",
    )
    summary = session._draw_summary_screen(dummy_quality)
    assert summary.shape == (session.screen_height, session.screen_width, 3)
    assert np.any(summary > 0)
