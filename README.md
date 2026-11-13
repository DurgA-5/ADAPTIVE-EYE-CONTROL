# ADAPTIVE-EYE-CONTROL

# AdaptiveEye: Intelligent Webcam-Based Eye-Controlled Computer System

> **AdaptiveEye: An Intelligent Webcam-Based Eye-Controlled Computer System with Adaptive Gaze Calibration, Confidence-Aware Cursor Stabilization, Biometric Security, and Hands-Free Gestures.**

---

## 1. Overview
AdaptiveEye is an advanced engineering and research initiative designed to provide seamless, hands-free human-computer interaction using an ordinary, consumer-grade laptop webcam. The system accurately estimates gaze position, stabilizes cursor movements, detects intentional blinks for system clicks, enables head-tilt scrolling, and dynamically adapts calibration while enforcing strict biometric identity verification to prevent unauthorized takeovers in multi-person environments.

---

## 2. Core Capabilities
- **Biometric Identity Lock & Multi-Person Tolerance:** Real-time multi-face detection and ArcFace deep feature verification ensures only the authorized user controls the cursor. Bystanders are highlighted in blue and ignored without locking the system.
- **Iris & 3D Landmark Mesh:** 478 sub-pixel landmark detection using MediaPipe Face Mesh with a 12-edge 3D perspective wireframe bounding box, surface point cloud, and green orientation axes.
- **Compound Head-Eye Coordination:** Fused ocular iris tracking (75%) and subtle 3D head pose orientation (25%) for effortless screen-wide cursor travel.
- **Adaptive Calibration:** Dynamic compensation for posture shifts, distance changes, and head motion with instant one-tap `[SPACE]` / `[C]` center calibration.
- **Confidence-Aware Stabilization:** Speed-adaptive 1€-style Exponential Moving Average and Kalman filtering for jitter-free fixation and instant saccadic response.
- **Intentional Blink & Double-Click Engine:** Discriminates between natural involuntary blinks (<220 ms) and intentional conscious single clicks (~220–650 ms), with a dedicated rapid double-blink state machine (<650 ms).
- **Head-Tilt Scrolling:** Effortless hands-free scrolling by tilting head UP ($\ge 11^\circ$) or DOWN ($\ge 11^\circ$).
- **Voice Speech & Audio Feedback:** Native spoken voice announcements ("Single Click", "Double Click", "Scroll Up", "Scroll Down") with audible confirmation chimes.
- **Zero-Trust Safety Interlocks:** Instant cursor freeze and failsafe disengagement upon identity loss or emergency corner override.

---

## 3. Directory Layout
```text
AdaptiveEye/
├── app/                       # Main application source modules
│   ├── core/                  # Base dataclasses, logging, voice announcer, safety controller
│   ├── camera/                # Webcam initialization, mirroring, and streaming
│   ├── face_detection/        # Multi-face detection pipeline
│   ├── face_verification/     # ArcFace verification, enrollment & hysteresis
│   ├── person_tracking/       # Identity lock & centroid matching
│   ├── eye_tracking/          # Landmark & iris localization with EAR calculation
│   ├── gaze_estimation/       # Vector gaze mapping, 3D head pose & wireframe overlay
│   ├── calibration/           # 9-point screen polynomial calibration
│   ├── adaptive_calibration/  # Real-time posture drift compensator
│   ├── stabilization/         # Speed-adaptive EMA and confidence Kalman filters
│   ├── cursor_control/        # Win32 OS mouse dispatch & ballistic acceleration
│   ├── blink_detection/       # Temporal blink FSM for single/double clicking
│   └── scrolling/             # Head-tilt continuous scrolling & edge engine
├── config/                    # YAML configuration files for all subsystems
├── data/                      # Enrolled biometric identity templates & datasets
├── models/weights/            # ArcFace ONNX model weights (w600k_r50.onnx)
└── tests/                     # Automated pytest test suite (120 unit tests)
```

---

## 4. Setup and Quickstart

### Prerequisites
- Windows 10/11 (64-bit)
- Python 3.10.x
- Integrated laptop webcam or external USB camera

### Installation
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Running the System
```powershell
# Launch full live real-time OS cursor control with blinking and head-tilt scrolling:
.\.venv\Scripts\python.exe main.py control-mouse

# Enroll authorized face identity:
.\.venv\Scripts\python.exe main.py enroll --name "DurgaPrasad"

# Run biometric verification:
.\.venv\Scripts\python.exe main.py verify

# Run 9-point screen calibration:
.\.venv\Scripts\python.exe main.py calibrate
```

### Live Keyboard Controls:
| Key | Function |
|---|---|
| **`[SPACE]`** or **`[C]`** | Lock center gaze calibration & center mouse |
| **`[M]`** | Toggle between Compound mode and Pure Iris mode |
| **`[F8]`** | Pause / Resume eye mouse control |
| **`[+]` / `[-]`** | Increase / Decrease reach sensitivity |
| **`[S]`** | Toggle scrolling ON / OFF |
| **`[Q]`** or **`[ESC]`** | Exit application cleanly |

---

## 5. Automated Verification Test Suite
```powershell
.\.venv\Scripts\pytest tests/ -v
```
*(All 120 unit tests pass with 100% success).*
