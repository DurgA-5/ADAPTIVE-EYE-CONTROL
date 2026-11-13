# AdaptiveEye System Architecture

## 1. System Pipeline Architecture

```text
       ┌────────────────────────┐
       │   Webcam Video Stream  │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   Lighting Adjustment  │  (CLAHE / Dynamic Brightness & Contrast)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │     Face Detection     │  (MediaPipe Face Detection)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   Identity Lock & Auth │  (ArcFace 512-d Embedding Verification)
       └───────────┬────────────┘
                   │ Authorized?
          No ──────┴────── Yes
          │                 │
          ▼                 ▼
   [PAUSE / LOCK]  ┌────────────────────────┐
                   │ Eye Landmark / Iris    │  (MediaPipe 478-pt Mesh)
                   └───────────┬────────────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │    Gaze Estimation     │  (Iris Displacement & Head Pose)
                   └───────────┬────────────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │  Adaptive Calibration  │  (9-point Polynomial / Homography)
                   └───────────┬────────────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │  Cursor Stabilization  │  (Confidence-Aware EMA / Kalman)
                   └───────────┬────────────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │ Safety Controller Check│  (Confidence Thresholds & Occlusion)
                   └───────────┬────────────┘
                               │ Passed
                               ▼
                   ┌────────────────────────┐
                   │ OS Cursor Dispatch     │  (Mouse Move, Blink Click, Dwell Scroll)
                   └────────────────────────┘
```

## 2. Safety Controller Invariants
1. **Zero-Trust Multi-Face Handling:** If an unauthorized face appears or the identity of the current user is ambiguous, cursor injection is suspended immediately.
2. **Confidence-Gated Movement:** If iris tracking confidence drops below calibrated limits, cursor position holds steady rather than jumping erratically.
3. **Intentionality Discriminator:** Clicks require deliberate eye aspect ratio (EAR) sustained profiles above a minimum duration threshold to prevent involuntary reflex clicks.
