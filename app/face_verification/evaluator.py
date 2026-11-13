"""AdaptiveEye Biometric Evaluation and Calibration Framework.

Calculates False Acceptance Rate (FAR), False Rejection Rate (FRR), Genuine
Acceptance Rate (GAR), Impostor Rejection Rate (IRR), and Equal Error Rate (EER)
to scientifically calibrate verification thresholds rather than guessing.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.core.logger import get_logger

logger = get_logger("BiometricEvaluator")


@dataclass
class ThresholdPoint:
    """Metrics evaluated at a specific threshold tau."""
    threshold: float
    far: float  # False Acceptance Rate
    frr: float  # False Rejection Rate
    gar: float  # Genuine Acceptance Rate (1 - FRR)
    irr: float  # Impostor Rejection Rate (1 - FAR)
    f1_score: float


@dataclass
class EvaluationReport:
    """Comprehensive biometric evaluation summary."""
    num_genuine_pairs: int
    num_impostor_pairs: int
    genuine_mean_sim: float
    genuine_std_sim: float
    impostor_mean_sim: float
    impostor_std_sim: float
    eer_threshold: float
    eer: float
    recommended_threshold: float  # Conservative threshold (FAR <= 0.1%)
    points: List[ThresholdPoint]


class BiometricEvaluator:
    """Evaluates biometric verification distributions, computes FAR/FRR, and finds optimal thresholds."""

    def __init__(self, threshold_step: float = 0.01) -> None:
        self.threshold_step = threshold_step

    def evaluate(
        self,
        genuine_scores: List[float],
        impostor_scores: List[float],
    ) -> EvaluationReport:
        """Compute biometric performance metrics across threshold spectrum.
        
        Args:
            genuine_scores: List of cosine similarities between genuine (authorized) samples.
            impostor_scores: List of cosine similarities between unauthorized (unknown) samples and the template.
            
        Returns:
            EvaluationReport containing EER, recommended operating thresholds, and FAR/FRR curves.
        """
        gen_arr = np.array(genuine_scores, dtype=np.float64)
        imp_arr = np.array(impostor_scores, dtype=np.float64)

        if len(gen_arr) == 0 or len(imp_arr) == 0:
            raise ValueError("Both genuine_scores and impostor_scores must contain at least 1 sample.")

        thresholds = np.arange(0.50, 0.99, self.threshold_step)
        points: List[ThresholdPoint] = []

        min_diff = float("inf")
        eer_thresh = 0.85
        eer_val = 0.0

        rec_thresh = 0.88  # Default recommendation
        found_conservative = False

        for th in thresholds:
            th_f = round(float(th), 4)

            # False Rejection: genuine sample rejected (score < threshold)
            fr_count = np.sum(gen_arr < th_f)
            frr = float(fr_count / len(gen_arr))

            # False Acceptance: impostor sample accepted (score >= threshold)
            fa_count = np.sum(imp_arr >= th_f)
            far = float(fa_count / len(imp_arr))

            gar = 1.0 - frr
            irr = 1.0 - far

            # Precision & Recall for F1
            tp = len(gen_arr) - fr_count
            fp = fa_count
            fn = fr_count
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            point = ThresholdPoint(
                threshold=th_f,
                far=round(far, 4),
                frr=round(frr, 4),
                gar=round(gar, 4),
                irr=round(irr, 4),
                f1_score=round(f1, 4),
            )
            points.append(point)

            # Check Equal Error Rate (where FAR ~= FRR)
            diff = abs(far - frr)
            if diff < min_diff:
                min_diff = diff
                eer_thresh = th_f
                eer_val = round((far + frr) / 2.0, 4)

            # Conservative threshold for high-security eye mouse: FAR <= 0.01 (1%)
            if far <= 0.01 and not found_conservative:
                rec_thresh = th_f
                found_conservative = True

        report = EvaluationReport(
            num_genuine_pairs=len(gen_arr),
            num_impostor_pairs=len(imp_arr),
            genuine_mean_sim=round(float(np.mean(gen_arr)), 4),
            genuine_std_sim=round(float(np.std(gen_arr)), 4),
            impostor_mean_sim=round(float(np.mean(imp_arr)), 4),
            impostor_std_sim=round(float(np.std(imp_arr)), 4),
            eer_threshold=eer_thresh,
            eer=eer_val,
            recommended_threshold=rec_thresh,
            points=points,
        )

        logger.info(
            "Evaluation complete: EER=%.2f%% @ thresh=%.3f | Rec Thresh=%.3f",
            report.eer * 100.0,
            report.eer_threshold,
            report.recommended_threshold,
        )
        return report

    def save_report(self, report: EvaluationReport, output_path: Path) -> None:
        """Save evaluation report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info("Evaluation report saved to %s", output_path)

    @staticmethod
    def print_summary(report: EvaluationReport) -> None:
        """Print human-readable evaluation summary table."""
        print("\n" + "=" * 70)
        print("          ADAPTIVEEYE BIOMETRIC CALIBRATION REPORT")
        print("=" * 70)
        print(f"Genuine Samples:   {report.num_genuine_pairs:<8} Mean Sim: {report.genuine_mean_sim:.4f} (std: {report.genuine_std_sim:.4f})")
        print(f"Impostor Samples:  {report.num_impostor_pairs:<8} Mean Sim: {report.impostor_mean_sim:.4f} (std: {report.impostor_std_sim:.4f})")
        print("-" * 70)
        print(f"Equal Error Rate (EER):      {report.eer:.2%} @ Threshold = {report.eer_threshold:.3f}")
        print(f"Recommended Auth Threshold:  {report.recommended_threshold:.3f} (enforces FAR <= 1.0%)")
        print("-" * 70)
        print("Threshold Sweep Sample:")
        print("  Thresh  |   FAR (%)   |   FRR (%)   |   GAR (%)   |   IRR (%)   |   F1   ")
        print("  " + "-" * 62)
        for p in report.points[::4]:  # Print every 4th point for readability
            print(
                f"  {p.threshold:6.2f}  |  "
                f"{p.far*100:8.2f}%  |  "
                f"{p.frr*100:8.2f}%  |  "
                f"{p.gar*100:8.2f}%  |  "
                f"{p.irr*100:8.2f}%  |  "
                f"{p.f1_score:5.3f}"
            )
        print("=" * 70 + "\n")
