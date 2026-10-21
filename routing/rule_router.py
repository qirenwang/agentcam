from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from data_utils.manifest import load_yaml
from degradation.metrics import quantile


YOLO_ONLY = "YOLO_ONLY"
EXPOSURE_ENHANCE_THEN_YOLO = "EXPOSURE_ENHANCE_THEN_YOLO"
SHARPEN_THEN_YOLO = "SHARPEN_THEN_YOLO"
RTDETR_VERIFY = "RTDETR_VERIFY"
UNCERTAINTY_SIGNAL = "UNCERTAINTY_SIGNAL"
TEMPORAL_RECOVERY = "TEMPORAL_RECOVERY"


@dataclass
class RouterInputs:
    blur_score: float
    overexposure_score: float
    saturated_pixel_ratio: float
    yolo_mean_confidence: float
    yolo_max_confidence: float
    num_yolo_detections: int
    low_confidence_detection_ratio: float
    latency_budget_ms: float
    temporal_inconsistent: bool = False


@dataclass
class RouterDecision:
    action: str
    reason: str


class RuleRouter:
    def __init__(self, config: Dict[str, object]) -> None:
        self.config = config
        self.thresholds: Dict[str, float] = {
            key: float(value) for key, value in dict(config.get("thresholds", {})).items()
        }
        router_cfg = dict(config.get("router", {}))
        self.enable_rtdetr = bool(router_cfg.get("enable_rtdetr", True))
        self.enable_temporal_recovery = bool(router_cfg.get("enable_temporal_recovery", False))
        self.default_latency_budget_ms = float(router_cfg.get("default_latency_budget_ms", 100.0))

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RuleRouter":
        return cls(load_yaml(path))

    def route(self, inputs: RouterInputs) -> RouterDecision:
        budget = inputs.latency_budget_ms or self.default_latency_budget_ms
        low_mean = self.thresholds.get("low_mean_confidence", 0.35)
        low_max = self.thresholds.get("low_max_confidence", 0.45)
        low_ratio = self.thresholds.get("low_confidence_ratio", 0.5)
        uncertain = (
            inputs.num_yolo_detections == 0
            or inputs.yolo_mean_confidence < low_mean
            or inputs.yolo_max_confidence < low_max
            or inputs.low_confidence_detection_ratio >= low_ratio
        )

        if self.enable_temporal_recovery and inputs.temporal_inconsistent:
            return RouterDecision(TEMPORAL_RECOVERY, "temporal inconsistency detected")

        if (
            inputs.overexposure_score >= self.thresholds.get("overexposure_score", 0.7)
            or inputs.saturated_pixel_ratio >= self.thresholds.get("saturated_pixel_ratio", 0.5)
        ) and budget >= self.thresholds.get("enhancement_min_budget_ms", 35.0):
            return RouterDecision(EXPOSURE_ENHANCE_THEN_YOLO, "overexposure threshold exceeded")

        if inputs.blur_score >= self.thresholds.get("blur_score", 0.7) and budget >= self.thresholds.get(
            "enhancement_min_budget_ms", 35.0
        ):
            return RouterDecision(SHARPEN_THEN_YOLO, "blur threshold exceeded")

        if uncertain and self.enable_rtdetr and budget >= self.thresholds.get("rtdetr_min_budget_ms", 80.0):
            return RouterDecision(RTDETR_VERIFY, "YOLO uncertainty threshold exceeded")

        if uncertain:
            return RouterDecision(UNCERTAINTY_SIGNAL, "YOLO uncertainty without verifier budget")

        return RouterDecision(YOLO_ONLY, "YOLO confidence and degradation scores within thresholds")


def load_router_config(path: Path | str) -> Dict[str, object]:
    return load_yaml(path)


def calibrate_thresholds_from_scores(config: Dict[str, object], scores_csv: Path | str) -> Dict[str, object]:
    calibrated = dict(config)
    calibration = dict(config.get("calibration", {}))
    if not calibration.get("enabled", False):
        return calibrated
    split = str(calibration.get("split", "val_from_train"))
    blur_values = []
    over_values = []
    with Path(scores_csv).open("r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") != split:
                continue
            if row.get("blur_score") not in {"", None}:
                blur_values.append(float(row["blur_score"]))
            if row.get("overexposure_score") not in {"", None}:
                over_values.append(float(row["overexposure_score"]))
    thresholds = dict(calibrated.get("thresholds", {}))
    if blur_values:
        thresholds["blur_score"] = quantile(blur_values, float(calibration.get("blur_quantile", 0.67)))
    if over_values:
        thresholds["overexposure_score"] = quantile(
            over_values, float(calibration.get("overexposure_quantile", 0.67))
        )
    calibrated["thresholds"] = thresholds
    return calibrated


def confidence_stats(scores: Sequence[float], low_conf_threshold: float = 0.30) -> Tuple[float, float, int, float]:
    if not scores:
        return 0.0, 0.0, 0, 1.0
    mean_conf = sum(scores) / len(scores)
    max_conf = max(scores)
    low_ratio = sum(1 for score in scores if score < low_conf_threshold) / len(scores)
    return mean_conf, max_conf, len(scores), low_ratio
