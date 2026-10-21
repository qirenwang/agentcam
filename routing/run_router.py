from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from data_utils.manifest import read_manifest
from detectors.base import Detection
from detectors.rtdetr_worker import RTDETRWorker
from detectors.yolo_worker import YOLOWorker
from evaluation.latency import timed_section
from routing.fusion import merge_yolo_and_rtdetr, replace_with_enhanced_if_better
from routing.rule_router import (
    RuleRouter,
    RouterInputs,
    calibrate_thresholds_from_scores,
    confidence_stats,
    load_router_config,
)
from workers.enhancement import enhance_to_cache


def load_scores(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", newline="") as f:
        return {row["image_path"]: row for row in csv.DictReader(f)}


def detections_to_dicts(detections: Sequence[Detection]) -> List[Dict[str, object]]:
    return [det.to_dict() for det in detections]


def route_dataset(
    data: Path,
    scores_path: Path,
    yolo_weights: str,
    rtdetr_weights: str,
    config_path: Path,
    split: str,
    out: Path,
    log_path: Path,
    limit: int = 0,
    dry_run: bool = False,
    disable_rtdetr: bool = False,
    latency_budget_ms: Optional[float] = None,
    cache_dir: Path = Path("artifacts/cache"),
) -> Dict[str, object]:
    rows = [row for row in read_manifest(data / "manifest.csv") if row.split == split]
    if limit and limit > 0:
        rows = rows[:limit]
    scores = load_scores(scores_path)
    config = calibrate_thresholds_from_scores(load_router_config(config_path), scores_path)
    if disable_rtdetr:
        router_cfg = dict(config.get("router", {}))
        router_cfg["enable_rtdetr"] = False
        config["router"] = router_cfg
    router = RuleRouter(config)
    low_conf_threshold = float(dict(config.get("thresholds", {})).get("low_confidence_detection_threshold", 0.30))

    yolo = YOLOWorker(yolo_weights, dry_run=dry_run)
    rtdetr = None
    if not disable_rtdetr:
        rtdetr = RTDETRWorker(rtdetr_weights, dry_run=dry_run)

    out.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    frames: List[Dict[str, object]] = []

    with log_path.open("w") as log_file:
        for row in rows:
            total_start = time.perf_counter()
            latencies = {
                "degradation_monitor_ms": 0.0,
                "yolo_ms": 0.0,
                "enhancement_ms": 0.0,
                "rtdetr_ms": 0.0,
                "router_ms": 0.0,
                "fusion_ms": 0.0,
                "total_ms": 0.0,
            }
            with timed_section(latencies, "degradation_monitor_ms"):
                score_row = scores.get(row.image_path)
                if score_row is None:
                    raise KeyError(f"No degradation score row for image: {row.image_path}")

            with timed_section(latencies, "yolo_ms"):
                yolo_detections = yolo.predict(row.image_path)

            yolo_scores = [det.score for det in yolo_detections]
            mean_conf, max_conf, num_det, low_ratio = confidence_stats(yolo_scores, low_conf_threshold)
            budget = latency_budget_ms if latency_budget_ms is not None else router.default_latency_budget_ms

            with timed_section(latencies, "router_ms"):
                decision = router.route(
                    RouterInputs(
                        blur_score=float(score_row.get("blur_score") or 0.0),
                        overexposure_score=float(score_row.get("overexposure_score") or 0.0),
                        saturated_pixel_ratio=float(score_row.get("saturated_pixel_ratio") or 0.0),
                        yolo_mean_confidence=mean_conf,
                        yolo_max_confidence=max_conf,
                        num_yolo_detections=num_det,
                        low_confidence_detection_ratio=low_ratio,
                        latency_budget_ms=budget,
                    )
                )

            final_detections = list(yolo_detections)
            if decision.action == "EXPOSURE_ENHANCE_THEN_YOLO":
                with timed_section(latencies, "enhancement_ms"):
                    enhanced_path = Path(row.image_path) if dry_run else enhance_to_cache(row.image_path, "exposure", cache_dir)
                with timed_section(latencies, "yolo_ms"):
                    enhanced_detections = yolo.predict(enhanced_path)
                with timed_section(latencies, "fusion_ms"):
                    final_detections = replace_with_enhanced_if_better(yolo_detections, enhanced_detections)
            elif decision.action == "SHARPEN_THEN_YOLO":
                with timed_section(latencies, "enhancement_ms"):
                    enhanced_path = Path(row.image_path) if dry_run else enhance_to_cache(row.image_path, "sharpen", cache_dir)
                with timed_section(latencies, "yolo_ms"):
                    enhanced_detections = yolo.predict(enhanced_path)
                with timed_section(latencies, "fusion_ms"):
                    final_detections = replace_with_enhanced_if_better(yolo_detections, enhanced_detections)
            elif decision.action == "RTDETR_VERIFY" and rtdetr is not None:
                with timed_section(latencies, "rtdetr_ms"):
                    rtdetr_detections = rtdetr.predict(row.image_path)
                with timed_section(latencies, "fusion_ms"):
                    final_detections = merge_yolo_and_rtdetr(yolo_detections, rtdetr_detections)

            latencies["total_ms"] = (time.perf_counter() - total_start) * 1000.0
            record = {
                "image_id": row.image_path,
                "image_path": row.image_path,
                "split": row.split,
                "degradation_type": row.degradation_type,
                "selected_action": decision.action,
                "action": decision.action,
                "reason": decision.reason,
                "scores": {
                    "blur_score": float(score_row.get("blur_score") or 0.0),
                    "overexposure_score": float(score_row.get("overexposure_score") or 0.0),
                    "saturated_pixel_ratio": float(score_row.get("saturated_pixel_ratio") or 0.0),
                    "near_white_pixel_ratio": float(score_row.get("near_white_pixel_ratio") or 0.0),
                },
                "yolo_confidence": {
                    "mean": mean_conf,
                    "max": max_conf,
                    "num_detections": num_det,
                    "low_confidence_detection_ratio": low_ratio,
                },
                "latency_ms": latencies,
                "predictions": detections_to_dicts(final_detections),
            }
            log_file.write(json.dumps(record) + "\n")
            frames.append(record)

    payload = {
        "split": split,
        "dry_run": dry_run,
        "disable_rtdetr": disable_rtdetr,
        "frames": frames,
    }
    out.write_text(json.dumps(payload, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AgentCam rule-based perception router.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--yolo_weights", required=True)
    parser.add_argument("--rtdetr_weights", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--latency_budget_ms", type=float)
    parser.add_argument("--disable_rtdetr", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", "--dry-run", action="store_true", dest="dry_run")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    route_dataset(
        data=args.data,
        scores_path=args.scores,
        yolo_weights=args.yolo_weights,
        rtdetr_weights=args.rtdetr_weights,
        config_path=args.config,
        split=args.split,
        out=args.out,
        log_path=args.log,
        limit=args.limit,
        dry_run=args.dry_run,
        disable_rtdetr=args.disable_rtdetr,
        latency_budget_ms=args.latency_budget_ms,
    )
    print(f"Wrote router predictions to {args.out}")
    print(f"Wrote router JSONL log to {args.log}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
