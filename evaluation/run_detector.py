from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from data_utils.manifest import read_manifest
from detectors.base import COCO_CAR_TRUCK_TO_DATASET, Detection
from detectors.rtdetr_worker import RTDETRWorker
from detectors.yolo_worker import YOLOWorker
from evaluation.latency import timed_section
from routing.fusion import merge_yolo_and_rtdetr
from workers.enhancement import enhance_to_cache


CLASS_MAPS: Dict[str, Optional[Dict[int, int]]] = {
    "none": None,
    "coco_car_truck": COCO_CAR_TRUCK_TO_DATASET,
}


def write_prediction_payload(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def run_detector(
    data: Path,
    split: str,
    detector: str,
    weights: str,
    out: Path,
    rtdetr_weights: Optional[str] = None,
    enhancement: str = "none",
    limit: int = 0,
    dry_run: bool = False,
    disable_missing_rtdetr: bool = False,
    conf: float = 0.001,
    imgsz: int = 640,
    iou: float = 0.7,
    max_det: int = 300,
    device: Optional[str] = None,
    class_map: str = "none",
    warmup_calls: int = 2,
) -> Dict[str, object]:
    if class_map not in CLASS_MAPS:
        raise ValueError(f"Unknown class_map '{class_map}'. Choices: {sorted(CLASS_MAPS)}")
    rows = [row for row in read_manifest(data / "manifest.csv") if row.split == split]
    if limit and limit > 0:
        rows = rows[:limit]

    worker_kwargs = {
        "device": device,
        "conf": conf,
        "imgsz": imgsz,
        "iou": iou,
        "max_det": max_det,
        "class_map": CLASS_MAPS[class_map],
        "dry_run": dry_run,
    }
    skipped = False
    worker = None
    rtdetr_worker = None
    if detector in {"yolo", "yolo_plus_rtdetr"}:
        worker = YOLOWorker(weights, **worker_kwargs)
    if detector in {"rtdetr", "yolo_plus_rtdetr"}:
        try:
            rtdetr_worker = RTDETRWorker(rtdetr_weights or weights, **worker_kwargs)
        except FileNotFoundError:
            if not disable_missing_rtdetr:
                raise
            skipped = True

    if rows and not dry_run:
        warmup_path = Path(rows[0].image_path)
        for active in (worker, rtdetr_worker):
            if active is not None:
                active.warmup(warmup_path, calls=warmup_calls)

    frames: List[Dict[str, object]] = []
    for row in rows:
        latencies = {"preprocess_ms": 0.0, "detector_ms": 0.0, "fusion_ms": 0.0, "total_ms": 0.0}
        start = time.perf_counter()
        image_path = Path(row.image_path)
        detector_input = image_path
        if enhancement != "none":
            with timed_section(latencies, "preprocess_ms"):
                detector_input = image_path if dry_run else enhance_to_cache(image_path, enhancement)

        detections: List[Detection] = []
        speed: Dict[str, Dict[str, float]] = {}
        if detector == "yolo" and worker is not None:
            with timed_section(latencies, "detector_ms"):
                detections = worker.predict(detector_input)
            speed["yolo"] = dict(worker.last_speed)
        elif detector == "rtdetr" and rtdetr_worker is not None:
            with timed_section(latencies, "detector_ms"):
                detections = rtdetr_worker.predict(detector_input)
            speed["rtdetr"] = dict(rtdetr_worker.last_speed)
        elif detector == "yolo_plus_rtdetr" and worker is not None and rtdetr_worker is not None:
            with timed_section(latencies, "detector_ms"):
                yolo_detections = worker.predict(detector_input)
                rtdetr_detections = rtdetr_worker.predict(detector_input)
            speed["yolo"] = dict(worker.last_speed)
            speed["rtdetr"] = dict(rtdetr_worker.last_speed)
            with timed_section(latencies, "fusion_ms"):
                detections = merge_yolo_and_rtdetr(yolo_detections, rtdetr_detections)

        latencies["total_ms"] = (time.perf_counter() - start) * 1000.0
        frames.append(
            {
                "image_id": row.image_path,
                "image_path": row.image_path,
                "split": row.split,
                "degradation_type": row.degradation_type,
                "latency_ms": latencies,
                "ultralytics_speed_ms": speed,
                "prediction_count": len(detections),
                "prediction_count_reached_max_det": len(detections) >= max_det,
                "predictions": [det.to_dict() for det in detections],
            }
        )

    payload = {
        "detector": detector,
        "weights": weights,
        "rtdetr_weights": rtdetr_weights,
        "enhancement": enhancement,
        "split": split,
        "dry_run": dry_run,
        "skipped": skipped,
        "config": {
            "conf": conf,
            "imgsz": imgsz,
            "iou": iou,
            "max_det": max_det,
            "device": device,
            "class_map": class_map,
            "warmup_calls": warmup_calls,
            "batch_size": 1,
        },
        "frames": frames,
    }
    write_prediction_payload(out, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a detector baseline on an AgentCam split.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/agentcam"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--detector", choices=["yolo", "rtdetr", "yolo_plus_rtdetr"], default="yolo")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--rtdetr_weights")
    parser.add_argument("--enhancement", choices=["none", "exposure", "sharpen"], default="none")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.001, help="Candidate-retention confidence for offline eval, not a deployment threshold.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--device", default=None)
    parser.add_argument("--class_map", choices=sorted(CLASS_MAPS), default="none", help="Use coco_car_truck for COCO-pretrained weights (2 car -> 0, 7 truck -> 1).")
    parser.add_argument("--warmup", type=int, default=2, dest="warmup_calls")
    parser.add_argument("--disable_missing_rtdetr", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", "--dry-run", action="store_true", dest="dry_run")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_detector(
        data=args.data,
        split=args.split,
        detector=args.detector,
        weights=args.weights,
        out=args.out,
        rtdetr_weights=args.rtdetr_weights,
        enhancement=args.enhancement,
        limit=args.limit,
        dry_run=args.dry_run,
        disable_missing_rtdetr=args.disable_missing_rtdetr,
        conf=args.conf,
        imgsz=args.imgsz,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        class_map=args.class_map,
        warmup_calls=args.warmup_calls,
    )
    print(f"Wrote detector predictions to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
