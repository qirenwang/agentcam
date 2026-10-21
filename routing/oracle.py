from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from evaluation.coco_eval import GroundTruth, Prediction, load_gt, load_predictions
from routing.fusion import iou_xyxy


def parse_candidate_args(items: Sequence[str]) -> List[Tuple[str, Path]]:
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected ACTION=prediction.json, got: {item}")
        name, path = item.split("=", 1)
        parsed.append((name, Path(path)))
    return parsed


def per_image_detection_gain(
    gts: Sequence[GroundTruth],
    preds: Sequence[Prediction],
    image_id: int,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    image_gts = [gt for gt in gts if gt.image_id == image_id]
    image_preds = sorted([pred for pred in preds if pred.image_id == image_id], key=lambda pred: pred.score, reverse=True)
    matched = set()
    tp = 0
    fp = 0
    for pred in image_preds:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(image_gts):
            if idx in matched or gt.category_id != pred.category_id:
                continue
            score = iou_xyxy(pred.xyxy, gt.xyxy)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_iou >= iou_threshold and best_idx >= 0:
            matched.add(best_idx)
            tp += 1
        else:
            fp += 1
    fn = max(0, len(image_gts) - tp)
    return {"tp": float(tp), "fp": float(fp), "fn": float(fn), "gain": float(tp - fp), "critical_miss": 1.0 if fn else 0.0}


def load_candidate_latency(path: Path, images: Dict[int, Dict[str, object]]) -> Dict[int, float]:
    payload = json.loads(path.read_text())
    file_to_id = {str(image.get("file_name")): image_id for image_id, image in images.items()}
    latencies: Dict[int, float] = defaultdict(float)
    if not isinstance(payload, dict):
        return latencies
    for frame in payload.get("frames", []):
        image_id = file_to_id.get(str(frame.get("image_path"))) or file_to_id.get(str(frame.get("image_id")))
        if image_id is None:
            continue
        latencies[int(image_id)] = float(frame.get("latency_ms", {}).get("total_ms", 0.0))
    return latencies


def run_oracle(
    gt_path: Path,
    candidates: Sequence[Tuple[str, Path]],
    out: Path,
    lambda_latency: float,
    beta_critical_miss: float,
    limit: int = 0,
) -> Dict[str, object]:
    images, gts = load_gt(gt_path)
    image_ids = sorted(images)
    if limit and limit > 0:
        image_ids = image_ids[:limit]
    candidate_predictions = {name: load_predictions(path, images) for name, path in candidates}
    candidate_latencies = {name: load_candidate_latency(path, images) for name, path in candidates}

    choices = []
    for image_id in image_ids:
        scored = []
        for name, _ in candidates:
            stats = per_image_detection_gain(gts, candidate_predictions[name], image_id)
            latency = candidate_latencies[name].get(image_id, 0.0)
            utility = stats["gain"] - lambda_latency * latency - beta_critical_miss * stats["critical_miss"]
            scored.append({"action": name, "utility": utility, "latency_ms": latency, **stats})
        best = max(scored, key=lambda row: row["utility"]) if scored else {}
        choices.append({"image_id": image_id, "file_name": images[image_id].get("file_name"), "best": best, "candidates": scored})

    action_counts = Counter(choice["best"].get("action", "") for choice in choices)
    payload = {
        "lambda_latency": lambda_latency,
        "beta_critical_miss": beta_critical_miss,
        "note": "Offline upper-bound analysis only. Do not use test labels to tune the deployed router.",
        "action_counts": dict(action_counts),
        "choices": choices,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute offline oracle action choices for AgentCam analysis.")
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], help="Candidate as ACTION=prediction.json")
    parser.add_argument("--out", type=Path, default=Path("artifacts/analysis/oracle.json"))
    parser.add_argument("--lambda_latency", type=float, default=0.001)
    parser.add_argument("--beta_critical_miss", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", "--dry-run", action="store_true", dest="dry_run")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = parse_candidate_args(args.candidate)
    if args.dry_run:
        print(json.dumps({"gt": str(args.gt), "candidates": [name for name, _ in candidates]}, indent=2))
        return 0
    run_oracle(
        gt_path=args.gt,
        candidates=candidates,
        out=args.out,
        lambda_latency=args.lambda_latency,
        beta_critical_miss=args.beta_critical_miss,
        limit=args.limit,
    )
    print(f"Wrote oracle analysis to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
