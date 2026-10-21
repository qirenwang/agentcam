from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluation.coco_eval import (
    GroundTruth,
    Prediction,
    load_categories,
    load_gt,
    load_predictions,
)
from routing.fusion import iou_xyxy


def matched_flags(
    gts: Sequence[GroundTruth],
    preds: Sequence[Prediction],
    iou_threshold: float,
) -> List[Tuple[float, int, bool]]:
    """(score, category_id, is_tp) per prediction, greedy-matched in score order.

    Greedy score-ordered matching assigns each prediction independently of any
    lower-scored prediction, so flags computed once are valid at every
    confidence threshold applied afterwards.
    """
    gt_by_key: Dict[Tuple[int, int], List[GroundTruth]] = defaultdict(list)
    for gt in gts:
        gt_by_key[(gt.image_id, gt.category_id)].append(gt)
    matched: set[Tuple[int, int, int]] = set()
    flags: List[Tuple[float, int, bool]] = []
    for pred in sorted(preds, key=lambda item: item.score, reverse=True):
        candidates = gt_by_key.get((pred.image_id, pred.category_id), [])
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(candidates):
            key = (gt.image_id, gt.category_id, idx)
            if key in matched:
                continue
            score = iou_xyxy(pred.xyxy, gt.xyxy)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        is_tp = best_iou >= iou_threshold and best_idx >= 0
        if is_tp:
            matched.add((pred.image_id, pred.category_id, best_idx))
        flags.append((pred.score, pred.category_id, is_tp))
    return flags


def sweep_curve(flags: Sequence[Tuple[float, int, bool]], gt_count: int) -> List[Dict[str, float]]:
    """Precision/recall/F1 at every distinct confidence boundary, highest first."""
    if not flags or gt_count <= 0:
        return []
    ordered = sorted(flags, key=lambda item: item[0], reverse=True)
    scores = np.asarray([item[0] for item in ordered], dtype=np.float64)
    tps = np.cumsum([1.0 if item[2] else 0.0 for item in ordered])
    kept = np.arange(1, len(ordered) + 1, dtype=np.float64)
    boundary = np.append(scores[:-1] != scores[1:], True)
    points = []
    for idx in np.flatnonzero(boundary):
        tp = float(tps[idx])
        retained = float(kept[idx])
        precision = tp / retained
        recall = tp / gt_count
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        points.append(
            {
                "threshold": float(scores[idx]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": retained - tp,
                "fn": float(gt_count) - tp,
                "predictions_kept": retained,
            }
        )
    return points


def select_operating_point(
    points: Sequence[Dict[str, float]],
    objective: str,
    min_precision: float,
) -> Optional[Dict[str, float]]:
    if not points:
        return None
    if objective == "f1":
        return max(points, key=lambda point: (point["f1"], point["threshold"]))
    if objective == "recall_at_precision":
        eligible = [point for point in points if point["precision"] >= min_precision]
        if not eligible:
            return None
        return max(eligible, key=lambda point: (point["recall"], point["threshold"]))
    raise ValueError(f"Unknown objective: {objective}")


def metrics_at_threshold(
    flags: Sequence[Tuple[float, int, bool]],
    gt_count: int,
    threshold: float,
) -> Dict[str, float]:
    kept = [item for item in flags if item[0] >= threshold]
    tp = float(sum(1 for item in kept if item[2]))
    fp = float(len(kept)) - tp
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, float(gt_count))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": float(gt_count) - tp,
        "gt": float(gt_count),
        "predictions_kept": float(len(kept)),
    }


def analyze_operating_point(
    gt_path: Path,
    pred_path: Path,
    iou_threshold: float = 0.5,
    objective: str = "f1",
    min_precision: float = 0.8,
    curve_out: Optional[Path] = None,
) -> Dict[str, object]:
    images, gts = load_gt(gt_path)
    categories = load_categories(gt_path, gts)
    preds = load_predictions(pred_path, images)
    if not gts:
        raise ValueError(f"No ground-truth annotations in {gt_path}")
    if not preds:
        raise ValueError(f"No predictions in {pred_path}")

    flags = matched_flags(gts, preds, iou_threshold)
    overall_points = sweep_curve(flags, len(gts))
    chosen = select_operating_point(overall_points, objective, min_precision)
    if chosen is None:
        raise ValueError(
            f"No operating point satisfies objective '{objective}'"
            + (f" with precision >= {min_precision}" if objective == "recall_at_precision" else "")
        )
    threshold = float(chosen["threshold"])

    gt_count_by_class = defaultdict(int)
    for gt in gts:
        gt_count_by_class[gt.category_id] += 1
    per_class: Dict[str, Dict[str, object]] = {}
    curve_rows: List[Dict[str, object]] = [{"scope": "overall", **point} for point in overall_points]
    for category_id in sorted(categories):
        gt_count = gt_count_by_class.get(category_id, 0)
        if gt_count == 0:
            continue
        class_flags = [item for item in flags if item[1] == category_id]
        class_points = sweep_curve(class_flags, gt_count)
        class_best = select_operating_point(class_points, "f1", min_precision) if class_points else None
        per_class[categories[category_id]] = {
            "category_id": category_id,
            **metrics_at_threshold(class_flags, gt_count, threshold),
            "own_f1_optimal_threshold": float(class_best["threshold"]) if class_best else None,
            "own_f1_at_optimal": float(class_best["f1"]) if class_best else None,
        }
        curve_rows.extend({"scope": categories[category_id], **point} for point in class_points)

    if curve_out is not None:
        curve_out.parent.mkdir(parents=True, exist_ok=True)
        fields = ["scope", "threshold", "precision", "recall", "f1", "tp", "fp", "fn", "predictions_kept"]
        with curve_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in curve_rows:
                writer.writerow({field: row.get(field, "") for field in fields})

    overall = metrics_at_threshold(flags, len(gts), threshold)
    return {
        "config": {
            "gt": str(gt_path),
            "pred": str(pred_path),
            "iou_threshold": iou_threshold,
            "objective": objective,
            "min_precision": min_precision if objective == "recall_at_precision" else None,
        },
        "chosen_threshold": threshold,
        "overall": {
            **overall,
            "preds_per_image_mean": overall["predictions_kept"] / max(1, len(images)),
            "image_count": len(images),
        },
        "per_class": per_class,
        "curve_point_count": len(overall_points),
        "notes": (
            "Threshold selected on this split only. Freeze it before touching the test split; "
            "never tune on test."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a deployment confidence threshold from saved predictions via PR sweep."
    )
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--objective", choices=["f1", "recall_at_precision"], default="f1")
    parser.add_argument("--min_precision", type=float, default=0.8)
    parser.add_argument("--out", type=Path, default=Path("artifacts/metrics/operating_point.json"))
    parser.add_argument("--curve_out", type=Path, default=None, help="Optional CSV with the full PR curve per scope.")
    parser.add_argument("--dry_run", "--dry-run", action="store_true", dest="dry_run")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = analyze_operating_point(
        gt_path=args.gt,
        pred_path=args.pred,
        iou_threshold=args.iou,
        objective=args.objective,
        min_precision=args.min_precision,
        curve_out=args.curve_out,
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2))
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote operating point analysis to {args.out}")
        if args.curve_out:
            print(f"Wrote PR curve to {args.curve_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
