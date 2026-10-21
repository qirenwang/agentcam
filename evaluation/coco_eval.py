from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from routing.fusion import iou_xyxy


IOU_THRESHOLDS = [0.50 + idx * 0.05 for idx in range(10)]


@dataclass
class GroundTruth:
    image_id: int
    category_id: int
    xyxy: List[float]


@dataclass
class Prediction:
    image_id: int
    category_id: int
    xyxy: List[float]
    score: float


def bbox_xywh_to_xyxy(bbox: Sequence[float]) -> List[float]:
    x, y, w, h = [float(value) for value in bbox]
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def load_gt(path: Path) -> Tuple[Dict[int, Dict[str, object]], List[GroundTruth]]:
    data = json.loads(path.read_text())
    images = {int(image["id"]): image for image in data.get("images", [])}
    gts = [
        GroundTruth(
            image_id=int(ann["image_id"]),
            category_id=int(ann["category_id"]),
            xyxy=bbox_xywh_to_xyxy(ann["bbox"]),
        )
        for ann in data.get("annotations", [])
    ]
    return images, gts


def load_categories(path: Path, gts: Sequence[GroundTruth]) -> Dict[int, str]:
    data = json.loads(path.read_text())
    categories = {int(cat["id"]): str(cat.get("name", cat["id"])) for cat in data.get("categories", [])}
    for gt in gts:
        categories.setdefault(gt.category_id, str(gt.category_id))
    return categories


def load_predictions(path: Path, images: Dict[int, Dict[str, object]]) -> List[Prediction]:
    data = json.loads(path.read_text())
    file_to_id = {str(image.get("file_name")): image_id for image_id, image in images.items()}
    predictions: List[Prediction] = []

    if isinstance(data, list):
        for pred in data:
            predictions.append(
                Prediction(
                    image_id=int(pred["image_id"]),
                    category_id=int(pred["category_id"]),
                    xyxy=bbox_xywh_to_xyxy(pred["bbox"]),
                    score=float(pred.get("score", 1.0)),
                )
            )
        return predictions

    for frame in data.get("frames", []):
        image_id = file_to_id.get(str(frame.get("image_path"))) or file_to_id.get(str(frame.get("image_id")))
        if image_id is None:
            continue
        for det in frame.get("predictions", []):
            box = det.get("xyxy")
            if box is None and det.get("bbox") is not None:
                box = bbox_xywh_to_xyxy(det["bbox"])
            if box is None:
                continue
            predictions.append(
                Prediction(
                    image_id=int(image_id),
                    category_id=int(det.get("class_id", det.get("category_id", 0))),
                    xyxy=[float(value) for value in box],
                    score=float(det.get("score", 1.0)),
                )
            )
    return predictions


def match_stats(
    gts: Sequence[GroundTruth],
    preds: Sequence[Prediction],
    iou_threshold: float,
    category_id: Optional[int] = None,
    image_filter: Optional[set[int]] = None,
) -> Dict[str, float]:
    """Greedy score-ordered matching; class-aware within each image."""
    selected_gts = [
        gt
        for gt in gts
        if (category_id is None or gt.category_id == category_id)
        and (image_filter is None or gt.image_id in image_filter)
    ]
    selected_preds = [
        pred
        for pred in preds
        if (category_id is None or pred.category_id == category_id)
        and (image_filter is None or pred.image_id in image_filter)
    ]
    gt_by_key: Dict[Tuple[int, int], List[GroundTruth]] = defaultdict(list)
    for gt in selected_gts:
        gt_by_key[(gt.image_id, gt.category_id)].append(gt)
    matched: set[Tuple[int, int, int]] = set()
    tp = 0
    fp = 0
    for pred in sorted(selected_preds, key=lambda item: item.score, reverse=True):
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
        if best_iou >= iou_threshold and best_idx >= 0:
            matched.add((pred.image_id, pred.category_id, best_idx))
            tp += 1
        else:
            fp += 1
    fn = max(0, len(selected_gts) - tp)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "gt": float(len(selected_gts)),
        "predictions": float(len(selected_preds)),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, len(selected_gts)),
    }


def average_precision(
    gts: Sequence[GroundTruth],
    preds: Sequence[Prediction],
    category_id: int,
    iou_threshold: float,
    image_filter: Optional[set[int]] = None,
) -> Optional[float]:
    """101-point interpolated AP for one category. None when the scope has no GT of it."""
    class_gts = [
        gt
        for gt in gts
        if gt.category_id == category_id and (image_filter is None or gt.image_id in image_filter)
    ]
    if not class_gts:
        return None
    class_preds = [
        pred
        for pred in preds
        if pred.category_id == category_id and (image_filter is None or pred.image_id in image_filter)
    ]
    if not class_preds:
        return 0.0
    gt_by_image: Dict[int, List[GroundTruth]] = defaultdict(list)
    for gt in class_gts:
        gt_by_image[gt.image_id].append(gt)
    matched: set[Tuple[int, int]] = set()
    tp_flags: List[int] = []
    for pred in sorted(class_preds, key=lambda item: item.score, reverse=True):
        candidates = gt_by_image.get(pred.image_id, [])
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(candidates):
            if (pred.image_id, idx) in matched:
                continue
            score = iou_xyxy(pred.xyxy, gt.xyxy)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_iou >= iou_threshold and best_idx >= 0:
            matched.add((pred.image_id, best_idx))
            tp_flags.append(1)
        else:
            tp_flags.append(0)
    flags = np.asarray(tp_flags, dtype=np.float64)
    cum_tp = np.cumsum(flags)
    cum_fp = np.cumsum(1.0 - flags)
    recalls = cum_tp / len(class_gts)
    precisions = cum_tp / np.maximum(1.0, cum_tp + cum_fp)
    sampled = []
    for threshold in np.linspace(0.0, 1.0, 101):
        eligible = precisions[recalls >= threshold]
        sampled.append(float(eligible.max()) if eligible.size else 0.0)
    return float(np.mean(sampled))


def evaluate_scope(
    gts: Sequence[GroundTruth],
    preds: Sequence[Prediction],
    categories: Dict[int, str],
    image_filter: Optional[set[int]] = None,
) -> Dict[str, object]:
    """Macro mAP over categories plus pooled counts at the retention confidence."""
    per_class: Dict[str, Dict[str, object]] = {}
    ap50_values: List[float] = []
    ap5095_values: List[float] = []
    for category_id in sorted(categories):
        aps = [
            average_precision(gts, preds, category_id, threshold, image_filter=image_filter)
            for threshold in IOU_THRESHOLDS
        ]
        if all(ap is None for ap in aps):
            continue
        ap50 = aps[0] if aps[0] is not None else 0.0
        valid = [ap for ap in aps if ap is not None]
        ap5095 = float(np.mean(valid)) if valid else 0.0
        stats = match_stats(gts, preds, 0.50, category_id=category_id, image_filter=image_filter)
        per_class[categories[category_id]] = {
            "category_id": category_id,
            "AP50": ap50,
            "AP50_95": ap5095,
            "precision": stats["precision"],
            "recall": stats["recall"],
            "tp": stats["tp"],
            "fp": stats["fp"],
            "fn": stats["fn"],
            "gt": stats["gt"],
            "predictions": stats["predictions"],
        }
        ap50_values.append(ap50)
        ap5095_values.append(ap5095)

    pooled = match_stats(gts, preds, 0.50, image_filter=image_filter)
    return {
        "mAP50": float(np.mean(ap50_values)) if ap50_values else 0.0,
        "mAP50-95": float(np.mean(ap5095_values)) if ap5095_values else 0.0,
        "precision": pooled["precision"],
        "recall": pooled["recall"],
        "critical_FN_rate": pooled["fn"] / max(1.0, pooled["gt"]),
        "tp": pooled["tp"],
        "fp": pooled["fp"],
        "fn": pooled["fn"],
        "num_gt": pooled["gt"],
        "num_predictions": pooled["predictions"],
        "per_class": per_class,
    }


def degradation_of(image: Dict[str, object]) -> str:
    value = str(image.get("degradation_type", "") or "").strip().lower()
    return value if value else "unknown"


def load_images_meta(path: Path, meta_field: str) -> Dict[str, str]:
    """file_name -> degradation label from a sidecar CSV (e.g. evaluated_images.csv)."""
    by_file: Dict[str, str] = {}
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            file_name = str(row.get("file_name", "")).strip()
            value = str(row.get(meta_field, "") or "").strip().lower()
            if file_name:
                by_file[file_name] = value if value else "unknown"
    return by_file


def evaluate_coco_like(
    gt_path: Path,
    pred_path: Path,
    images_meta: Optional[Path] = None,
    meta_field: str = "manual_degradation",
) -> Dict[str, object]:
    images, gts = load_gt(gt_path)
    categories = load_categories(gt_path, gts)
    preds = load_predictions(pred_path, images)

    overall = evaluate_scope(gts, preds, categories)

    meta_by_file = load_images_meta(images_meta, meta_field) if images_meta else {}
    degradation_groups: Dict[str, set[int]] = defaultdict(set)
    for image_id, image in images.items():
        label = meta_by_file.get(str(image.get("file_name", "")))
        degradation_groups[label if label else degradation_of(image)].add(image_id)
    by_degradation: Dict[str, Dict[str, object]] = {}
    for group, image_ids in sorted(degradation_groups.items()):
        scoped = evaluate_scope(gts, preds, categories, image_filter=image_ids)
        scoped.pop("per_class", None)
        scoped["image_count"] = len(image_ids)
        by_degradation[group] = scoped

    covered = sum(len(ids) for ids in degradation_groups.values())
    return {
        **{key: overall[key] for key in [
            "mAP50",
            "mAP50-95",
            "precision",
            "recall",
            "critical_FN_rate",
            "num_gt",
            "num_predictions",
        ]},
        "metric_notes": (
            "mAP is the macro mean of 101-point interpolated per-class AP. "
            "precision/recall are pooled at the candidate-retention confidence, not a deployment threshold."
        ),
        "image_count": len(images),
        "degradation_covered_image_count": covered,
        "per_class": overall["per_class"],
        "by_degradation": by_degradation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate AgentCam predictions against COCO-style GT.")
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--images_meta", type=Path, default=None, help="Sidecar CSV with file_name plus a degradation column.")
    parser.add_argument("--meta_field", default="manual_degradation")
    parser.add_argument("--out", type=Path, default=Path("artifacts/metrics/coco_eval.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", "--dry-run", action="store_true", dest="dry_run")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = evaluate_coco_like(args.gt, args.pred, images_meta=args.images_meta, meta_field=args.meta_field)
    if args.dry_run:
        print(json.dumps(metrics, indent=2))
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2))
        print(f"Wrote evaluation metrics to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
