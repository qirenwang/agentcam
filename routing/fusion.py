from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List

from detectors.base import Detection


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def replace_with_enhanced_if_better(
    yolo_detections: Iterable[Detection],
    enhanced_detections: Iterable[Detection],
    iou_threshold: float = 0.5,
) -> List[Detection]:
    enhanced = list(enhanced_detections)
    fused: List[Detection] = []
    for det in yolo_detections:
        replacement = det
        for candidate in enhanced:
            if candidate.class_id != det.class_id:
                continue
            if candidate.score <= det.score:
                continue
            if iou_xyxy(candidate.xyxy, det.xyxy) > iou_threshold:
                replacement = replace(
                    candidate,
                    source=f"{candidate.source}_replacement",
                    extra={**candidate.extra, "replaced_source": det.source, "replaced_score": det.score},
                )
                break
        fused.append(replacement)
    return fused


def class_aware_nms(detections: Iterable[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    kept: List[Detection] = []
    by_class = {}
    for det in detections:
        by_class.setdefault(det.class_id, []).append(det)
    for class_detections in by_class.values():
        remaining = sorted(class_detections, key=lambda det: det.score, reverse=True)
        while remaining:
            current = remaining.pop(0)
            kept.append(current)
            remaining = [det for det in remaining if iou_xyxy(current.xyxy, det.xyxy) <= iou_threshold]
    return sorted(kept, key=lambda det: det.score, reverse=True)


def merge_yolo_and_rtdetr(
    yolo_detections: Iterable[Detection],
    rtdetr_detections: Iterable[Detection],
    iou_threshold: float = 0.5,
) -> List[Detection]:
    return class_aware_nms([*yolo_detections, *rtdetr_detections], iou_threshold=iou_threshold)
