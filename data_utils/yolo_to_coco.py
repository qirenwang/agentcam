from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from data_utils.manifest import load_yaml, read_manifest
from data_utils.validate_yolo import names_from_data_yaml, parse_yolo_label_file


def yolo_record_to_coco_bbox(record, width: int, height: int) -> List[float]:
    box_w = record.width * width
    box_h = record.height * height
    x_min = (record.x_center * width) - box_w / 2.0
    y_min = (record.y_center * height) - box_h / 2.0
    return [x_min, y_min, box_w, box_h]


def manifest_to_coco(data_dir: Path, split: str, limit: int = 0) -> Dict[str, object]:
    rows = [row for row in read_manifest(data_dir / "manifest.csv") if row.split == split and row.has_label]
    if limit and limit > 0:
        rows = rows[:limit]
    names = names_from_data_yaml(data_dir / "data_trainval.yaml")
    categories = [{"id": idx, "name": name} for idx, name in enumerate(names)]
    images = []
    annotations = []
    annotation_id = 1

    for image_id, row in enumerate(rows, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": row.image_path,
                "width": row.width,
                "height": row.height,
                "split": row.split,
                "degradation_type": row.degradation_type,
            }
        )
        if row.is_empty_label:
            continue
        for record in parse_yolo_label_file(Path(row.label_path), class_count=len(names)):
            bbox = yolo_record_to_coco_bbox(record, row.width, row.height)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": record.class_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert AgentCam YOLO manifest labels to COCO JSON.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/agentcam"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=Path("artifacts/coco/test_gt.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", "--dry-run", action="store_true", dest="dry_run")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    coco = manifest_to_coco(args.data, args.split, args.limit)
    if args.dry_run:
        print(json.dumps({k: len(v) for k, v in coco.items() if isinstance(v, list)}, indent=2))
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(coco, indent=2))
        print(f"Wrote COCO annotations to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
