from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.coco_eval import evaluate_coco_like
from evaluation.operating_point import analyze_operating_point


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload))


def gt_payload() -> dict:
    """10 car GT with perfect predictions available, 1 truck GT that is missed."""
    images = []
    annotations = []
    for idx in range(10):
        images.append({"id": idx, "file_name": f"car_{idx}.png", "degradation_type": "blur"})
        annotations.append({"image_id": idx, "category_id": 0, "bbox": [10, 10, 20, 20]})
    images.append({"id": 10, "file_name": "truck_0.png", "degradation_type": "underexposure"})
    annotations.append({"image_id": 10, "category_id": 1, "bbox": [10, 10, 20, 20]})
    images.append({"id": 11, "file_name": "empty_0.png"})
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "car"}, {"id": 1, "name": "truck"}],
    }


class CocoEvalTests(unittest.TestCase):
    def test_macro_map_and_degradation_groups(self) -> None:
        preds = [
            {"image_id": idx, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.9}
            for idx in range(10)
        ]
        # Truck prediction that does not overlap its GT: AP(truck) = 0.
        preds.append({"image_id": 10, "category_id": 1, "bbox": [100, 100, 20, 20], "score": 0.1})
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = Path(tmp) / "gt.json"
            pred_path = Path(tmp) / "pred.json"
            write_json(gt_path, gt_payload())
            write_json(pred_path, preds)
            metrics = evaluate_coco_like(gt_path, pred_path)

        # Macro mean over classes: car AP50 = 1.0, truck AP50 = 0.0 -> 0.5.
        # The old pooled (micro) ranking would report ~0.909 here.
        self.assertAlmostEqual(metrics["per_class"]["car"]["AP50"], 1.0, places=6)
        self.assertAlmostEqual(metrics["per_class"]["truck"]["AP50"], 0.0, places=6)
        self.assertAlmostEqual(metrics["mAP50"], 0.5, places=6)

        self.assertEqual(set(metrics["by_degradation"]), {"blur", "underexposure", "unknown"})
        self.assertEqual(metrics["by_degradation"]["blur"]["mAP50"], 1.0)
        self.assertEqual(metrics["degradation_covered_image_count"], metrics["image_count"])
        self.assertEqual(metrics["per_class"]["truck"]["fn"], 1.0)


class OperatingPointTests(unittest.TestCase):
    def build_fixture(self, tmp: Path) -> tuple[Path, Path]:
        images = [{"id": idx, "file_name": f"img_{idx}.png"} for idx in range(1, 5)]
        annotations = [
            {"image_id": idx, "category_id": 0, "bbox": [10, 10, 20, 20]} for idx in range(1, 5)
        ]
        gt = {"images": images, "annotations": annotations, "categories": [{"id": 0, "name": "car"}]}
        preds = [
            {"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.9},
            {"image_id": 2, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.8},
            {"image_id": 1, "category_id": 0, "bbox": [200, 200, 20, 20], "score": 0.6},
            {"image_id": 3, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.4},
            {"image_id": 4, "category_id": 0, "bbox": [200, 200, 20, 20], "score": 0.2},
        ]
        gt_path = tmp / "gt.json"
        pred_path = tmp / "pred.json"
        write_json(gt_path, gt)
        write_json(pred_path, preds)
        return gt_path, pred_path

    def test_f1_max_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gt_path, pred_path = self.build_fixture(Path(tmp))
            payload = analyze_operating_point(gt_path, pred_path, objective="f1")
        # Curve by hand: t=0.9 F1=0.4; t=0.8 F1=2/3; t=0.6 F1=4/7; t=0.4 F1=0.75; t=0.2 F1=2/3.
        self.assertAlmostEqual(payload["chosen_threshold"], 0.4, places=6)
        self.assertAlmostEqual(payload["overall"]["f1"], 0.75, places=6)
        self.assertAlmostEqual(payload["overall"]["precision"], 0.75, places=6)
        self.assertAlmostEqual(payload["overall"]["recall"], 0.75, places=6)
        self.assertEqual(payload["overall"]["fn"], 1.0)

    def test_recall_at_precision_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gt_path, pred_path = self.build_fixture(Path(tmp))
            payload = analyze_operating_point(
                gt_path, pred_path, objective="recall_at_precision", min_precision=1.0
            )
        # Highest recall while precision stays 1.0 is the two top-scored TPs.
        self.assertAlmostEqual(payload["chosen_threshold"], 0.8, places=6)
        self.assertAlmostEqual(payload["overall"]["recall"], 0.5, places=6)

    def test_curve_csv_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gt_path, pred_path = self.build_fixture(Path(tmp))
            curve_path = Path(tmp) / "curve.csv"
            analyze_operating_point(gt_path, pred_path, curve_out=curve_path)
            content = curve_path.read_text().strip().splitlines()
        self.assertEqual(content[0].split(",")[0], "scope")
        # 5 overall boundaries + 5 car boundaries + header.
        self.assertEqual(len(content), 11)


if __name__ == "__main__":
    unittest.main()
