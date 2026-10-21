from __future__ import annotations

import unittest

from detectors.base import COCO_CAR_TRUCK_TO_DATASET
from detectors.rtdetr_worker import RTDETRWorker
from detectors.yolo_worker import YOLOWorker


class DetectorProtocolTests(unittest.TestCase):
    def test_dry_run_defaults_match_frozen_protocol(self) -> None:
        worker = YOLOWorker("weights/yolo26n.pt", dry_run=True)
        kwargs = worker.predict_kwargs("image.png")
        self.assertEqual(kwargs["imgsz"], 640)
        self.assertEqual(kwargs["iou"], 0.7)
        self.assertEqual(kwargs["max_det"], 300)
        self.assertNotIn("classes", kwargs)
        self.assertEqual(worker.predict("image.png"), [])

    def test_class_map_restricts_classes_and_remaps(self) -> None:
        worker = RTDETRWorker(
            "weights/rtdetr-l.pt", dry_run=True, class_map=COCO_CAR_TRUCK_TO_DATASET
        )
        self.assertEqual(worker.classes, [2, 7])
        self.assertEqual(worker.predict_kwargs("image.png")["classes"], [2, 7])
        self.assertEqual(worker.map_class(2), 0)
        self.assertEqual(worker.map_class(7), 1)
        with self.assertRaises(ValueError):
            worker.map_class(5)

    def test_explicit_classes_override_map_derived_filter(self) -> None:
        worker = YOLOWorker(
            "weights/yolo26n.pt", dry_run=True, classes=[2], class_map={2: 0}
        )
        self.assertEqual(worker.classes, [2])

    def test_config_dict_reports_protocol(self) -> None:
        worker = YOLOWorker("weights/yolo26n.pt", dry_run=True, conf=0.001)
        config = worker.config_dict()
        self.assertEqual(config["conf"], 0.001)
        self.assertEqual(config["max_det"], 300)
        self.assertIsNone(config["class_map"])


if __name__ == "__main__":
    unittest.main()
