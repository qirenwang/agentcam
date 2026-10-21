from __future__ import annotations

from detectors.base import UltralyticsWorker


class YOLOWorker(UltralyticsWorker):
    source_name = "yolo"

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is required for YOLOWorker. Install requirements.txt.") from exc
        return YOLO(self.weights)
