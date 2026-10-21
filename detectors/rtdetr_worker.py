from __future__ import annotations

from pathlib import Path

from detectors.base import UltralyticsWorker


class RTDETRWorker(UltralyticsWorker):
    source_name = "rtdetr"

    def _load_model(self):
        weight_path = Path(self.weights)
        if not weight_path.exists():
            raise FileNotFoundError(
                f"RT-DETR weights not found: {self.weights}. "
                "Provide trained weights or run router experiments with --disable_rtdetr."
            )
        try:
            from ultralytics import RTDETR
        except ImportError as exc:
            raise RuntimeError("Ultralytics RTDETR is required. Install requirements.txt or pass --disable_rtdetr.") from exc
        return RTDETR(str(weight_path))
