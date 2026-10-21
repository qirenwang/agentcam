from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


COCO_CAR_TRUCK_TO_DATASET: Dict[int, int] = {2: 0, 7: 1}


@dataclass
class Detection:
    image_id: str
    xyxy: List[float]
    score: float
    class_id: int
    source: str
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "image_id": self.image_id,
            "xyxy": [float(value) for value in self.xyxy],
            "score": float(self.score),
            "class_id": int(self.class_id),
            "source": self.source,
            "extra": self.extra,
        }


class DetectorWorker:
    source_name = "detector"

    def predict(self, image_path: str | Path) -> List[Detection]:
        raise NotImplementedError

    def predict_batch(self, image_paths: Iterable[str | Path]) -> Dict[str, List[Detection]]:
        return {str(path): self.predict(path) for path in image_paths}


class UltralyticsWorker(DetectorWorker):
    """Shared inference protocol for ultralytics-backed workers.

    ``class_map`` remaps native model class ids to dataset class ids
    (e.g. COCO 2 car -> 0, 7 truck -> 1 for COCO-pretrained weights).
    When ``class_map`` is set and ``classes`` is not, prediction is
    restricted to the mapped native ids; a prediction outside the map is
    an error rather than a silently wrong class id.
    """

    source_name = "ultralytics"

    def __init__(
        self,
        weights: str | Path,
        device: Optional[str] = None,
        conf: float = 0.25,
        imgsz: int = 640,
        iou: float = 0.7,
        max_det: int = 300,
        classes: Optional[Sequence[int]] = None,
        class_map: Optional[Mapping[int, int]] = None,
        half: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.weights = str(weights)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.iou = iou
        self.max_det = max_det
        self.class_map = {int(key): int(value) for key, value in class_map.items()} if class_map else None
        if classes is not None:
            self.classes: Optional[List[int]] = [int(value) for value in classes]
        elif self.class_map is not None:
            self.classes = sorted(self.class_map)
        else:
            self.classes = None
        self.half = half
        self.dry_run = dry_run
        self.model = None
        self.last_speed: Dict[str, float] = {}
        if dry_run:
            return
        self.model = self._load_model()

    def _load_model(self):
        raise NotImplementedError

    def config_dict(self) -> Dict[str, object]:
        return {
            "weights": self.weights,
            "device": self.device,
            "conf": self.conf,
            "imgsz": self.imgsz,
            "iou": self.iou,
            "max_det": self.max_det,
            "classes": self.classes,
            "class_map": self.class_map,
            "half": self.half,
        }

    def predict_kwargs(self, image_path: str | Path) -> Dict[str, object]:
        kwargs: Dict[str, object] = {
            "source": str(image_path),
            "conf": self.conf,
            "imgsz": self.imgsz,
            "iou": self.iou,
            "max_det": self.max_det,
            "half": self.half,
            "verbose": False,
        }
        if self.classes is not None:
            kwargs["classes"] = self.classes
        if self.device is not None:
            kwargs["device"] = self.device
        return kwargs

    def map_class(self, native_class_id: int) -> int:
        if self.class_map is None:
            return native_class_id
        if native_class_id not in self.class_map:
            raise ValueError(
                f"native class {native_class_id} is not in class_map {sorted(self.class_map)}; "
                "restrict prediction classes or extend the map."
            )
        return self.class_map[native_class_id]

    def warmup(self, image_path: str | Path, calls: int = 2) -> None:
        if self.dry_run or self.model is None:
            return
        for _ in range(max(0, calls)):
            self.model.predict(**self.predict_kwargs(image_path))

    def predict(self, image_path: str | Path) -> List[Detection]:
        if self.dry_run:
            return []
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} model is not initialized.")
        results = self.model.predict(**self.predict_kwargs(image_path))
        detections: List[Detection] = []
        self.last_speed = {}
        for result in results:
            speed = getattr(result, "speed", None) or {}
            self.last_speed = {
                "preprocess_ms": float(speed.get("preprocess", 0.0)),
                "inference_ms": float(speed.get("inference", 0.0)),
                "postprocess_ms": float(speed.get("postprocess", 0.0)),
            }
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy_values = boxes.xyxy.cpu().tolist()
            conf_values = boxes.conf.cpu().tolist()
            cls_values = boxes.cls.cpu().tolist()
            for xyxy, score, native_class in zip(xyxy_values, conf_values, cls_values):
                native_class_id = int(native_class)
                extra: Dict[str, object] = {"weights": self.weights}
                if self.class_map is not None:
                    extra["native_class_id"] = native_class_id
                detections.append(
                    Detection(
                        image_id=str(image_path),
                        xyxy=[float(value) for value in xyxy],
                        score=float(score),
                        class_id=self.map_class(native_class_id),
                        source=self.source_name,
                        extra=extra,
                    )
                )
        return detections
