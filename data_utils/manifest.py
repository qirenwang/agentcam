from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MANIFEST_COLUMNS = [
    "image_path",
    "label_path",
    "has_label",
    "is_empty_label",
    "original_split",
    "split",
    "degradation_type",
    "sequence_id",
    "frame_id",
    "width",
    "height",
]

GENERIC_SEQUENCE_PARTS = {
    "",
    ".",
    "data",
    "raw",
    "processed",
    "agentcam",
    "images",
    "image",
    "imgs",
    "labels",
    "label",
    "ann",
    "annotations",
    "train",
    "test",
    "val",
    "valid",
    "validation",
    "train_core",
    "val_from_train",
    "blur",
    "blurry",
    "overexposure",
    "over_exposure",
    "exposure",
    "glare",
    "saturated",
    "sunlight",
    "unknown",
}


@dataclass
class ManifestRow:
    image_path: str
    label_path: str
    has_label: bool
    is_empty_label: bool
    original_split: str
    split: str
    degradation_type: str
    sequence_id: str
    frame_id: str
    width: int
    height: int

    def to_csv_row(self) -> Dict[str, str]:
        return {
            "image_path": self.image_path,
            "label_path": self.label_path,
            "has_label": str(self.has_label).lower(),
            "is_empty_label": str(self.is_empty_label).lower(),
            "original_split": self.original_split,
            "split": self.split,
            "degradation_type": self.degradation_type,
            "sequence_id": self.sequence_id,
            "frame_id": self.frame_id,
            "width": str(self.width),
            "height": str(self.height),
        }

    @classmethod
    def from_csv_row(cls, row: Dict[str, str]) -> "ManifestRow":
        return cls(
            image_path=row["image_path"],
            label_path=row.get("label_path", ""),
            has_label=parse_bool(row.get("has_label", "false")),
            is_empty_label=parse_bool(row.get("is_empty_label", "false")),
            original_split=row.get("original_split", ""),
            split=row.get("split", ""),
            degradation_type=row.get("degradation_type", "unknown"),
            sequence_id=row.get("sequence_id", ""),
            frame_id=row.get("frame_id", ""),
            width=int(row.get("width") or 0),
            height=int(row.get("height") or 0),
        )


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_manifest(path: Path | str) -> List[ManifestRow]:
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [ManifestRow.from_csv_row(row) for row in reader]


def write_manifest(path: Path | str, rows: Sequence[ManifestRow]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def infer_original_split(relative_path: Path | str) -> Optional[str]:
    parts = [p.lower() for p in Path(relative_path).parts]
    train_aliases = {"train", "training", "training-data", "training_data"}
    test_aliases = {"test", "testing", "testing-data", "testing_data"}
    if any(part in train_aliases for part in parts):
        return "train"
    if any(part in test_aliases for part in parts):
        return "test"
    return None


def infer_degradation_type(path: Path | str) -> str:
    text = str(path).lower().replace("-", "_")
    if "blur" in text or "blurry" in text:
        return "blur"
    overexposure_tokens = [
        "overexposure",
        "over_exposure",
        "exposure",
        "glare",
        "saturated",
        "sunlight",
    ]
    if any(token in text for token in overexposure_tokens):
        return "overexposure"
    return "unknown"


def infer_sequence_frame_id(relative_path: Path | str) -> Tuple[str, str]:
    path = Path(relative_path)
    stem = path.stem
    frame_id = ""
    sequence_id = ""

    number_matches = list(re.finditer(r"\d+", stem))
    if number_matches:
        last = number_matches[-1]
        frame_id = last.group(0)
        prefix = stem[: last.start()].strip("_-. ")
        if prefix and prefix.lower() not in {"img", "image", "frame", "frm"}:
            sequence_id = prefix

    candidate_parts = []
    for part in path.parent.parts:
        lowered = part.lower()
        if lowered not in GENERIC_SEQUENCE_PARTS and not lowered.startswith("__macosx"):
            candidate_parts.append(part)
    if candidate_parts:
        sequence_id = "/".join(candidate_parts)

    return sequence_id, frame_id


def image_size(path: Path) -> Tuple[int, int]:
    if Image is None:
        raise RuntimeError("Pillow is required to inspect image dimensions.")
    with Image.open(path) as im:
        return im.size


def is_empty_label_file(path: Path) -> bool:
    return path.exists() and path.read_text().strip() == ""


def hash_relative_path(path: Path | str) -> str:
    return hashlib.sha1(str(path).replace(os.sep, "/").encode("utf-8")).hexdigest()[:12]


def safe_processed_name(relative_path: Path | str) -> str:
    rel = Path(relative_path)
    return f"{hash_relative_path(rel)}_{rel.name}"


def symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def write_text_file(path: Path | str, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def write_yaml(path: Path | str, payload: Dict[str, object]) -> None:
    try:
        import yaml
    except ImportError:  # pragma: no cover
        lines = []
        for key, value in payload.items():
            if isinstance(value, list):
                lines.append(f"{key}:")
                lines.extend(f"  - {item}" for item in value)
            else:
                lines.append(f"{key}: {value}")
        write_text_file(path, "\n".join(lines) + "\n")
        return

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def load_yaml(path: Path | str) -> Dict[str, object]:
    import yaml

    with Path(path).open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def yolo_class_ids_from_labels(label_paths: Iterable[Path], strict: bool = False) -> List[int]:
    class_ids = set()
    for label_path in label_paths:
        if not label_path.exists() or is_empty_label_file(label_path):
            continue
        for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            first = stripped.split()[0]
            try:
                class_id = int(first)
            except ValueError as exc:
                if strict:
                    raise ValueError(f"{label_path}: line {line_number}: class_id must be an integer") from exc
                continue
            if class_id < 0:
                if strict:
                    raise ValueError(f"{label_path}: line {line_number}: class_id must be non-negative")
                continue
            class_ids.add(class_id)
    return sorted(class_ids)


def class_names_from_labels(label_paths: Iterable[Path]) -> List[str]:
    class_ids = yolo_class_ids_from_labels(label_paths)
    if not class_ids:
        warnings.warn(
            "No raw category metadata or YOLO class IDs found; using placeholder class name class_0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return ["class_0"]
    max_class = max(class_ids)
    warnings.warn(
        "No raw category metadata found; using placeholder class names derived from YOLO label IDs.",
        RuntimeWarning,
        stacklevel=2,
    )
    if max_class < 0:
        return ["class_0"]
    return [f"class_{idx}" for idx in range(max_class + 1)]


def count_by(rows: Sequence[ManifestRow], *fields: str) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    for row in rows:
        key = tuple(str(getattr(row, field)) for field in fields)
        counts[key] = counts.get(key, 0) + 1
    return counts
