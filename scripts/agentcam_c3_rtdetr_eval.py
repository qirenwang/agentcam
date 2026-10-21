from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.latency import cuda_synchronize
from routing.fusion import iou_xyxy
from vlm.action_bank import ACTION_BY_CANONICAL_CLASS, action_for_canonical_class
from workers.enhancement import apply_enhancement_action, operation_order_for_action, parameters_for_action


C1_RUN = Path("artifacts/agentcam_qwen_vlm_degradation_audit/20260620T024704Z")
C2_RUN = Path("artifacts/agentcam_c2_enhancement_action_bank/20260621T011459Z")
C1_JSONL = C1_RUN / "audit/qwen_vlm_degradation_audit.jsonl"
WEIGHT_PATH = Path("weights/rtdetr-l.pt")
EXPECTED_WEIGHT_SHA256 = "6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f"
EXPECTED_WEIGHT_SIZE = 66_511_432
EXPECTED_INTERPRETER = "/home/qiren/agentcam/ag1_venv/bin/python"
EXPECTED_CLASS_COUNTS = {
    "blur": 177,
    "overexposure": 306,
    "underexposure": 1,
    "blur_overexposure": 109,
    "clear": 0,
    "unknown": 47,
}
EXPECTED_ACTION_COUNTS = {
    "blur_enhancement": 177,
    "overexposure_enhancement": 306,
    "underexposure_enhancement": 1,
    "overexposure_then_blur_enhancement": 109,
    "no_enhancement": 0,
    "defer_no_automatic_enhancement": 47,
}
ACTIONABLE_ACTIONS = {
    "no_enhancement",
    "blur_enhancement",
    "overexposure_enhancement",
    "underexposure_enhancement",
    "overexposure_then_blur_enhancement",
}
DEFER_ACTION = "defer_no_automatic_enhancement"
DATASET_CLASS_NAMES = {0: "car", 1: "truck"}
COCO_TO_DATASET = {2: 0, 7: 1}
DETECTOR_CONFIG = {
    "imgsz": 640,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 100,
    "device": 0,
    "batch": 1,
    "half": False,
    "classes": [2, 7],
    "save": False,
    "verbose": False,
}
SCOPE_FIELDS = [
    "image_id",
    "image_filename",
    "raw_image_path",
    "label_path",
    "original_source_split",
    "processed_split_provenance",
    "manual_degradation_reference",
    "vlm_degradation_canonical",
    "enhancement_action",
    "evaluation_scope",
    "included_raw_full",
    "included_paired_actionable",
    "deferred",
    "is_test_image",
]


@dataclass(frozen=True)
class GTObject:
    image_id: str
    class_id: int
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    image_id: str
    class_id: int
    score: float
    xyxy: tuple[float, float, float, float]


def run_command(command: list[str], cwd: Path = REPO_ROOT) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fields = ordered
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("_line_number", line_number)
            rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_test_path(path: str | Path) -> bool:
    normalized = str(path).replace("\\", "/").lower()
    return "/testing-data/" in normalized or normalized.startswith("testing-data/")


def assert_not_test_path(path: str | Path) -> None:
    if is_test_path(path):
        raise ValueError(f"test path is forbidden in C3: {path}")


def remap_coco_class(coco_class_id: int) -> int:
    if coco_class_id not in COCO_TO_DATASET:
        raise ValueError(f"unexpected native COCO class: {coco_class_id}")
    return COCO_TO_DATASET[coco_class_id]


def label_path_for_image(image_path: str | Path) -> Path:
    path = Path(image_path)
    if path.parent.name != "images":
        raise ValueError(f"cannot derive label path from non-images path: {image_path}")
    return path.parent.parent / "labels" / f"{path.stem}.txt"


def image_id_from_path(path: str | Path) -> str:
    return Path(path).stem


def is_actionable(action: str) -> bool:
    return action in ACTIONABLE_ACTIONS


def class_support_warnings(counts: dict[str, int]) -> list[str]:
    warnings: list[str] = []
    if counts.get("clear", 0) == 0:
        warnings.append("clear has zero VLM examples; no-enhancement action cannot be validated in C3.")
    if counts.get("underexposure", 0) == 1:
        warnings.append("underexposure has one VLM example; category-level conclusions are prohibited.")
    return warnings


def validate_dimension_preserved(raw_width: int, raw_height: int, output_width: int, output_height: int) -> bool:
    return raw_width == output_width and raw_height == output_height


def compare_hash_snapshots(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    changed = sorted(key for key, value in before.items() if after.get(key) != value)
    missing_after = sorted(key for key in before if key not in after)
    return {
        "unchanged": not changed and not missing_after,
        "changed": changed,
        "missing_after": missing_after,
    }


def ensure_paired_scope_ids(raw_ids: set[str], enhanced_ids: set[str], gt_ids: set[str]) -> None:
    if raw_ids != enhanced_ids or raw_ids != gt_ids:
        raise ValueError(
            "paired scope image IDs differ: "
            f"raw_only={sorted(raw_ids - enhanced_ids)[:5]}, "
            f"enhanced_only={sorted(enhanced_ids - raw_ids)[:5]}, "
            f"gt_mismatch={sorted((raw_ids ^ gt_ids))[:5]}"
        )


def ensure_metric_populations_not_mixed(left_name: str, left_ids: set[str], right_name: str, right_ids: set[str]) -> None:
    if left_ids != right_ids:
        raise ValueError(f"metric populations cannot be mixed: {left_name}={len(left_ids)} {right_name}={len(right_ids)}")


def preflight(run_dir: Path) -> dict[str, Any]:
    modules: dict[str, str | None] = {}
    for module_name in ["torch", "ultralytics", "PIL", "numpy", "cv2"]:
        module = importlib.import_module(module_name)
        modules[module_name] = getattr(module, "__file__", None)
    site_prefix = str(Path(EXPECTED_INTERPRETER).parents[1])
    outside_venv = {
        name: path for name, path in modules.items() if path is None or not str(Path(path).resolve()).startswith(site_prefix)
    }
    import site
    import torch
    import ultralytics

    cuda = {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "ultralytics_version": ultralytics.__version__,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    weight = {
        "path": str(WEIGHT_PATH),
        "exists": WEIGHT_PATH.exists(),
        "size_bytes": WEIGHT_PATH.stat().st_size if WEIGHT_PATH.exists() else None,
        "sha256": sha256_file(WEIGHT_PATH) if WEIGHT_PATH.exists() else None,
    }
    payload = {
        "commands": {
            "git_status_short": run_command(["git", "status", "--short"]),
            "git_diff_stat": run_command(["git", "diff", "--stat"]),
            "python_version": run_command([sys.executable, "--version"]),
        },
        "expected_interpreter": EXPECTED_INTERPRETER,
        "interpreter": {
            "sys_executable": sys.executable,
            "sys_prefix": sys.prefix,
            "sys_base_prefix": sys.base_prefix,
            "venv_active": sys.prefix != sys.base_prefix,
            "site_packages": site.getsitepackages(),
        },
        "modules": modules,
        "outside_venv": outside_venv,
        "cuda": cuda,
        "rtdetr_l_weight": weight,
        "inherited_system_pip_warning": True,
    }
    write_json(run_dir / "preflight_checks.json", payload)
    if sys.executable != EXPECTED_INTERPRETER:
        raise RuntimeError("wrong interpreter")
    if outside_venv:
        raise RuntimeError(f"modules resolve outside ag1_venv: {outside_venv}")
    if not cuda["cuda_available"]:
        raise RuntimeError("CUDA_NOT_AVAILABLE")
    if weight["size_bytes"] != EXPECTED_WEIGHT_SIZE or weight["sha256"] != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("RT-DETR-L checksum or size mismatch")
    return payload


def discover_c2_manifest(c2_run: Path = C2_RUN) -> tuple[Path, Path]:
    csv_candidates = sorted(c2_run.glob("*vlm_action_manifest.csv"))
    jsonl_candidates = sorted(c2_run.glob("*vlm_action_manifest.jsonl"))
    if len(csv_candidates) != 1 or len(jsonl_candidates) != 1:
        raise FileNotFoundError(f"expected one C2 action manifest CSV/JSONL in {c2_run}")
    return csv_candidates[0], jsonl_candidates[0]


def load_c2_manifest(c2_run: Path = C2_RUN) -> list[dict[str, Any]]:
    csv_path, jsonl_path = discover_c2_manifest(c2_run)
    rows = read_csv(csv_path)
    jsonl_rows = read_jsonl(jsonl_path)
    if len(rows) != len(jsonl_rows):
        raise ValueError("C2 CSV/JSONL manifest row counts differ")
    return [dict(row) for row in rows]


def bool_from_csv(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_c2_row(row: dict[str, Any]) -> dict[str, Any]:
    canonical = str(row["vlm_degradation_canonical"])
    action = str(row["enhancement_action"])
    if action != action_for_canonical_class(canonical):
        raise ValueError(f"action does not match canonical VLM class for image {row.get('image_id')}")
    raw_image_path = str(row["original_image_path"])
    label_path = label_path_for_image(raw_image_path)
    return {
        "image_id": str(row["image_id"]),
        "image_filename": str(row["image_filename"]),
        "raw_image_path": raw_image_path,
        "label_path": str(label_path),
        "original_source_split": str(row["original_source_split"]),
        "processed_split_provenance": str(row["processed_split_provenance"]),
        "manual_degradation_reference": str(row.get("manual_degradation_reference", "")),
        "vlm_degradation_canonical": canonical,
        "enhancement_action": action,
        "evaluation_scope": "deferred_unknown" if action == DEFER_ACTION else "paired_actionable",
        "included_raw_full": True,
        "included_paired_actionable": action != DEFER_ACTION,
        "deferred": action == DEFER_ACTION,
        "is_test_image": is_test_path(raw_image_path) or is_test_path(label_path),
    }


def verify_inputs(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    c1_rows = read_jsonl(C1_JSONL)
    c2_rows = load_c2_manifest(C2_RUN)
    c1_status = read_json(C1_RUN / "STATUS.json")
    c2_status = read_json(C2_RUN / "STATUS.json")
    c2_spec = read_json(C2_RUN / "ACTION_BANK_SPEC.json")
    c2_mapping = read_json(C2_RUN / "action_mapping_validation.json")

    c1_by_id = {str(row["image_id"]): row for row in c1_rows}
    c2_by_id = {str(row["image_id"]): row for row in c2_rows}
    normalized = [normalize_c2_row(row) for row in c2_rows]
    ids = [row["image_id"] for row in normalized]
    paths = [row["raw_image_path"] for row in normalized]
    class_counts = Counter(row["vlm_degradation_canonical"] for row in normalized)
    action_counts = Counter(row["enhancement_action"] for row in normalized)
    for canonical in EXPECTED_CLASS_COUNTS:
        class_counts.setdefault(canonical, 0)
    for action, expected_zero in EXPECTED_ACTION_COUNTS.items():
        action_counts.setdefault(action, expected_zero)
    split_counts = Counter(row["processed_split_provenance"] for row in normalized)
    source_splits = Counter(row["original_source_split"] for row in normalized)
    actionable = [row for row in normalized if row["included_paired_actionable"]]
    deferred = [row for row in normalized if row["deferred"]]
    failures: list[str] = []
    if len(c1_rows) != 640:
        failures.append("C1 record count is not 640")
    if len(c2_rows) != 640:
        failures.append("C2 action record count is not 640")
    if set(c1_by_id) != set(c2_by_id):
        failures.append("C1/C2 ID sets differ")
    if len(set(ids)) != 640:
        failures.append("duplicate image ID in C2 manifest")
    if len(set(paths)) != 640:
        failures.append("duplicate raw image path in C2 manifest")
    if dict(class_counts) != EXPECTED_CLASS_COUNTS:
        failures.append(f"class counts mismatch: {dict(class_counts)}")
    compact_action_counts = {key: action_counts.get(key, 0) for key in EXPECTED_ACTION_COUNTS if EXPECTED_ACTION_COUNTS[key] or action_counts.get(key, 0)}
    expected_compact = {key: value for key, value in EXPECTED_ACTION_COUNTS.items() if value}
    if compact_action_counts != expected_compact:
        failures.append(f"action counts mismatch: {compact_action_counts}")
    if len(actionable) != 593 or len(deferred) != 47:
        failures.append(f"actionable/deferred mismatch: {len(actionable)}/{len(deferred)}")
    if split_counts.get("train_core", 0) != 544 or split_counts.get("val_from_train", 0) != 96:
        failures.append(f"provenance split mismatch: {dict(split_counts)}")
    if source_splits != Counter({"training-data": 640}):
        failures.append(f"source split mismatch: {dict(source_splits)}")
    if any(row["is_test_image"] for row in normalized):
        failures.append("test path present")
    missing_sources = [row["raw_image_path"] for row in normalized if not Path(row["raw_image_path"]).exists()]
    missing_labels = [row["label_path"] for row in normalized if not Path(row["label_path"]).exists()]
    if missing_sources:
        failures.append(f"missing source images: {missing_sources[:3]}")
    if missing_labels:
        failures.append(f"missing label files: {missing_labels[:3]}")
    if any(str(row.get("action_source", "")) != "vlm_prediction_only" for row in c2_rows):
        failures.append("C2 action source was not vlm_prediction_only for every row")
    for row in normalized:
        c1 = c1_by_id.get(row["image_id"])
        if c1 is None or str(c1.get("image_path")) != row["raw_image_path"]:
            failures.append(f"C1/C2 join mismatch for {row['image_id']}")
            break
    if failures:
        raise RuntimeError("; ".join(failures))

    write_csv(run_dir / "evaluation_scope_manifest.csv", normalized, SCOPE_FIELDS)
    write_jsonl(run_dir / "evaluation_scope_manifest.jsonl", normalized)
    payload = {
        "source_c1_run": str(C1_RUN),
        "source_c2_run": str(C2_RUN),
        "authoritative_c1_jsonl": str(C1_JSONL),
        "authoritative_c2_manifest_csv": str(discover_c2_manifest(C2_RUN)[0]),
        "authoritative_c2_manifest_jsonl": str(discover_c2_manifest(C2_RUN)[1]),
        "c1_status": c1_status.get("status"),
        "c2_status": c2_status.get("status"),
        "c2_spec_stage": c2_spec.get("stage"),
        "c2_best_of_bank_active": c2_mapping.get("best_of_bank_active"),
        "c1_record_count": len(c1_rows),
        "c2_action_record_count": len(c2_rows),
        "unique_image_id_count": len(set(ids)),
        "unique_raw_image_path_count": len(set(paths)),
        "class_counts": dict(sorted(class_counts.items())),
        "action_counts": dict(sorted((key, action_counts.get(key, 0)) for key in EXPECTED_ACTION_COUNTS if action_counts.get(key, 0))),
        "actionable_count": len(actionable),
        "deferred_unknown_count": len(deferred),
        "train_core_provenance_count": split_counts.get("train_core", 0),
        "val_from_train_provenance_count": split_counts.get("val_from_train", 0),
        "original_source_split_counts": dict(source_splits),
        "test_image_count": sum(1 for row in normalized if row["is_test_image"]),
        "manual_label_used_for_action": False,
        "all_source_images_exist": True,
        "all_label_paths_exist": True,
    }
    write_json(run_dir / "input_integrity.json", payload)
    return normalized, payload


def parse_yolo_label(label_path: Path, image_id: str, width: int, height: int) -> list[GTObject]:
    assert_not_test_path(label_path)
    objects: list[GTObject] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return objects
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"invalid YOLO label line {label_path}:{line_number}")
        class_id = int(float(parts[0]))
        if class_id not in DATASET_CLASS_NAMES:
            raise ValueError(f"unexpected dataset class {class_id} in {label_path}")
        cx, cy, bw, bh = [float(value) for value in parts[1:]]
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        objects.append(
            GTObject(
                image_id=image_id,
                class_id=class_id,
                xyxy=(
                    max(0.0, min(float(width), x1)),
                    max(0.0, min(float(height), y1)),
                    max(0.0, min(float(width), x2)),
                    max(0.0, min(float(height), y2)),
                ),
            )
        )
    return objects


def image_shape(path: Path) -> tuple[int, int, int]:
    assert_not_test_path(path)
    with Image.open(path) as image:
        bands = image.getbands()
        return image.width, image.height, len(bands)


def snapshot_sources(run_dir: Path, scope_rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, list[GTObject]], dict[str, Any]]:
    image_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    image_meta: dict[str, dict[str, Any]] = {}
    gt_by_image: dict[str, list[GTObject]] = {}
    gt_counts = Counter()
    for row in scope_rows:
        image_id = row["image_id"]
        raw_path = Path(row["raw_image_path"])
        label_path = Path(row["label_path"])
        assert_not_test_path(raw_path)
        assert_not_test_path(label_path)
        width, height, channels = image_shape(raw_path)
        image_hash = sha256_file(raw_path)
        label_hash = sha256_file(label_path)
        objects = parse_yolo_label(label_path, image_id, width, height)
        for obj in objects:
            gt_counts[obj.class_id] += 1
        image_meta[image_id] = {
            "image_id": image_id,
            "raw_image_path": str(raw_path),
            "label_path": str(label_path),
            "width": width,
            "height": height,
            "channel_count": channels,
            "image_sha256": image_hash,
            "label_sha256": label_hash,
            "gt_object_count": len(objects),
        }
        gt_by_image[image_id] = objects
        image_rows.append(
            {
                "image_id": image_id,
                "raw_image_path": str(raw_path),
                "sha256": image_hash,
                "width": width,
                "height": height,
                "channel_count": channels,
            }
        )
        label_rows.append(
            {
                "image_id": image_id,
                "label_path": str(label_path),
                "sha256": label_hash,
                "gt_object_count": len(objects),
                "gt_car_count": sum(1 for obj in objects if obj.class_id == 0),
                "gt_truck_count": sum(1 for obj in objects if obj.class_id == 1),
            }
        )
    write_csv(run_dir / "source_image_hashes_before.csv", image_rows)
    write_csv(run_dir / "source_label_hashes_before.csv", label_rows)
    inventory = {
        "image_count": len(scope_rows),
        "total_gt_objects": sum(gt_counts.values()),
        "gt_objects_by_class": {str(key): value for key, value in sorted(gt_counts.items())},
        "gt_objects_by_class_name": {DATASET_CLASS_NAMES[key]: gt_counts.get(key, 0) for key in DATASET_CLASS_NAMES},
        "classes_only_car_truck": set(gt_counts).issubset(set(DATASET_CLASS_NAMES)),
    }
    write_json(run_dir / "ground_truth_inventory.json", inventory)
    return image_meta, gt_by_image, inventory


def validate_output_image(path: Path, expected_width: int, expected_height: int) -> dict[str, Any]:
    with Image.open(path) as output:
        image = output.copy()
    arr = np.asarray(image)
    channels = 1 if arr.ndim == 2 else int(arr.shape[2]) if arr.ndim == 3 else 0
    return {
        "readable": True,
        "width": image.width,
        "height": image.height,
        "channel_count": channels,
        "dimensions_preserved": validate_dimension_preserved(expected_width, expected_height, image.width, image.height),
        "pixel_values_finite": bool(np.isfinite(arr).all()),
        "pixel_values_in_expected_range": bool(arr.size > 0 and int(arr.min()) >= 0 and int(arr.max()) <= 255),
    }


def materialize_enhancements(
    run_dir: Path,
    scope_rows: list[dict[str, Any]],
    image_meta: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_dir = run_dir / "images/enhanced"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "full_enhancement_manifest.csv"
    jsonl_path = run_dir / "full_enhancement_manifest.jsonl"
    latency_path = run_dir / "enhancement_latency_per_image.csv"
    failures: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    fields = [
        "image_id",
        "image_filename",
        "raw_image_path",
        "enhanced_image_path",
        "vlm_degradation_canonical",
        "enhancement_action",
        "parameters",
        "operation_order",
        "operation_log",
        "source_hash_before",
        "source_hash_after",
        "source_hash_unchanged",
        "latency_ms",
        "status",
        "error",
        "width",
        "height",
        "channel_count",
        "dimensions_preserved",
        "pixel_values_finite",
        "pixel_values_in_expected_range",
    ]
    actionable = [row for row in scope_rows if row["included_paired_actionable"]]
    deferred = [row for row in scope_rows if row["deferred"]]
    with manifest_path.open("w", newline="", encoding="utf-8") as mf, jsonl_path.open("w", encoding="utf-8") as jf, latency_path.open(
        "w", newline="", encoding="utf-8"
    ) as lf:
        manifest_writer = csv.DictWriter(mf, fieldnames=fields)
        latency_writer = csv.DictWriter(lf, fieldnames=["image_id", "enhancement_action", "vlm_degradation_canonical", "latency_ms"])
        manifest_writer.writeheader()
        latency_writer.writeheader()
        for row in actionable:
            image_id = row["image_id"]
            raw_path = Path(row["raw_image_path"])
            action = row["enhancement_action"]
            canonical = row["vlm_degradation_canonical"]
            expected_action = action_for_canonical_class(canonical)
            if action != expected_action:
                raise RuntimeError(f"action changed for {image_id}: {action} != {expected_action}")
            out_path = output_dir / f"{image_id}.png"
            source_before = sha256_file(raw_path)
            error = ""
            status = "success"
            operation_log: list[str] = []
            latency_ms = 0.0
            validation: dict[str, Any] = {}
            try:
                start = time.perf_counter()
                with Image.open(raw_path) as image:
                    raw_image = image.copy()
                enhanced = apply_enhancement_action(raw_image, action, operation_log)
                if enhanced is None:
                    raise RuntimeError(f"action {action} did not create an enhanced image")
                enhanced.save(out_path, format="PNG")
                latency_ms = (time.perf_counter() - start) * 1000.0
                validation = validate_output_image(out_path, int(image_meta[image_id]["width"]), int(image_meta[image_id]["height"]))
                if not all(
                    [
                        validation["readable"],
                        validation["dimensions_preserved"],
                        validation["pixel_values_finite"],
                        validation["pixel_values_in_expected_range"],
                    ]
                ):
                    status = "failed_validation"
            except Exception as exc:
                status = "failed_exception"
                error = f"{type(exc).__name__}: {exc}"
            source_after = sha256_file(raw_path)
            record = {
                "image_id": image_id,
                "image_filename": row["image_filename"],
                "raw_image_path": str(raw_path),
                "enhanced_image_path": str(out_path) if out_path.exists() else "",
                "vlm_degradation_canonical": canonical,
                "enhancement_action": action,
                "parameters": json.dumps(parameters_for_action(action), sort_keys=True),
                "operation_order": json.dumps(operation_order_for_action(action)),
                "operation_log": json.dumps(operation_log),
                "source_hash_before": source_before,
                "source_hash_after": source_after,
                "source_hash_unchanged": source_before == source_after,
                "latency_ms": latency_ms,
                "status": status,
                "error": error,
                "width": validation.get("width", ""),
                "height": validation.get("height", ""),
                "channel_count": validation.get("channel_count", ""),
                "dimensions_preserved": validation.get("dimensions_preserved", False),
                "pixel_values_finite": validation.get("pixel_values_finite", False),
                "pixel_values_in_expected_range": validation.get("pixel_values_in_expected_range", False),
            }
            manifest_writer.writerow(record)
            jf.write(json.dumps(record, sort_keys=True) + "\n")
            latency_record = {
                "image_id": image_id,
                "enhancement_action": action,
                "vlm_degradation_canonical": canonical,
                "latency_ms": latency_ms,
            }
            latency_writer.writerow(latency_record)
            manifest_rows.append(record)
            latency_rows.append(latency_record)
            if status != "success" or source_before != source_after:
                failures.append(record)
    deferred_outputs = [row for row in deferred if (output_dir / f"{row['image_id']}.png").exists()]
    count_by_class = Counter(row["vlm_degradation_canonical"] for row in manifest_rows if row["status"] == "success")
    summary = {
        "success_count": sum(1 for row in manifest_rows if row["status"] == "success"),
        "failure_count": len(failures),
        "deferred_count": len(deferred),
        "deferred_output_count": len(deferred_outputs),
        "output_counts_by_vlm_class": dict(sorted(count_by_class.items())),
        "expected_output_counts": {
            "blur": 177,
            "overexposure": 306,
            "underexposure": 1,
            "blur_overexposure": 109,
            "clear": 0,
            "total": 593,
            "deferred": 0,
        },
    }
    write_csv(run_dir / "enhancement_failures.csv", failures, fields)
    write_json(run_dir / "enhancement_latency_summary.json", summarize_enhancement_latency(latency_rows))
    if failures or summary["success_count"] != 593 or summary["deferred_output_count"] != 0:
        raise RuntimeError(f"enhancement generation failed: {summary}")
    return manifest_rows, summary


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = min(len(ordered) - 1, max(0, math.ceil((len(ordered) - 1) * p)))
        return ordered[idx]

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": ordered[-1],
    }


def summarize_enhancement_latency(latency_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_action: dict[str, list[float]] = defaultdict(list)
    by_class: dict[str, list[float]] = defaultdict(list)
    all_values: list[float] = []
    for row in latency_rows:
        value = float(row["latency_ms"])
        all_values.append(value)
        by_action[str(row["enhancement_action"])].append(value)
        by_class[str(row["vlm_degradation_canonical"])].append(value)
    return {
        "measurement_notes": {
            "includes_image_decode": True,
            "includes_enhancement_transform": True,
            "includes_output_png_disk_write": True,
            "measured_separately_from_detector": True,
        },
        "overall": summarize_values(all_values),
        "by_action": {key: summarize_values(values) for key, values in sorted(by_action.items())},
        "by_vlm_class": {key: summarize_values(values) for key, values in sorted(by_class.items())},
    }


def recheck_source_integrity(
    run_dir: Path,
    scope_rows: list[dict[str, Any]],
    image_meta: dict[str, dict[str, Any]],
    filename: str,
) -> dict[str, Any]:
    before_image = {image_id: str(meta["image_sha256"]) for image_id, meta in image_meta.items()}
    before_label = {image_id: str(meta["label_sha256"]) for image_id, meta in image_meta.items()}
    after_image = {row["image_id"]: sha256_file(Path(row["raw_image_path"])) for row in scope_rows}
    after_label = {row["image_id"]: sha256_file(Path(row["label_path"])) for row in scope_rows}
    image_check = compare_hash_snapshots(before_image, after_image)
    label_check = compare_hash_snapshots(before_label, after_label)
    payload = {
        "source_images": image_check,
        "source_labels": label_check,
        "all_source_hashes_unchanged": image_check["unchanged"] and label_check["unchanged"],
        "source_data_modified": not (image_check["unchanged"] and label_check["unchanged"]),
        "test_image_accessed": False,
    }
    write_json(run_dir / filename, payload)
    return payload


def write_gt_integrity(
    run_dir: Path,
    scope_rows: list[dict[str, Any]],
    gt_by_image: dict[str, list[GTObject]],
) -> dict[str, Any]:
    all_ids = {row["image_id"] for row in scope_rows}
    actionable_ids = {row["image_id"] for row in scope_rows if row["included_paired_actionable"]}
    raw_gt_count = sum(len(gt_by_image[image_id]) for image_id in all_ids)
    paired_gt_count = sum(len(gt_by_image[image_id]) for image_id in actionable_ids)
    classes = sorted({obj.class_id for objects in gt_by_image.values() for obj in objects})
    payload = {
        "raw_full_image_count": len(all_ids),
        "paired_actionable_image_count": len(actionable_ids),
        "raw_full_gt_object_count": raw_gt_count,
        "paired_actionable_raw_gt_object_count": paired_gt_count,
        "paired_actionable_enhanced_gt_object_count": paired_gt_count,
        "raw_enhanced_gt_mapping_identical": True,
        "classes": classes,
        "classes_only_car_truck": set(classes).issubset(set(DATASET_CLASS_NAMES)),
        "test_record_count": sum(1 for row in scope_rows if row["is_test_image"]),
    }
    write_json(run_dir / "gt_integrity.json", payload)
    if payload["raw_full_image_count"] != 640 or payload["paired_actionable_image_count"] != 593:
        raise RuntimeError("GT scope counts are invalid")
    if not payload["classes_only_car_truck"] or payload["test_record_count"] != 0:
        raise RuntimeError("GT integrity validation failed")
    return payload


def flatten_gt(gt_by_image: dict[str, list[GTObject]], image_ids: set[str]) -> list[GTObject]:
    return [obj for image_id in sorted(image_ids) for obj in gt_by_image.get(image_id, [])]


def flatten_predictions(records: Sequence[dict[str, Any]], image_ids: set[str] | None = None) -> list[Prediction]:
    output: list[Prediction] = []
    for record in records:
        image_id = str(record["image_id"])
        if image_ids is not None and image_id not in image_ids:
            continue
        for pred in record.get("predictions", []):
            output.append(
                Prediction(
                    image_id=image_id,
                    class_id=int(pred["dataset_class_id"]),
                    score=float(pred["confidence"]),
                    xyxy=tuple(float(value) for value in pred["xyxy"]),  # type: ignore[arg-type]
                )
            )
    return output


def match_predictions(
    gts: Sequence[GTObject],
    preds: Sequence[Prediction],
    iou_threshold: float,
    class_id: int | None = None,
) -> dict[str, Any]:
    selected_gts = [gt for gt in gts if class_id is None or gt.class_id == class_id]
    selected_preds = [pred for pred in preds if class_id is None or pred.class_id == class_id]
    gt_by_key: dict[tuple[str, int], list[GTObject]] = defaultdict(list)
    for gt in selected_gts:
        gt_by_key[(gt.image_id, gt.class_id)].append(gt)
    matched: set[tuple[str, int, int]] = set()
    tp = 0
    fp = 0
    for pred in sorted(selected_preds, key=lambda item: item.score, reverse=True):
        candidates = gt_by_key.get((pred.image_id, pred.class_id), [])
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(candidates):
            key = (gt.image_id, gt.class_id, idx)
            if key in matched:
                continue
            score = iou_xyxy(pred.xyxy, gt.xyxy)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_iou >= iou_threshold and best_idx >= 0:
            matched.add((pred.image_id, pred.class_id, best_idx))
            tp += 1
        else:
            fp += 1
    fn = max(0, len(selected_gts) - tp)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt": len(selected_gts),
        "predictions": len(selected_preds),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, len(selected_gts)),
    }


def average_precision_for_class(
    gts: Sequence[GTObject],
    preds: Sequence[Prediction],
    class_id: int,
    iou_threshold: float,
) -> float | None:
    class_gts = [gt for gt in gts if gt.class_id == class_id]
    class_preds = [pred for pred in preds if pred.class_id == class_id]
    if not class_gts:
        return None
    gt_by_image: dict[str, list[GTObject]] = defaultdict(list)
    for gt in class_gts:
        gt_by_image[gt.image_id].append(gt)
    matched: set[tuple[str, int]] = set()
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    for pred in sorted(class_preds, key=lambda item: item.score, reverse=True):
        candidates = gt_by_image.get(pred.image_id, [])
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(candidates):
            key = (pred.image_id, idx)
            if key in matched:
                continue
            score = iou_xyxy(pred.xyxy, gt.xyxy)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_iou >= iou_threshold and best_idx >= 0:
            matched.add((pred.image_id, best_idx))
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)
    if not tp_flags:
        return 0.0
    cum_tp = np.cumsum(tp_flags)
    cum_fp = np.cumsum(fp_flags)
    recalls = cum_tp / max(1, len(class_gts))
    precisions = cum_tp / np.maximum(1, cum_tp + cum_fp)
    sampled = []
    for threshold in np.linspace(0.0, 1.0, 101):
        eligible = precisions[recalls >= threshold]
        sampled.append(float(np.max(eligible)) if eligible.size else 0.0)
    return float(np.mean(sampled))


def prediction_count_stats(records: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    counts = [int(record.get("prediction_count", 0)) for record in records]
    if not counts:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0}
    return {
        "mean": statistics.fmean(counts),
        "median": statistics.median(counts),
        "p95": summarize_values([float(value) for value in counts])["p95"],
        "max": max(counts),
    }


def evaluate_scope(
    scope_name: str,
    image_ids: set[str],
    gt_by_image: dict[str, list[GTObject]],
    prediction_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    records = [record for record in prediction_records if str(record["image_id"]) in image_ids]
    gts = flatten_gt(gt_by_image, image_ids)
    preds = flatten_predictions(records, image_ids)
    thresholds = [0.50 + 0.05 * idx for idx in range(10)]
    class_metrics: dict[str, Any] = {}
    ap50_values = []
    ap5095_values = []
    for class_id, class_name in DATASET_CLASS_NAMES.items():
        aps = [average_precision_for_class(gts, preds, class_id, threshold) for threshold in thresholds]
        ap50 = aps[0]
        valid_aps = [value for value in aps if value is not None]
        ap5095 = statistics.fmean(valid_aps) if valid_aps else None
        match = match_predictions(gts, preds, 0.50, class_id=class_id)
        class_metrics[class_name] = {
            "class_id": class_id,
            "gt_object_count": match["gt"],
            "prediction_count": match["predictions"],
            "AP50": ap50,
            "AP50-95": ap5095,
            "precision_conf_0.001_iou_0.50": match["precision"],
            "recall_conf_0.001_iou_0.50": match["recall"],
            "FN_iou_0.50": match["fn"],
            "TP_iou_0.50": match["tp"],
            "FP_iou_0.50": match["fp"],
        }
        if ap50 is not None:
            ap50_values.append(ap50)
        if ap5095 is not None:
            ap5095_values.append(ap5095)
    aggregate_match = match_predictions(gts, preds, 0.50)
    saturation_count = sum(1 for record in records if bool(record.get("prediction_count_reached_max_det")))
    return {
        "scope": scope_name,
        "image_count": len(image_ids),
        "gt_object_count": len(gts),
        "prediction_count": len(preds),
        "mAP50": statistics.fmean(ap50_values) if ap50_values else 0.0,
        "mAP50-95": statistics.fmean(ap5095_values) if ap5095_values else 0.0,
        "precision_conf_0.001_iou_0.50": aggregate_match["precision"],
        "recall_conf_0.001_iou_0.50": aggregate_match["recall"],
        "FN_iou_0.50": aggregate_match["fn"],
        "TP_iou_0.50": aggregate_match["tp"],
        "FP_iou_0.50": aggregate_match["fp"],
        "prediction_count_stats": prediction_count_stats(records),
        "max_det_saturation_count": saturation_count,
        "max_det_saturation_fraction": saturation_count / max(1, len(records)),
        "per_class": class_metrics,
        "metric_notes": "Precision/recall are measured at conf=0.001 for candidate retention, not a deployment threshold.",
    }


def matched_gt_count_by_image(
    image_ids: set[str],
    gt_by_image: dict[str, list[GTObject]],
    prediction_records: Sequence[dict[str, Any]],
) -> dict[str, int]:
    output: dict[str, int] = {}
    by_record = {str(record["image_id"]): record for record in prediction_records}
    for image_id in image_ids:
        gts = gt_by_image.get(image_id, [])
        preds = flatten_predictions([by_record[image_id]], {image_id}) if image_id in by_record else []
        output[image_id] = int(match_predictions(gts, preds, 0.50)["tp"])
    return output


def prediction_records_by_id(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["image_id"]): record for record in records}


def write_prediction_artifacts(
    base_path: Path,
    records: Sequence[dict[str, Any]],
) -> None:
    write_jsonl(base_path.with_suffix(".jsonl"), records)
    csv_rows: list[dict[str, Any]] = []
    for record in records:
        common = {
            "branch": record["branch"],
            "image_id": record["image_id"],
            "image_filename": record["image_filename"],
            "source_path": record["source_path"],
            "original_raw_image_path": record.get("original_raw_image_path", ""),
            "vlm_degradation_canonical": record["vlm_degradation_canonical"],
            "enhancement_action": record["enhancement_action"],
            "processed_split_provenance": record["processed_split_provenance"],
            "width": record["width"],
            "height": record["height"],
            "prediction_count": record["prediction_count"],
            "prediction_count_reached_max_det": record["prediction_count_reached_max_det"],
            "wall_clock_ms": record["latency_ms"]["wall_clock_ms"],
            "ultralytics_preprocess_ms": record["latency_ms"].get("ultralytics_preprocess_ms", ""),
            "ultralytics_inference_ms": record["latency_ms"].get("ultralytics_inference_ms", ""),
            "ultralytics_postprocess_ms": record["latency_ms"].get("ultralytics_postprocess_ms", ""),
        }
        if not record["predictions"]:
            csv_rows.append({**common, "native_coco_class_id": "", "dataset_class_id": "", "confidence": "", "x1": "", "y1": "", "x2": "", "y2": ""})
        for pred in record["predictions"]:
            csv_rows.append(
                {
                    **common,
                    "native_coco_class_id": pred["native_coco_class_id"],
                    "native_coco_class_name": pred["native_coco_class_name"],
                    "dataset_class_id": pred["dataset_class_id"],
                    "dataset_class_name": pred["dataset_class_name"],
                    "confidence": pred["confidence"],
                    "x1": pred["xyxy"][0],
                    "y1": pred["xyxy"][1],
                    "x2": pred["xyxy"][2],
                    "y2": pred["xyxy"][3],
                }
            )
    write_csv(base_path.with_suffix(".csv"), csv_rows)


def write_latency_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    rows = []
    for record in records:
        latency = record["latency_ms"]
        rows.append(
            {
                "image_id": record["image_id"],
                "branch": record["branch"],
                "prediction_count": record["prediction_count"],
                "wall_clock_ms": latency["wall_clock_ms"],
                "ultralytics_preprocess_ms": latency.get("ultralytics_preprocess_ms", ""),
                "ultralytics_inference_ms": latency.get("ultralytics_inference_ms", ""),
                "ultralytics_postprocess_ms": latency.get("ultralytics_postprocess_ms", ""),
                "ultralytics_total_ms": latency.get("ultralytics_total_ms", ""),
            }
        )
    write_csv(path, rows)


def detector_predict_one(model: Any, row: dict[str, Any], source_path: Path, branch: str) -> dict[str, Any]:
    assert_not_test_path(source_path)
    cuda_synchronize()
    start = time.perf_counter()
    results = model.predict(source=str(source_path), **DETECTOR_CONFIG)
    cuda_synchronize()
    wall_ms = (time.perf_counter() - start) * 1000.0
    if len(results) != 1:
        raise RuntimeError(f"expected one result for {source_path}, got {len(results)}")
    result = results[0]
    boxes = getattr(result, "boxes", None)
    predictions: list[dict[str, Any]] = []
    if boxes is not None and boxes.xyxy is not None:
        xyxy_values = boxes.xyxy.detach().cpu().tolist()
        conf_values = boxes.conf.detach().cpu().tolist()
        cls_values = boxes.cls.detach().cpu().tolist()
        for xyxy, confidence, native_class in zip(xyxy_values, conf_values, cls_values):
            native_class_id = int(native_class)
            dataset_class_id = remap_coco_class(native_class_id)
            predictions.append(
                {
                    "native_coco_class_id": native_class_id,
                    "native_coco_class_name": model.names[native_class_id],
                    "dataset_class_id": dataset_class_id,
                    "dataset_class_name": DATASET_CLASS_NAMES[dataset_class_id],
                    "confidence": float(confidence),
                    "xyxy": [float(value) for value in xyxy],
                }
            )
    speed = getattr(result, "speed", {}) or {}
    width = int(row.get("width", 0) or 0)
    height = int(row.get("height", 0) or 0)
    return {
        "branch": branch,
        "image_id": row["image_id"],
        "image_filename": row["image_filename"],
        "source_path": str(source_path),
        "original_raw_image_path": row["raw_image_path"],
        "vlm_degradation_canonical": row["vlm_degradation_canonical"],
        "enhancement_action": row["enhancement_action"],
        "processed_split_provenance": row["processed_split_provenance"],
        "original_source_split": row["original_source_split"],
        "width": width,
        "height": height,
        "prediction_count": len(predictions),
        "prediction_count_reached_max_det": len(predictions) >= int(DETECTOR_CONFIG["max_det"]),
        "latency_ms": {
            "wall_clock_ms": wall_ms,
            "ultralytics_preprocess_ms": float(speed.get("preprocess", 0.0)),
            "ultralytics_inference_ms": float(speed.get("inference", 0.0)),
            "ultralytics_postprocess_ms": float(speed.get("postprocess", 0.0)),
            "ultralytics_total_ms": float(speed.get("preprocess", 0.0)) + float(speed.get("inference", 0.0)) + float(speed.get("postprocess", 0.0)),
        },
        "predictions": predictions,
    }


def load_detector_and_warmup(run_dir: Path, warmup_row: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    from ultralytics import RTDETR

    model = RTDETR(str(WEIGHT_PATH))
    if model.names.get(2) != "car" or model.names.get(7) != "truck":
        raise RuntimeError(f"unexpected RT-DETR class names: {model.names.get(2)}, {model.names.get(7)}")
    warmup_calls = 2
    warmup_path = Path(warmup_row["raw_image_path"])
    assert_not_test_path(warmup_path)
    for _ in range(warmup_calls):
        model.predict(source=str(warmup_path), **DETECTOR_CONFIG)
        cuda_synchronize()
    device = ""
    dtype = ""
    try:
        param = next(model.model.parameters())
        device = str(param.device)
        dtype = str(param.dtype)
    except Exception:
        pass
    payload = {
        "model_load_count": 1,
        "warmup_call_count": warmup_calls,
        "warmup_image_id": warmup_row["image_id"],
        "warmup_image_path": str(warmup_path),
        "detector_config": DETECTOR_CONFIG,
        "model_names": {"2": model.names.get(2), "7": model.names.get(7)},
        "model_parameter_device_after_warmup": device,
        "model_parameter_dtype_after_warmup": dtype,
        "fp32_requested": True,
    }
    write_json(run_dir / "detector_load_audit.json", payload)
    return model, payload


def run_detector_branch(
    model: Any,
    rows: list[dict[str, Any]],
    source_by_id: dict[str, Path],
    branch: str,
    out_base: Path,
    latency_path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    out_base.parent.mkdir(parents=True, exist_ok=True)
    latency_path.parent.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows, start=1):
        source = source_by_id[row["image_id"]]
        record = detector_predict_one(model, row, source, branch)
        records.append(record)
        if idx % 50 == 0:
            print(f"{branch}: processed {idx}/{len(rows)}", flush=True)
    write_prediction_artifacts(out_base, records)
    write_latency_csv(latency_path, records)
    if len({record["image_id"] for record in records}) != len(rows):
        raise RuntimeError(f"duplicate or missing IDs in {branch}")
    return records


def write_raw_paired_view(run_dir: Path, raw_records: list[dict[str, Any]], actionable_ids: set[str]) -> list[dict[str, Any]]:
    paired = [record for record in raw_records if str(record["image_id"]) in actionable_ids]
    if len(paired) != 593:
        raise RuntimeError(f"raw paired count is {len(paired)}, expected 593")
    write_prediction_artifacts(run_dir / "predictions/raw_paired_actionable_593", paired)
    return paired


def write_metric_artifacts(
    run_dir: Path,
    scope_rows: list[dict[str, Any]],
    gt_by_image: dict[str, list[GTObject]],
    raw_full_records: list[dict[str, Any]],
    raw_paired_records: list[dict[str, Any]],
    enhanced_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    all_ids = {row["image_id"] for row in scope_rows}
    actionable_ids = {row["image_id"] for row in scope_rows if row["included_paired_actionable"]}
    enhanced_ids = {record["image_id"] for record in enhanced_records}
    raw_paired_ids = {record["image_id"] for record in raw_paired_records}
    ensure_paired_scope_ids(raw_paired_ids, enhanced_ids, actionable_ids)
    raw_full_metrics = evaluate_scope("raw_full_640", all_ids, gt_by_image, raw_full_records)
    raw_paired_metrics = evaluate_scope("raw_paired_actionable_593", actionable_ids, gt_by_image, raw_paired_records)
    enhanced_metrics = evaluate_scope("enhanced_actionable_593", actionable_ids, gt_by_image, enhanced_records)
    metrics_dir = run_dir / "metrics"
    write_json(metrics_dir / "raw_full_640.json", raw_full_metrics)
    write_json(metrics_dir / "raw_paired_actionable_593.json", raw_paired_metrics)
    write_json(metrics_dir / "enhanced_actionable_593.json", enhanced_metrics)
    delta = paired_delta(raw_paired_metrics, enhanced_metrics)
    write_json(metrics_dir / "paired_delta.json", delta)
    by_class_rows = by_vlm_class_metrics(scope_rows, gt_by_image, raw_paired_records, enhanced_records)
    write_csv(metrics_dir / "by_vlm_class.csv", by_class_rows)
    write_csv(metrics_dir / "per_class_car_truck.csv", per_class_rows(raw_full_metrics, raw_paired_metrics, enhanced_metrics))
    write_csv(metrics_dir / "prediction_count_saturation.csv", saturation_rows(raw_full_metrics, raw_paired_metrics, enhanced_metrics))
    diagnostics = per_image_diagnostics(scope_rows, gt_by_image, raw_paired_records, enhanced_records)
    write_csv(metrics_dir / "per_image_paired_diagnostics.csv", diagnostics)
    write_json(
        metrics_dir / "paired_uncertainty_interval.json",
        {
            "paired_bootstrap_ci_produced": False,
            "reason": "No existing reliable paired-bootstrap implementation is active in this project; C3 does not add a statistics framework.",
        },
    )
    write_json(
        run_dir / "paired_id_integrity.json",
        {
            "raw_paired_count": len(raw_paired_ids),
            "enhanced_count": len(enhanced_ids),
            "gt_paired_count": len(actionable_ids),
            "sets_equal": raw_paired_ids == enhanced_ids == actionable_ids,
        },
    )
    return raw_full_metrics, raw_paired_metrics, enhanced_metrics, delta, diagnostics


def paired_delta(raw: dict[str, Any], enhanced: dict[str, Any]) -> dict[str, Any]:
    keys = ["mAP50", "mAP50-95", "precision_conf_0.001_iou_0.50", "recall_conf_0.001_iou_0.50", "FN_iou_0.50", "prediction_count", "max_det_saturation_count"]
    delta = {key: enhanced[key] - raw[key] for key in keys}
    delta["prediction_count_mean"] = enhanced["prediction_count_stats"]["mean"] - raw["prediction_count_stats"]["mean"]
    delta["prediction_count_median"] = enhanced["prediction_count_stats"]["median"] - raw["prediction_count_stats"]["median"]
    delta["per_class"] = {}
    for class_name in DATASET_CLASS_NAMES.values():
        delta["per_class"][class_name] = {}
        for metric in ["AP50", "AP50-95", "precision_conf_0.001_iou_0.50", "recall_conf_0.001_iou_0.50", "FN_iou_0.50"]:
            left = raw["per_class"][class_name][metric]
            right = enhanced["per_class"][class_name][metric]
            delta["per_class"][class_name][metric] = None if left is None or right is None else right - left
    return delta


def by_vlm_class_metrics(
    scope_rows: list[dict[str, Any]],
    gt_by_image: dict[str, list[GTObject]],
    raw_records: list[dict[str, Any]],
    enhanced_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for canonical in ["blur", "overexposure", "blur_overexposure", "underexposure"]:
        image_ids = {row["image_id"] for row in scope_rows if row["vlm_degradation_canonical"] == canonical and row["included_paired_actionable"]}
        if not image_ids:
            continue
        raw_metrics = evaluate_scope(f"raw_{canonical}", image_ids, gt_by_image, raw_records)
        enhanced_metrics = evaluate_scope(f"enhanced_{canonical}", image_ids, gt_by_image, enhanced_records)
        for branch, metrics in [("raw", raw_metrics), ("enhanced", enhanced_metrics)]:
            rows.append(
                {
                    "vlm_degradation_canonical": canonical,
                    "branch": branch,
                    "image_count": metrics["image_count"],
                    "gt_object_count": metrics["gt_object_count"],
                    "prediction_count": metrics["prediction_count"],
                    "mAP50": metrics["mAP50"],
                    "mAP50-95": metrics["mAP50-95"],
                    "precision_conf_0.001_iou_0.50": metrics["precision_conf_0.001_iou_0.50"],
                    "recall_conf_0.001_iou_0.50": metrics["recall_conf_0.001_iou_0.50"],
                    "FN_iou_0.50": metrics["FN_iou_0.50"],
                    "prediction_count_mean": metrics["prediction_count_stats"]["mean"],
                    "max_det_saturation_count": metrics["max_det_saturation_count"],
                    "warning": "descriptive only; n=1 is not statistically meaningful" if canonical == "underexposure" else "",
                }
            )
        delta = paired_delta(raw_metrics, enhanced_metrics)
        rows.append(
            {
                "vlm_degradation_canonical": canonical,
                "branch": "enhanced_minus_raw",
                "image_count": len(image_ids),
                "gt_object_count": raw_metrics["gt_object_count"],
                "prediction_count": delta["prediction_count"],
                "mAP50": delta["mAP50"],
                "mAP50-95": delta["mAP50-95"],
                "precision_conf_0.001_iou_0.50": delta["precision_conf_0.001_iou_0.50"],
                "recall_conf_0.001_iou_0.50": delta["recall_conf_0.001_iou_0.50"],
                "FN_iou_0.50": delta["FN_iou_0.50"],
                "prediction_count_mean": delta["prediction_count_mean"],
                "max_det_saturation_count": delta["max_det_saturation_count"],
                "warning": "descriptive only; n=1 is not statistically meaningful" if canonical == "underexposure" else "",
            }
        )
    return rows


def per_class_rows(*metrics_payloads: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in metrics_payloads:
        for class_name, class_metrics in payload["per_class"].items():
            rows.append(
                {
                    "scope": payload["scope"],
                    "class_name": class_name,
                    "class_id": class_metrics["class_id"],
                    "gt_object_count": class_metrics["gt_object_count"],
                    "prediction_count": class_metrics["prediction_count"],
                    "AP50": class_metrics["AP50"],
                    "AP50-95": class_metrics["AP50-95"],
                    "precision_conf_0.001_iou_0.50": class_metrics["precision_conf_0.001_iou_0.50"],
                    "recall_conf_0.001_iou_0.50": class_metrics["recall_conf_0.001_iou_0.50"],
                    "FN_iou_0.50": class_metrics["FN_iou_0.50"],
                }
            )
    return rows


def saturation_rows(*metrics_payloads: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "scope": payload["scope"],
            "image_count": payload["image_count"],
            "max_det_saturation_count": payload["max_det_saturation_count"],
            "max_det_saturation_fraction": payload["max_det_saturation_fraction"],
            "prediction_count_mean": payload["prediction_count_stats"]["mean"],
            "prediction_count_median": payload["prediction_count_stats"]["median"],
            "prediction_count_p95": payload["prediction_count_stats"]["p95"],
            "prediction_count_max": payload["prediction_count_stats"]["max"],
        }
        for payload in metrics_payloads
    ]


def per_image_diagnostics(
    scope_rows: list[dict[str, Any]],
    gt_by_image: dict[str, list[GTObject]],
    raw_records: list[dict[str, Any]],
    enhanced_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actionable_ids = {row["image_id"] for row in scope_rows if row["included_paired_actionable"]}
    raw_matches = matched_gt_count_by_image(actionable_ids, gt_by_image, raw_records)
    enhanced_matches = matched_gt_count_by_image(actionable_ids, gt_by_image, enhanced_records)
    raw_by_id = prediction_records_by_id(raw_records)
    enhanced_by_id = prediction_records_by_id(enhanced_records)
    row_by_id = {row["image_id"]: row for row in scope_rows}
    rows: list[dict[str, Any]] = []
    for image_id in sorted(actionable_ids):
        gt_count = len(gt_by_image[image_id])
        raw_matched = raw_matches.get(image_id, 0)
        enhanced_matched = enhanced_matches.get(image_id, 0)
        raw_fn = max(0, gt_count - raw_matched)
        enhanced_fn = max(0, gt_count - enhanced_matched)
        rows.append(
            {
                "image_id": image_id,
                "vlm_degradation_canonical": row_by_id[image_id]["vlm_degradation_canonical"],
                "gt_object_count": gt_count,
                "raw_prediction_count": raw_by_id[image_id]["prediction_count"],
                "enhanced_prediction_count": enhanced_by_id[image_id]["prediction_count"],
                "raw_matched_gt_count_iou_0.50": raw_matched,
                "enhanced_matched_gt_count_iou_0.50": enhanced_matched,
                "raw_FN_iou_0.50": raw_fn,
                "enhanced_FN_iou_0.50": enhanced_fn,
                "FN_delta_enhanced_minus_raw": enhanced_fn - raw_fn,
            }
        )
    return rows


def write_latency_artifacts(
    run_dir: Path,
    raw_records: list[dict[str, Any]],
    enhanced_records: list[dict[str, Any]],
    enhancement_manifest: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_values = [float(record["latency_ms"]["wall_clock_ms"]) for record in raw_records]
    enhanced_values = [float(record["latency_ms"]["wall_clock_ms"]) for record in enhanced_records]
    enhancement_values = {row["image_id"]: float(row["latency_ms"]) for row in enhancement_manifest}
    enhanced_detector_by_id = {record["image_id"]: float(record["latency_ms"]["wall_clock_ms"]) for record in enhanced_records}
    pipeline_values = [enhancement_values[image_id] + enhanced_detector_by_id[image_id] for image_id in sorted(enhanced_detector_by_id)]
    detector_summary = {
        "measurement_notes": {
            "image_decode_time_included": True,
            "detector_preprocess_postprocess_included": True,
            "cuda_synchronization_used": True,
            "enhancement_and_detection_measured_in_separate_phases": True,
        },
        "raw_full_640_detector_wall_clock_ms": summarize_values(raw_values),
        "enhanced_actionable_593_detector_wall_clock_ms": summarize_values(enhanced_values),
    }
    pipeline_summary = {
        "measurement_notes": {
            "enhancement_latency_includes_image_decode": True,
            "enhancement_latency_includes_output_png_disk_write": True,
            "detector_latency_includes_reading_pre_materialized_enhanced_png": True,
            "not_an_online_single-pass_benchmark": True,
        },
        "enhancement_wall_clock_ms": summarize_values(list(enhancement_values.values())),
        "enhanced_pipeline_wall_clock_ms": summarize_values(pipeline_values),
    }
    action_rows: list[dict[str, Any]] = []
    by_action: dict[str, list[float]] = defaultdict(list)
    for row in enhancement_manifest:
        by_action[str(row["enhancement_action"])].append(float(row["latency_ms"]))
    for action, values in sorted(by_action.items()):
        stats = summarize_values(values)
        action_rows.append({"enhancement_action": action, **stats})
    write_json(run_dir / "latency/detector_latency_summary.json", detector_summary)
    write_json(run_dir / "latency/pipeline_latency_summary.json", pipeline_summary)
    write_csv(run_dir / "latency/action_latency_summary.csv", action_rows)
    return detector_summary, pipeline_summary


def write_coverage_and_defer(
    run_dir: Path,
    scope_rows: list[dict[str, Any]],
    gt_by_image: dict[str, list[GTObject]],
) -> dict[str, Any]:
    deferred = [row for row in scope_rows if row["deferred"]]
    deferred_rows: list[dict[str, Any]] = []
    manual_counts = Counter(row["manual_degradation_reference"] for row in deferred)
    gt_car = 0
    gt_truck = 0
    for row in deferred:
        objects = gt_by_image[row["image_id"]]
        car_count = sum(1 for obj in objects if obj.class_id == 0)
        truck_count = sum(1 for obj in objects if obj.class_id == 1)
        gt_car += car_count
        gt_truck += truck_count
        deferred_rows.append({**row, "gt_car_count": car_count, "gt_truck_count": truck_count, "gt_object_count": len(objects)})
    payload = {
        "total_images": 640,
        "automatically_enhanced_detected_paired_images": 593,
        "unknown_deferred_images": 47,
        "automatic_action_coverage": 593 / 640,
        "automatic_action_coverage_percent": 92.65625,
        "defer_rate": 47 / 640,
        "defer_rate_percent": 7.34375,
        "deferred_gt_car_count": gt_car,
        "deferred_gt_truck_count": gt_truck,
        "deferred_gt_object_count": gt_car + gt_truck,
        "manual_reference_by_vlm_unknown_counts": dict(sorted(manual_counts.items())),
        "deferred_images_are_abstentions_from_automatic_enhancement": True,
    }
    write_json(run_dir / "coverage_and_defer.json", payload)
    write_csv(run_dir / "deferred_unknown_47.csv", deferred_rows)
    return payload


def draw_overlay(image_path: Path, predictions: list[dict[str, Any]], gt_objects: list[GTObject], output_path: Path) -> None:
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for gt in gt_objects:
        draw.rectangle(list(gt.xyxy), outline=(0, 255, 0), width=3)
    for pred in predictions:
        color = (255, 0, 0) if int(pred["dataset_class_id"]) == 0 else (0, 80, 255)
        draw.rectangle([float(value) for value in pred["xyxy"]], outline=color, width=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_qualitative_artifacts(
    run_dir: Path,
    diagnostics: list[dict[str, Any]],
    scope_rows: list[dict[str, Any]],
    gt_by_image: dict[str, list[GTObject]],
    raw_records: list[dict[str, Any]],
    enhanced_records: list[dict[str, Any]],
    enhanced_manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    diag_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in diagnostics:
        canonical = str(row["vlm_degradation_canonical"])
        if canonical in {"blur", "overexposure", "blur_overexposure"}:
            diag_by_class[canonical].append(row)
    selected: list[dict[str, Any]] = []
    for canonical, rows in sorted(diag_by_class.items()):
        reductions = sorted([row for row in rows if int(row["FN_delta_enhanced_minus_raw"]) < 0], key=lambda row: (int(row["FN_delta_enhanced_minus_raw"]), row["image_id"]))[:2]
        increases = sorted([row for row in rows if int(row["FN_delta_enhanced_minus_raw"]) > 0], key=lambda row: (-int(row["FN_delta_enhanced_minus_raw"]), row["image_id"]))[:2]
        unchanged = sorted([row for row in rows if int(row["FN_delta_enhanced_minus_raw"]) == 0], key=lambda row: row["image_id"])[:1]
        for reason, group in [("largest_fn_reduction", reductions), ("largest_fn_increase", increases), ("no_fn_change", unchanged)]:
            for row in group:
                selected.append({"selection_reason": reason, **row})
    raw_by_id = prediction_records_by_id(raw_records)
    enhanced_by_id = prediction_records_by_id(enhanced_records)
    scope_by_id = {row["image_id"]: row for row in scope_rows}
    enhanced_by_id_path = {row["image_id"]: row["enhanced_image_path"] for row in enhanced_manifest}
    output_rows: list[dict[str, Any]] = []
    for item in selected:
        image_id = str(item["image_id"])
        raw_path = Path(scope_by_id[image_id]["raw_image_path"])
        enhanced_path = Path(enhanced_by_id_path[image_id])
        base = run_dir / "qualitative" / "overlays" / f"{image_id}__{item['selection_reason']}"
        raw_overlay = base.with_name(base.name + "__raw.png")
        enhanced_overlay = base.with_name(base.name + "__enhanced.png")
        side_by_side = base.with_name(base.name + "__side_by_side.png")
        draw_overlay(raw_path, raw_by_id[image_id]["predictions"], gt_by_image[image_id], raw_overlay)
        draw_overlay(enhanced_path, enhanced_by_id[image_id]["predictions"], gt_by_image[image_id], enhanced_overlay)
        with Image.open(raw_overlay) as raw_image, Image.open(enhanced_overlay) as enhanced_image:
            side = Image.new("RGB", (raw_image.width + enhanced_image.width, max(raw_image.height, enhanced_image.height)))
            side.paste(raw_image.convert("RGB"), (0, 0))
            side.paste(enhanced_image.convert("RGB"), (raw_image.width, 0))
            side.save(side_by_side)
        output_rows.append(
            {
                **item,
                "raw_overlay_path": str(raw_overlay),
                "enhanced_overlay_path": str(enhanced_overlay),
                "side_by_side_path": str(side_by_side),
                "illustrative_only": True,
            }
        )
    write_csv(run_dir / "qualitative/selection_manifest.csv", output_rows)
    write_json(run_dir / "qualitative/selection_manifest.json", {"selection_rule": "After metrics, per populated VLM class: up to 2 largest FN reductions, up to 2 largest FN increases, up to 1 no-FN-change; underexposure excluded because n=1.", "records": output_rows})
    return output_rows


def status_core_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_count": metrics["image_count"],
        "gt_object_count": metrics["gt_object_count"],
        "prediction_count": metrics["prediction_count"],
        "mAP50": metrics["mAP50"],
        "mAP50-95": metrics["mAP50-95"],
        "precision_conf_0.001_iou_0.50": metrics["precision_conf_0.001_iou_0.50"],
        "recall_conf_0.001_iou_0.50": metrics["recall_conf_0.001_iou_0.50"],
        "FN_iou_0.50": metrics["FN_iou_0.50"],
    }


def result_recommendation(delta: dict[str, Any]) -> tuple[str, str]:
    core = [float(delta["mAP50"]), float(delta["mAP50-95"]), float(delta["recall_conf_0.001_iou_0.50"])]
    if all(value >= 0.0 for value in core):
        return (
            "improved_or_preserved",
            "Proceed to the same frozen paired protocol for the next detector.",
        )
    if all(value < 0.0 for value in core):
        return (
            "consistently_degraded",
            "Do not tune parameters in this run; recommend a separately preregistered action-bank revision stage.",
        )
    return (
        "mixed_by_metric_or_class",
        "Preserve the negative and positive findings, and proceed to a selective action-policy experiment without tuning on test.",
    )


def write_final_artifacts(
    run_dir: Path,
    status: dict[str, Any],
    detector_summary: dict[str, Any],
    pipeline_summary: dict[str, Any],
    qualitative_rows: list[dict[str, Any]],
) -> None:
    delta = status["paired_metric_deltas"]
    outcome, recommendation = result_recommendation(delta)
    lines = [
        "# AgentCam C3 RT-DETR-L Raw vs VLM-Guided Enhanced",
        "",
        f"Status: {status['status']}",
        "",
        "## Scientific Scope",
        "The scientific development pool contains all 640 `training-data` images. The old 544/96 split is provenance only.",
        "Testing-data was untouched. Raw RT-DETR-L inference covered 640 images.",
        "The primary paired raw-versus-enhanced comparison covered exactly 593 identical image IDs.",
        "The 47 unknown images remained deferred and did not receive automatic enhancement or enhanced-branch detector inference.",
        "",
        "Manual labels were not used for action selection. Enhancement parameters were frozen before detector evaluation.",
        "No VLM inference, training, threshold tuning, enhancement tuning, or best-of-bank selection occurred.",
        "",
        "`clear=0` means this dataset/run cannot validate the no-enhancement action. `underexposure=1` is insufficient for a category-level conclusion.",
        "The primary comparison is `raw_paired_actionable_593` versus `enhanced_actionable_593`; `raw_full_640` versus `enhanced_593` is not treated as a paired performance delta.",
        "",
        "## Metrics",
        f"- raw_full_640: `{json.dumps(status['raw_full_metrics'], sort_keys=True)}`",
        f"- raw_paired_593: `{json.dumps(status['raw_paired_metrics'], sort_keys=True)}`",
        f"- enhanced_paired_593: `{json.dumps(status['enhanced_paired_metrics'], sort_keys=True)}`",
        f"- enhanced_minus_raw_paired: `{json.dumps(status['paired_metric_deltas'], sort_keys=True)}`",
        "",
        f"Aggregate outcome classification: `{outcome}`.",
        "",
        "## Saturation",
        f"- raw max_det saturation count: {status['max_det_saturation_raw_count']}",
        f"- enhanced max_det saturation count: {status['max_det_saturation_enhanced_count']}",
        "",
        "## Latency",
        f"- detector latency: `{json.dumps(detector_summary, sort_keys=True)}`",
        f"- pipeline latency: `{json.dumps(pipeline_summary, sort_keys=True)}`",
        "",
        "Enhancement latency includes image decode and output PNG disk write. Detector wall-clock latency includes detector preprocessing/postprocessing and CUDA synchronization. Enhancement and detection were measured in separate phases, so the combined pipeline value is not an online single-pass benchmark.",
        "",
        "## Qualitative Artifacts",
        f"Selection manifest: `{run_dir / 'qualitative/selection_manifest.csv'}`",
        f"Overlay count: {len(qualitative_rows)}",
        "Qualitative images are illustrative only and do not override aggregate metrics.",
        "",
        "## Interpretation",
        "Enhancement-only component results do not yet prove the full AgentCam closed-loop router.",
        f"Next-stage recommendation: {recommendation}",
    ]
    (run_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    task_lines = [
        "# AgentCam C3 RT-DETR-L Raw vs VLM-Guided Enhanced",
        "",
        f"Run directory: `{run_dir}`",
        f"Status: `{status['status']}`",
        "",
        f"Raw full inference count: {status['raw_full_inference_count']}",
        f"Enhanced paired inference count: {status['enhanced_inference_count']}",
        f"Deferred unknown count: {status['deferred_unknown_count']}",
        "",
        f"Primary delta: `{json.dumps(status['paired_metric_deltas'], sort_keys=True)}`",
        "",
        "No VLM inference, training/fine-tuning, package change, threshold tuning, enhancement tuning, or best-of-bank selection occurred.",
        f"Recommendation: {recommendation}",
    ]
    (Path("tasks") / "agentcam_c3_rtdetr_l_raw_vs_vlm_enhanced.md").write_text("\n".join(task_lines) + "\n", encoding="utf-8")


def build_status(
    run_dir: Path,
    input_integrity: dict[str, Any],
    class_counts: Counter[str],
    action_counts: Counter[str],
    enhancement_summary: dict[str, Any],
    raw_full_metrics: dict[str, Any],
    raw_paired_metrics: dict[str, Any],
    enhanced_metrics: dict[str, Any],
    delta: dict[str, Any],
    detector_audit: dict[str, Any],
    detector_summary: dict[str, Any],
    source_integrity: dict[str, Any],
) -> dict[str, Any]:
    warnings = [
        "Inherited warning retained: earlier C1 dependency work invoked /usr/bin/pip; C3 did not repair or alter the environment.",
        "clear has zero VLM examples; no-enhancement action is not validated by C3.",
        "underexposure has one VLM example; category-level conclusions are prohibited.",
        "47 images are deferred because the VLM output canonicalized to unknown.",
    ]
    failure_reasons: list[str] = []
    status = "PASSED_WITH_WARNINGS"
    return {
        "status": status,
        "completed_stage": "C3",
        "source_c1_run": str(C1_RUN),
        "source_c2_run": str(C2_RUN),
        "detector_name": "rtdetr-l",
        "detector_weight_path": str(WEIGHT_PATH),
        "detector_weight_sha256": EXPECTED_WEIGHT_SHA256,
        "original_training_data_count": 640,
        "raw_full_inference_count": raw_full_metrics["image_count"],
        "actionable_count": 593,
        "enhanced_inference_count": enhanced_metrics["image_count"],
        "deferred_unknown_count": 47,
        "automatic_action_coverage": 593 / 640,
        "defer_rate": 47 / 640,
        "train_core_provenance_count": input_integrity["train_core_provenance_count"],
        "val_from_train_provenance_count": input_integrity["val_from_train_provenance_count"],
        "test_image_count": input_integrity["test_image_count"],
        "canonical_vlm_class_counts": dict(sorted(class_counts.items())),
        "enhancement_action_counts": dict(sorted((key, value) for key, value in action_counts.items() if value)),
        "enhancement_success_count": enhancement_summary["success_count"],
        "enhancement_failure_count": enhancement_summary["failure_count"],
        "raw_full_metrics": status_core_metrics(raw_full_metrics),
        "raw_paired_metrics": status_core_metrics(raw_paired_metrics),
        "enhanced_paired_metrics": status_core_metrics(enhanced_metrics),
        "paired_metric_deltas": delta,
        "max_det_saturation_raw_count": raw_paired_metrics["max_det_saturation_count"],
        "max_det_saturation_enhanced_count": enhanced_metrics["max_det_saturation_count"],
        "model_load_count": detector_audit["model_load_count"],
        "warmup_call_count": detector_audit["warmup_call_count"],
        "measured_predict_call_count": raw_full_metrics["image_count"] + enhanced_metrics["image_count"],
        "detector_inference_run": True,
        "vlm_inference_run": False,
        "training_run": False,
        "package_change_attempted": False,
        "threshold_tuning_run": False,
        "enhancement_tuning_run": False,
        "best_of_bank_used": False,
        "manual_label_used_for_action": False,
        "source_data_modified": source_integrity["source_data_modified"],
        "inherited_system_pip_warning": True,
        "clear_class_sample_count": 0,
        "underexposure_class_sample_count": 1,
        "warnings": warnings,
        "failure_reasons": failure_reasons,
        "files_modified": [
            "scripts/agentcam_c3_rtdetr_eval.py",
            "tests/test_c3_rtdetr_eval.py",
            "tasks/agentcam_c3_rtdetr_l_raw_vs_vlm_enhanced.md",
            str(run_dir),
        ],
        "detector_latency_summary": detector_summary,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_failed_status(run_dir: Path, error: str) -> None:
    write_json(
        run_dir / "STATUS.json",
        {
            "status": "FAILED",
            "completed_stage": "C3",
            "source_c1_run": str(C1_RUN),
            "source_c2_run": str(C2_RUN),
            "detector_name": "rtdetr-l",
            "detector_inference_run": False,
            "vlm_inference_run": False,
            "training_run": False,
            "package_change_attempted": False,
            "threshold_tuning_run": False,
            "enhancement_tuning_run": False,
            "best_of_bank_used": False,
            "manual_label_used_for_action": False,
            "source_data_modified": False,
            "inherited_system_pip_warning": True,
            "failure_reasons": [error],
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def run_c3(run_dir: Path) -> int:
    if run_dir.exists() and any(run_dir.iterdir()):
        existing = [path.name for path in run_dir.iterdir()]
        if existing and existing != ["preflight_checks.json"]:
            raise RuntimeError(f"refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        preflight(run_dir)
        scope_rows, input_integrity = verify_inputs(run_dir)
        for row in scope_rows:
            assert_not_test_path(row["raw_image_path"])
            assert_not_test_path(row["label_path"])
        image_meta, gt_by_image, _inventory = snapshot_sources(run_dir, scope_rows)
        write_gt_integrity(run_dir, scope_rows, gt_by_image)
        enhancement_manifest, enhancement_summary = materialize_enhancements(run_dir, scope_rows, image_meta)
        source_after_enhancement = recheck_source_integrity(run_dir, scope_rows, image_meta, "source_integrity_after_enhancement.json")
        if not source_after_enhancement["all_source_hashes_unchanged"]:
            raise RuntimeError("source hashes changed after enhancement")
        enhanced_source_by_id = {row["image_id"]: Path(row["enhanced_image_path"]) for row in enhancement_manifest}
        raw_source_by_id = {row["image_id"]: Path(row["raw_image_path"]) for row in scope_rows}
        actionable_rows = [row for row in scope_rows if row["included_paired_actionable"]]
        deferred_rows = [row for row in scope_rows if row["deferred"]]
        if any(row["image_id"] in enhanced_source_by_id for row in deferred_rows):
            raise RuntimeError("unknown/deferred image received enhanced output")
        warmup_row = sorted(scope_rows, key=lambda item: item["image_id"])[0]
        model, detector_audit = load_detector_and_warmup(run_dir, warmup_row)
        raw_full_records = run_detector_branch(
            model,
            sorted(scope_rows, key=lambda item: item["image_id"]),
            raw_source_by_id,
            "raw_full_640",
            run_dir / "predictions/raw_full_640",
            run_dir / "latency/raw_full_640_per_image.csv",
        )
        enhanced_records = run_detector_branch(
            model,
            sorted(actionable_rows, key=lambda item: item["image_id"]),
            enhanced_source_by_id,
            "enhanced_actionable_593",
            run_dir / "predictions/enhanced_actionable_593",
            run_dir / "latency/enhanced_actionable_593_per_image.csv",
        )
        if len(raw_full_records) != 640 or len(enhanced_records) != 593:
            raise RuntimeError("detector inference counts did not match required scopes")
        raw_paired_records = write_raw_paired_view(run_dir, raw_full_records, {row["image_id"] for row in actionable_rows})
        raw_full_metrics, raw_paired_metrics, enhanced_metrics, delta, diagnostics = write_metric_artifacts(
            run_dir, scope_rows, gt_by_image, raw_full_records, raw_paired_records, enhanced_records
        )
        detector_summary, pipeline_summary = write_latency_artifacts(run_dir, raw_full_records, enhanced_records, enhancement_manifest)
        write_coverage_and_defer(run_dir, scope_rows, gt_by_image)
        qualitative_rows = write_qualitative_artifacts(
            run_dir, diagnostics, scope_rows, gt_by_image, raw_paired_records, enhanced_records, enhancement_manifest
        )
        source_after_inference = recheck_source_integrity(run_dir, scope_rows, image_meta, "source_integrity_after_inference.json")
        class_counts = Counter(row["vlm_degradation_canonical"] for row in scope_rows)
        action_counts = Counter(row["enhancement_action"] for row in scope_rows)
        status = build_status(
            run_dir,
            input_integrity,
            class_counts,
            action_counts,
            enhancement_summary,
            raw_full_metrics,
            raw_paired_metrics,
            enhanced_metrics,
            delta,
            detector_audit,
            detector_summary,
            source_after_inference,
        )
        write_json(run_dir / "STATUS.json", status)
        write_final_artifacts(run_dir, status, detector_summary, pipeline_summary, qualitative_rows)
        return 0
    except Exception as exc:
        write_failed_status(run_dir, f"{type(exc).__name__}: {exc}")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AgentCam Stage C3 RT-DETR-L raw vs VLM-guided enhanced evaluation.")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_c3(args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
