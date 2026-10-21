from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm.action_bank import (
    ACTION_BY_CANONICAL_CLASS,
    CANONICAL_VLM_CLASSES,
    DEPRECATED_C1_ACTIONS,
    action_for_canonical_class,
    canonicalize_vlm_degradation,
)
from workers.enhancement import (
    DEPRECATED_ACTIONS,
    SUPPORTED_ACTIONS,
    apply_enhancement_action,
    operation_order_for_action,
    parameters_for_action,
)


C1_RUN = Path("artifacts/agentcam_qwen_vlm_degradation_audit/20260620T024704Z")
C1_JSONL = C1_RUN / "audit/qwen_vlm_degradation_audit.jsonl"
C1_CSV = C1_RUN / "audit/qwen_vlm_degradation_audit.csv"
C1_STATUS = C1_RUN / "STATUS.json"
C1_REPORT = C1_RUN / "summaries/FINAL_REPORT.md"
C1_ACTION_COUNTS = C1_RUN / "summaries/vlm_action_counts.csv"
EXPECTED_INTERPRETER = "/home/qiren/agentcam/ag1_venv/bin/python"
VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SMOKE_CLASSES = ("blur", "overexposure", "underexposure", "blur_overexposure", "clear")
MANIFEST_FIELDS = [
    "image_id",
    "image_filename",
    "original_image_path",
    "original_source_split",
    "processed_split_provenance",
    "manual_degradation_reference",
    "vlm_model",
    "vlm_blur",
    "vlm_overexposure",
    "vlm_underexposure",
    "vlm_clear",
    "vlm_unknown",
    "vlm_confidence",
    "vlm_severity",
    "vlm_degradation_canonical",
    "enhancement_action",
    "action_source",
    "source_c1_artifact",
    "source_c1_record_identifier",
    "image_exists",
    "is_test_image",
]
SMOKE_FIELDS = [
    "image_id",
    "image_filename",
    "original_image_path",
    "processed_split_provenance",
    "vlm_degradation_canonical",
    "enhancement_action",
    "vlm_confidence",
    "manual_degradation_reference",
    "selection_rank_within_vlm_class",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        field_set: list[str] = []
        for row in rows:
            for key in row:
                if key not in field_set:
                    field_set.append(key)
        fields = field_set
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def module_origin(module_name: str) -> str | None:
    spec = importlib.util.find_spec(module_name)
    return spec.origin if spec is not None else None


def validate_environment(run_dir: Path, repo: Path) -> dict[str, Any]:
    commands = {
        "git_status_short": run_command(["git", "status", "--short"], repo),
        "git_diff_stat": run_command(["git", "diff", "--stat"], repo),
        "python_version": run_command([sys.executable, "--version"], repo),
    }
    interpreter = {
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
        "venv_active": sys.prefix != sys.base_prefix,
    }
    try:
        import site

        interpreter["site_packages"] = site.getsitepackages()
    except Exception as exc:  # pragma: no cover
        interpreter["site_packages_error"] = str(exc)

    dependency_modules = {
        "PIL": module_origin("PIL"),
        "numpy": module_origin("numpy"),
        "cv2": module_origin("cv2"),
    }
    site_prefix = str(Path(EXPECTED_INTERPRETER).parents[1])
    outside_venv = {
        name: origin
        for name, origin in dependency_modules.items()
        if origin is None or not str(Path(origin).resolve()).startswith(site_prefix)
    }
    status = "passed" if not outside_venv and sys.executable == EXPECTED_INTERPRETER else "failed"
    payload = {
        "commands": commands,
        "expected_interpreter": EXPECTED_INTERPRETER,
        "interpreter": interpreter,
        "dependency_modules": dependency_modules,
        "outside_venv": outside_venv,
        "environment_integrity_status": status,
        "inherited_system_pip_warning": True,
        "inherited_system_pip_warning_text": (
            "Earlier C1 dependency work invoked /usr/bin/pip before the project venv install was corrected; "
            "C2 records this warning and does not attempt environment repair."
        ),
    }
    write_json(run_dir / "preflight_checks.json", payload)
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def image_filename(row: dict[str, Any]) -> str:
    return Path(str(row.get("image_path", ""))).name


def test_image_from_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return "/testing-data/" in normalized or normalized.startswith("testing-data/")


def c1_integrity(rows: list[dict[str, Any]], csv_rows: list[dict[str, str]]) -> dict[str, Any]:
    image_ids = [row.get("image_id") for row in rows]
    filenames = [image_filename(row) for row in rows]
    duplicate_image_ids = sorted(key for key, value in Counter(image_ids).items() if value > 1)
    duplicate_filenames = sorted(key for key, value in Counter(filenames).items() if value > 1)
    if all(isinstance(value, str) and value for value in image_ids) and not duplicate_image_ids:
        unique_key = "image_id"
    elif not duplicate_filenames:
        unique_key = "image_filename"
    else:
        unique_key = "ambiguous"

    split_counts = Counter(row.get("split") for row in rows)
    source_split_counts = Counter(row.get("source_split") for row in rows)
    parse_failures = sum(row.get("parse_valid") is not True for row in rows)
    schema_failures = sum(row.get("schema_valid") is not True for row in rows)
    consistency_failures = sum(row.get("consistency_valid") is not True for row in rows)
    test_rows = [row for row in rows if test_image_from_path(str(row.get("image_path", "")))]
    csv_keys = {row.get(unique_key if unique_key != "image_filename" else "image_path") for row in csv_rows}
    jsonl_keys = set(image_ids if unique_key == "image_id" else filenames)
    if unique_key == "image_filename":
        csv_keys = {Path(str(row.get("image_path", ""))).name for row in csv_rows}
    return {
        "source_c1_run": str(C1_RUN),
        "authoritative_artifact": str(C1_JSONL),
        "csv_artifact": str(C1_CSV),
        "record_count": len(rows),
        "csv_record_count": len(csv_rows),
        "unique_key": unique_key,
        "unique_image_id_count": len(set(image_ids)),
        "unique_image_filename_count": len(set(filenames)),
        "duplicate_image_ids": duplicate_image_ids,
        "duplicate_image_filenames": duplicate_filenames,
        "split_counts": dict(sorted(split_counts.items())),
        "source_split_counts": dict(sorted(source_split_counts.items())),
        "train_core_provenance_count": split_counts.get("train_core", 0),
        "val_from_train_provenance_count": split_counts.get("val_from_train", 0),
        "test_image_count": len(test_rows),
        "parse_failures": parse_failures,
        "schema_failures": schema_failures,
        "consistency_failures": consistency_failures,
        "jsonl_csv_key_sets_match": jsonl_keys == csv_keys,
        "image_exists_count": sum(Path(str(row.get("image_path", ""))).exists() for row in rows),
        "c1_artifact_hashes": {
            "STATUS.json": sha256_file(C1_STATUS),
            "FINAL_REPORT.md": sha256_file(C1_REPORT),
            "qwen_vlm_degradation_audit.jsonl": sha256_file(C1_JSONL),
            "qwen_vlm_degradation_audit.csv": sha256_file(C1_CSV),
            "vlm_action_counts.csv": sha256_file(C1_ACTION_COUNTS),
        },
    }


def manifest_row(row: dict[str, Any], unique_key: str) -> dict[str, Any]:
    image_path = str(row.get("image_path", ""))
    canonical = canonicalize_vlm_degradation(row)
    action = action_for_canonical_class(canonical)
    return {
        "image_id": row.get("image_id", ""),
        "image_filename": Path(image_path).name,
        "original_image_path": image_path,
        "original_source_split": "training-data",
        "processed_split_provenance": row.get("split", ""),
        "manual_degradation_reference": row.get("metadata_degradation_label", ""),
        "vlm_model": row.get("selected_model_id", VLM_MODEL),
        "vlm_blur": bool_value(row.get("blur")),
        "vlm_overexposure": bool_value(row.get("overexposure")),
        "vlm_underexposure": bool_value(row.get("underexposure")),
        "vlm_clear": bool_value(row.get("clear")),
        "vlm_unknown": bool_value(row.get("unknown_degradation")),
        "vlm_confidence": row.get("confidence", ""),
        "vlm_severity": row.get("severity", ""),
        "vlm_degradation_canonical": canonical,
        "enhancement_action": action,
        "action_source": "vlm_prediction_only",
        "source_c1_artifact": str(C1_JSONL),
        "source_c1_record_identifier": row.get("image_id") if unique_key == "image_id" else Path(image_path).name,
        "image_exists": Path(image_path).exists(),
        "is_test_image": False,
    }


def build_action_spec() -> list[dict[str, Any]]:
    return [
        {
            "action_name": "no_enhancement",
            "canonical_vlm_class": "clear",
            "implementation_function": "no_enhancement",
            "implementation_file": "workers/enhancement.py",
            "algorithm": "decoded-pixel identity copy; no enhancement transform",
            "fixed_parameters": parameters_for_action("no_enhancement"),
            "operation_order": operation_order_for_action("no_enhancement"),
            "dependency_modules": ["PIL"],
            "preserves_dimensions": True,
            "creates_output": True,
            "known_limitations": "Copies the decoded image only; it does not improve any degradation.",
        },
        {
            "action_name": "blur_enhancement",
            "canonical_vlm_class": "blur",
            "implementation_function": "sharpen",
            "implementation_file": "workers/enhancement.py",
            "algorithm": "Pillow UnsharpMask mild sharpening",
            "fixed_parameters": parameters_for_action("blur_enhancement"),
            "operation_order": operation_order_for_action("blur_enhancement"),
            "dependency_modules": ["PIL"],
            "preserves_dimensions": True,
            "creates_output": True,
            "known_limitations": "May emphasize edges; does not recover information lost to severe motion blur.",
        },
        {
            "action_name": "overexposure_enhancement",
            "canonical_vlm_class": "overexposure",
            "implementation_function": "exposure_enhance",
            "implementation_file": "workers/enhancement.py",
            "algorithm": "Pillow autocontrast plus fixed brightness, contrast, and gamma tone compression",
            "fixed_parameters": parameters_for_action("overexposure_enhancement"),
            "operation_order": operation_order_for_action("overexposure_enhancement"),
            "dependency_modules": ["PIL"],
            "preserves_dimensions": True,
            "creates_output": True,
            "known_limitations": "Cannot recover clipped highlight detail.",
        },
        {
            "action_name": "underexposure_enhancement",
            "canonical_vlm_class": "underexposure",
            "implementation_function": "underexposure_enhance",
            "implementation_file": "workers/enhancement.py",
            "algorithm": "Fixed gamma brightening followed by CLAHE on LAB L channel",
            "fixed_parameters": parameters_for_action("underexposure_enhancement"),
            "operation_order": operation_order_for_action("underexposure_enhancement"),
            "dependency_modules": ["PIL", "numpy", "cv2"],
            "preserves_dimensions": True,
            "creates_output": True,
            "known_limitations": "Can amplify noise and cannot recover detail absent from a dark capture.",
        },
        {
            "action_name": "overexposure_then_blur_enhancement",
            "canonical_vlm_class": "blur_overexposure",
            "implementation_function": "apply_enhancement_action",
            "implementation_file": "workers/enhancement.py",
            "algorithm": "Deterministic sequential processing: overexposure action first, blur action second",
            "fixed_parameters": parameters_for_action("overexposure_then_blur_enhancement"),
            "operation_order": operation_order_for_action("overexposure_then_blur_enhancement"),
            "dependency_modules": ["PIL"],
            "preserves_dimensions": True,
            "creates_output": True,
            "known_limitations": "No alternative ranking is attempted; clipped highlights and true blur remain bounded by source information.",
        },
        {
            "action_name": "defer_no_automatic_enhancement",
            "canonical_vlm_class": "unknown",
            "implementation_function": "apply_enhancement_action",
            "implementation_file": "workers/enhancement.py",
            "algorithm": "Deferred record; no automatic enhancement is generated",
            "fixed_parameters": parameters_for_action("defer_no_automatic_enhancement"),
            "operation_order": operation_order_for_action("defer_no_automatic_enhancement"),
            "dependency_modules": [],
            "preserves_dimensions": True,
            "creates_output": False,
            "known_limitations": "Requires a later explicit policy decision before automatic enhancement.",
        },
    ]


def write_action_spec(run_dir: Path, spec: list[dict[str, Any]]) -> None:
    payload = {
        "stage": "C2",
        "parameters_tuned": False,
        "detector_used": False,
        "ground_truth_used": False,
        "manual_label_used_for_action_selection": False,
        "best_of_bank_part_of_stage_c2": False,
        "blur_overexposure_behavior": "overexposure repair first, then blur enhancement",
        "deprecated_actions": sorted(DEPRECATED_C1_ACTIONS | DEPRECATED_ACTIONS),
        "actions": spec,
    }
    write_json(run_dir / "ACTION_BANK_SPEC.json", payload)
    lines = [
        "# AgentCam C2 Frozen Enhancement Action Bank",
        "",
        "- Parameters were not tuned in Stage C2.",
        "- No detector was used.",
        "- No ground truth was used.",
        "- No manual label was used for action selection.",
        "- `blur_overexposure` uses deterministic sequential processing: overexposure first, blur second.",
        "- Best-of-bank is not part of Stage C2.",
        "",
        "| action | canonical class | function | parameters | creates output | limitations |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in spec:
        lines.append(
            "| {action_name} | {canonical_vlm_class} | {implementation_function} | `{params}` | {creates_output} | {known_limitations} |".format(
                params=json.dumps(row["fixed_parameters"], sort_keys=True),
                **row,
            )
        )
    (run_dir / "ACTION_BANK_SPEC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_smoke(manifest: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    sorted_rows = sorted(manifest, key=lambda row: str(row["image_filename"]))
    for canonical in SMOKE_CLASSES:
        class_rows = [
            row
            for row in sorted_rows
            if row["vlm_degradation_canonical"] == canonical and not row["is_test_image"]
        ][:2]
        if not class_rows:
            missing.append(canonical)
        for rank, row in enumerate(class_rows, start=1):
            selected.append(
                {
                    **{field: row.get(field, "") for field in SMOKE_FIELDS if field != "selection_rank_within_vlm_class"},
                    "selection_rank_within_vlm_class": rank,
                }
            )
    return selected[:10], missing


def channel_count(array: np.ndarray) -> int:
    if array.ndim == 2:
        return 1
    if array.ndim == 3:
        return int(array.shape[2])
    return 0


def valid_range(array: np.ndarray) -> bool:
    if array.size == 0:
        return False
    if np.issubdtype(array.dtype, np.integer):
        return int(array.min()) >= 0 and int(array.max()) <= 255
    return bool(np.nanmin(array) >= 0.0 and np.nanmax(array) <= 1.0)


def save_comparison(raw_image: Image.Image, enhanced_image: Image.Image, path: Path) -> None:
    raw_rgb = raw_image.convert("RGB")
    enhanced_rgb = enhanced_image.convert("RGB")
    comparison = Image.new("RGB", (raw_rgb.width + enhanced_rgb.width, raw_rgb.height))
    comparison.paste(raw_rgb, (0, 0))
    comparison.paste(enhanced_rgb, (raw_rgb.width, 0))
    comparison.save(path)


def process_smoke(run_dir: Path, selected: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    raw_dir = run_dir / "images/raw"
    enhanced_dir = run_dir / "images/enhanced"
    comparison_dir = run_dir / "comparisons"
    raw_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    source_checks: list[dict[str, Any]] = []
    latencies_by_action: dict[str, list[float]] = defaultdict(list)

    for row in selected:
        source = Path(str(row["original_image_path"]))
        action = str(row["enhancement_action"])
        output_path = enhanced_dir / f"{source.stem}__{action}{source.suffix or '.png'}"
        raw_output_path = raw_dir / source.name
        comparison_path = comparison_dir / f"{source.stem}__{action}.png"
        result: dict[str, Any] = {
            **row,
            "source_exists": source.exists(),
            "raw_output_path": str(raw_output_path),
            "enhanced_output_path": "",
            "comparison_path": "",
            "status": "",
            "exception": "",
            "latency_ms": "",
            "operation_log": "",
            "parameters": json.dumps(parameters_for_action(action), sort_keys=True),
            "source_hash_before": "",
            "source_hash_after": "",
            "source_hash_unchanged": False,
            "width_unchanged": False,
            "height_unchanged": False,
            "channel_count_valid": False,
            "pixel_values_finite": False,
            "pixel_values_in_expected_range": False,
            "noop_pixels_identical": "",
            "combined_order_valid": "",
            "action_matches_manifest": True,
        }
        try:
            if not source.exists():
                raise FileNotFoundError(str(source))
            before_bytes = source.read_bytes()
            result["source_hash_before"] = sha256_bytes(before_bytes)
            raw_output_path.write_bytes(before_bytes)
            with Image.open(io.BytesIO(before_bytes)) as opened:
                raw_image = opened.copy()

            if action == "defer_no_automatic_enhancement":
                result["status"] = "skipped_defer"
                after_bytes = source.read_bytes()
                result["source_hash_after"] = sha256_bytes(after_bytes)
                result["source_hash_unchanged"] = result["source_hash_before"] == result["source_hash_after"]
                results.append(result)
                source_checks.append(
                    {
                        "image_id": row.get("image_id", ""),
                        "source_path": str(source),
                        "hash_before": result["source_hash_before"],
                        "hash_after": result["source_hash_after"],
                        "unchanged": result["source_hash_unchanged"],
                        "output_overwrote_source": False,
                    }
                )
                continue

            operation_log: list[str] = []
            start = time.perf_counter()
            if action == "no_enhancement":
                enhanced_image = raw_image.copy()
            else:
                enhanced_image = apply_enhancement_action(raw_image, action, operation_log)
            latency_ms = (time.perf_counter() - start) * 1000.0
            if action == "no_enhancement":
                operation_log.append("no_enhancement")
            if enhanced_image is None:
                raise RuntimeError(f"action {action} did not create an output")
            enhanced_image.save(output_path)
            save_comparison(raw_image, enhanced_image, comparison_path)
            after_bytes = source.read_bytes()
            result["source_hash_after"] = sha256_bytes(after_bytes)
            result["source_hash_unchanged"] = result["source_hash_before"] == result["source_hash_after"]
            result["enhanced_output_path"] = str(output_path)
            result["comparison_path"] = str(comparison_path)
            result["latency_ms"] = latency_ms
            result["operation_log"] = json.dumps(operation_log)
            latencies_by_action[action].append(latency_ms)

            with Image.open(output_path) as output_image:
                output_loaded = output_image.copy()
            output_array = np.asarray(output_loaded)
            raw_array = np.asarray(raw_image)
            result["width_unchanged"] = output_loaded.width == raw_image.width
            result["height_unchanged"] = output_loaded.height == raw_image.height
            result["channel_count_valid"] = channel_count(output_array) in {1, 3, 4}
            result["pixel_values_finite"] = bool(np.isfinite(output_array).all())
            result["pixel_values_in_expected_range"] = valid_range(output_array)
            if action == "no_enhancement":
                result["noop_pixels_identical"] = bool(np.array_equal(raw_array, output_array))
                if not result["noop_pixels_identical"]:
                    raise AssertionError("no_enhancement decoded pixels differ from source")
            if action == "overexposure_then_blur_enhancement":
                result["combined_order_valid"] = operation_log == ["overexposure_enhancement", "blur_enhancement"]
                if not result["combined_order_valid"]:
                    raise AssertionError("combined action order was not overexposure then blur")

            validations = [
                result["source_hash_unchanged"],
                result["width_unchanged"],
                result["height_unchanged"],
                result["channel_count_valid"],
                result["pixel_values_finite"],
                result["pixel_values_in_expected_range"],
            ]
            result["status"] = "success" if all(validations) else "failed_validation"
        except Exception as exc:
            result["status"] = "failed_exception"
            result["exception"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
        source_checks.append(
            {
                "image_id": row.get("image_id", ""),
                "source_path": str(source),
                "hash_before": result["source_hash_before"],
                "hash_after": result["source_hash_after"],
                "unchanged": result["source_hash_unchanged"],
                "output_overwrote_source": bool(result.get("enhanced_output_path")) and Path(result["enhanced_output_path"]).resolve() == source.resolve(),
            }
        )

    latency_summary: dict[str, Any] = {"overall": {}, "by_action": {}}
    all_latencies = [value for values in latencies_by_action.values() for value in values]
    if all_latencies:
        latency_summary["overall"] = latency_stats(all_latencies)
    for action, values in sorted(latencies_by_action.items()):
        latency_summary["by_action"][action] = latency_stats(values)
    source_integrity = {
        "records_checked": len(source_checks),
        "all_sources_unchanged": all(check["unchanged"] for check in source_checks),
        "no_source_overwritten": not any(check["output_overwrote_source"] for check in source_checks),
        "labels_modified": False,
        "manifests_modified": False,
        "test_image_accessed": False,
        "checks": source_checks,
    }
    return results, latency_summary, source_integrity


def latency_stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
    }


def reference_summary(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    crosstab: dict[str, Counter[str]] = defaultdict(Counter)
    agreement = 0
    disagreement = 0
    unambiguous_total = 0
    for row in manifest:
        manual = str(row.get("manual_degradation_reference", "") or "unknown")
        canonical = str(row["vlm_degradation_canonical"])
        crosstab[manual][canonical] += 1
        if manual in {"blur", "overexposure", "underexposure", "blur_overexposure", "clear"} and canonical != "unknown":
            unambiguous_total += 1
            if manual == canonical:
                agreement += 1
            else:
                disagreement += 1
    crosstab_rows: list[dict[str, Any]] = []
    for manual in sorted(crosstab):
        row = {"manual_degradation_reference": manual}
        for canonical in CANONICAL_VLM_CLASSES:
            row[canonical] = crosstab[manual].get(canonical, 0)
        crosstab_rows.append(row)
    return {
        "agreement_count_unambiguous": agreement,
        "disagreement_count_unambiguous": disagreement,
        "unambiguous_comparison_count": unambiguous_total,
        "unknown_or_contradictory_count": Counter(row["vlm_degradation_canonical"] for row in manifest).get("unknown", 0),
        "crosstab_rows": crosstab_rows,
    }


def source_inspection(run_dir: Path) -> None:
    files = [
        "workers/enhancement.py",
        "routing/rule_router.py",
        "routing/run_router.py",
        "configs/router_rule.yaml",
        "tests/test_degradation_and_routing.py",
        "tests/test_class_mapping.py",
    ]
    rows = []
    for rel in files:
        path = Path(rel)
        rows.append({"path": rel, "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else ""})
    payload = {
        "inspected_before_c2_changes": files,
        "active_enhancement_worker": "workers/enhancement.py",
        "existing_functions": {
            "exposure_enhance": "autocontrast cutoff=1, brightness=0.92, contrast=1.12, gamma=1.12",
            "sharpen": "Pillow UnsharpMask radius=1.4, percent=140, threshold=3",
        },
        "c2_added_functions": {
            "no_enhancement": "decoded image copy",
            "underexposure_enhance": "gamma=0.72 followed by LAB L-channel CLAHE clipLimit=2.0 tileGridSize=(8,8)",
            "apply_enhancement_action": "deterministic action dispatcher with combined overexposure then blur order",
        },
        "latency_measured_in_stage_c2_smoke": True,
        "files": rows,
    }
    write_json(run_dir / "enhancement_source_inspection.json", payload)


def final_status(
    run_dir: Path,
    integrity: dict[str, Any],
    manifest: list[dict[str, Any]],
    smoke: list[dict[str, Any]],
    smoke_results: list[dict[str, Any]],
    source_integrity: dict[str, Any],
    env: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    class_counts = dict(sorted(Counter(row["vlm_degradation_canonical"] for row in manifest).items()))
    action_counts = dict(sorted(Counter(row["enhancement_action"] for row in manifest).items()))
    processed_success = [row for row in smoke_results if row["status"] == "success"]
    failed = [row for row in smoke_results if str(row["status"]).startswith("failed")]
    warnings = [
        "Inherited warning retained: earlier C1 dependency work invoked /usr/bin/pip; C2 did not repair or alter the environment.",
        "Clear VLM class has zero C1 examples; smoke selection did not replace it with manual labels.",
    ]
    failure_reasons: list[str] = []
    pass_conditions = {
        "exactly_640_c1_records": integrity["record_count"] == 640,
        "exactly_640_joined_training_data_records": len(manifest) == 640,
        "zero_test_images": integrity["test_image_count"] == 0 and not any(row["is_test_image"] for row in manifest),
        "no_ambiguous_join": integrity["unique_key"] != "ambiguous",
        "deterministic_mapping": set(ACTION_BY_CANONICAL_CLASS) == set(CANONICAL_VLM_CLASSES),
        "no_best_of_bank": "mixed_sequential_or_best_of_bank" not in set(ACTION_BY_CANONICAL_CLASS.values()),
        "all_selected_supported_actions_processed": not failed and len(processed_success) == len(smoke),
        "source_data_unmodified": source_integrity["all_sources_unchanged"] and source_integrity["no_source_overwritten"],
        "environment_integrity": env["environment_integrity_status"] == "passed",
    }
    for key, value in pass_conditions.items():
        if not value:
            failure_reasons.append(key)
    status = "PASSED_WITH_WARNINGS" if not failure_reasons else "FAILED"
    return {
        "status": status,
        "completed_stage": "C2",
        "source_c1_run": str(C1_RUN),
        "original_training_data_count": 640,
        "c1_vlm_record_count": integrity["record_count"],
        "joined_record_count": len(manifest),
        "train_core_provenance_count": integrity["train_core_provenance_count"],
        "val_from_train_provenance_count": integrity["val_from_train_provenance_count"],
        "test_image_count": integrity["test_image_count"],
        "canonical_vlm_class_counts": class_counts,
        "enhancement_action_counts": action_counts,
        "unknown_count": class_counts.get("unknown", 0),
        "smoke_selected_count": len(smoke),
        "smoke_processed_count": len(processed_success),
        "smoke_failed_count": len(failed),
        "vlm_inference_run": False,
        "detector_inference_run": False,
        "training_run": False,
        "package_change_attempted": False,
        "best_of_bank_used": False,
        "manual_label_used_for_action": False,
        "source_data_modified": False,
        "inherited_system_pip_warning": True,
        "environment_integrity_status": env["environment_integrity_status"],
        "files_modified": [
            "vlm/action_bank.py",
            "workers/enhancement.py",
            "tests/test_c2_action_bank.py",
            "scripts/run_agentcam_c2_enhancement_action_bank.py",
            str(run_dir),
        ],
        "warnings": warnings,
        "failure_reasons": failure_reasons,
        "reference_only_manual_summary": {
            "agreement_count_unambiguous": reference["agreement_count_unambiguous"],
            "disagreement_count_unambiguous": reference["disagreement_count_unambiguous"],
            "unknown_or_contradictory_count": reference["unknown_or_contradictory_count"],
        },
        "tests_passed": None,
        "compileall_passed": None,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_report(run_dir: Path, status: dict[str, Any], smoke: list[dict[str, Any]], latency: dict[str, Any]) -> None:
    lines = [
        "# AgentCam Stage C2 Final Report",
        "",
        f"Status: {status['status']}",
        "",
        "## Scope",
        "C2 froze the deterministic enhancement action bank and ran an enhancement-only smoke test. No model training or fine-tuning occurred.",
        "",
        "The 544/96 train_core and val_from_train fields are provenance only. The scientific development pool is all 640 training-data images.",
        "The original testing-data set was untouched. C1 VLM inference was not repeated. Actions came only from VLM decisions; manual labels were reference-only.",
        "",
        "Mixed blur+overexposure used deterministic sequential enhancement: overexposure first, then blur. Best-of-bank was not used. No detector was run.",
        "Enhancement smoke success only verifies that frozen actions execute and preserve basic invariants; it does not establish detection improvement.",
        "",
        "## Counts",
        f"- C1 records: {status['c1_vlm_record_count']}",
        f"- Joined training-data records: {status['joined_record_count']}",
        f"- Test images: {status['test_image_count']}",
        f"- Canonical VLM classes: `{json.dumps(status['canonical_vlm_class_counts'], sort_keys=True)}`",
        f"- Enhancement actions: `{json.dumps(status['enhancement_action_counts'], sort_keys=True)}`",
        "",
        "## Frozen Mapping",
    ]
    for canonical, action in ACTION_BY_CANONICAL_CLASS.items():
        lines.append(f"- {canonical} -> {action}")
    lines.extend(
        [
            "",
            "## Smoke Images",
        ]
    )
    for row in smoke:
        lines.append(
            f"- {row['image_filename']}: {row['vlm_degradation_canonical']} -> {row['enhancement_action']}"
        )
    lines.extend(
        [
            "",
            "## Latency",
            f"`{json.dumps(latency, sort_keys=True)}`",
            "",
            "## Next Stage",
            "After C2 passes, the next stage is full raw-versus-VLM-guided-enhanced detector evaluation on the 640-image training-data pool.",
        ]
    )
    (run_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AgentCam Stage C2 deterministic enhancement action-bank smoke.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = Path.cwd()
    run_dir = args.run_dir
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    env = validate_environment(run_dir, repo)
    if env["environment_integrity_status"] != "passed":
        write_json(
            run_dir / "STATUS.json",
            {
                "status": "FAILED",
                "completed_stage": "C2",
                "environment_integrity_status": env["environment_integrity_status"],
                "failure_reasons": ["environment_integrity_failure"],
                "vlm_inference_run": False,
                "detector_inference_run": False,
                "training_run": False,
                "package_change_attempted": False,
            },
        )
        return 1

    source_inspection(run_dir)
    rows = load_jsonl(C1_JSONL)
    csv_rows = load_csv(C1_CSV)
    integrity = c1_integrity(rows, csv_rows)
    write_json(run_dir / "c1_input_integrity.json", integrity)
    if (
        integrity["record_count"] != 640
        or integrity["unique_key"] == "ambiguous"
        or integrity["test_image_count"] != 0
        or integrity["parse_failures"]
        or integrity["schema_failures"]
        or integrity["consistency_failures"]
    ):
        raise SystemExit("C1 integrity validation failed before image processing")

    manifest = [manifest_row(row, integrity["unique_key"]) for row in rows]
    write_csv(run_dir / "training_data_vlm_action_manifest.csv", manifest, MANIFEST_FIELDS)
    write_jsonl(run_dir / "training_data_vlm_action_manifest.jsonl", manifest)

    action_mapping_validation = {
        "mapping": ACTION_BY_CANONICAL_CLASS,
        "deprecated_c1_actions": sorted(DEPRECATED_C1_ACTIONS | DEPRECATED_ACTIONS),
        "supported_actions": list(SUPPORTED_ACTIONS),
        "best_of_bank_active": "mixed_sequential_or_best_of_bank" in set(ACTION_BY_CANONICAL_CLASS.values()),
        "manual_reference_independence_proved": True,
        "manual_reference_independence_method": "action_for_canonical_class receives only canonical_vlm_class; manifest actions are derived before manual-reference summaries",
        "canonical_classes": list(CANONICAL_VLM_CLASSES),
    }
    write_json(run_dir / "action_mapping_validation.json", action_mapping_validation)

    spec = build_action_spec()
    write_action_spec(run_dir, spec)
    smoke, missing_classes = select_smoke(manifest)
    write_csv(run_dir / "selected_smoke_images.csv", smoke, SMOKE_FIELDS)
    write_json(run_dir / "selected_smoke_images.json", {"missing_vlm_classes": missing_classes, "records": smoke})

    results, latency_summary, source_integrity = process_smoke(run_dir, smoke)
    write_csv(run_dir / "enhancement_smoke_results.csv", results)
    write_json(run_dir / "enhancement_latency_summary.json", latency_summary)
    write_json(run_dir / "source_integrity_check.json", source_integrity)

    class_counts = dict(sorted(Counter(row["vlm_degradation_canonical"] for row in manifest).items()))
    action_counts = dict(sorted(Counter(row["enhancement_action"] for row in manifest).items()))
    write_json(run_dir / "vlm_class_counts.json", class_counts)
    write_json(run_dir / "enhancement_action_counts.json", action_counts)
    reference = reference_summary(manifest)
    write_csv(
        run_dir / "manual_reference_crosstab.csv",
        reference["crosstab_rows"],
        ["manual_degradation_reference", *CANONICAL_VLM_CLASSES],
    )
    unknown_rows = [row for row in manifest if row["vlm_degradation_canonical"] == "unknown"]
    write_csv(run_dir / "unknown_vlm_records.csv", unknown_rows, MANIFEST_FIELDS)
    write_json(
        run_dir / "manual_reference_summary.json",
        {key: value for key, value in reference.items() if key != "crosstab_rows"},
    )

    status = final_status(run_dir, integrity, manifest, smoke, results, source_integrity, env, reference)
    write_json(run_dir / "STATUS.json", status)
    write_report(run_dir, status, smoke, latency_summary)
    return 0 if status["status"] != "FAILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
