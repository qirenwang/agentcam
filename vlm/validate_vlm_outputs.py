from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


BOOLEAN_FIELDS = ["blur", "overexposure", "underexposure", "clear", "unknown_degradation"]
REQUIRED_FIELDS = [
    "image_id",
    *BOOLEAN_FIELDS,
    "severity",
    "affected_region",
    "confidence",
    "recommended_enhancement_action",
    "explanation",
]
SEVERITIES = {"none", "mild", "moderate", "severe"}
ACTIONS = {
    "raw_no_enhancement",
    "overexposure_highlight_compression",
    "underexposure_gamma_clahe",
    "blur_unsharp_or_deblur",
    "mixed_sequential_or_best_of_bank",
    "unknown_defer_or_vlm_judge",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                rows.append({"_line_number": line_number, "_json_error": str(exc), "_raw": stripped})
                continue
            if isinstance(value, dict):
                value.setdefault("_line_number", line_number)
                rows.append(value)
            else:
                rows.append({"_line_number": line_number, "_json_error": "row is not a JSON object", "_raw": value})
    return rows


def output_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    parsed = row.get("parsed_json")
    if isinstance(parsed, dict):
        return parsed
    return row


def word_count(text: str) -> int:
    return len([part for part in text.strip().split() if part])


def validate_schema_fields(payload: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
        return errors

    allowed = set(schema.get("properties", {}).keys()) or set(REQUIRED_FIELDS)
    extras = sorted(key for key in payload.keys() if key not in allowed and not key.startswith("_"))
    if extras:
        errors.append(f"unexpected fields: {', '.join(extras)}")

    if not isinstance(payload["image_id"], str):
        errors.append("image_id must be a string")
    for field in BOOLEAN_FIELDS:
        if not isinstance(payload[field], bool):
            errors.append(f"{field} must be a boolean")
    if payload["severity"] not in SEVERITIES:
        errors.append(f"severity must be one of {sorted(SEVERITIES)}")
    if not isinstance(payload["affected_region"], str):
        errors.append("affected_region must be a string")
    confidence = payload["confidence"]
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(float(confidence)):
        errors.append("confidence must be a finite number")
    elif float(confidence) < 0.0 or float(confidence) > 1.0:
        errors.append("confidence must be in [0, 1]")
    if payload["recommended_enhancement_action"] not in ACTIONS:
        errors.append(f"recommended_enhancement_action must be one of {sorted(ACTIONS)}")
    if not isinstance(payload["explanation"], str):
        errors.append("explanation must be a string")
    elif word_count(payload["explanation"]) > 40:
        errors.append("explanation must be 40 words or fewer")
    return errors


def expected_action(payload: Dict[str, Any]) -> Optional[str]:
    confidence = float(payload.get("confidence", 0.0))
    if confidence < 0.50:
        return "unknown_defer_or_vlm_judge"
    if payload.get("unknown_degradation") is True:
        return "unknown_defer_or_vlm_judge"
    if payload.get("clear") is True:
        return "raw_no_enhancement"

    labels = [name for name in ("blur", "overexposure", "underexposure") if payload.get(name) is True]
    if len(labels) > 1:
        return "mixed_sequential_or_best_of_bank"
    if labels == ["blur"]:
        return "blur_unsharp_or_deblur"
    if labels == ["overexposure"]:
        return "overexposure_highlight_compression"
    if labels == ["underexposure"]:
        return "underexposure_gamma_clahe"
    return None


def validate_consistency(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    blur = payload.get("blur")
    over = payload.get("overexposure")
    under = payload.get("underexposure")
    clear = payload.get("clear")
    unknown = payload.get("unknown_degradation")
    action = payload.get("recommended_enhancement_action")

    if clear is True:
        if blur or over or under or unknown:
            errors.append("clear=true must not coexist with blur, overexposure, underexposure, or unknown_degradation")
        if payload.get("severity") != "none":
            errors.append('clear=true implies severity="none"')
        if action != "raw_no_enhancement":
            errors.append('clear=true implies recommended_enhancement_action="raw_no_enhancement"')

    if unknown is True and action != "unknown_defer_or_vlm_judge":
        errors.append('unknown_degradation=true implies recommended_enhancement_action="unknown_defer_or_vlm_judge"')

    confidence = payload.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and float(confidence) < 0.50:
        if action != "unknown_defer_or_vlm_judge":
            errors.append('confidence < 0.50 implies recommended_enhancement_action="unknown_defer_or_vlm_judge"')

    expected = expected_action(payload)
    if expected is not None and action != expected:
        errors.append(f'expected recommended_enhancement_action="{expected}" from labels/confidence')
    return errors


def validate_row(row: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, bool, bool, List[str]]:
    if "_json_error" in row:
        return False, False, False, [str(row["_json_error"])]
    payload = output_payload(row)
    schema_errors = validate_schema_fields(payload, schema)
    if schema_errors:
        return True, False, False, schema_errors
    consistency_errors = validate_consistency(payload)
    return True, True, not consistency_errors, consistency_errors


def validate_file(input_jsonl: Path, schema_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    rows = load_jsonl(input_jsonl)
    invalid_rows: List[Dict[str, Any]] = []
    parse_failures = 0
    schema_failures = 0
    consistency_failures = 0
    for row in rows:
        parse_valid, schema_valid, consistency_valid, errors = validate_row(row, schema)
        if not parse_valid:
            parse_failures += 1
        elif not schema_valid:
            schema_failures += 1
        elif not consistency_valid:
            consistency_failures += 1
        if errors:
            invalid_rows.append(
                {
                    "line_number": row.get("_line_number"),
                    "image_id": output_payload(row).get("image_id"),
                    "parse_valid": parse_valid,
                    "schema_valid": schema_valid,
                    "consistency_valid": consistency_valid,
                    "errors": errors,
                    "row": row,
                }
            )
    report = {
        "input_jsonl": str(input_jsonl),
        "schema": str(schema_path),
        "total_rows": len(rows),
        "valid_rows": len(rows) - len(invalid_rows),
        "invalid_rows": len(invalid_rows),
        "parse_failures": parse_failures,
        "schema_failures": schema_failures,
        "consistency_failures": consistency_failures,
        "passed": not invalid_rows,
    }
    return report, invalid_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate strict JSON VLM degradation outputs.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--invalid-rows", type=Path, required=True)
    parser.add_argument("--allow-invalid", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report, invalid_rows = validate_file(args.input_jsonl, args.schema)
    args.validation_report.parent.mkdir(parents=True, exist_ok=True)
    args.invalid_rows.parent.mkdir(parents=True, exist_ok=True)
    args.validation_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with args.invalid_rows.open("w", encoding="utf-8") as f:
        for row in invalid_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if invalid_rows and not args.allow_invalid:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

