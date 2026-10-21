from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from vlm.validate_vlm_outputs import REQUIRED_FIELDS, validate_consistency, validate_schema_fields


ACTION_UNKNOWN = "unknown_defer_or_vlm_judge"
SMOKE_LABEL_ORDER = ["blur", "overexposure", "blur_overexposure", "underexposure", "clear"]
CSV_COLUMNS = [
    "image_id",
    "image_path",
    "split",
    "source_split",
    "folder_label",
    "metadata_degradation_label",
    "width",
    "height",
    "selected_model_id",
    "fallback_used",
    "blur",
    "overexposure",
    "underexposure",
    "clear",
    "unknown_degradation",
    "severity",
    "affected_region",
    "confidence",
    "recommended_enhancement_action",
    "explanation",
    "parse_valid",
    "schema_valid",
    "consistency_valid",
    "raw_response_path",
    "latency_ms",
]


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def image_id_from_path(path: str) -> str:
    return Path(path).stem


def select_rows(
    rows: List[Dict[str, str]],
    splits: Sequence[str],
    limit: int = 0,
    smoke_sample: bool = False,
    smoke_per_label: int = 2,
) -> List[Dict[str, str]]:
    allowed = set(splits)
    if "test" in allowed:
        raise ValueError("test split is forbidden for the Qwen VLM degradation audit")
    selected = [row for row in rows if row.get("split") in allowed]
    selected.sort(key=lambda row: (row.get("split", ""), row.get("manual_degradation", ""), row.get("image_path", "")))
    if smoke_sample:
        buckets: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in selected:
            buckets[row.get("manual_degradation") or row.get("degradation_type") or "unknown"].append(row)
        sample: List[Dict[str, str]] = []
        for label in SMOKE_LABEL_ORDER:
            take = 1 if label == "clear" else smoke_per_label
            sample.extend(buckets.get(label, [])[:take])
        return sample
    if limit and limit > 0:
        return selected[:limit]
    return selected


def load_completed(output_jsonl: Path) -> set[str]:
    completed: set[str] = set()
    if not output_jsonl.exists():
        return completed
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_id = row.get("image_id")
            if isinstance(image_id, str):
                completed.add(image_id)
    return completed


def extract_one_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value, None
        return None, "decoded JSON is not an object"
    except json.JSONDecodeError:
        pass

    matches = list(re.finditer(r"\{", stripped))
    decoded: List[Dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for match in matches:
        try:
            value, end = decoder.raw_decode(stripped[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            trailing = stripped[match.start() + end :].strip()
            decoded.append(value)
            if trailing and trailing not in {"```"}:
                continue
    if len(decoded) == 1:
        return decoded[0], None
    if not decoded:
        return None, "no JSON object found"
    return None, "more than one JSON object found"


def fallback_payload(image_id: str) -> Dict[str, Any]:
    return {
        "image_id": image_id,
        "blur": False,
        "overexposure": False,
        "underexposure": False,
        "clear": False,
        "unknown_degradation": True,
        "severity": "moderate",
        "affected_region": "unknown",
        "confidence": 0.0,
        "recommended_enhancement_action": ACTION_UNKNOWN,
        "explanation": "Invalid model output; deferred for review.",
    }


def normalize_for_csv(payload: Dict[str, Any], image_id: str) -> Dict[str, Any]:
    normalized = fallback_payload(image_id)
    for key in REQUIRED_FIELDS:
        if key in payload:
            normalized[key] = payload[key]
    return normalized


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def load_qwen_model(model_id: str, device: str):
    import torch
    import transformers
    from transformers import AutoProcessor

    model_errors: List[str] = []
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    class_names: List[str] = []
    lowered = model_id.lower()
    if "qwen3-vl" in lowered:
        class_names.append("Qwen3VLForConditionalGeneration")
    if "qwen2.5-vl" in lowered or "qwen2_5-vl" in lowered:
        class_names.append("Qwen2_5_VLForConditionalGeneration")
    class_names.extend(["AutoModelForImageTextToText", "AutoModelForVision2Seq"])
    for class_name in class_names:
        try:
            cls = getattr(transformers, class_name)
            model = cls.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
            model.to(device)
            model.eval()
            return processor, model
        except Exception as exc:  # pragma: no cover - depends on installed transformers/model support
            model_errors.append(f"{class_name}: {repr(exc)}")
    raise RuntimeError("; ".join(model_errors))


def run_model(processor: Any, model: Any, image_path: Path, prompt: str, device: str, max_new_tokens: int, temperature: float, do_sample: bool) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generate_kwargs)
    input_ids = inputs.get("input_ids")
    if input_ids is not None:
        generated_ids = generated_ids[:, input_ids.shape[1] :]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def audit_rows(
    rows: List[Dict[str, str]],
    manifest_root: Path,
    schema: Dict[str, Any],
    prompt: str,
    model_id: str,
    fallback_used: bool,
    processor: Any,
    model: Any,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    raw_dir = args.raw_response_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = load_completed(args.output_jsonl) if args.resume else set()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.output_jsonl.open("a" if args.resume else "w", encoding="utf-8") as jsonl:
        for row in rows:
            image_id = image_id_from_path(row["image_path"])
            if image_id in existing:
                continue
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = manifest_root / image_path
            if row.get("split") == "test":
                raise ValueError(f"test image encountered and refused: {row['image_path']}")
            with Image.open(image_path) as image:
                width, height = image.size
            start = time.perf_counter()
            raw_response = run_model(
                processor=processor,
                model=model,
                image_path=image_path,
                prompt=prompt.replace("{image_id}", image_id),
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            raw_response_path = raw_dir / f"{image_id}.txt"
            raw_response_path.write_text(raw_response, encoding="utf-8")
            parsed, parse_error = extract_one_json_object(raw_response)
            parse_valid = parsed is not None
            parsed_payload = parsed if parsed is not None else fallback_payload(image_id)
            schema_errors = validate_schema_fields(parsed_payload, schema) if parse_valid else ["parse failed"]
            consistency_errors = validate_consistency(parsed_payload) if parse_valid and not schema_errors else ["schema failed"]
            schema_valid = parse_valid and not schema_errors
            consistency_valid = schema_valid and not consistency_errors
            csv_payload = normalize_for_csv(parsed_payload if parse_valid and schema_valid and consistency_valid else fallback_payload(image_id), image_id)
            audit_row: Dict[str, Any] = {
                "image_id": image_id,
                "image_path": row["image_path"],
                "split": row.get("split", ""),
                "source_split": row.get("source_split", row.get("original_split", "")),
                "folder_label": row.get("degradation_type", ""),
                "metadata_degradation_label": row.get("manual_degradation", ""),
                "width": width,
                "height": height,
                "selected_model_id": model_id,
                "fallback_used": fallback_used,
                **csv_payload,
                "parse_valid": parse_valid,
                "schema_valid": schema_valid,
                "consistency_valid": consistency_valid,
                "raw_response_path": str(raw_response_path),
                "latency_ms": latency_ms,
                "raw_response": raw_response,
                "parsed_json": parsed if parse_valid else None,
                "parse_error": parse_error,
                "schema_errors": schema_errors,
                "consistency_errors": consistency_errors,
            }
            jsonl.write(json.dumps(audit_row, sort_keys=True) + "\n")
            jsonl.flush()
            output_rows.append(audit_row)
            write_csv(args.output_csv, output_rows)
    return output_rows


def write_validation_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    report = {
        "total_rows": len(rows),
        "parse_failures": sum(1 for row in rows if not row.get("parse_valid")),
        "schema_failures": sum(1 for row in rows if not row.get("schema_valid")),
        "consistency_failures": sum(1 for row in rows if not row.get("consistency_valid")),
        "invalid_rows": sum(1 for row in rows if not (row.get("parse_valid") and row.get("schema_valid") and row.get("consistency_valid"))),
        "passed": all(row.get("parse_valid") and row.get("schema_valid") and row.get("consistency_valid") for row in rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local Qwen-VL degradation assessment on AgentCam images.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/agentcam/manifest_with_manual_degradation.csv"))
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--splits", default="train_core,val_from_train")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-sample", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--raw-response-dir", type=Path, required=True)
    parser.add_argument("--model-load-audit", type=Path)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--fallback-model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", default="false")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.do_sample = str(args.do_sample).strip().lower() in {"1", "true", "yes", "y"}
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    if "test" in splits:
        raise ValueError("test split is forbidden")
    prompt = args.prompt.read_text(encoding="utf-8")
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    manifest_rows = read_manifest(args.manifest)
    selected_rows = select_rows(manifest_rows, splits=splits, limit=args.limit, smoke_sample=args.smoke_sample)
    if not selected_rows:
        raise ValueError("no images selected")

    errors: List[Tuple[str, str]] = []
    model_load_audit: Dict[str, Any] = {"attempts": []}
    fallback_used = False
    try:
        processor, model = load_qwen_model(args.model_id, args.device)
        selected_model_id = args.model_id
        model_load_audit["attempts"].append({"model_id": args.model_id, "status": "loaded"})
    except Exception as exc:
        errors.append((args.model_id, repr(exc)))
        model_load_audit["attempts"].append({"model_id": args.model_id, "status": "failed", "error": repr(exc)})
        fallback_used = True
        try:
            processor, model = load_qwen_model(args.fallback_model_id, args.device)
            selected_model_id = args.fallback_model_id
            model_load_audit["attempts"].append({"model_id": args.fallback_model_id, "status": "loaded"})
        except Exception as fallback_exc:
            errors.append((args.fallback_model_id, repr(fallback_exc)))
            model_load_audit["attempts"].append(
                {"model_id": args.fallback_model_id, "status": "failed", "error": repr(fallback_exc)}
            )
            if args.model_load_audit:
                args.model_load_audit.parent.mkdir(parents=True, exist_ok=True)
                args.model_load_audit.write_text(json.dumps(model_load_audit, indent=2, sort_keys=True), encoding="utf-8")
            raise RuntimeError(json.dumps({"model_load_errors": errors}, indent=2)) from fallback_exc
    model_load_audit.update({"selected_model_id": selected_model_id, "fallback_used": fallback_used})
    if args.model_load_audit:
        args.model_load_audit.parent.mkdir(parents=True, exist_ok=True)
        args.model_load_audit.write_text(json.dumps(model_load_audit, indent=2, sort_keys=True), encoding="utf-8")

    rows = audit_rows(
        rows=selected_rows,
        manifest_root=Path.cwd(),
        schema=schema,
        prompt=prompt,
        model_id=selected_model_id,
        fallback_used=fallback_used,
        processor=processor,
        model=model,
        args=args,
    )
    write_validation_report(args.validation_report, rows)
    print(
        json.dumps(
            {
                "selected_model_id": selected_model_id,
                "fallback_used": fallback_used,
                "processed_images": len(rows),
                "splits": splits,
                "test_images_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
