from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from data_utils.manifest import ManifestRow, load_yaml, read_manifest, write_text_file


@dataclass
class LabelRecord:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def parse_yolo_label_line(line: str, line_number: int = 1) -> LabelRecord:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"line {line_number}: expected 5 fields, got {len(parts)}")
    try:
        class_id = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"line {line_number}: class_id must be an integer") from exc
    try:
        values = [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"line {line_number}: bbox values must be floats") from exc
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"line {line_number}: bbox values must be in [0, 1]")
    if values[2] <= 0.0 or values[3] <= 0.0:
        raise ValueError(f"line {line_number}: bbox width and height must be positive")
    return LabelRecord(class_id, values[0], values[1], values[2], values[3])


def parse_yolo_label_file(path: Path, class_count: Optional[int] = None) -> List[LabelRecord]:
    records: List[LabelRecord] = []
    for idx, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = parse_yolo_label_line(line, idx)
        if class_count is not None and not (0 <= record.class_id < class_count):
            raise ValueError(f"line {idx}: class_id {record.class_id} is outside [0, {class_count - 1}]")
        records.append(record)
    return records


def names_from_data_yaml(path: Optional[Path]) -> List[str]:
    if path is None or not path.exists():
        return ["class_0"]
    data = load_yaml(path)
    names = data.get("names", ["class_0"])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(f"Unsupported names format in {path}")


def validate_rows(rows: Sequence[ManifestRow], class_count: int, limit: int = 0) -> Tuple[List[str], Dict[str, Counter]]:
    errors: List[str] = []
    counters: Dict[str, Counter] = {
        "original_split": Counter(),
        "split": Counter(),
        "degradation_type": Counter(),
        "label_state": Counter(),
        "split_degradation": Counter(),
    }
    basename_counts = Counter(Path(row.image_path).name for row in rows)

    selected_rows = list(rows)
    if limit and limit > 0:
        selected_rows = selected_rows[:limit]

    for row in selected_rows:
        image_path = Path(row.image_path)
        counters["original_split"][row.original_split] += 1
        counters["split"][row.split] += 1
        counters["degradation_type"][row.degradation_type] += 1
        counters["split_degradation"][(row.split, row.degradation_type)] += 1
        if basename_counts[image_path.name] > 1:
            errors.append(f"{image_path}: duplicate image basename {image_path.name}")

        try:
            with Image.open(image_path) as im:
                width, height = im.size
            if width <= 0 or height <= 0:
                errors.append(f"{image_path}: width and height must be positive")
        except Exception as exc:
            errors.append(f"{image_path}: image could not be opened ({exc})")

        if not row.has_label:
            counters["label_state"]["missing_label_unlabeled"] += 1
            if row.label_path:
                errors.append(f"{image_path}: has_label is false but label_path is populated")
            continue

        label_path = Path(row.label_path)
        if not label_path.exists():
            counters["label_state"]["missing_label_unlabeled"] += 1
            errors.append(f"{image_path}: manifest label file does not exist: {label_path}")
            continue

        if label_path.read_text().strip() == "":
            counters["label_state"]["empty_label_negative"] += 1
            if not row.is_empty_label:
                errors.append(f"{label_path}: manifest should mark empty label file as is_empty_label=true")
            continue

        counters["label_state"]["labeled_positive_or_mixed"] += 1
        try:
            parse_yolo_label_file(label_path, class_count=class_count)
        except ValueError as exc:
            errors.append(f"{label_path}: {exc}")

    return errors, counters


def format_validation_report(errors: List[str], counters: Dict[str, Counter]) -> str:
    lines = ["# YOLO Validation Report", ""]
    lines.append(f"- errors: {len(errors)}")
    lines.append("")
    for name, counter in counters.items():
        lines.append(f"## {name}")
        lines.append("")
        for key, value in sorted(counter.items(), key=lambda item: str(item[0])):
            if isinstance(key, tuple):
                key_text = " / ".join(str(part) for part in key)
            else:
                key_text = str(key)
            lines.append(f"- {key_text}: {value}")
        lines.append("")
    if errors:
        lines.append("## Errors")
        lines.append("")
        lines.extend(f"- {error}" for error in errors)
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate AgentCam YOLO labels and manifest.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/agentcam"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data_yaml", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/validation_report.md"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", "--dry-run", action="store_true", dest="dry_run")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest or args.data / "manifest.csv"
    data_yaml_path = args.data_yaml or args.data / "data_trainval.yaml"
    rows = read_manifest(manifest_path)
    names = names_from_data_yaml(data_yaml_path)
    errors, counters = validate_rows(rows, class_count=len(names), limit=args.limit)
    report = format_validation_report(errors, counters)
    if args.dry_run:
        print(report)
    else:
        write_text_file(args.out, report)
        print(f"Wrote validation report to {args.out}")
    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
