from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ANNOTATION_SOURCE = "manual_name_list_v1"

MANUAL_LABEL_COLUMNS = [
    "image_id",
    "image_filename",
    "source_split",
    "processed_split",
    "raw_category",
    "manual_degradation",
    "include_primary_eval",
    "evaluation_role",
    "annotation_status",
    "annotation_source",
]

ENRICHED_COLUMNS = [
    "source_split",
    "manual_degradation",
    "manual_degradation_raw",
    "include_primary_eval",
    "evaluation_role",
    "manual_annotation_status",
    "manual_annotation_source",
]

DUPLICATE_COLUMNS = [
    "source_split",
    "image_id",
    "raw_category",
    "manual_degradation",
    "line_number",
    "duplicate_of_line_number",
]
CONFLICT_COLUMNS = [
    "source_split",
    "image_id",
    "manual_degradations",
    "raw_categories",
    "line_numbers",
]
CROSS_SPLIT_COLUMNS = ["image_id", "source_splits"]
UNMATCHED_COLUMNS = [
    "source_split",
    "image_id",
    "raw_category",
    "manual_degradation",
    "line_number",
    "reason",
    "match_count",
]
UNLISTED_COLUMNS = [
    "source_split",
    "processed_split",
    "image_id",
    "image_path",
    "label_path",
    "reason",
]

DIAGNOSTIC_PATHS = {
    "duplicates": Path("artifacts/manual_degradation_duplicates.csv"),
    "conflicts": Path("artifacts/manual_degradation_conflicts.csv"),
    "cross_split": Path("artifacts/manual_degradation_cross_split.csv"),
    "unmatched": Path("artifacts/manual_degradation_unmatched.csv"),
    "unlisted": Path("artifacts/manual_degradation_unlisted.csv"),
}

SOURCE_SPLIT_RANK = {"training-data": 0, "testing-data": 1, "unknown": 9}
PROCESSED_SPLIT_RANK = {"train_core": 0, "val_from_train": 1, "test": 2}
ALLOWED_PROCESSED_SPLITS = {
    "training-data": {"train_core", "val_from_train"},
    "testing-data": {"test"},
}

CATEGORY_ORDER = [
    "blur",
    "overexposure",
    "blur_overexposure",
    "underexposure",
    "clear",
    "unknown",
]

DEFAULT_EXPECTED_COUNTS = {
    "training-data": {
        "blur": 234,
        "overexposure": 298,
        "blur_overexposure": 67,
        "underexposure": 37,
        "clear": 4,
    },
    "testing-data": {
        "blur": 43,
        "overexposure": 73,
        "blur_overexposure": 37,
        "underexposure": 6,
        "clear": 1,
    },
}

SPECIFIC_VALIDATIONS = {
    "202410311251146380097": ("training-data", "blur"),
    "202410311719239751521": ("training-data", "overexposure"),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_EXTENSIONS = {".txt"}


@dataclass(frozen=True)
class ManualRecord:
    image_id: str
    image_filename: str
    source_split: str
    raw_category: str
    manual_degradation: str
    line_number: int


@dataclass
class ManualAnalysis:
    records: List[ManualRecord]
    unique_records: List[ManualRecord]
    duplicates: List[Dict[str, str]]
    conflicts: List[Dict[str, str]]
    cross_split: List[Dict[str, str]]
    unique_pair_count: int


@dataclass
class MatchResult:
    manual_labels: List[Dict[str, str]]
    enriched_manifest: List[Dict[str, str]]
    unmatched: List[Dict[str, str]]
    unlisted: List[Dict[str, str]]
    matched_count: int


@dataclass
class ZipMetadata:
    entry_count: int = 0
    image_extensions: List[str] = None  # type: ignore[assignment]
    split_image_counts: Dict[str, int] = None  # type: ignore[assignment]
    duplicate_basenames: Dict[str, Dict[str, int]] = None  # type: ignore[assignment]
    basename_overlap: List[str] = None  # type: ignore[assignment]
    metadata_files: List[str] = None  # type: ignore[assignment]
    json_reports: List[Dict[str, object]] = None  # type: ignore[assignment]
    error: str = ""

    def __post_init__(self) -> None:
        if self.image_extensions is None:
            self.image_extensions = []
        if self.split_image_counts is None:
            self.split_image_counts = {}
        if self.duplicate_basenames is None:
            self.duplicate_basenames = {}
        if self.basename_overlap is None:
            self.basename_overlap = []
        if self.metadata_files is None:
            self.metadata_files = []
        if self.json_reports is None:
            self.json_reports = []


def normalized_line_iter(text: str) -> Iterable[Tuple[int, str]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u2028", "\n").replace("\u2029", "\n")
    for line_number, line in enumerate(normalized.split("\n"), start=1):
        stripped = line.strip()
        if stripped:
            yield line_number, stripped


def canonical_heading(value: str) -> str:
    value = value.strip().rstrip(":").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def heading_key(value: str) -> str:
    return canonical_heading(value).replace("_", " ").replace("-", " ")


def normalize_source_split(value: str) -> Optional[str]:
    key = heading_key(value)
    aliases = {
        "train": "training-data",
        "training": "training-data",
        "training data": "training-data",
        "trainingdata": "training-data",
        "training-data": "training-data",
        "training_data": "training-data",
        "test": "testing-data",
        "testing": "testing-data",
        "testing data": "testing-data",
        "testingdata": "testing-data",
        "testing-data": "testing-data",
        "testing_data": "testing-data",
    }
    return aliases.get(key) or aliases.get(key.replace(" ", ""))


def normalize_category(value: str) -> Optional[str]:
    raw = canonical_heading(value)
    compact = raw.replace(" ", "").replace("_", "").replace("-", "")
    aliases = {
        "blur": "blur",
        "exporsure": "overexposure",
        "exposure": "overexposure",
        "overexposure": "overexposure",
        "blur&overexposure": "blur_overexposure",
        "blurandoverexposure": "blur_overexposure",
        "bluroverexposure": "blur_overexposure",
        "underexposure": "underexposure",
        "clear": "clear",
    }
    return aliases.get(compact)


def parse_image_token(value: str) -> Optional[Tuple[str, str]]:
    token = value.strip().lstrip("-*").strip().rstrip(",;")
    match = re.fullmatch(r"(\d+)(\.[A-Za-z0-9]+)?", token)
    if not match:
        return None
    image_id = match.group(1)
    ext = match.group(2) or ""
    if ext and ext.lower() not in IMAGE_EXTENSIONS | LABEL_EXTENSIONS:
        return None
    return image_id, f"{image_id}{ext}" if ext else image_id


def parse_manual_text(text: str) -> List[ManualRecord]:
    records: List[ManualRecord] = []
    current_source = "unknown"
    current_raw_category = "unknown"
    current_degradation = "unknown"

    for line_number, line in normalized_line_iter(text):
        source_split = normalize_source_split(line)
        if source_split:
            current_source = source_split
            continue

        category = normalize_category(line)
        if category:
            current_raw_category = canonical_heading(line)
            current_degradation = category
            continue

        image = parse_image_token(line)
        if image:
            image_id, image_filename = image
            records.append(
                ManualRecord(
                    image_id=image_id,
                    image_filename=image_filename,
                    source_split=current_source,
                    raw_category=current_raw_category,
                    manual_degradation=current_degradation,
                    line_number=line_number,
                )
            )
    return records


def parse_manual_file(path: Path) -> List[ManualRecord]:
    return parse_manual_text(path.read_text(encoding="utf-8"))


def record_sort_key(record: ManualRecord) -> Tuple[int, str, int]:
    return (
        SOURCE_SPLIT_RANK.get(record.source_split, 9),
        record.image_id,
        record.line_number,
    )


def analyze_manual_records(records: Sequence[ManualRecord]) -> ManualAnalysis:
    sorted_records = sorted(records, key=record_sort_key)
    by_pair: Dict[Tuple[str, str], List[ManualRecord]] = defaultdict(list)
    by_image_id: Dict[str, set[str]] = defaultdict(set)
    for record in sorted_records:
        by_pair[(record.source_split, record.image_id)].append(record)
        if record.source_split != "unknown":
            by_image_id[record.image_id].add(record.source_split)

    duplicates: List[Dict[str, str]] = []
    conflicts: List[Dict[str, str]] = []
    unique_records: List[ManualRecord] = []

    for key in sorted(by_pair, key=lambda item: (SOURCE_SPLIT_RANK.get(item[0], 9), item[1])):
        pair_records = sorted(by_pair[key], key=lambda record: record.line_number)
        categories = sorted({record.manual_degradation for record in pair_records})
        if len(categories) > 1:
            conflicts.append(
                {
                    "source_split": key[0],
                    "image_id": key[1],
                    "manual_degradations": ";".join(categories),
                    "raw_categories": ";".join(record.raw_category for record in pair_records),
                    "line_numbers": ";".join(str(record.line_number) for record in pair_records),
                }
            )
            continue

        first = pair_records[0]
        unique_records.append(first)
        for duplicate in pair_records[1:]:
            duplicates.append(
                {
                    "source_split": duplicate.source_split,
                    "image_id": duplicate.image_id,
                    "raw_category": duplicate.raw_category,
                    "manual_degradation": duplicate.manual_degradation,
                    "line_number": str(duplicate.line_number),
                    "duplicate_of_line_number": str(first.line_number),
                }
            )

    cross_split = []
    for image_id, splits in sorted(by_image_id.items()):
        if len(splits) > 1:
            cross_split.append({"image_id": image_id, "source_splits": ";".join(sorted(splits))})

    return ManualAnalysis(
        records=sorted_records,
        unique_records=sorted(unique_records, key=record_sort_key),
        duplicates=duplicates,
        conflicts=conflicts,
        cross_split=cross_split,
        unique_pair_count=len(by_pair),
    )


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def source_split_for_manifest_row(row: Dict[str, str]) -> str:
    original_split = row.get("original_split", "").strip().lower()
    if original_split == "train":
        return "training-data"
    if original_split == "test":
        return "testing-data"

    path_text = row.get("image_path", "").lower()
    if "training-data" in path_text:
        return "training-data"
    if "testing-data" in path_text:
        return "testing-data"
    return "unknown"


def image_id_from_path(path_value: str) -> str:
    return Path(path_value).stem


def include_policy(manual_degradation: str, annotation_status: str = "matched") -> Tuple[str, str]:
    if annotation_status != "matched":
        return "false", annotation_status
    if manual_degradation == "blur":
        return "true", "primary_blur"
    if manual_degradation == "overexposure":
        return "true", "primary_overexposure"
    if manual_degradation == "blur_overexposure":
        return "false", "secondary_combined_degradation"
    if manual_degradation == "underexposure":
        return "false", "out_of_scope_underexposure"
    if manual_degradation == "clear":
        return "false", "out_of_scope_clear"
    return "false", "excluded_unknown"


def manual_label_row(record: ManualRecord, manifest_row: Dict[str, str]) -> Dict[str, str]:
    include_primary_eval, evaluation_role = include_policy(record.manual_degradation)
    return {
        "image_id": record.image_id,
        "image_filename": Path(manifest_row["image_path"]).name,
        "source_split": record.source_split,
        "processed_split": manifest_row["split"],
        "raw_category": record.raw_category,
        "manual_degradation": record.manual_degradation,
        "include_primary_eval": include_primary_eval,
        "evaluation_role": evaluation_role,
        "annotation_status": "matched",
        "annotation_source": ANNOTATION_SOURCE,
    }


def match_manual_to_manifest(
    analysis: ManualAnalysis,
    manifest_rows: Sequence[Dict[str, str]],
) -> MatchResult:
    manifest_by_key: Dict[Tuple[str, str], List[Tuple[int, Dict[str, str]]]] = defaultdict(list)
    for idx, row in enumerate(manifest_rows):
        source_split = source_split_for_manifest_row(row)
        image_id = image_id_from_path(row.get("image_path", ""))
        row["_manual_source_split"] = source_split
        row["_manual_image_id"] = image_id
        manifest_by_key[(source_split, image_id)].append((idx, row))

    matched_by_manifest_index: Dict[int, ManualRecord] = {}
    manual_labels: List[Dict[str, str]] = []
    unmatched: List[Dict[str, str]] = []

    for record in analysis.unique_records:
        key = (record.source_split, record.image_id)
        matches = manifest_by_key.get(key, [])
        allowed_splits = ALLOWED_PROCESSED_SPLITS.get(record.source_split, set())
        matches = [(idx, row) for idx, row in matches if row.get("split") in allowed_splits]
        if len(matches) != 1:
            unmatched.append(
                {
                    "source_split": record.source_split,
                    "image_id": record.image_id,
                    "raw_category": record.raw_category,
                    "manual_degradation": record.manual_degradation,
                    "line_number": str(record.line_number),
                    "reason": "zero_manifest_matches" if not matches else "multiple_manifest_matches",
                    "match_count": str(len(matches)),
                }
            )
            continue

        manifest_index, manifest_row = matches[0]
        if manifest_index in matched_by_manifest_index:
            previous = matched_by_manifest_index[manifest_index]
            unmatched.append(
                {
                    "source_split": record.source_split,
                    "image_id": record.image_id,
                    "raw_category": record.raw_category,
                    "manual_degradation": record.manual_degradation,
                    "line_number": str(record.line_number),
                    "reason": f"manifest_row_already_matched_by_line_{previous.line_number}",
                    "match_count": "1",
                }
            )
            continue
        matched_by_manifest_index[manifest_index] = record
        manual_labels.append(manual_label_row(record, manifest_row))

    unlisted: List[Dict[str, str]] = []
    enriched_manifest: List[Dict[str, str]] = []
    for idx, row in enumerate(manifest_rows):
        base = {key: value for key, value in row.items() if not key.startswith("_manual_")}
        source_split = row.get("_manual_source_split", source_split_for_manifest_row(row))
        image_id = row.get("_manual_image_id", image_id_from_path(row.get("image_path", "")))
        record = matched_by_manifest_index.get(idx)
        if record:
            include_primary_eval, evaluation_role = include_policy(record.manual_degradation)
            additions = {
                "source_split": source_split,
                "manual_degradation": record.manual_degradation,
                "manual_degradation_raw": record.raw_category,
                "include_primary_eval": include_primary_eval,
                "evaluation_role": evaluation_role,
                "manual_annotation_status": "matched",
                "manual_annotation_source": ANNOTATION_SOURCE,
            }
        else:
            unlisted.append(
                {
                    "source_split": source_split,
                    "processed_split": row.get("split", ""),
                    "image_id": image_id,
                    "image_path": row.get("image_path", ""),
                    "label_path": row.get("label_path", ""),
                    "reason": "manifest_row_has_no_manual_annotation",
                }
            )
            include_primary_eval, evaluation_role = include_policy("unknown", annotation_status="unlisted")
            additions = {
                "source_split": source_split,
                "manual_degradation": "unknown",
                "manual_degradation_raw": "",
                "include_primary_eval": include_primary_eval,
                "evaluation_role": evaluation_role,
                "manual_annotation_status": "unlisted",
                "manual_annotation_source": "",
            }
        enriched_manifest.append({**base, **additions})

    manual_labels.sort(
        key=lambda row: (
            SOURCE_SPLIT_RANK.get(row["source_split"], 9),
            row["image_id"],
            PROCESSED_SPLIT_RANK.get(row["processed_split"], 9),
        )
    )
    return MatchResult(
        manual_labels=manual_labels,
        enriched_manifest=enriched_manifest,
        unmatched=unmatched,
        unlisted=unlisted,
        matched_count=len(manual_labels),
    )


def write_diagnostics(
    duplicates: Sequence[Dict[str, str]],
    conflicts: Sequence[Dict[str, str]],
    cross_split: Sequence[Dict[str, str]],
    unmatched: Sequence[Dict[str, str]],
    unlisted: Sequence[Dict[str, str]],
    diagnostic_paths: Optional[Dict[str, Path]] = None,
) -> None:
    paths = diagnostic_paths or DIAGNOSTIC_PATHS
    write_csv_rows(paths["duplicates"], DUPLICATE_COLUMNS, duplicates)
    write_csv_rows(paths["conflicts"], CONFLICT_COLUMNS, conflicts)
    write_csv_rows(paths["cross_split"], CROSS_SPLIT_COLUMNS, cross_split)
    write_csv_rows(paths["unmatched"], UNMATCHED_COLUMNS, unmatched)
    write_csv_rows(paths["unlisted"], UNLISTED_COLUMNS, unlisted)


def count_rows(rows: Iterable[Dict[str, str]], *columns: str) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        key = tuple(row.get(column, "") for column in columns)
        counter[key if len(key) > 1 else key[0]] += 1
    return counter


def format_counter(counter: Counter) -> List[str]:
    lines: List[str] = []
    for key, value in sorted(counter.items(), key=lambda item: str(item[0])):
        if isinstance(key, tuple):
            label = " / ".join(str(part) for part in key)
        else:
            label = str(key)
        lines.append(f"- {label}: {value}")
    if not lines:
        lines.append("- none")
    return lines


def inspect_raw_zip(raw_path: Path) -> ZipMetadata:
    metadata = ZipMetadata()
    if not raw_path.exists():
        metadata.error = f"Raw ZIP does not exist: {raw_path}"
        return metadata

    try:
        with zipfile.ZipFile(raw_path) as zf:
            names = zf.namelist()
            metadata.entry_count = len(names)
            image_by_split: Dict[str, List[str]] = defaultdict(list)
            extensions = set()
            for name in names:
                path = Path(name)
                suffix = path.suffix.lower()
                if suffix in IMAGE_EXTENSIONS:
                    extensions.add(suffix)
                    if "training-data" in path.parts:
                        image_by_split["training-data"].append(name)
                    elif "testing-data" in path.parts:
                        image_by_split["testing-data"].append(name)

            metadata.image_extensions = sorted(extensions)
            metadata.split_image_counts = {split: len(paths) for split, paths in sorted(image_by_split.items())}
            for split, paths in image_by_split.items():
                basename_counts = Counter(Path(path).name for path in paths)
                metadata.duplicate_basenames[split] = {
                    basename: count for basename, count in sorted(basename_counts.items()) if count > 1
                }

            train_bases = {Path(path).name for path in image_by_split.get("training-data", [])}
            test_bases = {Path(path).name for path in image_by_split.get("testing-data", [])}
            metadata.basename_overlap = sorted(train_bases & test_bases)
            metadata.metadata_files = [
                name for name in names if Path(name).suffix.lower() in {".json", ".yaml", ".yml"}
            ]

            for name in metadata.metadata_files:
                if Path(name).suffix.lower() != ".json":
                    continue
                info = zf.getinfo(name)
                if info.file_size > 2_000_000:
                    continue
                data = json.loads(zf.read(name).decode("utf-8"))
                categories = data.get("categories", []) if isinstance(data, dict) else []
                images = data.get("images", []) if isinstance(data, dict) else []
                annotations = data.get("annotations", []) if isinstance(data, dict) else []
                category_ids = [category.get("id") for category in categories]
                category_names = [category.get("name") for category in categories]

                label_dir = str(Path(name).parent / "labels")
                label_ids = set()
                for label_name in names:
                    if not label_name.startswith(label_dir + "/") or not label_name.endswith(".txt"):
                        continue
                    label_text = zf.read(label_name).decode("utf-8").strip()
                    for line in label_text.splitlines():
                        stripped = line.strip()
                        if stripped:
                            label_ids.add(int(stripped.split()[0]))

                metadata.json_reports.append(
                    {
                        "path": name,
                        "image_count": len(images),
                        "annotation_count": len(annotations),
                        "category_ids": category_ids,
                        "category_names": category_names,
                        "yolo_label_ids": sorted(label_ids),
                        "category_ids_match_yolo_label_ids": sorted(category_ids) == sorted(label_ids),
                    }
                )
    except Exception as exc:  # pragma: no cover - defensive audit output
        metadata.error = str(exc)
    return metadata


def parse_yaml_names(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    names: List[str] = []
    in_names = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            rest = stripped[len("names:") :].strip()
            if rest.startswith("[") and rest.endswith("]"):
                return [item.strip().strip("'\"") for item in rest[1:-1].split(",") if item.strip()]
            continue
        if in_names:
            if stripped.startswith("- "):
                names.append(stripped[2:].strip().strip("'\""))
            elif stripped and not line.startswith(" "):
                break
    return names


def write_class_mapping_audit(
    path: Path,
    raw_metadata: ZipMetadata,
    data_trainval_yaml: Path,
    data_test_yaml: Path,
) -> None:
    trainval_names = parse_yaml_names(data_trainval_yaml)
    test_names = parse_yaml_names(data_test_yaml)
    yaml_names = trainval_names or test_names
    placeholder = bool(yaml_names) and all(re.fullmatch(r"class_\d+", name) for name in yaml_names)

    lines = [
        "# Class Mapping Audit",
        "",
        "## Inputs",
        "",
        f"- raw_zip_entries: {raw_metadata.entry_count}",
        f"- data_trainval_yaml: {data_trainval_yaml}",
        f"- data_test_yaml: {data_test_yaml}",
        "",
        "## Raw Annotation Metadata",
        "",
    ]
    if raw_metadata.error:
        lines.append(f"- error: {raw_metadata.error}")
    if not raw_metadata.json_reports:
        lines.append("- none")
    for report in raw_metadata.json_reports:
        lines.extend(
            [
                f"### {report['path']}",
                "",
                f"- image_count: {report['image_count']}",
                f"- annotation_count: {report['annotation_count']}",
                f"- category_ids: {report['category_ids']}",
                f"- category_names: {report['category_names']}",
                f"- yolo_label_ids: {report['yolo_label_ids']}",
                f"- category_ids_match_yolo_label_ids: {report['category_ids_match_yolo_label_ids']}",
                "",
            ]
        )
        for category_id, category_name in zip(report["category_ids"], report["category_names"]):
            lines.append(f"- {category_id} -> {category_name}")
        lines.append("")

    lines.extend(
        [
            "## YAML Names",
            "",
            f"- data_trainval.yaml names: {trainval_names}",
            f"- data_test.yaml names: {test_names}",
            f"- yaml_names_are_placeholders: {placeholder}",
            "",
            "## Conclusion",
            "",
        ]
    )
    if placeholder:
        lines.append(
            "- Fixing YAML/class-name generation is required before detector evaluation because raw metadata maps IDs to object names but YAML currently uses placeholders."
        )
    else:
        lines.append("- YAML class names are not placeholders.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def expected_total(expected_counts: Optional[Dict[str, Dict[str, int]]]) -> Optional[int]:
    if expected_counts is None:
        return None
    return sum(sum(categories.values()) for categories in expected_counts.values())


def evaluate_invariants(
    analysis: ManualAnalysis,
    match: MatchResult,
    manifest_rows: Sequence[Dict[str, str]],
    expected_counts: Optional[Dict[str, Dict[str, int]]] = DEFAULT_EXPECTED_COUNTS,
    specific_validations: Optional[Dict[str, Tuple[str, str]]] = SPECIFIC_VALIDATIONS,
) -> List[str]:
    failures: List[str] = []
    total_expected = expected_total(expected_counts)
    if total_expected is not None and len(analysis.records) != total_expected:
        failures.append(f"parsed record count {len(analysis.records)} != expected {total_expected}")
    if total_expected is not None and analysis.unique_pair_count != total_expected:
        failures.append(f"unique source-split/image-ID pair count {analysis.unique_pair_count} != expected {total_expected}")
    if analysis.duplicates:
        failures.append(f"exact duplicate count {len(analysis.duplicates)} != 0")
    if analysis.conflicts:
        failures.append(f"conflict count {len(analysis.conflicts)} != 0")
    if analysis.cross_split:
        failures.append(f"cross-split overlap count {len(analysis.cross_split)} != 0")
    if match.unmatched:
        failures.append(f"unmatched annotation count {len(match.unmatched)} != 0")
    if match.unlisted:
        failures.append(f"unlisted manifest count {len(match.unlisted)} != 0")
    if total_expected is not None and match.matched_count != total_expected:
        failures.append(f"matched count {match.matched_count} != expected {total_expected}")
    if len(match.enriched_manifest) != len(manifest_rows):
        failures.append(f"enriched manifest rows {len(match.enriched_manifest)} != manifest rows {len(manifest_rows)}")
    if total_expected is not None and len(match.enriched_manifest) != total_expected:
        failures.append(f"enriched manifest rows {len(match.enriched_manifest)} != expected {total_expected}")

    if expected_counts:
        by_source_category = Counter((row["source_split"], row["manual_degradation"]) for row in match.manual_labels)
        for source_split, categories in expected_counts.items():
            for category, expected in categories.items():
                actual = by_source_category[(source_split, category)]
                if actual != expected:
                    failures.append(f"{source_split} / {category} count {actual} != expected {expected}")

    for image_id, (expected_source, expected_category) in (specific_validations or {}).items():
        matches = [
            record
            for record in analysis.records
            if record.image_id == image_id
            and record.source_split == expected_source
            and record.manual_degradation == expected_category
        ]
        testing_matches = [record for record in analysis.records if record.image_id == image_id and record.source_split == "testing-data"]
        if len(matches) != 1:
            failures.append(
                f"specific ID {image_id} expected once as {expected_source}/{expected_category}, got {len(matches)}"
            )
        if testing_matches:
            failures.append(f"specific ID {image_id} appears in testing-data")
    return failures


def specific_validation_lines(
    records: Sequence[ManualRecord],
    specific_validations: Optional[Dict[str, Tuple[str, str]]] = SPECIFIC_VALIDATIONS,
) -> List[str]:
    lines: List[str] = []
    for image_id, (expected_source, expected_category) in (specific_validations or {}).items():
        expected_matches = [
            record
            for record in records
            if record.image_id == image_id
            and record.source_split == expected_source
            and record.manual_degradation == expected_category
        ]
        testing_matches = [record for record in records if record.image_id == image_id and record.source_split == "testing-data"]
        status = "PASS" if len(expected_matches) == 1 and not testing_matches else "FAIL"
        lines.append(
            f"- {image_id}: {status}; expected {expected_source}/{expected_category}; "
            f"matching_records={len(expected_matches)}; testing_data_records={len(testing_matches)}"
        )
    if not lines:
        lines.append("- none")
    return lines


def write_manual_audit(
    path: Path,
    annotations_path: Path,
    manifest_path: Path,
    raw_path: Path,
    scores_path: Optional[Path],
    analysis: ManualAnalysis,
    match: MatchResult,
    manifest_rows: Sequence[Dict[str, str]],
    raw_metadata: ZipMetadata,
    failures: Sequence[str],
    specific_validations: Optional[Dict[str, Tuple[str, str]]] = SPECIFIC_VALIDATIONS,
) -> None:
    source_counts = count_rows(
        (
            {
                "source_split": record.source_split,
                "manual_degradation": record.manual_degradation,
                "raw_category": record.raw_category,
            }
            for record in analysis.records
        ),
        "source_split",
    )
    raw_category_counts = count_rows(
        ({"source_split": record.source_split, "raw_category": record.raw_category} for record in analysis.records),
        "source_split",
        "raw_category",
    )
    normalized_counts = count_rows(
        (
            {"source_split": record.source_split, "manual_degradation": record.manual_degradation}
            for record in analysis.records
        ),
        "source_split",
        "manual_degradation",
    )
    processed_counts = count_rows(match.enriched_manifest, "split")
    by_processed_degradation = count_rows(match.enriched_manifest, "split", "manual_degradation")
    include_counts = count_rows(match.manual_labels, "include_primary_eval")

    training_covered = sum(1 for row in match.manual_labels if row["source_split"] == "training-data")
    testing_covered = sum(1 for row in match.manual_labels if row["source_split"] == "testing-data")
    enriched_exact = len(match.enriched_manifest) == len(manifest_rows)
    conclusion = "PASS" if not failures else "FAIL"

    lines = [
        "# Manual Degradation Audit",
        "",
        "## Inputs",
        "",
        f"- annotations: {annotations_path}",
        f"- manifest: {manifest_path}",
        f"- raw: {raw_path}",
        f"- scores: {scores_path if scores_path else ''}",
        "",
        "## Summary",
        "",
        f"- parsed_record_count: {len(analysis.records)}",
        f"- unique_source_split_image_id_pair_count: {analysis.unique_pair_count}",
        f"- exact_duplicate_count: {len(analysis.duplicates)}",
        f"- conflict_count: {len(analysis.conflicts)}",
        f"- cross_split_overlap_count: {len(analysis.cross_split)}",
        f"- matched_count: {match.matched_count}",
        f"- unmatched_count: {len(match.unmatched)}",
        f"- unlisted_manifest_count: {len(match.unlisted)}",
        "",
        "## Source Split Counts",
        "",
        *format_counter(source_counts),
        "",
        "## Processed Split Counts",
        "",
        *format_counter(processed_counts),
        "",
        "## Raw Category Counts",
        "",
        *format_counter(raw_category_counts),
        "",
        "## Normalized Category Counts",
        "",
        *format_counter(normalized_counts),
        "",
        "## Train Core Counts By Manual Degradation",
        "",
        *format_counter(Counter({key[1]: value for key, value in by_processed_degradation.items() if key[0] == "train_core"})),
        "",
        "## Val From Train Counts By Manual Degradation",
        "",
        *format_counter(
            Counter({key[1]: value for key, value in by_processed_degradation.items() if key[0] == "val_from_train"})
        ),
        "",
        "## Test Counts By Manual Degradation",
        "",
        *format_counter(Counter({key[1]: value for key, value in by_processed_degradation.items() if key[0] == "test"})),
        "",
        "## Include Primary Eval Counts",
        "",
        *format_counter(include_counts),
        "",
        "## Specific Validations",
        "",
        *specific_validation_lines(analysis.records, specific_validations=specific_validations),
        "",
        "## Coverage",
        "",
        f"- all_640_original_training_images_covered: {training_covered == 640}",
        f"- all_160_original_test_images_covered: {testing_covered == 160}",
        f"- training_covered_count: {training_covered}",
        f"- testing_covered_count: {testing_covered}",
        f"- enriched_manifest_has_exactly_800_rows: {len(match.enriched_manifest) == 800}",
        f"- enriched_manifest_row_count: {len(match.enriched_manifest)}",
        f"- enriched_manifest_matches_input_manifest_row_count: {enriched_exact}",
        "",
        "## Raw ZIP",
        "",
        f"- zip_entry_count: {raw_metadata.entry_count}",
        f"- image_extensions: {raw_metadata.image_extensions}",
        f"- split_image_counts: {raw_metadata.split_image_counts}",
        f"- basename_overlap_count: {len(raw_metadata.basename_overlap)}",
        "",
        "## Strict Invariants",
        "",
    ]
    if failures:
        lines.extend(f"- FAIL: {failure}" for failure in failures)
    else:
        lines.append("- all expected invariants satisfied")
    lines.extend(["", "## Conclusion", "", conclusion, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def stem_from_image_filename(filename: str) -> str:
    return Path(filename).stem


def float_values(rows: Sequence[Dict[str, str]], column: str) -> List[float]:
    values = []
    for row in rows:
        try:
            values.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def stats_for(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "max": ordered[-1],
    }


def format_stats(stats: Optional[Dict[str, float]]) -> str:
    if stats is None:
        return "n/a"
    return (
        f"min={stats['min']:.6g}, mean={stats['mean']:.6g}, "
        f"median={stats['median']:.6g}, max={stats['max']:.6g}"
    )


def write_score_alignment_audit(
    scores_path: Path,
    labels_path: Path,
    score_out: Path,
    report_path: Path,
) -> Dict[str, int]:
    scores = read_csv_rows(scores_path) if scores_path.exists() else []
    labels = read_csv_rows(labels_path) if labels_path.exists() else []
    labels_by_key: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for label in labels:
        labels_by_key[(stem_from_image_filename(label.get("image_filename", "")), label.get("processed_split", ""))].append(label)

    score_keys = set()
    joined: List[Dict[str, str]] = []
    unmatched_scores: List[Dict[str, str]] = []
    score_columns = list(scores[0].keys()) if scores else []
    label_extra_columns = [column for column in MANUAL_LABEL_COLUMNS if column not in {"image_filename", "processed_split"}]

    for score in scores:
        key = (Path(score.get("image_path", "")).stem, score.get("split", ""))
        score_keys.add(key)
        matches = labels_by_key.get(key, [])
        if len(matches) != 1:
            unmatched_scores.append(score)
            joined.append({**score, **{column: "" for column in label_extra_columns}})
            continue
        label = matches[0]
        joined.append({**score, **{column: label.get(column, "") for column in label_extra_columns}})

    unmatched_manual = [
        label
        for key, key_labels in labels_by_key.items()
        for label in key_labels
        if key not in score_keys or len(key_labels) != 1
    ]

    write_csv_rows(score_out, score_columns + label_extra_columns, joined)

    metric_columns = [
        column
        for column in ["blur_score", "overexposure_score", "exposure_score"]
        if score_columns and column in score_columns
    ]
    lines = [
        "# Manual Degradation Score Alignment Audit",
        "",
        "## Summary",
        "",
        f"- joined_row_count: {len(joined)}",
        f"- unmatched_score_rows: {len(unmatched_scores)}",
        f"- unmatched_manual_label_rows: {len(unmatched_manual)}",
        "",
        "## Statistics By Manual Category",
        "",
    ]
    categories = sorted({row.get("manual_degradation", "") for row in joined if row.get("manual_degradation")})
    for category in categories:
        category_rows = [row for row in joined if row.get("manual_degradation") == category]
        lines.extend([f"### {category}", ""])
        for metric in metric_columns:
            lines.append(f"- {metric}: {format_stats(stats_for(float_values(category_rows, metric)))}")
        lines.append("")

    lines.extend(["## Statistics By Split And Manual Category", ""])
    splits = sorted({row.get("split", "") for row in joined})
    for split in splits:
        for category in categories:
            group_rows = [
                row for row in joined if row.get("split") == split and row.get("manual_degradation") == category
            ]
            if not group_rows:
                continue
            lines.extend([f"### {split} / {category}", ""])
            for metric in metric_columns:
                lines.append(f"- {metric}: {format_stats(stats_for(float_values(group_rows, metric)))}")
            lines.append("")

    def append_ranked(title: str, rows: List[Dict[str, str]], metric: str, reverse: bool) -> None:
        lines.extend([f"## {title}", ""])
        if metric not in metric_columns:
            lines.append("- metric not available")
            lines.append("")
            return
        ranked = sorted(rows, key=lambda row: float(row.get(metric, "nan")), reverse=reverse)[:20]
        if not ranked:
            lines.append("- none")
        for row in ranked:
            lines.append(
                f"- {Path(row.get('image_path', '')).name} | split={row.get('split', '')} | "
                f"manual_degradation={row.get('manual_degradation', '')} | {metric}={row.get(metric, '')}"
            )
        lines.append("")

    append_ranked(
        "20 Pure-Blur Images With Lowest Blur Score",
        [row for row in joined if row.get("manual_degradation") == "blur"],
        "blur_score",
        False,
    )
    append_ranked(
        "20 Non-Blur Images With Highest Blur Score",
        [row for row in joined if row.get("manual_degradation") != "blur"],
        "blur_score",
        True,
    )
    append_ranked(
        "20 Pure-Overexposure Images With Lowest Overexposure Score",
        [row for row in joined if row.get("manual_degradation") == "overexposure"],
        "overexposure_score",
        False,
    )
    append_ranked(
        "20 Non-Overexposure Images With Highest Overexposure Score",
        [row for row in joined if row.get("manual_degradation") != "overexposure"],
        "overexposure_score",
        True,
    )

    lines.extend(
        [
            "## Blur Overexposure",
            "",
        ]
    )
    combined_rows = [row for row in joined if row.get("manual_degradation") == "blur_overexposure"]
    for metric in metric_columns:
        lines.append(f"- {metric}: {format_stats(stats_for(float_values(combined_rows, metric)))}")
    lines.extend(
        [
            "",
            "## Note",
            "",
            "- This audit reports score distributions and overlap only. No score threshold was selected or applied, and no image was classified from scores.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    return {
        "joined": len(joined),
        "unmatched_scores": len(unmatched_scores),
        "unmatched_manual": len(unmatched_manual),
    }


def run_import(
    annotations_path: Path,
    manifest_path: Path,
    raw_path: Path,
    scores_path: Optional[Path],
    labels_out: Path,
    manifest_out: Path,
    report_path: Path,
    score_report_path: Optional[Path],
    class_report_path: Path = Path("artifacts/class_mapping_audit.md"),
    strict: bool = False,
    expected_counts: Optional[Dict[str, Dict[str, int]]] = DEFAULT_EXPECTED_COUNTS,
    specific_validations: Optional[Dict[str, Tuple[str, str]]] = SPECIFIC_VALIDATIONS,
    diagnostic_paths: Optional[Dict[str, Path]] = None,
) -> int:
    raw_metadata = inspect_raw_zip(raw_path)
    write_class_mapping_audit(
        class_report_path,
        raw_metadata,
        manifest_path.parent / "data_trainval.yaml",
        manifest_path.parent / "data_test.yaml",
    )

    manifest_rows = read_csv_rows(manifest_path) if manifest_path.exists() else []
    manifest_columns = list(manifest_rows[0].keys()) if manifest_rows else []
    failures: List[str] = []
    if not annotations_path.exists():
        records: List[ManualRecord] = []
        failures.append(f"annotation file does not exist: {annotations_path}")
    else:
        records = parse_manual_file(annotations_path)

    analysis = analyze_manual_records(records)
    match = match_manual_to_manifest(analysis, manifest_rows)

    write_csv_rows(labels_out, MANUAL_LABEL_COLUMNS, match.manual_labels)
    write_csv_rows(manifest_out, manifest_columns + ENRICHED_COLUMNS, match.enriched_manifest)
    write_diagnostics(
        analysis.duplicates,
        analysis.conflicts,
        analysis.cross_split,
        match.unmatched,
        match.unlisted,
        diagnostic_paths=diagnostic_paths,
    )

    failures.extend(
        evaluate_invariants(
            analysis,
            match,
            manifest_rows,
            expected_counts=expected_counts,
            specific_validations=specific_validations,
        )
    )
    write_manual_audit(
        report_path,
        annotations_path,
        manifest_path,
        raw_path,
        scores_path,
        analysis,
        match,
        manifest_rows,
        raw_metadata,
        failures,
        specific_validations=specific_validations,
    )

    if scores_path and score_report_path:
        write_score_alignment_audit(
            scores_path=scores_path,
            labels_path=labels_out,
            score_out=scores_path.parent / "degradation_scores_with_manual_labels.csv",
            report_path=score_report_path,
        )

    return 1 if strict and failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import manual AgentCam degradation metadata.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--labels-out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--score-report", type=Path)
    parser.add_argument("--class-report", type=Path, default=Path("artifacts/class_mapping_audit.md"))
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run_import(
        annotations_path=args.annotations,
        manifest_path=args.manifest,
        raw_path=args.raw,
        scores_path=args.scores,
        labels_out=args.labels_out,
        manifest_out=args.manifest_out,
        report_path=args.report,
        score_report_path=args.score_report,
        class_report_path=args.class_report,
        strict=args.strict,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
