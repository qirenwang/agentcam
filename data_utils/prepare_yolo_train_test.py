from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from data_utils.manifest import (
    ManifestRow,
    class_names_from_labels,
    count_by,
    image_size,
    infer_degradation_type,
    infer_original_split,
    infer_sequence_frame_id,
    is_empty_label_file,
    is_image_file,
    safe_processed_name,
    symlink_or_copy,
    write_manifest,
    write_text_file,
    write_yaml,
    yolo_class_ids_from_labels,
)


IMAGE_DIR_NAMES = {"images", "image", "imgs"}
LABEL_DIR_NAMES = {"labels", "label", "anns", "annotations"}


@dataclass(frozen=True)
class CategoryMetadata:
    annotation_paths: Dict[str, str]
    split_mappings: Dict[str, Dict[int, str]]
    id_to_name: Dict[int, str]


def resolve_raw_root(raw: Path, out: Path, dry_run: bool = False) -> Optional[Path]:
    if raw.is_dir():
        return raw
    if not raw.exists():
        raise FileNotFoundError(f"Raw dataset path does not exist: {raw}")
    if raw.suffix.lower() != ".zip":
        raise ValueError(f"Expected a directory or .zip archive, got: {raw}")

    extract_root = out / "_raw_extracted" / raw.stem
    marker = extract_root / ".agentcam_extracted"
    if dry_run and not marker.exists():
        print(f"Dry run: would extract {raw} to {extract_root}")
        return None
    if marker.exists():
        return extract_root

    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw, "r") as zf:
        zf.extractall(extract_root)
    marker.write_text(raw.name + "\n")
    return extract_root


def category_mapping_from_coco_json(data: Dict[str, object], source: str) -> Dict[int, str]:
    categories = data.get("categories")
    if not categories:
        return {}
    if not isinstance(categories, list):
        raise ValueError(f"{source}: categories must be a list")

    mapping: Dict[int, str] = {}
    names = set()
    for idx, category in enumerate(categories, start=1):
        if not isinstance(category, dict):
            raise ValueError(f"{source}: category {idx} must be an object")
        category_id = category.get("id")
        if isinstance(category_id, bool) or not isinstance(category_id, int) or category_id < 0:
            raise ValueError(f"{source}: category {idx} has invalid non-negative integer id: {category_id!r}")
        name = category.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{source}: category {category_id} has invalid name: {name!r}")
        if category_id in mapping:
            raise ValueError(f"{source}: duplicate category id: {category_id}")
        if name in names:
            raise ValueError(f"{source}: duplicate category name: {name}")
        mapping[category_id] = name
        names.add(name)
    return dict(sorted(mapping.items()))


def _annotation_split(path: Path | str) -> Optional[str]:
    return infer_original_split(Path(path))


def _metadata_from_items(items: Sequence[Tuple[str, str, Dict[str, object]]]) -> Optional[CategoryMetadata]:
    split_mappings: Dict[str, Dict[int, str]] = {}
    annotation_paths: Dict[str, str] = {}
    missing_category_paths: List[str] = []

    for split, path, data in items:
        mapping = category_mapping_from_coco_json(data, path)
        if not mapping:
            missing_category_paths.append(path)
            continue
        if split in split_mappings and split_mappings[split] != mapping:
            raise ValueError(f"Conflicting category mappings within {split}: {annotation_paths[split]} and {path}")
        split_mappings[split] = mapping
        annotation_paths[split] = path

    if not split_mappings:
        return None
    if missing_category_paths:
        raise ValueError(
            "Some annotation JSON files are missing category metadata while others define categories: "
            + ", ".join(sorted(missing_category_paths))
        )

    unique_mappings = {tuple(mapping.items()) for mapping in split_mappings.values()}
    if len(unique_mappings) != 1:
        details = ", ".join(f"{split}={mapping}" for split, mapping in sorted(split_mappings.items()))
        raise ValueError(f"Category mappings disagree across source splits: {details}")

    id_to_name = dict(next(iter(unique_mappings)))
    return CategoryMetadata(
        annotation_paths=dict(sorted(annotation_paths.items())),
        split_mappings={split: dict(mapping) for split, mapping in sorted(split_mappings.items())},
        id_to_name=id_to_name,
    )


def extract_category_metadata(raw: Path) -> Optional[CategoryMetadata]:
    if raw.is_dir():
        items: List[Tuple[str, str, Dict[str, object]]] = []
        for path in sorted(raw.rglob("annotations.json")):
            split = _annotation_split(path.relative_to(raw))
            if split not in {"train", "test"}:
                continue
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"{path}: expected JSON object")
            items.append((split, str(path), data))
        return _metadata_from_items(items)

    if raw.suffix.lower() != ".zip":
        return None

    items = []
    with zipfile.ZipFile(raw, "r") as zf:
        for name in sorted(zf.namelist()):
            if Path(name).name != "annotations.json":
                continue
            split = _annotation_split(name)
            if split not in {"train", "test"}:
                continue
            data = json.loads(zf.read(name).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"{name}: expected JSON object")
            items.append((split, name, data))
    return _metadata_from_items(items)


def class_names_from_category_metadata(
    metadata: Optional[CategoryMetadata],
    label_paths: Iterable[Path],
) -> List[str]:
    if metadata is None:
        return class_names_from_labels(label_paths)

    label_class_ids = set(yolo_class_ids_from_labels(label_paths, strict=True))
    missing_ids = sorted(label_class_ids - set(metadata.id_to_name))
    if missing_ids:
        raise ValueError(f"YOLO label class IDs are not represented in raw category metadata: {missing_ids}")

    return [metadata.id_to_name[class_id] for class_id in sorted(metadata.id_to_name)]


def canonical_label_key(root: Path, path: Path, split: Optional[str]) -> str:
    rel = path.relative_to(root)
    parts = list(rel.parts)
    lowered = [p.lower() for p in parts]
    if split and split in lowered:
        parts = parts[lowered.index(split) + 1 :]
    cleaned = [p for p in parts if p.lower() not in IMAGE_DIR_NAMES | LABEL_DIR_NAMES]
    if not cleaned:
        cleaned = [path.name]
    cleaned[-1] = Path(cleaned[-1]).with_suffix("").name
    return "/".join(cleaned).lower()


def build_label_indexes(root: Path) -> Tuple[Dict[Tuple[str, str], List[Path]], Dict[Tuple[str, str], List[Path]]]:
    by_key: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    by_stem: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for label_path in sorted(root.rglob("*.txt")):
        rel = label_path.relative_to(root)
        split = infer_original_split(rel)
        if split not in {"train", "test"}:
            continue
        by_key[(split, canonical_label_key(root, label_path, split))].append(label_path)
        by_stem[(split, label_path.stem.lower())].append(label_path)
    return by_key, by_stem


def replace_image_dir_with_label_dir(root: Path, image_path: Path) -> Optional[Path]:
    rel = image_path.relative_to(root)
    parts = list(rel.parts)
    for idx, part in enumerate(parts[:-1]):
        if part.lower() in IMAGE_DIR_NAMES:
            parts[idx] = "labels"
            return (root / Path(*parts)).with_suffix(".txt")
    return None


def find_label_for_image(
    root: Path,
    image_path: Path,
    by_key: Dict[Tuple[str, str], List[Path]],
    by_stem: Dict[Tuple[str, str], List[Path]],
) -> Optional[Path]:
    rel = image_path.relative_to(root)
    split = infer_original_split(rel)
    if split is None:
        return None

    same_dir = image_path.with_suffix(".txt")
    if same_dir.exists():
        return same_dir

    mirrored = replace_image_dir_with_label_dir(root, image_path)
    if mirrored and mirrored.exists():
        return mirrored

    key_matches = by_key.get((split, canonical_label_key(root, image_path, split)), [])
    if len(key_matches) == 1:
        return key_matches[0]

    stem_matches = by_stem.get((split, image_path.stem.lower()), [])
    if len(stem_matches) == 1:
        return stem_matches[0]

    return None


def train_count(total: int, ratio: float) -> int:
    if total <= 1:
        return total
    return min(max(1, int(round(total * ratio))), total - 1)


def stratified_train_val_split(
    rows: Sequence[ManifestRow],
    train_val_ratio: float,
    seed: int,
) -> Tuple[List[ManifestRow], bool]:
    rng = random.Random(seed)
    if not rows:
        return [], False

    sequence_rows = [row for row in rows if row.sequence_id]
    unique_sequences = {row.sequence_id for row in sequence_rows}
    use_sequence_split = len(sequence_rows) == len(rows) and len(unique_sequences) >= 2

    result: List[ManifestRow] = []
    strata: Dict[str, List[ManifestRow]] = defaultdict(list)
    for row in rows:
        strata[row.degradation_type].append(row)

    for stratum_rows in strata.values():
        if use_sequence_split:
            groups: Dict[str, List[ManifestRow]] = defaultdict(list)
            for row in stratum_rows:
                groups[row.sequence_id].append(row)
            group_ids = sorted(groups)
            rng.shuffle(group_ids)
            n_train_groups = train_count(len(group_ids), train_val_ratio)
            train_groups = set(group_ids[:n_train_groups])
            for group_id in group_ids:
                split = "train_core" if group_id in train_groups else "val_from_train"
                result.extend(replace(row, split=split) for row in groups[group_id])
        else:
            shuffled = list(stratum_rows)
            rng.shuffle(shuffled)
            n_train = train_count(len(shuffled), train_val_ratio)
            for idx, row in enumerate(shuffled):
                split = "train_core" if idx < n_train else "val_from_train"
                result.append(replace(row, split=split))

    result.sort(key=lambda row: row.image_path)
    return result, use_sequence_split


def scan_dataset(root: Path, allow_unlabeled: bool, limit: int = 0) -> List[ManifestRow]:
    by_key, by_stem = build_label_indexes(root)
    image_paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and is_image_file(path) and infer_original_split(path.relative_to(root)) in {"train", "test"}
    ]
    if limit and limit > 0:
        image_paths = image_paths[:limit]

    rows: List[ManifestRow] = []
    for image_path in image_paths:
        rel = image_path.relative_to(root)
        original_split = infer_original_split(rel)
        if original_split not in {"train", "test"}:
            continue
        label_path = find_label_for_image(root, image_path, by_key, by_stem)
        has_label = label_path is not None and label_path.exists()
        if not has_label and not allow_unlabeled:
            raise ValueError(f"Missing label for {image_path}; pass --allow_unlabeled to keep it out of training/eval.")
        is_empty = bool(has_label and is_empty_label_file(label_path))
        sequence_id, frame_id = infer_sequence_frame_id(rel)
        try:
            width, height = image_size(image_path)
        except Exception:
            width, height = 0, 0
        rows.append(
            ManifestRow(
                image_path=str(image_path),
                label_path=str(label_path) if has_label else "",
                has_label=has_label,
                is_empty_label=is_empty,
                original_split=original_split,
                split=original_split,
                degradation_type=infer_degradation_type(rel),
                sequence_id=sequence_id,
                frame_id=frame_id,
                width=width,
                height=height,
            )
        )
    return rows


def assign_processed_splits(rows: Sequence[ManifestRow], train_val_ratio: float, seed: int) -> Tuple[List[ManifestRow], bool]:
    train_rows = [row for row in rows if row.original_split == "train"]
    test_rows = [replace(row, split="test") for row in rows if row.original_split == "test"]
    split_train_rows, used_sequence_split = stratified_train_val_split(train_rows, train_val_ratio, seed)
    combined = split_train_rows + test_rows
    combined.sort(key=lambda row: (row.split, row.image_path))
    return combined, used_sequence_split


def materialize_yolo_dirs(out: Path, rows: Sequence[ManifestRow]) -> None:
    for row in rows:
        if not row.has_label:
            continue
        image_src = Path(row.image_path)
        label_src = Path(row.label_path)
        image_name = safe_processed_name(image_src)
        label_name = Path(image_name).with_suffix(".txt").name
        symlink_or_copy(image_src, out / "images" / row.split / image_name)
        symlink_or_copy(label_src, out / "labels" / row.split / label_name)


def write_dataset_yaml(
    out: Path,
    rows: Sequence[ManifestRow],
    category_metadata: Optional[CategoryMetadata] = None,
) -> None:
    label_paths = [Path(row.label_path) for row in rows if row.has_label]
    names = class_names_from_category_metadata(category_metadata, label_paths)
    payload = {
        "path": str(out.resolve()),
        "train": "images/train_core",
        "val": "images/val_from_train",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    write_yaml(out / "data_trainval.yaml", payload)
    write_yaml(out / "data_test.yaml", {**payload, "train": "images/test", "val": "images/test"})


def format_counts(title: str, counts: Dict[Tuple[str, ...], int]) -> str:
    lines = [f"## {title}", ""]
    for key, value in sorted(counts.items()):
        lines.append(f"- {' / '.join(key)}: {value}")
    return "\n".join(lines)


def build_data_report(rows: Sequence[ManifestRow], used_sequence_split: bool, train_val_ratio: float) -> str:
    labeled_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        label_state = "labeled_empty" if row.is_empty_label else "labeled_positive_or_mixed" if row.has_label else "unlabeled"
        labeled_counts[(row.split, row.degradation_type, label_state)] += 1

    limitation = (
        "Sequence-level split was used because every original train image had an inferred sequence_id."
        if used_sequence_split
        else "Sequence_id could not be inferred for all original train images; image-level stratified split was used."
    )
    sections = [
        "# AgentCam Data Report",
        "",
        f"- train_val_ratio: {train_val_ratio}",
        f"- split_leakage_policy: {limitation}",
        "- label_policy: empty label files are labeled negative frames; missing label files are unlabeled and excluded from supervised training/evaluation.",
        "",
        format_counts("Counts by original split", count_by(rows, "original_split")),
        "",
        format_counts("Counts by processed split", count_by(rows, "split")),
        "",
        format_counts("Counts by split and degradation", count_by(rows, "split", "degradation_type")),
        "",
        "## Counts by split, degradation, and label state",
        "",
    ]
    for key, value in sorted(labeled_counts.items()):
        sections.append(f"- {' / '.join(key)}: {value}")
    sections.append("")
    return "\n".join(sections)


def prepare(
    raw: Path,
    out: Path,
    train_val_ratio: float,
    allow_unlabeled: bool,
    seed: int,
    limit: int = 0,
    dry_run: bool = False,
    report_path: Path = Path("artifacts/data_report.md"),
) -> List[ManifestRow]:
    category_metadata = extract_category_metadata(raw)
    raw_root = resolve_raw_root(raw, out, dry_run=dry_run)
    if raw_root is None:
        return []
    rows = scan_dataset(raw_root, allow_unlabeled=allow_unlabeled, limit=limit)
    rows, used_sequence_split = assign_processed_splits(rows, train_val_ratio=train_val_ratio, seed=seed)
    report = build_data_report(rows, used_sequence_split, train_val_ratio)

    if dry_run:
        print(report)
        return rows

    out.mkdir(parents=True, exist_ok=True)
    materialize_yolo_dirs(out, rows)
    write_manifest(out / "manifest.csv", rows)
    write_dataset_yaml(out, rows, category_metadata=category_metadata)
    write_text_file(report_path, report)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare train/test-only YOLO data for AgentCam V1.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train_val_ratio", type=float, default=0.85)
    parser.add_argument("--allow_unlabeled", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", "--dry-run", action="store_true", dest="dry_run")
    parser.add_argument("--report", type=Path, default=Path("artifacts/data_report.md"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    prepare(
        raw=args.raw,
        out=args.out,
        train_val_ratio=args.train_val_ratio,
        allow_unlabeled=args.allow_unlabeled,
        seed=args.seed,
        limit=args.limit,
        dry_run=args.dry_run,
        report_path=args.report,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
