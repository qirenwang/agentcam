from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from data_utils.import_manual_degradation import (
    analyze_manual_records,
    match_manual_to_manifest,
    parse_manual_text,
    run_import,
)


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


def manifest_row(image_id: str, processed_split: str) -> dict[str, str]:
    original_split = "test" if processed_split == "test" else "train"
    source_split = "testing-data" if processed_split == "test" else "training-data"
    return {
        "image_path": f"data/processed/agentcam/_raw/data/{source_split}/images/{image_id}.png",
        "label_path": f"data/processed/agentcam/_raw/data/{source_split}/labels/{image_id}.txt",
        "has_label": "true",
        "is_empty_label": "false",
        "original_split": original_split,
        "split": processed_split,
        "degradation_type": "unknown",
        "sequence_id": source_split,
        "frame_id": image_id,
        "width": "64",
        "height": "64",
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(path: Path) -> None:
    path.write_text("path: .\nnc: 2\nnames:\n- class_0\n- class_1\n")


def write_scores(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "image_path",
        "split",
        "blur_score",
        "overexposure_score",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def csv_row_count(path: Path) -> int:
    with path.open(newline="") as handle:
        return len(list(csv.DictReader(handle)))


def write_raw_zip(path: Path) -> None:
    categories = [{"id": 0, "name": "car"}, {"id": 1, "name": "truck"}]
    with zipfile.ZipFile(path, "w") as zf:
        for split in ["training-data", "testing-data"]:
            zf.writestr(
                f"data/{split}/annotations.json",
                json.dumps({"images": [{"id": 1}], "annotations": [], "categories": categories}),
            )
            zf.writestr(f"data/{split}/labels/100.txt", "0 0.5 0.5 0.2 0.2\n")
            zf.writestr(f"data/{split}/labels/200.txt", "1 0.5 0.5 0.2 0.2\n")
            zf.writestr(f"data/{split}/images/100.png", b"not-used")


class ManualDegradationImportTests(unittest.TestCase):
    def test_parser_accepts_line_separators_headings_aliases_and_extensions(self) -> None:
        text = "Training-Data:\nBlur:\n100.png\r\nEXPORSURE:\r101\u2028Testing Data\u2029Clear:\n200"
        records = parse_manual_text(text)
        self.assertEqual([record.image_id for record in records], ["100", "101", "200"])
        self.assertEqual([record.source_split for record in records], ["training-data", "training-data", "testing-data"])
        self.assertEqual([record.manual_degradation for record in records], ["blur", "overexposure", "clear"])
        self.assertEqual(records[0].image_filename, "100.png")
        self.assertEqual(records[1].image_filename, "101")

    def test_exact_duplicate_detection(self) -> None:
        records = parse_manual_text("training-data\nblur\n100\n100.png\n")
        analysis = analyze_manual_records(records)
        self.assertEqual(len(analysis.duplicates), 1)
        self.assertEqual(len(analysis.conflicts), 0)

    def test_conflicting_category_detection(self) -> None:
        records = parse_manual_text("training-data\nblur\n100\nexporsure\n100\n")
        analysis = analyze_manual_records(records)
        self.assertEqual(len(analysis.conflicts), 1)
        self.assertEqual(analysis.conflicts[0]["manual_degradations"], "blur;overexposure")

    def test_cross_split_duplicate_detection(self) -> None:
        records = parse_manual_text("training-data\nblur\n100\ntesting-data\nclear\n100\n")
        analysis = analyze_manual_records(records)
        self.assertEqual(len(analysis.cross_split), 1)
        self.assertEqual(analysis.cross_split[0]["image_id"], "100")

    def test_split_constrained_matching(self) -> None:
        records = parse_manual_text(
            "training-data\nblur\n100\nexporsure\n101\ntesting-data\nclear\n200\n"
        )
        analysis = analyze_manual_records(records)
        result = match_manual_to_manifest(
            analysis,
            [
                manifest_row("100", "train_core"),
                manifest_row("101", "val_from_train"),
                manifest_row("200", "test"),
            ],
        )
        self.assertEqual(result.matched_count, 3)
        self.assertEqual([row["processed_split"] for row in result.manual_labels], ["train_core", "val_from_train", "test"])

    def test_unmatched_manual_annotation(self) -> None:
        records = parse_manual_text("training-data\nblur\n999\n")
        analysis = analyze_manual_records(records)
        result = match_manual_to_manifest(analysis, [manifest_row("100", "train_core")])
        self.assertEqual(len(result.unmatched), 1)
        self.assertEqual(result.unmatched[0]["reason"], "zero_manifest_matches")

    def test_unlisted_manifest_row(self) -> None:
        records = parse_manual_text("training-data\nblur\n100\n")
        analysis = analyze_manual_records(records)
        result = match_manual_to_manifest(analysis, [manifest_row("100", "train_core"), manifest_row("101", "val_from_train")])
        self.assertEqual(len(result.unlisted), 1)
        self.assertEqual(result.unlisted[0]["image_id"], "101")

    def test_deterministic_output_ordering(self) -> None:
        records = parse_manual_text("testing-data\nclear\n200\ntraining-data\nblur\n101\n100\n")
        analysis = analyze_manual_records(records)
        result = match_manual_to_manifest(
            analysis,
            [
                manifest_row("101", "val_from_train"),
                manifest_row("200", "test"),
                manifest_row("100", "train_core"),
            ],
        )
        self.assertEqual([row["image_id"] for row in result.manual_labels], ["100", "101", "200"])

    def test_strict_mode_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.csv"
            annotations_path = root / "manual.txt"
            raw_path = root / "raw.zip"
            write_manifest(manifest_path, [manifest_row("100", "train_core")])
            write_yaml(root / "data_trainval.yaml")
            write_yaml(root / "data_test.yaml")
            write_raw_zip(raw_path)
            annotations_path.write_text("training-data\nblur\n999\n")

            code = run_import(
                annotations_path=annotations_path,
                manifest_path=manifest_path,
                raw_path=raw_path,
                scores_path=None,
                labels_out=root / "labels.csv",
                manifest_out=root / "manifest_with_manual.csv",
                report_path=root / "audit.md",
                score_report_path=None,
                class_report_path=root / "class.md",
                strict=True,
                expected_counts={"training-data": {"blur": 1}},
                specific_validations={},
                diagnostic_paths={
                    "duplicates": root / "duplicates.csv",
                    "conflicts": root / "conflicts.csv",
                    "cross_split": root / "cross_split.csv",
                    "unmatched": root / "unmatched.csv",
                    "unlisted": root / "unlisted.csv",
                },
            )
            self.assertEqual(code, 1)
            self.assertIn("FAIL", (root / "audit.md").read_text())

    def test_successful_complete_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.csv"
            annotations_path = root / "manual.txt"
            raw_path = root / "raw.zip"
            scores_path = root / "scores" / "degradation_scores.csv"
            rows = [
                manifest_row("100", "train_core"),
                manifest_row("101", "val_from_train"),
                manifest_row("200", "test"),
            ]
            write_manifest(manifest_path, rows)
            write_yaml(root / "data_trainval.yaml")
            write_yaml(root / "data_test.yaml")
            write_raw_zip(raw_path)
            write_scores(
                scores_path,
                [
                    {"image_path": rows[0]["image_path"], "split": "train_core", "blur_score": "0.1", "overexposure_score": "0.2"},
                    {"image_path": rows[1]["image_path"], "split": "val_from_train", "blur_score": "0.3", "overexposure_score": "0.4"},
                    {"image_path": rows[2]["image_path"], "split": "test", "blur_score": "0.5", "overexposure_score": "0.6"},
                ],
            )
            annotations_path.write_text("training-data\nblur\n100\nexporsure\n101\ntesting-data\nclear\n200\n")

            diagnostics = {
                "duplicates": root / "duplicates.csv",
                "conflicts": root / "conflicts.csv",
                "cross_split": root / "cross_split.csv",
                "unmatched": root / "unmatched.csv",
                "unlisted": root / "unlisted.csv",
            }
            code = run_import(
                annotations_path=annotations_path,
                manifest_path=manifest_path,
                raw_path=raw_path,
                scores_path=scores_path,
                labels_out=root / "labels.csv",
                manifest_out=root / "manifest_with_manual.csv",
                report_path=root / "audit.md",
                score_report_path=root / "score_audit.md",
                class_report_path=root / "class.md",
                strict=True,
                expected_counts={"training-data": {"blur": 1, "overexposure": 1}, "testing-data": {"clear": 1}},
                specific_validations={},
                diagnostic_paths=diagnostics,
            )
            self.assertEqual(code, 0)
            self.assertEqual(csv_row_count(root / "labels.csv"), 3)
            self.assertEqual(csv_row_count(root / "manifest_with_manual.csv"), 3)
            with (root / "manifest_with_manual.csv").open(newline="") as handle:
                fieldnames = csv.DictReader(handle).fieldnames or []
            self.assertNotIn("_manual_source_split", fieldnames)
            self.assertNotIn("_manual_image_id", fieldnames)
            for path in diagnostics.values():
                self.assertEqual(csv_row_count(path), 0)
            self.assertTrue((scores_path.parent / "degradation_scores_with_manual_labels.csv").exists())


if __name__ == "__main__":
    unittest.main()
