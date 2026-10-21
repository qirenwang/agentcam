from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from data_utils.manifest import ManifestRow, load_yaml
from data_utils.prepare_yolo_train_test import (
    CategoryMetadata,
    category_mapping_from_coco_json,
    class_names_from_category_metadata,
    extract_category_metadata,
    prepare,
    write_dataset_yaml,
)


def png_bytes(color=(20, 20, 20)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buffer, format="PNG")
    return buffer.getvalue()


def write_label(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def manifest_row(label_path: Path) -> ManifestRow:
    return ManifestRow(
        image_path="image.png",
        label_path=str(label_path),
        has_label=True,
        is_empty_label=False,
        original_split="train",
        split="train_core",
        degradation_type="unknown",
        sequence_id="",
        frame_id="",
        width=32,
        height=24,
    )


class ClassMappingTests(unittest.TestCase):
    def write_category_zip(
        self,
        path: Path,
        train_categories: list[dict[str, object]] | None = None,
        test_categories: list[dict[str, object]] | None = None,
    ) -> None:
        if train_categories is None:
            train_categories = [{"id": 0, "name": "car"}, {"id": 1, "name": "truck"}]
        if test_categories is None:
            test_categories = [{"id": 0, "name": "car"}, {"id": 1, "name": "truck"}]

        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("data/training-data/images/train_001.png", png_bytes())
            zf.writestr("data/training-data/labels/train_001.txt", "0 0.5 0.5 0.25 0.25\n")
            zf.writestr("data/testing-data/images/test_001.png", png_bytes(color=(30, 30, 30)))
            zf.writestr("data/testing-data/labels/test_001.txt", "1 0.5 0.5 0.25 0.25\n")
            zf.writestr(
                "data/training-data/annotations.json",
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "train_001.png"}],
                        "annotations": [{"id": 1, "image_id": 1, "category_id": 0}],
                        "categories": train_categories,
                    }
                ),
            )
            zf.writestr(
                "data/testing-data/annotations.json",
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "test_001.png"}],
                        "annotations": [{"id": 1, "image_id": 1, "category_id": 1}],
                        "categories": test_categories,
                    }
                ),
            )

    def test_extracts_expected_category_names_from_coco_zip_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "raw.zip"
            self.write_category_zip(zip_path)

            metadata = extract_category_metadata(zip_path)

            self.assertIsNotNone(metadata)
            assert metadata is not None
            self.assertEqual(metadata.id_to_name, {0: "car", 1: "truck"})
            self.assertEqual(metadata.split_mappings["train"], {0: "car", 1: "truck"})
            self.assertEqual(metadata.split_mappings["test"], {0: "car", 1: "truck"})
            self.assertEqual(
                metadata.annotation_paths,
                {
                    "test": "data/testing-data/annotations.json",
                    "train": "data/training-data/annotations.json",
                },
            )

    def test_category_mapping_order_is_deterministic_by_id(self) -> None:
        mapping = category_mapping_from_coco_json(
            {"categories": [{"id": 1, "name": "truck"}, {"id": 0, "name": "car"}]},
            "fixture",
        )
        self.assertEqual(list(mapping.items()), [(0, "car"), (1, "truck")])

    def test_inconsistent_train_test_category_names_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "raw.zip"
            self.write_category_zip(zip_path, test_categories=[{"id": 0, "name": "car"}, {"id": 1, "name": "van"}])

            with self.assertRaisesRegex(ValueError, "disagree across source splits"):
                extract_category_metadata(zip_path)

    def test_duplicate_category_ids_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate category id"):
            category_mapping_from_coco_json(
                {"categories": [{"id": 0, "name": "car"}, {"id": 0, "name": "truck"}]},
                "fixture",
            )

    def test_duplicate_category_names_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate category name"):
            category_mapping_from_coco_json(
                {"categories": [{"id": 0, "name": "vehicle"}, {"id": 1, "name": "vehicle"}]},
                "fixture",
            )

    def test_missing_category_metadata_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "raw.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("data/training-data/annotations.json", json.dumps({"images": [], "annotations": []}))
                zf.writestr("data/testing-data/annotations.json", json.dumps({"images": [], "annotations": []}))

            self.assertIsNone(extract_category_metadata(zip_path))

    def test_placeholder_fallback_emits_warning_when_metadata_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            label_path = tmp_path / "labels" / "image.txt"
            write_label(label_path, "0 0.5 0.5 0.25 0.25\n")

            with self.assertWarnsRegex(RuntimeWarning, "placeholder"):
                write_dataset_yaml(tmp_path / "out", [manifest_row(label_path)], category_metadata=None)

            data = load_yaml(tmp_path / "out" / "data_trainval.yaml")
            self.assertEqual(data["names"], ["class_0"])

    def test_yolo_label_id_not_represented_in_categories_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            label_path = Path(tmp) / "label.txt"
            write_label(label_path, "1 0.5 0.5 0.25 0.25\n")
            metadata = CategoryMetadata(
                annotation_paths={"train": "annotations.json"},
                split_mappings={"train": {0: "car"}},
                id_to_name={0: "car"},
            )

            with self.assertRaisesRegex(ValueError, "not represented"):
                class_names_from_category_metadata(metadata, [label_path])

    def test_prepare_zip_generates_yaml_with_real_names_and_preserves_split_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "raw.zip"
            out = tmp_path / "processed"
            self.write_category_zip(raw)

            prepare(
                raw=raw,
                out=out,
                train_val_ratio=0.5,
                allow_unlabeled=True,
                seed=7,
                report_path=tmp_path / "data_report.md",
            )

            trainval = load_yaml(out / "data_trainval.yaml")
            test = load_yaml(out / "data_test.yaml")
            self.assertEqual(trainval["path"], str(out.resolve()))
            self.assertEqual(trainval["train"], "images/train_core")
            self.assertEqual(trainval["val"], "images/val_from_train")
            self.assertEqual(trainval["test"], "images/test")
            self.assertEqual(trainval["nc"], 2)
            self.assertEqual(trainval["names"], ["car", "truck"])
            self.assertEqual(test["train"], "images/test")
            self.assertEqual(test["val"], "images/test")
            self.assertEqual(test["test"], "images/test")
            self.assertEqual(test["nc"], 2)
            self.assertEqual(test["names"], ["car", "truck"])


if __name__ == "__main__":
    unittest.main()
