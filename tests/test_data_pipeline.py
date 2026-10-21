from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from data_utils.manifest import infer_degradation_type, infer_original_split, read_manifest
from data_utils.prepare_yolo_train_test import prepare
from data_utils.validate_yolo import parse_yolo_label_line
from data_utils.yolo_to_coco import manifest_to_coco


def write_image(path: Path, color=(20, 20, 20)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


def write_label(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class DataPipelineTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> None:
        write_image(root / "train/images/blur/seqA_001.jpg")
        write_label(root / "train/labels/blur/seqA_001.txt", "0 0.5 0.5 0.25 0.25\n")

        write_image(root / "train/images/blur/seqB_001.jpg")
        write_label(root / "train/labels/blur/seqB_001.txt", "")

        write_image(root / "train/images/overexposure/seqC_001.jpg", color=(255, 255, 255))
        write_label(root / "train/labels/overexposure/seqC_001.txt", "0 0.5 0.5 0.5 0.5\n")

        write_image(root / "train/images/blur/seqD_001.jpg")

        write_image(root / "test/images/blur/test_001.jpg")
        write_label(root / "test/labels/blur/test_001.txt", "0 0.5 0.5 0.25 0.25\n")

    def test_yolo_label_parsing(self) -> None:
        record = parse_yolo_label_line("2 0.1 0.2 0.3 0.4")
        self.assertEqual(record.class_id, 2)
        with self.assertRaises(ValueError):
            parse_yolo_label_line("0 0.1 0.2 0.3")
        with self.assertRaises(ValueError):
            parse_yolo_label_line("cat 0.1 0.2 0.3 0.4")
        with self.assertRaises(ValueError):
            parse_yolo_label_line("0 0.1 0.2 1.3 0.4")

    def test_prepare_distinguishes_missing_and_empty_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "raw"
            out = tmp_path / "processed"
            self.make_dataset(raw)
            rows = prepare(
                raw=raw,
                out=out,
                train_val_ratio=0.5,
                allow_unlabeled=True,
                seed=7,
                report_path=tmp_path / "data_report.md",
            )
            manifest_rows = read_manifest(out / "manifest.csv")
            self.assertEqual(len(rows), 5)
            empty_rows = [row for row in manifest_rows if row.is_empty_label]
            missing_rows = [row for row in manifest_rows if not row.has_label]
            self.assertEqual(len(empty_rows), 1)
            self.assertEqual(len(missing_rows), 1)
            self.assertIn("train_core", {row.split for row in manifest_rows})
            self.assertIn("val_from_train", {row.split for row in manifest_rows})
            self.assertIn("test", {row.split for row in manifest_rows})
            self.assertTrue((out / "data_trainval.yaml").exists())
            self.assertTrue((out / "data_test.yaml").exists())

    def test_degradation_type_inference(self) -> None:
        self.assertEqual(infer_degradation_type("train/images/blur/frame.jpg"), "blur")
        self.assertEqual(infer_degradation_type("test/glare/frame.jpg"), "overexposure")
        self.assertEqual(infer_degradation_type("test/normal/frame.jpg"), "unknown")

    def test_original_split_inference_accepts_archive_names(self) -> None:
        self.assertEqual(infer_original_split("data/training-data/images/frame.jpg"), "train")
        self.assertEqual(infer_original_split("data/testing-data/images/frame.jpg"), "test")

    def test_yolo_to_coco_uses_labeled_frames_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "raw"
            out = tmp_path / "processed"
            self.make_dataset(raw)
            prepare(
                raw=raw,
                out=out,
                train_val_ratio=0.5,
                allow_unlabeled=True,
                seed=7,
                report_path=tmp_path / "data_report.md",
            )
            coco = manifest_to_coco(out, "train_core")
            self.assertGreaterEqual(len(coco["images"]), 1)
            self.assertTrue(all("bbox" in ann for ann in coco["annotations"]))


if __name__ == "__main__":
    unittest.main()
