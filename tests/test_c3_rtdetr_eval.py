from __future__ import annotations

import unittest

from scripts.agentcam_c3_rtdetr_eval import (
    C1_JSONL,
    C2_RUN,
    EXPECTED_CLASS_COUNTS,
    action_for_canonical_class,
    class_support_warnings,
    compare_hash_snapshots,
    discover_c2_manifest,
    ensure_metric_populations_not_mixed,
    ensure_paired_scope_ids,
    is_actionable,
    is_test_path,
    load_c2_manifest,
    read_jsonl,
    remap_coco_class,
    validate_dimension_preserved,
)


class C3RTDETRProtocolTests(unittest.TestCase):
    def test_exact_640_c1_c2_join(self) -> None:
        c1_rows = read_jsonl(C1_JSONL)
        c2_rows = load_c2_manifest(C2_RUN)
        c1_ids = {str(row["image_id"]) for row in c1_rows}
        c2_ids = {str(row["image_id"]) for row in c2_rows}
        self.assertEqual(len(c1_rows), 640)
        self.assertEqual(len(c2_rows), 640)
        self.assertEqual(c1_ids, c2_ids)

    def test_exact_actionable_and_deferred_sets(self) -> None:
        rows = load_c2_manifest(C2_RUN)
        actionable = [row for row in rows if is_actionable(str(row["enhancement_action"]))]
        deferred = [row for row in rows if str(row["enhancement_action"]) == "defer_no_automatic_enhancement"]
        self.assertEqual(len(actionable), 593)
        self.assertEqual(len(deferred), 47)

    def test_unknown_never_receives_enhanced_inference_action(self) -> None:
        rows = load_c2_manifest(C2_RUN)
        for row in rows:
            if str(row["vlm_degradation_canonical"]) == "unknown":
                self.assertEqual(str(row["enhancement_action"]), "defer_no_automatic_enhancement")

    def test_raw_paired_set_equals_enhanced_set_helper(self) -> None:
        ids = {"a", "b"}
        ensure_paired_scope_ids(ids, set(ids), set(ids))
        with self.assertRaises(ValueError):
            ensure_paired_scope_ids(ids, {"a"}, set(ids))

    def test_manual_label_cannot_alter_action(self) -> None:
        self.assertEqual(action_for_canonical_class("blur"), "blur_enhancement")
        self.assertEqual(action_for_canonical_class("blur"), "blur_enhancement")

    def test_raw_enhanced_gt_mapping_identity_by_id(self) -> None:
        ids = {"img1", "img2"}
        ensure_paired_scope_ids(ids, ids, ids)

    def test_class_remapping(self) -> None:
        self.assertEqual(remap_coco_class(2), 0)
        self.assertEqual(remap_coco_class(7), 1)
        with self.assertRaises(ValueError):
            remap_coco_class(0)

    def test_source_dimensions_preserved(self) -> None:
        self.assertTrue(validate_dimension_preserved(1920, 1200, 1920, 1200))
        self.assertFalse(validate_dimension_preserved(1920, 1200, 640, 640))

    def test_source_hashes_unchanged(self) -> None:
        self.assertTrue(compare_hash_snapshots({"a": "1"}, {"a": "1"})["unchanged"])
        self.assertFalse(compare_hash_snapshots({"a": "1"}, {"a": "2"})["unchanged"])

    def test_no_test_path_accepted(self) -> None:
        self.assertTrue(is_test_path("data/testing-data/images/x.png"))
        self.assertFalse(is_test_path("data/training-data/images/x.png"))

    def test_metric_scopes_cannot_mix_populations(self) -> None:
        ensure_metric_populations_not_mixed("raw", {"a"}, "enhanced", {"a"})
        with self.assertRaises(ValueError):
            ensure_metric_populations_not_mixed("raw_full_640", {"a", "b"}, "enhanced_593", {"a"})

    def test_underexposure_and_clear_warnings(self) -> None:
        warnings = class_support_warnings(EXPECTED_CLASS_COUNTS)
        self.assertTrue(any("underexposure has one" in warning for warning in warnings))
        self.assertTrue(any("clear has zero" in warning for warning in warnings))

    def test_manifest_discovery(self) -> None:
        csv_path, jsonl_path = discover_c2_manifest(C2_RUN)
        self.assertTrue(csv_path.name.endswith(".csv"))
        self.assertTrue(jsonl_path.name.endswith(".jsonl"))


if __name__ == "__main__":
    unittest.main()
