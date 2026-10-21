from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from vlm.action_bank import (
    ACTION_BY_CANONICAL_CLASS,
    DEPRECATED_C1_ACTIONS,
    action_for_canonical_class,
    action_from_vlm_record,
    canonicalize_vlm_degradation,
)
from workers.enhancement import (
    SUPPORTED_ACTIONS,
    apply_enhancement_action,
    no_enhancement,
    operation_order_for_action,
)


def record(
    blur: bool = False,
    overexposure: bool = False,
    underexposure: bool = False,
    clear: bool = False,
    unknown_degradation: bool = False,
    manual_degradation_reference: str = "blur",
) -> dict[str, object]:
    return {
        "blur": blur,
        "overexposure": overexposure,
        "underexposure": underexposure,
        "clear": clear,
        "unknown_degradation": unknown_degradation,
        "manual_degradation_reference": manual_degradation_reference,
    }


def sample_image() -> Image.Image:
    image = Image.new("RGB", (24, 16))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = (x * 7 % 256, y * 11 % 256, (x + y) * 5 % 256)
    return image


class C2ActionBankTests(unittest.TestCase):
    def test_single_label_canonicalization(self) -> None:
        self.assertEqual(canonicalize_vlm_degradation(record(blur=True)), "blur")
        self.assertEqual(canonicalize_vlm_degradation(record(overexposure=True)), "overexposure")
        self.assertEqual(canonicalize_vlm_degradation(record(underexposure=True)), "underexposure")

    def test_multi_label_canonicalization(self) -> None:
        self.assertEqual(canonicalize_vlm_degradation(record(blur=True, overexposure=True)), "blur_overexposure")

    def test_contradictory_outputs_are_unknown(self) -> None:
        self.assertEqual(canonicalize_vlm_degradation(record(overexposure=True, underexposure=True)), "unknown")
        self.assertEqual(canonicalize_vlm_degradation(record(blur=True, underexposure=True)), "unknown")
        self.assertEqual(canonicalize_vlm_degradation(record(blur=True, clear=True)), "unknown")
        self.assertEqual(canonicalize_vlm_degradation({"blur": "true"}), "unknown")

    def test_action_mapping(self) -> None:
        self.assertEqual(action_for_canonical_class("clear"), "no_enhancement")
        self.assertEqual(action_for_canonical_class("blur"), "blur_enhancement")
        self.assertEqual(action_for_canonical_class("overexposure"), "overexposure_enhancement")
        self.assertEqual(action_for_canonical_class("underexposure"), "underexposure_enhancement")
        self.assertEqual(action_for_canonical_class("blur_overexposure"), "overexposure_then_blur_enhancement")
        self.assertEqual(action_for_canonical_class("unknown"), "defer_no_automatic_enhancement")

    def test_manual_reference_cannot_affect_action(self) -> None:
        base = record(blur=True, manual_degradation_reference="blur")
        changed = record(blur=True, manual_degradation_reference="overexposure")
        self.assertEqual(action_from_vlm_record(base), action_from_vlm_record(changed))

    def test_no_op_pixel_identity(self) -> None:
        image = sample_image()
        output = no_enhancement(image)
        self.assertTrue(np.array_equal(np.asarray(image), np.asarray(output)))

    def test_output_dimension_preservation(self) -> None:
        image = sample_image()
        for action in (
            "no_enhancement",
            "blur_enhancement",
            "overexposure_enhancement",
            "underexposure_enhancement",
            "overexposure_then_blur_enhancement",
        ):
            with self.subTest(action=action):
                output = apply_enhancement_action(image, action)
                self.assertIsNotNone(output)
                assert output is not None
                self.assertEqual(output.size, image.size)

    def test_sequential_operation_order(self) -> None:
        log: list[str] = []
        output = apply_enhancement_action(sample_image(), "overexposure_then_blur_enhancement", log)
        self.assertIsNotNone(output)
        self.assertEqual(log, ["overexposure_enhancement", "blur_enhancement"])
        self.assertEqual(operation_order_for_action("overexposure_then_blur_enhancement"), log)

    def test_no_best_of_bank_action_remains_active(self) -> None:
        active_actions = set(ACTION_BY_CANONICAL_CLASS.values()) | set(SUPPORTED_ACTIONS)
        self.assertTrue(DEPRECATED_C1_ACTIONS.isdisjoint(active_actions))


if __name__ == "__main__":
    unittest.main()
