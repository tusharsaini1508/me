"""Tests for the deterministic synthetic fixture scaffold itself."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.fixtures.synthetic_receipts import DEFAULT_FIELDS, make_receipt_scene, write_fixture_set
from tests.geometry import quad_area_ratio, touches_image_edge


class SyntheticReceiptFixtureTests(unittest.TestCase):
    def test_valid_scene_is_deterministic_bgr_uint8(self) -> None:
        first = make_receipt_scene("valid")
        second = make_receipt_scene("valid")

        self.assertEqual(first.image.shape, (720, 960, 3))
        self.assertEqual(first.image.dtype, np.uint8)
        self.assertTrue(np.array_equal(first.image, second.image))
        self.assertTrue(np.array_equal(first.quad, second.quad))
        self.assertEqual(first.fields, DEFAULT_FIELDS)

    def test_scene_variants_cover_small_and_cutoff_geometry(self) -> None:
        valid = make_receipt_scene("valid")
        small = make_receipt_scene("small")
        cutoff = make_receipt_scene("cutoff")

        self.assertGreater(quad_area_ratio(valid.quad, valid.image.shape), 0.30)
        self.assertLess(quad_area_ratio(small.quad, small.image.shape), 0.08)
        self.assertFalse(touches_image_edge(valid.quad, valid.image.shape))
        self.assertTrue(touches_image_edge(cutoff.quad, cutoff.image.shape))

    def test_fixture_writer_creates_repeatable_smoke_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            written = write_fixture_set(temporary_directory)
            self.assertEqual(set(written), {"valid", "small", "cutoff", "labels"})
            for key in ("valid", "small", "cutoff", "labels"):
                self.assertTrue(written[key].is_file(), key)

            payload = json.loads(Path(written["labels"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["receipts"][0]["image"], written["valid"].name)
            self.assertEqual(payload["receipts"][0]["fields"], DEFAULT_FIELDS)


if __name__ == "__main__":  # pragma: no cover - convenient direct invocation
    unittest.main()
