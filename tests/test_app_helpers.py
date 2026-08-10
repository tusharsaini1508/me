"""Pure UI helper tests for clear failed-extraction messaging."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import app


class AppHelperTests(unittest.TestCase):
    def test_live_camera_component_is_disabled(self) -> None:
        self.assertIsNone(app._render_live_camera_component())
        self.assertFalse(hasattr(app, "components"))

    def test_ocr_runtime_issue_is_not_presented_as_a_bad_photo(self) -> None:
        self.assertTrue(
            app._is_ocr_runtime_issue(
                ["Local OCR is unavailable — install Tesseract OCR on this system and try again."]
            )
        )
        self.assertFalse(
            app._is_ocr_runtime_issue(["Image is too blurry — hold steady and retake it."])
        )

    def test_empty_failure_fields_remain_safe_and_reviewed(self) -> None:
        result = {
            "quality": {"pass": False, "issues": ["Image is too blurry."]},
            "fields": {
                name: {"value": None, "confidence": 0.0, "needs_review": True}
                for name in app.EXPECTED_FIELDS
            },
            "processing_ms": 1,
        }

        passed, issues = app._quality_details(result)
        rows, reviewed = app._field_rows(result)

        self.assertFalse(passed)
        self.assertTrue(issues)
        self.assertEqual(len(rows), len(app.EXPECTED_FIELDS))
        self.assertEqual(reviewed, list(app.EXPECTED_FIELDS))

    def test_ocr_runtime_issues_are_treated_as_actionable_setup_errors(self) -> None:
        result = {
            "quality": {"pass": False, "issues": ["Local OCR is unavailable — install Tesseract OCR on this system and try again."]},
            "fields": {},
            "processing_ms": 1,
        }

        self.assertTrue(app._should_accept_capture(result))

    def test_saved_scan_payload_contains_metadata_and_json_safe_content(self) -> None:
        payload = app._build_saved_scan_payload(
            source_name="receipt.png",
            fingerprint="abc123",
            result={"fields": {"merchant_name": {"value": "Cafe", "confidence": 0.91}}},
            preview=None,
            preview_error=None,
        )

        self.assertEqual(payload["source_name"], "receipt.png")
        self.assertEqual(payload["fingerprint"], "abc123")
        self.assertEqual(payload["result"]["fields"]["merchant_name"]["value"], "Cafe")
        self.assertIn("saved_at", payload)
        self.assertIsNone(payload["preview_error"])

    def test_persist_saved_scan_writes_json_to_disk(self) -> None:
        payload = app._build_saved_scan_payload(
            source_name="receipt.png",
            fingerprint="def456",
            result={"fields": {"total_amount": {"value": "12.34", "confidence": 0.87}}},
            preview=None,
            preview_error=None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_path = app._persist_saved_scan(payload, base_dir=Path(temp_dir))
            self.assertTrue(saved_path.exists())
            self.assertTrue(saved_path.suffix == ".json")
            loaded = json.loads(saved_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["fingerprint"], "def456")
            self.assertEqual(loaded["result"]["fields"]["total_amount"]["value"], "12.34")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

