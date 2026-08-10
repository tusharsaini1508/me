"""Pure UI helper tests for clear failed-extraction messaging."""

from __future__ import annotations

import unittest

import app


class AppHelperTests(unittest.TestCase):
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

