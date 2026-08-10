"""Smoke tests for the headless evaluation harness and label format."""

from __future__ import annotations

import tempfile
import unittest

import eval as evaluator
from tests.fixtures.synthetic_receipts import DEFAULT_FIELDS, write_fixture_set


class EvaluatorSmokeTests(unittest.TestCase):
    def test_generated_fixture_labels_are_scored_with_the_trust_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            write_fixture_set(temporary_directory)

            def extractor(_: str) -> dict[str, object]:
                return {
                    "quality": {"pass": True, "issues": []},
                    "fields": {
                        name: {
                            "value": value,
                            "confidence": 0.95,
                            "needs_review": False,
                        }
                        for name, value in DEFAULT_FIELDS.items()
                    },
                    "processing_ms": 1,
                }

            result = evaluator.evaluate(temporary_directory, extractor=extractor)

        self.assertEqual(result.images_discovered, 3)
        self.assertEqual(result.images_with_labels, 1)
        self.assertEqual(result.images_processed, 1)
        self.assertEqual(result.pipeline_errors, 0)
        self.assertEqual(result.overall.labelled, 4)
        self.assertEqual(result.overall.correct, 4)
        self.assertEqual(result.overall.trust_points, 4)
        self.assertEqual(result.overall.trust_score, 1.0)

    def test_reviewed_or_missing_values_receive_zero_credit_not_negative_credit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            write_fixture_set(temporary_directory)

            def extractor(_: str) -> dict[str, object]:
                return {
                    "quality": {"pass": False, "issues": ["Synthetic failure"]},
                    "fields": {
                        name: {"value": None, "confidence": 0.0, "needs_review": True}
                        for name in DEFAULT_FIELDS
                    },
                    "processing_ms": 1,
                }

            result = evaluator.evaluate(temporary_directory, extractor=extractor)

        self.assertEqual(result.overall.labelled, 4)
        self.assertEqual(result.overall.abstained, 4)
        self.assertEqual(result.overall.trust_points, 0)
        self.assertEqual(result.overall.zero_credit, 4)


if __name__ == "__main__":  # pragma: no cover - convenient direct invocation
    unittest.main()

