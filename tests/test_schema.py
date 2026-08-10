"""Contract tests for the small result-schema helper module.

The module is optional while the scaffold is being introduced.  Once
``src.schema`` exists, these tests intentionally fail for a partial helper API
instead of silently accepting a malformed pipeline response.
"""

from __future__ import annotations

import json
import math
import unittest
from collections.abc import Mapping
from typing import Any

try:
    from src import schema
except ImportError:  # The helpers are introduced independently of this scaffold.
    schema = None  # type: ignore[assignment]


EXPECTED_FIELD_NAMES = (
    "merchant_name",
    "transaction_date",
    "total_amount",
    "currency",
)


@unittest.skipIf(schema is None, "src.schema has not been added yet")
class SchemaContractTests(unittest.TestCase):
    def test_expected_fields_are_complete_and_ordered(self) -> None:
        self.assertEqual(tuple(schema.EXPECTED_FIELDS), EXPECTED_FIELD_NAMES)

    def test_empty_fields_are_safe_reviewed_abstentions(self) -> None:
        fields = schema.empty_fields()

        self.assertEqual(tuple(fields), EXPECTED_FIELD_NAMES)
        for name in EXPECTED_FIELD_NAMES:
            self._assert_field_contract(fields[name], expected_value=None, reviewed=True)
            self.assertEqual(fields[name]["confidence"], 0.0)

    def test_make_field_keeps_native_json_safe_values(self) -> None:
        field = schema.make_field("Cafe Nine", 0.91, False)

        self._assert_field_contract(field, expected_value="Cafe Nine", reviewed=False)
        self.assertEqual(field["confidence"], 0.91)
        # This catches accidental NumPy scalar values and NaN confidences.
        self.assertIsInstance(json.dumps(field, allow_nan=False), str)

    def test_make_result_fills_a_complete_failure_contract(self) -> None:
        result = schema.make_result(
            False,
            ("Image is too blurry — hold steady and retake.",),
            processing_ms=7.25,
        )

        self._assert_result_contract(result, passed=False, issues_nonempty=True)
        for name in EXPECTED_FIELD_NAMES:
            self._assert_field_contract(result["fields"][name], expected_value=None, reviewed=True)

    def test_make_result_preserves_supplied_complete_fields(self) -> None:
        fields = schema.empty_fields()
        fields["merchant_name"] = schema.make_field("Cafe Nine", 0.91, False)
        result = schema.make_result(True, (), fields=fields, processing_ms=12.0)

        self._assert_result_contract(result, passed=True, issues_nonempty=False)
        self._assert_field_contract(
            result["fields"]["merchant_name"], expected_value="Cafe Nine", reviewed=False
        )
        self.assertEqual(result["processing_ms"], 12.0)

    def _assert_result_contract(
        self, result: Any, *, passed: bool, issues_nonempty: bool
    ) -> None:
        self.assertIsInstance(result, Mapping)
        self.assertEqual(set(result), {"quality", "fields", "processing_ms"})
        self.assertIsInstance(result["quality"], Mapping)
        self.assertIs(result["quality"].get("pass"), passed)
        self.assertIsInstance(result["quality"].get("issues"), list)
        self.assertEqual(bool(result["quality"]["issues"]), issues_nonempty)
        self.assertIsInstance(result["fields"], Mapping)
        self.assertEqual(tuple(result["fields"]), EXPECTED_FIELD_NAMES)
        self.assertIsInstance(result["processing_ms"], (int, float))
        self.assertGreaterEqual(float(result["processing_ms"]), 0.0)
        self.assertIsInstance(json.dumps(result, allow_nan=False), str)

    def _assert_field_contract(
        self, field: Any, *, expected_value: Any, reviewed: bool
    ) -> None:
        self.assertIsInstance(field, Mapping)
        self.assertEqual(set(field), {"value", "confidence", "needs_review"})
        self.assertEqual(field["value"], expected_value)
        self.assertIsInstance(field["confidence"], float)
        self.assertTrue(math.isfinite(field["confidence"]))
        self.assertGreaterEqual(field["confidence"], 0.0)
        self.assertLessEqual(field["confidence"], 1.0)
        self.assertIs(field["needs_review"], reviewed)


if __name__ == "__main__":  # pragma: no cover - convenient direct invocation
    unittest.main()
