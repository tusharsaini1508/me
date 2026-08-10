"""Deployment configuration checks for Streamlit Cloud compatibility."""

from __future__ import annotations

import unittest
from pathlib import Path


class DeploymentConfigTests(unittest.TestCase):
    def test_cloud_packages_include_tesseract(self) -> None:
        packages_text = Path("packages.txt").read_text(encoding="utf-8")
        self.assertIn("tesseract-ocr", packages_text)
        self.assertIn("tesseract-ocr-eng", packages_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
