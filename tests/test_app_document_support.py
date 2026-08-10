import unittest
from pathlib import Path

import app


class DocumentSupportTests(unittest.TestCase):
    def test_supported_pdf_and_docx_names_are_accepted(self) -> None:
        self.assertTrue(app._is_supported_source_name("receipt.pdf"))
        self.assertTrue(app._is_supported_source_name("receipt.docx"))

    def test_legacy_doc_name_is_rejected(self) -> None:
        self.assertFalse(app._is_supported_source_name("receipt.doc"))

    def test_unsupported_extension_is_rejected(self) -> None:
        self.assertFalse(app._is_supported_source_name("receipt.txt"))


if __name__ == "__main__":
    unittest.main()
