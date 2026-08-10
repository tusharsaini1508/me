"""Streamlit interface for the receipt capture and extraction pipeline.

The app deliberately keeps all vision and OCR work in ``src.pipeline``.  Its
only responsibilities are safely receiving an upload, presenting the pipeline
result, and ensuring the temporary source image is removed after processing.
"""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
import tempfile
from contextlib import contextmanager
from numbers import Real
from pathlib import Path
from typing import Any, Iterator, Mapping

import streamlit as st


# Import the required headless entry point separately from the optional preview
# helper.  This lets the application explain a deployment/configuration problem
# instead of failing during Streamlit's initial import.
PIPELINE_IMPORT_ERROR: Exception | None = None
RECTIFY_IMPORT_ERROR: Exception | None = None

try:
    from src.pipeline import extract as pipeline_extract
except Exception as error:  # pragma: no cover - depends on deployment setup
    pipeline_extract = None
    PIPELINE_IMPORT_ERROR = error

try:
    from src.pipeline import rectify_preview
except Exception as error:  # Rectification preview is intentionally optional.
    rectify_preview = None
    RECTIFY_IMPORT_ERROR = error


MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SUPPORTED_DOCUMENT_SUFFIXES = {".pdf", ".docx"}
EXPECTED_FIELDS = ("merchant_name", "transaction_date", "total_amount", "currency")
OCR_RUNTIME_MARKERS = (
    "local ocr is unavailable",
    "tesseract executable",
    "text recognition could not complete",
)


@contextmanager
def temporary_image(upload_bytes: bytes, original_name: str) -> Iterator[str]:
    """Persist an upload just long enough for the path-based pipeline to read it.

    ``mkstemp`` avoids filename collisions.  The ``finally`` block also removes
    the file if extraction, rectification, or image decoding raises an error.
    """

    suffix = _safe_suffix(original_name)
    descriptor, temporary_path = tempfile.mkstemp(prefix="receipt_upload_", suffix=suffix)

    try:
        with os.fdopen(descriptor, "wb") as image_file:
            image_file.write(upload_bytes)
            image_file.flush()
        yield temporary_path
    finally:
        try:
            Path(temporary_path).unlink(missing_ok=True)
        except OSError:
            # A failed cleanup should not turn a completed extraction into a UI
            # error.  The operating system will still clear its temporary area.
            pass


def _safe_suffix(original_name: str) -> str:
    """Return a known image suffix without trusting an uploaded filename."""

    suffix = Path(original_name).suffix.lower()
    if suffix in ALLOWED_SUFFIXES:
        return suffix

    guessed_suffix = mimetypes.guess_extension(
        mimetypes.guess_type(original_name)[0] or ""
    )
    if guessed_suffix and guessed_suffix.lower() in ALLOWED_SUFFIXES:
        return guessed_suffix.lower()
    return ".png"


def _json_safe(value: Any) -> Any:
    """Convert common numeric/scalar pipeline values into JSON-safe values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value

    # NumPy scalar values expose ``item`` but importing NumPy solely for the UI
    # would make the interface less portable.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except Exception:
            pass
    return str(value)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _quality_details(result: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Read quality data defensively so malformed outputs remain inspectable."""

    quality = _as_mapping(result.get("quality"))
    passed = quality.get("pass") is True
    raw_issues = quality.get("issues", [])

    if isinstance(raw_issues, str):
        issues = [raw_issues]
    elif isinstance(raw_issues, (list, tuple)):
        issues = [str(issue) for issue in raw_issues if str(issue).strip()]
    else:
        issues = []

    if not quality:
        issues.append("The pipeline did not return a quality verdict.")
    elif not passed and not issues:
        issues.append("This image did not pass the capture-quality checks.")
    return passed, issues


def _field_rows(result: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Create a predictable review table from the documented response schema."""

    fields = _as_mapping(result.get("fields"))
    ordered_names = list(EXPECTED_FIELDS)
    ordered_names.extend(name for name in fields if name not in EXPECTED_FIELDS)

    rows: list[dict[str, Any]] = []
    review_names: list[str] = []
    for name in ordered_names:
        field = _as_mapping(fields.get(name))
        value = field.get("value")
        confidence = field.get("confidence")
        is_missing_required_field = name in EXPECTED_FIELDS and name not in fields
        needs_review = field.get("needs_review") is True or is_missing_required_field

        if needs_review:
            review_names.append(name)

        if isinstance(confidence, Real) and not isinstance(confidence, bool):
            confidence_display: str | None = f"{float(confidence):.0%}"
        else:
            confidence_display = None

        rows.append(
            {
                "Field": name,
                "Value": "—" if value is None else str(value),
                "Confidence": confidence_display or "—",
                "Needs review": (
                    "Yes — missing"
                    if is_missing_required_field
                    else ("Yes" if needs_review else "No")
                ),
            }
        )
    return rows, review_names


def _is_ocr_runtime_issue(issues: list[str]) -> bool:
    """Return whether a failure is server OCR setup, not a bad photo."""

    normalized = " ".join(issues).casefold()
    return any(marker in normalized for marker in OCR_RUNTIME_MARKERS)


def _render_quality(result: Mapping[str, Any]) -> tuple[bool, list[str]]:
    passed, issues = _quality_details(result)
    if passed:
        st.success("Capture quality passed")
    elif _is_ocr_runtime_issue(issues):
        st.error("Text recognition is not configured on this machine")
        st.info(
            "Install Tesseract OCR (including English language data), then restart the app. "
            "Your photo does not need to be retaken for this setup issue."
        )
    else:
        st.error("Retake recommended — capture quality did not pass")

    if issues:
        st.caption("Quality checks")
        for issue in issues:
            st.warning(issue, icon="⚠️")
    return passed, issues


def _render_rectified_preview(preview: Any, preview_error: str | None) -> None:
    st.subheader("Rectified receipt")
    if preview is None:
        st.info("No rectified receipt crop was available for this image.")
    else:
        # The pipeline standardizes OpenCV-style colour images as BGR.  Leaving
        # grayscale images alone avoids passing an invalid channel setting.
        shape = getattr(preview, "shape", ())
        try:
            if len(shape) == 3 and shape[-1] in (3, 4):
                st.image(preview, channels="BGR", caption="Deskewed receipt crop")
            else:
                st.image(preview, caption="Deskewed receipt crop")
        except Exception:
            st.info("A rectified crop was produced, but it could not be displayed in the browser.")

    if preview_error:
        st.caption("The extraction completed, but the optional preview could not be generated.")


def _render_result(result: Any, preview: Any, preview_error: str | None) -> None:
    if not isinstance(result, Mapping):
        st.error("The extractor returned an unexpected result format.")
        st.code(json.dumps(_json_safe(result), indent=2, ensure_ascii=False), language="json")
        return

    result_mapping = _as_mapping(result)
    quality_passed, _ = _render_quality(result_mapping)

    processing_ms = result_mapping.get("processing_ms")
    if isinstance(processing_ms, Real) and not isinstance(processing_ms, bool):
        st.caption(f"Processing time: {float(processing_ms):,.0f} ms")

    _render_rectified_preview(preview, preview_error)

    st.subheader("Extracted fields")
    if not quality_passed:
        st.info(
            "No fields were extracted because analysis stopped before OCR. "
            "Resolve the issue above, then analyze the image again."
        )
    else:
        rows, review_names = _field_rows(result_mapping)
        st.dataframe(rows, hide_index=True, use_container_width=True)
        if review_names:
            st.warning("Review required: " + ", ".join(f"`{name}`" for name in review_names))
        else:
            st.success("No extracted fields were flagged for review.")

    st.subheader("Extraction JSON")
    display_result = _json_safe(result_mapping)
    json_text = json.dumps(display_result, indent=2, ensure_ascii=False, allow_nan=False)
    st.code(json_text, language="json")
    st.download_button(
        "Download JSON",
        data=json_text,
        file_name="receipt-extraction.json",
        mime="application/json",
        use_container_width=False,
    )


def _should_accept_capture(result: Any) -> bool:
    """Only accept a capture when the pipeline quality gate passes."""

    if not isinstance(result, Mapping):
        return False
    passed, issues = _quality_details(result)
    return passed and not issues


def _process_upload(upload_bytes: bytes, original_name: str) -> tuple[Any, Any, str | None]:
    """Run the headless pipeline and optional crop helper against one upload."""

    if not callable(pipeline_extract):
        raise RuntimeError("The receipt extraction pipeline is unavailable.")

    with temporary_image(upload_bytes, original_name) as image_path:
        result = pipeline_extract(image_path)
        preview = None
        preview_error = None

        if callable(rectify_preview):
            try:
                preview = rectify_preview(image_path)
            except Exception as error:  # A result remains useful without a preview.
                preview_error = type(error).__name__
        return result, preview, preview_error


def _clear_stale_result(fingerprint: str) -> None:
    saved_result = st.session_state.get("receipt_analysis")
    if saved_result and saved_result.get("fingerprint") != fingerprint:
        st.session_state.pop("receipt_analysis", None)


def _is_supported_source_name(original_name: str) -> bool:
    suffix = Path(original_name).suffix.lower()
    return suffix in ALLOWED_SUFFIXES or suffix in SUPPORTED_DOCUMENT_SUFFIXES


def _prepare_source_payload(source_bytes: bytes, original_name: str) -> dict[str, Any]:
    """Normalize uploaded images and supported document files into a processable payload."""

    suffix = Path(original_name).suffix.lower()
    if suffix in ALLOWED_SUFFIXES:
        return {"kind": "image", "bytes": source_bytes, "name": original_name, "text": None}

    if suffix == ".pdf":
        temp_pdf_path: str | None = None
        temp_image_path: str | None = None
        pdf_document: Any | None = None
        try:
            import importlib.util

            pymupdf_spec = importlib.util.find_spec("pymupdf") or importlib.util.find_spec("fitz")
            if pymupdf_spec is not None:
                import pymupdf

                descriptor, temp_pdf_path = tempfile.mkstemp(prefix="receipt_pdf_", suffix=".pdf")
                with os.fdopen(descriptor, "wb") as pdf_file:
                    pdf_file.write(source_bytes)
                pdf_document = pymupdf.open(temp_pdf_path)
                if len(pdf_document) == 0:
                    raise ValueError("The PDF contains no pages to analyze.")

                image_descriptor, temp_image_path = tempfile.mkstemp(
                    prefix="receipt_pdf_page_", suffix=".png"
                )
                os.close(image_descriptor)
                page = pdf_document[0]
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                pixmap.save(temp_image_path)
                with open(temp_image_path, "rb") as image_file:
                    image_bytes = image_file.read()
                return {
                    "kind": "image",
                    "bytes": image_bytes,
                    "name": f"{Path(original_name).stem}.png",
                    "text": None,
                }

            return {
                "kind": "document",
                "bytes": None,
                "name": original_name,
                "text": "PDF parsing library is not available in this environment. "
                "Please upload an image or a Word document instead.",
            }
        finally:
            if pdf_document is not None:
                try:
                    pdf_document.close()
                except Exception:
                    pass
            for temp_path in (temp_pdf_path, temp_image_path):
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass

    if suffix in {".doc", ".docx"}:
        temp_doc_path: str | None = None
        try:
            import importlib.util

            docx_spec = importlib.util.find_spec("docx")
            if docx_spec is not None:
                from docx import Document

                descriptor, temp_doc_path = tempfile.mkstemp(prefix="receipt_doc_", suffix=".docx")
                with os.fdopen(descriptor, "wb") as doc_file:
                    doc_file.write(source_bytes)
                document = Document(temp_doc_path)
                text = "\n".join(
                    paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()
                )
                return {"kind": "document", "bytes": None, "name": original_name, "text": text}

            return {
                "kind": "document",
                "bytes": None,
                "name": original_name,
                "text": "Word document parsing library is not available in this environment. "
                "Please upload an image or a PDF instead.",
            }
        finally:
            if temp_doc_path and os.path.exists(temp_doc_path):
                try:
                    os.unlink(temp_doc_path)
                except OSError:
                        pass

    raise ValueError(f"Unsupported file type: {original_name}")


def main() -> None:
    st.set_page_config(page_title="Receipt Capture & Extraction", page_icon="🧾", layout="wide")

    st.title("Receipt Capture & Extraction")
    st.write(
        "Use live camera capture or upload a receipt photo. The image is analyzed only "
        "when the quality gate passes, so blurry or poorly lit captures are rejected."
    )

    with st.sidebar:
        st.header("Capture tips")
        st.markdown(
            "- Keep all four receipt edges in frame.\n"
            "- Use even light and avoid glare over text.\n"
            "- Hold the phone steady and close enough for small print."
        )
        st.caption(
            "Captures are processed locally and are removed from temporary storage after analysis."
        )

    camera_file = st.camera_input(
        "Live receipt capture",
        help="Capture a receipt directly from your camera. The app will only analyze it when it passes quality checks.",
    )
    uploaded_file = st.file_uploader(
        "Receipt photo or document",
        type=["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp", "pdf", "docx"],
        help="Use a clear phone photo, a receipt scan, a PDF, or a DOCX document.",
    )

    source_file = camera_file if camera_file is not None else uploaded_file
    if source_file is None:
        st.info("Choose a receipt photo, PDF, Word document, or use live capture to begin.")
        return

    source_bytes = source_file.getvalue() if hasattr(source_file, "getvalue") else None
    if not source_bytes:
        st.error("That file is empty. Please choose a valid image, PDF, or Word document.")
        return
    if len(source_bytes) > MAX_UPLOAD_BYTES:
        st.error("This file is larger than 15 MB. Please upload or capture a smaller file.")
        return

    source_name = getattr(source_file, "name", None) or (
        "live-capture.jpg" if camera_file is not None else "receipt-upload.png"
    )
    if not _is_supported_source_name(source_name):
        st.error("Unsupported file type. Please upload an image, PDF, or Word document.")
        return

    source_kind = "live capture" if camera_file is not None else "upload"
    try:
        source_payload = _prepare_source_payload(source_bytes, source_name)
    except Exception:
        st.error(
            "This file could not be read. Please upload a valid image, PDF, or DOCX document."
        )
        return
    fingerprint = hashlib.sha256(source_bytes).hexdigest()
    _clear_stale_result(fingerprint)

    source_column, details_column = st.columns((2, 1))
    with source_column:
        st.subheader("Source preview")
        if source_payload["kind"] == "document":
            st.info("Word document detected. The text content is shown below for review.")
            st.text_area("Extracted text", source_payload["text"] or "", height=220)
        else:
            try:
                st.image(source_payload["bytes"], caption=source_name, use_container_width=True)
            except Exception:
                st.error(
                    "The selected file could not be decoded as an image. "
                    "Please choose a valid receipt photo or PDF."
                )
                return
    with details_column:
        st.subheader("Capture details")
        st.write(f"**Source:** {source_kind.title()}")
        st.write(f"**File:** {source_name}")
        st.write(f"**Size:** {len(source_bytes) / 1024:.1f} KB")
        st.caption(
            "The source is kept only for this browser session; a temporary "
            "on-disk copy exists only during processing."
        )

    if pipeline_extract is None:
        st.error(
            "The receipt pipeline could not be loaded. Install the project "
            "dependencies and restart the app."
        )
        if PIPELINE_IMPORT_ERROR is not None:
            with st.expander("Technical details"):
                st.code(f"{type(PIPELINE_IMPORT_ERROR).__name__}: {PIPELINE_IMPORT_ERROR}")
        return

    run_column, clear_column = st.columns((1, 1))
    with run_column:
        button_label = "Analyze captured receipt" if camera_file is not None else "Analyze receipt"
        run_requested = st.button(button_label, type="primary", use_container_width=True)
    with clear_column:
        if st.button("Clear result", use_container_width=True):
            st.session_state.pop("receipt_analysis", None)
            st.session_state.pop("last_processed_fingerprint", None)
            st.rerun()

    last_processed_fingerprint = st.session_state.get("last_processed_fingerprint")
    should_auto_run = fingerprint != last_processed_fingerprint and source_file is not None

    if source_payload["kind"] == "document":
        st.info("Document upload detected. The text content is ready for review.")
        return

    if run_requested or should_auto_run:
        # Avoid showing an older result if a fresh attempt fails halfway through.
        st.session_state.pop("receipt_analysis", None)
        with st.spinner("Checking image quality and reading the receipt…"):
            try:
                result, preview, preview_error = _process_upload(
                    source_payload["bytes"],
                    source_payload["name"],
                )
            except Exception as error:
                st.error(
                    "We couldn't analyze this image. Try a clear, well-lit photo "
                    "with the whole receipt in frame."
                )
                with st.expander("Technical details"):
                    st.code(f"{type(error).__name__}: {error}")
            else:
                if not _should_accept_capture(result):
                    st.warning(
                        "The captured image did not pass the quality checks. Please retake it "
                        "with better focus, lighting, and all receipt edges visible."
                    )
                else:
                    st.session_state["receipt_analysis"] = {
                        "fingerprint": fingerprint,
                        "result": result,
                        "preview": preview,
                        "preview_error": preview_error,
                    }
                st.session_state["last_processed_fingerprint"] = fingerprint

    saved_result = st.session_state.get("receipt_analysis")
    if saved_result and saved_result.get("fingerprint") == fingerprint:
        st.divider()
        _render_result(
            saved_result.get("result"),
            saved_result.get("preview"),
            saved_result.get("preview_error"),
        )


if __name__ == "__main__":
    main()
