"""CPU-bounded receipt capture, rectification, OCR, and extraction pipeline.

The public contract is intentionally small:

``extract(image_path: str) -> dict``

All failures are converted into the documented response schema.  The module
does not load ML model weights or call remote services; image geometry uses
OpenCV and text recognition is delegated to a locally installed Tesseract
binary only after the capture-quality gate has passed.
"""

from __future__ import annotations

import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .parsing import OCRLine, lines_from_ocr_data, parse_receipt_fields
from .schema import ResultDict, make_result

try:  # Pillow reads image headers before OpenCV is asked to decode pixels.
    from PIL import Image, UnidentifiedImageError
except ImportError:  # pragma: no cover - Pillow is a pinned runtime dependency.
    Image = None  # type: ignore[assignment,misc]
    UnidentifiedImageError = OSError  # type: ignore[assignment,misc]


# Input limits protect the free-tier memory target.  Header dimensions are
# checked before decoding; all subsequent OpenCV work uses the smaller working
# image.  The values are deliberately documented constants rather than hidden
# resize magic, so they can be calibrated on the real evaluation set.
MAX_INPUT_FILE_BYTES = 24 * 1024 * 1024
MAX_SOURCE_PIXELS = 24_000_000
MIN_SOURCE_SHORT_SIDE = 480
MIN_SOURCE_PIXELS = 360_000
MAX_WORKING_DIMENSION = 1_800
MAX_WORKING_PIXELS = 2_600_000
MAX_RECTIFIED_DIMENSION = 1_800
MAX_RECTIFIED_PIXELS = 2_600_000
MIN_RECEIPT_AREA_RATIO = 0.12
EDGE_TOLERANCE_PIXELS = 8
BLUR_LAPLACIAN_VARIANCE_MIN = 20.0
OCR_TIMEOUT_SECONDS = 7
TESSERACT_ENVIRONMENT_VARIABLES: tuple[str, ...] = ("TESSERACT_CMD", "TESSERACT_PATH")


class OCRUnavailableError(RuntimeError):
    """Raised internally when local Tesseract cannot be loaded or started."""


class OCRExecutionError(RuntimeError):
    """Raised internally when a local OCR invocation fails or times out."""


@dataclass(frozen=True)
class QualityMetrics:
    """Fast pre-OCR measurements calculated on a bounded working image."""

    width: int
    height: int
    laplacian_variance: float
    brightness_mean: float
    brightness_stddev: float
    dark_fraction: float
    clipped_bright_fraction: float
    glare_fraction: float


@dataclass(frozen=True)
class ReceiptDetection:
    """A scored quadrilateral candidate for perspective rectification."""

    quad: np.ndarray
    area_ratio: float
    rectangularity: float
    aspect_ratio: float
    score: float
    touches_edge: bool


def _elapsed_ms(started_at: float) -> float:
    return max(0.0, (time.perf_counter() - started_at) * 1000.0)


def _native_float(value: object, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def resolve_tesseract_command(pytesseract_module: object) -> str | None:
    """Find a local Tesseract executable without depending solely on ``PATH``.

    Windows installers frequently update PATH only for new terminal sessions.
    The Streamlit process may therefore be unable to find a freshly installed
    engine even though it exists in the standard program directory.  This
    resolver checks an explicit environment override, PATH, and conventional
    Windows install locations, then configures pytesseract with the exact path.
    It performs only a handful of metadata checks and no subprocess call.
    """

    backend = getattr(pytesseract_module, "pytesseract", pytesseract_module)
    configured = getattr(backend, "tesseract_cmd", "tesseract")
    raw_candidates: list[str] = []
    for variable in TESSERACT_ENVIRONMENT_VARIABLES:
        value = os.environ.get(variable, "").strip().strip('"')
        if value:
            raw_candidates.append(value)
    if isinstance(configured, str) and configured.strip():
        raw_candidates.append(configured.strip().strip('"'))
    raw_candidates.append("tesseract")

    if os.name == "nt":
        for root_variable in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            root = os.environ.get(root_variable, "").strip()
            if not root:
                continue
            if root_variable == "LOCALAPPDATA":
                raw_candidates.append(str(Path(root) / "Programs" / "Tesseract-OCR" / "tesseract.exe"))
            else:
                raw_candidates.append(str(Path(root) / "Tesseract-OCR" / "tesseract.exe"))

    seen: set[str] = set()
    for raw_candidate in raw_candidates:
        candidate = raw_candidate.strip()
        if not candidate or candidate.casefold() in seen:
            continue
        seen.add(candidate.casefold())
        located = shutil.which(candidate)
        path = Path(located or candidate).expanduser()
        if not path.is_file():
            continue
        command = str(path.resolve())
        try:
            setattr(backend, "tesseract_cmd", command)
        except (AttributeError, TypeError):
            return None
        return command
    return None


def _header_dimensions(path: Path) -> tuple[int, int] | None:
    """Read only image metadata, returning ``(width, height)`` when possible."""

    if Image is None:
        return None
    try:
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, UnidentifiedImageError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def _resize_bounded(
    image: np.ndarray,
    *,
    max_dimension: int = MAX_WORKING_DIMENSION,
    max_pixels: int = MAX_WORKING_PIXELS,
) -> tuple[np.ndarray, float]:
    """Downscale an image before expensive vision/OCR stages, never upscale."""

    if image.ndim < 2:
        raise ValueError("Image does not have usable dimensions.")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Image has empty dimensions.")
    if max_dimension <= 0 or max_pixels <= 0:
        raise ValueError("Resize limits must be positive.")

    dimension_scale = max_dimension / float(max(height, width))
    pixel_scale = math.sqrt(max_pixels / float(height * width))
    scale = min(1.0, dimension_scale, pixel_scale)
    if scale >= 0.9999:
        return image, 1.0

    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    # Rounding may nudge a dimension one pixel above the cap.  Tighten it in
    # place instead of accepting an unbounded result.
    while resized_width * resized_height > max_pixels:
        if resized_width >= resized_height and resized_width > 1:
            resized_width -= 1
        elif resized_height > 1:
            resized_height -= 1
        else:  # pragma: no cover - impossible for positive maximum pixels.
            break
    resized_width = min(resized_width, max_dimension)
    resized_height = min(resized_height, max_dimension)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return resized, resized_width / float(width)


def _load_working_image(image_path: str | Path) -> tuple[np.ndarray | None, list[str]]:
    """Validate/decode an image and immediately apply the working-size bound."""

    try:
        path = Path(image_path).expanduser()
    except (TypeError, ValueError):
        return None, ["Choose a valid receipt image file and try again."]
    if not path.is_file():
        return None, ["The receipt image could not be found. Choose the image file and try again."]

    try:
        file_size = path.stat().st_size
    except OSError:
        return None, ["The receipt image could not be opened. Please choose the file again."]
    if file_size <= 0:
        return None, ["The selected image is empty. Choose a valid receipt photo."]
    if file_size > MAX_INPUT_FILE_BYTES:
        return None, ["This image file is too large. Upload a photo smaller than 24 MB."]

    dimensions = _header_dimensions(path)
    if dimensions is None:
        return None, ["The selected file is not a readable image. Choose a JPEG, PNG, or other supported photo."]
    source_width, source_height = dimensions
    source_pixels = source_width * source_height
    if source_pixels > MAX_SOURCE_PIXELS:
        return None, [
            "This photo has too many pixels to process safely. Resize it below 24 megapixels and try again."
        ]
    if min(source_width, source_height) < MIN_SOURCE_SHORT_SIDE or source_pixels < MIN_SOURCE_PIXELS:
        return None, [
            "Image resolution is too low — move closer to the receipt and retake the photo."
        ]

    try:
        decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except cv2.error:
        decoded = None
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
        return None, ["The selected image could not be decoded. Choose a clear receipt photo and try again."]
    try:
        working, _ = _resize_bounded(decoded)
    except (cv2.error, ValueError, MemoryError):
        return None, ["This image could not be prepared safely. Resize it and try again."]
    return working, []


def _analysis_gray(image: np.ndarray, maximum_dimension: int = 960) -> np.ndarray:
    """Return a small grayscale image for bounded quality measurements."""

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        conversion = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        gray = cv2.cvtColor(image, conversion)
    else:
        raise ValueError("Expected a grayscale, BGR, or BGRA image.")
    height, width = gray.shape[:2]
    if max(height, width) > maximum_dimension:
        scale = maximum_dimension / float(max(height, width))
        gray = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return gray


def measure_quality(image: np.ndarray) -> QualityMetrics:
    """Measure resolution, blur, exposure, and high-light glare signals."""

    gray = _analysis_gray(image)
    height, width = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian_variance = _native_float(cv2.Laplacian(blurred, cv2.CV_64F).var())
    brightness_mean = _native_float(np.mean(gray))
    brightness_stddev = _native_float(np.std(gray))
    dark_fraction = _native_float(np.mean(gray <= 28))
    clipped_bright_fraction = _native_float(np.mean(gray >= 252))

    # A receipt itself is commonly light; a glare signal should therefore only
    # count very bright connected regions, not ordinary off-white paper.
    glare_fraction = 0.0
    # Avoid allocating a component-label image unless clipped highlights are
    # actually present.  This keeps the usual photo path linear with a small
    # constant factor and avoids a needless ~4-byte-per-pixel temporary.
    if clipped_bright_fraction >= 0.01:
        bright_mask = np.where(gray >= 252, 255, 0).astype(np.uint8)
        component_count, _, statistics, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
        if component_count > 1:
            largest_index = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
            largest_component = int(statistics[largest_index, cv2.CC_STAT_AREA])
            component_width = int(statistics[largest_index, cv2.CC_STAT_WIDTH])
            component_height = int(statistics[largest_index, cv2.CC_STAT_HEIGHT])
            # A near-full-frame white component with normal contrast is a
            # scan/digital receipt background, not a local glare patch.  A
            # genuinely blown-out page still fails the separate low-contrast
            # overexposure check in ``assess_quality``.
            page_background = (
                component_width / float(max(1, width)) >= 0.90
                and component_height / float(max(1, height)) >= 0.90
                and brightness_stddev >= 20.0
            )
            if not page_background:
                glare_fraction = largest_component / float(max(1, gray.size))
    return QualityMetrics(
        width=int(width),
        height=int(height),
        laplacian_variance=laplacian_variance,
        brightness_mean=brightness_mean,
        brightness_stddev=brightness_stddev,
        dark_fraction=dark_fraction,
        clipped_bright_fraction=clipped_bright_fraction,
        glare_fraction=float(glare_fraction),
    )


def assess_quality(metrics: QualityMetrics) -> list[str]:
    """Turn pre-OCR measurements into concise, actionable capture failures."""

    issues: list[str] = []
    if metrics.laplacian_variance < BLUR_LAPLACIAN_VARIANCE_MIN:
        issues.append("Image is too blurry — hold steady, focus on the receipt, and retake it.")
    if metrics.brightness_mean < 48.0 and metrics.dark_fraction > 0.45:
        issues.append("Image is severely underexposed — use brighter, even lighting and retake it.")
    if metrics.brightness_mean > 247.0 and metrics.brightness_stddev < 18.0:
        issues.append("Image is severely overexposed — reduce direct light and retake it.")
    if metrics.glare_fraction > 0.11 or (
        metrics.clipped_bright_fraction > 0.23 and metrics.glare_fraction > 0.045
    ):
        issues.append("Strong glare is covering part of the receipt — tilt the phone or use softer light.")
    return issues


def order_quad_points(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Order a valid quadrilateral as top-left, top-right, bottom-right, bottom-left."""

    quad = np.asarray(points, dtype=np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError("Receipt corners must be four finite two-dimensional points.")
    sums = quad.sum(axis=1)
    differences = quad[:, 0] - quad[:, 1]
    indices = [int(np.argmin(sums)), int(np.argmax(differences)), int(np.argmax(sums)), int(np.argmin(differences))]
    if len(set(indices)) == 4:
        ordered = quad[indices]
    else:
        # Fallback for unusual diamond-like contours where sum/difference ties.
        by_y = quad[np.argsort(quad[:, 1], kind="stable")]
        top = by_y[:2][np.argsort(by_y[:2, 0], kind="stable")]
        bottom = by_y[2:][np.argsort(by_y[2:, 0], kind="stable")]
        ordered = np.array((top[0], top[1], bottom[1], bottom[0]), dtype=np.float32)
    if abs(cv2.contourArea(ordered.reshape(-1, 1, 2))) < 1.0:
        raise ValueError("Receipt corners have negligible area.")
    return ordered.astype(np.float32)


def _quad_edge_contact(quad: np.ndarray, image_shape: Sequence[int]) -> bool:
    height, width = int(image_shape[0]), int(image_shape[1])
    tolerance = max(EDGE_TOLERANCE_PIXELS, int(min(height, width) * 0.012))
    return bool(
        np.any(quad[:, 0] <= tolerance)
        or np.any(quad[:, 1] <= tolerance)
        or np.any(quad[:, 0] >= width - 1 - tolerance)
        or np.any(quad[:, 1] >= height - 1 - tolerance)
    )


def _quad_aspect_ratio(quad: np.ndarray) -> float:
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    width = (top + bottom) / 2.0
    height = (left + right) / 2.0
    if min(width, height) <= 0:
        return 0.0
    return float(min(width, height) / max(width, height))


def _candidate_detection(quad: np.ndarray, image_shape: Sequence[int]) -> ReceiptDetection | None:
    height, width = int(image_shape[0]), int(image_shape[1])
    area = abs(float(cv2.contourArea(quad.reshape(-1, 1, 2))))
    image_area = float(height * width)
    area_ratio = area / image_area
    if area_ratio < 0.006 or area_ratio > 0.995:
        return None
    rectangle = cv2.minAreaRect(quad.astype(np.float32))
    rectangle_area = float(rectangle[1][0] * rectangle[1][1])
    if rectangle_area <= 1.0:
        return None
    rectangularity = min(1.0, area / rectangle_area)
    aspect_ratio = _quad_aspect_ratio(quad)
    aspect_score = 1.0 if 0.18 <= aspect_ratio <= 0.95 else max(0.0, 1.0 - abs(aspect_ratio - 0.50))
    area_score = min(1.0, area_ratio / 0.36)
    score = 0.54 * area_score + 0.34 * rectangularity + 0.12 * aspect_score
    return ReceiptDetection(
        quad=quad,
        area_ratio=float(area_ratio),
        rectangularity=float(rectangularity),
        aspect_ratio=float(aspect_ratio),
        score=float(score),
        touches_edge=_quad_edge_contact(quad, image_shape),
    )


def _contours_for_receipt(gray: np.ndarray) -> list[np.ndarray]:
    """Generate a small set of contour proposals without OCR or ML models."""

    softened = cv2.GaussianBlur(gray, (5, 5), 0)
    low, high = 45, 135
    edges = cv2.Canny(softened, low, high)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2,
    )
    # The Otsu proposal complements Canny when the paper edge has a soft but
    # useful brightness boundary against a tabletop.
    _, threshold = cv2.threshold(softened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=1,
    )
    contour_sets: list[np.ndarray] = []
    for binary, mode in ((edges, cv2.RETR_LIST), (threshold, cv2.RETR_LIST)):
        found = cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found) == 2 else found[1]
        contour_sets.extend(contours)
    return contour_sets


def _looks_like_full_frame_document(gray: np.ndarray) -> bool:
    """Recognize a sharp, mostly white scan/digital receipt filling the frame.

    Full-frame scans often produce an image-border contour plus nested table
    contours.  Treating the largest nested table as the document loses headers
    and totals.  This narrow fallback is intentionally limited to bright,
    high-contrast document-like images; normal camera photos still use contour
    scoring and cut-off rejection.
    """

    # Use clipped-white pixels rather than merely light paper.  Phone photos
    # of a large receipt can be off-white too; only scan/digital backgrounds
    # normally contain this much pure white across the full frame.
    bright_fraction = _native_float(np.mean(gray >= 252))
    dark_fraction = _native_float(np.mean(gray <= 55))
    contrast = _native_float(np.std(gray))
    return bright_fraction >= 0.42 and dark_fraction <= 0.10 and contrast >= 28.0


def find_receipt_quad(image: np.ndarray) -> ReceiptDetection | None:
    """Find the highest-scoring plausible receipt quadrilateral in an image."""

    if image.ndim < 2:
        return None
    source_height, source_width = image.shape[:2]
    try:
        gray = _analysis_gray(image, maximum_dimension=MAX_WORKING_DIMENSION)
    except (cv2.error, ValueError):
        return None
    if _looks_like_full_frame_document(gray):
        full_quad = np.array(
            (
                (0.0, 0.0),
                (float(source_width - 1), 0.0),
                (float(source_width - 1), float(source_height - 1)),
                (0.0, float(source_height - 1)),
            ),
            dtype=np.float32,
        )
        return ReceiptDetection(
            quad=full_quad,
            area_ratio=1.0,
            rectangularity=1.0,
            aspect_ratio=float(min(source_width, source_height) / max(source_width, source_height)),
            score=1.0,
            # This is a scan/digital document fallback, not a physical receipt
            # whose corners happen to touch the camera frame.
            touches_edge=False,
        )
    image_area = gray.shape[0] * gray.shape[1]
    candidates: list[ReceiptDetection] = []
    seen: set[tuple[int, ...]] = set()
    try:
        contours = _contours_for_receipt(gray)
    except cv2.error:
        return None
    for contour in contours:
        area = abs(float(cv2.contourArea(contour)))
        if area < image_area * 0.006:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        # Two epsilon values help when a printed border creates a slightly
        # jagged contour while retaining a real four-corner perspective shape.
        approximations = (
            cv2.approxPolyDP(contour, 0.018 * perimeter, True),
            cv2.approxPolyDP(contour, 0.032 * perimeter, True),
        )
        for approximation in approximations:
            if len(approximation) != 4 or not cv2.isContourConvex(approximation):
                continue
            try:
                quad = order_quad_points(approximation.reshape(4, 2))
            except ValueError:
                continue
            # ``_analysis_gray`` may have reduced a direct caller's large
            # image.  Return corners in that caller's original coordinate
            # system so they are always safe to pass to ``rectify_receipt``.
            if gray.shape[0] != source_height or gray.shape[1] != source_width:
                quad = quad.copy()
                quad[:, 0] *= source_width / float(gray.shape[1])
                quad[:, 1] *= source_height / float(gray.shape[0])
            key = tuple(np.round(quad.reshape(-1) / 4.0).astype(int))
            if key in seen:
                continue
            seen.add(key)
            candidate = _candidate_detection(quad, image.shape)
            if candidate is not None:
                candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            -candidate.area_ratio,
            -candidate.rectangularity,
            candidate.touches_edge,
        )
    )
    return candidates[0]


def assess_detection(detection: ReceiptDetection | None) -> list[str]:
    """Return action-oriented size/cut-off failures for a receipt proposal."""

    if detection is None:
        return [
            "Receipt edges could not be found — place the whole receipt on a contrasting surface and retake it."
        ]
    if detection.touches_edge:
        return ["Receipt appears cut off at the image edge — include all four receipt corners and retake it."]
    if detection.area_ratio < MIN_RECEIPT_AREA_RATIO:
        return ["Receipt is too small in the frame — move closer and retake the photo."]
    return []


def rectify_receipt(
    image: np.ndarray,
    quad: ReceiptDetection | np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray | None:
    """Perspective-rectify a receipt with explicit dimensions and memory caps."""

    source_quad = quad.quad if isinstance(quad, ReceiptDetection) else quad
    try:
        corners = order_quad_points(source_quad)
    except (TypeError, ValueError):
        return None
    try:
        width_top = float(np.linalg.norm(corners[1] - corners[0]))
        width_bottom = float(np.linalg.norm(corners[2] - corners[3]))
        height_left = float(np.linalg.norm(corners[3] - corners[0]))
        height_right = float(np.linalg.norm(corners[2] - corners[1]))
    except (TypeError, ValueError):
        return None
    target_width = int(round(max(width_top, width_bottom)))
    target_height = int(round(max(height_left, height_right)))
    if min(target_width, target_height) < 32:
        return None

    scale = min(
        1.0,
        MAX_RECTIFIED_DIMENSION / float(max(target_width, target_height)),
        math.sqrt(MAX_RECTIFIED_PIXELS / float(target_width * target_height)),
    )
    target_width = max(1, int(round(target_width * scale)))
    target_height = max(1, int(round(target_height * scale)))
    if target_width * target_height > MAX_RECTIFIED_PIXELS:
        return None
    destination = np.array(
        (
            (0, 0),
            (target_width - 1, 0),
            (target_width - 1, target_height - 1),
            (0, target_height - 1),
        ),
        dtype=np.float32,
    )
    try:
        transform = cv2.getPerspectiveTransform(corners, destination)
        rectified = cv2.warpPerspective(
            image,
            transform,
            (target_width, target_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except cv2.error:
        return None
    if rectified is None or rectified.size == 0:
        return None
    return rectified


def prepare_for_ocr(rectified: np.ndarray) -> np.ndarray | None:
    """Create a high-contrast, bounded grayscale image for local Tesseract."""

    try:
        gray = _analysis_gray(rectified, maximum_dimension=MAX_RECTIFIED_DIMENSION)
    except (cv2.error, ValueError):
        return None
    height, width = gray.shape[:2]
    if min(height, width) < 32:
        return None
    # Upscaling a modest rectified crop is bounded and improves small receipt
    # glyphs.  It never creates a result above the same pixel/dimension limits.
    desired_short_side = 700
    if min(height, width) < desired_short_side:
        scale = min(
            desired_short_side / float(min(height, width)),
            MAX_RECTIFIED_DIMENSION / float(max(height, width)),
            math.sqrt(MAX_RECTIFIED_PIXELS / float(height * width)),
        )
        if scale > 1.02:
            gray = cv2.resize(
                gray,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_CUBIC,
            )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    # Otsu avoids per-image tuning and generally preserves receipt digits
    # without the expensive model-based preprocessing that is out of scope.
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def ocr_lines(image: np.ndarray) -> list[OCRLine]:
    """Run a time-bounded local Tesseract pass and return typed visual lines."""

    try:
        import pytesseract
    except ImportError as error:
        raise OCRUnavailableError("The Python pytesseract package is not installed.") from error

    if resolve_tesseract_command(pytesseract) is None:
        raise OCRUnavailableError("The Tesseract executable is not available on PATH or in a standard install location.")

    prepared = prepare_for_ocr(image)
    if prepared is None:
        raise OCRExecutionError("The rectified receipt could not be prepared for OCR.")
    try:
        data = pytesseract.image_to_data(
            prepared,
            config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT,
            timeout=OCR_TIMEOUT_SECONDS,
        )
    except getattr(pytesseract, "TesseractNotFoundError", RuntimeError) as error:
        raise OCRUnavailableError("The Tesseract executable is not available on PATH.") from error
    except RuntimeError as error:
        raise OCRExecutionError(str(error)) from error
    except Exception as error:  # noqa: BLE001 - third-party OCR failures are normalized by extract.
        raise OCRExecutionError(str(error)) from error
    if not isinstance(data, dict):
        raise OCRExecutionError("Tesseract returned an unexpected OCR result.")
    return lines_from_ocr_data(data)


def _failure(started_at: float, issues: Sequence[str]) -> ResultDict:
    return make_result(False, issues, processing_ms=_elapsed_ms(started_at))


def extract(image_path: str) -> ResultDict:
    """Extract reviewable receipt fields from one local image path.

    This function is intentionally exception-safe for malformed uploads,
    unavailable OCR dependencies, and third-party image/OCR errors.  It
    returns fields only after every capture and geometry gate passes.
    """

    started_at = time.perf_counter()
    try:
        image, load_issues = _load_working_image(image_path)
        if image is None:
            return _failure(started_at, load_issues)

        quality_issues = assess_quality(measure_quality(image))
        detection = find_receipt_quad(image)
        # Geometry is still cheap and bounded after a quality failure.  When a
        # receipt is both blurry and tiny/cut off, surfacing both actionable
        # causes is more useful than making the user fix them one at a time.
        if quality_issues:
            if detection is not None:
                quality_issues.extend(assess_detection(detection))
            return _failure(started_at, quality_issues)
        geometry_issues = assess_detection(detection)
        if geometry_issues:
            return _failure(started_at, geometry_issues)

        rectified = rectify_receipt(image, detection)
        if rectified is None:
            return _failure(
                started_at,
                ["Receipt perspective could not be corrected safely. Retake the photo with all edges visible."],
            )

        try:
            lines = ocr_lines(rectified)
        except OCRUnavailableError:
            return _failure(
                started_at,
                ["Local OCR is unavailable — install Tesseract OCR on this system and try again."],
            )
        except OCRExecutionError:
            # This is deliberately non-specific to avoid presenting a Tesseract
            # implementation detail as if it were a user's capture mistake.
            return _failure(
                started_at,
                ["Text recognition could not complete safely. Retake a clear receipt photo and try again."],
            )

        fields = parse_receipt_fields(lines)
        return make_result(True, (), fields=fields, processing_ms=_elapsed_ms(started_at))
    except Exception:  # noqa: BLE001 - public contract must not leak third-party exceptions.
        return _failure(
            started_at,
            ["The receipt could not be processed safely. Please retake the photo and try again."],
        )


def rectify_preview(image_path: str) -> np.ndarray | None:
    """Return a bounded BGR rectified receipt crop for the Streamlit preview.

    The preview intentionally skips OCR and returns ``None`` for invalid,
    too-small, or edge-cut-off receipts.  It never modifies the source path.
    """

    try:
        image, issues = _load_working_image(image_path)
        if image is None or issues:
            return None
        detection = find_receipt_quad(image)
        if assess_detection(detection):
            return None
        return rectify_receipt(image, detection)
    except Exception:  # noqa: BLE001 - preview failure must not break extraction UI.
        return None


__all__ = [
    "BLUR_LAPLACIAN_VARIANCE_MIN",
    "EDGE_TOLERANCE_PIXELS",
    "MAX_RECTIFIED_DIMENSION",
    "MAX_RECTIFIED_PIXELS",
    "MAX_WORKING_DIMENSION",
    "MAX_WORKING_PIXELS",
    "MIN_RECEIPT_AREA_RATIO",
    "OCRExecutionError",
    "OCRUnavailableError",
    "QualityMetrics",
    "ReceiptDetection",
    "assess_detection",
    "assess_quality",
    "extract",
    "find_receipt_quad",
    "measure_quality",
    "ocr_lines",
    "order_quad_points",
    "prepare_for_ocr",
    "rectify_preview",
    "rectify_receipt",
    "resolve_tesseract_command",
]
