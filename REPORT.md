# Receipt Capture & Extraction — implementation report

## Validation status

This repository has a deterministic unit suite for the response contract,
quality gates, perspective rectification, parsing rules, and evaluator trust
score. On 2026-08-10 it passed with:

```text
python -m unittest discover -s tests -v
30 tests, all passing
```

The Streamlit app was also started locally and returned `200 ok` from its
`/_stcore/health` endpoint. That is a launch smoke test only; it was not a
clean-environment installation or a deployed OCR validation.

After installing the local Tesseract executable, a supplied 1122×1402 sharp
digital receipt was also run end-to-end through the active virtual environment.
It returned the expected date (`2025-05-08`), total (`1250.0`), and currency
(`INR`); the merchant spelling is shown but remains reviewed because two OCR
copies differ by one character. Three warm runs had a median 538 ms wall time
on this machine. This is a functional smoke check, not a labelled accuracy or
memory benchmark. A supplied blurred photo was correctly rejected before OCR.

No rights-cleared real receipt-photo set is in this repository yet. Therefore
no accuracy, abstention, or trust-score metric is reported here, and no
synthetic metric is presented as a real-world result. Once at least twelve
labelled photos are collected with permission, run:

```text
python eval.py --data test/
```

and replace this section with the command output and collection details.

## Approach and thresholds

The pipeline is CPU-only and has no learned model weights. It reads image
dimensions before decoding, rejects extreme inputs, then downscales to a
maximum 1,800-pixel side / 2.6-megapixel working image before contour analysis
or OCR. It measures Laplacian variance for blur, brightness and clipped-white
fractions for exposure/glare, locates a scored quadrilateral with OpenCV, and
rectifies it with a bounded perspective transform.

Field extraction uses one local, time-bounded Tesseract pass. `src/parsing.py`
turns typed OCR lines into deterministic candidates: top receipt lines for the
merchant, clear or reviewed date formats, labelled total lines, and explicit
currency evidence. Missing, ambiguous, or low-confidence candidates are
flagged for review. A failed image/OCR stage always returns null reviewed
fields, never a high-confidence guess.

The current implementation thresholds are intentionally conservative starting
points, not empirically calibrated claims. The blur gate is a Laplacian
variance below 20; the receipt must occupy at least 12% of the frame and must
not touch an image edge. Field confidence is derived from OCR confidence,
receipt position, label strength, and competing candidates; unlabelled totals,
ambiguous numeric dates, and bare `$` currency evidence remain reviewed.
These thresholds must be tuned only after measuring a rights-cleared holdout
set.

## Evidenced failure cases

These are deterministic regression cases, not substitutes for real-photo
failure analysis.

1. A receipt that occupies about 4% of the frame is rejected with “move
   closer.” `tests/test_pipeline.py` verifies that the synthetic `small` scene
   falls below the 12% minimum.
2. A receipt whose left corners fall outside the image is rejected with an
   actionable cut-off message. The synthetic `cutoff` scene exercises that
   condition and produces no rectified preview.
3. If Tesseract is missing or cannot start, the pipeline returns a complete
   reviewed-abstention response with a local-OCR installation message. It does
   not throw from `extract()` or expose partial field guesses.

## With another week

I would collect and annotate diverse, rights-cleared receipt photos; log the
quality metrics and field confidence distributions; tune thresholds against a
held-out set; add OCR preprocessing variants only when evidence justifies
them; and measure peak memory / latency on the intended Streamlit host. I
would also deploy the app and test the installed Linux Tesseract binary using
representative uploads.

## AI-tool disclosure

OpenAI Codex was used to help implement and review the Python pipeline,
deterministic test fixtures, documentation, and verification commands. The
pipeline remains dependency-light and all field-ranking rules are explicit in
the repository for live walkthrough and review.
