# Receipt Capture & Extraction — Build Progress

This file is the source of truth for implementation progress. A box is checked only
after the corresponding code or artifact exists and has been verified locally.

## Engineering guardrails

- [x] Read and translate the take-home brief into an implementation plan.
- [x] Choose a CPU-only, memory-conscious architecture (OpenCV + Tesseract).
- [x] Define a single public pipeline contract used by the UI and evaluator.
- [x] Keep image processing bounded: downscale before expensive operations and avoid loading model weights.
- [x] Fail closed: return actionable quality failures or `needs_review`, never invent a high-confidence value.
- [x] Keep parsing deterministic, typed, and independently testable.

## Phase 1 — Foundation

- [ ] Add repository metadata, pinned dependencies, and local/deployment setup instructions (artifacts exist; clean-install verification remains).
- [x] Add the `src` package and exact output-schema helpers.
- [x] Add a smoke-test layout and deterministic fixture generator.

## Phase 2 — Image pipeline

- [x] Implement pre-OCR quality metrics: resolution, blur, exposure, and glare signals.
- [x] Implement receipt localization with contour/quad scoring.
- [x] Implement cut-off and too-small-in-frame rejection.
- [x] Implement safe perspective rectification with bounded output dimensions.
- [x] Verify geometry stages on synthetic perspective fixtures.

## Phase 3 — OCR and trustworthy extraction

- [x] Implement OCR token/line extraction with a clear failure path when Tesseract is unavailable.
- [x] Implement deterministic merchant, date, total, and currency candidate ranking.
- [ ] Calibrate field confidence and conservative review thresholds (awaiting a rights-cleared real-photo set).
- [x] Implement `extract(image_path: str) -> dict` with timing and exception safety.
- [x] Add contract tests for quality failure, valid output types, and parser normalization.

## Phase 4 — Product and evaluation

- [x] Add the Streamlit upload interface using the shared pipeline.
- [x] Add `eval.py --data test/` with accuracy, abstention, and trust score.
- [ ] Add at least 12 labelled real receipt photos and ground truth (requires rights-cleared image collection).
- [ ] Run evaluator and record real metrics; do not fabricate metrics.
- [x] Write `REPORT.md`, including three evidenced failure cases and AI-tool disclosure.

## Phase 5 — Release audit

- [ ] Test clean installation and Streamlit launch (local Streamlit health check passed; clean-environment install remains).
- [ ] Check CPU-only latency/memory expectations and quality-failure messaging (538 ms warm local smoke check and messaging verified; representative memory audit remains).
- [ ] Deploy to a free public host and record the live URL.
- [ ] Final schema, documentation, license, and reproducibility audit.

## Current focus

**Core pipeline complete and locally verified.** `src.pipeline.extract` is now
shared by the web UI and evaluator, with 30 deterministic unit tests covering
schema, quality gates, geometry, parsing, OCR discovery, OCR-unavailable
behavior, and the evaluation harness. A supplied sharp digital receipt now
extracts merchant, date, total, and currency locally; a supplied blurred photo
is correctly refused with an actionable retake message. Next: collect a
rights-cleared real-photo set, calibrate thresholds against it, record genuine
metrics, and complete the clean-install / deployment audit.
