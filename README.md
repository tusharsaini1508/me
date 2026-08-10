# Receipt Capture & Extraction

A CPU-only receipt-reading prototype designed to return structured data only when the image and extracted fields are trustworthy enough. It accepts a phone photo, checks capture quality, locates and rectifies the receipt, runs local OCR, and reports field-level confidence with an explicit review signal.

The project is being built for the accompanying computer-vision take-home brief. It deliberately favors calibrated abstention over confident guesses: when a photo cannot be read reliably, the result should explain how to retake it rather than invent a value.

## Required interface

The headless entry point is:

```python
from src.pipeline import extract

result = extract("path/to/receipt.jpg")
```

It returns this schema:

```json
{
  "quality": {
    "pass": true,
    "issues": []
  },
  "fields": {
    "merchant_name": {"value": "Cafe Nine", "confidence": 0.91, "needs_review": false},
    "transaction_date": {"value": "2026-03-14", "confidence": 0.88, "needs_review": false},
    "total_amount": {"value": 480.0, "confidence": 0.42, "needs_review": true},
    "currency": {"value": "INR", "confidence": 0.95, "needs_review": false}
  },
  "processing_ms": 1840
}
```

If `quality.pass` is `false`, `quality.issues` is non-empty and field values can be `null`. `confidence` remains a number in the range `[0, 1]`, and `needs_review` makes uncertainty visible to callers.

## Intended pipeline

1. Validate that the image can be decoded and has sufficient usable resolution.
2. Gate poor captures using fast CPU checks for blur, exposure, glare, document size, and edge cut-off.
3. Detect the largest plausible quadrilateral receipt contour and rectify it with a perspective transform.
4. Preprocess the rectified image and use local Tesseract OCR; no hosted OCR or vision API is used.
5. Parse the merchant, date, total, and ISO 4217 currency using receipt-aware rules and OCR evidence.
6. Calibrate each field's confidence and mark uncertain values for review.

The pipeline is intentionally bounded to lightweight, deterministic OpenCV and Tesseract operations so it can run on a CPU-only Streamlit deployment. The implementation should constrain oversized uploads before expensive operations and use linear scans over OCR tokens where practical.

## Repository layout

```text
app.py                 Streamlit upload and review interface
src/pipeline.py        Headless bounded capture/OCR `extract(image_path)` implementation
src/schema.py          Exact response-schema constructors and invariants
src/parsing.py         Typed, deterministic OCR token/line parsing rules
eval.py                Evaluation CLI
tests/                 Contract, geometry, parser, evaluator, and fixture tests
test/                  Labelled real receipt images and ground truth (to be added)
REPORT.md              Current threshold rationale and honest validation status
requirements.txt       Pinned Python dependencies
packages.txt           Streamlit Cloud Linux packages
```

The rights-cleared test data, measured metrics, and deployment URL are not yet
claimed by this implementation. `REPORT.md` documents the current verified
state and deliberately does not invent those missing results.

## Local setup

Use Python 3.11 for the verified local environment. Tesseract itself is a system dependency in addition to the Python package.

On Windows, install the Tesseract OCR binary and create an isolated environment
with the pinned packages. The pipeline checks `TESSERACT_CMD`, `PATH`, and the
standard `C:\Program Files\Tesseract-OCR\tesseract.exe` install location, so a
new terminal session is not required after the normal installer completes.
Verify the engine before launching the app (use the absolute path if the
installer has not refreshed this terminal's `PATH` yet):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
& 'C:\Program Files\Tesseract-OCR\tesseract.exe' --version
```

On Debian/Ubuntu, install the packages listed in `packages.txt` before installing Python dependencies:

```bash
sudo apt-get update
sudo apt-get install -y $(tr '\n' ' ' < packages.txt)
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

The interface is expected to show the capture-quality result, a rectified receipt preview when available, the extracted JSON, and review flags. Do not use receipt images containing personal information unless you have permission to process them.

## Run evaluation

Once a labelled test set and ground-truth files have been added under `test/`, run:

```bash
python eval.py --data test/
```

The evaluator is expected to report per-field accuracy, abstention rate, and the take-home trust score. Metrics should be generated from that command and recorded in `REPORT.md`; this README intentionally does not invent results.

## Run deterministic checks

The unit suite needs no real receipts or installed Tesseract binary: OCR is
mocked at the pipeline boundary, while image quality, geometry, response
schema, parsing, and evaluator behavior are checked with deterministic
synthetic scenes.

```bash
python -m unittest discover -s tests -v
```

To inspect the generated smoke inputs without checking them into the real
evaluation corpus:

```bash
python -c "from tests.fixtures.synthetic_receipts import write_fixture_set; write_fixture_set('tmp/smoke-fixtures')"
```

Synthetic scenes are only regression fixtures. They do not count toward the
required rights-cleared, labelled receipt-photo evaluation set.

## Streamlit Community Cloud deployment

1. Push the repository to a public GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and select `app.py` as the entry point.
3. Keep `requirements.txt` and `packages.txt` at the repository root so the platform installs the pinned Python and Linux dependencies during the build.
4. In **Advanced settings**, select Python 3.11 to match the verified local environment (Community Cloud otherwise defaults to Python 3.12).

**Note on OCR in Cloud Environments:**  
Streamlit Cloud deployments have system-level library constraints that prevent local Tesseract OCR installation. The app is designed with graceful degradation:
- Users can upload receipt images, PDFs, and Word documents
- The pipeline validates and processes images normally
- If Tesseract is unavailable, the app displays a clear message ("Text recognition is not configured on this machine") with setup instructions
- Users can still test image quality gates and document upload flows
- For full OCR functionality, run the app locally with Tesseract installed

Local testing with OCR:
```bash
# On your local machine with Tesseract installed:
streamlit run app.py
```

The app has no need for API keys or hosted vision services. If Streamlit secrets are later introduced for unrelated configuration, keep them in `.streamlit/secrets.toml`, which is ignored by Git.

## License

This project is available under the [MIT License](LICENSE).
