# Utility Bill Decoder — Women Tech Quest 2026 (Build Track)

Reads K-Electric, LESCO and IESCO electricity bills (photos / screenshots / scans, partly Urdu, personal data
hidden by grey boxes) and produces:

* **Level 1** — one schema-exact JSON object per bill → `output/level1.csv`
* **Level 2** — plain-English answers to customer questions, grounded in the bill → `output/level2.csv`

**Runtime:** Python 3.12 (3.11+ works). Tested on Python 3.12.3.

## Setup

Windows (PowerShell):
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then edit .env and set API_KEY
```
macOS / Linux:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env and set API_KEY
```

**Tesseract OCR (optional but recommended).** OCR text is extra evidence for the model. Without it the program
still runs (vision-only) and prints one warning. Windows: install Tesseract (UB-Mannheim build), then set
`TESSERACT_CMD` in `.env` if it is not on PATH. Linux: `sudo apt install tesseract-ocr`. Urdu OCR (`urd`
traineddata) is used automatically if installed; English-only is fine.

## Configuration (`.env`)

| Variable | Meaning |
|---|---|
| `MODEL_PROVIDER` | `google`, `anthropic` or `openai` |
| `MODEL_NAME` | one of the permitted models (default `gemini-3.5-flash-lite`) |
| `API_KEY` | provider API key (never commit it) |
| `FALLBACK_MODEL_NAME` | optional second *permitted* model of the same provider, used only if the first returns "not found" |
| `LLM_TIMEOUT`, `LLM_MAX_RETRIES`, `LLM_MIN_INTERVAL`, `LLM_TEMPERATURE` | request timeout, bounded retries, spacing between calls (free tier = 15 req/min), optional temperature |
| `TESSERACT_CMD` | optional path to the Tesseract binary |

Code refuses to start with any model outside the permitted list.

## Run

```bash
# 1. confirm the key and model work (one tiny request)
python main.py check-api

# 2. generate both CSVs for the test set
python main.py run --bills-dir test/bills \
    --level1-template test/csv/level1.csv --level2-template test/csv/level2.csv --out-dir output

# 3. audit the result against the templates
python main.py audit --out-dir output \
    --level1-template test/csv/level1.csv --level2-template test/csv/level2.csv

# 4. build and verify submission.zip (checks audit, secrets, 15 MB limit)
python package_submission.py --output-dir output \
    --level1-template test/csv/level1.csv --level2-template test/csv/level2.csv [--demo demo.mp4]
```

Re-running `run` resumes from `.cache/` (per-bill extraction and answers are checkpointed; failed answers are
retried). Use `--fresh` to ignore the cache, `--only KESC_0008 ...` to process specific bills.

**Validate on the five available training bills** (needs a live key; bill images are in `train/bills/`):
```bash
python main.py run --bills-dir train/bills --level2-template tests/sample_level2.csv --out-dir output_train
python tests/compare_reference.py --level1-csv output_train/level1.csv
```
`tests/reference/KESC_0008.json` is the official answer from the guide. The other reference files are the
author's own reading of the images (ambiguous fields left out), so treat differences there as "check the bill".

**Unit tests** (no API key needed; the model is simulated):
```powershell
$env:TRAIN_BILLS_DIR = "train\bills"
python -m unittest discover -s tests -v
```
macOS / Linux:
```bash
TRAIN_BILLS_DIR=train/bills python -m unittest discover -s tests -v
```

## Local web interface

Install dependencies as above, then start the local upload page:

```powershell
python web_app.py
```

Open **http://127.0.0.1:5000** in your browser. Upload up to 20 PNG/JPG/JPEG bill images (20 MB each);
the image filename without its extension becomes its bill ID. Optionally upload a questions CSV with
`bill_id`, `question_id`, and `question` columns (the page provides a downloadable template). Its bill IDs
must match the uploaded image filenames. When processing finishes, download the Level 1 CSV, optional
Level 2 CSV, and run report from the page.

The web server binds to `127.0.0.1` only. Uploaded images and cache files are removed after processing;
generated results remain under `web_runs/` until you remove them. Stop the server with **Ctrl+C**.

## Architecture

| File | Role |
|---|---|
| `main.py` | CLI (`check-api`, `run`, `audit`), orchestration, per-bill error handling, completeness report |
| `bill_processor.py` | image loading (EXIF fix), OCR preprocessing (upscale + deskew), Tesseract, extraction and Q&A pipelines, caching |
| `llm_client.py` | REST client for the permitted models only; retries with backoff, rate pacing, key redaction |
| `prompts.py` | prompt text |
| `schemas.py` | Level 1 schema, normalisation (numbers, `CR` credits, dates, enums), validation, PII scrub |
| `calculations.py` | safe expression evaluator + helpers (percent, mean, counts, …) used for every Level 2 number |
| `csv_writer.py` | CSV read/write with the `csv` module, template-preserving, output audit |
| `package_submission.py` | builds and verifies the ZIP |

**Pipeline.** (1) The original bill image (orientation-corrected, resized/upscaled) is sent to the model with OCR
text as supporting evidence. One call returns the Level 1 object plus the usage/billing history and the
transcribed charge table. (2) Python normalises and validates everything: numeric strings → numbers, `38209CR` →
`-38209`, subsidies negative, zero/blank lines dropped, enums enforced, dates → ISO, unreadable → `null`.
(3) Printed totals are cross-checked (charge lines vs subtotal, charges + taxes vs current bill). If something does
not add up, **one** bounded re-check call asks the model to re-inspect the image; the result is accepted only if it
reduces the problems, and printed values are never "corrected" by arithmetic. (4) For Level 2 the model writes
each answer with `{placeholders}` plus small expressions (`pct(total_taxes, current_bill)`,
`count_gt(units_prev, 200)`); Python evaluates them with a restricted evaluator, so arithmetic is never done "in
the model's head". Failed calculations get one retry, then a truthful "could not be answered" statement — never a
fabricated number.

**Interpretation decisions** (documented because bills are ambiguous):
* Several meter-register rows (peak/off-peak, import/export): the **first** consumption row is used for
  `previous_reading`/`current_reading`; rows are not summed.
* Summary-only layouts (LESCO/IESCO print "Total Electricity Charges / Subsidies / Net Electricity Charges /
  Taxes"): total → `energy`, subsidies → negative `subsidy`, separately printed FPA → `fpa`; `total_charges` is the
  printed net/subtotal; a single "Taxes" figure → `taxes: []` and `total_taxes`.
* `payable_after_due_date`: highest printed late-payment amount; `null` when only text such as "NOT TO BE PAID"
  is printed.
* "Previous months" in history questions excludes the current month, and answers state when this matters.

**Reliability.** Bounded retries with backoff and `Retry-After`; spacing to respect the free tier; one corrective
retry on invalid JSON; per-bill failures never stop the run (a failed bill gets a schema-valid all-null record and
truthful Level 2 statements); atomic CSV writes; a completeness report and `run_report.json`.

## AI usage

* **Models called by the program at runtime:** `gemini-3.5-flash-lite` (Google AI Studio API, provider `google`) —
  the default in `.env.example`. `gemini-3.1-flash-lite` is configured only as an optional fallback if the primary
  returns "not found". The code also supports `claude-haiku-4-5`, `gpt-4.1-mini` and `gpt-5-mini`, which are used
  **only if** `MODEL_PROVIDER`/`MODEL_NAME` are changed. No other LLM is ever called.
* **APIs / services:** Google AI Studio Gemini API (HTTPS via `requests`). No other external service. No invoice
  or receipt extraction service (Azure prebuilt-invoice, Textract AnalyzeExpense, Mindee) is used.
* **Local tools / libraries:** Tesseract OCR via `pytesseract`; OpenCV (headless) and Pillow for image
  preprocessing; NumPy; Flask for the localhost upload interface; `python-dotenv`; `requests`; Python standard
  library (`csv`, `ast`, `json`, `unittest`).
* **AI coding assistants used to write the code:** Claude (Anthropic, claude.ai chat) generated the initial
  implementation, tests and this README; Copilot in VS Code assisted with the local upload interface and subsequent
  audit improvements.
