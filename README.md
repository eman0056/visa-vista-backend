# Visa Vista OCR API

The backend OCR service accepts JPG, JPEG, and PNG images up to 20 MB and
processes them locally with PaddleOCR. Uploaded files are held in a temporary
directory only while the request is being processed and are removed afterward.

## Setup

Python 3.12 on Linux x86_64 is supported by the pinned CPU PaddlePaddle wheel.
From `backend/`, create and activate a virtual environment, then install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

The first OCR request initializes PaddleOCR and may download its recognition
models. The service does not store uploaded documents permanently.

## Test

```bash
python -m pip install -r requirements-dev.txt
pytest
```

`POST /ocr` expects one multipart field named `file`. Successful responses
contain concatenated text, per-line confidence scores, and `processing_time_ms`.
