import logging
import math
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 20 * 1024 * 1024
READ_CHUNK_SIZE = 64 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}


class OCRLine(BaseModel):
    text: str
    confidence: float


class OCRResponse(BaseModel):
    text: str
    lines: list[OCRLine]
    processing_time_ms: float


class OCRUnavailableError(RuntimeError):
    pass


class OCREngine(Protocol):
    def ocr(self, image_path: str, cls: bool = True) -> Any: ...


app = FastAPI(title="Visa Vista OCR API")
_engine: OCREngine | None = None
_engine_lock = threading.Lock()
_inference_lock = threading.Lock()


def get_ocr_engine() -> OCREngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                try:
                    from paddleocr import PaddleOCR

                    _engine = PaddleOCR(
                        use_angle_cls=True,
                        lang="en",
                        use_gpu=False,
                        show_log=False,
                    )
                except (ImportError, OSError, RuntimeError, ValueError) as exc:
                    logger.exception("Could not initialize PaddleOCR")
                    raise OCRUnavailableError("PaddleOCR could not be initialized.") from exc
    return _engine


def extract_lines(engine: OCREngine, image_path: str) -> list[OCRLine]:
    with _inference_lock:
        result = engine.ocr(image_path, cls=True)

    lines: list[OCRLine] = []
    for page in result or []:
        for item in page or []:
            if not item or len(item) < 2 or not item[1]:
                continue
            text, confidence = item[1]
            if not isinstance(text, str) or not text.strip():
                continue
            score = float(confidence)
            if not math.isfinite(score):
                continue
            lines.append(OCRLine(text=text.strip(), confidence=score))
    return lines


def run_ocr(image_path: str) -> list[OCRLine]:
    return extract_lines(get_ocr_engine(), image_path)


def validate_image(image_path: Path) -> None:
    try:
        with Image.open(image_path) as image:
            if image.format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(
                    status_code=415,
                    detail="The uploaded file content must be a JPG, JPEG, or PNG image.",
                )
            image.verify()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(
            status_code=422,
            detail="The uploaded image is unreadable or corrupted.",
        ) from exc


@app.post("/ocr", response_model=OCRResponse)
async def ocr_image(file: UploadFile = File(...)) -> OCRResponse:
    started_at = time.perf_counter()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        await file.close()
        raise HTTPException(
            status_code=415,
            detail="Unsupported file format. Upload a JPG, JPEG, or PNG image.",
        )

    try:
        with tempfile.TemporaryDirectory(prefix="visa-vista-ocr-") as temp_dir:
            image_path = Path(temp_dir) / f"upload{suffix}"
            size = 0
            with image_path.open("wb") as temporary_file:
                while chunk := await file.read(READ_CHUNK_SIZE):
                    size += len(chunk)
                    if size > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail="The uploaded image exceeds the 20 MB limit.",
                        )
                    temporary_file.write(chunk)

            if size == 0:
                raise HTTPException(status_code=400, detail="The uploaded image is empty.")

            validate_image(image_path)
            try:
                lines = await run_in_threadpool(run_ocr, str(image_path))
            except (OCRUnavailableError, OSError, RuntimeError, ValueError) as exc:
                logger.exception("PaddleOCR failed to process an image")
                raise HTTPException(
                    status_code=503,
                    detail="The OCR engine is unavailable or could not process the image.",
                ) from exc

            return OCRResponse(
                text="\n".join(line.text for line in lines),
                lines=lines,
                processing_time_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
    finally:
        await file.close()
