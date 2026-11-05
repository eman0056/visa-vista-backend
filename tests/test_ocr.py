from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main


class FakeOCREngine:
    def __init__(self) -> None:
        self.image_paths: list[Path] = []

    def ocr(self, image_path: str, cls: bool = True) -> list:
        self.image_paths.append(Path(image_path))
        return [
            [
                [
                    [[0, 0], [1, 0], [1, 1], [0, 1]],
                    ("FICTIONAL PASSPORT", 0.98),
                ],
                [
                    [[0, 2], [1, 2], [1, 3], [0, 3]],
                    ("SAMPLE PERSON", 0.95),
                ],
            ],
        ]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    engine = FakeOCREngine()
    monkeypatch.setattr(main, "get_ocr_engine", lambda: engine)
    return TestClient(main.app)


def image_bytes(image_format: str = "PNG") -> bytes:
    image = Image.new("RGB", (160, 80), color="white")
    output = BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def test_ocr_returns_text_lines_confidence_and_processing_time(
    client: TestClient,
) -> None:
    response = client.post(
        "/ocr",
        files={"file": ("fictional-passport.png", image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "FICTIONAL PASSPORT\nSAMPLE PERSON"
    assert body["lines"] == [
        {"text": "FICTIONAL PASSPORT", "confidence": 0.98},
        {"text": "SAMPLE PERSON", "confidence": 0.95},
    ]
    assert body["processing_time_ms"] >= 0


def test_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        "/ocr",
        files={"file": ("document.pdf", b"not an image", "application/pdf")},
    )

    assert response.status_code == 415


def test_rejects_image_with_unsupported_content(client: TestClient) -> None:
    response = client.post(
        "/ocr",
        files={"file": ("image.jpg", b"not a readable image", "image/jpeg")},
    )

    assert response.status_code == 422


def test_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/ocr",
        files={"file": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 400


def test_rejects_file_over_20_mb(client: TestClient) -> None:
    response = client.post(
        "/ocr",
        files={
            "file": (
                "oversized.png",
                b"x" * (main.MAX_FILE_SIZE + 1),
                "image/png",
            )
        },
    )

    assert response.status_code == 413


def test_temporary_image_is_removed_after_ocr(client: TestClient) -> None:
    response = client.post(
        "/ocr",
        files={"file": ("fictional-passport.png", image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    engine = main.get_ocr_engine()
    assert isinstance(engine, FakeOCREngine)
    assert len(engine.image_paths) == 1
    assert not engine.image_paths[0].exists()


def test_blank_image_returns_empty_text_and_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlankOCREngine:
        def ocr(self, image_path: str, cls: bool = True) -> list:
            return [None]

    monkeypatch.setattr(main, "get_ocr_engine", lambda: BlankOCREngine())
    with TestClient(main.app) as test_client:
        response = test_client.post(
            "/ocr",
            files={"file": ("blank.png", image_bytes(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["text"] == ""
    assert response.json()["lines"] == []
