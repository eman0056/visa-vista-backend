import asyncio

import pytest
from fastapi import HTTPException

import main as intake_main


class FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def test_save_ocr_results_uses_one_atomic_rpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []

    async def fake_supabase_request(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeResponse(204)

    monkeypatch.setattr(intake_main, "supabase_request", fake_supabase_request)
    document = {"id": "document-id"}
    pages = [{"document_id": "document-id"}]
    fields = [{"id": "field-id"}]
    flags = [{"id": "flag-id"}]

    asyncio.run(
        intake_main.save_ocr_results(
            None, "https://example.supabase.co", "service-key",
            document, pages, fields, flags,
        )
    )

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[1:5] == (
        "https://example.supabase.co",
        "service-key",
        "POST",
        "/rest/v1/rpc/save_ocr_document",
    )
    assert kwargs["json_body"] == {
        "p_document": document,
        "p_pages": pages,
        "p_fields": fields,
        "p_flags": flags,
    }


def test_save_ocr_results_reports_rpc_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_supabase_request(*args, **kwargs):
        return FakeResponse(400)

    monkeypatch.setattr(intake_main, "supabase_request", fake_supabase_request)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(intake_main.save_ocr_results(None, "url", "key", {}, [], [], []))

    assert exc_info.value.status_code == 502
