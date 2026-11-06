# Visa Vista OCR API

FastAPI backend for immigration document OCR and structured field extraction using OpenAI Vision.

## Supported formats

- JPG / JPEG
- PNG
- PDF
- Maximum upload size: 20 MB

## API endpoints

- `GET /health` — health check
- `POST /intake` — authenticate the Supabase user, retrieve a private uploaded object,
  run Vision extraction and validation, and save the result to the existing Supabase tables.

The intake endpoint accepts JSON containing `case_ref`, `storage_path`, `filename`, and
`mime_type`; it requires the signed-in user's `Authorization: Bearer <access-token>` header.
The browser uploads directly to the private `case-documents` bucket before calling this endpoint.

## Configuration

Set these server-side environment variables in the backend Vercel project:

- `SUPABASE_URL` — Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` — server-only Supabase service-role key
- `OPENAI_API_KEY` — required
- `OPENAI_MODEL` — optional; defaults to `gpt-4o-mini`
- `FRONTEND_ORIGINS` — comma-separated exact frontend origins allowed by CORS

The frontend Vercel project also needs `VITE_SUPABASE_URL`, `VITE_SUPABASE_PUBLISHABLE_KEY`,
and `VITE_OCR_API_URL` (the deployed backend origin). Never put service-role or OpenAI keys in
frontend variables or commit API keys or `.env` files.

## Database deployment

Before deploying the backend, apply `migrations/20261105_save_ocr_document.sql` to the Supabase
project (for example, through the Supabase SQL Editor). The intake endpoint uses the
`save_ocr_document` RPC to insert the document, page, extracted fields, and validation flags in
one PostgreSQL transaction; a failure rolls the entire RPC back. The migration grants RPC
execution only to `service_role`.

The migration expects the listed columns and existing public tables to match the OCR payloads.
Verify those columns and foreign-key relationships against the deployed schema before applying
the migration. Deploying the backend without applying the migration will cause document saves to
fail.

## Deployment

Set the backend Vercel project's Root Directory to `backend`. Vercel's Python runtime should
discover the FastAPI `app` in `main.py`; `requirements.txt` is in the same project root. Add the
server-side variables above, including the frontend's exact production origin in
`FRONTEND_ORIGINS`, then deploy. No local filesystem storage or OCR model is used.

Keep `case-documents` private, set its size limit to 20 MB, and restrict accepted MIME types to
PDF, JPEG, and PNG. Its authenticated upload policy must permit users to create objects only in
their own first-level user-ID folder; case access is additionally checked by the backend using
the caller's JWT and the `cases` table's RLS policy. Ensure the caller can read the requested
case through RLS. The backend downloads with its server-only service-role key and does not create
public URLs.
