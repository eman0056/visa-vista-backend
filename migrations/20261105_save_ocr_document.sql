CREATE OR REPLACE FUNCTION public.save_ocr_document(
    p_document jsonb,
    p_pages jsonb,
    p_fields jsonb,
    p_flags jsonb
)
RETURNS void
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $function$
BEGIN
    IF jsonb_typeof(p_document) IS DISTINCT FROM 'object'
       OR jsonb_typeof(p_pages) IS DISTINCT FROM 'array'
       OR jsonb_typeof(p_fields) IS DISTINCT FROM 'array'
       OR jsonb_typeof(p_flags) IS DISTINCT FROM 'array' THEN
        RAISE EXCEPTION 'Invalid OCR result payload'
            USING ERRCODE = '22023';
    END IF;

    INSERT INTO public.documents (
        id, case_id, doc_type, title, received_at, storage_path
    )
    SELECT
        document_row.id,
        document_row.case_id,
        document_row.doc_type,
        document_row.title,
        document_row.received_at,
        document_row.storage_path
    FROM jsonb_populate_record(
        NULL::public.documents,
        p_document
    ) AS document_row;

    INSERT INTO public.document_pages (
        document_id, page_number, label, image_path
    )
    SELECT
        page_row.document_id,
        page_row.page_number,
        page_row.label,
        page_row.image_path
    FROM jsonb_populate_recordset(
        NULL::public.document_pages,
        p_pages
    ) AS page_row;

    INSERT INTO public.extracted_fields (
        id, case_id, document_id, section, "key", label, value, confidence,
        source_page, source_region, flag_reason, edited, approved, approved_by
    )
    SELECT
        field_row.id,
        field_row.case_id,
        field_row.document_id,
        field_row.section,
        field_row."key",
        field_row.label,
        field_row.value,
        field_row.confidence,
        field_row.source_page,
        field_row.source_region,
        field_row.flag_reason,
        field_row.edited,
        field_row.approved,
        field_row.approved_by
    FROM jsonb_populate_recordset(
        NULL::public.extracted_fields,
        p_fields
    ) AS field_row;

    INSERT INTO public.validation_flags (
        id, case_id, document_id, field_id, severity, message, created_at, resolved
    )
    SELECT
        flag_row.id,
        flag_row.case_id,
        flag_row.document_id,
        flag_row.field_id,
        flag_row.severity,
        flag_row.message,
        flag_row.created_at,
        flag_row.resolved
    FROM jsonb_populate_recordset(
        NULL::public.validation_flags,
        p_flags
    ) AS flag_row;
END;
$function$;

REVOKE ALL ON FUNCTION public.save_ocr_document(jsonb, jsonb, jsonb, jsonb)
    FROM PUBLIC;
REVOKE ALL ON FUNCTION public.save_ocr_document(jsonb, jsonb, jsonb, jsonb)
    FROM anon, authenticated;
GRANT EXECUTE ON FUNCTION public.save_ocr_document(jsonb, jsonb, jsonb, jsonb)
    TO service_role;
