import re
from datetime import datetime, timezone

REQUIRED = {
    "Passport": ["full_name", "date_of_birth", "passport_number", "expiry_date"],
    "Visa": ["full_name", "visa_type", "visa_number", "expiry_date"],
    "Birth Certificate": ["full_name", "date_of_birth", "place_of_birth"],
    "I-94": ["full_name", "i94_number", "admit_until"],
    "USCIS Notice": ["receipt_number", "notice_date", "applicant_name"],
}


def mrz_digit(s):
    w = [7, 3, 1]
    t = 0
    for i, c in enumerate(s):
        if c == "<":
            v = 0
        elif c.isdigit():
            v = int(c)
        elif "A" <= c <= "Z":
            v = ord(c) - 55
        else:
            return None
        t += v * w[i % 3]
    return t % 10


def validate(llm_out, lines, case_ref, ms=None):
    fields = llm_out.get("fields") or []
    doc_type = llm_out.get("document_type") or "Unknown document type"
    conf = float(llm_out.get("classification_confidence") or 0)
    flags = []

    def add(rule, sev, msg, key=None):
        flags.append({"rule_type": rule, "severity": sev, "message": msg, "field_key": key})

    # 1. Classification
    if doc_type == "Unknown document type":
        add("other", "high", "Unknown document type: human review required before any filing.")
    elif conf < 0.8:
        add("other", "medium", f"Low classification confidence ({conf:.2f}).")

    # 2. Scan quality (MRZ lines ignored)
    low = [l for l in lines if l["confidence"] < 0.85 and "<" not in l["text"]]
    if low:
        add("low_scan_quality", "high" if len(low) >= 3 else "medium",
            f"{len(low)} OCR line(s) below 85% confidence.")

    # 3. Passport MRZ check digits
    mrz = {"checked": False, "results": []}
    if doc_type == "Passport":
        ocr_mrz = [re.sub(r"\s", "", l["text"]) for l in lines
                   if "<" in l["text"] and len(re.sub(r"\s", "", l["text"])) >= 25]
        l2 = re.sub(r"\s", "", (llm_out.get("mrz") or {}).get("line2") or (ocr_mrz[1] if len(ocr_mrz) > 1 else ""))
        if l2:
            mrz["checked"] = True
            if len(l2) != 44:
                add("low_scan_quality", "medium", f"MRZ line 2 has {len(l2)} characters, expected 44.")
            p = l2.ljust(44, "<")
            checks = [("Document number", p[0:9], p[9]),
                      ("Date of birth", p[13:19], p[19]),
                      ("Expiry date", p[21:27], p[27])]
            if len(l2) == 44:
                checks.append(("Composite", p[0:10] + p[13:20] + p[21:43], p[43]))
            for name, value, found in checks:
                exp = mrz_digit(value)
                ok = exp is not None and str(exp) == found
                mrz["results"].append({"check": name, "expected": exp, "found": found, "ok": ok})
                if not ok:
                    add("other", "high", f"MRZ check digit failed for {name} (expected {exp}, found {found}).")
        else:
            add("missing_document", "medium", "Passport MRZ lines were not found in the scan.")

    # 4. Ambiguous dates
    for f in fields:
        raw = str(f.get("raw_text") or f.get("value") or "")
        m = re.search(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b", raw)
        if m and int(m[1]) <= 12 and int(m[2]) <= 12 and m[1] != m[2]:
            add("ambiguous_date", "high",
                f'{f.get("label") or f.get("key")}: "{m[0]}" could be DD/MM or MM/DD.', f.get("key"))

    # 5. Required fields
    for key in REQUIRED.get(doc_type, []):
        f = next((x for x in fields if x.get("key") == key), None)
        if not f or not str(f.get("value") or "").strip():
            add("other", "medium", f"Required field missing: {key}.", key)

    # 6. Model warnings
    for w in llm_out.get("warnings") or []:
        add("other", "low", f"Model note: {w}")

    avg = sum(l["confidence"] for l in lines) / len(lines) if lines else None
    return {
        "case_ref": case_ref,
        "document_type": doc_type,
        "classification_confidence": conf,
        "status": "needs_review",
        "review_required": True,
        "fields": fields,
        "mrz_check": mrz,
        "flags": flags,
        "ocr": {"source": "paddleocr", "avg_confidence": avg,
                "low_confidence_lines": len(low), "processing_time_ms": ms},
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
