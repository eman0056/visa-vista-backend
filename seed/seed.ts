import "dotenv/config";
import { createClient } from "@supabase/supabase-js";
import { randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { buildSeed } from "../../frontend/src/data/mockData";

const { SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, DEMO_PASSWORD } = process.env;
if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY || !DEMO_PASSWORD) {
  throw new Error("Set SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY and DEMO_PASSWORD in backend/.env");
}

const sb = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  auth: { autoRefreshToken: false, persistSession: false },
});

const BUCKET = "case-documents";
const DOCS_DIR = path.resolve(process.cwd(), "../frontend/src/assets/docs");
const seed = buildSeed();

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, ".").replace(/^\.|\.$/g, "");
const lookup = (map: Map<string, string>, key?: string | null) => (key ? (map.get(key) ?? null) : null);

async function insert(table: string, rows: object[]) {
  for (let i = 0; i < rows.length; i += 200) {
    const { error } = await sb.from(table).insert(rows.slice(i, i + 200));
    if (error) throw new Error(`${table}: ${error.message}`);
  }
  console.log(`  ${table}: ${rows.length} rows`);
}

async function main() {
  const { count, error: countErr } = await sb.from("cases").select("*", { count: "exact", head: true });
  if (countErr) throw new Error(`cases: ${countErr.message} (did you run the migration?)`);
  if (count) throw new Error("cases is not empty. Seed already ran; audit_logs is append-only so it cannot re-run.");

  console.log("Creating demo users...");
  const roleByName = new Map<string, "Paralegal" | "Attorney">();
  const note = (name?: string, role?: string) => {
    if (!name) return;
    const isAttorney = role === "Attorney" || roleByName.get(name) === "Attorney";
    roleByName.set(name, isAttorney ? "Attorney" : "Paralegal");
  };
  seed.auditLogs.forEach((a) => note(a.approved_by, a.role));
  seed.fields.forEach((f) => note(f.approved_by));
  seed.deadlines.forEach((d) => note(d.signed_off_by, "Attorney"));

  const { data: existing } = await sb.auth.admin.listUsers({ page: 1, perPage: 1000 });
  const userIdByName = new Map<string, string>();
  for (const [name, role] of roleByName) {
    const email = `${slug(name)}@demo.example.com`;
    let id = existing?.users.find((u) => u.email === email)?.id;
    if (!id) {
      const { data, error } = await sb.auth.admin.createUser({
        email,
        password: DEMO_PASSWORD,
        email_confirm: true,
        user_metadata: { name },
      });
      if (error || !data.user) throw new Error(`createUser ${email}: ${error?.message}`);
      id = data.user.id;
    }
    userIdByName.set(name, id);
    const { error } = await sb.from("profiles").update({ role, name }).eq("id", id);
    if (error) throw new Error(`profiles ${email}: ${error.message}`);
    console.log(`  ${role.padEnd(9)} ${email}`);
  }

  console.log("Uploading sample scans...");
  const uploaded = new Set<string>();
  const imagePath = (img: string) => {
    if (img.startsWith("data:")) throw new Error("Image was inlined as data URI; cannot map to a file.");
    return `demo/${path.parse(img).name}.jpg`;
  };
  for (const doc of seed.documents) {
    for (const page of doc.pages) {
      const dest = imagePath(page.image);
      if (uploaded.has(dest)) continue;
      const file = path.join(DOCS_DIR, path.basename(dest));
      if (!existsSync(file)) throw new Error(`Missing image file: ${file}`);
      const { error } = await sb.storage.from(BUCKET).upload(dest, readFileSync(file), {
        contentType: "image/jpeg",
        upsert: true,
      });
      if (error) throw new Error(`upload ${dest}: ${error.message}`);
      uploaded.add(dest);
    }
  }
  console.log(`  ${uploaded.size} images in bucket "${BUCKET}"`);

  const caseIds = new Map(seed.cases.map((c) => [c.id, randomUUID()]));
  const docIds = new Map(seed.documents.map((d) => [d.id, randomUUID()]));
  const fieldIds = new Map(seed.fields.map((f) => [f.id, randomUUID()]));

  console.log("Inserting rows...");
  await insert(
    "cases",
    seed.cases.map((c) => ({
      id: caseIds.get(c.id),
      case_ref: c.id,
      client_name: c.client_name,
      case_type: c.case_type,
      status: c.status,
      attorney: c.attorney,
      opened_at: c.opened_at,
      last_updated: c.last_updated,
      summary: c.summary,
    })),
  );

  await insert(
    "documents",
    seed.documents.map((d) => ({
      id: docIds.get(d.id),
      case_id: caseIds.get(d.case_id),
      doc_type: d.doc_type,
      title: d.title,
      received_at: d.received_at,
      storage_path: d.pages[0] ? imagePath(d.pages[0].image) : null,
    })),
  );

  await insert(
    "document_pages",
    seed.documents.flatMap((d) =>
      d.pages.map((p) => ({
        document_id: docIds.get(d.id),
        page_number: p.page_number,
        label: p.label,
        image_path: imagePath(p.image),
      })),
    ),
  );

  await insert(
    "extracted_fields",
    seed.fields.map((f) => ({
      id: fieldIds.get(f.id),
      case_id: caseIds.get(f.case_id),
      document_id: docIds.get(f.document_id),
      section: f.section,
      key: f.key,
      label: f.label,
      value: f.value,
      confidence: f.confidence,
      source_page: f.source_page,
      source_region: f.source_region,
      flag_reason: f.flag_reason ?? null,
      edited: !!f.edited,
      approved: !!f.approved,
      approved_by: lookup(userIdByName, f.approved_by),
    })),
  );

  await insert(
    "validation_flags",
    seed.flags.map((f) => ({
      case_id: caseIds.get(f.case_id),
      document_id: lookup(docIds, f.document_id),
      field_id: lookup(fieldIds, f.field_id),
      severity: f.severity,
      message: f.message,
      created_at: f.created_at,
      resolved: !!f.resolved,
    })),
  );

  await insert(
    "deadlines",
    seed.deadlines.map((d) => ({
      case_id: caseIds.get(d.case_id),
      label: d.label,
      kind: d.kind,
      date: d.date.slice(0, 10),
      certainty: d.certainty,
      source_document_id: lookup(docIds, d.source_document_id),
      source_field_id: lookup(fieldIds, d.source_field_id),
      notes: d.notes ?? null,
      signed_off_by: lookup(userIdByName, d.signed_off_by),
      signed_off_at: d.signed_off_at ?? null,
    })),
  );

  await insert(
    "audit_logs",
    seed.auditLogs.map((a) => {
      const userId = userIdByName.get(a.approved_by);
      if (!userId) throw new Error(`No user for audit entry by "${a.approved_by}"`);
      return {
        case_id: caseIds.get(a.case_id),
        document_id: lookup(docIds, a.document_id),
        field: a.field,
        action: a.action,
        original_value: a.original_value,
        corrected_value: a.corrected_value,
        approved_by: userId,
        role: a.role,
        timestamp: a.timestamp,
        reason: a.reason ?? null,
      };
    }),
  );

  console.log("\nDone. Demo users share the DEMO_PASSWORD from your .env.");
}

main().catch((e) => {
  console.error("\nSeed failed:", e.message ?? e);
  process.exit(1);
});
