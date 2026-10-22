import type { JobInfo, JobWarning, StageRecord } from "../api/types";
import type { TFunc } from "../i18n";

/** The stage line of a job: the stage's name, or, in a regen round (`stage_round` > 0), what the round does. */
export function stageLabel(job: Pick<JobInfo, "stage" | "stage_round" | "round_total" | "round_lines">, i18n: { t: TFunc; has: (key: string) => boolean }): string {
  const { t, has } = i18n;
  const stage = job.stage;
  if (!stage) return "";
  const name = has(`stages.${stage}`) ? t(`stages.${stage}`) : stage;
  const rnd = job.stage_round ?? 0;
  if (rnd <= 0) return name;
  const p = { i: rnd, n: Math.max(rnd, job.round_total ?? 0), lines: job.round_lines ?? 0 };
  if (stage === "synth") return t("job.round_synth", p);
  if (stage === "score") return t("job.round_score", p);
  return t("job.round_stage", { ...p, stage: name });
}

/**
 * How much of the current stage's slot of the overall bar is filled. In a regen round the slot belongs to the quality
 * step and fills over the rounds (synth is the first half of a round, score the second), so the bar never shows the
 * job back at the voicing step.
 */
export function stageSlotFraction(job: Pick<JobInfo, "stage" | "stage_round" | "round_total">, own: number | null): number {
  const rnd = job.stage_round ?? 0;
  const total = Math.max(rnd, job.round_total ?? 0);
  if (rnd <= 0 || total <= 0) return own ?? 0;
  const half = 0.5 * (own ?? 0);
  return (rnd - 1 + (job.stage === "score" ? 0.5 + half : half)) / total;
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/** "12 мин", "1 ч 05 мин", "меньше минуты" */
export function fmtDuration(t: TFunc, seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "";
  const s = Math.max(0, Math.round(seconds));
  if (s < 45) return t("time.less_than_minute");
  const totalMin = Math.round(s / 60);
  if (totalMin < 60) return t("time.min", { n: totalMin });
  const h = Math.floor(totalMin / 60);
  const m = totalMin % 60;
  return m === 0 ? t("time.h", { n: h }) : t("time.h_min", { h, m: String(m).padStart(2, "0") });
}

export function fmtGb(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "";
  return v.toFixed(digits);
}

export function fmtMb(mb: number): string {
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
}

export function fmtMinutes(m: number | null | undefined): string {
  if (m === null || m === undefined || !Number.isFinite(m)) return "";
  return m < 10 ? m.toFixed(1) : String(Math.round(m));
}

export function fmtPercent(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "";
  return `${Math.round(v * 100)}%`;
}

export function fmtNumber(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "";
  return v.toFixed(digits);
}

export function fmtDate(iso: string | null | undefined, lang: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(lang === "en" ? "en-GB" : "ru-RU", { dateStyle: "medium", timeStyle: "short" });
}

export function basename(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] ?? path;
}

export function dirname(path: string): string {
  const i = Math.max(path.lastIndexOf("\\"), path.lastIndexOf("/"));
  return i > 0 ? path.slice(0, i) : path;
}

/** A stage is done when it has a `finished` stamp and no error (work/stage_status.json). */
export function stageDone(status: Record<string, StageRecord> | undefined, name: string): boolean {
  const r = status?.[name];
  return !!r && !!r.finished && !r.error;
}

export function toNumber(v: unknown, fallback = 0): number {
  const n = typeof v === "number" ? v : typeof v === "string" ? Number(v) : NaN;
  return Number.isFinite(n) ? n : fallback;
}

export interface Facet {
  id: string;
  name: string;
}

/** Facets may arrive as strings or as objects; normalize to {id, name}. */
export function normalizeFacets(items: unknown[] | undefined): Facet[] {
  const out: Facet[] = [];
  for (const it of items ?? []) {
    if (typeof it === "string" || typeof it === "number") {
      out.push({ id: String(it), name: String(it) });
    } else if (it && typeof it === "object") {
      const o = it as Record<string, unknown>;
      const id = o.id ?? o.speaker_id ?? o.scene ?? o.value ?? o.name;
      if (id === undefined || id === null) continue;
      out.push({ id: String(id), name: String(o.name ?? o.label ?? id) });
    }
  }
  return out;
}

/** Row of `result.files` of a download / install job: `{id, name, filename, total, done, state, verified, skipped, error}` (bytes). */
export interface FileProgress {
  id: string;
  name: string;
  /** environments have no bytes: only a state */
  isEnv: boolean;
  done: number | null;
  total: number | null;
  state: string;
  fraction: number | null;
  verified: boolean | null;
  skipped: boolean;
  error: string | null;
}

/** Normalize `result.files`: an array of objects (or a map name -> object, older shape). */
export function normalizeFiles(files: unknown): FileProgress[] {
  const rows: [string, Record<string, unknown>][] = [];
  if (Array.isArray(files)) {
    files.forEach((f, i) => {
      if (f && typeof f === "object") {
        const o = f as Record<string, unknown>;
        rows.push([String(o.id ?? o.name ?? o.filename ?? o.file ?? o.path ?? `#${i + 1}`), o]);
      } else if (typeof f === "string") {
        rows.push([f, { name: f }]);
      }
    });
  } else if (files && typeof files === "object") {
    for (const [k, v] of Object.entries(files as Record<string, unknown>)) {
      rows.push([k, v && typeof v === "object" ? { name: k, ...(v as Record<string, unknown>) } : { name: k, progress: v }]);
    }
  }
  return rows.map(([id, o]) => {
    const done = pickNum(o, ["done", "downloaded", "bytes_done", "done_bytes", "current"]);
    const total = pickNum(o, ["total", "size", "bytes_total", "total_bytes"]);
    let fraction = pickNum(o, ["progress", "fraction", "ratio"]);
    if (fraction === null && done !== null && total) fraction = done / total;
    if (fraction !== null && fraction > 1) fraction = fraction / 100;
    const skipped = o.skipped === true;
    const state = String(o.state ?? o.status ?? (skipped ? "done" : fraction !== null && fraction >= 1 ? "done" : "pending"));
    return {
      id,
      name: String(o.name ?? o.filename ?? o.file ?? id),
      isEnv: id.startsWith("env:"),
      done,
      total: total && total > 0 ? total : null,
      state,
      fraction: fraction === null ? null : clamp(fraction, 0, 1),
      verified: typeof o.verified === "boolean" ? o.verified : null,
      skipped,
      error: typeof o.error === "string" && o.error ? o.error : null,
    };
  });
}

/** `result.warnings` of a job as a clean list: only objects with a string `kind`, every other field kept when it is a string. */
export function jobWarnings(result: unknown): JobWarning[] {
  const raw = result && typeof result === "object" ? (result as Record<string, unknown>).warnings : null;
  if (!Array.isArray(raw)) return [];
  const out: JobWarning[] = [];
  for (const w of raw) {
    if (!w || typeof w !== "object" || typeof (w as Record<string, unknown>).kind !== "string") continue;
    const o = w as Record<string, unknown>;
    const item: Record<string, string> = {};
    for (const [k, v] of Object.entries(o)) if (typeof v === "string") item[k] = v;
    out.push(item as unknown as JobWarning);
  }
  return out;
}

/** "12.3 MB", "1.4 GB" from bytes. */
export function fmtBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return "";
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(bytes >= 1e8 ? 0 : 1)} MB`;
  if (bytes >= 1e3) return `${Math.round(bytes / 1e3)} KB`;
  return `${Math.round(bytes)} B`;
}

function pickNum(o: Record<string, unknown>, keys: string[]): number | null {
  for (const k of keys) {
    const v = o[k];
    if (typeof v === "number" && Number.isFinite(v)) return v;
  }
  return null;
}

/** Make a readable fallback title out of an id such as "app_manual" or "TROUBLESHOOTING". */
export function prettifyId(id: string): string {
  const s = id.replace(/[_-]+/g, " ").trim();
  return s ? s.charAt(0).toUpperCase() + s.slice(1).toLowerCase() : id;
}
