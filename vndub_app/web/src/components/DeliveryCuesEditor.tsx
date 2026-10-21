import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import type { ErrorInfo } from "../api/types";
import { useI18n } from "../i18n";
import { ErrorPanel } from "./ErrorPanel";
import { Button, Spinner, useToast } from "./ui";

// The delivery cue table (configs/delivery_cues.yaml) maps a cue word to emotion / intensity / tempo /
// loudness / events. Its exact nesting is owned by the pipeline, so this editor walks the JSON and
// edits every "leaf" object that has delivery fields, whatever its position in the tree.

type Path = (string | number)[];
type Obj = Record<string, unknown>;

const DELIVERY_FIELDS = ["emotion", "intensity", "tempo", "loudness", "events"] as const;
const EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted", "embarrassed", "other"];
const TEMPOS = ["slow", "normal", "fast"];
const LOUDNESS = ["soft", "normal", "loud"];
const NAME_FIELDS = ["cue", "pattern", "text", "word", "phrase", "match"];
const MAX_ROWS = 400;

interface Row {
  path: Path;
  group: string;
  name: string;
  nameField: string | null;
  node: Obj;
}

function isObj(v: unknown): v is Obj {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function isLeaf(v: unknown): v is Obj {
  return isObj(v) && DELIVERY_FIELDS.some((k) => k in v);
}

function collect(node: unknown, path: Path, out: Row[]): void {
  if (isLeaf(node)) {
    const nameField = NAME_FIELDS.find((f) => typeof node[f] === "string") ?? null;
    const last = path[path.length - 1];
    out.push({
      path,
      group: path.slice(0, -1).join(" / "),
      name: nameField ? String(node[nameField]) : String(last ?? ""),
      nameField,
      node,
    });
    return;
  }
  if (Array.isArray(node)) node.forEach((c, i) => collect(c, [...path, i], out));
  else if (isObj(node)) for (const [k, v] of Object.entries(node)) collect(v, [...path, k], out);
}

function setAt(root: unknown, path: Path, field: string, value: unknown): unknown {
  const copy = structuredClone(root);
  let cur: unknown = copy;
  for (const p of path) cur = (cur as Record<string | number, unknown>)[p];
  (cur as Obj)[field] = value;
  return copy;
}

export function DeliveryCuesEditor() {
  const { t, tOr } = useI18n();
  const toast = useToast();
  const [data, setData] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ErrorInfo | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [filter, setFilter] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setData(await api.deliveryCues());
      setDirty(false);
      setError(null);
    } catch (e) {
      setError(toErrorInfo(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const rows = useMemo(() => {
    const out: Row[] = [];
    collect(data, [], out);
    return out;
  }, [data]);

  const f = filter.trim().toLowerCase();
  const shown = rows.filter((r) => !f || `${r.group} ${r.name}`.toLowerCase().includes(f)).slice(0, MAX_ROWS);

  const edit = (row: Row, field: string, value: unknown) => {
    setData((d: unknown) => setAt(d, row.path, field, value));
    setDirty(true);
  };

  const save = async () => {
    setSaving(true);
    try {
      await api.putDeliveryCues(data);
      setDirty(false);
      toast.show(t("common.saved"));
    } catch (e) {
      setError(toErrorInfo(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="cues-editor" data-testid="cues-editor">
      <p className="hint">{t("cues.hint")}</p>
      {loading ? <Spinner label={t("app.loading")} /> : null}
      {error ? <ErrorPanel error={error} onRetry={() => void load()} /> : null}
      {!loading && rows.length === 0 && !error ? <p className="hint">{t("cues.empty")}</p> : null}
      {rows.length > 0 ? (
        <>
          <div className="row-actions">
            <label className="inline-field">
              <span className="sr-only">{t("cues.filter")}</span>
              <input type="search" value={filter} placeholder={t("cues.filter")} onChange={(e) => setFilter(e.target.value)} data-testid="cues-filter" />
            </label>
            <Button variant="primary" disabled={!dirty} busy={saving} onClick={() => void save()} data-testid="cues-save">
              {t("common.save")}
            </Button>
            {dirty ? <span className="hint">{t("common.unsaved")}</span> : null}
          </div>
          <div className="table-wrap cues-table">
            <table className="data-table" data-testid="cues-table">
              <thead>
                <tr>
                  <th scope="col">{t("cues.cue")}</th>
                  <th scope="col">{t("cues.emotion")}</th>
                  <th scope="col">{t("cues.intensity")}</th>
                  <th scope="col">{t("cues.tempo")}</th>
                  <th scope="col">{t("cues.loudness")}</th>
                  <th scope="col">{t("cues.events")}</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((r) => (
                  <tr key={r.path.join("/")}>
                    <th scope="row">
                      {r.nameField ? (
                        <input type="text" value={r.name} aria-label={t("cues.cue")} onChange={(e) => edit(r, r.nameField as string, e.target.value)} />
                      ) : (
                        r.name
                      )}
                      {r.group ? <span className="muted block">{tOr(`cues.group.${r.group}`, r.group)}</span> : null}
                    </th>
                    <td>{"emotion" in r.node ? <Pick value={String(r.node.emotion ?? "")} options={EMOTIONS} label={t("cues.emotion")} onChange={(v) => edit(r, "emotion", v)} tr={(o) => t(`emotion.${o}`)} /> : null}</td>
                    <td>
                      {"intensity" in r.node ? (
                        <input type="number" min={0} max={1} step={0.1} value={Number(r.node.intensity ?? 0)} aria-label={t("cues.intensity")} onChange={(e) => edit(r, "intensity", Number(e.target.value))} />
                      ) : null}
                    </td>
                    <td>{"tempo" in r.node ? <Pick value={String(r.node.tempo ?? "")} options={TEMPOS} label={t("cues.tempo")} onChange={(v) => edit(r, "tempo", v)} tr={(o) => t(`cues.tempo.${o}`)} /> : null}</td>
                    <td>{"loudness" in r.node ? <Pick value={String(r.node.loudness ?? "")} options={LOUDNESS} label={t("cues.loudness")} onChange={(v) => edit(r, "loudness", v)} tr={(o) => t(`cues.loudness.${o}`)} /> : null}</td>
                    <td>
                      {"events" in r.node ? (
                        <input
                          type="text"
                          value={Array.isArray(r.node.events) ? (r.node.events as unknown[]).join(", ") : String(r.node.events ?? "")}
                          aria-label={t("cues.events")}
                          onChange={(e) =>
                            edit(
                              r,
                              "events",
                              e.target.value
                                .split(",")
                                .map((s) => s.trim())
                                .filter(Boolean),
                            )
                          }
                        />
                      ) : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {rows.length > shown.length && !f ? <p className="hint">{t("cues.truncated", { n: MAX_ROWS })}</p> : null}
        </>
      ) : null}
    </div>
  );
}

function Pick({ value, options, label, onChange, tr }: { value: string; options: string[]; label: string; onChange: (v: string) => void; tr: (o: string) => string }) {
  const opts = options.includes(value) || value === "" ? options : [value, ...options];
  return (
    <select value={value} aria-label={label} onChange={(e) => onChange(e.target.value)}>
      {value === "" ? <option value="" /> : null}
      {opts.map((o) => (
        <option key={o} value={o}>
          {tr(o)}
        </option>
      ))}
    </select>
  );
}
