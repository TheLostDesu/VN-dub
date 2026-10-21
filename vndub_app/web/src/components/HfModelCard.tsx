import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import type { CustomModel, ErrorInfo, HfInspect, HfRecommended } from "../api/types";
import { useI18n } from "../i18n";
import { useJobTracker } from "../state/progress";
import { fmtBytes, fmtGb } from "../util/format";
import { ErrorPanel } from "./ErrorPanel";
import { FilesProgress, JobProgress } from "./JobProgress";
import { Button, Callout, Card, Checkbox, Chip, Dialog, TextField } from "./ui";

/** Plain-words names of the detected kinds (`models.hf.kind.<kind>`). */
const KIND_TONE = { builtin: "neutral", gguf_llm: "accent", rvc: "info", files: "neutral" } as const;

/** "Add a model by link" card of the Models screen: paste a Hugging Face link, check it, pick files, download. */
export function HfModelCard({ onBuiltin, onAdded }: { onBuiltin: (name: string) => void; onAdded: () => void }) {
  const { t, lang } = useI18n();
  const [url, setUrl] = useState("");
  const [inspecting, setInspecting] = useState(false);
  const [ins, setIns] = useState<HfInspect | null>(null);
  const [error, setError] = useState<ErrorInfo | null>(null);
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [reco, setReco] = useState<HfRecommended[]>([]);
  const [custom, setCustom] = useState<CustomModel[]>([]);
  const [added, setAdded] = useState<string | null>(null);
  const input = useRef<HTMLDivElement>(null);

  const loadCustom = useCallback(async () => {
    try {
      setCustom(await api.customModels());
    } catch {
      /* the list is a convenience; the card keeps working */
    }
  }, []);

  useEffect(() => {
    void loadCustom();
    api
      .hfRecommended()
      .then((r) => setReco(r.items))
      .catch(() => setReco([]));
  }, [loadCustom]);

  const add = useJobTracker({
    onFinal: () => void loadCustom(),
    onDone: (job) => {
      const entry = job.result.entry as { label?: string } | undefined;
      setAdded(entry?.label ?? ins?.repo ?? "");
      onAdded();
    },
  });

  const inspect = async (text = url) => {
    const value = text.trim();
    if (!value) return;
    setInspecting(true);
    setError(null);
    setAdded(null);
    setIns(null);
    add.reset();
    try {
      const r = await api.hfInspect(value);
      setIns(r);
      setChecked(new Set(r.suggested));
      if (r.kind === "builtin" && r.builtin) onBuiltin(r.builtin);
    } catch (e) {
      setError(toErrorInfo(e));
    } finally {
      setInspecting(false);
    }
  };

  const pickReco = (item: HfRecommended) => {
    setUrl(item.url);
    setIns(null);
    setError(null);
    input.current?.querySelector("input")?.focus();
  };

  const chosen = (ins?.files ?? []).filter((f) => checked.has(f.name));
  const chosenBytes = chosen.reduce((a, f) => a + f.size, 0);
  const free = ins?.free_gb ?? 0;
  const noSpace = ins !== null && chosenBytes / 1e9 > free;
  const downloadable = ins !== null && ins.kind !== "builtin";
  const startAdd = () => {
    if (!ins) return;
    setAdded(null);
    void add.start(() => api.hfAdd({ repo: ins.repo, revision: ins.revision, files: chosen.map((f) => f.name), kind: ins.kind }));
  };
  const toggle = (name: string, on: boolean) =>
    setChecked((cur) => {
      const next = new Set(cur);
      if (on) next.add(name);
      else next.delete(name);
      return next;
    });

  return (
    <Card testid="models-hf-card">
      <h2>{t("models.hf.title")}</h2>
      <p className="hint">{t("models.hf.intro")}</p>
      <div className="path-row" ref={input}>
        <TextField
          label={t("models.hf.url")}
          value={url}
          onChange={setUrl}
          placeholder="https://huggingface.co/owner/model"
          testid="models-hf-url"
          className="grow"
          hint={t("models.hf.url_hint")}
          onEnter={() => void inspect()}
        />
        <Button variant="primary" busy={inspecting} disabled={!url.trim() || add.busy} onClick={() => void inspect()} data-testid="models-hf-inspect">
          {t("models.hf.inspect")}
        </Button>
      </div>

      {reco.length > 0 ? (
        <div className="hf-reco" data-testid="models-hf-recommended">
          <p className="chip-row-label">{t("models.hf.recommended")}</p>
          <ul>
            {reco.map((it, i) => (
              <li key={it.repo}>
                <button type="button" className="hf-reco-btn" onClick={() => pickReco(it)} data-testid={`models-hf-reco-${i}`} data-repo={it.repo}>
                  <span className="hf-reco-title">{it.title[lang] ?? it.title.ru ?? it.repo}</span>
                  <span className="hf-reco-meta">{t("models.hf.reco_meta", { size: fmtGb(it.size_gb), license: it.license })}</span>
                  <span className="hint block">{it.note[lang] ?? it.note.ru ?? ""}</span>
                </button>
              </li>
            ))}
          </ul>
          <p className="hint">{t("models.hf.recommended_hint")}</p>
        </div>
      ) : null}

      {error ? <ErrorPanel error={error} onRetry={() => void inspect()} onDismiss={() => setError(null)} /> : null}

      {ins ? (
        <div className="hf-result" data-testid="models-hf-result" data-kind={ins.kind} data-repo={ins.repo}>
          <h3 className="hf-repo">
            <a href={ins.page_url} target="_blank" rel="noreferrer">
              {ins.repo}
            </a>
          </h3>
          <div className="chip-row">
            <Chip tone={KIND_TONE[ins.kind]} testid="models-hf-kind">
              {t(`models.hf.kind.${ins.kind}`)}
            </Chip>
            <Chip tone="neutral">{ins.license ? t("models.hf.license", { license: ins.license }) : t("models.hf.license_unknown")}</Chip>
            {ins.gated ? <Chip tone="warn">{t("models.hf.gated")}</Chip> : null}
            <Chip tone="neutral">{t("models.hf.revision", { rev: ins.revision })}</Chip>
          </div>
          <p>{t(`models.hf.kind.${ins.kind}_text`)}</p>
          {ins.warnings.map((w) => (
            <Callout key={w.code} tone={w.code === "gated" ? "warn" : "info"} testid={`models-hf-warn-${w.code}`}>
              {t(`models.hf.warn.${w.code}`, { ...w.params, page_url: ins.page_url })}
            </Callout>
          ))}

          {ins.kind === "builtin" && ins.builtin ? (
            <div className="row-actions">
              <Button onClick={() => onBuiltin(ins.builtin as string)} data-testid="models-hf-builtin">
                {t("models.hf.show_in_list")}
              </Button>
            </div>
          ) : null}

          {downloadable ? (
            <>
              <div className="row-actions">
                <span className="label">{t("models.hf.files")}</span>
                <Button small onClick={() => setChecked(new Set(ins.suggested))} data-testid="models-hf-pick-suggested">
                  {t("models.hf.pick_suggested")}
                </Button>
                <Button small variant="ghost" onClick={() => setChecked(new Set())} data-testid="models-hf-pick-none">
                  {t("models.hf.pick_none")}
                </Button>
              </div>
              <ul className="hf-files" data-testid="models-hf-files">
                {ins.files.map((f, i) => (
                  <li key={f.name}>
                    <Checkbox
                      checked={checked.has(f.name)}
                      onChange={(v) => toggle(f.name, v)}
                      label={
                        <>
                          <span className="hf-file-name">{f.name}</span> <span className="muted">{fmtBytes(f.size)}</span>
                        </>
                      }
                      testid={`models-hf-file-${i}`}
                    />
                  </li>
                ))}
              </ul>
              <p className={`models-total ${noSpace ? "is-bad" : ""}`} data-testid="models-hf-total" role="status">
                {t("models.hf.total", { n: chosen.length, size: fmtBytes(chosenBytes), free: fmtGb(free) })}
              </p>
              {noSpace ? (
                <Callout tone="bad" testid="models-hf-no-space">
                  {t("models.no_space", { need: fmtGb(chosenBytes / 1e9), free: fmtGb(free) })}
                </Callout>
              ) : null}
              <div className="row-actions">
                <Button variant="primary" onClick={startAdd} busy={add.busy} disabled={chosen.length === 0 || noSpace} data-testid="models-hf-add">
                  {t("models.hf.add")}
                </Button>
              </div>
            </>
          ) : null}
        </div>
      ) : null}

      {add.job ? (
        <>
          <JobProgress job={add.job} onRetry={startAdd} showLog testid="models-hf-job" />
          {add.job.result.files ? <FilesProgress files={add.job.result.files} /> : null}
        </>
      ) : null}
      {add.error && !add.job ? <ErrorPanel error={add.error} onRetry={startAdd} /> : null}
      {added !== null && add.job?.state === "done" ? (
        <Callout tone="good" testid="models-hf-done" role="status">
          {t("models.hf.added", { label: added })}
        </Callout>
      ) : null}

      <CustomModelsList items={custom} onChanged={() => void loadCustom()} />
    </Card>
  );
}

function CustomModelsList({ items, onChanged }: { items: CustomModel[]; onChanged: () => void }) {
  const { t } = useI18n();
  const [confirm, setConfirm] = useState<CustomModel | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ErrorInfo | null>(null);
  const [inUse, setInUse] = useState(false);

  const remove = async (m: CustomModel) => {
    setBusy(true);
    setError(null);
    setInUse(false);
    try {
      await api.deleteCustomModel(m.id);
      setConfirm(null);
      onChanged();
    } catch (e) {
      const info = toErrorInfo(e);
      setConfirm(null);
      if (info.code === "bad_request" && info.params.reason === "in_use") setInUse(true);
      else setError(info);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="hf-custom">
      <h3>{t("models.custom.title")}</h3>
      {inUse ? (
        <Callout tone="warn" testid="models-custom-in-use">
          {t("models.custom.in_use_error")}
        </Callout>
      ) : null}
      {error ? <ErrorPanel error={error} onDismiss={() => setError(null)} /> : null}
      {items.length === 0 ? (
        <p className="hint" data-testid="models-custom-empty">
          {t("models.custom.empty")}
        </p>
      ) : (
        <ul className="hf-custom-list" data-testid="models-custom-list">
          {items.map((m) => (
            <li key={m.id} data-testid={`models-custom-row-${m.id}`} data-kind={m.kind} data-present={m.present}>
              <div className="hf-custom-main">
                <strong>{m.label}</strong>{" "}
                <a href={`https://huggingface.co/${m.repo}`} target="_blank" rel="noreferrer" className="muted">
                  {m.repo}
                </a>
                <div className="chip-row">
                  <Chip tone={m.kind === "gguf_llm" ? "accent" : m.kind === "rvc" ? "info" : "neutral"}>{t(`models.custom.kind.${m.kind}`)}</Chip>
                  {m.present ? <Chip tone={m.usable === false ? "warn" : "good"}>{t(m.usable === false ? "models.custom.unusable" : "models.custom.ready")}</Chip> : <Chip tone="warn">{t("models.custom.incomplete")}</Chip>}
                  {m.in_use ? <Chip tone="accent">{t("models.custom.in_use")}</Chip> : null}
                  {m.license ? <Chip tone="neutral">{m.license}</Chip> : null}
                  {m.disk_gb || m.size_gb ? <span className="muted">{t("models.custom.size", { size: fmtBytes((m.disk_gb || m.size_gb) * 1e9) })}</span> : null}
                </div>
                {m.kind === "rvc" ? <p className="hint">{t("models.custom.rvc_note")}</p> : null}
                {m.kind === "files" ? <p className="hint">{t("models.custom.files_note")}</p> : null}
              </div>
              <Button small variant="ghost" onClick={() => setConfirm(m)} data-testid={`models-custom-delete-${m.id}`}>
                {t("models.custom.delete")}
              </Button>
            </li>
          ))}
        </ul>
      )}
      <Dialog
        open={confirm !== null}
        onClose={() => setConfirm(null)}
        title={t("models.custom.delete_title")}
        testid="models-custom-delete-dialog"
        actions={
          <>
            <Button onClick={() => setConfirm(null)}>{t("common.cancel")}</Button>
            <Button variant="danger" busy={busy} data-autofocus onClick={() => confirm && void remove(confirm)} data-testid="models-custom-delete-confirm">
              {t("models.custom.delete")}
            </Button>
          </>
        }
      >
        {confirm ? <p>{t("models.custom.delete_text", { label: confirm.label, size: fmtBytes((confirm.disk_gb || confirm.size_gb) * 1e9) })}</p> : null}
      </Dialog>
    </section>
  );
}
