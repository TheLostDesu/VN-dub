import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import type { ErrorInfo, Glossary, ImportPreview, TranslationSettings } from "../api/types";
import { DeliveryCuesEditor } from "../components/DeliveryCuesEditor";
import { ErrorPanel } from "../components/ErrorPanel";
import { GlossaryEditor } from "../components/GlossaryEditor";
import { JobProgress } from "../components/JobProgress";
import { StepFooter } from "../components/StepFooter";
import { Button, Callout, Card, Checkbox, Chip, Collapsible, LicenseBadge, PageTitle, RadioGroup, SelectField, Spinner, TextField, useToast } from "../components/ui";
import { GenderMissingNote, useGenderInfo } from "../components/GenderNote";
import { useI18n } from "../i18n";
import { useJobTracker } from "../state/progress";
import { useProject } from "../state/project";
import { useSettings } from "../state/settings";
import { stageDone } from "../util/format";
import { useAutosave, type SaveStatus } from "../util/useAutosave";

/** Languages the user can pick as the source language, next to "auto" (the order of the list). */
const SRC_LANGS = ["ja", "en", "zh", "ko", "ru", "uk", "de", "fr", "es", "it", "pt"] as const;
const TRANSLATE_JOBS = ["translate"];

function SaveIndicator({ status }: { status: SaveStatus }) {
  const t = useI18n().t;
  return (
    <span className="save-indicator" role="status" aria-live="polite" data-testid="save-indicator" data-status={status}>
      {status === "saving" ? t("common.saving") : status === "saved" ? t("common.saved") : status === "error" ? t("common.save_failed") : ""}
    </span>
  );
}

export default function TranslateScreen() {
  const { t, langName } = useI18n();
  const toast = useToast();
  const nav = useNavigate();
  const { project, reload } = useProject();
  const { reload: reloadSettings } = useSettings();
  const [tr, setTr] = useState<TranslationSettings | null>(null);
  const [glossary, setGlossary] = useState<Glossary | null>(null);
  const [loadError, setLoadError] = useState<ErrorInfo | null>(null);
  const [dirty, setDirty] = useState(false);
  const [preview, setPreview] = useState<ImportPreview | null>(null);
  const [previewError, setPreviewError] = useState<ErrorInfo | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [pickNote, setPickNote] = useState(false);
  const [translatedHere, setTranslatedHere] = useState(false);

  // the license policy is one global setting that this form also writes: the Settings screen must read it again
  const trSave = useAutosave<TranslationSettings>(
    async (v) => {
      const res = await api.putTranslation(v);
      void reloadSettings();
      return res;
    },
    700,
    () => toast.show(t("common.save_failed")),
  );
  const glSave = useAutosave<Glossary>((v) => api.putGlossary(v), 700, () => toast.show(t("common.save_failed")));
  // kinds + project: the job (running, or done with its warnings) is shown again after a visit to another screen
  const job = useJobTracker({
    onDone: () => {
      setTranslatedHere(true);
      setDirty(false);
      void reload();
      // the translation may have guessed genders and found names: the glossary on the server has them now
      void load();
    },
    kinds: TRANSLATE_JOBS,
    project: project?.slug ?? null,
  });

  const genderInfo = useGenderInfo(translatedHere);

  const load = useCallback(async () => {
    try {
      const [a, g] = await Promise.all([api.translation(), api.glossary()]);
      setTr(a);
      setGlossary(g);
      setLoadError(null);
    } catch (e) {
      setLoadError(toErrorInfo(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const trRef = useRef(tr);
  trRef.current = tr;

  const updateTr = (patch: Partial<TranslationSettings>) => {
    const cur = trRef.current;
    if (!cur) return;
    const next = { ...cur, ...patch };
    setTr(next);
    setDirty(true);
    trSave.schedule(next);
  };
  const updateAdv = (patch: Partial<TranslationSettings["advanced"]>) => {
    const cur = trRef.current;
    if (!cur) return;
    updateTr({ advanced: { ...cur.advanced, ...patch } });
  };
  const updateGlossary = (g: Glossary) => {
    setGlossary(g);
    setDirty(true);
    glSave.schedule(g);
  };

  const runPreview = async (path: string) => {
    if (!path.trim()) return;
    setPreviewBusy(true);
    setPreviewError(null);
    try {
      setPreview(await api.previewImport(path.trim()));
    } catch (e) {
      setPreview(null);
      setPreviewError(toErrorInfo(e));
    } finally {
      setPreviewBusy(false);
    }
  };

  const pickImport = async (kind: "file" | "folder") => {
    setPickNote(false);
    try {
      const res = kind === "file" ? await api.pickFile("translation", ["*.json", "*.jsonl", "*.csv", "*.xlsx", "*.tsv"]) : await api.pickFolder("translation");
      if (res.unsupported) {
        setPickNote(true);
      } else if (res.path) {
        updateTr({ import_path: res.path });
        void runPreview(res.path);
      }
    } catch (e) {
      setPreviewError(toErrorInfo(e));
    }
  };

  const startTranslate = async () => {
    const saved = await Promise.all([trSave.flush(), glSave.flush()]);
    // settings that are not on the server would be translated with the old values: stop and let the error panel say why
    if (saved.some((ok) => !ok)) {
      toast.show(t("translate.save_failed_no_start"));
      return;
    }
    void job.start(() => api.translate());
  };

  if (loadError) return <ErrorPanel error={loadError} onRetry={() => void load()} />;
  if (!tr || !glossary) return <Spinner label={t("app.loading")} />;

  const translated = translatedHere || stageDone(project?.stage_status, "translate");
  const stale = translated && dirty;
  const importMode = tr.source === "import";
  // the language the lines are in: the user's choice, or what the detector found when the choice is "auto"
  const srcAuto = (tr.src_lang ?? "auto") === "auto";
  const srcDetected = tr.src_lang_detected ?? null;
  const srcCode = srcAuto ? (srcDetected?.lang ?? null) : tr.src_lang;
  const sameLang = !!srcCode && srcCode === tr.tgt_lang;
  const srcDetectedHint = !srcAuto
    ? null
    : srcDetected
      ? t("translate.source_lang.detected", { lang: langName(srcDetected.lang), confidence: Math.round(Math.min(1, Math.max(0, srcDetected.confidence)) * 100) })
      : t("translate.source_lang.unknown");

  return (
    <>
      <PageTitle title={t("translate.title")} intro={t("translate.intro")} testid="translate-title" />
      <div className="save-line">
        <SaveIndicator status={trSave.status === "idle" ? glSave.status : trSave.status} />
      </div>
      {trSave.error ? <ErrorPanel error={trSave.error} /> : null}
      {glSave.error ? <ErrorPanel error={glSave.error} /> : null}

      <Card testid="translate-basics">
        <SelectField
          label={t("translate.source_lang")}
          value={tr.src_lang ?? "auto"}
          options={[{ value: "auto", label: t("translate.source_lang.auto") }, ...SRC_LANGS.map((c) => ({ value: c, label: langName(c) }))]}
          onChange={(v) => updateTr({ src_lang: v })}
          testid="translate-src-lang"
        />
        {srcDetectedHint ? (
          <p className="hint src-lang-hint" data-testid="translate-src-lang-detected">
            {srcDetectedHint}
          </p>
        ) : null}

        <SelectField
          label={t("translate.target_lang")}
          value={tr.tgt_lang}
          options={[
            { value: "ru", label: t("lang.ru") },
            { value: "en", label: t("lang.en") },
          ]}
          onChange={(v) => updateTr({ tgt_lang: v })}
          testid="translate-lang"
        />
        {sameLang ? (
          <Callout tone="warn" testid="translate-same-lang">
            {t("translate.same_lang")}
          </Callout>
        ) : null}

        <RadioGroup
          name="tr-source"
          legend={t("translate.source")}
          value={tr.source}
          onChange={(v) => updateTr({ source: v })}
          testid="translate-source"
          options={[
            { value: "auto", label: t("translate.source.auto"), hint: t("translate.source.auto_hint"), testid: "translate-source-auto" },
            { value: "import", label: t("translate.source.import"), hint: t("translate.source.import_hint"), testid: "translate-source-import" },
          ]}
        />

        {tr.source === "auto" && !tr.llm_available ? (
          <Callout tone="warn" title={t("translate.no_llm.title")} testid="translate-no-llm">
            <p>{t("translate.no_llm.text")}</p>
            <Button onClick={() => nav("/models")} data-testid="translate-open-models">
              {t("actions.open_models")}
            </Button>
          </Callout>
        ) : null}

        {importMode ? (
          <div className="import-box" data-testid="translate-import-box">
            <div className="path-row">
              <TextField label={t("translate.import.path")} value={tr.import_path} onChange={(v) => updateTr({ import_path: v })} onBlur={() => void runPreview(tr.import_path)} testid="translate-import-path" className="grow" onEnter={() => void runPreview(tr.import_path)} />
            </div>
            <div className="row-actions">
              <Button onClick={() => void pickImport("file")} data-testid="translate-pick-file">
                {t("translate.import.pick_file")}
              </Button>
              <Button onClick={() => void pickImport("folder")} data-testid="translate-pick-folder">
                {t("translate.import.pick_folder")}
              </Button>
              <Button variant="primary" busy={previewBusy} disabled={!tr.import_path.trim()} onClick={() => void runPreview(tr.import_path)} data-testid="translate-preview-button">
                {t("translate.import.preview")}
              </Button>
            </div>
            {pickNote ? <p className="hint">{t("translate.import.type_path")}</p> : null}
            <p className="hint">{t("translate.import.formats")}</p>
            {previewError ? <ErrorPanel error={previewError} /> : null}
            {preview ? (
              <div data-testid="translate-preview">
                <p>
                  <Chip tone={preview.matched > 0 ? "good" : "bad"}>{t("translate.import.matched", { matched: preview.matched, total: preview.total })}</Chip>
                  <span className="muted"> {t("translate.import.format", { format: preview.format })}</span>
                </p>
                {preview.matched === 0 ? <Callout tone="warn">{t("translate.import.none_matched")}</Callout> : null}
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th scope="col">{t("translate.import.col_src")}</th>
                        <th scope="col">{t("translate.import.col_tgt")}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {preview.preview.slice(0, 5).map((p) => (
                        <tr key={p.line_id}>
                          <td lang="ja">{p.text_src}</td>
                          <td>{p.text_tgt}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}
          </div>
        ) : null}

        <RadioGroup
          name="tr-honorifics"
          legend={t("translate.honorifics")}
          value={tr.honorifics}
          onChange={(v) => updateTr({ honorifics: v })}
          testid="translate-honorifics"
          options={[
            { value: "drop", label: t("translate.honorifics.drop"), hint: t("translate.honorifics.drop_hint"), testid: "honorifics-drop" },
            { value: "keep", label: t("translate.honorifics.keep"), hint: t("translate.honorifics.keep_hint"), testid: "honorifics-keep" },
            { value: "adapt", label: t("translate.honorifics.adapt"), hint: t("translate.honorifics.adapt_hint"), testid: "honorifics-adapt" },
          ]}
        />

        <RadioGroup
          name="tr-address"
          legend={t("translate.address")}
          value={tr.address_default}
          onChange={(v) => updateTr({ address_default: v })}
          testid="translate-address"
          options={[
            { value: "ты", label: t("translate.address.ty"), hint: t("translate.address.ty_hint"), testid: "address-ty" },
            { value: "вы", label: t("translate.address.vy"), hint: t("translate.address.vy_hint"), testid: "address-vy" },
          ]}
        />

        <RadioGroup
          name="tr-player"
          legend={t("translate.player")}
          value={tr.player_gender ?? "unknown"}
          onChange={(v) => updateTr({ player_gender: v as TranslationSettings["player_gender"] })}
          testid="translate-player"
          options={[
            { value: "m", label: t("translate.player.m"), hint: t("translate.player.m_hint"), testid: "player-m" },
            { value: "f", label: t("translate.player.f"), hint: t("translate.player.f_hint"), testid: "player-f" },
            { value: "unknown", label: t("translate.player.unknown"), hint: t("translate.player.unknown_hint"), testid: "player-unknown" },
          ]}
        />

        <RadioGroup
          name="tr-license"
          legend={t("translate.license")}
          value={tr.license_policy}
          onChange={(v) => updateTr({ license_policy: v })}
          testid="translate-license"
          options={[
            { value: "research", label: t("translate.license.research"), hint: t("translate.license.research_hint"), testid: "license-research" },
            { value: "commercial", label: t("translate.license.commercial"), hint: t("translate.license.commercial_hint"), testid: "license-commercial" },
          ]}
        />
      </Card>

      <Card testid="translate-glossary">
        <h2>{t("glossary.title")}</h2>
        <GenderMissingNote items={glossary.characters.map((c) => ({ name: c.tgt || c.src || c.speaker_id, gender: c.gender }))} />
        <GlossaryEditor value={glossary} onChange={updateGlossary} genderInfo={genderInfo} />
      </Card>

      <Collapsible title={t("translate.advanced.title")} testid="translate-advanced-toggle" keepMounted>
        <div className="advanced-grid" data-testid="translate-advanced">
          <div className="field">
            <label htmlFor="n-cand">{t("translate.advanced.n_candidates")}</label>
            <input id="n-cand" type="number" min={1} max={8} step={1} value={tr.advanced.n_candidates} onChange={(e) => updateAdv({ n_candidates: Math.max(1, Math.min(8, Number(e.target.value) || 1)) })} data-testid="advanced-n-candidates" />
            <span className="hint">{t("translate.advanced.n_candidates_hint")}</span>
          </div>

          <fieldset className="radio-group">
            <legend>{t("translate.advanced.routes")}</legend>
            {(["A", "B", "C"] as const).map((r) => (
              <Checkbox
                key={r}
                checked={!!tr.advanced.routes[r]}
                onChange={(v) => updateAdv({ routes: { ...tr.advanced.routes, [r]: v } })}
                label={t(`route.${r}.name`)}
                hint={t(`route.${r}.hint`)}
                testid={`advanced-route-${r}`}
              />
            ))}
          </fieldset>

          <fieldset className="radio-group">
            <legend>{t("translate.advanced.backends")}</legend>
            <p className="hint">{t("translate.advanced.backends_hint")}</p>
            <ul className="backend-list" data-testid="advanced-backends">
              {tr.advanced.backends.map((b, i) => (
                <li key={b.name}>
                  <Checkbox
                    checked={b.enabled}
                    onChange={(v) => updateAdv({ backends: tr.advanced.backends.map((x, j) => (j === i ? { ...x, enabled: v } : x)) })}
                    label={<BackendName name={b.name} />}
                    testid={`advanced-backend-${b.name}`}
                  />
                  <span className="backend-meta">
                    {b.role ? <Chip><RoleName role={b.role} /></Chip> : null}
                    {b.vram_gb !== null && b.vram_gb !== undefined ? <Chip tone="info">{t("translate.advanced.vram", { n: b.vram_gb })}</Chip> : null}
                    <LicenseBadge license={b.license} commercialOk={b.commercial_ok} />
                    {tr.license_policy === "commercial" && b.commercial_ok === false ? <Chip tone="bad">{t("translate.advanced.filtered")}</Chip> : null}
                  </span>
                </li>
              ))}
            </ul>
          </fieldset>

          <Checkbox checked={tr.advanced.asr_check} onChange={(v) => updateAdv({ asr_check: v })} label={t("translate.advanced.asr_check")} hint={t("translate.advanced.asr_check_hint")} testid="advanced-asr-check" />

          {/* kept mounted while collapsed (this section and the one around it): the editor holds its unsaved draft in its own state */}
          <Collapsible title={t("cues.title")} testid="cues-toggle" keepMounted>
            <DeliveryCuesEditor />
          </Collapsible>
        </div>
      </Collapsible>

      <Card testid="translate-run">
        <h2>{t("translate.run.title")}</h2>
        <p>{translated ? t("translate.run.done_text") : t("translate.run.text")}</p>
        {stale ? (
          <Callout tone="info" testid="translate-stale">
            {t("translate.run.stale")}
          </Callout>
        ) : null}
        <div className="row-actions">
          <Button variant="primary" busy={job.busy} onClick={() => void startTranslate()} data-testid="translate-run-button">
            {translated ? t("translate.run.again") : importMode ? t("translate.run.import") : t("translate.run.start")}
          </Button>
          {translated && !job.busy ? <Chip tone="good" testid="translate-done-chip">{t("translate.run.ready")}</Chip> : null}
        </div>
        {job.job ? <JobProgress job={job.job} onRetry={() => void startTranslate()} showLog stageLabel={() => null} testid="translate-job" /> : null}
        {job.error && !job.job ? <ErrorPanel error={job.error} onRetry={() => void startTranslate()} /> : null}
      </Card>

      <StepFooter step="translate" canNext={translated && !job.busy} nextHint={t("translate.next_hint")} />
    </>
  );
}

function RoleName({ role }: { role: string }) {
  const { tOr } = useI18n();
  return <>{tOr(`models.role.${role}`, role)}</>;
}

function BackendName({ name }: { name: string }) {
  const { tOr } = useI18n();
  return <>{tOr(`models.${name}.name`, name)}</>;
}
