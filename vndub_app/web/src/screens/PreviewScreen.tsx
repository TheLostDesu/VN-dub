import { useCallback, useEffect, useState } from "react";
import { api } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import type { ErrorInfo, PreviewItem } from "../api/types";
import { ErrorPanel } from "../components/ErrorPanel";
import { JobProgress } from "../components/JobProgress";
import { StepFooter } from "../components/StepFooter";
import { ABPlayer } from "../components/WavePlayer";
import { Button, Callout, Card, Chip, PageTitle, RadioGroup, Spinner } from "../components/ui";
import { useI18n } from "../i18n";
import { useActiveJob, useJobTracker } from "../state/progress";
import { useProject } from "../state/project";
import { useWizard } from "../state/wizard";

export default function PreviewScreen() {
  const { t, has } = useI18n();
  const wiz = useWizard();
  const { project, saveState } = useProject();
  const saved = project?.state.preview;
  const [mode, setMode] = useState<"first_scene" | "first_n">(saved?.mode ?? "first_scene");
  const [n, setN] = useState<number>(saved?.n ?? 40);
  const [items, setItems] = useState<PreviewItem[] | null>(null);
  const [error, setError] = useState<ErrorInfo | null>(null);

  const loadItems = useCallback(async () => {
    try {
      const res = await api.previewItems();
      setItems(res.items ?? []);
      setError(null);
    } catch (e) {
      setItems([]);
      setError(toErrorInfo(e));
    }
  }, []);

  const job = useJobTracker({ onDone: () => void loadItems() });
  const active = useActiveJob(["preview"]);
  useEffect(() => {
    if (active && !job.job) job.attach(active);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active?.id]);

  useEffect(() => {
    void loadItems();
  }, [loadItems]);

  const run = async () => {
    const count = Math.max(1, Math.min(500, Math.round(n) || 40));
    setN(count);
    await saveState({ preview: { mode, n: count } });
    void job.start(() => api.previewRun(mode, count));
  };

  const ready = (items ?? []).some((i) => !!i.dub_url);

  return (
    <>
      <PageTitle title={t("preview.title")} intro={t("preview.intro")} testid="preview-title" />

      <Card testid="preview-setup">
        <RadioGroup
          name="preview-mode"
          legend={t("preview.mode")}
          value={mode}
          onChange={setMode}
          testid="preview-mode"
          options={[
            { value: "first_scene", label: t("preview.mode.first_scene"), hint: t("preview.mode.first_scene_hint"), testid: "preview-mode-scene" },
            { value: "first_n", label: t("preview.mode.first_n"), hint: t("preview.mode.first_n_hint"), testid: "preview-mode-n" },
          ]}
        />
        {mode === "first_n" ? (
          <div className="field narrow">
            <label htmlFor="preview-n">{t("preview.n_label")}</label>
            <input id="preview-n" type="number" min={1} max={500} value={n} onChange={(e) => setN(Number(e.target.value))} data-testid="preview-n" />
          </div>
        ) : null}
        <div className="row-actions">
          <Button variant="primary" busy={job.busy} onClick={() => void run()} data-testid="preview-run">
            {items && items.length > 0 ? t("preview.run_again") : t("preview.run")}
          </Button>
          <span className="hint">{t("preview.run_hint")}</span>
        </div>
        {job.job ? <JobProgress job={job.job} onRetry={() => void run()} showLog testid="preview-job" /> : null}
        {job.error && !job.job ? <ErrorPanel error={job.error} onRetry={() => void run()} /> : null}
      </Card>

      {items === null ? <Spinner label={t("app.loading")} /> : null}
      {error && !job.job ? <ErrorPanel error={error} onRetry={() => void loadItems()} /> : null}

      {items && items.length > 0 ? (
        <Card testid="preview-items">
          <h2>{t("preview.results")}</h2>
          <p className="hint">{t("preview.results_hint")}</p>
          <ul className="preview-list">
            {items.map((it) => (
              <li key={it.line_id} className="preview-item" data-testid={`preview-item-${it.line_id}`}>
                <div className="preview-text">
                  <p className="preview-name">
                    <strong>{it.name || it.speaker_id}</strong>
                    {it.status && has(`review.status.${it.status}`) ? <Chip tone={it.status === "ok" ? "good" : it.status === "check" ? "warn" : "bad"}>{t(`review.status.${it.status}`)}</Chip> : null}
                  </p>
                  <p className="preview-tgt">{it.text_tgt}</p>
                  <p className="preview-src muted" lang="ja">
                    {it.text_src}
                  </p>
                </div>
                {/* both columns are always there: a missing recording shows an empty player of the same size */}
                <ABPlayer originalUrl={it.original_url} dubUrl={it.dub_url} kind="preview" testid={`preview-ab-${it.line_id}`} />
              </li>
            ))}
          </ul>
        </Card>
      ) : items && !job.busy ? (
        <Callout tone="info" testid="preview-empty">
          {t("preview.empty")}
        </Callout>
      ) : null}

      <Card testid="preview-decision">
        <div className="row-actions">
          <Button onClick={() => wiz.goto("casting")} data-testid="preview-change-voices">
            {t("preview.change_voices")}
          </Button>
          <Button onClick={() => wiz.goto("translate")} data-testid="preview-change-settings">
            {t("preview.change_settings")}
          </Button>
        </div>
      </Card>

      <StepFooter step="preview" canNext={ready && !job.busy} nextLabel={t("preview.approve")} nextTestid="preview-approve" nextHint={t("preview.next_hint")} onNext={() => wiz.next("preview", { autostart: true })} />
    </>
  );
}
