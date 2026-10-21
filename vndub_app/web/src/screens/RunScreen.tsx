import { useEffect, useRef } from "react";
import { useLocation } from "react-router-dom";
import { api } from "../api/endpoints";
import { ErrorPanel } from "../components/ErrorPanel";
import { JobProgress } from "../components/JobProgress";
import { StepFooter } from "../components/StepFooter";
import { Button, Callout, Card, Checkbox, Gauge, PageTitle } from "../components/ui";
import { useI18n } from "../i18n";
import { useActiveJob, useGpu, useJobTracker } from "../state/progress";
import { useProject } from "../state/project";
import { useSettings } from "../state/settings";
import { useWizard } from "../state/wizard";
import { stageDone } from "../util/format";

const RUN_KINDS = ["full", "stage"];

export default function RunScreen() {
  const { t } = useI18n();
  const wiz = useWizard();
  const loc = useLocation();
  const { project } = useProject();
  const { settings, update } = useSettings();
  const gpu = useGpu();
  const shutdown = !!settings.shutdown_when_done;
  const tracker = useJobTracker();
  const active = useActiveJob(RUN_KINDS);
  const autostarted = useRef(false);

  useEffect(() => {
    if (active && !tracker.job) tracker.attach(active);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active?.id]);

  const start = () => {
    void tracker.start(() => api.runFull(shutdown));
  };

  // "Мне нравится, запускай всё" on the preview screen arrives here with autostart.
  useEffect(() => {
    const st = loc.state as { autostart?: boolean } | null;
    if (st?.autostart && !autostarted.current && !active && !tracker.job) {
      autostarted.current = true;
      start();
      window.history.replaceState({}, "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loc.state]);

  const job = tracker.job;
  const finished = job?.state === "done" || (!job && stageDone(project?.stage_status, "report"));
  const running = tracker.busy;

  return (
    <>
      <PageTitle title={t("run.title")} intro={t("run.intro")} testid="run-title" />

      <Card testid="run-setup">
        <Checkbox checked={shutdown} onChange={(v) => void update({ shutdown_when_done: v })} label={t("run.shutdown")} hint={t("run.shutdown_hint")} testid="run-shutdown" />
        {shutdown ? (
          <Callout tone="warn" testid="run-shutdown-warning">
            {t("run.shutdown_warning")}
          </Callout>
        ) : null}
        <div className="row-actions">
          <Button variant="primary" busy={tracker.starting} disabled={running} onClick={start} data-testid="run-start">
            {finished ? t("run.start_again") : t("run.start")}
          </Button>
        </div>
        <p className="hint" data-testid="run-closable">
          {t("run.closable")}
        </p>
      </Card>

      {tracker.error && !job ? <ErrorPanel error={tracker.error} onRetry={start} /> : null}

      {job ? (
        <Card testid="run-progress">
          <JobProgress job={job} onRetry={start} showStages showLog testid="run-job" />
          {gpu ? <Gauge used={gpu.used_gb} total={gpu.total_gb} label={t("run.gpu")} testid="run-gpu" /> : null}
        </Card>
      ) : null}

      {finished && !running ? (
        <Callout tone="good" title={t("run.done.title")} testid="run-done">
          <p>{t("run.done.text")}</p>
          <Button variant="primary" onClick={() => void wiz.next("run")} data-testid="run-to-review">
            {t("run.done.go")}
          </Button>
        </Callout>
      ) : null}

      <StepFooter step="run" canNext={!!finished && !running} nextHint={t("run.next_hint")} />
    </>
  );
}
