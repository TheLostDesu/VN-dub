import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { api } from "../api/endpoints";
import type { PatchCoverage } from "../api/types";
import { ErrorPanel } from "../components/ErrorPanel";
import { JobProgress } from "../components/JobProgress";
import { CoverageCallout } from "../components/PatchCoverage";
import { StepFooter } from "../components/StepFooter";
import { Button, Callout, Card, Checkbox, Gauge, PageTitle } from "../components/ui";
import { useI18n } from "../i18n";
import { useActiveJob, useGpu, useJobTracker } from "../state/progress";
import { useProject } from "../state/project";
import { useSettings } from "../state/settings";
import { useWizard } from "../state/wizard";
import { stageDone } from "../util/format";

const RUN_KINDS = ["full", "stage"];

/** `result.coverage` of a finished full run (what the patch built by it covers), null when the job carries none. */
function coverageOf(result: Record<string, unknown> | undefined): PatchCoverage | null {
  const c = result?.coverage;
  return c && typeof c === "object" && typeof (c as PatchCoverage).installed === "number" ? (c as PatchCoverage) : null;
}

export default function RunScreen() {
  const { t } = useI18n();
  const wiz = useWizard();
  const loc = useLocation();
  const { project } = useProject();
  const { settings, update } = useSettings();
  const gpu = useGpu();
  const shutdown = !!settings.shutdown_when_done;
  // kinds + project: the run (running, or done with its warnings) is shown again after a visit to another screen
  const tracker = useJobTracker({ kinds: RUN_KINDS, project: project?.slug ?? null });
  const active = useActiveJob(RUN_KINDS, project?.slug ?? null);
  const autostarted = useRef(false);

  const start = () => {
    void tracker.start(() => api.runFull(shutdown));
  };

  // "Мне нравится, запускай всё" on the preview screen arrives here with autostart.
  useEffect(() => {
    const st = loc.state as { autostart?: boolean } | null;
    // a finished run that the tracker picked up again must not stop a new one: only a run in progress does
    if (st?.autostart && !autostarted.current && !active && !tracker.busy) {
      autostarted.current = true;
      start();
      window.history.replaceState({}, "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loc.state]);

  const job = tracker.job;
  const finished = job?.state === "done" || (!job && stageDone(project?.stage_status, "report"));
  const running = tracker.busy;

  // What the patch covers: the finished job carries it; the app opened on an already finished run asks the server.
  const jobCoverage = job?.state === "done" ? coverageOf(job.result) : null;
  const [fetchedCoverage, setFetchedCoverage] = useState<PatchCoverage | null>(null);
  useEffect(() => {
    if (!finished || running || jobCoverage) return;
    let alive = true;
    api
      .patchCoverage()
      .then((r) => alive && setFetchedCoverage(r.coverage ?? null))
      .catch(() => alive && setFetchedCoverage(null));
    return () => {
      alive = false;
    };
  }, [finished, running, jobCoverage, job?.id]);
  const coverage = jobCoverage ?? (finished && !running ? fetchedCoverage : null);
  const staleImport = (project?.stale ?? []).some((s) => s.stage === "parse");

  return (
    <>
      <PageTitle title={t("run.title")} intro={t("run.intro")} testid="run-title" />

      {staleImport && !running ? (
        <Callout tone="warn" title={t("stale_import.title")} testid="run-stale-import">
          <p>{t("stale_import.run_text")}</p>
          <Button onClick={() => wiz.goto("game")} data-testid="run-stale-import-go">
            {t("stale_import.go")}
          </Button>
        </Callout>
      ) : null}

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

      {coverage && finished && !running ? <CoverageCallout coverage={coverage} onReview={() => void wiz.next("run")} /> : null}

      <StepFooter step="run" canNext={!!finished && !running} nextHint={t("run.next_hint")} />
    </>
  );
}
