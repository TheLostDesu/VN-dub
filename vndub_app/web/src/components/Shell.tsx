import { useEffect, useState } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { api } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import { STEP_IDS, type ErrorInfo, type StepId } from "../api/types";
import { useI18n } from "../i18n";
import { useActiveJob, useConnected, ingestJob } from "../state/progress";
import { useProject } from "../state/project";
import { furthestStep, pathForJobKind, STEP_NUMBER, stepForJobKind, stepPath, useWizard } from "../state/wizard";
import { stageLabel } from "../util/format";
import { ErrorPanel } from "./ErrorPanel";
import { Button, Checkbox, Dialog, ProgressBar } from "./ui";

/** The running job as a chip in the top bar: title, stage, percentage; a click opens the page that shows the job. */
function TopBarJob() {
  const { t, tOr, has } = useI18n();
  const nav = useNavigate();
  const job = useActiveJob();
  if (!job) return null;
  const title = tOr(job.title_key, t(`jobs.${job.kind}`));
  const stageName = stageLabel(job, { t, has });
  const pct = job.total > 0 ? Math.round(Math.min(1, Math.max(0, job.done / job.total)) * 100) : null;
  const note = job.state === "paused" ? t("job.state.paused") : job.state === "queued" ? t("job.state.queued") : stageName || (pct === null ? t("job.working") : "");
  const dest = pathForJobKind(job.kind);
  const full = [t("job.running_now", { title }), note, pct !== null ? `${pct}%` : ""].filter(Boolean).join(" · ");
  const body = (
    <>
      {pct === null && job.state !== "paused" ? <span className="spinner" aria-hidden="true" /> : null}
      <span className="topbar-job-title">{title}</span>
      {note ? <span className="topbar-job-stage">{note}</span> : null}
      {pct !== null ? <span className="topbar-job-pct">{pct}%</span> : null}
      {pct !== null ? <span className="topbar-job-fill" style={{ width: `${pct}%` }} aria-hidden="true" /> : null}
    </>
  );
  return dest ? (
    <button type="button" className="topbar-job" onClick={() => nav(dest)} title={full} data-testid="topbar-job" data-kind={job.kind}>
      {body}
    </button>
  ) : (
    <span className="topbar-job" title={full} data-testid="topbar-job" data-kind={job.kind}>
      {body}
    </span>
  );
}

export function TopBar() {
  const t = useI18n().t;
  const { project, step } = useProject();
  const loc = useLocation();
  const inWizard = loc.pathname.startsWith("/step/");
  return (
    <header className="topbar">
      <div className="topbar-left">
        <NavLink to={project ? stepPath(step) : "/step/game"} className="brand" data-testid="nav-home" aria-label={t("app.home")}>
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
            <path d="M4 14v-4m4 8V6m4 12V4m4 14V8m4 6v-4" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" fill="none" />
          </svg>
          <span>{t("app.name")}</span>
        </NavLink>
        {project ? (
          <span className="project-name" data-testid="project-name" title={project.game_dir}>
            {project.name}
          </span>
        ) : null}
      </div>
      <TopBarJob />
      <nav className="topbar-nav" aria-label={t("app.menu")}>
        {!inWizard && project ? (
          <NavLink to={stepPath(step)} data-testid="nav-wizard">
            {t("nav.wizard")}
          </NavLink>
        ) : null}
        <NavLink to="/models" data-testid="nav-models">
          {t("nav.models")}
        </NavLink>
        <NavLink to="/settings" data-testid="nav-settings">
          {t("nav.settings")}
        </NavLink>
        <NavLink to="/help" data-testid="nav-help">
          {t("nav.help")}
        </NavLink>
        <NavLink to="/step/doctor" data-testid="nav-doctor">
          {t("nav.doctor")}
        </NavLink>
      </nav>
    </header>
  );
}

export function Stepper({ current }: { current: StepId }) {
  const t = useI18n().t;
  const wiz = useWizard();
  const { project, completed, step } = useProject();
  const reach = project ? furthestStep(step, completed) : 1;
  return (
    <nav className="stepper" aria-label={t("stepper.label")} data-testid="stepper">
      <ol>
        {STEP_IDS.map((id) => {
          const n = STEP_NUMBER[id];
          const done = completed.includes(id);
          const isCurrentStep = id === current;
          const reachable = isCurrentStep || id === "doctor" || (project !== null && n <= reach);
          const isCurrent = id === current;
          return (
            <li key={id} className={`${isCurrent ? "is-current" : ""} ${done ? "is-done" : ""}`}>
              <button
                type="button"
                className="step-btn"
                aria-current={isCurrent ? "step" : undefined}
                aria-disabled={!reachable}
                disabled={!reachable}
                onClick={() => wiz.goto(id)}
                data-testid={`step-${id}`}
              >
                <span className="step-num" aria-hidden="true">
                  {done && !isCurrent ? (
                    <svg viewBox="0 0 24 24" width="14" height="14">
                      <path d="M5 12.5l4.5 4.5L19 7.5" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                    </svg>
                  ) : (
                    n
                  )}
                </span>
                <span className="step-title">{t(`steps.${id}`)}</span>
                {done ? <span className="sr-only">{t("stepper.done")}</span> : null}
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

/** Sticky bar when the program's service has not answered for a few seconds: what is on screen is then out of date. */
export function ConnectionBanner() {
  const t = useI18n().t;
  const connected = useConnected();
  if (connected) return null;
  return (
    <div className="connection-banner" role="alert" data-testid="connection-banner">
      <strong>{t("connection.lost.title")}</strong> {t("connection.lost.text")}
    </div>
  );
}

/** Slim banner for a job that is running while the user looks at another screen. */
export function ActiveJobBar({ currentStep }: { currentStep: StepId | null }) {
  const { t, tOr } = useI18n();
  const nav = useNavigate();
  const loc = useLocation();
  const job = useActiveJob();
  if (!job) return null;
  const target = stepForJobKind(job.kind);
  if (target && target === currentStep) return null;
  const dest = pathForJobKind(job.kind);
  // already on the page that shows this job (Models / Tools tab, Settings): the page has its own progress panel
  if (!target && dest && loc.pathname === dest) return null;
  const frac = job.total > 0 ? job.done / job.total : null;
  const label = tOr(job.title_key, t(`jobs.${job.kind}`));
  return (
    <div className="active-job-bar" role="status" data-testid="active-job-bar">
      <span className="active-job-title">{t("job.running_now", { title: label })}</span>
      <div className="active-job-progress">
        <ProgressBar value={frac} label={label} size="sm" />
      </div>
      {dest ? (
        <Button small onClick={() => nav(dest)} data-testid="active-job-show">
          {t("job.show")}
        </Button>
      ) : null}
    </div>
  );
}

// Games for which the "continue where you left off?" question was dismissed in this session. Per game: closing it for one
// game must not hide the question for another one that also has an interrupted job.
const resumeDismissed = new Set<string>();

/** "Продолжить с того места, где остановились?" when the last job was interrupted. */
export function ResumePrompt() {
  const { t, tOr } = useI18n();
  const { project, reload } = useProject();
  const wiz = useWizard();
  const active = useActiveJob();
  const [, rerender] = useState(0);
  const dismissed = !!project && resumeDismissed.has(project.slug);
  const [error, setError] = useState<ErrorInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [keepShutdown, setKeepShutdown] = useState(false);

  const available = !!project?.resume?.available && !dismissed && !active;
  useEffect(() => {
    setError(null);
  }, [available]);
  if (!available || !project) return null;
  const kind = project.resume.job?.kind ?? "";
  const label = project.resume.job ? tOr(project.resume.job.title_key, t(`jobs.${kind}`)) : "";
  // "turn the computer off when done" of the interrupted run is dropped on a continue unless the user ticks it again
  const hadShutdown = project.resume.job?.params?.shutdown_when_done === true;
  // the other interrupted jobs stay interrupted and are offered next time; one title per kind
  const others = Array.from(new Set((project.resume.candidates ?? []).map((j) => tOr(j.title_key, t(`jobs.${j.kind}`)))));

  const dismiss = () => {
    resumeDismissed.add(project.slug);
    rerender((n) => n + 1);
  };
  const go = async () => {
    setBusy(true);
    setError(null);
    try {
      const job = await api.resumeProject({ keep_shutdown: hadShutdown && keepShutdown });
      ingestJob(job);
      await reload();
      resumeDismissed.add(project.slug);
      rerender((n) => n + 1);
      const target = stepForJobKind(job.kind) ?? stepForJobKind(kind);
      if (target) wiz.goto(target);
    } catch (e) {
      setError(toErrorInfo(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog
      open
      onClose={dismiss}
      title={t("resume.title")}
      testid="resume-dialog"
      actions={
        <>
          <Button onClick={dismiss} data-testid="resume-dismiss">
            {t("resume.later")}
          </Button>
          <Button variant="primary" busy={busy} onClick={() => void go()} data-testid="resume-continue" data-autofocus>
            {t("resume.continue")}
          </Button>
        </>
      }
    >
      <p>{t("resume.text")}</p>
      {label ? (
        <p className="hint">
          {t("resume.was_doing")}: {label}
        </p>
      ) : null}
      {others.length > 0 ? (
        <p className="hint" data-testid="resume-candidates">
          {t("resume.others", { titles: others.join(", ") })}
        </p>
      ) : null}
      {hadShutdown ? (
        <Checkbox checked={keepShutdown} onChange={setKeepShutdown} label={t("resume.keep_shutdown")} hint={t("resume.keep_shutdown_hint")} testid="resume-keep-shutdown" />
      ) : null}
      {error ? <ErrorPanel error={error} /> : null}
    </Dialog>
  );
}
