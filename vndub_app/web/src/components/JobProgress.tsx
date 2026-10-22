import { Fragment, useEffect, useRef, useState, type ReactNode } from "react";
import { api } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import { ACTIVE_STATES, type ErrorInfo, type JobInfo } from "../api/types";
import { useI18n } from "../i18n";
import { ingestJob, useJobLog } from "../state/progress";
import { fmtBytes, fmtDuration, fmtNumber, normalizeFiles } from "../util/format";
import { ErrorPanel } from "./ErrorPanel";
import { Button, Chip, Collapsible, ProgressBar, copyText } from "./ui";

/** Every server log line starts with the local time `HH:MM:SS `; a line without it is shown as it is. */
const LOG_TS = /^(\d{2}:\d{2}:\d{2}) /;

function LogLine({ line }: { line: string }) {
  const m = LOG_TS.exec(line);
  if (!m) return <>{line}</>;
  return (
    <>
      <span className="log-ts">{m[1]}</span>
      {line.slice(8)}
    </>
  );
}

export function LogViewer({ jobId, fallback }: { jobId: string; fallback?: string[] }) {
  const t = useI18n().t;
  const lines = useJobLog(jobId, fallback);
  const ref = useRef<HTMLPreElement>(null);
  const stick = useRef(true);
  const linesRef = useRef(lines);
  linesRef.current = lines;
  const [copied, setCopied] = useState(false);
  const copiedTimer = useRef<number | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [lines.length]);
  useEffect(
    () => () => {
      if (copiedTimer.current !== null) window.clearTimeout(copiedTimer.current);
    },
    [],
  );

  // the whole log file of the job from the server; when that is not possible, the lines this viewer already has
  const copyLog = async () => {
    let all = linesRef.current;
    try {
      const res = await api.jobLog(jobId);
      if (Array.isArray(res.lines) && res.lines.length > 0) all = res.lines;
    } catch {
      /* older server or the job is gone: copy what is shown */
    }
    if (await copyText(all.join("\n"))) {
      setCopied(true);
      if (copiedTimer.current !== null) window.clearTimeout(copiedTimer.current);
      copiedTimer.current = window.setTimeout(() => setCopied(false), 1500);
    }
  };

  return (
    <div className="log-box">
      <div className="log-head">
        <Button small onClick={() => void copyLog()} data-testid="job-log-copy" aria-live="polite">
          {copied ? t("common.copied") : t("common.copy")}
        </Button>
      </div>
      <pre
        ref={ref}
        className="log-viewer"
        role="log"
        aria-label={t("job.log")}
        tabIndex={0}
        data-testid="job-log"
        onScroll={(e) => {
          const el = e.currentTarget;
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
        }}
      >
        {lines.length === 0
          ? t("job.log_empty")
          : lines.map((line, i) => (
              <Fragment key={i}>
                <LogLine line={line} />
                {"\n"}
              </Fragment>
            ))}
      </pre>
    </div>
  );
}

const FILE_STATES = ["pending", "downloading", "verifying", "extracting", "installing", "done", "failed"];

/** Per-file rows of a download / install job (bytes from `result.files[]`, environments have only a state). */
export function FilesProgress({ files }: { files: unknown }) {
  const { t, tOr, has } = useI18n();
  const rows = normalizeFiles(files);
  if (rows.length === 0) return null;
  return (
    <ul className="files-progress" data-testid="files-progress">
      {rows.map((f) => {
        const failed = f.state === "failed" || f.state === "error";
        const name = f.isEnv ? tOr(`envs.${f.name}.name`, f.name) : f.name;
        const active = f.state === "downloading" || f.state === "installing" || f.state === "extracting" || f.state === "verifying";
        const value = f.state === "done" ? 1 : f.fraction !== null ? f.fraction : active ? null : 0;
        const stateKey = FILE_STATES.includes(f.state) ? `job.file.${f.state}` : "";
        return (
          <li key={f.id} data-state={f.state} data-testid={`file-row-${f.id}`}>
            <span className="file-name">{name}</span>
            <ProgressBar value={value} label={name} size="sm" tone={failed ? "bad" : f.state === "done" ? "good" : "accent"} />
            <span className="file-state">
              {f.skipped && f.state === "done" ? t("job.file.skipped") : stateKey ? t(stateKey) : f.state}
              {f.total !== null && f.state !== "pending" && !f.isEnv && !(f.skipped && f.state === "done") ? ` (${f.state === "done" ? fmtBytes(f.total) : `${fmtBytes(f.done ?? 0)} / ${fmtBytes(f.total)}`})` : ""}
              {f.state === "done" && f.verified === true ? `, ${t("job.file.verified")}` : ""}
            </span>
            {failed && f.error ? (
              <span className="file-error">{has(`errors.${f.error}.title`) ? t(`errors.${f.error}.title`) : f.error}</span>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}

/** Jobs whose done / total are megabytes, environments or files rather than lines. */
const BYTE_JOBS = ["models_download", "models_hf_add", "tools_install", "ffmpeg_download"];
const FILE_JOBS = ["install_apply", "install_rollback", "install_zip"];

function counterKey(kind: string): string {
  if (BYTE_JOBS.includes(kind)) return "job.counter_mb";
  if (kind === "envs_install") return "job.counter_envs";
  if (FILE_JOBS.includes(kind)) return "job.counter_files";
  return "job.counter";
}

function showsLineRate(kind: string): boolean {
  return !BYTE_JOBS.includes(kind) && kind !== "envs_install" && !FILE_JOBS.includes(kind) && kind !== "report";
}

interface Props {
  job: JobInfo;
  /** one-line layout without stage list and log */
  compact?: boolean;
  /** show the list of stages with their state (full run) */
  showStages?: boolean;
  /** offer the collapsible log viewer */
  showLog?: boolean;
  onRetry?: () => void;
  /** replace the label of a stage (e.g. one bar for analyze + text_emotion) */
  stageLabel?: (stage: string) => string | null;
  hideControls?: boolean;
  testid?: string;
  footer?: ReactNode;
  /** replace the fraction of the stage bar (e.g. one bar for two stages) */
  fractionOverride?: (job: JobInfo) => number | null | undefined;
}

export function JobProgress({ job, compact, showStages, showLog, onRetry, stageLabel, hideControls, testid = "job-progress", footer, fractionOverride }: Props) {
  const { t, has, tOr } = useI18n();
  const [busy, setBusy] = useState<string | null>(null);
  const [controlError, setControlError] = useState<ErrorInfo | null>(null);

  const active = ACTIVE_STATES.includes(job.state);
  const title = tOr(job.title_key, t(`jobs.${job.kind}`));
  const stage = job.stage;
  const custom = stage && stageLabel ? stageLabel(stage) : null;
  const stageName = custom ?? (stage ? (has(`stages.${stage}`) ? t(`stages.${stage}`) : stage) : "");
  const hasTotal = job.total > 0;
  const own = job.state === "done" ? 1 : hasTotal ? Math.min(1, job.done / job.total) : null;
  const over = fractionOverride ? fractionOverride(job) : undefined;
  const stageFrac = over !== undefined ? over : own;
  const multi = job.stage_count > 1;
  const finished = job.state === "done";
  // a finished multi-stage job has nothing left to compare: one full bar, not a stage bar plus an identical overall bar
  const twoBars = multi && !compact && !finished;
  const overall = multi ? (finished ? 1 : Math.min(1, (Math.max(0, job.stage_index - 1) + (own ?? 0)) / job.stage_count)) : null;
  const eta = job.eta_s ?? job.stage_eta_s;

  const control = async (name: "pause" | "resume" | "cancel") => {
    setBusy(name);
    setControlError(null);
    try {
      const fn = name === "pause" ? api.pauseJob : name === "resume" ? api.resumeJob : api.cancelJob;
      ingestJob(await fn(job.id));
    } catch (e) {
      setControlError(toErrorInfo(e));
    } finally {
      setBusy(null);
    }
  };

  const stateChip =
    job.state === "paused" ? (
      <Chip tone="warn">{t("job.state.paused")}</Chip>
    ) : job.state === "queued" ? (
      <Chip tone="info">{t("job.state.queued")}</Chip>
    ) : job.state === "cancelled" ? (
      <Chip tone="warn">{t("job.state.cancelled")}</Chip>
    ) : job.state === "interrupted" ? (
      <Chip tone="warn">{t("job.state.interrupted")}</Chip>
    ) : job.state === "failed" ? (
      <Chip tone="bad">{t("job.state.failed")}</Chip>
    ) : finished ? (
      <Chip tone="good">{t("job.state.done")}</Chip>
    ) : null;

  return (
    <div className={`job-progress ${compact ? "is-compact" : ""}`} data-testid={testid} data-state={job.state} data-kind={job.kind}>
      <div className="job-head">
        <h3 className="job-title" data-testid="job-title">
          {title}
        </h3>
        {stateChip}
      </div>

      {job.state !== "failed" ? (
        <>
          {stageName && !compact && !finished ? (
            <p className="job-stage" data-testid="job-stage">
              {multi ? t("job.stage_of", { i: job.stage_index, n: job.stage_count }) + ": " : ""}
              <strong>{stageName}</strong>
            </p>
          ) : null}
          <ProgressBar value={twoBars ? stageFrac : finished ? 1 : (overall ?? stageFrac)} label={stageName || title} testid="job-bar" />
          <div className="job-numbers">
            {hasTotal ? (
              <span data-testid="job-counter">{t(counterKey(job.kind), { done: job.state === "done" ? job.total : job.done, total: job.total })}</span>
            ) : active ? (
              <span>{t("job.working")}</span>
            ) : null}
            {active && eta !== null && eta !== undefined && eta > 0 ? (
              <span data-testid="job-eta">{t("job.eta", { time: fmtDuration(t, eta) })}</span>
            ) : null}
            {active && job.rate_per_min && showsLineRate(job.kind) ? (
              <span data-testid="job-rate">{t("job.rate", { n: job.rate_per_min >= 10 ? Math.round(job.rate_per_min) : fmtNumber(job.rate_per_min, 1) })}</span>
            ) : null}
          </div>
          {job.item && active && !compact ? (
            <p className="job-item" data-testid="job-item">
              {t("job.current_item")}: <code>{job.item}</code>
            </p>
          ) : null}
          {twoBars ? (
            <div className="job-overall">
              <span className="sr-only">{t("job.overall")}</span>
              <ProgressBar value={overall} label={t("job.overall")} size="sm" testid="job-overall" />
            </div>
          ) : null}
          {job.message_key && has(job.message_key) && active ? <p className="job-message">{t(job.message_key, job.message_params)}</p> : null}
          {job.kind && showStages && job.stages.length > 1 ? (
            <ol className="stage-list" data-testid="job-stage-list">
              {job.stages.map((s, i) => {
                const idx = i + 1;
                const st = finished || idx < job.stage_index ? "done" : idx === job.stage_index && active ? "current" : "pending";
                return (
                  <li key={s} className={`stage-${st}`} data-state={st}>
                    <span className="stage-dot" aria-hidden="true" />
                    <span>{has(`stages.${s}`) ? t(`stages.${s}`) : s}</span>
                    <span className="sr-only">{t(`job.stage_state.${st}`)}</span>
                  </li>
                );
              })}
            </ol>
          ) : null}
        </>
      ) : null}

      {job.state === "failed" && job.error ? <ErrorPanel error={job.error} onRetry={onRetry} /> : null}
      {job.state === "interrupted" && onRetry ? (
        <p>
          <Button variant="primary" onClick={onRetry} data-testid="job-continue">
            {t("job.continue")}
          </Button>
        </p>
      ) : null}
      {controlError ? <ErrorPanel error={controlError} testid="job-control-error" /> : null}

      {!hideControls && active ? (
        <div className="job-controls">
          {job.state === "paused" ? (
            <Button onClick={() => void control("resume")} busy={busy === "resume"} data-testid="job-resume">
              {t("job.resume")}
            </Button>
          ) : (
            <Button onClick={() => void control("pause")} busy={busy === "pause"} data-testid="job-pause">
              {t("job.pause")}
            </Button>
          )}
          <Button variant="danger" onClick={() => void control("cancel")} busy={busy === "cancel" || !!job.cancel_requested} disabled={!!job.cancel_requested} data-testid="job-cancel">
            {job.cancel_requested ? t("job.cancelling") : t("job.cancel")}
          </Button>
        </div>
      ) : null}

      {footer}
      {showLog && !compact ? (
        <Collapsible title={t("job.show_log")} testid="job-log-toggle">
          <LogViewer jobId={job.id} />
        </Collapsible>
      ) : null}
    </div>
  );
}
