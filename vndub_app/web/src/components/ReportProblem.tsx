import { api } from "../api/endpoints";
import { useT } from "../i18n";
import { useJobTracker } from "../state/progress";
import { dirname } from "../util/format";
import { JobProgress } from "./JobProgress";
import { Button, Callout } from "./ui";

const REPORT_JOBS = ["report"];

/** "Собрать отчёт о проблеме": builds the zip (job) and shows where it is. */
export function ReportProblem({ label, variant = "primary" }: { label: string; variant?: "primary" | "secondary" }) {
  const t = useT();
  // kinds: the report keeps building when the user leaves the screen; activeOnly: the button lives in several places
  // (Settings, every error panel), so a finished report is shown only where it was started
  const tracker = useJobTracker({ kinds: REPORT_JOBS, activeOnly: true });
  const path = typeof tracker.job?.result?.path === "string" ? (tracker.job.result.path as string) : "";
  return (
    <div className="report-problem" data-testid="report-problem">
      <Button variant={variant} busy={tracker.busy} onClick={() => void tracker.start(() => api.report())} data-testid="report-button">
        {label}
      </Button>
      {tracker.job && tracker.busy ? <JobProgress job={tracker.job} compact /> : null}
      {tracker.job?.state === "done" && path ? (
        <Callout tone="good" testid="report-done">
          <p>{t("report.done")}</p>
          <p>
            <code>{path}</code>
          </p>
          <Button small onClick={() => void api.openFolder("path", dirname(path)).catch(() => undefined)} data-testid="report-open-folder">
            {t("report.show_folder")}
          </Button>
        </Callout>
      ) : null}
      {tracker.error ? <p className="error-hint">{t(`errors.${tracker.error.code}.title`, tracker.error.params)}</p> : null}
    </div>
  );
}
