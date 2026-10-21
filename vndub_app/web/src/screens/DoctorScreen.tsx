import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, isJob, type DefenderHelp } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import type { DoctorReport, DoctorRow, ErrorInfo } from "../api/types";
import { ErrorPanel } from "../components/ErrorPanel";
import { JobProgress } from "../components/JobProgress";
import { StepFooter } from "../components/StepFooter";
import { Button, Callout, Card, CopyButton, Dialog, PageTitle, Spinner } from "../components/ui";
import { useI18n } from "../i18n";
import { useJobTracker } from "../state/progress";
import { markFirstRunDone } from "../util/firstRun";

/** Row params for interpolation: the server's params plus `value` when it is a plain value (never an object). */
function rowParams(row: DoctorRow): Record<string, unknown> {
  const p: Record<string, unknown> = { ...row.params };
  const v = row.value;
  if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") p.value = v;
  return p;
}

const MIN_VRAM_GB = 7.5; // a "8 GB" card reports about 8.0; anything below 7.5 is a smaller card

export default function DoctorScreen() {
  const { t, has } = useI18n();
  const nav = useNavigate();
  const [report, setReport] = useState<DoctorReport | null>(null);
  const [error, setError] = useState<ErrorInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [defender, setDefender] = useState<DefenderHelp | null>(null);
  const [fixing, setFixing] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setReport(await api.doctor());
      setError(null);
    } catch (e) {
      setError(toErrorInfo(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const tracker = useJobTracker({ onFinal: () => void load() });

  useEffect(() => {
    markFirstRunDone();
    void load();
  }, [load]);

  const fix = async (row: DoctorRow) => {
    if (!row.fix) return;
    const action = row.fix.action;
    // navigation actions of errors.ACTIONS (envs / models rows) just open the page
    if (action === "open_tools") return nav("/tools");
    if (action === "open_models") return nav("/models");
    setFixing(action);
    try {
      const res = await api.doctorFix(action);
      if (isJob(res)) tracker.attach(res);
      else if (action === "add_defender_exclusion_help" && "command" in res) setDefender(res as DefenderHelp);
      else void load();
    } catch (e) {
      setError(toErrorInfo(e));
    } finally {
      setFixing(null);
    }
  };

  const gpu = report?.gpu ?? null;
  const gpuProblem = report ? (gpu === null ? "none" : gpu.vram_gb < MIN_VRAM_GB ? "low" : null) : null;
  const fail = report?.summary === "fail";

  return (
    <>
      <PageTitle title={t("doctor.title")} intro={t("doctor.intro")} testid="doctor-title" />

      {loading && !report ? <Spinner label={t("doctor.checking")} /> : null}
      {error ? <ErrorPanel error={error} onRetry={() => void load()} /> : null}

      {report ? (
        <>
          <Callout tone={report.summary === "ok" ? "good" : report.summary === "warn" ? "warn" : "bad"} testid="doctor-summary" role="status">
            <strong>{t(`doctor.summary.${report.summary}`)}</strong>
            {gpu ? (
              <span className="muted">
                {" "}
                {gpu.name}, {gpu.vram_gb.toFixed(1)} GB
              </span>
            ) : null}
          </Callout>

          {gpuProblem ? (
            <Callout tone="warn" title={t(`doctor.gpu_warning.${gpuProblem}.title`)} testid="doctor-gpu-warning">
              <p>{t(`doctor.gpu_warning.${gpuProblem}.text`, { vram: gpu ? gpu.vram_gb.toFixed(1) : "0" })}</p>
            </Callout>
          ) : null}

          <Card>
            <ul className="doctor-rows" data-testid="doctor-rows">
              {report.rows.map((row) => {
                const params = rowParams(row);
                const reason = typeof row.params.reason === "string" ? row.params.reason : "";
                const statusKey = reason && has(`doctor.${row.id}.${row.status}.${reason}`) ? `doctor.${row.id}.${row.status}.${reason}` : `doctor.${row.id}.${row.status}`;
                const fixAction = row.fix?.action ?? "";
                const fixLabelKey = row.fix ? (has(row.fix.label_key) ? row.fix.label_key : has(`doctor.fix.${fixAction}`) ? `doctor.fix.${fixAction}` : `actions.${fixAction}`) : "";
                return (
                  <li key={row.id} className={`doctor-row status-${row.status}`} data-testid={`doctor-row-${row.id}`} data-status={row.status}>
                    <span className={`status-dot status-${row.status}`} aria-hidden="true" />
                    <div className="doctor-row-main">
                      <p className="doctor-row-title">
                        {has(`doctor.${row.id}.title`) ? t(`doctor.${row.id}.title`) : row.id}
                        <span className="sr-only">: {t(`doctor.status.${row.status}`)}</span>
                      </p>
                      <p className="doctor-row-text">{has(statusKey) ? t(statusKey, params) : ""}</p>
                    </div>
                    <div className="doctor-row-actions">
                      {row.fix ? (
                        <Button small variant={row.status === "red" ? "primary" : "secondary"} busy={fixing === fixAction} onClick={() => void fix(row)} data-testid={`doctor-fix-${fixAction}`}>
                          {t(fixLabelKey)}
                        </Button>
                      ) : null}
                    </div>
                  </li>
                );
              })}
            </ul>
            <div className="row-actions">
              <Button onClick={() => void load()} busy={loading} data-testid="doctor-refresh">
                {t("doctor.refresh")}
              </Button>
            </div>
          </Card>
        </>
      ) : null}

      {tracker.job ? <JobProgress job={tracker.job} onRetry={() => void load()} /> : null}

      <Dialog
        open={defender !== null}
        onClose={() => setDefender(null)}
        title={t("doctor.defender_help.title")}
        testid="defender-help"
        wide
        actions={
          <Button variant="primary" onClick={() => setDefender(null)} data-autofocus>
            {t("common.ok")}
          </Button>
        }
      >
        {defender && !defender.applicable ? (
          <p>{t("doctor.defender_help.not_applicable")}</p>
        ) : defender ? (
          <>
            <p>{t("doctor.defender_help.intro")}</p>
            <h3>{t("doctor.defender_help.option_settings")}</h3>
            <p data-testid="defender-settings-path">{has(defender.settings_path_key) ? t(defender.settings_path_key) : t("doctor.defender.settings_path")}</p>
            <p>
              {t("doctor.defender_help.folder")}: <code data-testid="defender-path">{defender.path}</code> <CopyButton text={defender.path} testid="defender-copy-path" />
            </p>
            <h3>{t("doctor.defender_help.option_shell")}</h3>
            <p>
              {t("doctor.defender_help.command_label")}
              {defender.admin_required ? ` ${t("doctor.defender_help.admin_required")}` : ""}
            </p>
            <pre className="command-box" data-testid="defender-command">
              {defender.command}
            </pre>
            <CopyButton text={defender.command} testid="defender-copy-command" />
            <p className="hint">{t("doctor.defender_help.we_do_not_run")}</p>
          </>
        ) : null}
      </Dialog>

      <StepFooter step="doctor" hideBack nextLabel={fail ? t("doctor.continue_anyway") : t("common.next")} />
    </>
  );
}
