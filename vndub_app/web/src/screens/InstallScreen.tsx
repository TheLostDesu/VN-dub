import { useCallback, useEffect, useState } from "react";
import { api } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import type { ErrorInfo, InstallPlan } from "../api/types";
import { ErrorPanel } from "../components/ErrorPanel";
import { JobProgress } from "../components/JobProgress";
import { Markdown } from "../components/Markdown";
import { StepFooter } from "../components/StepFooter";
import { Button, Callout, Card, Checkbox, Chip, Collapsible, Dialog, PageTitle, SelectField, Spinner, useToast } from "../components/ui";
import { useI18n } from "../i18n";
import { useJobTracker } from "../state/progress";
import { dirname, fmtDate } from "../util/format";

export default function InstallScreen() {
  const { t, tOr, has, lang } = useI18n();
  const toast = useToast();
  const [plan, setPlan] = useState<InstallPlan | null>(null);
  const [error, setError] = useState<ErrorInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [backup, setBackup] = useState(true);
  const [backupId, setBackupId] = useState("");
  const [confirmRollback, setConfirmRollback] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const p = await api.installPlan();
      setPlan(p);
      setError(null);
      setBackupId((cur) => (cur && p.backups.some((b) => b.id === cur) ? cur : p.backups[0]?.id ?? ""));
    } catch (e) {
      setError(toErrorInfo(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const apply = useJobTracker({ onDone: () => void load() });
  const rollback = useJobTracker({ onDone: () => { toast.show(t("install.rollback_done")); void load(); } });
  const zip = useJobTracker();

  const zipPath = typeof zip.job?.result?.path === "string" ? (zip.job.result.path as string) : "";
  const installed = apply.job?.state === "done";
  const busy = apply.busy || rollback.busy;

  const startApply = () => void apply.start(() => api.installApply(backup));

  return (
    <>
      <PageTitle title={t("install.title")} intro={t("install.intro")} testid="install-title" />

      {loading && !plan ? <Spinner label={t("app.loading")} /> : null}
      {error ? <ErrorPanel error={error} onRetry={() => void load()} /> : null}

      {plan ? (
        <>
          <Card testid="install-plan">
            <div className="game-head">
              <h2>{t("install.what.title")}</h2>
              <Chip tone="info" testid="install-engine">
                {t("game.engine")}: {tOr(`engines.${plan.engine}`, plan.engine)}
              </Chip>
            </div>
            <div className="engine-steps" data-testid="install-engine-steps">
              <h3>{t("install.engine.title")}</h3>
              <p>{has(`install.engine.${plan.engine}.steps`) ? t(`install.engine.${plan.engine}.steps`) : t("install.engine.generic.steps")}</p>
            </div>
            {plan.install_md ? (
              <Collapsible title={t("install.md.title")} testid="install-md-toggle" keepMounted>
                <Markdown source={plan.install_md} testid="install-md" />
              </Collapsible>
            ) : (
              <p className="hint">{t("install.no_md")}</p>
            )}
            {plan.files.length > 0 ? (
              <Collapsible title={t("install.files.title", { n: plan.files.length })} testid="install-files-toggle">
                <ul className="file-list" data-testid="install-files">
                  {plan.files.slice(0, 500).map((f) => (
                    <li key={f.target}>
                      <code>{f.target}</code>
                      {f.exists ? <Chip tone="warn">{t("install.files.replace")}</Chip> : <Chip tone="neutral">{t("install.files.new")}</Chip>}
                    </li>
                  ))}
                </ul>
              </Collapsible>
            ) : null}
          </Card>

          {!plan.auto_possible ? (
            <Callout tone="warn" title={t("install.manual.title")} testid="install-manual">
              <p>{tOr(plan.manual_reason_key, t("install.manual.default_reason"))}</p>
              <p>{t("install.manual.text")}</p>
              <p className="hint">{t("install.manual.no_claim")}</p>
            </Callout>
          ) : null}

          <Card testid="install-actions">
            {plan.auto_possible ? (
              <>
                <Checkbox checked={backup} onChange={setBackup} label={t("install.backup")} hint={t("install.backup_hint")} testid="install-backup" />
                <div className="row-actions">
                  <Button variant="primary" onClick={startApply} busy={apply.busy} disabled={rollback.busy} data-testid="install-apply">
                    {t("install.apply")}
                  </Button>
                </div>
              </>
            ) : null}

            {apply.job ? <JobProgress job={apply.job} onRetry={startApply} testid="install-apply-job" /> : null}
            {apply.error && !apply.job ? <ErrorPanel error={apply.error} onRetry={startApply} /> : null}
            {installed ? (
              <Callout tone="good" title={t("install.done.title")} testid="install-done" role="status">
                <p>{t("install.done.text")}</p>
              </Callout>
            ) : null}

            <hr />
            <h3>{t("install.rollback.title")}</h3>
            {plan.backups.length > 0 ? (
              <div className="row-actions">
                <SelectField
                  label={t("install.rollback.pick")}
                  value={backupId}
                  options={plan.backups.map((b) => ({ value: b.id, label: `${fmtDate(b.created, lang) || b.id}${Array.isArray(b.files) ? ` (${b.files.length})` : typeof b.files === "number" ? ` (${b.files})` : ""}` }))}
                  onChange={setBackupId}
                  testid="install-backup-select"
                />
                <Button onClick={() => setConfirmRollback(true)} disabled={!backupId || busy} data-testid="install-rollback">
                  {t("install.rollback")}
                </Button>
              </div>
            ) : (
              <p className="hint" data-testid="install-no-backups">
                {t("install.rollback.none")}
              </p>
            )}
            {rollback.job ? <JobProgress job={rollback.job} testid="install-rollback-job" /> : null}
            {rollback.error && !rollback.job ? <ErrorPanel error={rollback.error} /> : null}

            <hr />
            <div className="row-actions">
              <Button onClick={() => void api.openFolder("patch").catch(() => undefined)} data-testid="install-open-patch">
                {t("install.open_patch")}
              </Button>
              <Button onClick={() => void zip.start(() => api.installZip())} busy={zip.busy} data-testid="install-zip">
                {t("install.zip")}
              </Button>
            </div>
            <p className="hint">{t("install.zip_hint")}</p>
            {zip.job && zip.busy ? <JobProgress job={zip.job} testid="install-zip-job" /> : null}
            {zip.error ? <ErrorPanel error={zip.error} /> : null}
            {zip.job?.state === "done" && zipPath ? (
              <Callout tone="good" testid="install-zip-done" role="status">
                <p>{t("install.zip_done")}</p>
                <p>
                  <code>{zipPath}</code>
                </p>
                <Button small onClick={() => void api.openFolder("path", dirname(zipPath)).catch(() => undefined)} data-testid="install-zip-open">
                  {t("report.show_folder")}
                </Button>
              </Callout>
            ) : null}
          </Card>
        </>
      ) : null}

      <Dialog
        open={confirmRollback}
        onClose={() => setConfirmRollback(false)}
        title={t("install.rollback.confirm_title")}
        testid="install-rollback-dialog"
        actions={
          <>
            <Button onClick={() => setConfirmRollback(false)}>{t("common.cancel")}</Button>
            <Button
              variant="danger"
              data-autofocus
              onClick={() => {
                setConfirmRollback(false);
                void rollback.start(() => api.installRollback(backupId));
              }}
              data-testid="install-rollback-confirm"
            >
              {t("install.rollback")}
            </Button>
          </>
        }
      >
        <p>{t("install.rollback.confirm_text")}</p>
      </Dialog>

      <StepFooter step="install" />
    </>
  );
}
