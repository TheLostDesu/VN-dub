import { useNavigate } from "react-router-dom";
import { api } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import type { ErrorInfo } from "../api/types";
import { useI18n } from "../i18n";
import { ReportProblem } from "./ReportProblem";
import { Button, Collapsible, CopyButton, useToast } from "./ui";

/** Performs the button of an error (`actions.<name>`). Returns false when nothing could be done here. */
export function useErrorActions(onRetry?: () => void) {
  const nav = useNavigate();
  const toast = useToast();
  return async (action: string): Promise<void> => {
    try {
      switch (action) {
        case "retry":
          onRetry?.();
          break;
        case "open_doctor":
          nav("/step/doctor");
          break;
        case "open_models":
          nav("/models");
          break;
        case "open_tools":
          nav("/tools");
          break;
        case "open_settings":
          nav("/settings");
          break;
        case "open_data_dir":
          await api.openFolder("data_dir");
          break;
        case "open_logs":
          await api.openFolder("logs");
          break;
        case "choose_game_dir":
          nav("/step/game");
          break;
        case "open_casting":
          nav("/step/casting");
          break;
        case "open_translation":
          nav("/step/translate");
          break;
        case "show_manual":
          nav("/help");
          break;
        default:
          break;
      }
    } catch (e) {
      toast.show(toErrorInfo(e).code);
    }
  };
}

function str(v: unknown): string {
  return typeof v === "string" ? v : v === undefined || v === null ? "" : String(v);
}

type Tr = (key: string, params?: Record<string, unknown>) => string;

/** Make the params of an error readable: human names for tools / models / stages, plain fallbacks when absent. */
function enrichParams(raw: Record<string, unknown>, t: Tr, has: (k: string) => boolean, tOr: (k: string | null | undefined, f: string) => string): Record<string, unknown> {
  const p: Record<string, unknown> = { ...raw };
  const tool = str(raw.tool);
  const model = str(raw.model);
  const env = str(raw.env);
  const stage = str(raw.stage);
  p.tool = tool ? tOr(`tools.${tool}.name`, tool) : t("errors.param.tool");
  p.model = model ? tOr(`models.${model}.name`, model) : t("errors.param.model");
  p.env = env ? tOr(`envs.${env}.name`, env) : t("errors.param.env");
  p.stage = stage ? (has(`stage_noun.${stage}`) ? t(`stage_noun.${stage}`) : stage) : t("errors.param.stage");
  p.path = str(raw.path) || t("errors.param.path");
  p.filename = str(raw.filename) || t("errors.param.filename");
  return p;
}

/** Manual download card: shown when a tool or model cannot be fetched automatically. */
export function ManualDownloadCard({ url, filename, targetDir }: { url: string; filename: string; targetDir?: string }) {
  const t = useI18n().t;
  return (
    <div className="manual-card" data-testid="manual-download-card">
      <h4>{t("manual.download.title")}</h4>
      <ol>
        <li>
          {t("manual.download.step1")}
          <div className="manual-url">
            <code>{url}</code> <CopyButton text={url} testid="manual-copy-url" />
          </div>
        </li>
        <li>
          {t("manual.download.step2")}
          <div className="manual-url">
            <code>{filename}</code> <CopyButton text={filename} />
          </div>
        </li>
        <li>
          {targetDir ? t("manual.download.target") : t("manual.download.step3")}
          {targetDir ? (
            <div className="manual-url" data-testid="manual-target-dir">
              <code>{targetDir}</code> <CopyButton text={targetDir} testid="manual-copy-target" />
            </div>
          ) : null}
        </li>
      </ol>
      {targetDir ? (
        <Button small onClick={() => void api.openFolder("path", targetDir).catch(() => undefined)} data-testid="manual-open-target">
          {t("manual.download.open_target")}
        </Button>
      ) : (
        <Button small onClick={() => void api.openFolder("data_dir").catch(() => undefined)} data-testid="manual-open-data">
          {t("actions.open_data_dir")}
        </Button>
      )}
    </div>
  );
}

export function ErrorPanel({
  error,
  onRetry,
  onDismiss,
  testid = "error-panel",
}: {
  error: ErrorInfo;
  onRetry?: () => void;
  onDismiss?: () => void;
  testid?: string;
}) {
  const { t, has, tOr } = useI18n();
  const run = useErrorActions(onRetry);
  const known = has(`errors.${error.code}.title`);
  const code = known ? error.code : "internal";
  const params = enrichParams(error.params, t, has, tOr);
  const title = t(`errors.${code}.title`, params);
  const hint = has(`errors.${code}.hint`) ? t(`errors.${code}.hint`, params) : "";
  const action = error.action || "none";
  const hasAction = action !== "none" && !(action === "retry" && !onRetry) && has(`actions.${action}`);
  const details = [error.details, known ? "" : `code: ${error.code}`].filter(Boolean).join("\n");
  const manual =
    error.code === "download_unreachable" && str(error.params.url)
      ? { url: str(error.params.url), filename: str(error.params.filename), targetDir: str(error.params.target_dir) }
      : null;

  return (
    <div className="error-panel" role="alert" data-testid={testid} data-code={error.code}>
      <div className="error-text">
        <p className="error-title" data-testid="error-title">
          {title}
        </p>
        {hint ? (
          <p className="error-hint" data-testid="error-hint">
            {hint}
          </p>
        ) : null}
      </div>
      {manual ? <ManualDownloadCard url={manual.url} filename={manual.filename} targetDir={manual.targetDir || undefined} /> : null}
      {(error.code === "hf_gated" || error.code === "hf_not_found") && str(error.params.page_url) ? (
        <p className="error-link">
          <a href={str(error.params.page_url)} target="_blank" rel="noreferrer" data-testid="error-hf-page">
            {str(error.params.page_url)}
          </a>{" "}
          <CopyButton text={str(error.params.page_url)} />
        </p>
      ) : null}
      <div className="error-actions">
        {action === "report_problem" ? (
          <ReportProblem label={t("actions.report_problem")} />
        ) : hasAction && !(action === "show_manual" && manual) ? (
          <Button variant="primary" onClick={() => void run(action)} data-testid="error-action">
            {t(`actions.${action}`)}
          </Button>
        ) : null}
        {onRetry && action !== "retry" ? (
          <Button onClick={onRetry} data-testid="error-retry">
            {t("actions.retry")}
          </Button>
        ) : null}
        {onDismiss ? (
          <Button variant="ghost" onClick={onDismiss} data-testid="error-dismiss">
            {t("common.dismiss")}
          </Button>
        ) : null}
      </div>
      {details ? (
        <Collapsible title={t("errors.show_details")} testid="error-details-toggle">
          <pre className="details" data-testid="error-details">
            {details}
          </pre>
        </Collapsible>
      ) : null}
    </div>
  );
}

