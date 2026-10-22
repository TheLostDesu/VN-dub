import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import type { EnvRow, ErrorInfo, ModelRow, ModelsReport, ToolRow } from "../api/types";
import { ErrorPanel, ManualDownloadCard } from "../components/ErrorPanel";
import { HfModelCard } from "../components/HfModelCard";
import { FilesProgress, JobProgress } from "../components/JobProgress";
import { Button, Callout, Card, Chip, Collapsible, LicenseBadge, PageTitle, RadioGroup, Spinner, Tabs } from "../components/ui";
import { useI18n } from "../i18n";
import { useJobTracker } from "../state/progress";
import { fmtGb, fmtMb } from "../util/format";

/**
 * Choices the user made on this screen. /models and /tools are two routes, so switching the tab unmounts the whole tab
 * component; module-level memory keeps the choices for the rest of the UI session (a page reload resets them).
 */
const choices: {
  /** ticked models; `null` = the screen was never initialised (the first load then pre-selects what is missing) */
  selected: Set<string> | null;
  torch: TorchChoice;
} = { selected: null, torch: "auto" };

type TorchChoice = "auto" | "cuda" | "cpu";
const MODEL_JOBS = ["models_download"];
const CHECK_JOBS = ["models_check"];
const ENV_JOBS = ["envs_install"];
const TOOL_JOBS = ["tools_install"];

export default function ModelsScreen({ tab }: { tab: "models" | "tools" }) {
  const { t } = useI18n();
  const nav = useNavigate();
  return (
    <>
      <PageTitle title={tab === "models" ? t("models.title") : t("tools.title")} intro={tab === "models" ? t("models.intro") : t("tools.intro")} testid="models-title" />
      <Tabs
        label={t("models.tabs")}
        value={tab}
        onChange={(v) => nav(v === "models" ? "/models" : "/tools")}
        tabs={[
          { value: "models", label: t("models.tab.models"), testid: "tab-models" },
          { value: "tools", label: t("models.tab.tools"), testid: "tab-tools" },
        ]}
      />
      {tab === "models" ? <ModelsTab /> : <ToolsTab />}
    </>
  );
}

// ---- models --------------------------------------------------------------------------------------

type PresetName = "min" | "recommended" | "all";

/** The backend environment a weight needs, when the row says it is not installed (`reason: env_missing`). */
function missingEnv(m: ModelRow): string | null {
  if (m.reason !== "env_missing") return null;
  const e = m.reason_params?.env;
  return (typeof e === "string" && e) || m.env || null;
}

function ModelsTab() {
  const { t, tOr } = useI18n();
  const nav = useNavigate();
  const [report, setReport] = useState<ModelsReport | null>(null);
  const [error, setError] = useState<ErrorInfo | null>(null);
  const [selected, setSelected] = useState<Set<string>>(() => new Set(choices.selected ?? []));
  const initialised = useRef(choices.selected !== null);
  const [flash, setFlash] = useState<string | null>(null);
  const flashTimer = useRef<number | null>(null);

  const load = useCallback(async () => {
    try {
      const r = await api.models();
      setReport(r);
      setError(null);
      return r;
    } catch (e) {
      setError(toErrorInfo(e));
      return null;
    }
  }, []);

  useEffect(() => {
    void load().then((r) => {
      if (r && !initialised.current) {
        // pre-select what the recommended set still lacks, but only when nothing was chosen on an earlier visit
        initialised.current = true;
        setSelected(new Set(r.rows.filter((m) => !m.present && r.presets.recommended.includes(m.name)).map((m) => m.name)));
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // remember the choice for the next visit of the tab
  useEffect(() => {
    if (initialised.current) choices.selected = selected;
  }, [selected]);

  // the jobs keep running on the server when the tab is left: each tracker picks its job up again on return
  const download = useJobTracker({ onFinal: () => void load(), kinds: MODEL_JOBS });
  const check = useJobTracker({ onFinal: () => void load(), kinds: CHECK_JOBS });
  const envJob = useJobTracker({ onFinal: () => void load(), kinds: ENV_JOBS });

  const rows = report?.rows ?? [];
  // selected weights whose backend environment is not installed yet: they must wait for the environment
  const blocked = rows.filter((m) => selected.has(m.name) && missingEnv(m) !== null);
  const blockedEnvs = Array.from(new Set(blocked.map((m) => missingEnv(m) as string)));
  const toDownload = rows.filter((m) => selected.has(m.name) && !m.present && missingEnv(m) === null);
  const toCheck = rows.filter((m) => selected.has(m.name) && m.present && missingEnv(m) === null);
  const totalGb = rows.filter((m) => selected.has(m.name) && !m.present).reduce((a, m) => a + (m.size_gb || 0), 0);
  const free = report?.disk.free_gb ?? 0;
  const notEnough = report !== null && totalGb > free;

  const applyPreset = (p: PresetName) => {
    if (!report) return;
    setSelected(new Set(report.presets[p]));
  };
  const activePreset = (p: PresetName): boolean => {
    if (!report) return false;
    const names = report.presets[p];
    return names.length === selected.size && names.every((n) => selected.has(n));
  };

  // "this link is a model the table already lists": select its row, scroll to it and light it up for a moment
  const showBuiltin = (name: string) => {
    setSelected((cur) => new Set(cur).add(name));
    setFlash(name);
    if (flashTimer.current !== null) window.clearTimeout(flashTimer.current);
    flashTimer.current = window.setTimeout(() => setFlash(null), 3500);
    window.setTimeout(() => document.querySelector(`[data-testid="model-row-${name}"]`)?.scrollIntoView({ block: "center", behavior: "smooth" }), 50);
  };
  useEffect(
    () => () => {
      if (flashTimer.current !== null) window.clearTimeout(flashTimer.current);
    },
    [],
  );

  const toggle = (name: string, on: boolean) =>
    setSelected((cur) => {
      const next = new Set(cur);
      if (on) next.add(name);
      else next.delete(name);
      return next;
    });

  if (error && !report) return <ErrorPanel error={error} onRetry={() => void load()} />;
  if (!report) return <Spinner label={t("app.loading")} />;

  const startDownload = () => void download.start(() => api.downloadModels(toDownload.map((m) => m.name)));
  const startCheck = () => void check.start(() => api.checkModels((toCheck.length > 0 ? toCheck : rows.filter((m) => m.present && missingEnv(m) === null)).map((m) => m.name)));
  const startEnvs = () => void envJob.start(() => api.installEnvs(blockedEnvs, "auto"));

  return (
    <>
      <HfModelCard onBuiltin={showBuiltin} onAdded={() => void load()} />

      <Card testid="models-presets">
        <div className="row-actions" role="group" aria-label={t("models.presets")}>
          <span className="chip-row-label">{t("models.presets")}</span>
          {(["min", "recommended", "all"] as const).map((p) => (
            <button key={p} type="button" className={`chip-btn ${activePreset(p) ? "is-on" : ""}`} aria-pressed={activePreset(p)} onClick={() => applyPreset(p)} data-testid={`models-preset-${p}`}>
              {t(`models.preset.${p}`)}
            </button>
          ))}
        </div>
        <p className="hint">{t(activePreset("min") ? "models.preset.min_hint" : activePreset("all") ? "models.preset.all_hint" : "models.preset.recommended_hint")}</p>
      </Card>

      <Card testid="models-table-card">
        <div className="table-wrap">
          <table className="data-table is-fixed models-table" data-testid="models-table">
            <colgroup>
              <col className="col-select" />
              <col className="col-name" />
              <col className="col-num col-size" />
              <col className="col-num col-vram" />
              <col className="col-license" />
              <col className="col-state" />
              <col className="col-check" />
            </colgroup>
            <thead>
              <tr>
                <th scope="col">
                  <span className="sr-only">{t("models.col.select")}</span>
                </th>
                <th scope="col">{t("models.col.name")}</th>
                <th scope="col" className="num">
                  {t("models.col.size")}
                </th>
                <th scope="col" className="num">
                  {t("models.col.vram")}
                </th>
                <th scope="col">{t("models.col.license")}</th>
                <th scope="col">{t("models.col.status")}</th>
                <th scope="col">{t("models.col.check")}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((m) => (
                <ModelRowView key={m.name} m={m} checked={selected.has(m.name)} onToggle={(v) => toggle(m.name, v)} recommended={report.presets.recommended.includes(m.name)} flash={flash === m.name} />
              ))}
            </tbody>
          </table>
        </div>
        <p className={`models-total ${notEnough ? "is-bad" : ""}`} data-testid="models-total" role="status">
          {t("models.total", { size: fmtGb(totalGb), free: fmtGb(free) })}
        </p>
        {notEnough ? (
          <Callout tone="bad" testid="models-no-space">
            {t("models.no_space", { need: fmtGb(totalGb), free: fmtGb(free) })}
          </Callout>
        ) : null}
        {blockedEnvs.length > 0 ? (
          <Callout tone="warn" title={t("models.env_needed.title")} testid="models-env-needed">
            <p>{t("models.env_needed.text", { envs: blockedEnvs.map((e) => tOr(`envs.${e}.name`, e)).join(", ") })}</p>
            <div className="row-actions">
              <Button variant="primary" busy={envJob.busy} onClick={startEnvs} data-testid="models-install-envs">
                {t("models.env_needed.install")}
              </Button>
              <Button onClick={() => nav("/tools")} data-testid="models-open-tools">
                {t("models.env_needed.open")}
              </Button>
            </div>
          </Callout>
        ) : null}
        {envJob.job ? (
          <>
            <JobProgress job={envJob.job} onRetry={startEnvs} showLog testid="models-envs-job" />
            {envJob.job.result.files ? <FilesProgress files={envJob.job.result.files} /> : null}
          </>
        ) : null}
        {envJob.error && !envJob.job ? <ErrorPanel error={envJob.error} onRetry={startEnvs} /> : null}
        <p className="hint">{t("models.disk_in", { path: report.disk.data_dir })}</p>
        <div className="row-actions">
          <Button variant="primary" onClick={startDownload} busy={download.busy} disabled={toDownload.length === 0 || check.busy || envJob.busy} data-testid="models-download">
            {t("models.download")}
          </Button>
          <Button onClick={startCheck} busy={check.busy} disabled={download.busy || rows.every((m) => !m.present)} data-testid="models-check">
            {t("models.check")}
          </Button>
        </div>
        <p className="hint">{t("models.check_hint")}</p>

        {download.job ? (
          <>
            <JobProgress job={download.job} onRetry={startDownload} showLog testid="models-download-job" />
            {download.job.result.files ? <FilesProgress files={download.job.result.files} /> : null}
          </>
        ) : null}
        {download.error && !download.job ? <ErrorPanel error={download.error} onRetry={startDownload} /> : null}
        {check.job ? <JobProgress job={check.job} onRetry={startCheck} testid="models-check-job" /> : null}
        {check.error && !check.job ? <ErrorPanel error={check.error} onRetry={startCheck} /> : null}
      </Card>
    </>
  );
}

function ModelRowView({ m, checked, onToggle, recommended, flash }: { m: ModelRow; checked: boolean; onToggle: (v: boolean) => void; recommended: boolean; flash: boolean }) {
  const { t, tOr, has } = useI18n();
  const name = tOr(`models.${m.name}.name`, m.name);
  const purpose = tOr(`models.${m.name}.purpose`, "");
  const reasonKey = m.reason ? `models.reason.${m.reason}` : "";
  const reason = reasonKey && has(reasonKey) ? t(reasonKey, m.reason_params) : "";
  return (
    <tr data-testid={`model-row-${m.name}`} data-present={m.present} className={flash ? "is-flash" : undefined}>
      <td>
        <input type="checkbox" checked={checked} onChange={(e) => onToggle(e.target.checked)} aria-label={t("models.select_one", { name })} data-testid={`model-check-${m.name}`} />
      </td>
      <th scope="row">
        <span className="model-name">{name}</span>
        {purpose ? (
          <span className="hint model-purpose" title={purpose}>
            {purpose}
          </span>
        ) : null}
        {m.role || recommended ? (
          <span className="model-chips">
            {m.role ? <Chip>{tOr(`models.role.${m.role}`, m.role)}</Chip> : null}
            {recommended ? (
              <Chip tone="accent" testid={`model-recommended-${m.name}`}>
                {t("models.chip.recommended")}
              </Chip>
            ) : null}
          </span>
        ) : null}
      </th>
      <td className="num">{fmtGb(m.size_gb)}</td>
      <td className="num">{m.vram_gb !== null && m.vram_gb !== undefined ? fmtGb(m.vram_gb) : ""}</td>
      <td>
        <LicenseBadge license={m.license} commercialOk={m.commercial_ok} />
      </td>
      <td>
        {!m.present ? (
          <Chip tone="neutral">{t("models.status.missing")}</Chip>
        ) : m.usable ? (
          <Chip tone="good">{t("models.status.ready")}</Chip>
        ) : (
          <Chip tone="warn" title={reason}>
            {t("models.status.unusable")}
          </Chip>
        )}
      </td>
      <td data-testid={`model-selftest-${m.name}`}>
        {m.selftest ? (
          m.selftest.ok ? (
            <span>
              <Chip tone="good">{t("models.selftest.ok")}</Chip>
              {m.selftest.peak_vram_gb !== null && m.selftest.peak_vram_gb !== undefined ? <span className="muted"> {t("models.selftest.peak", { n: fmtGb(m.selftest.peak_vram_gb) })}</span> : null}
            </span>
          ) : (
            <Chip tone="bad">{t("models.selftest.failed")}</Chip>
          )
        ) : (
          <span className="muted">{t("models.selftest.none")}</span>
        )}
      </td>
    </tr>
  );
}

// ---- tools and environments ----------------------------------------------------------------------

function ToolsTab() {
  const { t, tOr } = useI18n();
  const [tools, setTools] = useState<ToolRow[] | null>(null);
  const [envs, setEnvs] = useState<EnvRow[] | null>(null);
  const [error, setError] = useState<ErrorInfo | null>(null);
  const [torch, setTorchState] = useState<TorchChoice>(choices.torch);
  const setTorch = (v: TorchChoice) => {
    choices.torch = v;
    setTorchState(v);
  };

  const load = useCallback(async () => {
    try {
      const [a, b] = await Promise.all([api.tools(), api.envs()]);
      setTools(a);
      setEnvs(b);
      setError(null);
    } catch (e) {
      setError(toErrorInfo(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const toolJob = useJobTracker({ onFinal: () => void load(), kinds: TOOL_JOBS });
  const envJob = useJobTracker({ onFinal: () => void load(), kinds: ENV_JOBS });

  if (error && !tools) return <ErrorPanel error={error} onRetry={() => void load()} />;
  if (!tools || !envs) return <Spinner label={t("app.loading")} />;

  const missingTools = tools.filter((x) => !x.present && !x.manual);
  const missingEnvs = envs.filter((x) => !x.present);
  const manual = tools.filter((x) => !x.present && x.manual);

  return (
    <>
      <Card testid="tools-card">
        <h2>{t("tools.list")}</h2>
        <div className="table-wrap">
          <table className="data-table is-fixed tools-table" data-testid="tools-table">
            <colgroup>
              <col className="col-name" />
              <col className="col-for" />
              <col className="col-num" />
              <col className="col-state" />
            </colgroup>
            <thead>
              <tr>
                <th scope="col">{t("tools.col.name")}</th>
                <th scope="col">{t("tools.col.needed_for")}</th>
                <th scope="col" className="num">
                  {t("models.col.size")}
                </th>
                <th scope="col">{t("models.col.status")}</th>
              </tr>
            </thead>
            <tbody>
              {tools.map((x) => (
                <tr key={x.name} data-testid={`tool-row-${x.name}`} data-present={x.present}>
                  <th scope="row">
                    {tOr(`tools.${x.name}.name`, x.name)}
                    {x.version ? <span className="muted"> {x.version}</span> : null}
                    <span className="hint block model-purpose" title={tOr(`tools.${x.name}.purpose`, "")}>
                      {tOr(`tools.${x.name}.purpose`, "")}
                    </span>
                  </th>
                  <td>{x.required_for.length ? x.required_for.join(", ") : t("tools.all_engines")}</td>
                  <td className="num">{x.size_mb ? fmtMb(x.size_mb) : ""}</td>
                  <td>{x.present ? <Chip tone="good">{t("tools.present")}</Chip> : x.manual ? <Chip tone="warn">{t("game.tool_manual")}</Chip> : <Chip tone="neutral">{t("models.status.missing")}</Chip>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="row-actions">
          <Button variant="primary" busy={toolJob.busy} disabled={missingTools.length === 0 || envJob.busy} onClick={() => void toolJob.start(() => api.installTools(missingTools.map((x) => x.name)))} data-testid="tools-install">
            {t("tools.install")}
          </Button>
          {missingTools.length === 0 ? <span className="hint">{t("tools.all_present")}</span> : null}
        </div>
        {toolJob.job ? (
          <>
            <JobProgress job={toolJob.job} onRetry={() => void toolJob.start(() => api.installTools(missingTools.map((x) => x.name)))} showLog testid="tools-install-job" />
            {toolJob.job.result.files ? <FilesProgress files={toolJob.job.result.files} /> : null}
          </>
        ) : null}
        {toolJob.error && !toolJob.job ? <ErrorPanel error={toolJob.error} /> : null}

        {manual.length > 0 ? (
          <Collapsible title={t("tools.manual.title", { n: manual.length })} testid="tools-manual-toggle">
            <p className="hint">{t("tools.manual.text")}</p>
            {manual.map((x) => (
              <div key={x.name}>
                <h4>{tOr(`tools.${x.name}.name`, x.name)}</h4>
                <ManualDownloadCard url={x.url} filename={x.filename} />
              </div>
            ))}
          </Collapsible>
        ) : null}
      </Card>

      <Card testid="envs-card">
        <h2>{t("envs.title")}</h2>
        <p className="hint">{t("envs.intro")}</p>
        <ul className="env-list" data-testid="envs-list">
          {envs.map((e) => (
            <li key={e.name} data-testid={`env-row-${e.name}`} data-present={e.present}>
              <span>{tOr(`envs.${e.name}.name`, e.name)}</span>
              {e.present ? <Chip tone="good">{t("envs.present", { size: fmtGb(e.size_gb) })}</Chip> : <Chip tone="neutral">{t("models.status.missing")}</Chip>}
              {e.torch ? <Chip>{t(`envs.torch.${e.torch}`)}</Chip> : null}
            </li>
          ))}
        </ul>
        <RadioGroup
          name="envs-torch"
          legend={t("envs.torch")}
          value={torch}
          onChange={setTorch}
          testid="envs-torch"
          options={[
            { value: "auto", label: t("envs.torch.auto"), hint: t("envs.torch.auto_hint"), testid: "envs-torch-auto" },
            { value: "cuda", label: t("envs.torch.cuda"), testid: "envs-torch-cuda" },
            { value: "cpu", label: t("envs.torch.cpu"), hint: t("envs.torch.cpu_hint"), testid: "envs-torch-cpu" },
          ]}
        />
        <div className="row-actions">
          <Button variant="primary" busy={envJob.busy} disabled={missingEnvs.length === 0 || toolJob.busy} onClick={() => void envJob.start(() => api.installEnvs(missingEnvs.map((x) => x.name), torch))} data-testid="envs-install">
            {t("envs.install")}
          </Button>
          {missingEnvs.length === 0 ? <span className="hint">{t("envs.all_present")}</span> : null}
        </div>
        {envJob.job ? <JobProgress job={envJob.job} onRetry={() => void envJob.start(() => api.installEnvs(missingEnvs.map((x) => x.name), torch))} showLog testid="envs-install-job" /> : null}
        {envJob.error && !envJob.job ? <ErrorPanel error={envJob.error} /> : null}
      </Card>
    </>
  );
}
