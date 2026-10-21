import { useEffect, useRef, useState } from "react";
import { api } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import type { CustomModel, ErrorInfo, ModelRow } from "../api/types";
import { ErrorPanel } from "../components/ErrorPanel";
import { ReportProblem } from "../components/ReportProblem";
import { Button, Callout, Card, Dialog, PageTitle, RadioGroup, SelectField, Slider, TextField, useToast } from "../components/ui";
import { useI18n, type Lang } from "../i18n";
import { useSettings } from "../state/settings";

export default function SettingsScreen() {
  const { t, tOr } = useI18n();
  const toast = useToast();
  const { settings, update, reload } = useSettings();
  const [llmRows, setLlmRows] = useState<ModelRow[]>([]);
  const [customLlm, setCustomLlm] = useState<CustomModel[]>([]);
  const [error, setError] = useState<ErrorInfo | null>(null);
  const [health, setHealth] = useState<{ version: string; profile: string; mock: boolean } | null>(null);
  const [showPath, setShowPath] = useState(false);
  const [newDir, setNewDir] = useState("");
  const [confirmDir, setConfirmDir] = useState<string | null>(null);
  const [gpu, setGpu] = useState(settings.gpu_budget_gb);
  const gpuTimer = useRef<number | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    // translation models that are on the disk: built in (role llm) and the ones added by link
    api
      .models()
      .then((r) => setLlmRows(r.rows.filter((m) => m.role === "llm")))
      .catch(() => setLlmRows([]));
    api
      .customModels()
      .then((r) => setCustomLlm(r.filter((m) => m.kind === "gguf_llm")))
      .catch(() => setCustomLlm([]));
  }, []);

  // "" = the default (Gemma 4 12B); a stale choice (model deleted meanwhile) is still shown so it can be changed
  const chosenLlm = typeof settings.translation_llm === "string" ? settings.translation_llm : "";
  const llmOptions: { value: string; label: string }[] = [{ value: "", label: t("settings.translation_llm.default") }];
  for (const m of llmRows) {
    if (m.name !== "gemma4_12b" && m.present) llmOptions.push({ value: m.name, label: tOr(`models.${m.name}.name`, m.name) });
  }
  for (const m of customLlm) {
    if (m.present) llmOptions.push({ value: m.id, label: `${m.label} (${m.repo})` });
  }
  if (chosenLlm && !llmOptions.some((o) => o.value === chosenLlm)) llmOptions.push({ value: chosenLlm, label: tOr(`models.${chosenLlm}.name`, chosenLlm) });

  const setGpuBudget = (v: number) => {
    setGpu(v);
    if (gpuTimer.current !== null) window.clearTimeout(gpuTimer.current);
    gpuTimer.current = window.setTimeout(() => void update({ gpu_budget_gb: v }), 600);
  };
  useEffect(
    () => () => {
      if (gpuTimer.current !== null) window.clearTimeout(gpuTimer.current);
    },
    [],
  );

  const pickDir = async () => {
    setError(null);
    try {
      const res = await api.pickFolder("data_dir");
      if (res.unsupported) setShowPath(true);
      else if (res.path) setConfirmDir(res.path);
    } catch (e) {
      setError(toErrorInfo(e));
    }
  };

  const applyDir = async (path: string) => {
    setConfirmDir(null);
    try {
      await api.setDataDir(path);
      await reload();
      setShowPath(false);
      setNewDir("");
      toast.show(t("settings.data_dir.changed"));
    } catch (e) {
      setError(toErrorInfo(e));
    }
  };

  return (
    <>
      <PageTitle title={t("settings.title")} intro={t("settings.intro")} testid="settings-title" />
      {error ? <ErrorPanel error={error} onDismiss={() => setError(null)} /> : null}

      <Card testid="settings-data">
        <h2>{t("settings.data_dir.title")}</h2>
        <p className="hint">{t("settings.data_dir.hint")}</p>
        <p>
          <code data-testid="settings-data-dir">{typeof settings.data_dir === "string" ? settings.data_dir : ""}</code>
        </p>
        <div className="row-actions">
          <Button onClick={() => void api.openFolder("data_dir").catch((e) => setError(toErrorInfo(e)))} data-testid="settings-open-data">
            {t("actions.open_data_dir")}
          </Button>
          <Button onClick={() => void pickDir()} data-testid="settings-change-data">
            {t("settings.data_dir.change")}
          </Button>
        </div>
        {showPath ? (
          <div className="path-row">
            <TextField label={t("settings.data_dir.path")} value={newDir} onChange={setNewDir} testid="settings-data-path" className="grow" onEnter={() => newDir.trim() && setConfirmDir(newDir.trim())} />
            <Button variant="primary" disabled={!newDir.trim()} onClick={() => setConfirmDir(newDir.trim())} data-testid="settings-data-apply">
              {t("settings.data_dir.apply")}
            </Button>
          </div>
        ) : null}
      </Card>

      <Card testid="settings-language">
        <SelectField
          label={t("settings.language")}
          value={(settings.ui_lang === "en" ? "en" : "ru") as Lang}
          options={[
            { value: "ru", label: t("lang.ru") },
            { value: "en", label: t("lang.en") },
          ]}
          onChange={(v) => void update({ ui_lang: v })}
          testid="settings-lang"
        />
      </Card>

      <Card testid="settings-gpu">
        <Slider label={t("settings.gpu_budget")} value={gpu} min={4} max={24} step={0.5} onChange={setGpuBudget} format={(v) => `${v.toFixed(1)} GB`} testid="settings-gpu-budget" />
        <p className="hint">{t("settings.gpu_budget_hint")}</p>
      </Card>

      <Card testid="settings-translation-llm">
        <SelectField label={t("settings.translation_llm.title")} value={chosenLlm} options={llmOptions} onChange={(v) => void update({ translation_llm: v })} testid="settings-translation-llm-select" />
        <p className="hint">{t("settings.translation_llm.hint")}</p>
      </Card>

      <Card testid="settings-license">
        <RadioGroup
          name="settings-license"
          legend={t("settings.license")}
          value={settings.license_policy === "commercial" ? "commercial" : "research"}
          onChange={(v) => void update({ license_policy: v })}
          testid="settings-license-policy"
          options={[
            { value: "research", label: t("translate.license.research"), hint: t("translate.license.research_hint"), testid: "settings-license-research" },
            { value: "commercial", label: t("translate.license.commercial"), hint: t("translate.license.commercial_hint"), testid: "settings-license-commercial" },
          ]}
        />
      </Card>

      <Card testid="settings-report">
        <h2>{t("settings.report.title")}</h2>
        <p className="hint">{t("settings.report.hint")}</p>
        <div className="row-actions">
          <ReportProblem label={t("settings.report.button")} variant="secondary" />
          <Button onClick={() => void api.openFolder("logs").catch((e) => setError(toErrorInfo(e)))} data-testid="settings-open-logs">
            {t("actions.open_logs")}
          </Button>
        </div>
      </Card>

      {health ? (
        <p className="hint" data-testid="settings-version">
          {t("settings.version", { version: health.version, profile: health.profile })}
          {health.mock ? ` ${t("settings.mock")}` : ""}
        </p>
      ) : null}
      {health?.mock ? <Callout tone="info">{t("settings.mock_hint")}</Callout> : null}

      <Dialog
        open={confirmDir !== null}
        onClose={() => setConfirmDir(null)}
        title={t("settings.data_dir.confirm_title")}
        testid="settings-data-dialog"
        actions={
          <>
            <Button onClick={() => setConfirmDir(null)}>{t("common.cancel")}</Button>
            <Button variant="primary" data-autofocus onClick={() => confirmDir && void applyDir(confirmDir)} data-testid="settings-data-confirm">
              {t("settings.data_dir.apply")}
            </Button>
          </>
        }
      >
        <p>
          <code>{confirmDir}</code>
        </p>
        <p>{t("settings.data_dir.confirm_text")}</p>
      </Dialog>
    </>
  );
}
