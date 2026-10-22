import { useCallback, useEffect, useState } from "react";
import { api } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import type { ErrorInfo, JobInfo, MissingTool, ProjectLite, ProjectSummary, SummaryCharacter, ToolRow } from "../api/types";
import { ErrorPanel } from "../components/ErrorPanel";
import { FilesProgress, JobProgress } from "../components/JobProgress";
import { KrkrDumpGuide } from "../components/ManualGuides";
import { StepFooter } from "../components/StepFooter";
import { Button, Callout, Card, Chip, Collapsible, PageTitle, ProgressBar, Spinner, TextField } from "../components/ui";
import { useI18n } from "../i18n";
import { useJobTracker } from "../state/progress";
import { useProject } from "../state/project";
import { fmtMinutes, fmtPercent } from "../util/format";

const GAME_KINDS = ["import", "recover_text", "asr_check"];
const TOOL_KINDS = ["tools_install"];
const ANALYSIS_STAGES = ["analyze", "text_emotion"];

function noTextCount(v: ProjectSummary["no_text_lines"] | undefined): number {
  if (Array.isArray(v)) return v.length;
  return typeof v === "number" ? v : 0;
}

/** One bar for analyze + text_emotion: the share of that pair that is finished. */
function analysisFraction(job: JobInfo): number | null | undefined {
  if (!job.stage || !ANALYSIS_STAGES.includes(job.stage)) return undefined;
  const group = job.stages.filter((s) => ANALYSIS_STAGES.includes(s));
  if (group.length < 2) return undefined;
  const idx = group.indexOf(job.stage);
  const own = job.total > 0 ? Math.min(1, job.done / job.total) : 0;
  return (idx + own) / group.length;
}

export default function GameScreen() {
  const { t, has, tOr, langName } = useI18n();
  const { project, open, reload, loading } = useProject();
  const engineName = (id: string | null, label: string | null): string => (id ? tOr(`engines.${id}`, label ?? id) : label ?? "");

  const [summary, setSummary] = useState<ProjectSummary | null>(null);
  const [summaryLoaded, setSummaryLoaded] = useState(false);
  const [pickError, setPickError] = useState<ErrorInfo | null>(null);
  const [showPath, setShowPath] = useState(false);
  const [path, setPath] = useState("");
  const [opening, setOpening] = useState(false);
  const [recent, setRecent] = useState<ProjectLite[]>([]);
  const [toolRows, setToolRows] = useState<ToolRow[] | null>(null);

  const loadSummary = useCallback(async () => {
    try {
      const s = await api.summary();
      setSummary(s);
      return s;
    } catch {
      setSummary(null);
      return null;
    } finally {
      setSummaryLoaded(true);
    }
  }, []);

  // The jobs keep running when the screen is left: the trackers follow them again on return (a running job, else the
  // finished panel of the open game, exactly as it looked when the user left).
  const gameJob = useJobTracker({ onFinal: () => void loadSummary(), kinds: GAME_KINDS, project: project?.slug ?? null });
  const toolJob = useJobTracker({ onDone: () => void loadSummary(), kinds: TOOL_KINDS });

  useEffect(() => {
    api.projects().then(setRecent).catch(() => setRecent([]));
  }, [project?.slug]);

  useEffect(() => {
    setSummary(null);
    setSummaryLoaded(false);
    if (project) void loadSummary();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project?.slug]);

  const startImport = useCallback(() => {
    void gameJob.start(() => api.importGame());
  }, [gameJob]);

  const openGame = useCallback(
    async (target: { game_dir: string } | { slug: string }) => {
      setPickError(null);
      setOpening(true);
      try {
        const p = await open(target);
        const s = await api.summary().catch(() => null);
        setSummary(s);
        setSummaryLoaded(true);
        // a new game (nothing imported yet) is analysed right away: one click from folder to result
        if (!s || s.lines_total === 0) await gameJob.start(() => api.importGame());
        else void reload();
        return p;
      } catch (e) {
        setPickError(toErrorInfo(e));
        return null;
      } finally {
        setOpening(false);
      }
    },
    [open, gameJob, reload],
  );

  const pick = async () => {
    setPickError(null);
    try {
      const res = await api.pickFolder("game");
      if (res.unsupported) {
        setShowPath(true);
      } else if (res.path) {
        await openGame({ game_dir: res.path });
      }
    } catch (e) {
      setPickError(toErrorInfo(e));
    }
  };

  const importing = gameJob.busy;
  const jobFailed = gameJob.job?.state === "failed";
  const imported = !!summary && summary.lines_total > 0;
  const missing = summary?.tools_missing ?? [];
  // a tool can be installed by the app only if the installer manifest knows it and it is not manual
  const installable = (name: string): boolean => (toolRows ? toolRows.some((r) => r.name === name && !r.manual) : true);
  const missingKey = missing.map((m) => m.tool).join(",");
  useEffect(() => {
    if (!missingKey) return;
    api.tools().then(setToolRows).catch(() => setToolRows(null));
  }, [missingKey]);

  return (
    <>
      <PageTitle title={t("game.title")} intro={t("game.intro")} testid="game-title" />

      {/* ---- choose a game ---------------------------------------------------------------- */}
      <Card testid="game-picker">
        {/* label, then the field with all its buttons on one line; without the field: the two ways to choose a game */}
        <div className="path-row">
          {showPath ? (
            <>
              <TextField label={t("game.path_label")} value={path} onChange={setPath} placeholder={t("game.path_placeholder")} testid="game-path-input" onEnter={() => path.trim() && void openGame({ game_dir: path.trim() })} className="grow" />
              <Button variant="primary" disabled={!path.trim() || importing} busy={opening} onClick={() => void openGame({ game_dir: path.trim() })} data-testid="game-path-open">
                {t("game.open")}
              </Button>
            </>
          ) : null}
          <Button variant={project || showPath ? "secondary" : "primary"} onClick={() => void pick()} busy={opening && !showPath} disabled={importing} data-testid="game-pick-folder">
            {project ? t("game.pick_another") : t("game.pick_folder")}
          </Button>
          {!showPath ? (
            <Button variant="ghost" onClick={() => setShowPath(true)} data-testid="game-manual-toggle">
              {t("game.enter_path")}
            </Button>
          ) : null}
        </div>
        {pickError ? <ErrorPanel error={pickError} onRetry={() => void pick()} /> : null}
        {recent.filter((p) => p.slug !== project?.slug).length > 0 ? (
          <div className="recent" data-testid="game-recent">
            <p className="hint">{t("game.recent")}</p>
            <ul>
              {recent.filter((p) => p.slug !== project?.slug).slice(0, 6).map((p) => (
                <li key={p.slug}>
                  <Button small variant="ghost" onClick={() => void openGame({ slug: p.slug })} disabled={importing} data-testid={`game-recent-${p.slug}`} title={p.game_dir}>
                    {p.name}
                  </Button>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </Card>

      {loading && !project ? <Spinner label={t("app.loading")} /> : null}

      {project ? (
        <Card testid="game-card">
          <div className="game-head">
            <div>
              <h2 data-testid="game-name">{project.name}</h2>
              <p className="muted path" title={project.game_dir}>
                {project.game_dir}
              </p>
            </div>
            <div className="chip-line">
              {summary && engineName(summary.engine.id, summary.engine.label) ? (
                <Chip tone="info" testid="game-engine">
                  {t("game.engine")}: {engineName(summary.engine.id, summary.engine.label)}
                  {summary.engine.version ? ` ${summary.engine.version}` : ""}
                </Chip>
              ) : project.engine ? (
                <Chip tone="info" testid="game-engine">
                  {t("game.engine")}: {engineName(project.engine, null)}
                </Chip>
              ) : null}
              {summary?.languages?.src ? (
                <Chip tone="info" testid="game-text-lang">
                  {t("game.text_lang")}: {langName(summary.languages.src)}
                  {summary.languages.src_detected ? ` (${t("game.text_lang.detected")})` : ""}
                </Chip>
              ) : null}
              {summary?.languages?.audio && summary.languages.audio !== summary.languages.src ? (
                <Chip tone="info" testid="game-audio-lang">
                  {t("game.audio_lang")}: {langName(summary.languages.audio)}
                </Chip>
              ) : null}
            </div>
          </div>

          {!gameJob.job && !imported && summaryLoaded && !opening ? (
            <div className="row-actions">
              <Button variant="primary" onClick={startImport} data-testid="game-import">
                {t("game.analyze")}
              </Button>
              <span className="hint">{t("game.analyze_hint")}</span>
            </div>
          ) : null}

          {gameJob.job ? (
            <JobProgress
              job={gameJob.job}
              onRetry={startImport}
              showLog
              stageLabel={(s) => (ANALYSIS_STAGES.includes(s) ? t("game.analyzing") : null)}
              fractionOverride={analysisFraction}
              testid="game-job"
            />
          ) : null}
          {gameJob.error && !gameJob.job ? <ErrorPanel error={gameJob.error} onRetry={startImport} /> : null}
        </Card>
      ) : null}

      {/* ---- tools that are missing ---------------------------------------------------------- */}
      {project && missing.length > 0 ? (
        <Card testid="game-tools-missing">
          <h2>{t("game.tools_missing.title")}</h2>
          <p>{t("game.tools_missing.text")}</p>
          <ul className="tool-cards">
            {missing.map((m) => (
              <MissingToolCard
                key={m.tool}
                tool={m}
                installable={installable(m.tool)}
                busy={toolJob.busy}
                onInstall={() => void toolJob.start(() => api.installTools([m.tool]))}
              />
            ))}
          </ul>
          {toolJob.job ? (
            <>
              <JobProgress job={toolJob.job} onRetry={() => toolJob.reset()} testid="tools-job" />
              {toolJob.job.result.files ? <FilesProgress files={toolJob.job.result.files} /> : null}
            </>
          ) : null}
          {toolJob.error && !toolJob.job ? <ErrorPanel error={toolJob.error} /> : null}
          {summary?.engine.encrypted || missing.some((m) => /krkr/i.test(`${m.manual_key ?? ""} ${m.tool}`)) ? <KrkrDumpGuide url={missing.find((m) => /krkr/i.test(m.tool))?.url} /> : null}
          {!importing ? (
            <div className="row-actions">
              <Button onClick={startImport} data-testid="game-reimport">
                {t("game.analyze_again")}
              </Button>
            </div>
          ) : null}
        </Card>
      ) : null}

      {/* ---- the result ------------------------------------------------------------------------ */}
      {project && summary && imported && !importing ? (
        <SummaryView
          summary={summary}
          onRecoverText={() => void gameJob.start(() => api.recoverText())}
          onAsrCheck={() => void gameJob.start(() => api.asrCheck())}
          onReimport={startImport}
          busy={importing}
          has={has}
        />
      ) : null}

      {project && summaryLoaded && !imported && !importing && !jobFailed && gameJob.job?.state === "done" ? (
        <Callout tone="warn" testid="game-empty" role="status">
          {t("game.empty")}
        </Callout>
      ) : null}

      <StepFooter step="game" canNext={imported && !importing} nextHint={t("game.next_hint")} />
    </>
  );
}

function MissingToolCard({ tool, installable, busy, onInstall }: { tool: MissingTool; installable: boolean; busy: boolean; onInstall: () => void }) {
  const { t, tOr } = useI18n();
  const isManual = !!tool.manual_key || !installable;
  return (
    <li className="tool-card" data-testid={`tool-missing-${tool.tool}`}>
      <div>
        <strong>{tOr(`tools.${tool.tool}.name`, tool.tool)}</strong>
        <p className="hint">{tOr(`tools.${tool.tool}.purpose`, t("game.tool_needed"))}</p>
        {isManual && tool.url ? (
          <p className="hint">
            {t("manual.download.step1")} <code>{tool.url}</code>
          </p>
        ) : null}
      </div>
      {!isManual ? (
        <Button variant="primary" busy={busy} onClick={onInstall} data-testid={`tool-install-${tool.tool}`}>
          {t("game.tool_install")}
        </Button>
      ) : (
        <Chip tone="warn">{t("game.tool_manual")}</Chip>
      )}
    </li>
  );
}

function Stat({ label, value, testid, tone }: { label: string; value: number | string; testid: string; tone?: "warn" }) {
  return (
    <div className={`stat ${tone === "warn" ? "stat-warn" : ""}`} data-testid={testid}>
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}

function SummaryView({
  summary,
  onRecoverText,
  onAsrCheck,
  onReimport,
  busy,
  has,
}: {
  summary: ProjectSummary;
  onRecoverText: () => void;
  onAsrCheck: () => void;
  onReimport: () => void;
  busy: boolean;
  has: (key: string) => boolean;
}) {
  const { t } = useI18n();
  const eng = summary.engine;
  // encrypted_archives has its own callout and drm:<kind> codes duplicate eng.drm
  const warnings = eng.warnings.filter((w) => w !== "encrypted_archives" && !w.startsWith("drm:"));
  const noText = noTextCount(summary.no_text_lines);
  const chars = [...summary.characters].sort((a, b) => b.lines - a.lines);
  const unvoicedChars = chars.filter((c) => c.mode !== "voiced");
  const noVoiceAtAll = summary.voiced === 0;
  const conf = summary.mapping.confidence;
  const confTone = conf === null ? "neutral" : conf >= 0.9 ? "good" : conf >= 0.7 ? "warn" : "bad";
  const confKey = conf === null ? "unknown" : conf >= 0.9 ? "high" : conf >= 0.7 ? "medium" : "low";

  return (
    <div className="summary" data-testid="game-summary">
      {/* engine warnings */}
      {eng.encrypted || eng.drm.length > 0 || warnings.length > 0 ? (
        <Card testid="game-engine-warnings">
          {eng.encrypted ? (
            <Callout tone="warn" title={t("game.encrypted.title")} testid="game-encrypted">
              <p>{t("game.encrypted.text")}</p>
            </Callout>
          ) : null}
          {eng.drm.length > 0 ? (
            <Callout tone="warn" title={t("game.drm.title")} testid="game-drm">
              <ul>
                {eng.drm.map((d) => (
                  <li key={d}>{has(`engine.drm.${d}`) ? t(`engine.drm.${d}`) : t("engine.drm.unknown", { name: d })}</li>
                ))}
              </ul>
            </Callout>
          ) : null}
          {warnings.map((w) => (
            <Callout key={w} tone="info" testid="game-engine-warning">
              {has(`summary.warning.${w}`) ? t(`summary.warning.${w}`) : t("summary.warning.other")}
            </Callout>
          ))}
        </Card>
      ) : null}

      <Card testid="game-stats">
        <h2>{t("game.found.title")}</h2>
        <div className="stats">
          <Stat label={t("game.found.lines")} value={summary.lines_total} testid="stat-lines" />
          <Stat label={t("game.found.voiced")} value={summary.voiced} testid="stat-voiced" />
          <Stat label={t("game.found.unvoiced")} value={summary.unvoiced} testid="stat-unvoiced" tone={summary.unvoiced > 0 ? "warn" : undefined} />
          <Stat label={t("game.found.voice_files")} value={summary.voice_files} testid="stat-voice-files" />
        </div>

        {noVoiceAtAll ? (
          <Callout tone="info" testid="game-mode-b" title={t("game.mode_b.title")}>
            <p>{t("game.mode_b.text")}</p>
          </Callout>
        ) : (
          <p className="hint" data-testid="game-analysis">
            {t("game.analysis", { audio: summary.analysis.audio_passports, text: summary.analysis.text_passports })}
          </p>
        )}
      </Card>

      {/* mapping confidence */}
      {!noVoiceAtAll ? (
        <Card testid="game-mapping">
          <h2>{t("game.mapping.title")}</h2>
          <div className="mapping-line">
            <ProgressBar value={conf} label={t("game.mapping.title")} tone={confTone === "good" ? "good" : confTone === "warn" ? "warn" : confTone === "bad" ? "bad" : "accent"} testid="mapping-bar" />
            <Chip tone={confTone} testid="mapping-confidence">
              {conf === null ? t("game.mapping.unknown") : fmtPercent(conf)}
            </Chip>
          </div>
          <p data-testid="mapping-text">{t(`game.mapping.${confKey}`)}</p>
          <p className="hint">
            {t("game.mapping.counts", { ok: summary.mapping.ok, review: summary.mapping.review, unmatched: summary.mapping.unmatched })}
            {summary.mapping.mode === "no_asr" ? ` ${t("game.mapping.no_asr")}` : ""}
          </p>
          {summary.mapping.suggest_asr_check ? (
            <Callout tone="warn" testid="game-asr-suggest" title={t("game.asr.suggest_title")}>
              <p>{t("game.asr.suggest_text")}</p>
              <Button onClick={onAsrCheck} disabled={busy} data-testid="game-asr-check">
                {t("game.asr.button")}
              </Button>
            </Callout>
          ) : null}
        </Card>
      ) : null}

      {/* lines with voice but no text: the only place where speech recognition is offered */}
      {noText > 0 ? (
        <Card testid="game-no-text">
          <Callout tone="warn" title={t("game.no_text.title", { n: noText })}>
            <p>{t("game.no_text.text")}</p>
            <Button variant="primary" onClick={onRecoverText} disabled={busy} data-testid="game-recover-text">
              {t("game.no_text.button")}
            </Button>
          </Callout>
        </Card>
      ) : null}

      {/* characters */}
      <Card testid="game-characters">
        <h2>{t("game.characters.title")}</h2>
        {unvoicedChars.length > 0 && !noVoiceAtAll ? (
          <Callout tone="warn" testid="game-unvoiced-list">
            <p>
              <strong>{t("game.characters.unvoiced_list", { names: unvoicedChars.map((c) => c.name_tgt || c.name_src || c.speaker_id).join(", ") })}</strong>
            </p>
            <p>{t("game.characters.unvoiced_hint")}</p>
          </Callout>
        ) : null}
        <div className="table-wrap">
          <table className="data-table" data-testid="game-characters-table">
            <thead>
              <tr>
                <th scope="col">{t("game.characters.name")}</th>
                <th scope="col" className="num">
                  {t("game.characters.lines")}
                </th>
                <th scope="col">{t("game.characters.voice_split")}</th>
                <th scope="col" className="num">
                  {t("game.characters.minutes")}
                </th>
                <th scope="col">{t("game.characters.voice")}</th>
              </tr>
            </thead>
            <tbody>
              {chars.map((c) => (
                <CharacterRow key={c.speaker_id} c={c} />
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* problems: collapsible and non-blocking */}
      {summary.problem_files.length > 0 ? (
        <Collapsible title={t("game.problems.title", { n: summary.problem_files.length })} testid="game-problems-toggle">
          <p className="hint">{t("game.problems.text")}</p>
          <ul className="problem-list" data-testid="game-problems">
            {summary.problem_files.slice(0, 200).map((p) => (
              <li key={p.line_id}>
                <code>{p.line_id}</code>
                <span>{p.reasons.map((r) => reasonText(r, t, has)).join("; ") || (has(`game.problems.status.${p.status}`) ? t(`game.problems.status.${p.status}`) : t("game.problems.status.unknown"))}</span>
              </li>
            ))}
          </ul>
        </Collapsible>
      ) : null}

      <Collapsible title={t("game.advanced.title")} testid="game-advanced-toggle">
        <p className="hint">{t("game.asr.text")}</p>
        <div className="row-actions">
          <Button onClick={onAsrCheck} disabled={busy || noVoiceAtAll} data-testid="game-asr-check-advanced">
            {t("game.asr.button")}
          </Button>
          <Button onClick={onReimport} disabled={busy} data-testid="game-reimport-advanced">
            {t("game.analyze_again")}
          </Button>
        </div>
      </Collapsible>
    </div>
  );
}

function reasonText(reason: string, t: (k: string, p?: Record<string, unknown>) => string, has: (k: string) => boolean): string {
  const [code, ...rest] = reason.split(":");
  const key = `game.reason.${code}`;
  const value = rest.join(":");
  return has(key) ? t(key, { value }) : t("game.reason.unknown", { code: reason });
}

function CharacterRow({ c }: { c: SummaryCharacter }) {
  const { t } = useI18n();
  const total = Math.max(1, c.lines);
  const vp = (c.voiced_lines / total) * 100;
  const mode = c.mode;
  return (
    <tr data-testid={`game-char-${c.speaker_id}`} data-mode={mode}>
      <th scope="row">
        {c.name_tgt || c.name_src || c.speaker_id}
        {c.name_src && c.name_tgt && c.name_src !== c.name_tgt ? <span className="muted"> ({c.name_src})</span> : null}
      </th>
      <td className="num">{c.lines}</td>
      <td>
        <div className="split-bar" role="img" aria-label={t("game.characters.split_label", { voiced: c.voiced_lines, unvoiced: c.unvoiced_lines })}>
          <span className="split-voiced" style={{ width: `${vp}%` }} />
        </div>
        <span className="hint">{t("game.characters.split_label", { voiced: c.voiced_lines, unvoiced: c.unvoiced_lines })}</span>
      </td>
      <td className="num">{fmtMinutes(c.minutes)}</td>
      <td>
        <Chip tone={mode === "voiced" ? "good" : mode === "mixed" ? "warn" : "bad"}>{t(`game.characters.mode.${mode}`)}</Chip>
      </td>
    </tr>
  );
}
