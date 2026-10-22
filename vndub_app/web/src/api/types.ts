// Wire types of the VNDub backend. The authoritative contract is vndub_app/API.md.

export type StepId = "doctor" | "game" | "translate" | "casting" | "preview" | "run" | "review" | "install";

export const STEP_IDS: StepId[] = ["doctor", "game", "translate", "casting", "preview", "run", "review", "install"];

// ---- errors ---------------------------------------------------------------------------------

export interface ErrorInfo {
  code: string;
  params: Record<string, unknown>;
  action: string;
  details: string;
}

// ---- settings -------------------------------------------------------------------------------

/** Effective absolute folders (GET /api/settings → `paths`); `python` / `uv_cache` are uv's, always below `envs`. */
export interface SettingsPaths {
  data: string;
  models: string;
  envs: string;
  tools: string;
  python: string;
  uv_cache: string;
}

export interface Settings {
  ui_lang: string;
  profile: string;
  gpu_budget_gb: number;
  license_policy: string;
  /** "" = <data>/models */
  models_dir: string;
  /** "" = <data>/envs (uv's Python builds and download cache live in <envs>/_uv) */
  envs_dir?: string;
  paths?: SettingsPaths;
  player_volume: Record<string, number>;
  last_project: string | null;
  shutdown_when_done: boolean;
  /** "" = the default translation model; else a registry `llm` backend name (built in or added by link) */
  translation_llm?: string;
  /** write DEBUG lines to <data>/logs (absent until the backend knows the setting: read it as `?? false`) */
  debug_log?: boolean;
  data_dir?: string;
  [extra: string]: unknown;
}

// ---- jobs -----------------------------------------------------------------------------------

export type JobState = "queued" | "running" | "paused" | "done" | "failed" | "cancelled" | "interrupted";

export interface JobInfo {
  id: string;
  kind: string;
  title_key: string;
  params: Record<string, unknown>;
  project: string | null;
  gpu: boolean;
  cancel_requested?: boolean;
  state: JobState;
  created: string;
  started: string | null;
  finished: string | null;
  stages: string[];
  stage: string | null;
  stage_index: number;
  stage_count: number;
  /**
   * A regen round in progress: score found lines without a passing candidate and synth + score run again for them
   * inside the quality step (`stage_index` stays at score). 0 / absent outside a round; `round_total` = the rounds the
   * config allows, `round_lines` = the lines this round re-does.
   */
  stage_round?: number;
  round_total?: number;
  round_lines?: number;
  done: number;
  total: number;
  item: string | null;
  eta_s: number | null;
  stage_eta_s: number | null;
  rate_per_min: number | null;
  message_key: string | null;
  message_params: Record<string, unknown>;
  error: ErrorInfo | null;
  /** may hold `warnings` (see JobWarning): what the run skipped or substituted */
  result: Record<string, unknown>;
  /** app settings at submit time (gpu_budget_gb, translation_llm, license_policy, models_dir, envs_dir, profile, src_lang, tgt_lang); {} on old records */
  settings?: Record<string, unknown>;
  log_tail?: string[];
}

/**
 * One entry of `JobInfo.result.warnings`. `kind`: `substitution` (requested, used, reason), `unavailable` (stage, group,
 * reason), `mixed_models` (models: "a: 812, b: 140"), `settings_changed` (keys: "gpu_budget_gb, translation_llm").
 */
export interface JobWarning {
  kind: string;
  requested?: string;
  used?: string;
  stage?: string;
  group?: string;
  reason?: string;
  models?: string;
  keys?: string;
}

/** GET /api/jobs/{id}/log: the whole per-job log file (or the tail the server keeps when the file is gone). */
export interface JobLog {
  path: string | null;
  lines: string[];
}

export const ACTIVE_STATES: JobState[] = ["queued", "running", "paused"];
export const FINAL_STATES: JobState[] = ["done", "failed", "cancelled", "interrupted"];

export type ProgressEvent =
  | { type: "hello"; jobs: JobInfo[] }
  | { type: "job"; job: JobInfo }
  | { type: "log"; job_id: string; line: string }
  | { type: "gpu"; used_gb: number; total_gb: number };

// ---- project --------------------------------------------------------------------------------

export interface AppState {
  step: StepId;
  completed: StepId[];
  translation: {
    tgt_lang: string;
    /** "auto" (detect from the game text) or a language code */
    src_lang: string;
    source: "auto" | "import";
    import_path: string;
    honorifics: "drop" | "keep" | "adapt";
    address_default: string;
  };
  advanced: { asr_check: boolean; [k: string]: unknown };
  config_overrides: Record<string, unknown>;
  preview: { mode: "first_scene" | "first_n"; n: number; line_ids: string[] };
  review: { touched: string[] };
  updated: string;
  [extra: string]: unknown;
}

export interface StageRecord {
  started?: string | null;
  finished?: string | null;
  error?: string | null;
  elapsed_s?: number;
  summary?: Record<string, unknown>;
  [extra: string]: unknown;
}

export interface ResumeInfo {
  available: boolean;
  /** the job a resume continues: the newest interrupted one among the heavy kinds, else the newest */
  job: JobInfo | null;
  next_stage: string | null;
  /** the other interrupted jobs (they stay interrupted and are offered next time); absent on an older server */
  candidates?: JobInfo[];
}

export interface ProjectInfo {
  slug: string;
  name: string;
  game_dir: string;
  engine: string | null;
  project_yaml: Record<string, unknown> | null;
  state: AppState;
  stage_status: Record<string, StageRecord>;
  active_job: JobInfo | null;
  resume: ResumeInfo;
}

export interface ProjectLite {
  slug: string;
  name: string;
  game_dir: string;
  engine: string | null;
  step: StepId | string;
  updated: string;
}

export interface EngineInfo {
  id: string | null;
  /** proper name of a known engine; null for the generic fallback (the UI uses engines.<id>) */
  label: string | null;
  version?: string | null;
  encrypted: boolean;
  drm: string[];
  warnings: string[];
}

export interface MissingTool {
  tool: string;
  url?: string;
  pip?: string;
  manual_key?: string;
}

export interface SummaryCharacter {
  speaker_id: string;
  name_src: string;
  name_tgt?: string;
  lines: number;
  voiced_lines: number;
  unvoiced_lines: number;
  minutes: number;
  mode: "voiced" | "unvoiced" | "mixed";
}

export interface ProblemFile {
  line_id: string;
  status: string;
  reasons: string[];
}

export interface ProjectSummary {
  engine: EngineInfo;
  tools_missing: MissingTool[];
  voice_files: number;
  lines_total: number;
  voiced: number;
  unvoiced: number;
  characters: SummaryCharacter[];
  mapping: {
    mode: string;
    ok: number;
    review: number;
    unmatched: number;
    confidence: number | null;
    suggest_asr_check: boolean;
  };
  // Either the list of voiced line ids without text or just their number (the contract says "no_text_lines").
  no_text_lines: string[] | number;
  problem_files: ProblemFile[];
  analysis: { audio_passports: number; text_passports: number };
  /** languages of the game: `src` is null before the import, `src_detected` is true when it came from the detector */
  languages?: { src: string | null; src_detected: boolean; audio: string | null; tgt: string };
}

/** What the language detector found in the parsed script lines. */
export interface SrcLangDetected {
  lang: string;
  confidence: number;
  letters: number;
  lines: number;
}

// ---- translation & settings -----------------------------------------------------------------

export interface BackendRow {
  name: string;
  enabled: boolean;
  vram_gb: number | null;
  license: string;
  commercial_ok: boolean | null;
  role: string;
}

export interface TranslationSettings {
  tgt_lang: string;
  /** the user's choice: "auto" or a language code (auto, ja, en, zh, ko, ru, uk, de, fr, es, it, pt) */
  src_lang: string;
  /** the code the pipeline uses right now (the detected one when the choice is auto) */
  src_lang_effective: string;
  /** null before the game was imported or when nothing could be detected */
  src_lang_detected: SrcLangDetected | null;
  source: "auto" | "import";
  import_path: string;
  honorifics: "drop" | "keep" | "adapt";
  address_default: string;
  license_policy: string;
  llm_available: boolean;
  advanced: {
    n_candidates: number;
    routes: { A: boolean; B: boolean; C: boolean };
    asr_check: boolean;
    backends: BackendRow[];
  };
}

export interface ImportPreview {
  format: string;
  matched: number;
  total: number;
  preview: { line_id: string; text_src: string; text_tgt: string }[];
}

export interface Glossary {
  characters: { speaker_id: string; src: string; tgt: string; gender: "m" | "f" | "unknown" }[];
  terms: { src: string; tgt: string; note: string }[];
}

// ---- casting --------------------------------------------------------------------------------

export interface Tuning {
  pitch_st: number;
  speed: number;
  expressiveness: number;
  energy: number;
  timbre_preset: "" | "younger" | "older" | "softer" | "sharper";
  character_note: string;
}

export const DEFAULT_TUNING: Tuning = {
  pitch_st: 0,
  speed: 1,
  expressiveness: 0.5,
  energy: 0,
  timbre_preset: "",
  character_note: "",
};

export interface Clip {
  kind: "neutral" | "emotional" | "longest";
  line_id: string;
  url: string;
  text_src: string;
  emotion: string | null;
}

export interface Chosen {
  design_id: string;
  route: string;
  kind: string;
}

export interface RvcInfo {
  state: "none" | "queued" | "training" | "done" | "failed";
  progress: number;
}

export interface Character {
  speaker_id: string;
  name_src: string;
  name_tgt: string;
  gender: "m" | "f" | "unknown";
  age_hint: string;
  description: string;
  lines: number;
  voiced_lines: number;
  minutes: number;
  mode: "voiced" | "unvoiced" | "mixed";
  emotion_hist: Record<string, number>;
  clips: Clip[];
  chosen: Chosen | null;
  tuning: Tuning;
  rvc: RvcInfo;
}

export interface DesignSample {
  line_id: string;
  wav?: string;
  text: string;
  text_tgt?: string;
  url: string;
}

export interface Design {
  design_id: string;
  kind: "voice_design" | "reference" | "clone";
  backend: string;
  prompt: string;
  description: string;
  samples: DesignSample[];
  spk_sim: number | null;
  f0_delta_st: number | null;
  stars: number | null;
  rank_score: number;
  chosen?: boolean;
  license: string;
  commercial_ok: boolean | null;
  created: string;
}

export interface VoiceLibraryEntry {
  id: string;
  name: string;
  gender: string;
  description: string;
  sample_url: string;
  source_project: string;
}

export interface ListenItem {
  line_id: string;
  text_src: string;
  text_tgt: string;
  original_url: string | null;
  result_url: string;
}

export interface TuningListenItem {
  line_id: string;
  kind: "neutral" | "emotional";
  url: string;
  text_tgt: string;
}

// ---- preview / review -----------------------------------------------------------------------

export interface PreviewItem {
  line_id: string;
  speaker_id: string;
  name: string;
  text_src: string;
  text_tgt: string;
  original_url: string | null;
  dub_url: string | null;
  status: string;
}

export type ReviewStatus = "ok" | "check" | "failed";

export interface ReviewLine {
  line_id: string;
  speaker_id: string;
  name: string;
  scene: string;
  status: ReviewStatus;
  composite: number | null;
  text_src: string;
  text_tgt: string;
  winner_label: string | null;
  flags: string[];
  delivery: { source: "audio" | "text"; layer?: string } | null;
  touched: boolean;
}

export interface Verdict {
  metric: "voice" | "emotion" | "clarity" | "length" | "quality";
  key: string;
  level: "good" | "ok" | "bad";
  value: number | null;
  tooltip_params: Record<string, unknown>;
}

export interface Candidate {
  id: string;
  label: string;
  url: string;
  backend: string;
  route: string;
  composite: number | null;
  passed: boolean;
  is_winner: boolean;
  verdicts: Verdict[];
}

export interface WhyItem {
  gate: string;
  key: string;
  params: Record<string, unknown>;
}

export interface ReviewRef {
  path: string;
  url: string;
  emotion: string | null;
}

export interface ReviewDetail {
  line: ReviewLine;
  original_url: string | null;
  candidates: Candidate[];
  why: WhyItem[];
  style_prompt: string;
  text_tts: string;
  refs: ReviewRef[];
  tuning_override: Partial<Tuning> | null;
}

export interface ReviewFacets {
  speakers: unknown[];
  scenes: unknown[];
}

// ---- doctor / tools / models / install ------------------------------------------------------

export interface DoctorRow {
  id: string;
  status: "green" | "yellow" | "red";
  /** may be a plain value or an object (driver, envs, models rows): only primitives are interpolated */
  value: unknown;
  params: Record<string, unknown>;
  fix: { action: string; label_key: string } | null;
}

export interface DoctorReport {
  summary: "ok" | "warn" | "fail";
  gpu: { name: string; vram_gb: number; driver: string; cuda: string } | null;
  rows: DoctorRow[];
  data_dir?: string;
  mock?: boolean;
}

export interface ToolRow {
  name: string;
  version: string;
  present: boolean;
  required_for: string[];
  size_mb: number;
  url: string;
  filename: string;
  sha256: string;
  manual: boolean;
}

export interface EnvRow {
  name: string;
  present: boolean;
  size_gb: number;
  torch: "cuda" | "cpu" | null;
}

export interface ModelRow {
  name: string;
  role: string;
  env: string;
  size_gb: number;
  vram_gb: number | null;
  license: string;
  commercial_ok: boolean | null;
  present: boolean;
  usable: boolean;
  /** code of models.reason.<code> ("" when usable) */
  reason: string | null;
  reason_params?: Record<string, unknown>;
  presets: string[];
  selftest: { ok: boolean; peak_vram_gb: number | null; at: string } | null;
}

/** Kind of a model found by link (`POST /api/models/hf/inspect`). */
export type HfKind = "builtin" | "gguf_llm" | "rvc" | "files";

export interface HfWarning {
  code: string;
  params: Record<string, unknown>;
}

export interface HfInspect {
  repo: string;
  revision: string;
  sha: string;
  page_url: string;
  files: { name: string; size: number }[];
  total_size: number;
  license: string | null;
  gated: boolean;
  private: boolean;
  kind: HfKind;
  suggested: string[];
  suggested_size: number;
  /** name of the built-in backend (row of the models table) when the repo is one the app already lists */
  builtin: string | null;
  warnings: HfWarning[];
  free_gb: number;
  focus: string | null;
}

/** A model added by link (`GET /api/models/custom`). */
export interface CustomModel {
  id: string;
  repo: string;
  revision: string;
  files: string[];
  kind: Exclude<HfKind, "builtin">;
  dir: string;
  size_gb: number;
  license: string | null;
  label: string;
  added_at: string;
  present: boolean;
  disk_gb: number;
  /** can the pipeline use it right now (gguf_llm only; null for other kinds) */
  usable: boolean | null;
  in_use: boolean;
}

export interface HfRecommended {
  repo: string;
  url: string;
  kind: string;
  file: string;
  size_gb: number;
  license: string;
  title: Record<string, string>;
  note: Record<string, string>;
}

export interface ModelsReport {
  rows: ModelRow[];
  presets: { min: string[]; recommended: string[]; all: string[] };
  disk: { free_gb: number; data_dir: string };
}

export interface InstallPlan {
  engine: string;
  auto_possible: boolean;
  manual_reason_key: string | null;
  install_md: string;
  files: { target: string; source: string; exists: boolean }[];
  patch_dir: string;
  backups: { id: string; created: string; files: number | unknown[] }[];
}

export interface HelpDoc {
  id: string;
  title_key: string;
}

export interface GpuInfo {
  name?: string;
  total_gb?: number;
  used_gb?: number;
  available?: boolean;
}
