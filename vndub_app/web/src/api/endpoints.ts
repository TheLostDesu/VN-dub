// Typed wrappers over the HTTP API of vndub_app/API.md. Everything the contract leaves ambiguous
// (payload shapes of review actions, request bodies of PUT endpoints) is decided here, in one place.
import { http } from "./client";
import type {
  AppState,
  Character,
  Design,
  DoctorReport,
  EnvRow,
  CustomModel,
  GpuInfo,
  Glossary,
  HelpDoc,
  HfInspect,
  HfRecommended,
  ImportPreview,
  InstallPlan,
  JobInfo,
  JobLog,
  ModelsReport,
  PatchCoverage,
  PreviewItem,
  ProjectInfo,
  ProjectLite,
  ProjectSummary,
  ReviewActionResult,
  ReviewDetail,
  ReviewLinesResponse,
  Settings,
  ToolRow,
  Tuning,
  TranslationSettings,
  VoiceLibraryEntry,
} from "./types";

export type JsonObject = Record<string, unknown>;

/** Answer of POST /api/doctor/fix/add_defender_exclusion_help (nothing is changed on the machine). */
export interface DefenderHelp {
  ok: boolean;
  path: string;
  command: string;
  settings_path_key: string;
  admin_required: boolean;
  applicable: boolean;
}

export type ReviewActionName =
  | "accept"
  | "reject"
  | "regen"
  | "set_text"
  | "set_ref"
  | "keep_original"
  | "set_src_text"
  | "tuning";

export const api = {
  // ---- general ----------------------------------------------------------------------------
  health: () => http.get<{ ok: boolean; version: string; profile: string; mock: boolean }>("/api/health"),
  settings: () => http.get<Settings>("/api/settings"),
  putSettings: (patch: Partial<Settings>) => http.put<Settings>("/api/settings", patch),
  setDataDir: (path: string) => http.post<Settings>("/api/settings/data-dir", { path }),
  pickFolder: (purpose: string) => http.post<{ path: string | null; unsupported?: boolean }>("/api/system/pick-folder", { purpose }),
  pickFile: (purpose: string, filters?: string[]) =>
    http.post<{ path: string | null; unsupported?: boolean }>("/api/system/pick-file", { purpose, filters }),
  openFolder: (kind: "data_dir" | "logs" | "patch" | "backups" | "work" | "path", path?: string) =>
    http.post<{ ok: boolean }>("/api/system/open", path ? { kind, path } : { kind }),
  report: () => http.post<JobInfo>("/api/system/report"),
  help: () => http.get<HelpDoc[]>("/api/help"),
  helpDoc: (id: string) => http.get<{ markdown: string }>(`/api/help/${encodeURIComponent(id)}`),
  gpu: () => http.get<GpuInfo>("/api/gpu"),

  // ---- jobs ---------------------------------------------------------------------------------
  job: (id: string) => http.get<JobInfo & { log_tail?: string[] }>(`/api/jobs/${encodeURIComponent(id)}`),
  jobLog: (id: string) => http.get<JobLog>(`/api/jobs/${encodeURIComponent(id)}/log`),
  jobs: (active?: boolean) => http.get<JobInfo[]>("/api/jobs", { active: active ? 1 : undefined }),
  pauseJob: (id: string) => http.post<JobInfo>(`/api/jobs/${encodeURIComponent(id)}/pause`),
  resumeJob: (id: string) => http.post<JobInfo>(`/api/jobs/${encodeURIComponent(id)}/resume`),
  cancelJob: (id: string) => http.post<JobInfo>(`/api/jobs/${encodeURIComponent(id)}/cancel`),

  // ---- project ------------------------------------------------------------------------------
  projects: () => http.get<ProjectLite[]>("/api/projects"),
  openProject: (target: { game_dir: string } | { slug: string }) => http.post<ProjectInfo>("/api/project/open", target),
  project: () => http.get<ProjectInfo>("/api/project"),
  putState: (patch: Partial<AppState> | JsonObject) => http.put<AppState>("/api/project/state", patch),
  /** continue the interrupted job; `keep_shutdown` keeps its "turn the computer off when done" (dropped otherwise) */
  resumeProject: (opts: { keep_shutdown?: boolean } = {}) => http.post<JobInfo>("/api/project/resume", opts.keep_shutdown ? { keep_shutdown: true } : {}),
  importGame: () => http.post<JobInfo>("/api/project/import"),
  recoverText: () => http.post<JobInfo>("/api/project/recover-text"),
  asrCheck: () => http.post<JobInfo>("/api/project/asr-check"),
  summary: () => http.get<ProjectSummary>("/api/project/summary"),
  runStage: (name: string, body: { only?: string[]; limit?: number; force?: boolean; sets?: JsonObject } = {}) =>
    http.post<JobInfo>(`/api/stages/${encodeURIComponent(name)}/run`, body),

  // ---- translation & settings (Step 2) -------------------------------------------------------
  translation: () => http.get<TranslationSettings>("/api/project/translation"),
  putTranslation: (value: TranslationSettings) => http.put<TranslationSettings>("/api/project/translation", value),
  previewImport: (path: string) => http.post<ImportPreview>("/api/project/translation/preview-import", { path }),
  glossary: () => http.get<Glossary>("/api/project/glossary"),
  putGlossary: (value: Glossary) => http.put<Glossary>("/api/project/glossary", value),
  translate: () => http.post<JobInfo>("/api/project/translate"),
  deliveryCues: () => http.get<unknown>("/api/project/delivery-cues"),
  putDeliveryCues: (value: unknown) => http.put<unknown>("/api/project/delivery-cues", value),

  // ---- casting (Step 3 / 3b) -----------------------------------------------------------------
  characters: () => http.get<Character[]>("/api/characters"),
  putCharacter: (sid: string, patch: Partial<Pick<Character, "gender" | "age_hint" | "description" | "name_tgt">>) =>
    http.put<Character>(`/api/characters/${encodeURIComponent(sid)}`, patch),
  proposals: (sid: string) => http.get<Design[]>(`/api/casting/${encodeURIComponent(sid)}/proposals`),
  generateDesigns: (sid: string, n = 3, description?: string) =>
    http.post<JobInfo>(`/api/casting/${encodeURIComponent(sid)}/generate`, description ? { n, description } : { n }),
  uploadSample: (sid: string, file: File) => http.upload<Design>(`/api/casting/${encodeURIComponent(sid)}/upload`, file),
  choose: (sid: string, design_id: string, route: string) =>
    http.post<Character>(`/api/casting/${encodeURIComponent(sid)}/choose`, { design_id, route }),
  // a GPU job (kind casting_auto): proposes designs for characters without any, picks the top-ranked voice for
  // everybody, queues rvc_train for route B. result = {chosen: {sid: design_id}, generated: [sid]}
  autoCast: (n?: number) => http.post<JobInfo>("/api/casting/auto", n ? { n } : {}),
  listen: (sid: string, n = 3) => http.post<JobInfo>(`/api/casting/${encodeURIComponent(sid)}/listen`, { n }),
  voices: () => http.get<VoiceLibraryEntry[]>("/api/voices"),
  saveVoice: (speaker_id: string, name: string) => http.post<VoiceLibraryEntry>("/api/voices", { speaker_id, name }),
  fromLibrary: (sid: string, voice_id: string) =>
    http.post<Design>(`/api/casting/${encodeURIComponent(sid)}/from-library`, { voice_id }),
  tuning: (sid: string) => http.get<Tuning>(`/api/characters/${encodeURIComponent(sid)}/tuning`),
  putTuning: (sid: string, tuning: Tuning) => http.put<Tuning>(`/api/characters/${encodeURIComponent(sid)}/tuning`, tuning),
  tuningListen: (sid: string, tuning: Tuning) =>
    http.post<JobInfo>(`/api/characters/${encodeURIComponent(sid)}/tuning/listen`, { tuning }),
  tuningApplySimilar: (sid: string) =>
    http.post<{ updated: string[] }>(`/api/characters/${encodeURIComponent(sid)}/tuning/apply-similar`),

  // ---- preview / run ---------------------------------------------------------------------------
  previewRun: (mode: "first_scene" | "first_n", n: number) => http.post<JobInfo>("/api/preview/run", { mode, n }),
  previewItems: () => http.get<{ line_ids: string[]; items: PreviewItem[] }>("/api/preview/items"),
  runFull: (shutdown_when_done: boolean) => http.post<JobInfo>("/api/run/full", { shutdown_when_done }),

  // ---- review (Step 6) --------------------------------------------------------------------------
  reviewLines: (q: { speaker?: string; scene?: string; status?: string; q?: string; min?: number; max?: number }) =>
    http.get<ReviewLinesResponse>("/api/review/lines", q),
  reviewLine: (id: string) => http.get<ReviewDetail>(`/api/review/lines/${encodeURIComponent(id)}`),
  // Payload shapes (API.md fixes only set_src_text and tuning; the others follow vndub_app/reviewdata.py):
  // accept / reject {candidate: <candidate.id>}, regen {text?, style_prompt?}, set_text {text, style_prompt?},
  // set_ref {ref: <refs[].path>}, keep_original {}, set_src_text {text}, tuning {Tuning fields}.
  reviewAction: (id: string, action: ReviewActionName, payload: JsonObject = {}) =>
    http.post<ReviewActionResult>(`/api/review/lines/${encodeURIComponent(id)}/action`, { action, payload }),
  preferRoute: (route: string, speaker_id?: string) =>
    http.post<{ updated: number }>("/api/review/prefer-route", speaker_id ? { route, speaker_id } : { route }),
  reviewApply: () => http.post<JobInfo>("/api/review/apply"),

  // ---- doctor / tools / models / install --------------------------------------------------------
  doctor: () => http.get<DoctorReport>("/api/doctor"),
  doctorFix: (action: string) => http.post<JobInfo | DefenderHelp | { ok: boolean; path?: string; url?: string; killed?: number }>(`/api/doctor/fix/${encodeURIComponent(action)}`),
  tools: () => http.get<ToolRow[]>("/api/tools"),
  installTools: (names: string[]) => http.post<JobInfo>("/api/tools/install", { names }),
  envs: () => http.get<EnvRow[]>("/api/envs"),
  installEnvs: (envs: string[], torch: "auto" | "cuda" | "cpu") => http.post<JobInfo>("/api/envs/install", { envs, torch }),
  models: () => http.get<ModelsReport>("/api/models"),
  downloadModels: (names: string[]) => http.post<JobInfo>("/api/models/download", { names }),
  checkModels: (names: string[]) => http.post<JobInfo>("/api/models/check", { names }),
  // ---- models by Hugging Face link ----
  hfInspect: (url: string) => http.post<HfInspect>("/api/models/hf/inspect", { url }),
  hfAdd: (body: { repo: string; revision: string; files: string[]; kind: string; label?: string }) => http.post<JobInfo>("/api/models/hf/add", body),
  hfRecommended: () => http.get<{ items: HfRecommended[]; checked: string | null }>("/api/models/hf/recommended"),
  customModels: () => http.get<CustomModel[]>("/api/models/custom"),
  deleteCustomModel: (id: string) => http.del<{ ok: boolean }>(`/api/models/custom/${encodeURIComponent(id)}`),
  installPlan: () => http.get<InstallPlan>("/api/install/plan"),
  /** what the built patch covers; null while there is no patch (the finished full run carries the same in `result.coverage`) */
  patchCoverage: () => http.get<{ coverage: PatchCoverage | null }>("/api/patch/coverage"),
  installApply: (backup: boolean) => http.post<JobInfo>("/api/install/apply", { backup }),
  installRollback: (backup_id: string) => http.post<JobInfo>("/api/install/rollback", { backup_id }),
  installZip: () => http.post<JobInfo>("/api/install/zip"),
};

/** Is this response a JobInfo (as opposed to a plain `{ok}`)? */
export function isJob(v: unknown): v is JobInfo {
  return typeof v === "object" && v !== null && "id" in v && "state" in v && "kind" in v;
}
