import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ApiError, toErrorInfo } from "../api/client";
import { api, type JsonObject } from "../api/endpoints";
import { FINAL_STATES, STEP_IDS, type AppState, type ErrorInfo, type ProjectInfo, type StepId } from "../api/types";
import { useProgressState } from "./progress";

interface ProjectValue {
  project: ProjectInfo | null;
  loading: boolean;
  error: ErrorInfo | null;
  reload: () => Promise<ProjectInfo | null>;
  open: (target: { game_dir: string } | { slug: string }) => Promise<ProjectInfo>;
  /** PUT /api/project/state (deep-merged by the server); the answer replaces the local copy. */
  saveState: (patch: JsonObject) => Promise<AppState | null>;
  step: StepId;
  completed: StepId[];
}

const Ctx = createContext<ProjectValue | null>(null);

function asStep(v: unknown): StepId {
  return (STEP_IDS as string[]).includes(String(v)) ? (v as StepId) : "game";
}

export function ProjectProvider({ children }: { children: ReactNode }) {
  const [project, setProject] = useState<ProjectInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ErrorInfo | null>(null);
  const progress = useProgressState();
  const handled = useRef<Set<string>>(new Set());
  const reloadTimer = useRef<number | null>(null);

  const reload = useCallback(async () => {
    try {
      const p = await api.project();
      setProject(p);
      setError(null);
      return p;
    } catch (e) {
      if (e instanceof ApiError && (e.info.code === "no_project" || e.status === 404)) {
        setProject(null);
        setError(null);
      } else {
        setError(toErrorInfo(e));
      }
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  // A finished job changes stage_status / state / resume: refresh the project once per job.
  useEffect(() => {
    let fresh = false;
    for (const j of Object.values(progress.jobs)) {
      if (FINAL_STATES.includes(j.state)) {
        const key = `${j.id}:${j.state}`;
        if (!handled.current.has(key)) {
          handled.current.add(key);
          fresh = true;
        }
      }
    }
    if (fresh) {
      if (reloadTimer.current !== null) window.clearTimeout(reloadTimer.current);
      reloadTimer.current = window.setTimeout(() => void reload(), 400);
    }
  }, [progress.jobs, reload]);

  const open = useCallback(async (target: { game_dir: string } | { slug: string }) => {
    const p = await api.openProject(target);
    setProject(p);
    setError(null);
    return p;
  }, []);

  const saveState = useCallback(async (patch: JsonObject) => {
    try {
      const next = await api.putState(patch);
      setProject((p) => (p ? { ...p, state: next } : p));
      return next;
    } catch (e) {
      setError(toErrorInfo(e));
      return null;
    }
  }, []);

  const value = useMemo<ProjectValue>(
    () => ({
      project,
      loading,
      error,
      reload,
      open,
      saveState,
      step: asStep(project?.state?.step),
      completed: (project?.state?.completed ?? []).map(asStep),
    }),
    [project, loading, error, reload, open, saveState],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useProject(): ProjectValue {
  const v = useContext(Ctx);
  if (!v) throw new Error("useProject outside ProjectProvider");
  return v;
}
