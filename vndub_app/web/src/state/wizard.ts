import { useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { STEP_IDS, type StepId } from "../api/types";
import { useProject } from "./project";

export const STEP_NUMBER: Record<StepId, number> = {
  doctor: 0,
  game: 1,
  translate: 2,
  casting: 3,
  preview: 4,
  run: 5,
  review: 6,
  install: 7,
};

export function stepPath(step: StepId): string {
  return `/step/${step}`;
}

/** Which wizard step shows a job of this kind. */
export function stepForJobKind(kind: string): StepId | null {
  switch (kind) {
    case "import":
    case "recover_text":
    case "asr_check":
      return "game";
    case "translate":
      return "translate";
    case "casting_auto":
    case "casting_generate":
    case "casting_listen":
    case "tuning_listen":
    case "rvc_train":
      return "casting";
    case "preview":
      return "preview";
    case "full":
    case "stage":
      return "run";
    case "review_apply":
      return "review";
    case "install_apply":
    case "install_rollback":
    case "install_zip":
      return "install";
    case "ffmpeg_download":
    case "vcredist_install":
      return "doctor";
    default:
      return null;
  }
}

/** Route that shows a job of this kind: a wizard step or one of the plain pages (the "show" button of the job bar). */
export function pathForJobKind(kind: string): string | null {
  const step = stepForJobKind(kind);
  if (step) return stepPath(step);
  if (kind.startsWith("models")) return "/models"; // models_download, models_check, models_hf_add
  if (kind.startsWith("tools") || kind.startsWith("envs")) return "/tools"; // tools_install, envs_install
  if (kind === "report") return "/settings"; // the report button of the Settings screen
  return null;
}

/** Furthest step the user may open: everything before it, plus the next one after the last completed. */
export function furthestStep(step: StepId, completed: StepId[]): number {
  const done = completed.map((s) => STEP_NUMBER[s]);
  const maxDone = done.length ? Math.max(...done) : 0;
  return Math.max(STEP_NUMBER[step], maxDone + 1);
}

export function useWizard() {
  const nav = useNavigate();
  const { project, step, completed, saveState } = useProject();

  const goto = useCallback(
    (target: StepId, state?: unknown) => {
      nav(stepPath(target), state === undefined ? undefined : { state });
    },
    [nav],
  );

  /** Mark a step as finished (persisted in app_state.completed). */
  const complete = useCallback(
    async (s: StepId) => {
      if (!project) return;
      const next = Array.from(new Set([...completed, s]));
      if (next.length !== completed.length) await saveState({ completed: next });
    },
    [project, completed, saveState],
  );

  const next = useCallback(
    async (from: StepId, state?: unknown) => {
      await complete(from);
      const i = STEP_IDS.indexOf(from);
      const target = STEP_IDS[Math.min(STEP_IDS.length - 1, i + 1)] as StepId;
      goto(target, state);
    },
    [complete, goto],
  );

  const back = useCallback(
    (from: StepId) => {
      const i = STEP_IDS.indexOf(from);
      goto(STEP_IDS[Math.max(0, i - 1)] as StepId);
    },
    [goto],
  );

  return { step, completed, goto, next, back, complete };
}
