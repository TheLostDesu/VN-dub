import { useCallback, useEffect, useRef, useState } from "react";
import { toErrorInfo } from "../api/client";
import type { ErrorInfo } from "../api/types";

export type SaveStatus = "idle" | "saving" | "saved" | "error";

/**
 * Debounced saving of a whole value. `schedule(v)` restarts the timer, `flush()` saves at once and says whether the
 * values are on the server (await it before starting a job that depends on the saved values: `false` = the save failed,
 * do not start). Pending changes are written when the component unmounts; a failure then has no screen to show it, so
 * `onLeaveError` (a toast) is called instead.
 */
export function useAutosave<T>(save: (value: T) => Promise<unknown>, delay = 700, onLeaveError?: () => void) {
  const [status, setStatus] = useState<SaveStatus>("idle");
  const [error, setError] = useState<ErrorInfo | null>(null);
  const pending = useRef<{ value: T } | null>(null);
  const timer = useRef<number | null>(null);
  const saveRef = useRef(save);
  saveRef.current = save;
  const leaveErrorRef = useRef(onLeaveError);
  leaveErrorRef.current = onLeaveError;
  const mounted = useRef(true);

  const run = useCallback(async (): Promise<boolean> => {
    const p = pending.current;
    if (!p) return true;
    pending.current = null;
    if (mounted.current) setStatus("saving");
    try {
      await saveRef.current(p.value);
      if (mounted.current) {
        setStatus("saved");
        setError(null);
      }
      return true;
    } catch (e) {
      // keep the value (unless a newer one arrived) so that flush() and the save on leave try it again
      if (!pending.current) pending.current = p;
      if (mounted.current) {
        setStatus("error");
        setError(toErrorInfo(e));
      }
      return false;
    }
  }, []);

  const schedule = useCallback(
    (value: T) => {
      pending.current = { value };
      if (timer.current !== null) window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => {
        timer.current = null;
        void run();
      }, delay);
    },
    [delay, run],
  );

  const flush = useCallback(async (): Promise<boolean> => {
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
    return run();
  }, [run]);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      if (timer.current !== null) window.clearTimeout(timer.current);
      const p = pending.current;
      if (p) {
        pending.current = null;
        void saveRef.current(p.value).catch(() => leaveErrorRef.current?.());
      }
    };
  }, []);

  return { status, error, schedule, flush };
}
