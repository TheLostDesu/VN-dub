import { useCallback, useEffect, useRef, useState } from "react";
import { toErrorInfo } from "../api/client";
import type { ErrorInfo } from "../api/types";

export type SaveStatus = "idle" | "saving" | "saved" | "error";

/**
 * Debounced saving of a whole value. `schedule(v)` restarts the timer, `flush()` saves at once
 * (await it before starting a job that depends on the saved values). Pending changes are written
 * when the component unmounts.
 */
export function useAutosave<T>(save: (value: T) => Promise<unknown>, delay = 700) {
  const [status, setStatus] = useState<SaveStatus>("idle");
  const [error, setError] = useState<ErrorInfo | null>(null);
  const pending = useRef<{ value: T } | null>(null);
  const timer = useRef<number | null>(null);
  const saveRef = useRef(save);
  saveRef.current = save;
  const mounted = useRef(true);

  const run = useCallback(async () => {
    const p = pending.current;
    if (!p) return;
    pending.current = null;
    if (mounted.current) setStatus("saving");
    try {
      await saveRef.current(p.value);
      if (mounted.current) {
        setStatus("saved");
        setError(null);
      }
    } catch (e) {
      if (mounted.current) {
        setStatus("error");
        setError(toErrorInfo(e));
      }
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

  const flush = useCallback(async () => {
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
    await run();
  }, [run]);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      if (timer.current !== null) window.clearTimeout(timer.current);
      const p = pending.current;
      if (p) {
        pending.current = null;
        void saveRef.current(p.value).catch(() => undefined);
      }
    };
  }, []);

  return { status, error, schedule, flush };
}
