// Tells the user a job ended while the window was not in front: the page title gets a "[готово]" / "[ошибка]" prefix
// until the window is looked at again. One place for the whole app (App.tsx starts it), not per screen.
import { useEffect, useRef } from "react";
import { useI18n } from "../i18n";
import { onJobFinal } from "./progress";

export function useJobAttention(): void {
  const { t } = useI18n();
  const tRef = useRef(t);
  tRef.current = t;

  useEffect(() => {
    let base: string | null = null; // the title to come back to
    const restore = () => {
      if (base !== null && !document.hidden) {
        document.title = base;
        base = null;
      }
    };
    const off = onJobFinal((job) => {
      if (!document.hidden) return;
      // a cancelled job was stopped by the user, an interrupted one by a restart: neither needs a signal
      const key = job.state === "done" ? "app.title_done" : job.state === "failed" ? "app.title_failed" : null;
      if (!key) return;
      if (base === null) base = document.title;
      document.title = tRef.current(key);
    });
    document.addEventListener("visibilitychange", restore);
    return () => {
      off();
      document.removeEventListener("visibilitychange", restore);
      if (base !== null) document.title = base;
    };
  }, []);
}
