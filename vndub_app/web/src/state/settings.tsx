import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import type { ErrorInfo, Settings } from "../api/types";
import { DEFAULT_LANG, I18nProvider, normalizeLang, translate, type Lang } from "../i18n";
import { ErrorPanel } from "../components/ErrorPanel";

interface SettingsValue {
  settings: Settings;
  lang: Lang;
  /** PUT /api/settings with a partial object; the result replaces the local copy. */
  update: (patch: Partial<Settings>) => Promise<Settings | null>;
  /** Remember the volume of one kind of player (debounced). */
  setPlayerVolume: (kind: string, volume: number) => void;
  reload: () => Promise<void>;
}

const FALLBACK: Settings = {
  ui_lang: DEFAULT_LANG,
  profile: "default",
  gpu_budget_gb: 7.5,
  license_policy: "research",
  models_dir: "",
  player_volume: {},
  last_project: null,
  shutdown_when_done: false,
};

const Ctx = createContext<SettingsValue | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [error, setError] = useState<ErrorInfo | null>(null);
  const volumeTimer = useRef<number | null>(null);
  const settingsRef = useRef<Settings | null>(null);
  settingsRef.current = settings;

  const reload = useCallback(async () => {
    try {
      setSettings(await api.settings());
      setError(null);
    } catch (e) {
      setError(toErrorInfo(e));
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  const update = useCallback(async (patch: Partial<Settings>) => {
    // optimistic: the language and volumes should react at once
    setSettings((s) => (s ? { ...s, ...patch } : s));
    try {
      const next = await api.putSettings(patch);
      setSettings((s) => ({ ...(s ?? FALLBACK), ...next }));
      return next;
    } catch (e) {
      setError(toErrorInfo(e));
      return null;
    }
  }, []);

  const setPlayerVolume = useCallback(
    (kind: string, volume: number) => {
      const cur = settingsRef.current;
      if (!cur) return;
      const player_volume = { ...cur.player_volume, [kind]: volume };
      setSettings({ ...cur, player_volume });
      if (volumeTimer.current !== null) window.clearTimeout(volumeTimer.current);
      volumeTimer.current = window.setTimeout(() => {
        api.putSettings({ player_volume }).catch(() => undefined);
      }, 600);
    },
    [],
  );

  const value = useMemo<SettingsValue | null>(
    () =>
      settings
        ? { settings, lang: normalizeLang(settings.ui_lang), update, setPlayerVolume, reload }
        : null,
    [settings, update, setPlayerVolume, reload],
  );

  useEffect(() => {
    document.documentElement.lang = value?.lang ?? DEFAULT_LANG;
  }, [value?.lang]);

  if (!value) {
    return (
      <I18nProvider lang={DEFAULT_LANG}>
        <div className="splash" data-testid="splash">
          {error ? (
            <div className="splash-card">
              <h1>VNDub</h1>
              <ErrorPanel error={error} onRetry={() => void reload()} />
            </div>
          ) : (
            <p role="status">{translate(DEFAULT_LANG, "app.loading")}</p>
          )}
        </div>
      </I18nProvider>
    );
  }
  return (
    <Ctx.Provider value={value}>
      <I18nProvider lang={value.lang}>{children}</I18nProvider>
    </Ctx.Provider>
  );
}

export function useSettings(): SettingsValue {
  const v = useContext(Ctx);
  if (!v) throw new Error("useSettings outside SettingsProvider");
  return v;
}
