import { createContext, useCallback, useContext, useMemo, type ReactNode } from "react";
import en from "@i18n/en.json";
import ru from "@i18n/ru.json";

// One flat table per language (dotted keys, `{param}` interpolation). The tables are imported at build
// time from app/i18n/{ru,en}.json: the same files the tests check for completeness.

export type Lang = "ru" | "en";
export type Params = Record<string, unknown>;
export type TFunc = (key: string, params?: Params) => string;

const TABLES: Record<Lang, Record<string, string>> = { ru: ru as Record<string, string>, en: en as Record<string, string> };

export const LANGS: Lang[] = ["ru", "en"];
export const DEFAULT_LANG: Lang = "ru";

const warned = new Set<string>();

export function normalizeLang(value: unknown): Lang {
  return value === "en" ? "en" : "ru";
}

export function hasKey(lang: Lang, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(TABLES[lang], key);
}

function interpolate(template: string, params?: Params): string {
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (whole, name: string) => {
    const v = params[name];
    return v === undefined || v === null ? whole : String(v);
  });
}

export function translate(lang: Lang, key: string, params?: Params): string {
  const table = TABLES[lang];
  if (Object.prototype.hasOwnProperty.call(table, key)) return interpolate(table[key] as string, params);
  if (import.meta.env.DEV && !warned.has(`${lang}:${key}`)) {
    warned.add(`${lang}:${key}`);
    console.warn(`[i18n] missing key "${key}" (${lang})`);
  }
  return key; // visible fallback
}

/** Marks a string literal as a translation key without translating it (the i18n test collects these). */
export function tk(key: string): string {
  return key;
}

interface I18nValue {
  lang: Lang;
  t: TFunc;
  /** Translate if the key exists, otherwise return `fallback` (no warning). */
  tOr: (key: string | null | undefined, fallback: string, params?: Params) => string;
  has: (key: string) => boolean;
}

const I18nContext = createContext<I18nValue>({
  lang: DEFAULT_LANG,
  t: (k, p) => translate(DEFAULT_LANG, k, p),
  tOr: (k, f, p) => (k && hasKey(DEFAULT_LANG, k) ? translate(DEFAULT_LANG, k, p) : f),
  has: (k) => hasKey(DEFAULT_LANG, k),
});

export function I18nProvider({ lang, children }: { lang: Lang; children: ReactNode }) {
  const t = useCallback<TFunc>((key, params) => translate(lang, key, params), [lang]);
  const value = useMemo<I18nValue>(
    () => ({
      lang,
      t,
      has: (k) => hasKey(lang, k),
      tOr: (k, f, p) => (k && hasKey(lang, k) ? translate(lang, k, p) : f),
    }),
    [lang, t],
  );
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nValue {
  return useContext(I18nContext);
}

export function useT(): TFunc {
  return useContext(I18nContext).t;
}
