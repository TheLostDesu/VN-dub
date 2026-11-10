import { useEffect, useState } from "react";
import { api } from "../api/endpoints";
import { useI18n } from "../i18n";
import { Callout, Chip } from "./ui";

/** What the project summary knows about a character's gender: the value and who set it (absent source = not recorded). */
export type GenderInfo = Record<string, { gender: string; source: string | null }>;

/**
 * Gender (and where it came from) per speaker, from the project summary. The summary's `gender_source` is a newer field: an
 * older server (or no summary yet) gives an empty map, and the screens then simply show no "guessed" marks.
 */
export function useGenderInfo(refreshKey: unknown = 0): GenderInfo {
  const [info, setInfo] = useState<GenderInfo>({});
  useEffect(() => {
    let alive = true;
    api
      .summary()
      .then((s) => {
        if (!alive) return;
        const out: GenderInfo = {};
        for (const c of s.characters ?? []) {
          const row = c as { speaker_id: string; gender?: string; gender_source?: string | null };
          out[row.speaker_id] = { gender: row.gender ?? "", source: row.gender_source ?? null };
        }
        setInfo(out);
      })
      .catch(() => alive && setInfo({}));
    return () => {
      alive = false;
    };
  }, [refreshKey]);
  return info;
}

/** True when the gender shown is still the one the model guessed (a later edit by the user makes it the user's own). */
export function isGuessed(info: GenderInfo, speakerId: string, gender: string): boolean {
  const e = info[speakerId];
  return !!e && e.source === "guess" && e.gender === gender && gender !== "unknown" && gender !== "";
}

export function GuessChip({ testid }: { testid?: string }) {
  const t = useI18n().t;
  return (
    <Chip tone="info" title={t("gender.guessed_title")} testid={testid}>
      {t("gender.guessed")}
    </Chip>
  );
}

/** A note above the character lists: which characters have no gender yet (the translation then avoids gendered forms). */
export function GenderMissingNote({ items, testid }: { items: { name: string; gender: string }[]; testid?: string }) {
  const t = useI18n().t;
  const missing = items.filter((i) => !i.gender || i.gender === "unknown");
  if (missing.length === 0) return null;
  const shown = missing.slice(0, 8).map((i) => i.name);
  const names = missing.length > shown.length ? `${shown.join(", ")} ${t("gender.missing_more", { n: missing.length - shown.length })}` : shown.join(", ");
  return (
    <Callout tone="info" testid={testid ?? "gender-missing-note"}>
      {t("gender.missing", { n: missing.length, names })}
    </Callout>
  );
}
