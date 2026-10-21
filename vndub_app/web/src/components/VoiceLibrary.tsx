import { useCallback, useEffect, useState } from "react";
import { api } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import type { Character, ErrorInfo, VoiceLibraryEntry } from "../api/types";
import { useI18n } from "../i18n";
import { ErrorPanel } from "./ErrorPanel";
import { WavePlayer } from "./WavePlayer";
import { Button, Chip, SelectField, Spinner, useToast } from "./ui";

/** "Библиотека голосов": performer designs saved from earlier projects. */
export function VoiceLibrary({ characters, onImported }: { characters: Character[]; onImported: (sid: string) => void }) {
  const { t, tOr } = useI18n();
  const toast = useToast();
  const [voices, setVoices] = useState<VoiceLibraryEntry[] | null>(null);
  const [error, setError] = useState<ErrorInfo | null>(null);
  const [target, setTarget] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setVoices(await api.voices());
      setError(null);
    } catch (e) {
      setError(toErrorInfo(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const first = characters[0]?.speaker_id ?? "";
  const use = async (v: VoiceLibraryEntry) => {
    const sid = target[v.id] ?? first;
    if (!sid) return;
    setBusy(v.id);
    try {
      await api.fromLibrary(sid, v.id);
      onImported(sid);
      const ch = characters.find((c) => c.speaker_id === sid);
      toast.show(t("casting.library.added", { name: ch?.name_tgt || ch?.name_src || sid }));
    } catch (e) {
      setError(toErrorInfo(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="voice-library" data-testid="voice-library">
      <p className="hint">{t("casting.library.intro")}</p>
      {voices === null && !error ? <Spinner label={t("app.loading")} /> : null}
      {error ? <ErrorPanel error={error} onRetry={() => void load()} /> : null}
      {voices !== null && voices.length === 0 ? <p data-testid="library-empty">{t("casting.library.empty")}</p> : null}
      <ul className="library-list">
        {(voices ?? []).map((v) => (
          <li key={v.id} className="library-item" data-testid={`library-item-${v.id}`}>
            <div className="library-head">
              <strong>{v.name}</strong>
              {v.gender ? <Chip>{tOr(`gender.${v.gender}`, v.gender)}</Chip> : null}
              {v.source_project ? <span className="muted">{t("casting.library.from", { project: v.source_project })}</span> : null}
            </div>
            {v.description ? <p className="hint">{v.description}</p> : null}
            <WavePlayer url={v.sample_url} kind="library" label={v.name} compact testid={`library-player-${v.id}`} />
            <div className="row-actions">
              <SelectField
                label={t("casting.library.use_for")}
                value={target[v.id] ?? first}
                options={characters.map((c) => ({ value: c.speaker_id, label: c.name_tgt || c.name_src || c.speaker_id }))}
                onChange={(sid) => setTarget((cur) => ({ ...cur, [v.id]: sid }))}
                testid={`library-target-${v.id}`}
              />
              <Button variant="primary" busy={busy === v.id} onClick={() => void use(v)} disabled={characters.length === 0} data-testid={`library-use-${v.id}`}>
                {t("casting.library.use")}
              </Button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
