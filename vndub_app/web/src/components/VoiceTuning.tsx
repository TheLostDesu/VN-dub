import { useState, type ReactNode } from "react";
import { DEFAULT_TUNING, type Tuning, type TuningListenItem } from "../api/types";
import { useI18n } from "../i18n";
import { WavePlayer } from "./WavePlayer";
import { Button, Chip, Slider, TextField } from "./ui";

export function withDefaults(v: Partial<Tuning> | null | undefined): Tuning {
  return { ...DEFAULT_TUNING, ...(v ?? {}) };
}

const PRESETS: Exclude<Tuning["timbre_preset"], "">[] = ["younger", "older", "softer", "sharper"];

/**
 * Step 3b: a handful of sliders and presets, nothing more. Controlled: the parent decides what
 * "change" means (save to the character, or write an override for one line).
 */
export function VoiceTuningPanel({
  value,
  onChange,
  onReset,
  onListen,
  listenBusy,
  listenItems,
  onApplySimilar,
  lineMode,
  prefix,
  children,
}: {
  value: Tuning;
  onChange: (v: Tuning) => void;
  onReset: () => void;
  onListen?: () => void;
  listenBusy?: boolean;
  listenItems?: TuningListenItem[] | null;
  onApplySimilar?: () => void;
  lineMode?: boolean;
  prefix: string;
  children?: ReactNode;
}) {
  const { t } = useI18n();
  const [open, setOpen] = useState(true);
  const set = (patch: Partial<Tuning>) => onChange({ ...value, ...patch });
  const sign = (v: number) => (v > 0 ? `+${v}` : String(v));

  return (
    <section className="tuning-panel" data-testid={`${prefix}-tuning`} aria-label={lineMode ? t("tuning.title_line") : t("tuning.title")}>
      <h4 className="tuning-title">
        <button type="button" className="collapsible-toggle" aria-expanded={open} onClick={() => setOpen(!open)} data-testid={`${prefix}-tuning-toggle`}>
          <span className="chevron" aria-hidden="true" />
          {lineMode ? t("tuning.title_line") : t("tuning.title")}
        </button>
      </h4>
      {open ? (
        <div className="tuning-body">
          {lineMode ? <p className="hint">{t("tuning.line_hint")}</p> : null}
          <div className="tuning-sliders">
            <Slider label={t("tuning.pitch")} value={value.pitch_st} min={-6} max={6} step={0.5} onChange={(v) => set({ pitch_st: v })} format={(v) => `${sign(v)} ${t("tuning.unit_st")}`} minLabel={t("tuning.lower")} maxLabel={t("tuning.higher")} testid={`${prefix}-pitch`} />
            <Slider label={t("tuning.speed")} value={value.speed} min={0.8} max={1.2} step={0.05} onChange={(v) => set({ speed: Math.round(v * 100) / 100 })} format={(v) => `${v.toFixed(2)}x`} minLabel={t("tuning.slower")} maxLabel={t("tuning.faster")} testid={`${prefix}-speed`} />
            <Slider label={t("tuning.expressiveness")} value={value.expressiveness} min={0} max={1} step={0.05} onChange={(v) => set({ expressiveness: Math.round(v * 100) / 100 })} format={(v) => `${Math.round(v * 100)}%`} minLabel={t("tuning.calm")} maxLabel={t("tuning.vivid")} testid={`${prefix}-expressiveness`} />
            <Slider label={t("tuning.energy")} value={value.energy} min={-1} max={1} step={0.1} onChange={(v) => set({ energy: Math.round(v * 10) / 10 })} format={(v) => (v === 0 ? t("tuning.energy_normal") : sign(Math.round(v * 10) / 10).toString())} minLabel={t("tuning.softer")} maxLabel={t("tuning.louder")} testid={`${prefix}-energy`} />
          </div>

          <div className="chip-row" role="group" aria-label={t("tuning.timbre")}>
            <span className="chip-row-label">{t("tuning.timbre")}</span>
            {PRESETS.map((p) => (
              <button
                key={p}
                type="button"
                className={`chip-btn ${value.timbre_preset === p ? "is-on" : ""}`}
                aria-pressed={value.timbre_preset === p}
                onClick={() => set({ timbre_preset: value.timbre_preset === p ? "" : p })}
                data-testid={`${prefix}-timbre-${p}`}
              >
                {t(`tuning.timbre.${p}`)}
              </button>
            ))}
          </div>

          <TextField label={t("tuning.note")} value={value.character_note} onChange={(v) => set({ character_note: v })} placeholder={t("tuning.note_placeholder")} testid={`${prefix}-note`} />

          {children}

          <div className="row-actions">
            {onListen ? (
              <Button variant="primary" busy={listenBusy} onClick={onListen} data-testid={`${prefix}-listen`}>
                {t("tuning.listen")}
              </Button>
            ) : null}
            <Button onClick={onReset} data-testid={`${prefix}-reset`}>
              {t("tuning.reset")}
            </Button>
            {onApplySimilar ? (
              <Button onClick={onApplySimilar} data-testid={`${prefix}-apply-similar`}>
                {t("tuning.apply_similar")}
              </Button>
            ) : null}
          </div>

          {listenItems && listenItems.length > 0 ? (
            <ul className="tuning-results" data-testid={`${prefix}-tuning-results`}>
              {listenItems.map((it) => (
                <li key={`${it.line_id}-${it.kind}`}>
                  <div className="tuning-result-head">
                    <Chip tone={it.kind === "emotional" ? "accent" : "neutral"}>{t(`tuning.sample.${it.kind}`)}</Chip>
                    <span className="sample-text">{it.text_tgt}</span>
                  </div>
                  <WavePlayer url={it.url} kind="tuning" label={t(`tuning.sample.${it.kind}`)} compact />
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
