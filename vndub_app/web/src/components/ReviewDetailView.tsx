import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import { api, type JsonObject, type ReviewActionName } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import { DEFAULT_TUNING, type Candidate, type ErrorInfo, type ReviewDetail, type Tuning, type Verdict } from "../api/types";
import { useI18n } from "../i18n";
import { fmtNumber } from "../util/format";
import { ErrorPanel } from "./ErrorPanel";
import { VoiceTuningPanel, withDefaults } from "./VoiceTuning";
import { WavePlayer, type WavePlayerHandle } from "./WavePlayer";
import { Button, Callout, Chip, Collapsible, SelectField, TextField } from "./ui";

export interface ReviewDetailHandle {
  toggle: () => void;
  playOriginal: () => void;
  playCandidate: (index: number) => void;
  accept: () => void;
  regenerate: () => void;
}

const EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted", "embarrassed"];

function VerdictChip({ v }: { v: Verdict }) {
  const { t, has } = useI18n();
  const key = `verdict.${v.metric}.${v.key}`;
  const label = has(key) ? t(key) : `${v.metric}: ${v.key}`;
  const tipKey = `verdict.tooltip.${v.metric}`;
  const numbers = { ...v.tooltip_params, value: v.value === null || v.value === undefined ? "-" : fmtNumber(v.value, 2) };
  const tip = has(tipKey) ? t(tipKey, numbers) : "";
  return (
    <span className={`chip chip-verdict level-${v.level}`} title={tip} tabIndex={0} data-testid={`verdict-${v.metric}`} data-level={v.level}>
      {label}
      {tip ? <span className="sr-only"> ({tip})</span> : null}
    </span>
  );
}

interface Props {
  lineId: string;
  /** an action was applied on the server: the list should refresh; `advance` asks to go to the next line */
  onChanged: (action: ReviewActionName, advance: boolean) => void;
}

export const ReviewDetailView = forwardRef<ReviewDetailHandle, Props>(function ReviewDetailView({ lineId, onChanged }, ref) {
  const { t, has } = useI18n();
  const [detail, setDetail] = useState<ReviewDetail | null>(null);
  const [error, setError] = useState<ErrorInfo | null>(null);
  const [actionError, setActionError] = useState<ErrorInfo | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [textTgt, setTextTgt] = useState("");
  const [textSrc, setTextSrc] = useState("");
  const [style, setStyle] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [tuning, setTuning] = useState<Tuning>({ ...DEFAULT_TUNING });
  const [emotion, setEmotion] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const players = useRef<Record<string, WavePlayerHandle | null>>({});
  const lastPlayed = useRef<string>("original");
  const reqId = useRef(0);

  const load = useCallback(async () => {
    const id = ++reqId.current;
    try {
      const d = await api.reviewLine(lineId);
      if (id !== reqId.current) return;
      setDetail(d);
      setError(null);
      setTextTgt(d.line.text_tgt ?? "");
      setTextSrc(d.line.text_src ?? "");
      setStyle(d.style_prompt ?? "");
      setTuning(withDefaults(d.tuning_override));
      setEmotion("");
      const winner = d.candidates.find((c) => c.is_winner) ?? d.candidates[0];
      setSelected(winner ? winner.id : null);
      lastPlayed.current = "original";
    } catch (e) {
      if (id === reqId.current) setError(toErrorInfo(e));
    }
  }, [lineId]);

  useEffect(() => {
    setDetail(null);
    setActionError(null);
    setNotice(null);
    void load();
  }, [load]);

  const act = useCallback(
    async (action: ReviewActionName, payload: JsonObject = {}, advance = false) => {
      setBusy(action);
      setActionError(null);
      setNotice(null);
      try {
        await api.reviewAction(lineId, action, payload);
        onChanged(action, advance);
        if (!advance) {
          await load();
          setNotice(t("review.action_saved"));
        }
      } catch (e) {
        setActionError(toErrorInfo(e));
      } finally {
        setBusy(null);
      }
    },
    [lineId, onChanged, load, t],
  );

  const cands = detail?.candidates ?? [];
  const selectedCand = cands.find((c) => c.id === selected) ?? null;

  const accept = useCallback(() => {
    if (!selectedCand) return;
    void act("accept", { candidate: selectedCand.id }, true);
  }, [act, selectedCand]);

  const regenerate = useCallback(() => {
    const payload: JsonObject = {};
    if (detail && textTgt !== detail.line.text_tgt) payload.text = textTgt;
    if (detail && style !== (detail.style_prompt ?? "")) payload.style_prompt = style;
    void act("regen", payload);
  }, [act, detail, textTgt, style]);

  useImperativeHandle(
    ref,
    () => ({
      toggle: () => players.current[lastPlayed.current]?.toggle(),
      playOriginal: () => {
        lastPlayed.current = "original";
        players.current.original?.toggle();
      },
      playCandidate: (i: number) => {
        const c = cands[i];
        if (!c) return;
        setSelected(c.id);
        lastPlayed.current = c.id;
        players.current[c.id]?.toggle();
      },
      accept,
      regenerate,
    }),
    [cands, accept, regenerate],
  );

  if (error) return <ErrorPanel error={error} onRetry={() => void load()} />;
  if (!detail) return <p className="loading" role="status">{t("app.loading")}</p>;

  const line = detail.line;
  const fromAsr = line.flags?.includes("text_from_asr");
  const textDelivery = line.delivery?.source === "text";
  const layer = line.delivery?.layer ?? "neutral";
  const textChanged = textTgt !== line.text_tgt;

  const setEmotionNote = (e: string) => {
    setEmotion(e);
    setTuning((cur) => ({ ...cur, character_note: e ? t("review.emotion_note", { emotion: t(`emotion.${e}`) }) : "" }));
  };

  return (
    <div className="review-detail" data-testid="review-detail" data-line={line.line_id}>
      <header className="review-detail-head">
        <h2>
          {line.name || line.speaker_id}
          <span className="muted"> {line.scene ? ` / ${line.scene}` : ""}</span>
        </h2>
        <div className="chip-line">
          <Chip tone={line.status === "ok" ? "good" : line.status === "check" ? "warn" : "bad"} testid="review-detail-status">
            {t(`review.status.${line.status}`)}
          </Chip>
          {line.composite !== null && line.composite !== undefined ? <Chip title={t("review.composite_title")}>{t("review.composite", { v: fmtNumber(line.composite, 2) })}</Chip> : null}
          {line.touched ? <Chip tone="info">{t("review.touched")}</Chip> : null}
          {fromAsr ? (
            <Chip tone="warn" testid="review-asr-badge" title={t("review.asr_badge_title")}>
              {t("review.asr_badge")}
            </Chip>
          ) : null}
          {textDelivery ? (
            <Chip tone="accent" testid="review-delivery-chip">
              {t("review.delivery.text", { layer: has(`review.delivery.layer.${layer}`) ? t(`review.delivery.layer.${layer}`) : layer })}
            </Chip>
          ) : null}
          <code className="muted">{line.line_id}</code>
        </div>
      </header>

      {/* ---- why this line is here ---- */}
      {detail.why.length > 0 ? (
        <Callout tone="warn" title={t("review.why.title")} testid="review-why">
          <ul>
            {detail.why.map((w, i) => (
              <li key={`${w.key}-${i}`}>{has(`why.${w.key}`) ? t(`why.${w.key}`, w.params) : t("why.unknown", { gate: w.gate })}</li>
            ))}
          </ul>
        </Callout>
      ) : null}

      {/* ---- texts ---- */}
      <div className="review-texts">
        {fromAsr ? (
          <div data-testid="review-src-edit">
            <Callout tone="warn">{t("review.asr_warning")}</Callout>
            <TextField label={t("review.text_src")} value={textSrc} onChange={setTextSrc} multiline rows={2} testid="review-text-src" />
            <Button small onClick={() => void act("set_src_text", { text: textSrc })} busy={busy === "set_src_text"} disabled={textSrc === line.text_src} data-testid="review-save-src">
              {t("review.save_src")}
            </Button>
          </div>
        ) : (
          <div>
            <p className="label">{t("review.text_src")}</p>
            <p className="review-src" lang="ja" data-testid="review-text-src-readonly">
              {line.text_src}
            </p>
          </div>
        )}
        <TextField label={t("review.text_tgt")} value={textTgt} onChange={setTextTgt} multiline rows={3} testid="review-text-tgt" hint={detail.text_tts && detail.text_tts !== textTgt ? t("review.text_tts", { text: detail.text_tts }) : undefined} />
        {textChanged ? (
          <div className="row-actions">
            <Button small onClick={() => void act("set_text", { text: textTgt })} busy={busy === "set_text"} data-testid="review-save-text">
              {t("review.save_text")}
            </Button>
            <Button small variant="ghost" onClick={() => setTextTgt(line.text_tgt)}>
              {t("review.revert_text")}
            </Button>
          </div>
        ) : null}
      </div>

      {/* ---- original ---- */}
      <div className="review-original">
        <WavePlayer
          ref={(h) => {
            players.current.original = h;
          }}
          url={detail.original_url}
          kind="review-original"
          label={t("player.original")}
          showLabel
          testid="review-original"
          onPlay={() => {
            lastPlayed.current = "original";
          }}
        />
        {!detail.original_url ? <p className="hint">{t("review.no_original")}</p> : null}
      </div>

      {/* ---- candidates ---- */}
      {cands.length === 0 ? (
        <Callout tone="info" testid="review-no-candidates">
          {t("review.no_candidates")}
        </Callout>
      ) : (
        <ul className="candidates" data-testid="review-candidates">
          {cands.slice(0, 3).map((c, i) => (
            <CandidateCard
              key={c.id}
              c={c}
              index={i}
              selected={selected === c.id}
              busy={busy === "accept"}
              playerRef={(h) => {
                players.current[c.id] = h;
              }}
              onSelect={() => setSelected(c.id)}
              onPlay={() => {
                lastPlayed.current = c.id;
                setSelected(c.id);
              }}
              onAccept={() => void act("accept", { candidate: c.id }, true)}
            />
          ))}
        </ul>
      )}

      {/* ---- actions ---- */}
      <div className="review-actions" data-testid="review-actions">
        <Button variant="primary" onClick={accept} disabled={!selectedCand} busy={busy === "accept"} data-testid="review-accept">
          {t("review.accept")}
        </Button>
        <Button onClick={regenerate} busy={busy === "regen"} data-testid="review-regen">
          {t("review.regen")}
        </Button>
        <Button onClick={() => selectedCand && void act("reject", { candidate: selectedCand.id }, true)} disabled={!selectedCand} busy={busy === "reject"} data-testid="review-reject">
          {t("review.reject")}
        </Button>
        <Button onClick={() => void act("keep_original", {}, true)} busy={busy === "keep_original"} disabled={!detail.original_url} data-testid="review-keep-original">
          {t("review.keep_original")}
        </Button>
      </div>
      {notice ? (
        <p className="hint" role="status" data-testid="review-notice">
          {notice}
        </p>
      ) : null}
      {actionError ? <ErrorPanel error={actionError} onDismiss={() => setActionError(null)} /> : null}

      {/* ---- style, other reference, per-line tuning ---- */}
      <Collapsible title={t("review.style.title")} testid="review-style-toggle">
        <TextField label={t("review.style.label")} value={style} onChange={setStyle} multiline rows={2} testid="review-style" hint={t("review.style.hint")} />
      </Collapsible>

      {detail.refs.length > 0 ? (
        <Collapsible title={t("review.refs.title")} testid="review-refs-toggle">
          <p className="hint">{t("review.refs.hint")}</p>
          <ul className="refs">
            {detail.refs.map((r) => (
              <li key={r.path}>
                <div className="chip-line">
                  {r.emotion ? <Chip>{has(`emotion.${r.emotion}`) ? t(`emotion.${r.emotion}`) : r.emotion}</Chip> : null}
                </div>
                <WavePlayer url={r.url} kind="review-ref" label={t("review.refs.sample")} compact />
                <Button small onClick={() => void act("set_ref", { ref: r.path })} busy={busy === "set_ref"} data-testid="review-use-ref">
                  {t("review.refs.use")}
                </Button>
              </li>
            ))}
          </ul>
        </Collapsible>
      ) : null}

      <VoiceTuningPanel
        value={tuning}
        onChange={setTuning}
        onReset={() => {
          setTuning({ ...DEFAULT_TUNING });
          setEmotion("");
          void act("tuning", { ...DEFAULT_TUNING });
        }}
        lineMode
        prefix="review-line"
      >
        {textDelivery ? (
          <SelectField
            label={t("review.emotion")}
            value={emotion}
            options={[{ value: "", label: t("review.emotion_as_is") }, ...EMOTIONS.map((e) => ({ value: e, label: t(`emotion.${e}`) }))]}
            onChange={setEmotionNote}
            testid="review-emotion"
          />
        ) : null}
        <div className="row-actions">
          <Button variant="primary" onClick={() => void act("tuning", { ...tuning })} busy={busy === "tuning"} data-testid="review-line-tuning-apply">
            {t("review.tuning_apply")}
          </Button>
        </div>
      </VoiceTuningPanel>
    </div>
  );
});

function CandidateCard({
  c,
  index,
  selected,
  busy,
  playerRef,
  onSelect,
  onPlay,
  onAccept,
}: {
  c: Candidate;
  index: number;
  selected: boolean;
  busy: boolean;
  playerRef: (h: WavePlayerHandle | null) => void;
  onSelect: () => void;
  onPlay: () => void;
  onAccept: () => void;
}) {
  const { t, tOr } = useI18n();
  return (
    <li className={`candidate ${selected ? "is-selected" : ""} ${c.is_winner ? "is-winner" : ""}`} data-testid={`candidate-${c.label}`} data-selected={selected} onClick={onSelect}>
      <div className="candidate-head">
        <Chip tone={c.is_winner ? "good" : "neutral"} testid={`candidate-label-${c.label}`}>
          {t("review.candidate", { label: c.label })}
        </Chip>
        {c.is_winner ? <Chip tone="good">{t("review.winner")}</Chip> : null}
        {!c.passed ? <Chip tone="warn">{t("review.not_passed")}</Chip> : null}
        <span className="muted">
          {tOr(`models.${c.backend}.name`, c.backend)}, {t("review.route", { route: c.route })}
        </span>
        <span className="key-hint" aria-hidden="true">
          {index + 2 <= 4 ? `[${index + 2}]` : ""}
        </span>
      </div>
      <WavePlayer ref={playerRef} url={c.url} kind="review-candidate" label={t("review.candidate", { label: c.label })} compact testid={`candidate-player-${c.label}`} onPlay={onPlay} />
      <div className="verdicts" data-testid={`candidate-verdicts-${c.label}`}>
        {c.verdicts.map((v) => (
          <VerdictChip key={v.metric} v={v} />
        ))}
      </div>
      <div className="row-actions">
        <Button small onClick={onAccept} busy={busy} data-testid={`candidate-accept-${c.label}`}>
          {t("review.accept_this", { label: c.label })}
        </Button>
      </div>
    </li>
  );
}
