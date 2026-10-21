import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import type { ErrorInfo, ReviewFacets, ReviewLine, ReviewStatus } from "../api/types";
import { ErrorPanel } from "../components/ErrorPanel";
import { JobProgress } from "../components/JobProgress";
import { ReviewDetailView, type ReviewDetailHandle } from "../components/ReviewDetailView";
import { StepFooter } from "../components/StepFooter";
import { VirtualList } from "../components/VirtualList";
import { Button, Callout, Card, Chip, PageTitle, SelectField, Spinner, useToast } from "../components/ui";
import { useI18n } from "../i18n";
import { useJobTracker } from "../state/progress";
import { fmtNumber, normalizeFacets } from "../util/format";

const ROW_H = 64;

function isTextTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
}

function isActivatable(el: EventTarget | null): boolean {
  return el instanceof HTMLElement && (el.tagName === "BUTTON" || el.tagName === "A" || el.tagName === "SUMMARY" || el.getAttribute("role") === "tab");
}

export default function ReviewScreen() {
  const { t } = useI18n();
  const toast = useToast();
  const [speaker, setSpeaker] = useState("");
  const [scene, setScene] = useState("");
  const [status, setStatus] = useState<"" | ReviewStatus>("");
  const [query, setQuery] = useState("");
  const [debouncedQ, setDebouncedQ] = useState("");
  const [minC, setMinC] = useState("");
  const [maxC, setMaxC] = useState("");
  const [items, setItems] = useState<ReviewLine[] | null>(null);
  const [facets, setFacets] = useState<ReviewFacets>({ speakers: [], scenes: [] });
  const [error, setError] = useState<ErrorInfo | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [applyResult, setApplyResult] = useState<{ changed: number; lines: number } | null>(null);
  const [bulkBusy, setBulkBusy] = useState(false);
  const detail = useRef<ReviewDetailHandle>(null);
  const reqId = useRef(0);
  const itemsRef = useRef<ReviewLine[] | null>(null);
  itemsRef.current = items;
  const selectedRef = useRef<string | null>(null);
  selectedRef.current = selected;

  useEffect(() => {
    const id = window.setTimeout(() => setDebouncedQ(query.trim()), 300);
    return () => window.clearTimeout(id);
  }, [query]);

  const load = useCallback(async () => {
    const id = ++reqId.current;
    try {
      const res = await api.reviewLines({
        speaker: speaker || undefined,
        scene: scene || undefined,
        status: status || undefined,
        q: debouncedQ || undefined,
        min: minC !== "" && Number.isFinite(Number(minC)) ? Number(minC) : undefined,
        max: maxC !== "" && Number.isFinite(Number(maxC)) ? Number(maxC) : undefined,
      });
      if (id !== reqId.current) return;
      setItems(res.items ?? []);
      setFacets(res.facets ?? { speakers: [], scenes: [] });
      setError(null);
      const cur = selectedRef.current;
      if (!cur || !(res.items ?? []).some((i) => i.line_id === cur)) setSelected((res.items ?? [])[0]?.line_id ?? null);
    } catch (e) {
      if (id === reqId.current) setError(toErrorInfo(e));
    }
  }, [speaker, scene, status, debouncedQ, minC, maxC]);

  useEffect(() => {
    void load();
  }, [load]);

  const apply = useJobTracker({
    onDone: (j) => {
      const changed = Number(j.result?.changed ?? 0);
      const raw = j.result?.lines;
      const lines = Array.isArray(raw) ? raw.length : Number(raw ?? 0) || 0;
      setApplyResult({ changed, lines });
      void load();
    },
  });

  const index = useMemo(() => (items ?? []).findIndex((i) => i.line_id === selected), [items, selected]);
  const touchedCount = (items ?? []).filter((i) => i.touched).length;
  const speakers = normalizeFacets(facets.speakers);
  const scenes = normalizeFacets(facets.scenes);

  const move = useCallback((delta: number) => {
    const list = itemsRef.current ?? [];
    if (list.length === 0) return;
    const cur = list.findIndex((i) => i.line_id === selectedRef.current);
    const next = Math.max(0, Math.min(list.length - 1, (cur < 0 ? 0 : cur) + delta));
    setSelected((list[next] as ReviewLine).line_id);
  }, []);

  // After accept / reject / keep original: go to the next line of the current list, refresh in the background.
  const onChanged = useCallback(
    (_action: string, advance: boolean) => {
      if (advance) move(1);
      void load();
    },
    [move, load],
  );

  // Keyboard: Space play/pause, 1/2/3(/4) original/A/B/C, Enter accept, R regenerate, arrows previous/next.
  // Physical keys (e.code) are used so the shortcuts work on the Russian layout too.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (isTextTarget(e.target)) return;
      if (document.querySelector('[role="dialog"]')) return;
      const h = detail.current;
      switch (e.code) {
        case "Space":
          if (isActivatable(e.target)) return;
          e.preventDefault();
          h?.toggle();
          return;
        case "Digit1":
        case "Numpad1":
          e.preventDefault();
          h?.playOriginal();
          return;
        case "Digit2":
        case "Numpad2":
          e.preventDefault();
          h?.playCandidate(0);
          return;
        case "Digit3":
        case "Numpad3":
          e.preventDefault();
          h?.playCandidate(1);
          return;
        case "Digit4":
        case "Numpad4":
          e.preventDefault();
          h?.playCandidate(2);
          return;
        case "KeyR":
          e.preventDefault();
          h?.regenerate();
          return;
        case "ArrowDown":
        case "ArrowRight":
          e.preventDefault();
          move(1);
          return;
        case "ArrowUp":
        case "ArrowLeft":
          e.preventDefault();
          move(-1);
          return;
        default:
          break;
      }
      if (e.key === "Enter") {
        if (isActivatable(e.target)) return;
        e.preventDefault();
        h?.accept();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [move]);

  const preferA = async () => {
    setBulkBusy(true);
    try {
      const res = await api.preferRoute("A");
      toast.show(t("review.prefer_a_done", { n: res.updated }));
      await load();
    } catch (e) {
      setError(toErrorInfo(e));
    } finally {
      setBulkBusy(false);
    }
  };

  const counts = useMemo(() => {
    const c = { ok: 0, check: 0, failed: 0 };
    for (const i of items ?? []) c[i.status] += 1;
    return c;
  }, [items]);

  return (
    <>
      <PageTitle title={t("review.title")} intro={t("review.intro")} testid="review-title" />

      <Card testid="review-toolbar">
        <div className="review-filters">
          <SelectField label={t("review.filter.speaker")} value={speaker} options={[{ value: "", label: t("review.filter.all") }, ...speakers.map((s) => ({ value: s.id, label: s.name }))]} onChange={setSpeaker} testid="review-filter-speaker" />
          <SelectField label={t("review.filter.scene")} value={scene} options={[{ value: "", label: t("review.filter.all") }, ...scenes.map((s) => ({ value: s.id, label: s.name }))]} onChange={setScene} testid="review-filter-scene" />
          <SelectField
            label={t("review.filter.status")}
            value={status}
            options={[
              { value: "", label: t("review.filter.all") },
              { value: "check", label: t("review.status.check") },
              { value: "ok", label: t("review.status.ok") },
              { value: "failed", label: t("review.status.failed") },
            ]}
            onChange={(v) => setStatus(v as "" | ReviewStatus)}
            testid="review-filter-status"
          />
          <div className="field">
            <label htmlFor="rv-min">{t("review.filter.composite")}</label>
            <div className="range-pair">
              <input id="rv-min" type="number" min={0} max={1} step={0.05} value={minC} placeholder="0" aria-label={t("review.filter.min")} onChange={(e) => setMinC(e.target.value)} data-testid="review-filter-min" />
              <span aria-hidden="true">-</span>
              <input type="number" min={0} max={1} step={0.05} value={maxC} placeholder="1" aria-label={t("review.filter.max")} onChange={(e) => setMaxC(e.target.value)} data-testid="review-filter-max" />
            </div>
          </div>
          <div className="field grow">
            <label htmlFor="rv-q">{t("review.filter.search")}</label>
            <input id="rv-q" type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t("review.filter.search_placeholder")} data-testid="review-search" />
          </div>
        </div>
        <div className="row-actions">
          <Button onClick={() => void preferA()} busy={bulkBusy} data-testid="review-prefer-a" title={t("review.prefer_a_hint")}>
            {t("review.prefer_a")}
          </Button>
          <Button variant="primary" onClick={() => void apply.start(() => api.reviewApply())} busy={apply.busy} data-testid="review-apply">
            {t("review.apply")}
            {touchedCount > 0 ? <span className="count-badge" aria-label={t("review.touched_count", { n: touchedCount })}>{touchedCount}</span> : null}
          </Button>
          <span className="hint">{t("review.prefer_a_hint")}</span>
        </div>
        {apply.job ? <JobProgress job={apply.job} onRetry={() => void apply.start(() => api.reviewApply())} testid="review-apply-job" /> : null}
        {apply.error && !apply.job ? <ErrorPanel error={apply.error} /> : null}
        {applyResult && !apply.busy ? (
          <Callout tone="good" testid="review-apply-result" role="status">
            {t("review.apply_result", { changed: applyResult.changed, lines: applyResult.lines })}
          </Callout>
        ) : null}
      </Card>

      {error ? <ErrorPanel error={error} onRetry={() => void load()} /> : null}
      {items === null && !error ? <Spinner label={t("app.loading")} /> : null}

      {items ? (
        <div className="review-layout">
          <section className="review-list-pane" aria-label={t("review.list")}>
            <p className="hint" data-testid="review-counts" role="status">
              {t("review.counts", { total: items.length, check: counts.check, failed: counts.failed })}
            </p>
            {items.length === 0 ? (
              <Callout tone="info" testid="review-empty">
                {t("errors.nothing_to_review.title")}
              </Callout>
            ) : (
              <VirtualList
                items={items}
                rowHeight={ROW_H}
                keyOf={(i) => i.line_id}
                scrollToIndex={index}
                label={t("review.list")}
                testid="review-list"
                renderRow={(it) => (
                  <button type="button" className={`review-row ${it.line_id === selected ? "is-selected" : ""} status-${it.status}`} aria-current={it.line_id === selected ? "true" : undefined} onClick={() => setSelected(it.line_id)} data-testid={`review-row-${it.line_id}`} data-status={it.status}>
                    <span className={`status-dot status-${it.status === "ok" ? "green" : it.status === "check" ? "yellow" : "red"}`} aria-hidden="true" />
                    <span className="review-row-main">
                      <span className="review-row-name">
                        {it.name || it.speaker_id}
                        <span className="sr-only">: {t(`review.status.${it.status}`)}</span>
                      </span>
                      <span className="review-row-text">{it.text_tgt || it.text_src}</span>
                    </span>
                    <span className="review-row-meta">
                      {it.touched ? <Chip tone="info">{t("review.touched")}</Chip> : null}
                      {it.flags?.includes("text_from_asr") ? <Chip tone="warn">{t("review.asr_badge_short")}</Chip> : null}
                      {it.delivery?.source === "text" ? <Chip tone="accent">{t("review.delivery.short")}</Chip> : null}
                      {it.composite !== null && it.composite !== undefined ? <span className="muted">{fmtNumber(it.composite, 2)}</span> : null}
                    </span>
                  </button>
                )}
              />
            )}
          </section>

          <section className="review-detail-pane" aria-label={t("review.detail")}>
            {selected ? <ReviewDetailView key={selected} ref={detail} lineId={selected} onChanged={onChanged} /> : <p className="hint">{t("review.select_line")}</p>}
            <details className="keys-help">
              <summary>{t("review.keys.title")}</summary>
              <ul>
                <li>{t("review.keys.space")}</li>
                <li>{t("review.keys.digits")}</li>
                <li>{t("review.keys.enter")}</li>
                <li>{t("review.keys.r")}</li>
                <li>{t("review.keys.arrows")}</li>
              </ul>
            </details>
            <div className="row-actions">
              <Button small onClick={() => move(-1)} disabled={index <= 0} data-testid="review-prev">
                {t("review.prev")}
              </Button>
              <Button small onClick={() => move(1)} disabled={index < 0 || index >= items.length - 1} data-testid="review-next">
                {t("review.next")}
              </Button>
            </div>
          </section>
        </div>
      ) : null}

      <StepFooter step="review" canNext={items !== null && !apply.busy} nextLabel={t("review.to_install")} />
    </>
  );
}
