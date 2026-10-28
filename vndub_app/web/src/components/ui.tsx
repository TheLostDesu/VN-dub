import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type ReactNode,
} from "react";
import { useI18n, useT } from "../i18n";
import { clamp } from "../util/format";

// ---- Button ---------------------------------------------------------------------------------

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  small?: boolean;
  busy?: boolean;
};

export function Button({ variant = "secondary", small, busy, className, children, disabled, type, ...rest }: ButtonProps) {
  const cls = ["btn", `btn-${variant}`, small ? "btn-small" : "", busy ? "is-busy" : "", className ?? ""].filter(Boolean).join(" ");
  return (
    <button type={type ?? "button"} className={cls} disabled={disabled || busy} aria-busy={busy || undefined} {...rest}>
      {busy ? <span className="spinner" aria-hidden="true" /> : null}
      {children}
    </button>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="spinner-wrap" role="status">
      <span className="spinner" aria-hidden="true" />
      {label ? <span>{label}</span> : null}
    </span>
  );
}

// ---- Chip / badge / callout -----------------------------------------------------------------

export type Tone = "neutral" | "good" | "warn" | "bad" | "info" | "accent";

export function Chip({ tone = "neutral", children, title, testid }: { tone?: Tone; children: ReactNode; title?: string; testid?: string }) {
  return (
    <span className={`chip chip-${tone}`} title={title} data-testid={testid}>
      {children}
    </span>
  );
}

export function Callout({
  tone = "info",
  title,
  children,
  testid,
  role,
}: {
  tone?: Tone;
  title?: ReactNode;
  children?: ReactNode;
  testid?: string;
  role?: "alert" | "status";
}) {
  return (
    <div className={`callout callout-${tone}`} data-testid={testid} role={role}>
      {title ? <p className="callout-title">{title}</p> : null}
      {children}
    </div>
  );
}

export function LicenseBadge({ license, commercialOk }: { license: string; commercialOk: boolean | null | undefined }) {
  const t = useT();
  return (
    <span className="license-badges">
      <Chip tone="neutral" title={t("license.badge_title")}>
        {license || t("license.unknown")}
      </Chip>
      {commercialOk === false ? <Chip tone="warn">{t("license.non_commercial")}</Chip> : null}
      {commercialOk === true ? <Chip tone="good">{t("license.commercial_ok")}</Chip> : null}
    </span>
  );
}

// ---- Collapsible ----------------------------------------------------------------------------

export function Collapsible({
  title,
  children,
  defaultOpen = false,
  open: controlledOpen,
  onToggle,
  testid,
  badge,
  className,
  keepMounted,
}: {
  title: ReactNode;
  children: ReactNode;
  /** keep the content in the DOM (hidden) while collapsed */
  keepMounted?: boolean;
  defaultOpen?: boolean;
  open?: boolean;
  onToggle?: (open: boolean) => void;
  testid?: string;
  badge?: ReactNode;
  className?: string;
}) {
  const [inner, setInner] = useState(defaultOpen);
  const open = controlledOpen ?? inner;
  const id = useId();
  const toggle = () => {
    const next = !open;
    if (controlledOpen === undefined) setInner(next);
    onToggle?.(next);
  };
  return (
    <section className={`collapsible ${open ? "is-open" : ""} ${className ?? ""}`}>
      <h3 className="collapsible-head">
        <button type="button" className="collapsible-toggle" aria-expanded={open} aria-controls={id} onClick={toggle} data-testid={testid}>
          <span className="chevron" aria-hidden="true" />
          <span className="collapsible-title">{title}</span>
          {badge ? <span className="collapsible-badge">{badge}</span> : null}
        </button>
      </h3>
      <div id={id} className="collapsible-body" hidden={!open}>
        {open || keepMounted ? children : null}
      </div>
    </section>
  );
}

// ---- Dialog ---------------------------------------------------------------------------------

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function Dialog({
  open,
  title,
  onClose,
  children,
  actions,
  testid,
  wide,
}: {
  open: boolean;
  title: ReactNode;
  onClose: () => void;
  children?: ReactNode;
  actions?: ReactNode;
  testid?: string;
  wide?: boolean;
}) {
  const t = useT();
  const ref = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const returnFocus = useRef<HTMLElement | null>(null);
  // callers pass a new closure on every render: depending on it would run the effect again and pull the focus back to the
  // first button each time the parent re-renders (a job tick, a keystroke in a field of the dialog)
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!open) return;
    returnFocus.current = document.activeElement as HTMLElement | null;
    const node = ref.current;
    const first = node?.querySelector<HTMLElement>("[data-autofocus]") ?? node?.querySelector<HTMLElement>(FOCUSABLE);
    first?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onCloseRef.current();
        return;
      }
      if (e.key !== "Tab" || !node) return;
      const items = Array.from(node.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((el) => el.offsetParent !== null);
      if (items.length === 0) return;
      const a = items[0] as HTMLElement;
      const z = items[items.length - 1] as HTMLElement;
      if (e.shiftKey && document.activeElement === a) {
        e.preventDefault();
        z.focus();
      } else if (!e.shiftKey && document.activeElement === z) {
        e.preventDefault();
        a.focus();
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("keydown", onKey, true);
      returnFocus.current?.focus?.();
    };
  }, [open]);

  if (!open) return null;
  return (
    <div className="dialog-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div ref={ref} className={`dialog ${wide ? "dialog-wide" : ""}`} role="dialog" aria-modal="true" aria-labelledby={titleId} data-testid={testid}>
        <div className="dialog-head">
          <h2 id={titleId}>{title}</h2>
          <button type="button" className="icon-btn" onClick={onClose} aria-label={t("common.close")} data-testid={testid ? `${testid}-close` : undefined}>
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
              <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" fill="none" />
            </svg>
          </button>
        </div>
        <div className="dialog-body">{children}</div>
        {actions ? <div className="dialog-actions">{actions}</div> : null}
      </div>
    </div>
  );
}

// ---- Toast ----------------------------------------------------------------------------------

interface ToastValue {
  show: (text: string) => void;
}
const ToastCtx = createContext<ToastValue>({ show: () => undefined });

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<{ id: number; text: string }[]>([]);
  const nextId = useRef(1);
  const show = useCallback((text: string) => {
    const id = nextId.current++;
    setItems((cur) => [...cur.slice(-2), { id, text }]);
    window.setTimeout(() => setItems((cur) => cur.filter((x) => x.id !== id)), 4500);
  }, []);
  const value = useMemo(() => ({ show }), [show]);
  return (
    <ToastCtx.Provider value={value}>
      {children}
      <div className="toasts" role="status" aria-live="polite" data-testid="toasts">
        {items.map((i) => (
          <div key={i.id} className="toast">
            {i.text}
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

export function useToast(): ToastValue {
  return useContext(ToastCtx);
}

// ---- progress bar ---------------------------------------------------------------------------

export function ProgressBar({
  value,
  label,
  testid,
  tone = "accent",
  size = "md",
}: {
  /** 0..1, or null for an indeterminate bar */
  value: number | null;
  label: string;
  testid?: string;
  tone?: "accent" | "good" | "warn" | "bad";
  size?: "sm" | "md";
}) {
  const pct = value === null ? null : Math.round(clamp(value, 0, 1) * 100);
  return (
    <div
      className={`progress progress-${size} ${pct === null ? "is-indeterminate" : ""}`}
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct ?? undefined}
      data-testid={testid}
      data-value={pct ?? ""}
    >
      <div className={`progress-fill progress-${tone}`} style={pct === null ? undefined : { width: `${pct}%` }} />
    </div>
  );
}

// ---- form controls --------------------------------------------------------------------------

export function Checkbox({
  checked,
  onChange,
  label,
  hint,
  testid,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: ReactNode;
  hint?: ReactNode;
  testid?: string;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className="check">
      <input id={id} type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} data-testid={testid} />
      <label htmlFor={id}>
        <span className="check-label">{label}</span>
        {hint ? <span className="hint">{hint}</span> : null}
      </label>
    </div>
  );
}

export interface RadioOption<V extends string> {
  value: V;
  label: ReactNode;
  hint?: ReactNode;
  testid?: string;
  disabled?: boolean;
}

export function RadioGroup<V extends string>({
  name,
  legend,
  value,
  options,
  onChange,
  testid,
  inline,
}: {
  name: string;
  legend: ReactNode;
  value: V;
  options: RadioOption<V>[];
  onChange: (v: V) => void;
  testid?: string;
  inline?: boolean;
}) {
  const uid = useId();
  return (
    <fieldset className={`radio-group ${inline ? "is-inline" : ""}`} data-testid={testid}>
      <legend>{legend}</legend>
      {options.map((o) => {
        const id = `${uid}-${o.value}`;
        return (
          <div className="check" key={o.value}>
            <input id={id} type="radio" name={`${name}-${uid}`} checked={value === o.value} disabled={o.disabled} onChange={() => onChange(o.value)} data-testid={o.testid} />
            <label htmlFor={id}>
              <span className="check-label">{o.label}</span>
              {o.hint ? <span className="hint">{o.hint}</span> : null}
            </label>
          </div>
        );
      })}
    </fieldset>
  );
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
  minLabel,
  maxLabel,
  testid,
  disabled,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  minLabel?: string;
  maxLabel?: string;
  testid?: string;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className="slider">
      <div className="slider-head">
        <label htmlFor={id}>{label}</label>
        <output htmlFor={id}>{format ? format(value) : String(value)}</output>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} data-testid={testid} />
      {minLabel || maxLabel ? (
        <div className="slider-ends" aria-hidden="true">
          <span>{minLabel}</span>
          <span>{maxLabel}</span>
        </div>
      ) : null}
    </div>
  );
}

export function TextField({
  label,
  value,
  onChange,
  onBlur,
  placeholder,
  testid,
  hint,
  type = "text",
  multiline,
  disabled,
  rows,
  className,
  onEnter,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  onBlur?: () => void;
  placeholder?: string;
  testid?: string;
  hint?: ReactNode;
  type?: string;
  multiline?: boolean;
  disabled?: boolean;
  rows?: number;
  className?: string;
  onEnter?: () => void;
}) {
  const id = useId();
  return (
    <div className={`field ${className ?? ""}`}>
      <label htmlFor={id}>{label}</label>
      {multiline ? (
        <textarea id={id} value={value} rows={rows ?? 3} disabled={disabled} placeholder={placeholder} onChange={(e) => onChange(e.target.value)} onBlur={onBlur} data-testid={testid} />
      ) : (
        <input
          id={id}
          type={type}
          value={value}
          disabled={disabled}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
          onBlur={onBlur}
          onKeyDown={(e) => {
            if (e.key === "Enter" && onEnter) onEnter();
          }}
          data-testid={testid}
        />
      )}
      {hint ? <span className="hint">{hint}</span> : null}
    </div>
  );
}

export function SelectField<V extends string>({
  label,
  value,
  options,
  onChange,
  testid,
  disabled,
  hideLabel,
  className,
}: {
  label: string;
  value: V;
  options: { value: V; label: string }[];
  onChange: (v: V) => void;
  testid?: string;
  disabled?: boolean;
  hideLabel?: boolean;
  className?: string;
}) {
  const id = useId();
  return (
    <div className={`field ${className ?? ""}`}>
      <label htmlFor={id} className={hideLabel ? "sr-only" : undefined}>
        {label}
      </label>
      <select id={id} value={value} disabled={disabled} onChange={(e) => onChange(e.target.value as V)} data-testid={testid}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

// ---- stars / histogram / gauge ----------------------------------------------------------------

export function Stars({ value, title, testid }: { value: number; title?: string; testid?: string }) {
  const t = useT();
  const n = clamp(Math.round(value), 0, 5);
  return (
    <span className="stars" role="img" aria-label={t("casting.stars_label", { n })} title={title} data-testid={testid} data-stars={n}>
      {[1, 2, 3, 4, 5].map((i) => (
        <svg key={i} viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" className={i <= n ? "star on" : "star"}>
          <path d="M12 2.5l2.9 6 6.6.9-4.8 4.6 1.2 6.5L12 17.4 6.1 20.5l1.2-6.5L2.5 9.4l6.6-.9z" />
        </svg>
      ))}
    </span>
  );
}

export function Histogram({ data, testid }: { data: Record<string, number>; testid?: string }) {
  const { t, tOr } = useI18n();
  const rows = Object.entries(data)
    .filter(([, v]) => v > 0.001)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6);
  const top = rows.length ? (rows[0] as [string, number])[1] : 1;
  if (rows.length === 0) return null;
  return (
    <ul className="histogram" data-testid={testid} aria-label={t("casting.emotions")}>
      {rows.map(([label, share]) => (
        <li key={label}>
          <span className="hist-label">{tOr(`emotion.${label}`, label)}</span>
          <span className="hist-bar" aria-hidden="true">
            <span style={{ width: `${Math.max(4, (share / top) * 100)}%` }} />
          </span>
          <span className="hist-value">{Math.round(share * 100)}%</span>
        </li>
      ))}
    </ul>
  );
}

export function Gauge({ used, total, label, testid }: { used: number; total: number; label: string; testid?: string }) {
  const frac = total > 0 ? clamp(used / total, 0, 1) : 0;
  const tone = frac > 0.92 ? "bad" : frac > 0.8 ? "warn" : "accent";
  return (
    <div className="gauge" data-testid={testid}>
      <div className="gauge-head">
        <span>{label}</span>
        <span>
          {used.toFixed(1)} / {total.toFixed(1)} GB
        </span>
      </div>
      <ProgressBar value={frac} label={label} tone={tone} size="sm" />
    </div>
  );
}

// ---- tabs -----------------------------------------------------------------------------------

export function Tabs<V extends string>({
  tabs,
  value,
  onChange,
  label,
}: {
  tabs: { value: V; label: ReactNode; testid?: string }[];
  value: V;
  onChange: (v: V) => void;
  label: string;
}) {
  return (
    <div className="tabs" role="tablist" aria-label={label}>
      {tabs.map((tb) => (
        <button
          key={tb.value}
          type="button"
          role="tab"
          aria-selected={tb.value === value}
          tabIndex={tb.value === value ? 0 : -1}
          className={`tab ${tb.value === value ? "is-active" : ""}`}
          onClick={() => onChange(tb.value)}
          onKeyDown={(e) => {
            const i = tabs.findIndex((x) => x.value === value);
            if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
              e.preventDefault();
              const n = (i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length;
              onChange((tabs[n] as { value: V }).value);
            }
          }}
          data-testid={tb.testid}
        >
          {tb.label}
        </button>
      ))}
    </div>
  );
}

// ---- copy -----------------------------------------------------------------------------------

/** Put text on the clipboard: the async API first, a hidden textarea when it is not allowed (no secure context, no permission). */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through to the textarea */
  }
  return legacyCopy(text);
}

function legacyCopy(text: string): boolean {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  try {
    return document.execCommand("copy");
  } catch {
    return false;
  } finally {
    document.body.removeChild(ta);
  }
}

export function CopyButton({ text, testid }: { text: string; testid?: string }) {
  const t = useT();
  const toast = useToast();
  return (
    <Button
      small
      onClick={() => {
        void copyText(text).then((ok) => {
          if (ok) toast.show(t("common.copied"));
        });
      }}
      data-testid={testid}
    >
      {t("common.copy")}
    </Button>
  );
}

// ---- layout bits ----------------------------------------------------------------------------

export function Card({ children, className, testid }: { children: ReactNode; className?: string; testid?: string }) {
  return (
    <section className={`card ${className ?? ""}`} data-testid={testid}>
      {children}
    </section>
  );
}

export function PageTitle({ title, intro, testid }: { title: ReactNode; intro?: ReactNode; testid?: string }) {
  return (
    <header className="page-title">
      <h1 data-testid={testid} tabIndex={-1}>
        {title}
      </h1>
      {intro ? <p className="intro">{intro}</p> : null}
    </header>
  );
}
