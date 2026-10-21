import { useT } from "../i18n";
import { CopyButton } from "./ui";

// Simple inline illustrations (currentColor + CSS variables, so they follow the theme).

function IllustrationFolder() {
  return (
    <svg viewBox="0 0 160 100" role="img" aria-hidden="true" className="illustration">
      <rect x="10" y="22" width="140" height="68" rx="6" className="ill-fill" />
      <path d="M10 28a6 6 0 0 1 6-6h34l10 10h84a6 6 0 0 1 6 6" className="ill-stroke" fill="none" />
      <rect x="24" y="46" width="44" height="30" rx="4" className="ill-card" />
      <text x="46" y="65" textAnchor="middle" className="ill-text">game.exe</text>
      <rect x="82" y="46" width="52" height="30" rx="4" className="ill-card ill-accent" />
      <text x="108" y="65" textAnchor="middle" className="ill-text">version.dll</text>
      <path d="M96 40l12 0" className="ill-stroke" />
    </svg>
  );
}

function IllustrationGame() {
  return (
    <svg viewBox="0 0 160 100" role="img" aria-hidden="true" className="illustration">
      <rect x="14" y="10" width="132" height="80" rx="6" className="ill-fill" />
      <rect x="14" y="10" width="132" height="12" rx="6" className="ill-card" />
      <circle cx="24" cy="16" r="2.5" className="ill-accent-fill" />
      <circle cx="33" cy="16" r="2.5" className="ill-accent-fill" />
      <rect x="24" y="62" width="112" height="22" rx="4" className="ill-card" />
      <path d="M32 70h70M32 77h44" className="ill-stroke" />
      <path d="M118 36l6 6 12-14" className="ill-stroke ill-ok" fill="none" />
    </svg>
  );
}

function IllustrationApp() {
  return (
    <svg viewBox="0 0 160 100" role="img" aria-hidden="true" className="illustration">
      <rect x="10" y="14" width="64" height="72" rx="6" className="ill-fill" />
      <path d="M20 34h44M20 46h44M20 58h30" className="ill-stroke" />
      <path d="M82 50h24" className="ill-stroke" />
      <path d="M100 42l10 8-10 8" className="ill-stroke" fill="none" />
      <rect x="114" y="28" width="40" height="44" rx="6" className="ill-card ill-accent" />
      <path d="M124 52l8 8 14-18" className="ill-stroke ill-ok" fill="none" />
    </svg>
  );
}

/** Three-step instruction for encrypted KiriKiri Z games (KrkrDump), the manual path of Step 1. */
export function KrkrDumpGuide({ url }: { url?: string }) {
  const t = useT();
  return (
    <div className="manual-guide" data-testid="krkrdump-guide">
      <h4>{t("manual.krkrdump.title")}</h4>
      <p className="hint">{t("manual.krkrdump.intro")}</p>
      <ol className="guide-steps">
        <li>
          <IllustrationFolder />
          <div>
            <strong>{t("manual.krkrdump.step1.title")}</strong>
            <p>{t("manual.krkrdump.step1.text")}</p>
            {url ? (
              <div className="manual-url">
                <code>{url}</code> <CopyButton text={url} testid="krkrdump-copy-url" />
              </div>
            ) : null}
          </div>
        </li>
        <li>
          <IllustrationGame />
          <div>
            <strong>{t("manual.krkrdump.step2.title")}</strong>
            <p>{t("manual.krkrdump.step2.text")}</p>
          </div>
        </li>
        <li>
          <IllustrationApp />
          <div>
            <strong>{t("manual.krkrdump.step3.title")}</strong>
            <p>{t("manual.krkrdump.step3.text")}</p>
          </div>
        </li>
      </ol>
    </div>
  );
}
