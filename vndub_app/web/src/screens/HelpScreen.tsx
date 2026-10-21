import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import type { ErrorInfo, HelpDoc } from "../api/types";
import { ErrorPanel } from "../components/ErrorPanel";
import { Markdown } from "../components/Markdown";
import { PageTitle, Spinner } from "../components/ui";
import { useI18n } from "../i18n";
import { prettifyId } from "../util/format";

export default function HelpScreen() {
  const { t, tOr } = useI18n();
  const nav = useNavigate();
  const { id } = useParams();
  const [docs, setDocs] = useState<HelpDoc[] | null>(null);
  const [markdown, setMarkdown] = useState<string | null>(null);
  const [error, setError] = useState<ErrorInfo | null>(null);

  useEffect(() => {
    api
      .help()
      .then((d) => {
        setDocs(d);
        setError(null);
      })
      .catch((e) => setError(toErrorInfo(e)));
  }, []);

  const current = id ?? docs?.[0]?.id ?? null;

  useEffect(() => {
    if (!current) return;
    setMarkdown(null);
    api
      .helpDoc(current)
      .then((r) => {
        setMarkdown(r.markdown);
        setError(null);
      })
      .catch((e) => setError(toErrorInfo(e)));
  }, [current]);

  return (
    <>
      <PageTitle title={t("help.title")} intro={t("help.intro")} testid="help-title" />
      {error ? <ErrorPanel error={error} onRetry={() => nav(0)} /> : null}
      {docs === null && !error ? <Spinner label={t("app.loading")} /> : null}
      {docs ? (
        <div className="help-layout">
          <nav className="help-nav" aria-label={t("help.docs")} data-testid="help-nav">
            <ul>
              {docs.map((d) => (
                <li key={d.id}>
                  <button type="button" className={`help-link ${d.id === current ? "is-active" : ""}`} aria-current={d.id === current ? "page" : undefined} onClick={() => nav(`/help/${encodeURIComponent(d.id)}`)} data-testid={`help-doc-${d.id}`}>
                    {tOr(d.title_key, prettifyId(d.id))}
                  </button>
                </li>
              ))}
            </ul>
          </nav>
          <article className="help-body" data-testid="help-body">
            {markdown === null && !error ? <Spinner label={t("app.loading")} /> : null}
            {markdown !== null ? <Markdown source={markdown} testid="help-markdown" /> : null}
          </article>
        </div>
      ) : null}
    </>
  );
}
