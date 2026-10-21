import MarkdownIt from "markdown-it";
import { useMemo } from "react";

// Raw HTML in the source is escaped (html: false) and markdown-it refuses javascript:/data: links,
// so the output of render() is safe to inject. Links open outside the app window.
const md = new MarkdownIt({ html: false, linkify: false, typographer: false, breaks: false });

const defaultLink = md.renderer.rules.link_open;
md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
  const token = tokens[idx];
  if (token) {
    token.attrSet("target", "_blank");
    token.attrSet("rel", "noopener noreferrer");
  }
  return defaultLink ? defaultLink(tokens, idx, options, env, self) : self.renderToken(tokens, idx, options);
};

export function Markdown({ source, testid }: { source: string; testid?: string }) {
  const html = useMemo(() => md.render(source), [source]);
  // eslint-disable-next-line react/no-danger
  return <div className="markdown" data-testid={testid} dangerouslySetInnerHTML={{ __html: html }} />;
}
