import { Component, type ErrorInfo, type ReactNode } from "react";
import { useLocation } from "react-router-dom";
import { useI18n } from "../i18n";
import { Button, Callout } from "./ui";

/** What the user sees instead of a blank window when a screen throws while rendering. */
function CrashCard({ error }: { error: Error }) {
  const t = useI18n().t;
  return (
    <div className="splash" data-testid="app-error">
      <div className="splash-card">
        <h1>VNDub</h1>
        <Callout tone="bad" title={t("app.crash.title")} role="alert">
          <p>{t("app.crash.text")}</p>
          <pre className="command-box" data-testid="app-error-text">
            {error.message || String(error)}
          </pre>
        </Callout>
        <p>
          <Button variant="primary" onClick={() => window.location.reload()} data-testid="app-error-reload">
            {t("app.crash.reload")}
          </Button>
        </p>
      </div>
    </div>
  );
}

interface BoundaryProps {
  children: ReactNode;
  /** a change of this value (the route) lets the children try again */
  resetKey: string;
}

class Boundary extends Component<BoundaryProps, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error): { error: Error } {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("The screen crashed:", error, info.componentStack);
  }

  componentDidUpdate(prev: BoundaryProps): void {
    if (this.state.error && prev.resetKey !== this.props.resetKey) this.setState({ error: null });
  }

  render(): ReactNode {
    return this.state.error ? <CrashCard error={this.state.error} /> : this.props.children;
  }
}

/** Wraps the routes (inside the router and the providers, so the card is translated and a navigation clears it). */
export function ErrorBoundary({ children }: { children: ReactNode }) {
  const loc = useLocation();
  return <Boundary resetKey={loc.pathname}>{children}</Boundary>;
}
