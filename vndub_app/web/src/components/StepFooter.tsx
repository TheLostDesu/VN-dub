import type { ReactNode } from "react";
import type { StepId } from "../api/types";
import { useT } from "../i18n";
import { useWizard } from "../state/wizard";
import { Button } from "./ui";

/** Back / Next buttons at the bottom of every wizard step. */
export function StepFooter({
  step,
  canNext = true,
  nextLabel,
  nextHint,
  onNext,
  hideBack,
  extra,
  nextTestid,
}: {
  step: StepId;
  canNext?: boolean;
  nextLabel?: string;
  /** shown next to a disabled Next button: what is still missing */
  nextHint?: ReactNode;
  /** override the default "go to the next step" behaviour (must navigate itself) */
  onNext?: () => void | Promise<void>;
  hideBack?: boolean;
  extra?: ReactNode;
  /** data-testid of the Next button (default "nav-next") */
  nextTestid?: string;
}) {
  const t = useT();
  const wiz = useWizard();
  return (
    <footer className="step-footer">
      {!hideBack && step !== "doctor" ? (
        <Button onClick={() => wiz.back(step)} data-testid="nav-back">
          {t("common.back")}
        </Button>
      ) : (
        <span />
      )}
      <div className="step-footer-right">
        {extra}
        {!canNext && nextHint ? <span className="hint" data-testid="nav-next-hint">{nextHint}</span> : null}
        {step !== "install" ? (
          <Button variant="primary" disabled={!canNext} onClick={() => (onNext ? void onNext() : void wiz.next(step))} data-testid={nextTestid ?? "nav-next"}>
            {nextLabel ?? t("common.next")}
          </Button>
        ) : null}
      </div>
    </footer>
  );
}
