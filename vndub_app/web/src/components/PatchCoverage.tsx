import type { PatchCoverage } from "../api/types";
import { useI18n } from "../i18n";
import { Button, Callout, Chip } from "./ui";

/** "ни один дубль не прошёл проверку: 3; ждут новой озвучки: 1" -- the reasons lines were left out of the patch. */
export function useLeftOutReasons(coverage: PatchCoverage): string {
  const { t, tOr } = useI18n();
  return Object.entries(coverage.left_out_reasons ?? {})
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([reason, n]) => `${tOr(`coverage.reason.${reason}`, t("coverage.reason.other"))}: ${n}`)
    .join("; ");
}

/**
 * The compact "what is in the patch" callout of the finished full run: how many lines are dubbed, how many only provisionally
 * (the best take failed the quality checks but is installed), how many were left out (with the reasons) and how many narration
 * lines are never dubbed. Warn tone when anything is provisional or left out.
 */
export function CoverageCallout({ coverage, onReview, testid = "run-coverage" }: { coverage: PatchCoverage; onReview?: () => void; testid?: string }) {
  const { t } = useI18n();
  const reasons = useLeftOutReasons(coverage);
  const warn = coverage.provisional > 0 || coverage.left_out > 0;
  return (
    <Callout tone={warn ? "warn" : "info"} title={t("coverage.title")} testid={testid} role="status">
      <ul className="coverage-list">
        <li data-testid={`${testid}-installed`}>{t("coverage.installed", { installed: coverage.installed, total: coverage.dub_lines })}</li>
        {coverage.provisional > 0 ? (
          <li data-testid={`${testid}-provisional`}>
            {t("coverage.provisional", { n: coverage.provisional })}
            {onReview ? (
              <>
                {" "}
                <Button small onClick={onReview} data-testid={`${testid}-review`}>
                  {t("coverage.provisional.go")}
                </Button>
              </>
            ) : null}
          </li>
        ) : null}
        {coverage.left_out > 0 ? <li data-testid={`${testid}-left-out`}>{t("coverage.left_out", { n: coverage.left_out, reasons })}</li> : null}
        {coverage.narration > 0 ? <li data-testid={`${testid}-narration`}>{t("coverage.narration", { n: coverage.narration })}</li> : null}
        {coverage.own_voice > 0 ? <li data-testid={`${testid}-own-voice`}>{t("coverage.own_voice", { n: coverage.own_voice })}</li> : null}
      </ul>
    </Callout>
  );
}

/** The three numbers next to the patch's file count (Install screen): dubbed lines in the patch, provisional ones, left out. */
export function CoverageCounts({ coverage, testid = "install-coverage" }: { coverage: PatchCoverage; testid?: string }) {
  const { t } = useI18n();
  const reasons = useLeftOutReasons(coverage);
  return (
    <div className="chip-line coverage-counts" data-testid={testid}>
      <Chip tone="good" testid={`${testid}-installed`}>
        {t("coverage.installed", { installed: coverage.installed, total: coverage.dub_lines })}
      </Chip>
      <Chip tone={coverage.provisional > 0 ? "warn" : "neutral"} title={t("coverage.chip.provisional_title")} testid={`${testid}-provisional`}>
        {t("coverage.chip.provisional", { n: coverage.provisional })}
      </Chip>
      <Chip tone={coverage.left_out > 0 ? "warn" : "neutral"} title={coverage.left_out > 0 ? t("coverage.chip.left_out_title", { reasons }) : undefined} testid={`${testid}-left-out`}>
        {t("coverage.chip.left_out", { n: coverage.left_out })}
      </Chip>
    </div>
  );
}
