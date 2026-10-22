import type { Glossary } from "../api/types";
import { useT } from "../i18n";
import { Button } from "./ui";

/** Table of character names (target spelling, gender) and free glossary terms. */
export function GlossaryEditor({ value, onChange }: { value: Glossary; onChange: (g: Glossary) => void }) {
  const t = useT();
  const setChar = (i: number, patch: Partial<Glossary["characters"][number]>) =>
    onChange({ ...value, characters: value.characters.map((c, j) => (j === i ? { ...c, ...patch } : c)) });
  const setTerm = (i: number, patch: Partial<Glossary["terms"][number]>) =>
    onChange({ ...value, terms: value.terms.map((c, j) => (j === i ? { ...c, ...patch } : c)) });

  return (
    <div className="glossary" data-testid="glossary">
      <h3>{t("glossary.characters")}</h3>
      <p className="hint">{t("glossary.characters_hint")}</p>
      <div className="table-wrap">
        <table className="data-table is-fixed is-form" data-testid="glossary-characters">
          <colgroup>
            <col className="col-gl-src" />
            <col className="col-gl-tgt" />
            <col className="col-gl-gender" />
          </colgroup>
          <thead>
            <tr>
              <th scope="col">{t("glossary.src")}</th>
              <th scope="col">{t("glossary.tgt")}</th>
              <th scope="col">{t("glossary.gender")}</th>
            </tr>
          </thead>
          <tbody>
            {value.characters.map((c, i) => (
              <tr key={c.speaker_id} data-testid={`glossary-char-${c.speaker_id}`}>
                <th scope="row">{c.src || c.speaker_id}</th>
                <td>
                  <input type="text" value={c.tgt} aria-label={t("glossary.tgt_for", { name: c.src || c.speaker_id })} onChange={(e) => setChar(i, { tgt: e.target.value })} data-testid={`glossary-tgt-${c.speaker_id}`} />
                </td>
                <td>
                  <select value={c.gender} aria-label={t("glossary.gender_for", { name: c.src || c.speaker_id })} onChange={(e) => setChar(i, { gender: e.target.value as "m" | "f" | "unknown" })} data-testid={`glossary-gender-${c.speaker_id}`}>
                    <option value="m">{t("gender.m")}</option>
                    <option value="f">{t("gender.f")}</option>
                    <option value="unknown">{t("gender.unknown")}</option>
                  </select>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3>{t("glossary.terms")}</h3>
      <p className="hint">{t("glossary.terms_hint")}</p>
      {value.terms.length > 0 ? (
        <div className="table-wrap">
          <table className="data-table is-fixed is-form" data-testid="glossary-terms">
            <colgroup>
              <col className="col-gl-term" />
              <col className="col-gl-term" />
              <col className="col-gl-note" />
              <col className="col-gl-remove" />
            </colgroup>
            <thead>
              <tr>
                <th scope="col">{t("glossary.src")}</th>
                <th scope="col">{t("glossary.tgt")}</th>
                <th scope="col">{t("glossary.note")}</th>
                <th scope="col">
                  <span className="sr-only">{t("glossary.remove")}</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {value.terms.map((term, i) => (
                <tr key={i}>
                  <td>
                    <input type="text" value={term.src} aria-label={t("glossary.src")} onChange={(e) => setTerm(i, { src: e.target.value })} data-testid={`glossary-term-src-${i}`} />
                  </td>
                  <td>
                    <input type="text" value={term.tgt} aria-label={t("glossary.tgt")} onChange={(e) => setTerm(i, { tgt: e.target.value })} data-testid={`glossary-term-tgt-${i}`} />
                  </td>
                  <td>
                    <input type="text" value={term.note} aria-label={t("glossary.note")} onChange={(e) => setTerm(i, { note: e.target.value })} />
                  </td>
                  <td>
                    <Button small variant="ghost" onClick={() => onChange({ ...value, terms: value.terms.filter((_, j) => j !== i) })} aria-label={t("glossary.remove")} data-testid={`glossary-term-remove-${i}`}>
                      {t("glossary.remove")}
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      <Button onClick={() => onChange({ ...value, terms: [...value.terms, { src: "", tgt: "", note: "" }] })} data-testid="glossary-add-term">
        {t("glossary.add_term")}
      </Button>
    </div>
  );
}
