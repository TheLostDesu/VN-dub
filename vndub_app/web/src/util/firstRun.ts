const FIRST_RUN_KEY = "vndub.first_run_done";

/** The first-run Diagnostics screen is offered once; this remembers that it was shown. */
export function markFirstRunDone(): void {
  try {
    window.localStorage.setItem(FIRST_RUN_KEY, "1");
  } catch {
    /* storage may be unavailable: Diagnostics is then simply offered again */
  }
}

export function firstRunDone(): boolean {
  try {
    return window.localStorage.getItem(FIRST_RUN_KEY) === "1";
  } catch {
    return true;
  }
}
