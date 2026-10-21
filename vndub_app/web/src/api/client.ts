import type { ErrorInfo } from "./types";

/** A backend error in the wire format of vndub_app/errors.py (or a synthesized one for network problems). */
export class ApiError extends Error {
  readonly info: ErrorInfo;
  readonly status: number;

  constructor(info: ErrorInfo, status = 0) {
    super(`${info.code}${info.details ? `: ${info.details.split("\n")[0]}` : ""}`);
    this.name = "ApiError";
    this.info = info;
    this.status = status;
  }
}

export function makeError(code: string, details = "", action = "none", params: Record<string, unknown> = {}): ErrorInfo {
  return { code, params, action, details };
}

function isErrorBody(v: unknown): v is { error: Partial<ErrorInfo> } {
  return typeof v === "object" && v !== null && "error" in v && typeof (v as { error: unknown }).error === "object" && (v as { error: unknown }).error !== null;
}

export function normalizeErrorInfo(raw: Partial<ErrorInfo>): ErrorInfo {
  return {
    code: typeof raw.code === "string" && raw.code ? raw.code : "internal",
    params: raw.params && typeof raw.params === "object" ? raw.params : {},
    action: typeof raw.action === "string" && raw.action ? raw.action : "none",
    details: typeof raw.details === "string" ? raw.details : "",
  };
}

/** Turn anything that was thrown into an ErrorInfo the UI can show. */
export function toErrorInfo(e: unknown): ErrorInfo {
  if (e instanceof ApiError) return e.info;
  if (e instanceof Error) return makeError("internal", `${e.name}: ${e.message}`, "report_problem");
  return makeError("internal", String(e), "report_problem");
}

type Method = "GET" | "POST" | "PUT" | "DELETE";

async function request<T>(method: Method, path: string, body?: unknown, form?: FormData, signal?: AbortSignal): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  const init: RequestInit = { method, signal, headers };
  if (form) {
    init.body = form;
  } else if (body !== undefined) {
    init.body = JSON.stringify(body);
    headers["Content-Type"] = "application/json";
  }
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError(makeError("server_unreachable", String(e), "retry"), 0);
  }
  const text = await res.text();
  let data: unknown = undefined;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = undefined;
    }
  }
  if (!res.ok) {
    if (isErrorBody(data)) throw new ApiError(normalizeErrorInfo(data.error), res.status);
    throw new ApiError(makeError("internal", `HTTP ${res.status} ${res.statusText}\n${text.slice(0, 2000)}`, "report_problem"), res.status);
  }
  return data as T;
}

export type Query = Record<string, string | number | boolean | undefined | null>;

export const http = {
  get: <T>(path: string, query?: Query, signal?: AbortSignal): Promise<T> => {
    let url = path;
    if (query) {
      const qs = new URLSearchParams();
      for (const [k, v] of Object.entries(query)) {
        if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
      }
      const s = qs.toString();
      if (s) url += `?${s}`;
    }
    return request<T>("GET", url, undefined, undefined, signal);
  },
  post: <T>(path: string, body?: unknown): Promise<T> => request<T>("POST", path, body ?? {}),
  put: <T>(path: string, body?: unknown): Promise<T> => request<T>("PUT", path, body ?? {}),
  del: <T>(path: string): Promise<T> => request<T>("DELETE", path),
  upload: <T>(path: string, file: File, field = "file"): Promise<T> => {
    const form = new FormData();
    form.append(field, file, file.name);
    return request<T>("POST", path, undefined, form);
  },
};
