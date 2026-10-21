// Live job state: one WebSocket (/api/progress) feeds a tiny external store; components subscribe with hooks.
// The socket reconnects forever; while it is down (or silent) active jobs are refreshed with GET /api/jobs/{id}.
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { api } from "../api/endpoints";
import { toErrorInfo } from "../api/client";
import { ACTIVE_STATES, FINAL_STATES, type ErrorInfo, type JobInfo, type ProgressEvent } from "../api/types";

export interface GpuReading {
  used_gb: number;
  total_gb: number;
  at: number;
}

export interface ProgressState {
  connected: boolean;
  jobs: Record<string, JobInfo>;
  logs: Record<string, string[]>;
  seen: Record<string, number>; // last time (ms) an event for the job arrived
  gpu: GpuReading | null;
}

const LOG_CAP = 600;
const GPU_STALE_MS = 8000;

let state: ProgressState = { connected: false, jobs: {}, logs: {}, seen: {}, gpu: null };
const listeners = new Set<() => void>();

function emit(next: ProgressState): void {
  state = next;
  listeners.forEach((l) => l());
}

function subscribe(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

function getSnapshot(): ProgressState {
  return state;
}

export function ingestJob(job: JobInfo): void {
  emit({ ...state, jobs: { ...state.jobs, [job.id]: job }, seen: { ...state.seen, [job.id]: Date.now() } });
}

function ingestLog(jobId: string, line: string): void {
  const cur = state.logs[jobId] ?? [];
  const next = cur.length >= LOG_CAP ? [...cur.slice(cur.length - LOG_CAP + 1), line] : [...cur, line];
  emit({ ...state, logs: { ...state.logs, [jobId]: next }, seen: { ...state.seen, [jobId]: Date.now() } });
}

function setLogIfEmpty(jobId: string, lines: string[]): void {
  if ((state.logs[jobId]?.length ?? 0) > 0) return;
  emit({ ...state, logs: { ...state.logs, [jobId]: lines.slice(-LOG_CAP) } });
}

function handleEvent(ev: ProgressEvent): void {
  switch (ev.type) {
    case "hello": {
      const jobs = { ...state.jobs };
      const seen = { ...state.seen };
      const now = Date.now();
      for (const j of ev.jobs) {
        jobs[j.id] = j;
        seen[j.id] = now;
      }
      emit({ ...state, jobs, seen });
      void reconcile(new Set(ev.jobs.map((j) => j.id)));
      break;
    }
    case "job":
      ingestJob(ev.job);
      break;
    case "log":
      ingestLog(ev.job_id, ev.line);
      break;
    case "gpu":
      emit({ ...state, gpu: { used_gb: ev.used_gb, total_gb: ev.total_gb, at: Date.now() } });
      break;
  }
}

/** After a (re)connect: jobs we believed active but the server no longer lists as active may have finished. */
async function reconcile(listed: Set<string>): Promise<void> {
  const stale = Object.values(state.jobs).filter((j) => ACTIVE_STATES.includes(j.state) && !listed.has(j.id));
  for (const j of stale) {
    try {
      ingestJob(await api.job(j.id));
    } catch {
      /* the job may be gone after a server restart: keep the last known state */
    }
  }
}

let socket: WebSocket | null = null;
let started = false;
let retry = 0;
let timer: number | null = null;

function connect(): void {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  let ws: WebSocket;
  try {
    ws = new WebSocket(`${proto}://${window.location.host}/api/progress`);
  } catch {
    scheduleReconnect();
    return;
  }
  socket = ws;
  ws.onopen = () => {
    retry = 0;
    emit({ ...state, connected: true });
  };
  ws.onmessage = (m) => {
    try {
      handleEvent(JSON.parse(String(m.data)) as ProgressEvent);
    } catch {
      /* ignore malformed frames */
    }
  };
  ws.onclose = () => {
    if (socket === ws) socket = null;
    emit({ ...state, connected: false });
    scheduleReconnect();
  };
  ws.onerror = () => {
    try {
      ws.close();
    } catch {
      /* already closed */
    }
  };
}

function scheduleReconnect(): void {
  if (timer !== null) return;
  retry += 1;
  const delay = Math.min(15000, 500 * 2 ** Math.min(retry, 5));
  timer = window.setTimeout(() => {
    timer = null;
    connect();
  }, delay);
}

/** Start the socket once for the whole app. */
export function startProgress(): void {
  if (started) return;
  started = true;
  connect();
}

// ---- hooks --------------------------------------------------------------------------------

export function useProgressState(): ProgressState {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

/** A live job: from the store, refreshed by polling when the socket is down or quiet. */
export function useJob(id: string | null | undefined, initial?: JobInfo | null): JobInfo | null {
  const s = useProgressState();
  const job = id ? s.jobs[id] ?? (initial && initial.id === id ? initial : null) : null;
  const seen = id ? s.seen[id] ?? 0 : 0;
  const stateName = job?.state;
  const seenRef = useRef(seen);
  seenRef.current = seen;

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    const refresh = async () => {
      try {
        const j = await api.job(id);
        if (!cancelled) ingestJob(j);
      } catch {
        /* keep the last known state */
      }
    };
    if (!state.jobs[id]) void refresh();
    const iv = window.setInterval(() => {
      const active = !stateName || ACTIVE_STATES.includes(stateName);
      // no socket: poll every second; socket up but silent for a while: poll as a safety net
      if (active && (!state.connected || Date.now() - seenRef.current > 5000)) void refresh();
    }, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(iv);
    };
  }, [id, stateName]);

  return job;
}

export function useJobLog(id: string | null | undefined, initial?: string[]): string[] {
  const s = useProgressState();
  useEffect(() => {
    if (!id) return;
    if ((state.logs[id]?.length ?? 0) > 0) return;
    if (initial && initial.length > 0) {
      setLogIfEmpty(id, initial);
      return;
    }
    let cancelled = false;
    api
      .job(id)
      .then((j) => {
        if (!cancelled && j.log_tail) setLogIfEmpty(id, j.log_tail);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [id, initial]);
  return id ? s.logs[id] ?? [] : [];
}

/** The most recent job that is still queued, running or paused (optionally of given kinds). */
export function useActiveJob(kinds?: string[]): JobInfo | null {
  const s = useProgressState();
  return useMemo(() => {
    const act = Object.values(s.jobs).filter((j) => ACTIVE_STATES.includes(j.state) && (!kinds || kinds.includes(j.kind)));
    act.sort((a, b) => (a.created < b.created ? 1 : -1));
    return act[0] ?? null;
  }, [s.jobs, kinds]);
}

/** GPU memory reading, or null when no `gpu` event arrived recently (no NVIDIA GPU, or no GPU job running). */
export function useGpu(): GpuReading | null {
  const s = useProgressState();
  const [, tick] = useState(0);
  useEffect(() => {
    const iv = window.setInterval(() => tick((n) => n + 1), 3000);
    return () => window.clearInterval(iv);
  }, []);
  const g = s.gpu;
  if (!g || Date.now() - g.at > GPU_STALE_MS) return null;
  return g;
}

export interface JobTracker {
  job: JobInfo | null;
  starting: boolean;
  /** true while a request is being sent or the job is queued / running / paused */
  busy: boolean;
  /** error of the failed start request or of the failed job */
  error: ErrorInfo | null;
  start: (fn: () => Promise<JobInfo>) => Promise<JobInfo | null>;
  attach: (job: JobInfo) => void;
  reset: () => void;
}

/**
 * Start a job and follow it. `onDone` fires once when the tracked job ends in `done`;
 * `onFinal` fires once for any final state.
 */
export function useJobTracker(opts?: { onDone?: (job: JobInfo) => void; onFinal?: (job: JobInfo) => void }): JobTracker {
  const [jobId, setJobId] = useState<string | null>(null);
  const [initial, setInitial] = useState<JobInfo | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<ErrorInfo | null>(null);
  const job = useJob(jobId, initial);
  const optsRef = useRef(opts);
  optsRef.current = opts;
  const firedFor = useRef<string | null>(null);

  const attach = useCallback((j: JobInfo) => {
    ingestJob(j);
    setInitial(j);
    setJobId(j.id);
    setStartError(null);
    firedFor.current = null;
  }, []);

  const start = useCallback(
    async (fn: () => Promise<JobInfo>) => {
      setStarting(true);
      setStartError(null);
      try {
        const j = await fn();
        attach(j);
        return j;
      } catch (e) {
        setStartError(toErrorInfo(e));
        return null;
      } finally {
        setStarting(false);
      }
    },
    [attach],
  );

  const reset = useCallback(() => {
    setJobId(null);
    setInitial(null);
    setStartError(null);
    firedFor.current = null;
  }, []);

  useEffect(() => {
    if (!job || !FINAL_STATES.includes(job.state)) return;
    const key = `${job.id}:${job.state}`;
    if (firedFor.current === key) return;
    firedFor.current = key;
    optsRef.current?.onFinal?.(job);
    if (job.state === "done") optsRef.current?.onDone?.(job);
  }, [job]);

  const error = startError ?? (job?.state === "failed" ? job.error ?? { code: "internal", params: {}, action: "report_problem", details: "" } : null);
  const busy = starting || (!!job && ACTIVE_STATES.includes(job.state));
  return { job, starting, busy, error, start, attach, reset };
}
