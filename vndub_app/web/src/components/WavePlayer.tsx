import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { useT } from "../i18n";
import { useSettings } from "../state/settings";
import { Button } from "./ui";

// Only one player sounds at a time: starting one pauses the one that was playing.
let currentlyPlaying: { pause: () => void } | null = null;

export interface WavePlayerHandle {
  play: () => void;
  pause: () => void;
  toggle: () => void;
  isPlaying: () => boolean;
}

interface Props {
  url: string | null | undefined;
  /** the kind of player: its volume is remembered in Settings.player_volume[kind] */
  kind: string;
  /** what the player plays, for the screen reader ("Оригинал", "Вариант A" ...) */
  label: string;
  compact?: boolean;
  height?: number;
  testid?: string;
  onFinish?: () => void;
  onPlay?: () => void;
  /** show the label above the waveform */
  showLabel?: boolean;
}

function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
}

export const WavePlayer = forwardRef<WavePlayerHandle, Props>(function WavePlayer(
  { url, kind, label, compact, height, testid, onFinish, onPlay, showLabel },
  ref,
) {
  const t = useT();
  const { settings, setPlayerVolume } = useSettings();
  const volume = typeof settings.player_volume?.[kind] === "number" ? (settings.player_volume[kind] as number) : 1;
  const hostRef = useRef<HTMLDivElement>(null);
  const waveRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const loadedUrl = useRef<string | null>(null);
  const pendingPlay = useRef(false);
  const volumeRef = useRef(volume);
  volumeRef.current = volume;
  const finishRef = useRef(onFinish);
  finishRef.current = onFinish;
  const playRef = useRef(onPlay);
  playRef.current = onPlay;

  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  const [duration, setDuration] = useState(0);
  const [pos, setPos] = useState(0);

  const create = useCallback(() => {
    if (wsRef.current || !waveRef.current || !url) return wsRef.current;
    const ws = WaveSurfer.create({
      container: waveRef.current,
      url,
      height: height ?? (compact ? 36 : 56),
      waveColor: cssVar("--wave", "#9aa7b8"),
      progressColor: cssVar("--wave-progress", "#2b6cb0"),
      cursorColor: cssVar("--wave-cursor", "#1a365d"),
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
    });
    loadedUrl.current = url;
    ws.setVolume(volumeRef.current);
    ws.on("ready", () => {
      setReady(true);
      setFailed(false);
      setDuration(ws.getDuration());
      if (pendingPlay.current) {
        pendingPlay.current = false;
        void ws.play().catch(() => undefined);
      }
    });
    ws.on("play", () => {
      if (currentlyPlaying && currentlyPlaying !== handle) currentlyPlaying.pause();
      currentlyPlaying = handle;
      setPlaying(true);
      playRef.current?.();
    });
    ws.on("pause", () => {
      if (currentlyPlaying === handle) currentlyPlaying = null;
      setPlaying(false);
    });
    ws.on("finish", () => {
      setPlaying(false);
      if (currentlyPlaying === handle) currentlyPlaying = null;
      finishRef.current?.();
    });
    ws.on("timeupdate", (s: number) => setPos(s));
    ws.on("error", () => {
      setFailed(true);
      setReady(false);
      pendingPlay.current = false;
    });
    wsRef.current = ws;
    return ws;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, height, compact]);

  const handle = useRef<WavePlayerHandle>({
    play: () => undefined,
    pause: () => undefined,
    toggle: () => undefined,
    isPlaying: () => false,
  }).current;

  handle.play = () => {
    const ws = create();
    if (!ws) return;
    if (!ready && ws.getDuration() === 0) {
      pendingPlay.current = true;
    } else {
      void ws.play().catch(() => undefined);
    }
  };
  handle.pause = () => {
    pendingPlay.current = false;
    wsRef.current?.pause();
  };
  handle.toggle = () => {
    if (wsRef.current?.isPlaying()) handle.pause();
    else handle.play();
  };
  handle.isPlaying = () => !!wsRef.current?.isPlaying();

  useImperativeHandle(ref, () => handle, [handle]);

  // Create the waveform only when the player scrolls into view (Casting can list many players).
  useEffect(() => {
    const host = hostRef.current;
    if (!host || !url) return;
    if (typeof IntersectionObserver === "undefined") {
      create();
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          create();
          io.disconnect();
        }
      },
      { rootMargin: "200px" },
    );
    io.observe(host);
    return () => io.disconnect();
  }, [url, create]);

  // A new url (regenerated audio) reloads the existing instance.
  useEffect(() => {
    const ws = wsRef.current;
    if (ws && url && loadedUrl.current !== url) {
      loadedUrl.current = url;
      setReady(false);
      setFailed(false);
      setPlaying(false);
      void ws.load(url).catch(() => setFailed(true));
    }
  }, [url]);

  useEffect(() => {
    wsRef.current?.setVolume(volume);
  }, [volume]);

  useEffect(
    () => () => {
      if (currentlyPlaying === handle) currentlyPlaying = null;
      wsRef.current?.destroy();
      wsRef.current = null;
    },
    [handle],
  );

  if (!url) {
    // same box as a real player (label row + player row), so a missing recording does not shift its neighbour
    return (
      <div className={`wave-player is-empty ${compact ? "is-compact" : ""}`} data-testid={testid}>
        {showLabel ? <div className="wave-label">{label}</div> : null}
        <div className="wave-row">
          <span className="hint">{t("player.no_audio")}</span>
        </div>
      </div>
    );
  }

  return (
    <div ref={hostRef} className={`wave-player ${compact ? "is-compact" : ""} ${playing ? "is-playing" : ""}`} data-testid={testid} data-playing={playing} data-ready={ready}>
      {showLabel ? <div className="wave-label">{label}</div> : null}
      <div className="wave-row">
        <button
          type="button"
          className="play-btn"
          onClick={() => handle.toggle()}
          aria-label={playing ? t("player.pause", { label }) : t("player.play", { label })}
          aria-pressed={playing}
          data-testid={testid ? `${testid}-play` : undefined}
        >
          {playing ? (
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
              <path d="M7 5h4v14H7zM13 5h4v14h-4z" fill="currentColor" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
              <path d="M8 5v14l11-7z" fill="currentColor" />
            </svg>
          )}
        </button>
        <div className="wave-surface" ref={waveRef} aria-hidden="true" />
        <span className="wave-time" aria-hidden="true">
          {ready ? `${fmtTime(pos)} / ${fmtTime(duration)}` : failed ? "" : "-:--"}
        </span>
        <input
          className="wave-volume"
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={volume}
          aria-label={t("player.volume", { label })}
          onChange={(e) => setPlayerVolume(kind, Number(e.target.value))}
          data-testid={testid ? `${testid}-volume` : undefined}
        />
      </div>
      {failed ? (
        <p className="hint wave-error" role="status">
          {t("player.error")}
        </p>
      ) : null}
    </div>
  );
});

/** Original and dubbed side by side, with a button that plays them one after the other. */
export function ABPlayer({
  originalUrl,
  dubUrl,
  labelA,
  labelB,
  kind = "ab",
  testid,
}: {
  originalUrl: string | null | undefined;
  dubUrl: string | null | undefined;
  labelA?: string;
  labelB?: string;
  kind?: string;
  testid?: string;
}) {
  const t = useT();
  const a = useRef<WavePlayerHandle>(null);
  const b = useRef<WavePlayerHandle>(null);
  const chain = useRef(false);
  const la = labelA ?? t("player.original");
  const lb = labelB ?? t("player.dubbed");
  const sequential = !!originalUrl && !!dubUrl;
  return (
    <div className="ab-player" data-testid={testid}>
      <div className="ab-col">
        <WavePlayer ref={a} url={originalUrl} kind={`${kind}-original`} label={la} showLabel compact onFinish={() => {
          if (chain.current) {
            chain.current = false;
            b.current?.play();
          }
        }} testid={testid ? `${testid}-a` : undefined} />
      </div>
      <div className="ab-col">
        <WavePlayer ref={b} url={dubUrl} kind={`${kind}-dub`} label={lb} showLabel compact testid={testid ? `${testid}-b` : undefined} />
      </div>
      {sequential ? (
        <Button
          small
          onClick={() => {
            chain.current = true;
            a.current?.play();
          }}
          data-testid={testid ? `${testid}-seq` : undefined}>
          {t("player.play_both")}
        </Button>
      ) : null}
    </div>
  );
}
