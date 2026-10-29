"""Where the desktop app keeps its data, settings and projects.

Layout of the data folder (DEFAULT ``%LOCALAPPDATA%\\VNDub``, ``~/.local/share/vndub`` on Linux)::

    <data>/settings.json            app settings (UI language, GPU budget, license policy ...)
    <data>/jobs.json                job records (resume after restart)
    <data>/throughput_v2.json       measured seconds per voiced line per stage (ETA)
    <data>/logs/app.log             rotating app log (INFO; DEBUG with Settings.debug_log)
    <data>/logs/jobs/               one log file per job (<YYYYMMDD-HHMMSS>-<kind>-<id>.log), the newest 30
    <data>/configs/tools.local.yaml external tool paths written by the installer
    <data>/tools/                   external tools (ffmpeg, vgmstream ...)
    <data>/envs/                    backend virtual environments (VNDUB_ENVS_DIR; Settings.envs_dir moves it)
    <data>/envs/_uv/cache/          uv's download cache (UV_CACHE_DIR): next to the venvs so uv can hardlink wheels into them
    <data>/envs/_uv/python/         CPython builds uv installs for the app (UV_PYTHON_INSTALL_DIR)
    <data>/models/                  model weights (VNDUB_MODELS_DIR; Settings.models_dir moves it)
    <data>/voices/<id>.yaml + .wav  voice library (performer designs reused across projects)
    <data>/backups/<game>/<ts>/     files replaced in the game by the patch installer
    <data>/projects/<slug>/         one project per game:
        project_meta.json           {game_dir, name, created}
        app_state.json              wizard state (step, choices, config overrides)
        pipeline.yaml               generated pipeline config (extends the bundled profile)
        glossary.yaml               project glossary (translate.glossary)
        work/  out/                 the pipeline's work and out dirs

The data folder location itself is stored outside it, in ``location.json`` under
``%APPDATA%\\VNDub`` (``~/.config/vndub``). ``VNDUB_DATA`` overrides everything (tests).
The pipeline profile is chosen by ``Settings.profile`` or ``VNDUB_APP_PROFILE``:
``default`` -> ``configs/pipeline.yaml``, ``test`` -> ``configs/pipeline.test.yaml``,
``apptest`` -> ``configs/pipeline.apptest.yaml`` (all mocks, seconds per run).
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from vndub.util import now_iso

_WRITE_LOCK = threading.RLock()
APP_NAME = "VNDub"
PROFILES = {"default": "pipeline.yaml", "test": "pipeline.test.yaml", "apptest": "pipeline.apptest.yaml"}


# ----------------------------------------------------------------------------
# bundle / repo resources
# ----------------------------------------------------------------------------


def resource_root() -> Path:
    """Repo root in dev mode; the PyInstaller bundle dir when frozen.

    Holds ``configs/``, ``docs/``, ``app/`` (i18n, tools manifest), ``vndub_app/web/dist``.
    """
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def resource(*parts: str) -> Path:
    return resource_root().joinpath(*parts)


def web_dist() -> Path:
    return resource("vndub_app", "web", "dist")


# ----------------------------------------------------------------------------
# data folder
# ----------------------------------------------------------------------------


def _config_home() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / APP_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "vndub"


def default_data_dir() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_NAME
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "vndub"


def location_file() -> Path:
    return _config_home() / "location.json"


def data_dir() -> Path:
    env = os.environ.get("VNDUB_DATA")
    if env:
        p = Path(env)
    else:
        p = default_data_dir()
        lf = location_file()
        if lf.exists():
            try:
                p = Path(json.loads(lf.read_text(encoding="utf-8"))["data_dir"])
            except (ValueError, KeyError, OSError):
                pass
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_data_dir(path: Path) -> Path:
    """Persist a new data folder location (takes effect for new paths immediately).

    Existing data is not moved (DECISIONS: "Desktop app").
    """
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    lf = location_file()
    lf.parent.mkdir(parents=True, exist_ok=True)
    lf.write_text(json.dumps({"data_dir": str(path)}, ensure_ascii=False), encoding="utf-8")
    return path


def sub(*parts: str) -> Path:
    p = data_dir().joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir() -> Path:
    return sub("logs")


def tools_dir() -> Path:
    return sub("tools")


def envs_dir() -> Path:
    return sub("envs")


def models_dir() -> Path:
    return sub("models")


def voices_dir() -> Path:
    return sub("voices")


def backups_dir() -> Path:
    return sub("backups")


def projects_dir() -> Path:
    return sub("projects")


def tools_local_yaml() -> Path:
    return sub("configs") / "tools.local.yaml"


# ----------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------


class Settings(BaseModel):
    model_config = ConfigDict(extra="allow")

    ui_lang: str = "ru"                         # ru | en
    profile: str = "default"                    # default | test | apptest
    gpu_budget_gb: float = 7.5
    license_policy: str = "research"            # research | commercial
    models_dir: str = ""                        # "" = <data>/models
    envs_dir: str = ""                          # "" = <data>/envs (also holds uv's Python builds and cache: <envs>/_uv)
    player_volume: dict[str, float] = Field(default_factory=dict)
    last_project: str | None = None
    shutdown_when_done: bool = False
    translation_llm: str = ""                   # "" = the profile's default (gemma4_12b); else a registry llm backend name
    debug_log: bool = False                     # DEBUG lines in logs/app.log (WebSocket clients, every HTTP request); INFO otherwise


def settings_file() -> Path:
    return data_dir() / "settings.json"


def load_settings() -> Settings:
    f = settings_file()
    data: dict[str, Any] = {}
    if f.exists():
        try:
            data = json.loads(f.read_text(encoding="utf-8")) or {}
        except (ValueError, OSError):
            data = {}
    s = Settings.model_validate(data)
    env = os.environ.get("VNDUB_APP_PROFILE")
    if env:
        s.profile = env
    return s


def save_settings(s: Settings) -> Settings:
    write_json_atomic(settings_file(), s.model_dump(mode="json"))
    return s


def effective_models_dir(s: Settings | None = None) -> Path:
    s = s or load_settings()
    if s.models_dir:
        return Path(s.models_dir)
    env = os.environ.get("VNDUB_MODELS_DIR")
    return Path(env) if env else models_dir()


# The VNDUB_ENVS_DIR value the app itself exported (``installer.configure_environment``). Anything else found in
# the variable was set from outside (a shell, a dev's environment, a test) and is never overwritten.
_EXPORTED_ENVS: str | None = None


def mark_envs_exported(path: str | None) -> None:
    """Remember the ``VNDUB_ENVS_DIR`` value the app exported itself (``None`` forgets it)."""
    global _EXPORTED_ENVS
    _EXPORTED_ENVS = path


def external_envs_dir() -> Path | None:
    """``VNDUB_ENVS_DIR`` when it was set from outside the app, else ``None``."""
    env = os.environ.get("VNDUB_ENVS_DIR")
    return Path(env) if env and env != _EXPORTED_ENVS else None


def effective_envs_dir(s: Settings | None = None) -> Path:
    """The folder the settings ask for: ``Settings.envs_dir``, else an outside ``VNDUB_ENVS_DIR``, else ``<data>/envs``.

    What is in use is decided by ``installer.configure_environment`` (an outside variable beats the setting there,
    and a dev checkout may keep its ``.venvs``): ask ``installer.envs_dir()`` for that.
    """
    s = s or load_settings()
    if s.envs_dir:
        return Path(s.envs_dir)
    return external_envs_dir() or envs_dir()


# ----------------------------------------------------------------------------
# projects
# ----------------------------------------------------------------------------


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s or "game"


@dataclass(frozen=True)
class ProjectHandle:
    slug: str
    root: Path

    @property
    def work(self) -> Path:
        # ``project_meta.json: work_dir`` binds the project to an existing CLI work dir (``vndub review --app``)
        custom = self.meta().get("work_dir")
        return Path(custom) if custom else self.root / "work"

    @property
    def out(self) -> Path:
        custom = self.meta().get("out_dir")
        return Path(custom) if custom else self.root / "out"

    @property
    def meta_file(self) -> Path:
        return self.root / "project_meta.json"

    @property
    def state_file(self) -> Path:
        return self.root / "app_state.json"

    @property
    def config_file(self) -> Path:
        return self.root / "pipeline.yaml"

    @property
    def glossary_file(self) -> Path:
        return self.root / "glossary.yaml"

    def meta(self) -> dict[str, Any]:
        return read_json(self.meta_file, {}) or {}

    @property
    def game_dir(self) -> Path:
        return Path(self.meta().get("game_dir", ""))

    @property
    def name(self) -> str:
        return self.meta().get("name") or self.slug


def project_handle(slug: str) -> ProjectHandle:
    h = ProjectHandle(slug, projects_dir() / slug)
    if not h.meta_file.exists():
        from vndub_app.errors import AppError

        raise AppError("project_not_found", params={"slug": slug}, action="choose_game_dir", status=404)
    return h


def list_projects() -> list[ProjectHandle]:
    out = []
    for d in sorted(projects_dir().iterdir()) if projects_dir().exists() else []:
        if (d / "project_meta.json").exists():
            out.append(ProjectHandle(d.name, d))
    return out


def find_project_for_game(game_dir: Path) -> ProjectHandle | None:
    g = str(Path(game_dir).resolve())
    for h in list_projects():
        if str(Path(h.meta().get("game_dir", "")).resolve()) == g:
            return h
    return None


def create_project(game_dir: Path, name: str | None = None) -> ProjectHandle:
    """Create (or return the existing) project for a game folder."""
    game_dir = Path(game_dir).resolve()
    existing = find_project_for_game(game_dir)
    if existing:
        return existing
    base = slugify(name or game_dir.name)
    slug, i = base, 2
    while (projects_dir() / slug).exists():
        slug, i = f"{base}-{i}", i + 1
    h = ProjectHandle(slug, projects_dir() / slug)
    h.root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(h.meta_file, {"game_dir": str(game_dir), "name": name or game_dir.name, "created": now_iso()})
    write_app_state(h, default_app_state())
    write_project_config(h)
    return h


# ----------------------------------------------------------------------------
# wizard state
# ----------------------------------------------------------------------------

WIZARD_STEPS = ("doctor", "game", "translate", "casting", "preview", "run", "review", "install")


def default_app_state() -> dict[str, Any]:
    return {
        "step": "game",
        "completed": [],
        "translation": {
            "tgt_lang": "ru",
            "src_lang": "auto",        # auto = detected from the script text | a language code
            "source": "auto",          # auto | import
            "import_path": "",
            "honorifics": "drop",      # drop | keep | adapt
            "address_default": "ты",   # ты | вы
        },
        "advanced": {"asr_check": False},
        "config_overrides": {},        # dotted key -> value, applied to pipeline.yaml
        "preview": {"mode": "first_scene", "n": 40, "line_ids": []},
        "review": {"touched": []},
        "updated": now_iso(),
    }


def read_app_state(h: ProjectHandle) -> dict[str, Any]:
    st = read_json(h.state_file, None)
    if not st:
        return default_app_state()
    base = default_app_state()
    for k, v in base.items():
        st.setdefault(k, v)
    return st


def write_app_state(h: ProjectHandle, state: dict[str, Any]) -> dict[str, Any]:
    state = dict(state)
    state["updated"] = now_iso()
    write_json_atomic(h.state_file, state)
    return state


# ----------------------------------------------------------------------------
# generated pipeline config
# ----------------------------------------------------------------------------


def base_config_path(profile: str | None = None) -> Path:
    profile = profile or load_settings().profile
    return resource("configs", PROFILES.get(profile, PROFILES["default"]))


def _set_dotted(d: dict[str, Any], key: str, val: Any) -> None:
    cur = d
    parts = key.split(".")
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = cur[p] = {}
        cur = nxt
    cur[parts[-1]] = val


DEFAULT_TRANSLATION_LLM = "gemma4_12b"


def chosen_translation_llm(s: Settings) -> str:
    """The translation model picked in Settings when it is not the default and still exists, else ``""``.

    A model added by link is a registry backend only while its entry is in ``custom_models.json``, so a name that
    is gone (deleted model) silently falls back to the default instead of breaking every run.
    """
    name = (getattr(s, "translation_llm", "") or "").strip()
    if not name or name == DEFAULT_TRANSLATION_LLM:
        return ""
    try:
        from vndub.backends import registry

        from vndub_app import hf_custom

        hf_custom.sync_env()
        spec = registry.get(name)
    except Exception:
        return ""
    return name if spec.role == "llm" and not spec.name.startswith("mock") else ""


def settings_snapshot(slug: str | None = None, settings: Settings | None = None) -> dict[str, Any]:
    """What a job's behaviour depends on, as plain JSON: kept on the job record (``JobRecord.settings``) so a resume can
    tell that the GPU budget, the translation model, the license policy, the folders, the profile or the project's
    languages changed since the job was submitted. ``src_lang`` / ``tgt_lang`` are included for a known project."""
    s = settings or load_settings()
    snap: dict[str, Any] = {
        "gpu_budget_gb": float(s.gpu_budget_gb),
        "translation_llm": chosen_translation_llm(s) or DEFAULT_TRANSLATION_LLM,
        "license_policy": s.license_policy,
        "models_dir": str(effective_models_dir(s)),
        "envs_dir": str(effective_envs_dir(s)),
        "profile": s.profile,
    }
    if slug:
        try:
            tr = read_app_state(project_handle(slug)).get("translation", {})
        except Exception:  # an unknown project: the languages are simply not known
            tr = {}
        if tr:
            snap["src_lang"] = str(tr.get("src_lang") or "auto")
            snap["tgt_lang"] = str(tr.get("tgt_lang") or "ru")
    return snap


def write_project_config(h: ProjectHandle, settings: Settings | None = None) -> Path:
    """(Re)generate ``<project>/pipeline.yaml`` from the bundled profile + app choices.

    Relative resource paths of the bundled profile are made absolute so the config works
    from any cwd (packaged app). App DEFAULTS: ``link.mode: no_asr`` (spec 0.1 mode A; the
    Advanced checkbox ``asr_check`` switches it to ``asr``), project glossary, data-folder
    models dir and the tool paths the installer knows (``tools.local.yaml``, installed or
    bundled with the app: ``installer.tool_paths``). ``app_state.config_overrides``
    (dotted keys) are applied last.
    """
    from vndub_app import installer  # imports this module

    s = settings or load_settings()
    st = read_app_state(h)
    base = base_config_path(s.profile)
    root = resource_root()
    cfg: dict[str, Any] = {
        "extends": str(base),
        "license_policy": s.license_policy,
        "tools_file": str(root / "configs" / "tools.yaml"),
        "tools": installer.tool_paths(),
        "gpu": {"vram_budget_gb": float(s.gpu_budget_gb)},
        "project": {
            "game_dir": str(h.game_dir),
            "work_dir": str(h.work),
            "out_dir": str(h.out),
            "models_dir": str(effective_models_dir(s)),
            "tgt_lang": st.get("translation", {}).get("tgt_lang", "ru"),
            "src_lang": st.get("translation", {}).get("src_lang") or "auto",
        },
        "link": {"mode": "asr" if st.get("advanced", {}).get("asr_check") else "no_asr"},
        "translate": {
            "glossary": str(h.glossary_file),
            "interjections": str(root / "configs" / "interjections.yaml"),
        },
        "text_emotion": {"cues": str(root / "configs" / "delivery_cues.yaml")},
        # mode B: the app dubs speaker lines without original audio too (route P); the laughter table like the other bundled
        # tables (the packaged app runs from its exe folder, where a relative configs/ path finds nothing)
        "synth": {"voice_unvoiced": True, "laughter": {"table": str(root / "configs" / "laughter.yaml")}},
    }
    tr = st.get("translation", {})
    if tr.get("source") == "import" and tr.get("import_path"):
        cfg["translate"]["import"] = tr["import_path"]
    llm = chosen_translation_llm(s)
    if llm:
        cfg["translate"]["llm"] = {"backend": llm}
    raw_base = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
    mock_dict = (raw_base.get("translate") or {}).get("mock_dictionary")
    if mock_dict:
        cfg["translate"]["mock_dictionary"] = str(root / mock_dict)
    for k, v in (st.get("config_overrides") or {}).items():
        _set_dotted(cfg, k, v)
    h.root.mkdir(parents=True, exist_ok=True)
    text = "# generated by the VNDub app; edit choices in the app, not here\n" + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    with _WRITE_LOCK:  # request threads and job threads both regenerate it
        tmp = h.config_file.with_name(f"{h.config_file.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        replace_atomic(tmp, h.config_file)
    return h.config_file


# ----------------------------------------------------------------------------
# small json helpers
# ----------------------------------------------------------------------------


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


REPLACE_TRIES = 10  # os.replace attempts before a PermissionError is given up on ...
REPLACE_PAUSE_S = 0.03  # ... with this pause between them (0.3 s at most)


def replace_atomic(tmp: Path, path: Path) -> None:
    """``os.replace(tmp, path)`` that waits out a reader. On Windows the replace fails with ``PermissionError`` (a sharing
    violation) while another thread has ``path`` open, and ``load_settings`` opens ``settings.json`` on nearly every request.
    Retries a few times with a short pause; when it still fails ``tmp`` is deleted (no stray temp file) and the error raised."""
    try:
        for attempt in range(REPLACE_TRIES):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == REPLACE_TRIES - 1:
                    raise
                time.sleep(REPLACE_PAUSE_S)
    except BaseException:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_json_atomic(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    try:
        payload = text.encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate (an odd NTFS file name in a log line): ASCII escapes keep it and json.loads restores it
        payload = json.dumps(data, ensure_ascii=True, indent=2).encode("utf-8")
    with _WRITE_LOCK:  # several threads write settings / state / job records; a shared temp name would collide
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_bytes(payload)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        replace_atomic(tmp, path)
