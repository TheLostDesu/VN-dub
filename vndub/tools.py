"""External CLI tools: lookup from ``tools.yaml`` and ``vndub tools check``.

Tools are never vendored. ``configs/tools.yaml`` maps tool name -> path (empty
means "search PATH for the default executable name"). Inline ``tools:`` in the
pipeline config override the file.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ToolInfo:
    exe: str  # default executable name searched on PATH
    purpose: str
    url: str
    engines: tuple[str, ...] = ()
    required: bool = False
    pip: str = ""  # pip-installable alternative


TOOL_INFO: dict[str, ToolInfo] = {
    "ffmpeg": ToolInfo("ffmpeg", "decode/encode audio, atempo stretch fallback", "https://www.gyan.dev/ffmpeg/builds/ (winget install Gyan.FFmpeg)", required=True),
    "ffprobe": ToolInfo("ffprobe", "probe codec/sr/channels of voice files", "ships with ffmpeg", required=True),
    "vgmstream_cli": ToolInfo("vgmstream-cli", "decode proprietary game audio (hca, at9, fsb, ovk...)", "https://github.com/vgmstream/vgmstream/releases (nightly Windows build)"),
    "rubberband": ToolInfo("rubberband", "high-quality time stretch in master", "https://breakfastquay.com/rubberband/ (Windows CLI zip)"),
    "unrpa": ToolInfo("unrpa", "Ren'Py .rpa extraction", "https://github.com/Lattyware/unrpa", engines=("renpy",), pip="unrpa"),
    "unrpyc": ToolInfo("unrpyc", "Ren'Py .rpyc -> .rpy decompiler (v2.x)", "https://github.com/CensoredUsername/unrpyc/releases", engines=("renpy",)),
    "xp3pack": ToolInfo("Xp3Pack", "KiriKiri patchN.xp3 builder (KirikiriTools)", "https://github.com/arcusmaximus/KirikiriTools/releases", engines=("kirikiri",)),
    "krkrdump": ToolInfo("KrkrDumpLoader", "KiriKiri Z runtime dumper (encrypted xp3)", "https://github.com/crskycode/KrkrDump/releases", engines=("kirikiri",)),
    "freemote_psbdecompile": ToolInfo("PsbDecompile", ".scn (PSB) -> JSON", "https://github.com/UlyssesWu/FreeMote/releases", engines=("kirikiri",)),
    "siglus_ssu": ToolInfo("siglus-ssu", "Siglus Scene.pck / .ovk extraction, koe_master.csv", "https://github.com/Jirehlov/SiglusSceneScriptUtility", engines=("siglus",)),
    "bgi_extractor": ToolInfo("bgi-extractor", "BGI/Ethornell archives -> speaker+text+voice JSON", "https://github.com/mirinnano/bgi-extractor", engines=("bgi",)),
    "catsystem2_cli": ToolInfo("TriggersTools.CatSystem2", "CatSystem2 KIF .int extract + CST decompile", "https://github.com/trigger-segfault/TriggersTools.CatSystem2", engines=("catsystem2",)),
    "exkifint": ToolInfo("exkifint", "CatSystem2 .int extractor (v2)", "asmodean tools (legacy)", engines=("catsystem2",)),
    "pfs_unpacker": ToolInfo("pfs-rs", "Artemis .pfs unpack/pack", "https://github.com/sakarie9/pfs-rs", engines=("artemis",)),
    "yrt": ToolInfo("yrt", "YU-RIS YPF extract / ybn decompile", "https://github.com/exsatsukirin/YU-RIS-TOOLS", engines=("yuris",)),
    "majiro_tools": ToolInfo("MajiroTools", "Majiro .mjo/.arc tools", "https://github.com/AtomCrafty/MajiroTools", engines=("majiro",)),
    "assetripper": ToolInfo("AssetRipper", "Unity asset export (--headless)", "https://github.com/AssetRipper/AssetRipper/releases", engines=("unity",)),
    "garbro_console": ToolInfo("gameres", "GARbro console fallback extractor", "https://github.com/crskycode/GARbro"),
    "arc_unpacker": ToolInfo("arc_unpacker", "legacy universal extractor", "https://github.com/vn-tools/arc_unpacker"),
    "vntextpatch": ToolInfo("VNTextPatch", "text+names for 22 engines (extractlocal)", "https://github.com/arcusmaximus/VNTranslationTools/releases"),
    "llama_server": ToolInfo("llama-server", "local LLM server for translate", "https://github.com/ggml-org/llama.cpp/releases (Windows CUDA 12 build)"),
    "applio_dir": ToolInfo("", "Applio checkout used by the rvc env (if not in .venvs/rvc/src/Applio)", "https://github.com/IAHispano/Applio"),
    "audiocpp": ToolInfo("audiocpp", "audio.cpp CLI for Fish S2 Pro GGUF", "https://github.com/andreaabcj-cmd/fish-audio-s2-pro-realtime-local-tts"),
    "zonos2_cli": ToolInfo("zonos2", "zonos2.cpp CLI", "https://github.com/Zyphra/zonos2.cpp/releases"),
    "moss_tts_cli": ToolInfo("moss-tts", "MOSS-TTS llama.cpp CLI", "https://github.com/OpenMOSS/MOSS-TTS"),
}


@dataclass
class Tools:
    paths: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, tools_file: Path | None, overrides: dict[str, str] | None = None) -> "Tools":
        data: dict[str, Any] = {}
        if tools_file is not None and tools_file.exists():
            data = yaml.safe_load(tools_file.read_text(encoding="utf-8")) or {}
            local = tools_file.with_name("tools.local.yaml")
            if local.exists():
                data.update(yaml.safe_load(local.read_text(encoding="utf-8")) or {})
        data.update({k: v for k, v in (overrides or {}).items()})
        return cls({k: ("" if v is None else str(v)) for k, v in data.items()})

    def resolve(self, name: str) -> Path | None:
        """Configured path if it exists, else the default executable on PATH."""
        configured = self.paths.get(name, "")
        if configured:
            p = Path(configured).expanduser()
            if p.exists():
                return p
            found = shutil.which(configured)
            return Path(found) if found else None
        info = TOOL_INFO.get(name)
        if info and info.exe:
            found = shutil.which(info.exe)
            if found:
                return Path(found)
        return None

    def require(self, name: str, purpose: str = "") -> Path:
        p = self.resolve(name)
        if p is None:
            info = TOOL_INFO.get(name)
            url = f" Download: {info.url}" if info else ""
            raise ToolMissing(name, f"tool '{name}' not found{(' (' + purpose + ')') if purpose else ''}; set it in configs/tools.yaml.{url}")
        return p

    def run(self, name: str, args: list[str], cwd: Path | None = None, timeout: float | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
        exe = self.require(name)
        return subprocess.run([str(exe), *args], cwd=cwd, timeout=timeout, check=check, capture_output=True, text=True, encoding="utf-8", errors="replace")

    def check(self, engine: str | None = None) -> list[dict[str, Any]]:
        rows = []
        for name, info in TOOL_INFO.items():
            if engine and info.engines and engine not in info.engines:
                continue
            p = self.resolve(name)
            rows.append(
                {
                    "tool": name,
                    "present": p is not None,
                    "path": str(p) if p else (self.paths.get(name) or ""),
                    "required": info.required,
                    "engines": ",".join(info.engines) or "all",
                    "purpose": info.purpose,
                    "url": info.url,
                    "pip": info.pip,
                }
            )
        return rows


class ToolMissing(RuntimeError):
    def __init__(self, name: str, message: str) -> None:
        self.name = name
        super().__init__(message)
