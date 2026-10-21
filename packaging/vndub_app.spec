# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: one-folder build ``dist/VNDub/VNDub.exe`` (windowed, no console).

The packaged app holds only the light core (the pipeline's own imports: numpy, scipy, soundfile, librosa,
pydantic ...) plus the API server and the window shell. Backend venvs, models and external tools are
installed into the data folder by the in-app installer, so torch and every ML package are excluded.

Bundled as data (``appdirs.resource_root()`` = the bundle folder): ``configs/``, ``app/`` (i18n,
tools manifest), ``docs/``, ``README.md`` (in-app help), ``adapters/`` (sources: they run inside the backend envs), ``envs/``
(requirement files), ``scripts/make_envs.*``, ``vndub_app/web/dist`` (when the SPA is built),
``vndub/report/templates`` and ``bin/uv.exe`` (copied to ``packaging/_staging`` by build_app.ps1).

Build:  uv run pyinstaller packaging/vndub_app.spec --noconfirm --distpath dist --workpath build/pyinstaller
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821  (SPECPATH is provided by PyInstaller)
STAGING = ROOT / "packaging" / "_staging"
ICON = ROOT / "packaging" / "VNDub.ico"

SKIP_PARTS = {"__pycache__", ".pytest_cache", "node_modules", ".git", ".DS_Store"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
SKIP_FILES = {"tools.local.yaml"}  # per-machine tool paths never ship


def tree(src, dest):
    """(file, dest folder) pairs for every file below ``src`` (no caches, no per-machine files)."""
    src = Path(src)
    out = []
    if not src.is_dir():
        return out
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if p.is_dir() or SKIP_PARTS & set(rel.parts) or p.suffix in SKIP_SUFFIXES or p.name in SKIP_FILES:
            continue
        out.append((str(p), (Path(dest) / rel.parent).as_posix()))
    return out


datas = []
datas += tree(ROOT / "configs", "configs")
datas += tree(ROOT / "app", "app")
datas += tree(ROOT / "docs", "docs")
datas += tree(ROOT / "adapters", "adapters")
datas += tree(ROOT / "envs", "envs")
datas += tree(ROOT / "vndub" / "report" / "templates", "vndub/report/templates")
datas += tree(ROOT / "vndub_app" / "web" / "dist", "vndub_app/web/dist")
for script in ("make_envs.ps1", "make_envs.sh"):
    datas.append((str(ROOT / "scripts" / script), "scripts"))
datas.append((str(ROOT / "README.md"), "."))  # shown by the in-app help (vndub_app/routes/general.py HELP_DOCS)
if (STAGING / "uv.exe").is_file():
    datas.append((str(STAGING / "uv.exe"), "bin"))
elif (STAGING / "uv").is_file():
    datas.append((str(STAGING / "uv"), "bin"))

# modules the pipeline and the server load by name (importlib) are invisible to the static analysis
hiddenimports = []
hiddenimports += collect_submodules("vndub")
hiddenimports += collect_submodules("vndub_app")
hiddenimports += collect_submodules("uvicorn")
hiddenimports += ["python_multipart", "multipart", "anyio._backends._asyncio", "h11", "httptools"]
# huggingface_hub 2.x imports most of itself lazily and talks through httpx2/httpcore2
for optional in ("huggingface_hub", "httpx2", "httpcore2", "httpx", "httpcore", "websockets", "wsproto", "hf_xet", "webview", "clr_loader", "pythonnet"):
    try:
        __import__(optional)
    except Exception:
        continue
    hiddenimports += collect_submodules(optional)
hiddenimports = sorted(set(hiddenimports))

# heavy things only the backend envs (or the Gradio review tool / the tests) need
excludes = [
    "torch", "torchaudio", "torchvision", "tensorflow", "onnxruntime", "sympy", "transformers",
    "gradio", "gradio_client", "hf_gradio", "matplotlib", "pandas", "PIL", "tkinter", "IPython", "notebook",
    "pytest", "_pytest", "playwright", "PyInstaller",
]

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VNDub",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # windowed: no console; stdout/stderr are replaced by vndub_app.shell
    icon=str(ICON) if ICON.is_file() else None,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VNDub",
)
