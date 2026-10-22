"""Doctor: rows from fake probes (no GPU / 6 GB / 8 GB / no ffmpeg / low disk ...), fixes, endpoints."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import pytest
from o_helpers import RangeServer, client_for, drop_runner, fresh_runner, make_env, tool_entry, wait_job, write_manifest, zip_bytes

from vndub_app import appdirs, doctor, installer, openers

GOOD_TOOLS = {"ffmpeg": {"present": True, "path": "C:/t/ffmpeg.exe", "required": True}, "ffprobe": {"present": True, "path": "C:/t/ffprobe.exe", "required": True}, "rubberband": {"present": False, "path": None, "required": False}}
ALL_ENVS = {"needed": ["asr", "hf", "score"], "present": ["asr", "hf", "score"]}
FULL_MODELS = {"default_present": 31, "default_total": 31, "default_missing_gb": 0.0, "min_present": 7, "min_total": 7}
GOOD_VC = doctor.VcRuntime(True, "14.44.35211", "registry")
GOOD_GIT = {"path": "C:/Program Files/Git/cmd/git.exe", "source": "system", "frozen": False, "envs": ["funasr", "hf", "rvc", "score", "seedvc"]}


def gpu(name: str = "NVIDIA GeForce RTX 4070", mib: int = 12282, driver: str = "572.16", cuda: str | None = "12.8", count: int = 1) -> doctor.GpuInfo:
    return doctor.parse_nvidia_smi(f"{name}, {mib}, {driver}\n" * count, f"| NVIDIA-SMI {driver}  Driver Version: {driver}  CUDA Version: {cuda} |" if cuda else "")  # type: ignore[return-value]


def probes(**kw: Any) -> doctor.Probes:
    base = dict(
        gpu=lambda: gpu(),
        disk_free_gb=lambda p: 500.0,
        internet=lambda: True,
        tools=lambda: GOOD_TOOLS,
        vcredist=lambda: GOOD_VC,
        vcredist_bundled=lambda: True,
        git=lambda: GOOD_GIT,
        envs=lambda: ALL_ENVS,
        models=lambda: FULL_MODELS,
        data_dir=lambda: Path("C:/VNDub"),
        profile=lambda: "default",
        platform="win32",
    )
    base.update(kw)
    return doctor.Probes(**base)


def rows(p: doctor.Probes) -> dict[str, dict[str, Any]]:
    rep = doctor.run_doctor(p)
    return {r["id"]: r for r in rep["rows"]}


@pytest.fixture
def o_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    data = make_env(monkeypatch, tmp_path)
    fresh_runner()
    yield data
    doctor.set_probes(None)
    installer.set_manifest_override(None)
    openers.set_opener(None)
    drop_runner()


# ---- nvidia-smi parsing -------------------------------------------------------------------------------


def test_parse_nvidia_smi_reads_name_vram_driver_cuda() -> None:
    g = doctor.parse_nvidia_smi("NVIDIA GeForce RTX 3070, 8192, 551.23\n", "| NVIDIA-SMI 551.23   Driver Version: 551.23   CUDA Version: 12.4 |")
    assert g is not None
    assert (g.name, g.vram_gb, g.driver, g.cuda, g.count) == ("NVIDIA GeForce RTX 3070", 8.0, "551.23", "12.4", 1)


def test_parse_nvidia_smi_edge_cases() -> None:
    assert doctor.parse_nvidia_smi("") is None
    assert doctor.parse_nvidia_smi("garbage") is None
    g = doctor.parse_nvidia_smi("NVIDIA GeForce RTX 4060 Laptop GPU, [N/A], 560.94\n")
    assert g is not None and g.vram_gb is None and g.cuda is None
    two = doctor.parse_nvidia_smi("NVIDIA A, 24576, 560.1\nNVIDIA B, 8192, 560.1\n", "CUDA Version: 12.6")
    assert two is not None and two.name == "NVIDIA A" and two.count == 2  # device 0 is what the pipeline uses
    comma = doctor.parse_nvidia_smi("NVIDIA RTX, Special Edition, 8192, 551.23\n")
    assert comma is not None and comma.name == "NVIDIA RTX, Special Edition"


# ---- GPU rows -------------------------------------------------------------------------------------------


def test_no_nvidia_gpu_is_red_with_plain_reasons(o_env: Path) -> None:
    r = rows(probes(gpu=lambda: None))
    assert r["gpu"]["status"] == "red" and r["gpu"]["params"]["reason"] == "no_nvidia" and r["gpu"]["fix"] is None
    assert r["driver"]["status"] == "red" and r["driver"]["fix"]["action"] == "open_driver_page"
    assert r["vram"]["status"] == "red" and r["vram"]["params"]["reason"] == "no_nvidia" and r["vram"]["params"]["need_gb"] == 8
    rep = doctor.run_doctor(probes(gpu=lambda: None))
    assert rep["gpu"] is None and rep["summary"] == "fail" and rep["torch"] == "cpu"


@pytest.mark.parametrize(
    ("mib", "status", "reason"),
    [
        (24576, "green", None),
        (8192, "green", None),
        (8188, "green", None),  # RTX 4060 8 GB reports 8188 MiB
        (7600, "yellow", "tight"),
        (6144, "yellow", "tight"),  # 6 GB card
        (6141, "yellow", "tight"),
        (4096, "red", "low_vram"),
        (2048, "red", "low_vram"),
    ],
)
def test_vram_thresholds(o_env: Path, mib: int, status: str, reason: str | None) -> None:
    r = rows(probes(gpu=lambda: gpu(mib=mib)))["vram"]
    assert r["status"] == status
    assert r["params"].get("reason") == reason
    assert r["value"] == pytest.approx(mib / 1024, abs=0.06)


def test_six_gb_card_is_yellow_not_red(o_env: Path) -> None:
    rep = doctor.run_doctor(probes(gpu=lambda: gpu(mib=6144)))
    assert next(x for x in rep["rows"] if x["id"] == "vram")["status"] == "yellow"
    assert rep["summary"] == "warn" and rep["gpu"]["vram_gb"] == 6.0


def test_unknown_vram_is_yellow(o_env: Path) -> None:
    g = replace(gpu(), vram_gb=None)
    r = rows(probes(gpu=lambda: g))["vram"]
    assert r["status"] == "yellow" and r["params"]["reason"] == "unknown"


@pytest.mark.parametrize(
    ("cuda", "status", "reason"),
    [("12.8", "green", None), ("13.0", "green", None), ("12.4", "yellow", "old_driver"), ("12.1", "yellow", "old_driver"), ("11.8", "red", "too_old"), (None, "yellow", "cuda_unknown")],
)
def test_driver_rows(o_env: Path, cuda: str | None, status: str, reason: str | None) -> None:
    r = rows(probes(gpu=lambda: gpu(cuda=cuda)))["driver"]
    assert r["status"] == status and r["params"].get("reason") == reason
    assert (r["fix"] is None) == (status == "green")
    assert r["fix"] is None or r["fix"]["action"] == "open_driver_page"
    assert r["value"] == (f"572.16 (CUDA {cuda})" if cuda else "572.16")


def test_torch_choice(o_env: Path) -> None:
    assert doctor.choose_torch(gpu()) == "cuda"
    assert doctor.choose_torch(gpu(cuda="12.2")) == "cuda"
    assert doctor.choose_torch(gpu(cuda="11.8")) == "cpu"
    assert doctor.choose_torch(gpu(cuda=None)) == "cuda"
    assert doctor.choose_torch(None) == "cpu"
    doctor.set_probes(probes(gpu=lambda: None))
    assert doctor.choose_torch() == "cpu"
    doctor.set_probes(probes())
    assert doctor.choose_torch() == "cuda"


# ---- other rows -----------------------------------------------------------------------------------------


def test_missing_ffmpeg_is_red_with_a_download_fix(o_env: Path) -> None:
    tools = {**GOOD_TOOLS, "ffmpeg": {"present": False, "path": None, "required": True}, "ffprobe": {"present": False, "path": None, "required": True}}
    r = rows(probes(tools=lambda: tools))["ffmpeg"]
    assert r["status"] == "red" and r["fix"] == {"action": "download_ffmpeg", "label_key": "doctor.fix.download_ffmpeg"}
    assert set(r["params"]["missing"].split(", ")) == {"ffmpeg", "ffprobe"}
    only_probe = {**GOOD_TOOLS, "ffprobe": {"present": False, "path": None, "required": True}}
    assert rows(probes(tools=lambda: only_probe))["ffmpeg"]["params"]["missing"] == "ffprobe"
    assert rows(probes())["ffmpeg"]["status"] == "green"


def reg_reader(**values: int):  # type: ignore[no-untyped-def]
    return values.get


def dlls(**versions: tuple[int, int, int, int]):  # type: ignore[no-untyped-def]
    return lambda name: versions.get(name)


NEW_DLLS = {"msvcp140.dll": (14, 44, 35211, 0), "vcruntime140_1.dll": (14, 44, 35211, 0)}


@pytest.mark.parametrize(
    ("registry", "files", "ok", "version", "source"),
    [
        ({"Installed": 1, "Major": 14, "Minor": 44, "Bld": 35211}, {}, True, "14.44.35211", "registry"),
        ({"Installed": 1, "Major": 14, "Minor": 40, "Bld": 33807}, {}, True, "14.40.33807", "registry"),  # the lowest accepted
        ({"Installed": 1, "Major": 14, "Minor": 39, "Bld": 33519}, {}, False, "14.39.33519", "registry"),  # VS 2022 17.9: too old
        ({"Installed": 1, "Major": 14, "Minor": 29, "Bld": 30133}, {}, False, "14.29.30133", "registry"),  # VS 2019
        ({"Installed": 1, "Major": 15, "Minor": 0, "Bld": 1}, {}, True, "15.0.1", "registry"),  # a future major version
        ({"Installed": 1, "Major": 14, "Minor": 51}, {}, True, "14.51.0", "registry"),  # Bld may be absent
        ({"Installed": 0, "Major": 14, "Minor": 44, "Bld": 35211}, {}, False, None, "none"),  # Installed=0 is not installed
        ({"Installed": 1, "Major": 14, "Minor": 44}, {}, True, "14.44.0", "registry"),
        ({}, {}, False, None, "none"),  # nothing at all
        # the registry says nothing (or something old) but the DLLs are new: accepted through System32
        ({}, NEW_DLLS, True, "14.44.35211", "files"),
        ({"Installed": 1, "Major": 14, "Minor": 38, "Bld": 1}, NEW_DLLS, True, "14.44.35211", "files"),
        # one DLL missing / old: the fallback does not accept it
        ({}, {"msvcp140.dll": (14, 44, 35211, 0)}, False, "14.44.35211", "files"),
        ({}, {"msvcp140.dll": (14, 44, 35211, 0), "vcruntime140_1.dll": (14, 29, 30133, 0)}, False, "14.29.30133", "files"),
    ],
)
def test_vc_runtime_decision(registry: dict[str, int], files: dict[str, tuple[int, int, int, int]], ok: bool, version: str | None, source: str) -> None:
    rt = doctor.decide_vcruntime(reg_reader(**registry), dlls(**files))
    assert (rt.ok, rt.version, rt.source) == (ok, version, source)


def test_vc_runtime_threshold_is_the_one_of_the_installer_script() -> None:
    """The Doctor row and VNDub.iss decide with the same key and the same minimum version."""
    iss = (Path(__file__).resolve().parents[2] / "packaging" / "VNDub.iss").read_bytes().decode("utf-8-sig")
    assert f"VCRuntimeKey = '{doctor.VC_REG_KEY}'" in iss
    assert f"VCMinMajor = {doctor.VC_MIN[0]};" in iss and f"VCMinMinor = {doctor.VC_MIN[1]};" in iss


def test_vcredist_row_is_green_when_the_runtime_is_there(o_env: Path) -> None:
    r = rows(probes())["vcredist"]
    assert r["status"] == "green" and r["value"] == "14.44.35211" and r["fix"] is None
    assert r["params"]["version"] == "14.44.35211" and r["params"]["min_version"] == "14.40"


def test_vcredist_row_is_red_with_the_bundled_installer_as_the_fix(o_env: Path) -> None:
    missing = probes(vcredist=lambda: doctor.VcRuntime(False, None, "none"), vcredist_bundled=lambda: True)
    r = rows(missing)["vcredist"]
    assert r["status"] == "red" and r["params"]["reason"] == "missing" and r["value"] is None
    assert r["fix"] == {"action": "install_vcredist", "label_key": "doctor.fix.install_vcredist"}
    rep = doctor.run_doctor(missing)
    assert rep["summary"] == "fail"  # no model starts without it
    old = probes(vcredist=lambda: doctor.VcRuntime(False, "14.29.30133", "registry"))
    r = rows(old)["vcredist"]
    assert r["status"] == "red" and r["params"]["reason"] == "too_old" and r["params"]["version"] == "14.29.30133"


def test_vcredist_row_opens_the_microsoft_page_in_a_dev_run(o_env: Path) -> None:
    dev = probes(vcredist=lambda: doctor.VcRuntime(False, None, "none"), vcredist_bundled=lambda: False)
    r = rows(dev)["vcredist"]
    assert r["status"] == "red" and r["fix"] == {"action": "open_vcredist_page", "label_key": "doctor.fix.open_vcredist_page"}


def test_vcredist_row_is_not_a_thing_off_windows_and_yellow_when_the_probe_fails(o_env: Path) -> None:
    lin = rows(probes(platform="linux", vcredist=lambda: doctor.VcRuntime(False, None, "none")))["vcredist"]
    assert lin["status"] == "green" and lin["fix"] is None and lin["params"]["reason"] == "not_applicable"

    def boom() -> doctor.VcRuntime:
        raise OSError("registry unavailable")

    unknown = rows(probes(vcredist=boom))["vcredist"]
    assert unknown["status"] == "yellow" and unknown["fix"] is None  # unknown, never an HTTP 500
    assert doctor.run_doctor(probes(vcredist=boom))["summary"] == "warn"


def test_git_row_says_where_git_comes_from(o_env: Path) -> None:
    sysgit = rows(probes())["git"]
    assert sysgit["status"] == "green" and sysgit["value"] == GOOD_GIT["path"] and sysgit["params"]["source"] == "system" and "reason" not in sysgit["params"]
    bundled = rows(probes(git=lambda: {**GOOD_GIT, "path": "C:/VNDub/git/cmd/git.exe", "source": "bundled", "frozen": True}))["git"]
    assert bundled["status"] == "green" and bundled["params"]["source"] == "bundled" and bundled["params"]["reason"] == "bundled"


def test_git_row_is_a_warning_naming_the_envs_that_need_it(o_env: Path) -> None:
    none = {"path": None, "source": None, "frozen": False, "envs": ["rvc", "seedvc", "hf"]}
    r = rows(probes(git=lambda: none))["git"]
    assert r["status"] == "yellow" and r["value"] is None and r["fix"] is None
    assert r["params"] == {"envs": "rvc, seedvc, hf"}
    assert doctor.run_doctor(probes(git=lambda: none))["summary"] == "warn"  # a warning, not a failure
    lost = rows(probes(git=lambda: {**none, "frozen": True}))["git"]  # the packaged app should have carried it
    assert lost["status"] == "yellow" and lost["params"]["reason"] == "bundled_missing"


def test_real_runtime_probes_run_here(o_env: Path) -> None:
    """The real registry / version.dll / PATH probes on this machine (any answer is fine, a crash is not)."""
    import sys

    rt = doctor.probe_vcredist()
    assert isinstance(rt, doctor.VcRuntime)
    if sys.platform.startswith("win"):
        assert rt.applicable and (rt.ok or rt.version is None or rt.version.count(".") == 2)
        k32 = doctor._file_version(str(Path(os.environ["SystemRoot"]) / "System32" / "kernel32.dll"))
        assert k32 is not None and k32[0] >= 6
        assert doctor._file_version(str(Path(os.environ["SystemRoot"]) / "System32" / "no-such-file.dll")) is None
    g = doctor.probe_git()
    assert set(g) == {"path", "source", "frozen", "envs"} and g["source"] in (None, "system", "bundled")
    assert "rvc" in g["envs"] and "asr" not in g["envs"]


@pytest.mark.parametrize(("free", "status"), [(500.0, "green"), (95.0, "green"), (70.0, "yellow"), (30.0, "red"), (4.0, "red")])
def test_disk_row_compares_free_space_with_what_is_still_to_download(o_env: Path, free: float, status: str) -> None:
    models = {**FULL_MODELS, "default_missing_gb": 64.6}
    envs = {"needed": ["asr", "hf", "score", "rvc"], "present": ["asr"]}  # 3 envs x 4 GB missing
    need = doctor.disk_need_gb(models, envs)
    assert need == pytest.approx(64.6 + 12 + 10)
    r = rows(probes(disk_free_gb=lambda p: free, models=lambda: models, envs=lambda: envs))["disk"]
    assert r["status"] == status
    assert r["params"]["free_gb"] == pytest.approx(free) and r["params"]["need_gb"] == pytest.approx(need, abs=0.1)
    assert (r["fix"] is None) == (status == "green") and (r["fix"] is None or r["fix"]["action"] == "open_data_dir")


def test_everything_installed_needs_only_working_room(o_env: Path) -> None:
    assert rows(probes(disk_free_gb=lambda p: 20.0))["disk"]["status"] == "green"
    assert rows(probes(disk_free_gb=lambda p: 3.0))["disk"]["status"] == "red"


def test_internet_row_and_probe_failures(o_env: Path) -> None:
    assert rows(probes(internet=lambda: False))["internet"]["status"] == "yellow"

    def boom() -> bool:
        raise OSError("network down")

    r = rows(probes(internet=boom, gpu=lambda: (_ for _ in ()).throw(RuntimeError("nvidia-smi crashed"))))
    assert r["internet"]["status"] == "yellow" and r["gpu"]["status"] == "red"  # a failing probe is an unknown, never an HTTP 500


def test_defender_row_is_a_manual_card_on_windows_only(o_env: Path) -> None:
    r = rows(probes())["defender"]
    assert r["status"] == "yellow" and r["fix"]["action"] == "add_defender_exclusion_help" and r["params"]["path"] == str(Path("C:/VNDub"))
    lin = rows(probes(platform="linux"))["defender"]
    assert lin["status"] == "green" and lin["fix"] is None


def test_envs_and_models_rows(o_env: Path) -> None:
    some = {"needed": ["asr", "hf", "score"], "present": ["asr"]}
    none = {"needed": ["asr", "hf", "score"], "present": []}
    assert rows(probes(envs=lambda: some))["envs"]["status"] == "yellow"
    r = rows(probes(envs=lambda: none))["envs"]
    assert r["status"] == "red" and r["fix"]["action"] == "open_tools" and r["fix"]["label_key"] == "actions.open_tools"
    assert r["value"]["missing"] == ["asr", "hf", "score"]
    part = {"default_present": 10, "default_total": 31, "default_missing_gb": 30.0, "min_present": 7, "min_total": 7}
    zero = {"default_present": 4, "default_total": 31, "default_missing_gb": 60.0, "min_present": 1, "min_total": 7}
    assert rows(probes(models=lambda: part))["models"]["status"] == "yellow"
    m = rows(probes(models=lambda: zero))["models"]
    assert m["status"] == "red" and m["fix"]["action"] == "open_models"


def test_mock_profile_does_not_ask_for_envs_or_models(o_env: Path) -> None:
    rep = doctor.run_doctor(probes(profile=lambda: "apptest", envs=lambda: {"needed": ["asr"], "present": []}, models=lambda: {"default_total": 5, "default_present": 0, "min_total": 2, "min_present": 0}))
    by = {r["id"]: r for r in rep["rows"]}
    assert rep["mock"] is True and by["envs"]["status"] == "green" and by["models"]["status"] == "green"


def test_summary_ignores_the_standing_defender_hint(o_env: Path) -> None:
    rep = doctor.run_doctor(probes())
    assert rep["summary"] == "ok"
    assert {r["id"] for r in rep["rows"]} == set(doctor.ROW_IDS)
    assert doctor.summarize([{"id": "x", "status": "yellow"}]) == "warn"
    assert doctor.summarize([{"id": "x", "status": "red"}, {"id": "y", "status": "yellow"}]) == "fail"
    assert doctor.summarize([{"id": "defender", "status": "red"}]) == "ok"


def test_rows_carry_ids_and_codes_only(o_env: Path) -> None:
    """No display text from the backend: statuses, ids, numbers, i18n keys and machine reasons."""
    rep = doctor.run_doctor(probes(gpu=lambda: None))
    blob = json.dumps(rep, ensure_ascii=False)
    assert not any("\u0400" <= ch <= "\u04ff" for ch in blob), "Cyrillic text in a doctor payload"
    for r in rep["rows"]:
        assert r["status"] in ("green", "yellow", "red")
        if r["fix"]:
            assert set(r["fix"]) == {"action", "label_key"}


# ---- endpoints and fixes -----------------------------------------------------------------------------------


def test_doctor_endpoint_uses_the_injected_probes(o_env: Path) -> None:
    doctor.set_probes(probes(gpu=lambda: gpu(mib=8188)))
    c = client_for(doctor)
    body = c.get("/api/doctor").json()
    assert body["summary"] == "ok" and body["gpu"] == {"name": "NVIDIA GeForce RTX 4070", "vram_gb": 8.0, "driver": "572.16", "cuda": "12.8"}
    assert [r["id"] for r in body["rows"]] == list(doctor.ROW_IDS)
    assert body["torch"] == "cuda" and body["profile"] == "default" and body["data_dir"]


def test_real_probes_run_here_without_crashing(o_env: Path) -> None:
    """The real probes (nvidia-smi absent on the build machine is a legitimate case)."""
    rep = doctor.run_doctor(replace(doctor.Probes(), internet=lambda: False))
    assert [r["id"] for r in rep["rows"]] == list(doctor.ROW_IDS)
    assert rep["summary"] in ("ok", "warn", "fail")


def test_fix_open_data_dir_and_driver_page_and_defender_help(o_env: Path) -> None:
    opened: list[Path] = []
    openers.set_opener(opened.append)
    c = client_for(doctor)
    r = c.post("/api/doctor/fix/open_data_dir").json()
    assert r["ok"] is True and opened == [appdirs.data_dir()]
    d = c.post("/api/doctor/fix/open_driver_page").json()
    assert d == {"ok": True, "url": "https://www.nvidia.com/Download/index.aspx"} and opened == [appdirs.data_dir()]  # nothing was opened
    h = c.post("/api/doctor/fix/add_defender_exclusion_help").json()
    assert h["ok"] and h["command"] == f"Add-MpPreference -ExclusionPath '{appdirs.data_dir()}'" and h["admin_required"] is True
    assert h["settings_path_key"] == "doctor.defender.settings_path"
    bad = c.post("/api/doctor/fix/format_c")
    assert bad.status_code == 404 and bad.json()["error"]["code"] == "bad_request"


def test_defender_command_quotes_apostrophes() -> None:
    assert doctor.defender_command(Path("C:/Users/O'Neil/VNDub")) == "Add-MpPreference -ExclusionPath '" + str(Path("C:/Users/O'Neil/VNDub")).replace("'", "''") + "'"
    assert "''" in doctor.defender_command(Path("C:/Users/O'Neil/VNDub"))


def test_download_ffmpeg_fix_is_a_job_that_installs_the_tool(o_env: Path, tmp_path: Path) -> None:
    exe = b"MZ fake ffmpeg"
    archive = zip_bytes({"ffmpeg-x/bin/ffmpeg.exe": exe, "ffmpeg-x/bin/ffprobe.exe": b"MZ fake ffprobe"})
    with RangeServer({"ffmpeg.zip": archive}) as srv:
        e = tool_entry("ffmpeg", srv.url("ffmpeg.zip"), "ffmpeg.zip", archive, provides={"ffmpeg": "ffmpeg-x/bin/ffmpeg.exe", "ffprobe": "ffmpeg-x/bin/ffprobe.exe"}, required=True)
        installer.set_manifest_override(write_manifest(tmp_path / "m.json", [e]))
        c = client_for(doctor)
        job = c.post("/api/doctor/fix/download_ffmpeg").json()
        assert job["kind"] == "ffmpeg_download" and job["gpu"] is False
        done = wait_job(job)
    assert done["state"] == "done", done.get("error")
    local = installer.read_local_tools()
    assert Path(local["ffmpeg"]).read_bytes() == exe and Path(local["ffprobe"]).exists()
    assert appdirs.tools_local_yaml().exists()
