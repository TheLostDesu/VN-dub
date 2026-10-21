"""The pinned tools manifest (``app/tools_manifest.json``) and its validator."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from vndub_app import installer

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "app" / "tools_manifest.json"


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def entry(m: dict[str, Any], name: str) -> dict[str, Any]:
    return next(e for e in m["tools"] if e["name"] == name)


def problems(m: dict[str, Any], **kw: Any) -> list[str]:
    return installer.validate_manifest(m, **kw)


def mutated(m: dict[str, Any], name: str, **changes: Any) -> dict[str, Any]:
    m2 = copy.deepcopy(m)
    e = entry(m2, name)
    for k, v in changes.items():
        if v is KeyError:
            e.pop(k, None)
        else:
            e[k] = v
    return m2


# ---- the shipped manifest -------------------------------------------------------------------------


def test_shipped_manifest_is_valid(manifest: dict[str, Any]) -> None:
    assert problems(manifest) == []
    assert installer.load_manifest(MANIFEST)["schema"] == 1


def test_shipped_manifest_lists_the_required_tools(manifest: dict[str, Any]) -> None:
    names = {e["name"] for e in manifest["tools"]}
    assert {"ffmpeg", "rubberband", "vgmstream_cli", "llama_server", "xp3pack", "freemote_psbdecompile", "bgi_extractor", "unrpa", "unrpyc", "siglus_ssu"} <= names
    # every provided tool key is one the pipeline looks up
    from vndub.tools import TOOL_INFO

    for e in manifest["tools"]:
        assert set(e["provides"]) <= set(TOOL_INFO), e["name"]


def test_shipped_downloads_are_https_and_hash_pinned(manifest: dict[str, Any]) -> None:
    for e in manifest["tools"]:
        if e.get("manual"):
            continue
        if e["kind"] == "download":
            assert e["url"].startswith("https://"), e["name"]
            assert e["verify"] == "sha256" and len(e["sha256"]) == 64, e["name"]
            assert isinstance(e["size"], int) and e["size"] > 0, e["name"]
            for x in e["extra"]:
                assert x["url"].startswith("https://") and x["verify"] == "sha256", (e["name"], x["filename"])
        else:
            assert e["pip"]["hashes"] and all(h.startswith("sha256:") for h in e["pip"]["hashes"]), e["name"]


def test_unverified_entries_are_explicit(manifest: dict[str, Any]) -> None:
    """Nothing is unverified silently: the only exception is spelled out and listed by manifest_warnings."""
    unverified = [e["name"] for e in manifest["tools"] if e["verify"] != "sha256"]
    assert unverified == ["bgi_extractor"]
    assert entry(manifest, "bgi_extractor")["unverified_reason"]
    warns = installer.manifest_warnings(manifest)
    assert any("bgi_extractor" in w and "not hash-verified" in w for w in warns)
    assert not any(w.startswith("ffmpeg") for w in warns)


def test_engines_of_the_manifest_are_real_engines(manifest: dict[str, Any]) -> None:
    from vndub.engines import ENGINES

    for e in manifest["tools"]:
        assert all(x == "*" or x in ENGINES for x in e["engines"]), e["name"]
    assert entry(manifest, "unrpyc")["engines"] == ["renpy"]
    assert entry(manifest, "xp3pack")["engines"] == ["kirikiri"]


# ---- the validator rejects what it must ---------------------------------------------------------------


def test_null_sha256_needs_an_explicit_verify_mode(manifest: dict[str, Any]) -> None:
    bad = mutated(manifest, "rubberband", sha256=None)  # verify is still "sha256"
    assert any("sha256 is null but verify is 'sha256'" in p for p in problems(bad))
    by_size = mutated(manifest, "rubberband", sha256=None, verify="size")
    assert problems(by_size) == []
    no_size = mutated(manifest, "rubberband", sha256=None, size=None, verify="size")
    assert any("verify 'size' needs a size" in p for p in problems(no_size))


def test_verify_none_needs_a_reason_and_no_hash(manifest: dict[str, Any]) -> None:
    assert problems(mutated(manifest, "rubberband", sha256=None, verify="none", unverified_reason="the publisher lists no hash")) == []
    bad = mutated(manifest, "rubberband", sha256=None, verify="none")
    assert any("needs a non-empty unverified_reason" in p for p in problems(bad))
    contradiction = mutated(manifest, "rubberband", verify="none", unverified_reason="x")
    assert any("contradicts a given sha256" in p for p in problems(contradiction))


def test_bad_field_values_are_reported(manifest: dict[str, Any]) -> None:
    cases = {
        "http url": (mutated(manifest, "ffmpeg", url="http://example.com/x.zip"), "url must start with https://"),
        "path in filename": (mutated(manifest, "ffmpeg", filename="../x.zip"), "filename must be a plain file name"),
        "bad sha": (mutated(manifest, "ffmpeg", sha256="abc"), "sha256 must be 64 lowercase hex"),
        "bad archive": (mutated(manifest, "ffmpeg", archive="rar"), "archive must be one of"),
        "bad size": (mutated(manifest, "ffmpeg", size=-5), "size must be a positive integer"),
        "bad kind": (mutated(manifest, "ffmpeg", kind="exe"), "kind must be one of"),
        "bad platform": (mutated(manifest, "ffmpeg", platforms=["amiga"]), "platforms must be"),
        "bad engine": (mutated(manifest, "ffmpeg", engines=["nope"]), "engines must be"),
        "unknown key": (mutated(manifest, "ffmpeg", colour="red"), "unknown keys"),
        "absolute provides": (mutated(manifest, "ffmpeg", provides={"ffmpeg": "C:/x/ffmpeg.exe"}), "relative posix path"),
        "dotdot provides": (mutated(manifest, "ffmpeg", provides={"ffmpeg": "../ffmpeg.exe"}), "relative posix path"),
        "unknown tool key": (mutated(manifest, "ffmpeg", provides={"hammer": "bin/hammer.exe"}), "not a tool in vndub.tools.TOOL_INFO"),
        "no provides": (mutated(manifest, "ffmpeg", provides={}), "provides must map"),
        "bad version": (mutated(manifest, "ffmpeg", version="1/2"), "version must be usable as a folder name"),
        "no version": (mutated(manifest, "ffmpeg", version=KeyError), "version must be a non-empty string"),
        "none-archive mismatch": (mutated(manifest, "xp3pack", filename="Other.exe"), "provides must equal filename"),
    }
    for label, (m, needle) in cases.items():
        assert any(needle in p for p in problems(m)), f"{label}: {problems(m)}"


def test_pip_entries_must_be_pinned_and_hashed(manifest: dict[str, Any]) -> None:
    unpinned = copy.deepcopy(manifest)
    entry(unpinned, "unrpa")["pip"]["requirement"] = "unrpa>=2.3"
    assert any("pinned as name==version" in p for p in problems(unpinned))
    nohash = copy.deepcopy(manifest)
    entry(nohash, "unrpa")["pip"]["hashes"] = []
    assert any("pip.hashes" in p for p in problems(nohash))
    badhash = copy.deepcopy(manifest)
    entry(badhash, "unrpa")["pip"]["hashes"] = ["md5:abc"]
    assert any("pip.hashes" in p for p in problems(badhash))
    with_url = copy.deepcopy(manifest)
    entry(with_url, "unrpa")["url"] = "https://example.com/x"
    assert any("no url" in p for p in problems(with_url))


def test_wrapper_rules(manifest: dict[str, Any]) -> None:
    bad = copy.deepcopy(manifest)
    entry(bad, "unrpyc")["needs_pytools"] = False
    assert any("a wrapper needs needs_pytools" in p for p in problems(bad))
    bad2 = copy.deepcopy(manifest)
    entry(bad2, "unrpyc")["wrapper"]["script"] = "../evil.py"
    assert any("wrapper.script must be a relative posix path" in p for p in problems(bad2))


def test_duplicates_and_structure(manifest: dict[str, Any]) -> None:
    dup = copy.deepcopy(manifest)
    dup["tools"].append(copy.deepcopy(entry(dup, "ffmpeg")))
    assert any("duplicate name" in p for p in problems(dup))
    assert problems({"schema": 2, "tools": manifest["tools"]})
    assert problems({"schema": 1, "tools": []})
    assert problems([]) == ["manifest must be a JSON object"]


def test_extra_files_are_validated(manifest: dict[str, Any]) -> None:
    bad = copy.deepcopy(manifest)
    entry(bad, "llama_server")["extra"][0]["sha256"] = None
    assert any("extra[0]" in p and "sha256 is null" in p for p in problems(bad))


def test_manual_entries_cannot_claim_a_hash(manifest: dict[str, Any]) -> None:
    bad = copy.deepcopy(manifest)
    e = entry(bad, "bgi_extractor")
    e["verify"], e["sha256"] = "sha256", "0" * 64
    assert any("manual entries cannot claim sha256" in p for p in problems(bad))


def test_load_manifest_raises_with_all_problems(tmp_path: Path, manifest: dict[str, Any]) -> None:
    bad = mutated(manifest, "ffmpeg", sha256="zz", archive="rar")
    f = tmp_path / "m.json"
    f.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(installer.ManifestError) as ei:
        installer.load_manifest(f)
    assert len(ei.value.errors) >= 2
    with pytest.raises(installer.ManifestError):
        installer.load_manifest(tmp_path / "missing.json")


def test_allow_http_only_for_tests(manifest: dict[str, Any]) -> None:
    m = mutated(manifest, "ffmpeg", url="http://127.0.0.1:1/x.zip")
    assert problems(m)
    assert problems(m, allow_http=True) == []
    assert any("plain http" in w for w in installer.manifest_warnings(m))
