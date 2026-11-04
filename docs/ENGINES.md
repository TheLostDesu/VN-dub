# Engines: extraction, parsing, packing

vndub never rebuilds game scripts. Scripts are parsed only to learn
**line -> speaker -> voice file**; the dub is written back under the **same file
name, codec, sample rate and channel count**, as loose files or a patch archive
that the engine prefers over its original archives.

Stages involved: `detect` (engine + `work/project.yaml`), `extract`
(`work/raw/`, decoded `work/audio/src|src16/`, `work/audio_index.jsonl`),
`parse` (`work/manifest/lines.jsonl`, `characters.yaml`,
`unbound_audio.jsonl`), `pack` (`out/<engine>/`, `INSTALL.md`,
`manifest_patch.json`). Engine plugins live in `vndub/engines/<name>.py`.

`voice_file` in `lines.jsonl` is the path as the game addresses it, relative to
the engine's voice root (Ren'Py: `game/`; KiriKiri: archive root; NScripter:
the exe dir; Tyrano: `data/`). Script references are matched to extracted files
case-insensitively, extension-less and with `\` or `/` separators
(`VoiceResolver`). References that resolve to nothing are kept and flagged
`voice_missing`.

## Overview

| Engine | Detected by | Extraction | Voice mapping | Pack | Status |
|---|---|---|---|---|---|
| Ren'Py | `game/*.rpy`, `*.rpyc`, `*.rpa` (+ `renpy/`) | loose; `unrpa`; `unrpyc` for `.rpyc` | `voice "f"`, `voice sustain`, `config.auto_voice` ids, `dialogue.tab` | loose `out/renpy/game/...` + voice-track and text-patch scripts | complete |
| KiriKiri 2/Z | `*.xp3`, loose `.ks`, `.scn` | built-in XP3 reader (unencrypted); dump folders; FreeMote for `.scn` | KAG macros + conventions; `.scn` JSON | `patchN.xp3` (built-in writer or Xp3Pack), or `unencrypted/` | complete |
| NScripter / ONScripter | `nscript.dat`, `0.txt`, `arc*.nsa`, `*.sar`, `*.ns2` | built-in XOR 0x84 decoder, NSA/SAR/NS2 reader | `dwave CH,"file"` -> next text line | loose beside the exe | complete |
| Siglus | `Scene.pck` + `Gameexe.dat`, `koe_master.csv` | `siglus-ssu -k` / `-x` | `koe_master.csv` | dump layout + instructions (`.ovk` repack not available) | parser complete, pack manual |
| BGI / Ethornell | `*.arc` + `._bp` / `BGI.exe`, `bgi_json/` | `bgi-extractor -t -f json` (bundled with the app) | JSON speaker/text/voice id | loose + instructions | parser complete (tested with the bundled exe on a synthetic archive), pack manual |
| CatSystem2 | `*.int` (KIF) + `*.cst` | TriggersTools CLI / `exkifint` | decompiled CST (0x21 name, 0x20 text, 0x30 `pcm`) | loose + CatSceneEditor instructions | parser defensive (voice command UNVERIFIED) |
| Artemis | `*.pfs` | `pfs-rs` / `pfs_unpacker` | Lua `.ast` (`text.ja`, `name`, `{"vo", file=}`) | `files/` + pfs pack command | parser complete (voice key layout UNVERIFIED) |
| YU-RIS | `*.ypf` | `yrt` | TODO `.yst` parser -> use VNTextPatch import | instructions | thin wrapper |
| Majiro | `*.mjo` + `*.arc` | MajiroTools | TODO `.mjil` parser -> use VNTextPatch import | instructions | thin wrapper |
| Unity / Naninovel | `*_Data/globalgamemanagers`, `*.nani` | UnityPy (AudioClip export) or AssetRipper `--headless` | `@voice Path`, auto-voice by `\|#textId\|` | `out/unity/replace_audioclips.py` + checklist | EXPERIMENTAL |
| TyranoScript | `data/scenario/*.ks` | none (plain files) | `[voconfig vostorage="x_{number}.ogg"]` + `#name` auto-increment | loose `out/tyrano/data/...` | complete |
| generic | fallback | loose voice files | VNTextPatch JSON import + `<prefix>_<index>` assignment | loose | complete (flags `prefix_guess`) |

## External tools

None are vendored. Configure paths in `configs/tools.yaml` (empty = search
`PATH`; per-machine overrides in `configs/tools.local.yaml`), then check with
`uv run vndub tools check [--engine NAME]`.

| tools.yaml key | Executable | Used for | Download |
|---|---|---|---|
| `ffmpeg` | ffmpeg | decode/encode audio, atempo stretch fallback (**required**) | https://www.gyan.dev/ffmpeg/builds/ (`winget install Gyan.FFmpeg`) |
| `ffprobe` | ffprobe | probe codec/sr/channels (**required**) | ships with ffmpeg |
| `vgmstream_cli` | vgmstream-cli | proprietary game audio (hca, at9, fsb, ovk, wem ...) | https://github.com/vgmstream/vgmstream/releases |
| `rubberband` | rubberband | time stretch in `master` | https://breakfastquay.com/rubberband/ |
| `unrpa` | unrpa | Ren'Py `.rpa` | https://github.com/Lattyware/unrpa (`pip install unrpa`, or `uv sync --extra extract`) |
| `unrpyc` | unrpyc | Ren'Py `.rpyc` -> `.rpy` (v2.x) | https://github.com/CensoredUsername/unrpyc/releases |
| `xp3pack` | Xp3Pack | KiriKiri patch archives (optional; built-in writer otherwise) | https://github.com/arcusmaximus/KirikiriTools/releases |
| `krkrdump` | KrkrDumpLoader | KiriKiri Z runtime dump of encrypted archives | https://github.com/crskycode/KrkrDump/releases |
| `freemote_psbdecompile` | PsbDecompile | KiriKiri Z `.scn` -> JSON | https://github.com/UlyssesWu/FreeMote/releases |
| `siglus_ssu` | siglus-ssu | Siglus `Scene.pck`, `.ovk` -> `KOE(...).ogg` + `koe_master.csv` | https://github.com/Jirehlov/SiglusSceneScriptUtility |
| `bgi_extractor` | bgi-extractor | BGI archives -> speaker/text/voice JSON | https://github.com/mirinnano/bgi-extractor |
| `catsystem2_cli` | TriggersTools.CatSystem2 | CatSystem2 KIF + CST | https://github.com/trigger-segfault/TriggersTools.CatSystem2 |
| `exkifint` | exkifint | CatSystem2 `.int` (v2) | asmodean tools (legacy) |
| `pfs_unpacker` | pfs-rs | Artemis `.pfs` | https://github.com/sakarie9/pfs-rs |
| `yrt` | yrt | YU-RIS YPF / ybn | https://github.com/exsatsukirin/YU-RIS-TOOLS |
| `majiro_tools` | MajiroTools | Majiro `.mjo` / `.arc` | https://github.com/AtomCrafty/MajiroTools |
| `assetripper` | AssetRipper | Unity export (`--headless`) | https://github.com/AssetRipper/AssetRipper/releases |
| `garbro_console` | gameres | GARbro console fallback extractor | https://github.com/crskycode/GARbro |
| `arc_unpacker` | arc_unpacker | legacy universal extractor | https://github.com/vn-tools/arc_unpacker |
| `vntextpatch` | VNTextPatch | text + names for 22 engines (`extractlocal`) | https://github.com/arcusmaximus/VNTranslationTools/releases |

(Model-side tools - `llama_server`, `applio_dir`, `audiocpp`, `zonos2_cli`,
`moss_tts_cli` - are described in [MODELS.md](MODELS.md).)

## Ren'Py

**Extraction.** Loose files in `game/` are used directly. `*.rpa` archives are
extracted with `unrpa -mp work/raw/renpy/rpa <archive>` when configured
(otherwise a warning). `.rpyc` files without a matching `.rpy` are decompiled
with `unrpyc` (v2.x supports Ren'Py 7/8). Voice files are taken from the voice
dirs (`extract.voice_dirs` + `voices`, `audio/voice`); without them every audio
file outside `bgm/`, `music/`, `se/`, `sfx/`, `sound/` ... is considered.

**Parsing** (`vndub/engines/renpy.py`): `define x = Character("Name", voice_tag="tag")`
(also `_("Name")`, `Character(None)`, `kind=nvl`, `DynamicCharacter`);
speaker id = `voice_tag` when set, else a slug of the name. Labels give the
scene (local labels `.x` -> `global_x`). Say statements: `who [attrs] [@ tmp]
"what" [nointeract] [id x] [(args)]`, narrator `"what"`, `"Name" "what"`,
`extend`. Text tags (`{w}`, `{i}` ...) are removed from `text_src`; the raw
statement is kept in `raw`. Menus become a record with `choices` and the
`menu` flag. `init python`, `python`, `screen`, `transform`, `style`,
`translate` and ATL `image` blocks are skipped (but `config.auto_voice = ...`
is read from `init python` / `define config.auto_voice`).

* `voice "file"` binds to the next say statement.
* `voice sustain`: the previous voice keeps playing during the next say. That
  line gets `has_voice: false`, `voice_file: null`, flag `voice_sustain` and
  `sustain_of: <head line_id>`; the line owning the file gets flag
  `sustain_head` and `sustain_lines: [...]`. Downstream stages concatenate the
  texts for the head line (the file contains both sentences).
* `config.auto_voice = "voice/{id}.ogg"`: `{id}` is the dialogue identifier,
  computed as in `renpy/translation/__init__.py`: md5 over each statement of
  the translate block as `get_code() + "\r\n"` (a `voice` statement is
  translatable and belongs to the block of the following say), first 8 hex
  digits, prefixed with the current translation label (dots -> `_`; hidden
  labels and `_`-labels do not change it); collisions get `_1`, `_2`.
  Checked against Ren'Py master (`translation/__init__.py`, `parser.py`, `lexer.py`, `00voice.rpy`):
  every script file is restructured on its own (label and `_n` suffixes start over per file);
  `label`, `menu name:` and `call x from name` each make a Label node, so a name without a leading `_` moves
  the translation label for the says after it (`menu name:` also declares the global label that `.local` labels hang on;
  `call ... from _x` only sets an alternate label and leaves the identifier alone); a menu's own say
  (`e "Which way?"` before the choices) is a `nointeract` say, a bare string in a menu is a caption (no id);
  a triple-quoted string is cut at blank lines and each paragraph is its own say (own click, own id, own line number);
  `with` clauses are part of the code. At run time Ren'Py asks for
  `config.auto_voice.format(id=<id>)` at every interaction (`voice_interact`, skipped when `config.has_voice` is
  False: the generated `vndub_auto_voice.rpy` sets it True at `init 1000`).
  `Say.get_code()` = `who [attributes] [@ temporary] "encoded what" [nointeract]
  [id x] [arguments] [with expr]`, where the string is re-encoded by `encode_say_string`
  (escapes `\`, newline, `"`, and a space after a space) after Ren'Py's lexer
  processing (whitespace runs collapsed, escapes resolved).
  **Verified:** the Ren'Py documentation example under `label start`,
  `e "Thank you for taking a look at the Ren'Py translation framework."`,
  gives `start_636ae3f5` (test `tests/test_renpy_ids.py`). The translatable block that is hashed with a say is
  exactly what Ren'Py's `Restructurer` groups: the `voice` / `voice sustain` statements and `nvl clear` right before
  it, in script order, flushed by any other statement (1.0.0: before, one pending voice survived a `show` in between,
  two voice statements kept only the last, and `nvl clear` was skipped). Ren'Py 8.5.1 added
  `config.tlid_only_considers_say` (default True from 8.6): with it, voice statements no longer change ids. `parse`
  reads the game's Ren'Py version (`renpy/vc_version.py` or `renpy/__init__.py` next to the game folder) and uses that
  default (a parse note says which); `parse.engine_opts.renpy.tlid_only_considers_say: true|false` overrides it, and
  `dialogue.tab` beats both. Since 1.0.0 character and image names may be any Ren'Py word (Cyrillic, CJK), and
  `Character(kind=e)` inherits its name and voice tag from `e` (only an explicit `None` name is narration).
* **Scripts**: `game/**/*.rpy`, every `.rpa` archive (unrpa) and every `.rpyc` without its `.rpy` (unrpyc, `--clobber`,
  the decompiled tree mirrors the game's folders so two `script.rpyc` in different chapters do not collide). Only the
  engine's own `renpy/` folder is skipped (1.0.0: the filter used to drop everything extracted from archives, because
  the extraction folder is itself called `renpy/`).
* **Fallback when ids cannot be reproduced** (unknown constructs, newer
  Ren'Py rules): an unclaimed file under the auto_voice directory is bound to the one unvoiced line of its label only
  when that label has exactly one of each (flag `auto_voice_order`; `link` verifies it by ASR). With more, nothing
  is guessed (the files are named `<label>_<hash>`, so their order says nothing) and a parse note names the label.
* **Lines with a `voice` statement of their own** whose file is missing or computed at run time are NOT dubbed through
  `config.auto_voice` (Ren'Py never asks it for such a line); they are counted as `own_voice` in the patch coverage
  and listed in INSTALL.md.
* **Stale files**: a full `pack` removes from `out/renpy` the files the previous `manifest_patch.json` listed and the
  voice files of its own folders that the new patch does not contain (a line rejected in review, a re-import that
  renamed ids), nothing else; `--only` packs never delete.
* `parse.engine_opts.renpy.dialogue_tab`: the launcher's *Extract Dialogue*
  TSV (`Identifier, Character, Dialogue, Filename, Line Number`); its ids
  override computed ones by (file, line) (flag `dialogue_tab_id`).

**Packing.** `out/renpy/game/<voice_file>` loose files: Ren'Py checks the
filesystem before archives. If the game sets `config.force_archives = True`,
repack into an `.rpa` that sorts after the originals (`rpatool -c`); noted in
INSTALL.md.

**The voice track** (1.0.3). The dub plays on Ren'Py's own `voice` channel, which has a mixer of its own (never the
music mixer), but Ren'Py binds that mixer at `init 1600` only if `config.has_voice` is True at that moment. A game that
never had voice usually switches it off in `options.rpy`: the preferences have no voice slider and the dub is drowned by the music. So `pack`
writes a second generated script, `game/vndub_voice_mix.rpy`, next to `vndub_auto_voice.rpy`, at `init 1000` (after the
game's `options.rpy`, long before 1600), which:

* sets `config.has_voice = True`, and `config.sample_voice` to one of the patch's files when the game has none, so the
  standard preferences screen shows a "Voice Volume" slider with a Test button;
* turns on Ren'Py's ducking: `config.emphasize_audio_channels = ["voice"]`, `emphasize_audio_time = 0.3`, and the
  player preference `emphasize_audio` (Ren'Py leaves it OFF by default; it is switched on once), so the music and the
  sound effects drop to `duck` (default 0.3 = 30 %, about -10 dB) while a line plays;
* adds an in-game panel (screen `vndub_voice_panel`): voice volume, "music while a line plays" (stored in
  `persistent.vndub_duck`), "lower the music while a line plays", "mute the dub", "test the voice" and "close". Texts are
  Russian when the dub's target language is Russian, else English; the font is DejaVuSans, so Cyrillic always renders;
* opens the panel with a hotkey (`K_F8`, bound through `config.underlay`: it works in the game and in the menus) and
  with a button "Озвучка" / "Dub" in the top right corner of the game's preferences screen (a
  `config.always_shown_screens` screen on Ren'Py 7.5 / 8.0+, an interact callback on older versions; shown only while the
  `preferences` screen is up).

Config `pack.renpy.voice_mix`: `enabled: auto | true | false` (auto = on when the patch holds lines that had no
original voice, i.e. the game was never mixed for voice; a voiced game keeps its own mix unless `true`; a bare
`voice_mix: true|false` is read as `enabled`), `duck: 0.3` (music / sfx volume while a line plays, 1.0 = no ducking),
`key: K_F8`. `manifest_patch.json` carries `voice_mix` (`{rpy, key, keysym, duck, enabled}` or `null`), INSTALL.md the
section "The voice track in the game", the pack summary `voice_mix: true|false`. A full `pack` removes a
`vndub_voice_mix.rpy` it generated earlier when the feature is now off (the same rule as the stale auto_voice script).
To undo it in a game by hand, delete `game/vndub_voice_mix.rpy` and `game/vndub_voice_mix.rpyc`.
**Not yet run in a real Ren'Py** (no SDK on the build machine): the script follows the Ren'Py source and docs; open the
game once, press F8 and open the preferences screen.

**The translated text** (1.1.0). Until 1.0.3 the patch carried only voice: the dub spoke Russian over the game's English
text. `pack` now also writes `game/tl/<language>/vndub_dialogue.rpy`, one `translate <language> <dialogue id>:` block per
translated say line (the shape `renpy translate` writes; the identifier is the same one the auto-voice files are named by,
a menu's own say keeps its `nointeract`, a `voice` statement of the original group is repeated), plus a
`translate <language> strings:` block for menu choices (`choices_tgt` of the menu's translation record) and menu captions,
and `game/vndub_language.rpy` (`define config.language = "<language>"`: Ren'Py's `_init_language` reads
`RENPY_LANGUAGE or config.language or _preferences.language`, so the game starts in the dub's language every time). The
language name is the conventional lowercase one (`ru` -> `russian`, `ja` -> `japanese`, ...; `pack.renpy.text.language`
overrides). Ren'Py refuses duplicate identifiers and duplicate `old` strings, so ids and strings the game already
translates under `game/tl/<language>/` are skipped and listed, as are lines whose translation lost an interpolation or flow
tag (`[name]`, `{w}`), menus without translated choices, and lines with their own `voice` expression. Config
`pack.renpy.text: {enabled: auto | true | false, language: "", font: ""}` (auto = the project has a translation file;
`font` = a .ttf/.otf inside `game/` set through a `translate <language> python:` block for games whose font lacks
Cyrillic; Ren'Py's bundled DejaVuSans has it). `manifest_patch.json` carries `text` (`{rpy, language_rpy, language, lines,
strings, skipped, skipped_count, skipped_reasons, font, enabled}` or `null`), INSTALL.md the section "The translated text",
the desktop app's install step a note. A full `pack` removes the two scripts when the feature is off. To undo it in a game
by hand, delete `game/vndub_language.rpy`, `game/tl/<language>/vndub_dialogue.rpy` and their `.rpyc` twins. Narration and
menu choices are translated for this patch as text-only lines (`translate.include_narration`, auto = on for Ren'Py); they
never reach synth. **Not yet run in a real Ren'Py**: the first real run is the check (the text is in the dub's language,
a menu too, the game reports no "duplicate" error).

## KiriKiri 2 / KiriKiri Z

**Extraction.** Load order (later overrides earlier): the game folder (loose
files), dump folders (`unencrypted/` from KirikiriTools `version.dll` +
`extract-unencrypted.txt`, `KrkrDump/` output, `extract/`), `data.xp3`, other
archives, `patch.xp3`, `patch2.xp3` ... Plain archives are read by the
built-in reader (`vndub/engines/xp3.py`: version-2 header with the 0x17
"cushion", old-style headers, raw/zlib index, multi-segment zlib members,
Adler-32 check) into `work/raw/xp3/<archive>/`. Encryption is detected
heuristically (index does not decompress, or sampled `.ogg/.wav/.ks` members do
not look like their format) and reported: dump such games at runtime with
**KrkrDump** (KiriKiri Z) or **KirikiriTools `version.dll`**, then point
`--project` at the dump (or keep the dump folder inside the game dir). `.scn`
(PSB) scenes are converted with FreeMote `PsbDecompile` to JSON.

**Parsing KAG `.ks`** (two phases, `vndub/engines/kirikiri.py`):

1. *Macro learning* over all `.ks`: inside `[macro name=X] ... [endmacro]`, a
   voice tag (`playse`, `voice`, `playvoice`, ...) whose `storage` is a macro
   parameter (`%s`, `%s|default`, `&mp.s`) gives the voice attribute; `%t` /
   `mp.t` inside `[ch text=...]`, `[emb exp=...]`, `[eval exp=...]` gives the
   name attribute; a body printing a fixed `【名前】` makes a fixed-name macro.
   `engine_opts.kag.macro_names` are always treated as name macros and
   `name_attr` / `voice_attr` override learning.
2. *Streaming*: per scenario file, "current speaker" and "pending voice" are
   kept; a line is emitted when text ends at `[l]`, `[p]`, `[s]`, `[cm]` or a
   blank line. `[r]` is a line break inside the message (set
   `engine_opts.kag.break_on_r: true` to split there). Speaker sources: name
   macros (`[nm t="寧々" s=nen_0001]`), `[ns]名前[nse]`, inline `【名前】`, `#名前`
   lines. `[playse storage=X]` outside a voice macro counts as a voice only when
   X is in a voice folder or has a `<prefix>_<index>` name shared with other
   voices (otherwise it is a sound effect). `*label` lines are stored as
   `label`; scene = file stem. Encoding: BOM / UTF-8 / CP932 sniffing
   (`engine_opts.kag.encoding` forces one).

**Parsing `.scn` JSON**: `scenes[].texts[]` entries `[name, display_name, text, ...]`;
voices = any nested dict with a `voice` key (string or `{name: ...}`);
language-nested text variants are handled defensively. Speaker/voice positions
differ per title (UNVERIFIED in general).

**Packing.** `out/kirikiri/patchN.xp3` where N is the first unused number next
to the game exe (`patch.xp3`, then `patch2.xp3` ...). Entries are flat at the
archive root (KiriKiri resolves storages by name through auto-paths);
`pack.kirikiri.subdir` keeps a folder and INSTALL.md shows the
`Storages.addAutoPath(System.exePath + "patchN.xp3>voice/")` line. Built with
Xp3Pack when configured, else the built-in writer (unencrypted, version 2).
Encrypted games usually reject unencrypted patches: install KirikiriTools
`version.dll` (accepts unencrypted archives) or use
`pack.kirikiri.mode: unencrypted_dir` (`out/kirikiri/unencrypted/`, read by
`version.dll`).

## NScripter / ONScripter

**Extraction.** `nscript.dat` / `pscript.dat` (every byte XOR 0x84, Shift-JIS)
or plain `0.txt`..`99.txt`, decoded to `work/raw/nscripter/script.txt`.
Archives `arc.nsa`, `arc1.nsa`..`arc9.nsa`, `*.sar`, `*.ns2` share one
namespace (`vndub/engines/nsa.py`: SAR, NSA with stored/LZSS/NBZ members, NS2;
SPB bitmaps are skipped - images are never voices). The first archive holding
a name wins; loose files beside the exe override archive members.

**Parsing.** `dwave CH,"path"` (also `dwaveload`, `dwaveplay`, `dwaveloop`)
binds to the next text line; `:`-separated commands are split outside quotes;
`;` comments stripped. Text lines start with a non-ASCII char or `` ` ``/`^`;
`@`, `\`, `/`, `!wNNN`, `!dNNN`, `!sNNN`, `#RRGGBB`, `$var`, `%var` are removed.
Speakers: `engine_opts.nscripter.speaker_regex` (default `^【(?P<name>[^】]+)】`),
then the `名前「……」` convention (name of at most 8 chars directly before the
opening bracket, closing bracket at the end). `*label` lines give the scene.

**Packing.** Loose files in the same relative paths beside the exe.

## Siglus

`siglus-ssu -k Scene.pck <koe dir> <out>` dumps every voice as
`KOE(#########).ogg` in per-character folders plus `koe_master.csv`
(`koe_no, character, text, duration, callsite`); `siglus-ssu -x` dumps the
scripts. A `koe_master.csv` already present in the game dir or
`work/raw/siglus` is used directly. The CSV is the mapping: one line per KOE,
scene from the callsite (`s01:12` -> `s01`) or `koe_no // 100000` (global id =
scene x 100000 + entry). Packing: siglus-ssu does not rebuild `.ovk`; vndub
writes the dump layout (`out/siglus/koe_dump/`) and INSTALL.md lists the options
(SiglusTools, UNVERIFIED for ovk).

## BGI / Ethornell

`bgi-extractor -t -f json --no-archive --no-single-file --no-presets -o <out>
<archive>` for every `*.arc` (the packaged app ships its own build of the tool:
upstream publishes source only). That writes one
`01_scripts/<archive>/<script>.json` per script, a list of `{"type":
"line"|"choice", "speaker"?, "text", "voice"?}`; without the `--no-*` switches
the same records are also merged per archive and for the whole game, and every
line would be read three times. Choices are skipped; the JSON is read with
`strict=False` (the tool writes tabs into strings unescaped). For JSON from
other tools the parser also accepts `{lines|messages|entries|texts: [...]}` with
speaker in `speaker|name|character|chara`, text in `text|line|message|msg`,
voice in `voice|voice_id|vo|voice_file`, and voice ids embedded as
`(yuki_000003)` at the start of the text. Already-exported JSON in `bgi_json/`
(or anywhere in the game dir) is used without the tool. Repacking is
weak (BGIKit/EthornellTools archived): loose files + instructions.

## CatSystem2

`.int` (KIF) archives via TriggersTools.CatSystem2 or `exkifint` (v2; v3 not);
KIF archives containing `__key__.dat` are flagged encrypted. Decompiled CST
text is parsed by type code: `0x21` speaker, `0x20` message, `0x30` command
(voice command `pcm <name>` - UNVERIFIED); accepted forms `0x21 name`, `[21] name`,
`21: name` or JSON `[{type, content}]`. Voices live in `pcm_<x>.int`; rebuild
with CatSceneEditor (unencrypted only).

## Artemis

`pfs-rs unpack <root.pfs> <out>` (or pfs_unpacker). `.ast` files are Lua
tables read by a small built-in Lua-table reader; per `block_*`: text under
`text.<src_lang>` (`ja` first, tag tables like `{"rt2"}` skipped), speaker from
a `name` entry, voice from any `{"vo"|"voice", file=...}` entry or `vo`/`voice`
key (layout UNVERIFIED across games). Compiled `.asb`/`.iet` scripts are not
supported (use VNTextPatch). Pack: `out/artemis/files/` + the pfs pack command
(name the patch so it loads after the original, e.g. `root.pfs.001`).

## YU-RIS and Majiro (thin wrappers)

Archive extraction through `yrt` / MajiroTools. Voice-call parsers for
`.yst` / `.mjil` are TODO (not verified on real games). Use VNTextPatch for text:

```
VNTextPatch extractlocal <game> <json dir>
vndub parse --project <game> --import-vntextpatch <json dir>
```

then `vndub link` verifies the prefix-based voice assignment by ASR.
Repacking: YPF_Tool / ypf-repacker need the archive key; Majiro searches
archives by priority, so a patch `.arc` overrides `voice.arc`.

## Unity / Naninovel (EXPERIMENTAL)

Extraction: UnityPy (`uv sync --extra extract`) exports AudioClips from
`*_Data` to `work/raw/unity/audio/` and records their containers in
`clips.json`; otherwise AssetRipper `--headless`. Naninovel `.nani` scripts:
`Author: text` lines, narration lines, `@voice Path/To/Clip` binds to the next
line, `# label`; auto-voice by text id (`|#2670eb4|` ->
`Naninovel/Voice/<Script>/<textId>`) when such a clip exists. Unity has no
loose-file override: `out/unity/` contains the clips, `clips.json` and
`replace_audioclips.py` (UnityPy, writes `*.patched` asset files) plus a
checklist. Clip metadata mismatches can crash the game; FSB5 re-encoding (UABEA)
may be needed.

## TyranoScript

No archive (`data/scenario/*.ks`, `data/sound`, `data/voice`; also
`resources/app/` for NW.js builds; packed `app.asar` must be unpacked first).
`[voconfig name="akane" vostorage="akane_{number}.ogg" number=N]` + `[vostart]`:
each message of `#akane` (also `#akane:face`) auto-plays the next number;
`[vostop]` pauses it; counters persist across files (`first.ks` first).
`[chara_new name= jname=]` gives display names (speaker id = the ASCII key).
Explicit `[playse storage=...voice...]` before a message binds as well.
Pack: loose `out/tyrano/data/...`.

## Generic fallback

For any engine without a voice-aware parser: text + names from a VNTextPatch
JSON export (`[{"name": ..., "message": ...}]` per script file) and voices
grouped by `engine_opts.generic.prefix_regex` (default
`^(?P<prefix>[A-Za-z]+?)_?(?P<index>\d+)$`), sorted by index and assigned to the
speaker's lines in script order. Speaker -> prefix comes from
`engine_opts.generic.prefix_map`, then `characters.yaml` `voice_prefix`, then
name similarity, then by matching ranks of line counts and file counts (only
when the counts are within 30 %). Every binding is flagged `prefix_guess`.
GARbro console (`gameres -x`) or `arc_unpacker` can be used manually to unpack
unknown archives into the game folder first.

## Audio

`extract` decodes every voice file with ffmpeg (`vgmstream-cli` for
`extract.vgmstream_exts`) to `work/audio/src/<id>.wav` (original rate, mono),
`work/audio/src_stereo/<id>.wav` (stereo originals, when `keep_stereo_copy`)
and `work/audio/src16/<id>.wav` (16 kHz mono). Audio id = file stem,
disambiguated with parent folders on collisions. Items are cached by content
hash (re-running decodes only new/changed files); decode failures are recorded
per file in `audio_index.jsonl` (`error`). `project.yaml: voice_codec` is
probed from a loose voice by `detect`, or taken from the most common codec in
the audio index when voices were archived. `master` re-encodes to the
original codec/rate/channels and file name.

## Fixtures

`tests/fixtures/` holds synthetic mini games for every complete parser; they are
generated by `uv run python scripts/make_fixtures.py [renpy kag nscripter siglus bgi tyrano]`.
`renpy_mini` is built from `scripts/fixture_data.py` (3 voiced characters plus an
unvoiced protagonist `hero` with narration cues around his lines, a menu,
`voice` statements, one `voice sustain`; a mixed game, see `docs/PIPELINE.md`); its Japanese voices come from the
Windows SAPI ja-JP voice when available (pitch/rate per character, delivery per
line), else from synthetic speech-like signals (`--synthetic`); existing voice files
are kept on regeneration (`--rerender-voices` renders them again). Golden parser
outputs are in `tests/fixtures/golden/`; refresh them after regenerating with
`VNDUB_UPDATE_GOLDEN=1 uv run pytest tests/test_engine_goldens.py tests/test_engine_kag.py`.
