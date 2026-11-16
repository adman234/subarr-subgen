"""Runtime smoke for the per-file skip reason contract (patch 0065, coaxk/subarr#669).

Run INSIDE the built image (the patched tree needs torch, faster_whisper, av...),
with the PATCHED upstream tree mounted:

    docker run --rm -v "$PWD/upstream:/work:ro" -v "$PWD/tests:/smoke:ro" \
      -w /work -e PYTHONPATH=/work --entrypoint python3 \
      ghcr.io/coaxk/subarr-subgen:<tag> /smoke/smoke_skip_reasons.py

WHY THIS EXISTS: /batch used to return one `skipped` count, so subarr could not
tell "an English subtitle is embedded in the file" (benign) from "the audio is
a language this subgen skips" (worth a look). Every skip landed in subarr's
Issues bucket as "reason not in /batch response". Patch 0065 records which
should_skip_file branch fired and /batch reports `skip_reasons`.

Each branch is driven by setting the module globals that gate it and stubbing
the filesystem/ffprobe helpers, so no media file is needed. Every override is
restored after each case, so cases cannot leak into each other.

Not wired into CI (like the other smokes): it needs the built image.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.environ.get("SUBGEN_TREE", "/work"))

import subgen  # noqa: E402
from subgen import LanguageCode  # noqa: E402

FAILS = []
EN = LanguageCode.from_string("en")
FR = LanguageCode.from_string("fr")
NONE = LanguageCode.NONE


def check(name, cond, detail=""):
    print(
        f"{'PASS' if cond else '*** FAIL ***'}  {name}{('  -> ' + detail) if detail else ''}"
    )
    if not cond:
        FAILS.append(name)


# Every gate should_skip_file reads, set to "nothing fires". A case overrides
# only what its branch needs.
NEUTRAL = {
    "transcribe_or_translate": "transcribe",
    "lrc_for_audio_files": False,
    "skip_unknown_language": False,
    "skip_if_no_audio_language_but_subtitles_exist": False,
    "limit_to_preferred_audio_languages": False,
    "preferred_audio_languages": [],
    "skip_audio_languages": [],
    "skip_if_target_subtitle_exists": False,
    "subtitle_language_name": "",
    "skip_if_internal_sub_language": NONE,
    "skip_subtitle_languages": [],
    "skip_if_external_sub_exists": False,
    "only_match_subgen_subtitles": False,
    "has_internal_subtitle_in_language": lambda *a, **kw: False,
    "has_external_subtitle_in_language": lambda *a, **kw: False,
    "get_subtitle_languages": lambda *a, **kw: [],
    "name_subtitle": lambda *a, **kw: "/nonexistent/output.en.srt",
}


def run_case(overrides, file_path="/nonexistent/show/ep.mkv", target=EN, audio=None):
    saved = {k: getattr(subgen, k) for k in {**NEUTRAL, **overrides}}
    try:
        for k, v in {**NEUTRAL, **overrides}.items():
            setattr(subgen, k, v)
        skipped = subgen.should_skip_file(
            file_path, target, audio_langs=audio if audio is not None else [FR]
        )
        return skipped, subgen.current_skip_reason()
    finally:
        for k, v in saved.items():
            setattr(subgen, k, v)


tmp = tempfile.mkdtemp()
song = os.path.join(tmp, "song.mp3")
open(song, "w").close()
open(os.path.join(tmp, "song.lrc"), "w").close()
existing_output = os.path.join(tmp, "ep.subgen.en.srt")
open(existing_output, "w").close()

yes = lambda *a, **kw: True  # noqa: E731

CASES = [
    # (name, overrides, kwargs for run_case, expected reason)
    (
        "1 lrc beside an audio file",
        {"lrc_for_audio_files": True},
        {"file_path": song},
        "lrc_exists",
    ),
    (
        "2a unknown audio + SKIP_UNKNOWN_LANGUAGE",
        {"skip_unknown_language": True},
        {"target": NONE},
        "unknown_language",
    ),
    (
        "2b unknown audio + any internal sub",
        {
            "skip_if_no_audio_language_but_subtitles_exist": True,
            "get_subtitle_languages": lambda *a, **kw: [FR],
        },
        {"target": NONE},
        "unknown_language_has_subs",
    ),
    (
        "3a no preferred audio track",
        {"limit_to_preferred_audio_languages": True, "preferred_audio_languages": [EN]},
        {"audio": [FR]},
        "no_preferred_audio",
    ),
    (
        "3b skipped audio language",
        {"skip_audio_languages": [FR]},
        {"audio": [FR]},
        "audio_lang",
    ),
    # The motivating case: the target-language sub is EMBEDDED, nothing on disk.
    (
        "4 target sub embedded only",
        {
            "skip_if_target_subtitle_exists": True,
            "has_internal_subtitle_in_language": yes,
        },
        {},
        "internal_sub",
    ),
    (
        "4 target sub as a sidecar only",
        {
            "skip_if_target_subtitle_exists": True,
            "has_external_subtitle_in_language": yes,
        },
        {},
        "external_sub",
    ),
    # Both present: embedded wins, because it is the one the user cannot see
    # by listing the folder (the exact confusion behind #669).
    (
        "4 target sub embedded AND sidecar",
        {
            "skip_if_target_subtitle_exists": True,
            "has_internal_subtitle_in_language": yes,
            "has_external_subtitle_in_language": yes,
        },
        {},
        "internal_sub",
    ),
    (
        "4b custom SUBTITLE_LANGUAGE_NAME sidecar exists",
        {
            "skip_if_target_subtitle_exists": True,
            "subtitle_language_name": "en",
            "has_external_subtitle_in_language": yes,
        },
        {"target": NONE},
        "named_sub",
    ),
    (
        "4c subgen's own output already exists",
        {
            "skip_if_target_subtitle_exists": True,
            "name_subtitle": lambda *a, **kw: existing_output,
        },
        {},
        "output_exists",
    ),
    (
        "5a SKIP_IF_INTERNAL_SUBTITLES_LANGUAGE matched",
        {"skip_if_internal_sub_language": EN, "has_internal_subtitle_in_language": yes},
        {},
        "internal_sub_configured",
    ),
    (
        "5b skipped subtitle language embedded",
        {
            "skip_subtitle_languages": [FR],
            "get_subtitle_languages": lambda *a, **kw: [FR],
        },
        {},
        "skip_sub_language",
    ),
    (
        "6 SKIP_IF_EXTERNAL_SUBTITLES_EXIST named sidecar",
        {
            "skip_if_external_sub_exists": True,
            "subtitle_language_name": "en",
            "has_external_subtitle_in_language": yes,
        },
        {},
        "named_sub",
    ),
]

check(
    "SKIP_REASONS vocabulary is exported",
    isinstance(getattr(subgen, "SKIP_REASONS", None), frozenset),
)
vocab = getattr(subgen, "SKIP_REASONS", frozenset())

for name, overrides, kwargs, expected in CASES:
    try:
        skipped, reason = run_case(overrides, **kwargs)
    except Exception as e:  # noqa: BLE001
        check(f"branch {name}", False, f"raised {type(e).__name__}: {e}")
        continue
    check(f"branch {name}: skipped", skipped is True)
    check(f"branch {name}: reason {expected}", reason == expected, repr(reason))
    check(f"branch {name}: reason in SKIP_REASONS", expected in vocab)

# A file that is NOT skipped must clear a reason left over from the previous
# call, or the next tally would blame the wrong rule.
try:
    run_case({"skip_audio_languages": [FR]}, audio=[FR])
    skipped, reason = run_case({})
    check("not skipped returns False", skipped is False)
    check("not skipped clears the previous reason", reason is None, repr(reason))
    skipped = subgen.should_skip_file(
        "/nonexistent/x.mkv", EN, audio_langs=[EN], bypass_skip=True
    )
    check(
        "bypass_skip clears the reason too",
        subgen.current_skip_reason() is None,
        repr(subgen.current_skip_reason()),
    )
except Exception as e:  # noqa: BLE001
    check("not-skipped reset", False, f"raised {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# /batch tally: transcribe_existing reports skip_reasons beside skipped.
# ---------------------------------------------------------------------------


KEEP = object()


def batch(results):
    """Walk a temp dir of len(results) files; each fake gen_subtitles_queue call
    plays one (return value, reason) pair."""
    root = tempfile.mkdtemp()
    for i in range(len(results)):
        open(os.path.join(root, f"{i}.mkv"), "w").close()
    plays = iter(results)

    def fake_gsq(path, *a, **kw):
        ret, reason = next(plays)
        if reason is not KEEP:  # KEEP = a path that never reached should_skip_file
            subgen._skip_reason.set(reason)
        return ret

    orig = (subgen.gen_subtitles_queue, subgen.monitor, subgen.path_mapping)
    subgen.gen_subtitles_queue, subgen.monitor, subgen.path_mapping = (
        fake_gsq,
        False,
        (lambda p: p),
    )
    try:
        return subgen.transcribe_existing(root)
    finally:
        subgen.gen_subtitles_queue, subgen.monitor, subgen.path_mapping = orig


try:
    counts = batch(
        [
            ("skipped", "internal_sub"),
            ("skipped", "internal_sub"),
            ("skipped", "audio_lang"),
            ("queued", None),
        ]
    )
    check("/batch skipped count unchanged", counts.get("skipped") == 3, repr(counts))
    check(
        "/batch skip_reasons tallied",
        counts.get("skip_reasons") == {"internal_sub": 2, "audio_lang": 1},
        repr(counts.get("skip_reasons")),
    )
    counts = batch([("queued", None)])
    check(
        "/batch skip_reasons present and empty with no skips",
        counts.get("skip_reasons") == {},
        repr(counts.get("skip_reasons")),
    )
    # A skip whose branch recorded nothing (a future upstream branch we have
    # not tagged) must still be counted, as 'unknown', so the map sums to skipped.
    counts = batch([("skipped", None)])
    check(
        "/batch untagged skip counts as unknown",
        counts.get("skip_reasons") == {"unknown": 1},
        repr(counts.get("skip_reasons")),
    )
    # Files are walked in sorted order (0.mkv, 1.mkv). The second skip never
    # reached should_skip_file, so it must NOT inherit the first file's reason.
    counts = batch([("skipped", "internal_sub"), ("skipped", KEEP)])
    check(
        "/batch a later skip does not inherit the previous reason",
        counts.get("skip_reasons") == {"internal_sub": 1, "unknown": 1},
        repr(counts.get("skip_reasons")),
    )
except Exception as e:  # noqa: BLE001
    check("/batch tally", False, f"raised {type(e).__name__}: {e}")

print()
print(f"{len(FAILS)} failure(s)" if FAILS else "all checks passed")
sys.exit(1 if FAILS else 0)
