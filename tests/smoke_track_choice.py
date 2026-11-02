"""Runtime smoke for the #633 track-choice contract (patch 0062).

Run INSIDE the built image (the patched tree needs torch, faster_whisper, av...),
with the PATCHED upstream tree mounted:

    docker run --rm -v "$PWD/upstream:/work:ro" -v "$PWD/tests:/smoke:ro"       -w /work -e PYTHONPATH=/work --entrypoint python3       ghcr.io/coaxk/subarr-subgen:<tag> /smoke/smoke_track_choice.py

WHY THIS EXISTS: subarr (coaxk/subarr#633) predicts which audio language subgen
will transcribe by mirroring choose_transcribe_language from the capabilities
block. tests/fixtures/track_choice_cases.json is the shared contract: the same
bytes live in subarr as tests/fixtures/subgen_track_choice_cases.json. This
smoke proves the fixture's expected_source matches what real subgen returns,
and that /queue advertises the inputs subarr needs, null entries included.

Not wired into CI (smoke_patched_behaviour.py is not either): it needs the
built image, so run it locally against it.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.environ.get("SUBGEN_TREE", "/work"))

import subgen  # noqa: E402
from subgen import LanguageCode  # noqa: E402

FAILS = []

# Update this together with subarr's copy of the fixture (tests/
# test_per_file_task_633.py). A change on one side only fails one suite loudly.
FIXTURE_SHA256 = "e00b59d02962493ec0930c492e97a5838543f8454e9041ca79bd42f403de7b1d"

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "track_choice_cases.json")


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else '*** FAIL ***'}  {name}{('  -> ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def lang(value):
    return LanguageCode.NONE if value is None else LanguageCode.from_string(value)


def build_tracks(raw):
    """Build audio_tracks the way upstream get_audio_tracks does."""
    return [
        {"language": LanguageCode.from_iso_639_2(tag or "Unknown"), "default": bool(default)}
        for tag, default in raw
    ]


def main():
    raw = open(FIXTURE, "rb").read()
    check("fixture sha256 matches the constant", hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256,
          hashlib.sha256(raw).hexdigest())
    data = json.loads(raw)
    check("fixture version is 1", data["version"] == 1)
    cases = data["cases"]
    check("fixture has cases", len(cases) >= 13, str(len(cases)))

    orig_fd = subgen.force_detected_language_to
    orig_pref = subgen.preferred_audio_languages
    try:
        for case in cases:
            subgen.force_detected_language_to = lang(case["force_detected"])
            subgen.preferred_audio_languages = [lang(p) for p in case["preferred"]]
            forced = LanguageCode.from_string(case["forced"]) if case["forced"] else None
            result = subgen.choose_transcribe_language(
                "x.mkv", forced, audio_tracks=build_tracks(case["tracks"]))
            got = result.to_iso_639_1()
            check(f"choose: {case['name']}", got == case["expected_source"],
                  f"expected {case['expected_source']!r} got {got!r}")
            # The conversion subarr never sees (preferred -> advertised) is part
            # of the contract: subarr feeds preferred_advertised to its resolver.
            adv = subgen.queue_status()["capabilities"]["preferred_audio_languages"]
            check(f"advertised: {case['name']}", adv == case["preferred_advertised"],
                  f"expected {case['preferred_advertised']!r} got {adv!r}")

        # The capabilities block must reflect the module globals, null entries
        # kept in place, duplicates kept, order kept.
        subgen.force_detected_language_to = LanguageCode.from_string("ja")
        subgen.preferred_audio_languages = [lang(p) for p in [None, "eng", "eng", "de"]]
        caps = subgen.queue_status()["capabilities"]
        check("caps: preferred_audio_languages present", "preferred_audio_languages" in caps)
        check("caps: preferred list keeps order, null in place, duplicates",
              caps.get("preferred_audio_languages") == [None, "en", "en", "de"],
              repr(caps.get("preferred_audio_languages")))
        check("caps: force_detected_language_to advertised",
              caps.get("force_detected_language_to") == "ja", repr(caps.get("force_detected_language_to")))
        json.dumps(caps)  # must be JSON-serialisable
        check("caps: block is JSON-serialisable", True)

        subgen.force_detected_language_to = LanguageCode.NONE
        subgen.preferred_audio_languages = [lang("eng")]
        caps = subgen.queue_status()["capabilities"]
        check("caps: unset force_detected is null (key present)",
              "force_detected_language_to" in caps and caps["force_detected_language_to"] is None)
        check("caps: single preferred", caps.get("preferred_audio_languages") == ["en"])

        subgen.preferred_audio_languages = []
        caps = subgen.queue_status()["capabilities"]
        check("caps: empty preferred is an empty list, not missing",
              caps.get("preferred_audio_languages") == [])
    finally:
        subgen.force_detected_language_to = orig_fd
        subgen.preferred_audio_languages = orig_pref

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("all track-choice smokes passed")


if __name__ == "__main__":
    main()
