#!/usr/bin/env python3
"""Validate that the patched upstream/subgen.py meets subarr's structural
contract. Runs after apply-patches.sh.

Gates (in order):
  1. compile() — patched source must compile in-memory without SyntaxError
  2. AST: transcribe_existing has 'reverse' arg
  3. AST: batch() has 'reverse' arg
  4. AST: gen_subtitles_queue returns at least one string constant
  5. text: language_specific_kwargs present (from patch 0001)
  6. text: '@app.get("/queue")' present (from patch 0007)
  7. text: 'JSONResponse' imported (from patch 0006)
  8. text: '[v4.2 PATCH] _queued / _processing track' present (from patch 0007)
  9. text: 'Eager model load on boot' present (from patch 0002)

Exits non-zero with a clear message on first failure. This is the
"would subarr-subgen actually work?" gate; a clean run means the image
is safe to build.

Mirrors update_subgen_v4.py's STEP 7 (compile) and STEP 8 (AST) gates.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "upstream" / "subgen.py"
LAUNCHER = REPO / "upstream" / "launcher.py"


def fail(msg: str) -> None:
    print(f"VALIDATE: FAIL — {msg}", file=sys.stderr)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"VALIDATE: ok — {msg}")


def fn_args(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return [a.arg for a in n.args.args]
    return None


def fn_returns_string_constant(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            for sub in ast.walk(n):
                if (
                    isinstance(sub, ast.Return)
                    and isinstance(sub.value, ast.Constant)
                    and isinstance(sub.value.value, str)
                ):
                    return True
    return False


def validate_launcher() -> None:
    """Patch 0044: this fork must never re-download its own code at runtime.

    Upstream's launcher fetches subgen.py / launcher.py / language_code.py from
    McCloudS/subgen at container start, which silently replaces the entire patch
    stack with vanilla code while the image tag and OCI labels still advertise a
    patch rev (coaxk/subarr-subgen#59).

    The real contract here is an ABSENCE (no download call survives), and an
    absence check is exactly the kind that can pass while measuring nothing. So
    it is paired with three positives: a guard that is defined but never called,
    or a launch target that BRANCH can still rename, would each leave the
    download reachable while the absence assertion stayed green.
    """
    if not LAUNCHER.is_file():
        fail(f"{LAUNCHER} not found - did you run apply-patches.sh?")
    code = LAUNCHER.read_text(encoding="utf-8")

    try:
        compile(code, str(LAUNCHER), "exec")
    except SyntaxError as e:
        fail(f"launcher.py compile() failed: {e}")
    ok("launcher.py compile() passed")

    tree = ast.parse(code)

    # Positive 1: the guard exists.
    if not any(
        isinstance(n, ast.FunctionDef) and n.name == "warn_self_update_disabled"
        for n in ast.walk(tree)
    ):
        fail("launcher.py has no warn_self_update_disabled() - patch 0044 not landed")
    ok("patch 0044 (warn_self_update_disabled defined)")

    # Positive 2: and it is actually CALLED. A defined-but-unwired guard would
    # satisfy the assertion above while doing nothing at runtime.
    if not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "warn_self_update_disabled"
        for n in ast.walk(tree)
    ):
        fail("warn_self_update_disabled() is defined but never called - patch 0044")
    ok("patch 0044 (warn_self_update_disabled wired into main)")

    # Positive 3: the launch target is a literal. BRANCH=<name> renames it to
    # subgen-<name>.py upstream, which this image does not contain, and that
    # re-arms the download-if-missing arm with no opt-in from the operator.
    if 'subgen_script_to_run = "subgen.py"' not in code:
        fail("subgen_script_to_run is not pinned to a literal subgen.py - patch 0044")
    ok("patch 0044 (launch target pinned to the baked subgen.py)")

    # Positive 4 (patch 0046): --install is reported like the other inputs.
    if "getattr(args, 'install', False)" not in code:
        fail("--install is not reported by self_update_requests() - patch 0046")
    ok("patch 0046 (--install reported as an ignored input)")

    # ── The contract itself ──────────────────────────────────────────────
    # As of patch 0046 this fork downloads NOTHING at runtime, so the gate is
    # simply that download_from_github has zero call sites. That is strictly
    # stronger than the filename allow-list this replaced, which could only
    # catch the three names it happened to know about and let requirements.txt
    # through because it was passed by variable rather than as a literal.
    callers = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "download_from_github"
    ]
    if callers:
        fail(
            "launcher.py still downloads at runtime, %d call site(s) at line(s) %s"
            " - patch 0044/0046 not landed"
            % (len(callers), ", ".join(str(n.lineno) for n in callers))
        )
    ok("patch 0044/0046 (download_from_github has zero call sites)")

    # And nothing re-resolves the pinned dependency set at runtime either.
    installers = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "install_packages_from_requirements"
    ]
    if installers:
        fail(
            "launcher.py still pip-installs at runtime (line %s) - patch 0046 not landed"
            % ", ".join(str(n.lineno) for n in installers)
        )
    ok("patch 0046 (no runtime pip install over the pinned deps)")


def main() -> int:
    if not TARGET.is_file():
        fail(f"{TARGET} not found — did you run apply-patches.sh?")
    code = TARGET.read_text(encoding="utf-8")

    # Gate 1: compile
    try:
        compile(code, str(TARGET), "exec")
    except SyntaxError as e:
        fail(f"compile() failed: {e}")
    ok("compile() passed")

    tree = ast.parse(code)

    # Gate 2-4: function signatures + returns
    te_args = fn_args(tree, "transcribe_existing")
    if not te_args or "reverse" not in te_args:
        fail(f"transcribe_existing missing 'reverse' arg (got {te_args})")
    ok(f"transcribe_existing{tuple(te_args)} has reverse")

    ba_args = fn_args(tree, "batch")
    if not ba_args or "reverse" not in ba_args:
        fail(f"batch() missing 'reverse' arg (got {ba_args})")
    ok(f"batch{tuple(ba_args)} has reverse")

    # Gate: patch 0024 (per-request ignore_forced threading)
    if "ignore_forced" not in (ba_args or []):
        fail(
            f"batch() missing 'ignore_forced' arg (got {ba_args}) — patch 0024 not landed"
        )
    ok("batch() has ignore_forced (patch 0024)")

    ssf_args = fn_args(tree, "should_skip_file")
    if not ssf_args or "ignore_forced_override" not in ssf_args:
        fail(
            f"should_skip_file missing 'ignore_forced_override' (got {ssf_args}) — patch 0024"
        )
    ok("should_skip_file has ignore_forced_override (patch 0024)")

    hisl_args = fn_args(tree, "has_internal_subtitle_in_language")
    if not hisl_args or "ignore_forced_override" not in hisl_args:
        fail(
            f"has_internal_subtitle_in_language missing 'ignore_forced_override' (got {hisl_args}) — patch 0024"
        )
    ok("has_internal_subtitle_in_language has ignore_forced_override (patch 0024)")

    if not fn_returns_string_constant(tree, "gen_subtitles_queue"):
        fail(
            "gen_subtitles_queue has no string-constant return — patch 0005 not landed"
        )
    ok("gen_subtitles_queue returns dispatch strings")

    # Gate 5-9: text presence
    text_checks = [
        ("language_specific_kwargs", "patch 0001 (per-lang kwargs)"),
        ("Eager model load on boot", "patch 0002 (eager-load)"),
        ("[v4.2 PATCH] _queued / _processing track", "patch 0007 (DQ type-tracking)"),
        ('@app.get("/queue")', "patch 0007 (/queue endpoint)"),
        (
            "from fastapi.responses import StreamingResponse, JSONResponse",
            "patch 0006 (JSONResponse import)",
        ),
        ("[v4.1 PATCH] Structured dispatch counts", "patch 0003 (structured counts)"),
        ("return JSONResponse(content=result", "patch 0004 (/batch JSONResponse)"),
        ('@app.post("/config")', "patch 0022 (/config endpoint)"),
        ('"runtime_config": True', "patch 0022 (runtime_config capability)"),
        ("subarr_subgen_release_tag", "patch 0022 (release tag emission)"),
        (
            '"concurrent_transcriptions": concurrent_transcriptions',
            "patch 0023 (concurrent_transcriptions capability)",
        ),
        (
            '"request_ignore_forced": True',
            "patch 0024 (request_ignore_forced capability)",
        ),
        ("_eff_ignore_forced", "patch 0024 (per-request forced override logic)"),
        ("def path_is_allowed", "patch 0025 (#13 containment helper)"),
        ("SUBGEN_PATH_ALLOWLIST", "patch 0025 (#13 containment allowlist env)"),
        ("if not path_is_allowed(audio_path):", "patch 0025 (#13 /asr containment)"),
        (
            "if not path_is_allowed(path_mapping(directory)):",
            "patch 0025 (#13 /batch containment)",
        ),
        ("if not path_is_allowed(mapped):", "patch 0025 (#13 /detect containment)"),
        (
            "track: Union[int, None] = Query(default=None, ge=0, le=31)",
            "patch 0026 (#17 detect track param)",
        ),
        (
            '_out_kwargs["map"] = f"0:a:{track}"',
            "patch 0026 (#17 per-track ffmpeg map)",
        ),
        ("model_load_lock = threading.RLock()", "patch 0027 (#6 swap RLock)"),
        ("[PATCH 0027 / #6 item 2] live model", "patch 0027 (#6 live model in caps)"),
        ("model_load_lock.acquire()", "patch 0027 (#6 atomic swap)"),
        (
            '"detect_language_track": True',
            "patch 0027 (#17 detect_language_track capability)",
        ),
        ("def _perform_config_switch", "patch 0029 (#6 async switch fn)"),
        ("def _async_config_switch", "patch 0029 (#6 async switch thread)"),
        ("wait: bool = Query(True)", "patch 0029 (#6 /config wait param)"),
        (
            '"config_switch": dict(_config_switch_state)',
            "patch 0029 (#6 /queue config_switch)",
        ),
        ('"async_config": True', "patch 0029 (#6 async_config capability)"),
        # patch 0048 (subarr#498). subarr must be able to SEE the effective skip
        # list, because audio_language_override substitutes into the skip check
        # rather than bypassing it, so forwarding a verified 'en' to an install
        # that skips English audio skips the file instead of transcribing it.
        (
            '"skip_audio_languages": [',
            "patch 0048 (#498 effective skip list advertised)",
        ),
        # patch 0050 (subarr#505). subarr must be able to SEE what this instance
        # can PRODUCE, not just what it will skip. Whisper writes one language
        # per job -- always English in translate mode -- so without this subarr
        # queues a row wanting a language this instance cannot make, forever.
        (
            '"transcribe_or_translate": transcribe_or_translate,',
            "patch 0050 (#505 output task advertised)",
        ),
        (
            '"subtitle_language_name": subtitle_language_name or "",',
            "patch 0050 (#505 output naming advertised)",
        ),
        # patch_rev is cumulative -- each bump patch overwrites the last, so the
        # only value observable in the fully-patched tree is the newest one.
        # Assert against the LAST bump patch in the series and update this needle
        # whenever a new bump patch is added (0030 -> v4.17 went stale when 0032
        # landed v4.18, and this check failed silently behind an apply failure).
        (
            "subarr_subgen_patch_rev = 'v4.34'",
            "patch 0057 (patch_rev bump v4.34, latest)",
        ),
        # --- patch 0054 (#69: direct-task claim) ------------------------------
        # perform_model_cleanup() reads "is the system idle" and then unloads
        # the model, holding model_cleanup_lock across both. Incrementing
        # active_direct_tasks alone leaves a window between that read and the
        # unload in which a request runs against a model being torn down, which
        # surfaced as "No model replica is available in this thread" on live
        # Bazarr detect calls. The claim must take model_cleanup_lock, and BOTH
        # direct paths must go through it -- a re-port that keeps the helper but
        # restores a bare increment at either call site brings the race back.
        ("def direct_task_claim():", "patch 0054 (#69 direct-task claim exists)"),
        (
            "\n".join(
                [
                    "    with model_cleanup_lock:",
                    "        with active_direct_tasks_lock:",
                    "            active_direct_tasks += 1",
                ]
            ),
            "patch 0054 (#69 claim excludes the cleanup)",
        ),
        ("_claim = direct_task_claim()", "patch 0054 (#69 /detect-language claims)"),
        ("with direct_task_claim():", "patch 0054 (#69 robust detect claims)"),
        # --- patch 0039 (#458 follow-on: per-request bypass_skip) -------------
        # The bypass must reach should_skip_file. Every link in the chain is
        # asserted separately: a missing kwarg anywhere in
        # /batch -> transcribe_existing -> gen_subtitles_queue -> should_skip_file
        # silently falls back to the default False, so the endpoint accepts
        # bypass_skip=true, returns 200, and still skips the file. That reads to
        # the user as "the button does nothing" with no error anywhere.
        (
            "bypass_skip: bool = Query(default=False),",
            "patch 0039 (bypass_skip on /batch)",
        ),
        ("bypass_skip=bypass_skip,", "patch 0039 (threaded from /batch)"),
        (
            "bypass_skip: bool = False, **task_kwargs",
            "patch 0039 (gen_subtitles_queue accepts it)",
        ),
        ("bypass_skip=bypass_skip):", "patch 0039 (reaches should_skip_file)"),
        ("    if bypass_skip:", "patch 0039 (the early-out exists)"),
        ('"bypass_skip": True,', "patch 0039 (capability advertised)"),
        # --- patch 0037 (#458 image-based subs are not coverage) -------------
        # The env gate exists and defaults OFF. Image subs are the norm on DVD
        # and Blu-ray rips, so an accidental default-on would queue thousands
        # of unannounced transcriptions.
        (
            "ignore_image_subtitles = convert_to_bool(os.getenv('IGNORE_IMAGE_SUBTITLES', False))",
            "patch 0037 (#458 IGNORE_IMAGE_SUBTITLES env, default off)",
        ),
        # BOTH naming schemes are in the deny set. ffprobe says
        # 'hdmv_pgs_subtitle', PyAV says 'pgssub', and THIS code path reads
        # PyAV -- so a set carrying only the ffprobe spellings matches 1 of 4
        # and the whole patch becomes a silent no-op that still applies clean
        # and still passes every structural check. Assert the PyAV spellings
        # specifically: they are the ones that actually fire at runtime.
        ("'hdmv_pgs_subtitle', 'pgssub'", "patch 0037 (#458 PGS: both spellings)"),
        ("'dvd_subtitle', 'dvdsub'", "patch 0037 (#458 VobSub: both spellings)"),
        ("'dvb_subtitle', 'dvbsub'", "patch 0037 (#458 DVB: both spellings)"),
        # The internal (embedded stream) check consults the codec.
        (
            "if is_image_subtitle_codec(_codec_name):",
            "patch 0037 (#458 internal check reads the codec)",
        ),
        # The external (sidecar) check consults the filename.
        (
            "if ignore_image_subtitles and is_image_subtitle_file(file_path):",
            "patch 0037 (#458 external check screens sidecars)",
        ),
        # Reported as the RUNTIME value so subarr can tell an actionable gap
        # from an un-fillable one (the #79 precedent).
        (
            '"ignore_image_subtitles": bool(ignore_image_subtitles),',
            "patch 0037 (#458 runtime capability exposed)",
        ),
        # patch 0041 (subarr#483). Upstream excludes forced EMBEDDED tracks and
        # says why in its own docstring, but never screened SIDECARS, so a
        # .en.forced.srt counted as full English coverage and the file was
        # skipped forever. Assert the sidecar screen specifically: the embedded
        # half already existed and would mask a dropped patch.
        (
            'any(part.lower() == "forced" for part in subtitle_parts)',
            "patch 0041 (#483 external check screens forced sidecars)",
        ),
        # The per-request override must reach the sidecar check too, or a caller
        # can bypass forced-exclusion for embedded tracks and silently not for
        # external ones.
        (
            "only_match_subgen_subtitles=only_match_subgen_subtitles, ignore_forced_override=ignore_forced_override",
            "patch 0041 (#483 override reaches the external check)",
        ),
    ]
    for needle, label in text_checks:
        if needle not in code:
            fail(f"text: missing — {label} (needle: {needle!r})")
        ok(label)

    validate_launcher()

    print()
    print("VALIDATE: all gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
