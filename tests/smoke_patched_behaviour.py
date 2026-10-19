"""Runtime smokes for our patched transcribe_existing / SKIP_STARTUP_SCAN behaviour.

Run INSIDE the built image (the patched tree needs torch, faster_whisper, av...):

    docker run --rm -v "$PWD/upstream:/work:ro" -v "$PWD/tests:/smoke:ro"       -w /work -e PYTHONPATH=/work --entrypoint python3       ghcr.io/coaxk/subarr-subgen:<tag> /smoke/smoke_patched_behaviour.py

WHY THIS EXISTS: upstream ships tests for the .subgen_skip marker and
SKIP_STARTUP_SCAN, but it stubs gen_subtitles_queue with a lambda that does not
accept our patch-added kwargs (audio_language_override, ignore_forced_override),
so those tests die on TypeError before reaching their assertions. The doubles
here accept **kwargs, so they actually exercise the behaviour.

The last check is the important one: it pins patch 0033. Upstream gates
SKIP_STARTUP_SCAN inside transcribe_existing(), which /batch also calls, so
upstream semantics would silently no-op every transcription subarr requests.
If a future port drops 0033, that check fails.
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.environ.get("SUBGEN_TREE", "/work"))

import subgen  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else '*** FAIL ***'}  {name}{('  -> ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)

def _tree():
    """root/a.mkv, root/b.mkv, root/keep/c.mkv, root/skipme/.subgen_skip + d.mkv"""
    root = tempfile.mkdtemp()
    for rel in ("a.mkv", "b.mkv"):
        open(os.path.join(root, rel), "w").close()
    os.makedirs(os.path.join(root, "keep"))
    open(os.path.join(root, "keep", "c.mkv"), "w").close()
    os.makedirs(os.path.join(root, "skipme"))
    open(os.path.join(root, "skipme", ".subgen_skip"), "w").close()
    open(os.path.join(root, "skipme", "d.mkv"), "w").close()
    return root


def run(root, **overrides):
    """Call transcribe_existing with a kwargs-tolerant queue double."""
    seen = []

    def fake_gsq(path, *a, **kw):
        seen.append(path)
        return "queued"

    orig_gsq = subgen.gen_subtitles_queue
    orig_monitor = subgen.monitor
    orig_pm = subgen.path_mapping
    subgen.gen_subtitles_queue = fake_gsq
    subgen.monitor = False
    subgen.path_mapping = lambda p: p
    try:
        counts = subgen.transcribe_existing(root, **overrides)
    finally:
        subgen.gen_subtitles_queue = orig_gsq
        subgen.monitor = orig_monitor
        subgen.path_mapping = orig_pm
    return seen, counts


root = _tree()

# --- 1. upstream's .subgen_skip marker still prunes (carried over verbatim) ---
seen, counts = run(root)
names = sorted(os.path.basename(p) for p in seen)
check("marker prunes the marked subtree", "d.mkv" not in names, f"walked={names}")
check(
    "unmarked files still walked",
    names == ["a.mkv", "b.mkv", "c.mkv"],
    f"walked={names}",
)

# --- 2. our reverse-sort still works ---
fwd, _ = run(root)
rev, _ = run(root, reverse=True)
check(
    "reverse=True inverts order",
    rev == list(reversed(fwd)),
    f"fwd[0]={os.path.basename(fwd[0])} rev[0]={os.path.basename(rev[0])}",
)

# --- 3. our structured counts still returned ---
check("counts dict returned", isinstance(counts, dict), f"type={type(counts).__name__}")
check(
    "counts.walked matches files queued",
    counts.get("walked") == len(seen),
    f"walked={counts.get('walked')} seen={len(seen)}",
)
check(
    "counts.queued tallied",
    counts.get("queued") == len(seen),
    f"queued={counts.get('queued')}",
)
check(
    "counts echoes reverse flag",
    counts.get("reverse") is False,
    f"reverse={counts.get('reverse')}",
)

# --- 4. THE DIVERGENCE: SKIP_STARTUP_SCAN must NOT gag an explicit call ---
orig_flag = subgen.skip_startup_scan
subgen.skip_startup_scan = True
try:
    seen_skip, counts_skip = run(root)
finally:
    subgen.skip_startup_scan = orig_flag
check(
    "SKIP_STARTUP_SCAN=1 does NOT no-op an explicit /batch-style call",
    len(seen_skip) == 3 and counts_skip.get("walked") == 3,
    f"walked={counts_skip.get('walked')} (upstream semantics would give 0)",
)


def boot(skip_flag):
    """Run lifespan startup with a fake Thread and report whether a scan spawned."""
    spawned = []

    class FakeThread:
        def __init__(self, *a, **kw):
            spawned.append(kw.get("target", a[0] if a else None))

        def start(self):
            pass

        def join(self, *a, **kw):
            pass

    orig_thread = subgen.threading.Thread
    orig_flag = subgen.skip_startup_scan
    orig_folders = subgen.transcribe_folders
    subgen.threading.Thread = FakeThread
    subgen.skip_startup_scan = skip_flag
    subgen.transcribe_folders = "/tmp"

    async def drive():
        async with subgen.lifespan(None):
            pass

    try:
        asyncio.run(drive())
    finally:
        subgen.threading.Thread = orig_thread
        subgen.skip_startup_scan = orig_flag
        subgen.transcribe_folders = orig_folders
    return spawned


off = boot(False)
check(
    "SKIP_STARTUP_SCAN unset -> boot scan thread IS started",
    len(off) == 1 and getattr(off[0], "__name__", "") == "transcribe_existing",
    f"spawned={[getattr(t, '__name__', t) for t in off]}",
)

on = boot(True)
check(
    "SKIP_STARTUP_SCAN=1 -> boot scan thread is NOT started",
    len(on) == 0,
    f"spawned={[getattr(t, '__name__', t) for t in on]}",
)


# ---------------------------------------------------------------------------
# patch 0037 (#458): image-based subtitles are not coverage
#
# The failure mode this guards is a SILENT NO-OP, not a crash. ffprobe names a
# stream's codec 'hdmv_pgs_subtitle' but PyAV's codec_context.name -- which is
# what has_internal_subtitle_in_language actually reads -- says 'pgssub'. A
# deny set carrying only the ffprobe spellings applies clean, passes every
# structural gate, and matches nothing at runtime. These checks exercise the
# real predicates so that cannot pass unnoticed.
# ---------------------------------------------------------------------------

check(
    "#458 PyAV decoder spellings are denied (the silent-no-op case)",
    all(subgen.is_image_subtitle_codec(n) for n in ("pgssub", "dvdsub", "dvbsub", "xsub")),
    "these are what PyAV reports, not the ffprobe names",
)
check(
    "#458 ffprobe codec_name spellings are denied too",
    all(
        subgen.is_image_subtitle_codec(n)
        for n in ("hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub")
    ),
)
check(
    "#458 text codecs survive, including the long tail and both srt spellings",
    not any(
        subgen.is_image_subtitle_codec(n)
        for n in ("subrip", "srt", "ass", "ssa", "mov_text", "microdvd",
                  "subviewer", "stl", "jacosub", "sami", "mpl2", "cc_dec")
    ),
)
check(
    "#458 dvb_teletext and eia_608 are NOT denied (ffmpeg decodes both to text)",
    not subgen.is_image_subtitle_codec("dvb_teletext")
    and not subgen.is_image_subtitle_codec("eia_608"),
)
check(
    "#458 an unknown codec is treated as TEXT (fails safe, does not skip work)",
    not subgen.is_image_subtitle_codec("some_future_format")
    and not subgen.is_image_subtitle_codec(None),
)

# The ambiguous .sub: MicroDVD writes text to .sub, VobSub writes a bitmap to
# .sub and ALWAYS pairs it with a .idx. Resolve by looking, not by guessing.
_subdir = tempfile.mkdtemp()
_lone = os.path.join(_subdir, "microdvd_movie.sub")
open(_lone, "w").close()
_vob = os.path.join(_subdir, "vobsub_movie.sub")
open(_vob, "w").close()
open(os.path.join(_subdir, "vobsub_movie.idx"), "w").close()

check(
    "#458 a .sub WITH a sibling .idx is VobSub -> image",
    subgen.is_image_subtitle_file(_vob),
)
check(
    "#458 a LONE .sub is MicroDVD text -> not image",
    not subgen.is_image_subtitle_file(_lone),
    "guessing here would throw away real text subtitles",
)
check(
    "#458 .idx and .pgs are unconditionally image",
    subgen.is_image_subtitle_file(os.path.join(_subdir, "x.idx"))
    and subgen.is_image_subtitle_file(os.path.join(_subdir, "x.pgs")),
)
check(
    "#458 .srt/.ass sidecars are never image",
    not subgen.is_image_subtitle_file(os.path.join(_subdir, "x.srt"))
    and not subgen.is_image_subtitle_file(os.path.join(_subdir, "x.ass")),
)
check(
    "#458 IGNORE_IMAGE_SUBTITLES defaults OFF",
    subgen.ignore_image_subtitles is False,
    "image subs are the norm on DVD/Blu-ray rips; default-on would queue "
    "thousands of unannounced transcriptions",
)

# ---------------------------------------------------------------------------
# patch 0039 (#458 follow-on): per-request bypass_skip
#
# The failure mode is silent: if the kwarg is dropped anywhere along
# /batch -> transcribe_existing -> gen_subtitles_queue -> should_skip_file it
# defaults to False, the endpoint still returns 200, and the file is still
# skipped. To the user that is "the button does nothing", with no error.
# ---------------------------------------------------------------------------

import inspect as _inspect

_chain = {
    "should_skip_file": subgen.should_skip_file,
    "gen_subtitles_queue": subgen.gen_subtitles_queue,
    "transcribe_existing": subgen.transcribe_existing,
}
for _name, _fn in _chain.items():
    check(
        f"#458/0039 {_name} accepts bypass_skip",
        "bypass_skip" in _inspect.signature(_fn).parameters,
    )

# The early-out must short-circuit BEFORE any filesystem or ffprobe work, which
# is what makes it safe to call on anything. A path that does not exist proves
# it: if any check ran first this would raise or stall, not return False.
check(
    "#458/0039 bypass_skip returns False without touching the file",
    subgen.should_skip_file(
        "/nonexistent/definitely/not/here.mkv",
        subgen.LanguageCode.from_string("en"),
        audio_langs=[subgen.LanguageCode.from_string("en")],
        bypass_skip=True,
    ) is False,
)

check(
    "#458/0039 default is False so existing callers are unchanged",
    _inspect.signature(subgen.should_skip_file).parameters["bypass_skip"].default is False,
)


# ---------------------------------------------------------------------------
# patch 0058 (#77): a failed model load hands its CUDA pool memory back
#
# The structural gate in validate-patched.py proves the call sites exist. These
# prove the behaviour that makes the call safe to sit on a failure path: the
# load error still reaches the caller (the /config swap's 422 and rollback
# depend on it), and the helper never raises, with or without a GPU.
# ---------------------------------------------------------------------------

_released = []
_orig_loader = subgen.stable_whisper.load_faster_whisper
_orig_release = subgen.release_cuda_pool_after_failed_load
_orig_model = subgen.model


def _oom_loader(*a, **kw):
    raise RuntimeError("CUDA failed with error out of memory")


subgen.stable_whisper.load_faster_whisper = _oom_loader
subgen.release_cuda_pool_after_failed_load = lambda *a, **kw: _released.append(1)
subgen.model = None
_raised = None
try:
    subgen.start_model()
except RuntimeError as e:
    _raised = e
finally:
    subgen.stable_whisper.load_faster_whisper = _orig_loader
    subgen.release_cuda_pool_after_failed_load = _orig_release
    subgen.model = _orig_model

check(
    "#77/0058 start_model re-raises the load error",
    _raised is not None and "out of memory" in str(_raised),
    f"raised={_raised!r}",
)
check(
    "#77/0058 start_model releases the pool exactly once on a failed load",
    len(_released) == 1,
    f"calls={len(_released)}",
)

_orig_device = subgen.transcribe_device
subgen.transcribe_device = "cpu"
try:
    check(
        "#77/0058 helper is a no-op on CPU",
        subgen.release_cuda_pool_after_failed_load() is None,
    )
finally:
    subgen.transcribe_device = _orig_device

_orig_cdll = subgen.ctypes.CDLL


def _no_libcuda(name, *a, **kw):
    if "libcuda" in str(name):
        raise OSError(f"{name}: cannot open shared object file")
    return _orig_cdll(name, *a, **kw)


subgen.transcribe_device = "cuda"
subgen.ctypes.CDLL = _no_libcuda
try:
    check(
        "#77/0058 helper returns None, not an exception, without libcuda",
        subgen.release_cuda_pool_after_failed_load() is None,
    )
finally:
    subgen.ctypes.CDLL = _orig_cdll
    subgen.transcribe_device = _orig_device

print()
print("RESULT:", "ALL SMOKES PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
