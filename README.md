# subarr-subgen + Parakeet (fork)

This fork adds **patch 0058**: a `TRANSCRIBE_BACKEND=parakeet` option that
transcribes subtitles with NVIDIA **Parakeet TDT 0.6B v3** (int8) on the **CPU**
through [onnx-asr](https://github.com/istupakov/onnx-asr). No GPU or VRAM is used.

- **Images** (built by `.github/workflows/fork-image.yml` after a real-model
  smoke test, `scripts/parakeet-smoke.py`):

  | Branch | Image | Dockerfile | Use |
  |---|---|---|---|
  | `main` | `ghcr.io/adman234/subarr-subgen:latest` (`:cpu-<sha7>`) | `docker/Dockerfile.cpu` (ubuntu:22.04, CPU torch) | lean, no GPU at all |
  | `gpu` | `ghcr.io/adman234/subarr-subgen:gpu` (`:gpu-<sha7>`) | `docker/Dockerfile` (upstream CUDA base) | Whisper detection/fallback on an NVIDIA GPU; Parakeet still on CPU |

  `docker/Dockerfile` is upstream's CUDA file, kept unchanged on `main` so new
  subarr-subgen releases merge cleanly; only the `gpu` branch builds it.
- **Unraid templates:**
  - CPU: `wget -O /boot/config/plugins/dockerMan/templates-user/my-subarr-subgen-parakeet.xml https://raw.githubusercontent.com/adman234/subarr-subgen/main/unraid/subarr-subgen-parakeet.xml`
  - GPU: `wget -O /boot/config/plugins/dockerMan/templates-user/my-subarr-subgen-parakeet-gpu.xml https://raw.githubusercontent.com/adman234/subarr-subgen/main/unraid/subarr-subgen-parakeet-gpu.xml`
- **Language:** the language the caller sends (Bazarr's `?language=`, media
  metadata) is always used. Only when it is missing does `WHISPER_MODEL` detect
  it (`DETECT_LANGUAGE_LENGTH` seconds, up to ten 30 s windows).
- **Coverage:** Parakeet v3 handles bg, hr, cs, da, nl, en, et, fi, fr, de, el,
  hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk. Other languages and
  `task=translate` go to `WHISPER_MODEL` (`PARAKEET_FALLBACK=whisper`) or fail
  (`PARAKEET_FALLBACK=none`).
- **Memory:** models load on the first job (`EAGER_MODEL_LOAD=false`, new
  default) and unload with the existing `CLEAR_VRAM_ON_COMPLETE` /
  `MODEL_CLEANUP_DELAY` cleanup. Parakeet and Silero VAD are stored under
  `MODEL_PATH/onnx-asr/`.

| Variable | Default | Meaning |
|---|---|---|
| `TRANSCRIBE_BACKEND` | `whisper` | `parakeet` enables this backend |
| `PARAKEET_MODEL` | `nemo-parakeet-tdt-0.6b-v3` | any onnx-asr Parakeet model |
| `PARAKEET_QUANTIZATION` | `int8` | `none` for fp32 |
| `PARAKEET_THREADS` | `0` | 0 = one per CPU the container is pinned to |
| `PARAKEET_FALLBACK` | `whisper` | `none` to fail unsupported languages instead |
| `PARAKEET_LANGUAGES` | (model default) | override the supported-language list |
| `PARAKEET_MAX_SEGMENT_S` | `20` | longest VAD segment sent to Parakeet |
| `PARAKEET_VAD_THRESHOLD` / `_MIN_SILENCE_MS` / `_SPEECH_PAD_MS` | `0.5` / `300` / `100` | Silero VAD tuning |
| `EAGER_MODEL_LOAD` | `false` | `true` restores patch 0002's load-at-startup |

With the Parakeet backend, Whisper-only settings (`SUBGEN_KWARGS`, per-language
kwargs and prompts) apply only to the Whisper fallback.

---

# subarr-subgen

Pre-built [McCloudS/subgen](https://github.com/McCloudS/subgen) with the patches
[subarr](https://github.com/coaxk/subarr) requires.

## What this is

Subarr orchestrates [subgen](https://github.com/McCloudS/subgen) to drive
Whisper-based subtitle generation across your library. To do that cleanly it
needs three capabilities upstream subgen doesn't ship:

- **`POST /batch`** — bulk scan with one `scan_id` wrapping N files, used by
  subarr's scan runner to submit prioritised gap lists in one round-trip.
- **`GET /queue`** — queue introspection (type-tracked, dedup-aware) used by
  subarr's header counter, completion watcher, and activity feed.
- **Per-language `SUBGEN_KWARGS_LANG_XX` env-var overrides** — applied in
  subgen's transcribe pipeline so each language can use its own
  faster-whisper kwargs.

Plus a handful of smaller fixes (eager model load on boot, reverse-sort in
`transcribe_existing`, structured response shapes).

This repo follows the **distro patch-stack** pattern: vanilla upstream subgen
is a git submodule pinned to a specific commit; our changes live as discrete
`.patch` files in `patches/`; a build script applies them and produces a
docker image we publish to GHCR.

## Quickstart

```yaml
# compose.yaml
services:
  subgen:
    image: ghcr.io/coaxk/subarr-subgen:2026.05.3-r1
    # ... rest is identical to upstream mccloud/subgen ...
```

That's it. Same env-vars as upstream, same volumes, same ports. Drop-in
replacement.

**One variable upstream does not have: `SUBGEN_PATH_ALLOWLIST`.** The
path-accepting endpoints (`/batch ?directory=`, `/asr ?path=`,
`/detect_language_robust ?path=`) answer unauthenticated on your LAN, so this
image only serves paths under a colon-separated allowlist, default `/media`
(patch 0025). If your media is mounted at `/data` inside the container, set
`SUBGEN_PATH_ALLOWLIST=/data`, or every request is refused with 403
`directory is outside the allowed media root`. An empty value disables the
guard; do not do that on a network you share.

If you don't run subarr, you probably don't need this. Use upstream
`mccloud/subgen:latest` instead.

### One difference from upstream: this image never self-updates

**`UPDATE`, `LAUNCHER_UPDATE`, `BRANCH` and `--install` are ignored here.**
Upstream's launcher re-downloads `subgen.py` from McCloudS/subgen every time the
container starts, and `--install` re-installs upstream's `requirements.txt` over
the pinned dependency set. This image ships a *patched* `subgen.py` and pins its
dependencies at build time, so either would replace what you pulled: the first
swaps every patch for vanilla code while the image tag still claims a patch rev,
the second moves faster-whisper / ctranslate2 / stable-ts out from under the
baked `SUBGEN_KWARGS` and the CUDA build they were chosen against. The
symptom of the first is a `404` on `/queue` and subarr quietly dropping to compat mode with no
explanation ([#59](https://github.com/coaxk/subarr-subgen/issues/59)).

This matters most if you are **migrating an existing upstream compose file**,
because `UPDATE=True` is a common setting to carry across. Nothing breaks if you
leave it set: the container logs a notice naming the setting and runs the correct
baked code anyway. Remove it to silence the notice.

To move to a newer upstream, pull a newer image tag. To run vanilla subgen, use
the upstream image.

## Patches included

| # | Patch | What it adds |
|---|---|---|
| 0001 | `per-language-kwargs` | Reads `SUBGEN_KWARGS_LANG_<CODE>` env vars and merges into transcribe kwargs per detected language |
| 0002 | `eager-model-load` | Loads the Whisper model on container start (default upstream loads lazily on first request) |
| 0003 | `reverse-sort-transcribe-existing` | Adds `?reverse=true` to bulk scans so newest files transcribe first |
| 0004 | `batch-structured-response` | `/batch` returns `{scan_id, dispatched, skipped, errors}` JSON instead of plain text |
| 0005 | `gsq-return-str` | `gen_subtitles_queue` returns a string scan_id used by `/batch` |
| 0006 | `fastapi-jsonresponse-import` | Widens FastAPI import to include `JSONResponse` |
| 0007 | `deduplicated-queue-type-tracking-and-queue-endpoint` | `DeduplicatedQueue` tracks `(path, type)` instead of bare paths; adds `GET /queue` route |

See [`patches/`](./patches/) for the full diffs.

## Pinning + upgrades

Image tags follow `<upstream_version>-r<patch_rev>`:

- `2026.05.3-r1` — first build against upstream subgen 2026.05.3, patch revision 1
- `:latest` — newest release (any stability)
- `:stable` — manually promoted after ≥7 days in `:latest` without reported regression

Always pin to a specific `-r<N>` tag in production. See [`docs/tagging.md`](./docs/tagging.md).

## Reporting issues

- **Is the bug in subgen itself** (transcription quality, language detection, whisper model loading)? File at [McCloudS/subgen](https://github.com/McCloudS/subgen/issues).
- **Is the bug in one of our patches** (`/batch` shape wrong, `/queue` returning bad data, per-lang kwargs not being honoured)? File here.
- **Is the bug in how subarr drives subgen**? File at [coaxk/subarr](https://github.com/coaxk/subarr/issues).

If you're unsure, file here and we'll redirect.

## Building from source

```bash
git clone --recursive https://github.com/coaxk/subarr-subgen
cd subarr-subgen
./scripts/build.sh
```

Produces a local `subarr-subgen:dev` image.

## How the rebase-test works

Every Monday at 06:00 UTC a workflow fetches the latest upstream subgen from
`McCloudS/subgen` and tries to apply our patches against it. If everything
applies clean and smoke tests pass, we open a PR to bump the submodule pin.
If a patch fails to apply, a sticky GitHub issue is opened so we know
upstream restructured something we depend on. See [`docs/upstream-sync.md`](./docs/upstream-sync.md).

## Dependabot action bumps go red on a comment, not a problem

Dependabot regularly bumps a pinned action's SHA but writes the wrong version
comment next to it (it has moved a pin to v3.0.2's commit while labelling it
`# v3.0.1`). zizmor's `ref-version-mismatch` catches that and reds CI on
basically every actions-group bump. The pin itself is fine; only the comment
lies.

Fix it with:

```bash
GH_TOKEN=$(gh auth token) python scripts/fix-action-pins.py
```

**Do not just run `zizmor --fix=unsafe-only`.** It works, but it also strips
`# zizmor: ignore[...]` suppressions off the lines it rewrites, silently - it
has eaten the `superfluous-actions` suppression on the release step twice. The
script runs the same auto-fix, then puts the suppressions back, and **refuses
the whole result if any pinned SHA actually moved**, so a comment tidy-up can
never quietly become a supply-chain change.

## License + attribution

This repo's tooling (build scripts, patches, CI workflows) is MIT licensed.

The upstream subgen code itself remains under its own license; see
[`NOTICE`](./NOTICE) for attribution. We don't redistribute upstream source
in this repo — only patches against it. The published docker image contains
upstream code + our patches as derived work, per upstream's license terms.
