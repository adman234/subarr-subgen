# Project summary: local speech-to-text on Unraid (Home Assistant + subtitles)

Last updated 2026-11-07. Covers the chat that started 2026-10-07 plus the
fork commits made since (up to `6a4a941` on `adman234/subarr-subgen` main).

## Goal

Replace two GPU Whisper services on the Unraid server with NVIDIA **Parakeet
TDT 0.6B v3** running on the **CPU**, using **no VRAM**:

1. **Home Assistant voice** (was `wyoming-faster-whisper`): now the
   **wyoming-onnx-stt** container.
2. **Subtitle generation for Bazarr** (was `subarr-subgen`, i.e. subgen with
   Whisper `large-v3-turbo` on the GPU): now the fork
   **adman234/subarr-subgen** with a Parakeet backend.

Server: i5-13600KF (6 P-cores / 12 threads + 8 E-cores), RTX 5060 Ti 16 GB.
CPU pinning is set manually in Unraid (plan was E-cores, CPUs 12-19).

---

## Part 1: Home Assistant STT (wyoming-onnx-stt)

**Upstream:** https://github.com/lmoe/wyoming-onnx-stt (image
`lmo3/wyoming-onnx-stt:latest`, v0.1.0, CUDA 12.6 base, ~4.5 GB).

**Template:** `unraid/wyoming-onnx-stt.xml` in `adman234/-other` (main).

```bash
wget -O /boot/config/plugins/dockerMan/templates-user/my-wyoming-onnx-stt.xml \
  https://raw.githubusercontent.com/adman234/-other/main/unraid/wyoming-onnx-stt.xml
```

**Settings in use (CPU, int8):**

| Field | Value |
|---|---|
| Model (`STT_MODEL`) | `nemo-parakeet-tdt-0.6b-v3` |
| Device (`STT_DEVICE`) | `cpu` |
| Post Arguments | `--quantization int8` |
| Extra Parameters | empty (removed `--runtime=nvidia`) |
| NVIDIA_* variables | removed |
| VRAM patch path mapping | removed (only affects GPU mode) |
| Port | host 10301 → container 10300 |
| Zeroconf (`STT_ZEROCONF`) | blank (bridge mode would advertise the Docker IP) |
| Warmup | true |
| CPU pinning | E-cores (set manually) |

Home Assistant: Wyoming Protocol integration, host = Unraid IP, port 10301.

**Findings along the way:**
- *Failed to connect* at first was the model still downloading. The port only
  opens after download + load + warmup.
- **Port mapping:** only the *host* port may change. The container port must
  stay 10300.
- **Quantization in Post Arguments:** a blank `STT_QUANTIZATION` crashes the
  app. Unraid passes `''`, and pydantic rejects it for `Literal["int8","fp16"]`.
  So quantization is set via Post Arguments instead.
- **VRAM:** GPU fp32 used 4.2 GB, which the user rejected. I wrote a patched
  `handler.py` that sets the ONNX Runtime CUDA arena to `kSameAsRequested` and
  heuristic cuDNN search, plus an optional `STT_CUDA_MEM_LIMIT_MB`. It's in
  `unraid/wyoming-onnx-stt-patch/` in `-other` and mounted over the app's
  `handler.py`. It was abandoned in favour of CPU + int8, so it's not in use.
- **FP16 caveat:** fp16 only loads if the model's Hugging Face repo publishes
  fp16 files. I couldn't verify that for Parakeet v3; I believe it only ships
  fp32 and int8.
- **Microphone in the HA web UI on Edge:** the mic needs HTTPS. Either
  `edge://flags/#unsafely-treat-insecure-origin-as-secure` → add
  `http://<ha-ip>:8123` → Enabled, or serve HA over HTTPS. Then allow the site
  microphone permission, plus Windows microphone privacy settings.

**Estimates given (not measured):**

| Option | Latency, 3-5 s command | WER (English) | VRAM |
|---|---|---|---|
| fp32 + patch (GPU) | ~50-100 ms | ~6.3% | ~3 GB |
| int8 + patch (GPU) | ~80-300 ms | ~6.4-6.6% | ~1-1.5 GB |
| **CPU + int8 (chosen)** | ~0.3-1 s (8 E-cores: rtf ~0.08-0.2) | ~6.4-6.6% | 0 |
| old faster-whisper large-v3-turbo int8_float16 | ~200-500 ms | ~7.5-8% | ~1.5-2 GB |

---

## Part 2: Subtitles (fork of subarr-subgen with Parakeet)

### Background
- **subgen** (McCloudS/subgen) is the transcription server Bazarr's Whisper
  provider talks to.
- **subarr-subgen** (coaxk) is subgen plus 50+ patches that the subarr GUI
  needs. It's a *patch-stack* repo: upstream subgen is a git submodule in
  `upstream/`, the changes are `.patch` files in `patches/` listed in
  `patches/series`, and `scripts/apply-patches.sh` applies them.
- Bazarr's Whisper provider calls:
  - `POST /asr?task=transcribe&language=xx&output=srt&encode=false` with raw
    16 kHz s16le PCM;
  - `POST /detect-language` when the audio language is unknown.

### Repo and branches: https://github.com/adman234/subarr-subgen

The user forked it on GitHub; Claude can't create repos or forks from the
session.

| Branch | Image | Dockerfile | Notes |
|---|---|---|---|
| `main` | `ghcr.io/adman234/subarr-subgen:latest` (+ `:cpu-<sha7>`) | `docker/Dockerfile.cpu` (ubuntu:22.04, CPU torch 2.13.0 / torchaudio 2.11.0) | lean, no CUDA. **This is the one in use.** |
| `gpu` | `ghcr.io/adman234/subarr-subgen:gpu` (+ `:gpu-<sha7>`) | `docker/Dockerfile` (upstream's CUDA 12.9 base, cu129 torch) | Whisper detection/fallback on the GPU; Parakeet still on CPU |

`docker/Dockerfile` (CUDA) stays unchanged on `main` too, so upstream merges
stay clean; only the `gpu` branch builds it. The GHCR package was set to
**Public** so Unraid can pull without credentials.

### What the fork adds (patches after upstream's numbers)
- **`0064-parakeet-backend.patch`** (originally 0058; renumbered on each
  upstream merge):
  - `upstream/parakeet_backend.py` (new): `ParakeetHybridModel` replaces
    subgen's global `model`.
  - `transcribe()` returns a stable-ts `WhisperResult`, so subgen's
    regrouping, SRT/VTT/LRC writing and offsets work unchanged.
  - Words are built from onnx-asr token timestamps, VAD-segmented by Silero.
  - **Language:** the caller's language (Bazarr `?language=`, metadata) is
    always used. Only when missing does `WHISPER_MODEL` detect it, with
    faster-whisper `detect_language` over up to ten 30 s windows of
    `DETECT_LANGUAGE_LENGTH`.
  - **Fallback:** non-Parakeet languages and `task=translate` go to
    `WHISPER_MODEL` (`PARAKEET_FALLBACK=whisper`) or fail (`none`).
  - **Loading/unloading:** lazy-loads Parakeet and Whisper separately;
    `model.model.unload_model()` frees both, so `CLEAR_VRAM_ON_COMPLETE` +
    `MODEL_CLEANUP_DELAY` unload from RAM.
  - **Model storage:** Parakeet and Silero go to
    `MODEL_PATH/onnx-asr/<name>[-int8]`. Downloads go to a `.partial` dir and
    get a `.subgen-complete` marker, because onnx-asr treats any existing dir
    as offline.
  - **Decoding:** audio is decoded via the `ffmpeg` binary, not PyAV.
  - **subgen.py changes:** `TRANSCRIBE_BACKEND`, `EAGER_MODEL_LOAD` (default
    false, which makes patch 0002's startup load opt-in), a
    `_detect_language_only()` helper at the three detect sites, and a startup
    log line. Also adds `onnx-asr[cpu,hub]` to requirements.
- **`0065-parakeet-monotonic-word-timestamps.patch`** (later session): the TDT
  decoder could stamp repeated tokens with the same frame. Then stable-ts
  raised `UnsortedException` and Bazarr reported "Downloaded subtitles isn't
  valid". Words now never start before the previous word's end.
- **`0066-parakeet-bound-onnx-memory.patch`** (later session): defaults to
  `PARAKEET_BATCH_SIZE=1` and `PARAKEET_CPU_MEM_ARENA=false`, to lower peak
  RAM. I had estimated ~3 GB per movie job before this patch; not measured.
- **Other fork-only files:**
  - `docker/Dockerfile.cpu`;
  - `docker/constraints.txt` (`onnx-asr==0.12.0`) and the Dockerfile `COPY`
    of `parakeet_backend.py`;
  - `scripts/parakeet-smoke.py` (real-model test on the JFK clip);
  - `tests/test_parakeet_backend.py` (fake-model unit tests);
  - `.github/workflows/fork-image.yml`;
  - `unraid/*.xml` and `unraid/icon.{png,svg}`;
  - README section;
  - `image-staleness.yml` / `sticky-issue.sh` changed to check this fork's
    image and create missing labels.

### Settings and environment variables

| Variable | Default (code) | Template (CPU) | Meaning |
|---|---|---|---|
| `TRANSCRIBE_BACKEND` | `whisper` | `parakeet` | backend |
| `PARAKEET_MODEL` | `nemo-parakeet-tdt-0.6b-v3` | same | v2 = English-only |
| `PARAKEET_QUANTIZATION` | `int8` | `int8` | `none` = fp32 |
| `PARAKEET_THREADS` | `0` | `0` | 0 = one per pinned CPU (`sched_getaffinity`) |
| `PARAKEET_FALLBACK` | `whisper` | `whisper` | or `none` |
| `PARAKEET_LANGUAGES` | model default | unset | override the language list |
| `PARAKEET_MAX_SEGMENT_S` | `20` | unset | VAD max segment |
| `PARAKEET_BATCH_SIZE` | `1` | unset | segments per encoder call |
| `PARAKEET_CPU_MEM_ARENA` | `false` | unset | ORT CPU arena |
| `PARAKEET_VAD_THRESHOLD` / `_MIN_SILENCE_MS` / `_SPEECH_PAD_MS` | `0.5` / `300` / `100` | unset | Silero VAD |
| `EAGER_MODEL_LOAD` | `false` | `false` | load at startup |
| `WHISPER_MODEL` | `medium` | `small` | detection + fallback only |
| `TRANSCRIBE_DEVICE` / `COMPUTE_TYPE` | `cpu` / `auto` | `cpu` / `int8` | Whisper device |
| `CONCURRENT_TRANSCRIPTIONS` | `2` | `1` | best on CPU |
| `DETECT_LANGUAGE_LENGTH` | `30` | `300` | seconds sampled when no language given |
| `CLEAR_VRAM_ON_COMPLETE` / `MODEL_CLEANUP_DELAY` | `true` / `30` | `true` / `600` | unload when idle |
| `PROCESS_ADDED_MEDIA` / `PROCESS_MEDIA_ON_PLAY` | `true` / `true` | `false` / `false` | Plex/Jellyfin triggers off (Bazarr only); Plex vars left in, blank |
| `SUBGEN_PATH_ALLOWLIST` | `/media` | `/data/media` | **must cover the media mount, or every request returns 403** |
| `WEBHOOKPORT` + port | 9000 | 9006 / 9006 | |

Paths: `/mnt/user/data/media/` → `/data/media`; `/mnt/cache/appdata/subgen/` →
`/subgen/models` (Parakeet ~650 MB lands in `onnx-asr/`).

GPU template differences: `:gpu` image, `--runtime=nvidia`, `NVIDIA_*` vars,
`TRANSCRIBE_DEVICE=cuda`, `WHISPER_MODEL=large-v3-turbo`,
`COMPUTE_TYPE=int8_float16`.

### Install and update on Unraid
```bash
wget -O /boot/config/plugins/dockerMan/templates-user/my-subarr-subgen-parakeet.xml \
  https://raw.githubusercontent.com/adman234/subarr-subgen/main/unraid/subarr-subgen-parakeet.xml
# icon cache (if the old icon sticks):
rm -f /var/lib/docker/unraid/images/subarr-subgen-parakeet-icon.png
```
- **Bazarr:** set the Whisper provider endpoint to `http://<unraid-ip>:9006` and
  the transcription timeout to about `10800`.
- **Healthy log lines:**
  - `Loading Parakeet model nemo-parakeet-tdt-0.6b-v3 (quantization=int8, threads=N)`
  - `Parakeet transcription complete: …`
  - ~10 min after the queue empties: `Parakeet backend: models unloaded`

### CI / build
- **`fork-image.yml`** runs on push to `main` or `gpu`, and on manual dispatch:
  - **smoke job:** apply patches → validate → pytest → real-model smoke test.
    It downloads Parakeet int8 + Whisper tiny and transcribes JFK; the first
    real run took 12 s including download.
  - **image job:** build and push to GHCR with a per-variant registry
    cache (`:buildcache-cpu` / `:buildcache-gpu`).
- **Upstream's workflows** (`PR`, `security`, `codeql`, …) also run on the
  fork. `release.yml` only runs on tags and still names coaxk images; it's
  unused.
- **Local checks before pushing:**
  ```bash
  bash scripts/apply-patches.sh && python3 scripts/validate-patched.py && python3 -m pytest -q
  ```
  plus `zizmor --persona=regular .github/workflows/`.

---

## Notes for continuing

1. **The `gpu` branch is 2 commits behind `main`.** `main` is at `6a4a941`,
   `gpu` at `545be3e`. The `:gpu` image therefore **lacks the 0065
   UnsortedException fix and the 0066 memory bounds**. Fix:
   `git push origin origin/main:refs/heads/gpu` (fast-forward), which rebuilds
   `:gpu`. Keep `gpu` == `main` going forward; the workflow picks the
   Dockerfile from the branch name.
2. **Merging a new upstream subarr-subgen release:**
   - fetch `coaxk/subarr-subgen`, then merge into `main`;
   - renumber the Parakeet patches (0064-0066) past upstream's newest number,
     in both the file names and `patches/series`;
   - re-run apply / validate / pytest;
   - update `scripts/parakeet-smoke.py` and the test path references if they
     name the patch;
   - push `main`, then fast-forward `gpu`.
   - If upstream changes torch/torchaudio versions, base image or the layer
     order in `docker/Dockerfile`, **mirror it in `docker/Dockerfile.cpu`**:
     same pins, `/whl/cpu` index, `+cpu` asserts.
3. **Keep `UPDATE=false`.** The image ignores `UPDATE` anyway (subarr-subgen
   patch 0044); upstream self-update would drop all patches.
4. **Parakeet v3 languages:** bg hr cs da nl en et fi fr de el hu it lv lt mt
   pl pt ro sk sl es sv ru uk. Anything else goes to Whisper `small` on CPU,
   which is slow and lower quality. Parakeet can't translate, and turbo
   translates poorly anyway. Parakeet auto-detects per segment and may
   occasionally pick a close language.
5. **Whisper-only subarr features** (`SUBGEN_KWARGS`, per-language kwargs and
   prompts, safe-decode preset, model swap via `/config`) only affect the
   Whisper detection/fallback path under the Parakeet backend.
6. **Known upstream quirk (not fixed):**
   - `asr_task_worker` reads `task_data['output_format']`, but `/asr` stores
     `output`, so `/asr` always returns SRT. Bazarr asks for SRT, so it's
     harmless.
   - `/detect_language_robust` uses faster-whisper's `decode_audio` (PyAV).
     It works with the pinned `av==17.1.0`, but breaks on newer PyAV.
7. **CPU contention:** subgen and wyoming-onnx-stt both pinned to the E-cores
   means voice commands slow down while a movie transcribes. Options: pin
   subgen elsewhere, or accept it.
8. **Possible future ideas (discussed, not built):**
   - a Wyoming server inside the subgen fork (one container, one shared
     model, voice priority between VAD segments);
   - running Parakeet on the GPU in the `:gpu` image (needs
     `onnxruntime-gpu`, which conflicts with faster-whisper's `onnxruntime`);
   - contributing a `TRANSCRIBE_BACKEND=parakeet` PR to coaxk/subarr-subgen
     so upstream maintains it.
9. **Housekeeping in `adman234/-other`:**
   - the wyoming template's `<TemplateURL>` and the patch `wget` URL still
     point at branch `claude/festive-bohr-ssj3f6`, not `main`. Raw still
     served 200 on 2026-11-07; repoint to `main` to be safe.
   - a stray `unraid/wyoming-onnx-stt-patch/__pycache__/handler.cpython-313.pyc`
     got committed on `main` and can be deleted.
10. **Session limits seen:**
    - Hugging Face downloads are blocked from the Claude cloud container, so
      real-model tests have to run in GitHub Actions;
    - creating repos or forks via the API is denied: the user forks manually,
      then the session attaches it with push access;
    - `download.pytorch.org` is unreachable locally (CI is fine).

## Handy commands
```bash
docker stats --no-stream subarr-subgen-parakeet wyoming-onnx-stt   # RAM check (idle should drop after unload)
docker ps -a --filter publish=9006 --format 'table {{.Names}}\t{{.Ports}}'
ss -ltnp | grep ':9006'
echo '{"type":"describe"}' | nc -w 3 <unraid-ip> 10301           # Wyoming probe
# containers by template creation date (≈ install date), newest first
for f in /boot/config/plugins/dockerMan/templates-user/my-*.xml; do printf '%s\t%s\n' "$(stat -c %w "$f")" "$(basename "$f" .xml | sed 's/^my-//')"; done | sort -r
```
