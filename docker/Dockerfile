# subarr-subgen — patched McCloudS/subgen image.
#
# Layer structure mirrors upstream's Dockerfile (apt + torch + pip in separate
# layers for cache efficiency). NOTE: this image bumps the CUDA base to 12.8
# (upstream is still on 12.3) — see the FROM line. Build context is build/,
# which scripts/build.sh produces by:
#
#   1. cp -a upstream/. build/   (vanilla source tree)
#   2. apply patches in build/   (via apply-patches.sh in upstream/, then
#      build.sh recopies the patched tree)
#
# So when this Dockerfile COPYs subgen.py, it's the patched one.
#
# Differs from upstream Dockerfile by:
#   - CUDA base bumped 12.3 -> 12.8 (Blackwell/sm_120 support — see below)
#   - adding OCI image labels for version/source provenance
#   - taking UPSTREAM_VERSION + PATCH_REV build-args
#
# Same deps, same entrypoint, same CMD — drop-in.

# CUDA 12.8 base (was 12.3.2-base): CUDA 12.3's cuBLAS has no sm_120 kernels, so
# every transcription fails with CUBLAS_STATUS_NOT_SUPPORTED on NVIDIA Blackwell
# / RTX 50xx GPUs (#163) — CTranslate2 4.5+ has the Blackwell kernels but the
# cuBLAS underneath was too old. 12.8 ships sm_120 cuBLAS; the cudnn-runtime
# flavour also bundles cuBLAS + cuDNN at the matching version (the old -base
# relied on whatever cuBLAS happened to be present). Kept on ubuntu22.04
# (python3.10) to minimise the blast radius.
# [#27] 12.8 -> 12.9: the torch fix for CVE-2025-3000 (torch 2.13) is only
# published on the cu129 index and up; cu128 stops at 2.11. The base CUDA
# version must match the wheel index (tests/test_dockerfile_layers.py).
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04

# [#63] LAYER ORDER IS A DOWNLOAD-SIZE DECISION. Every layer below a changed
# instruction is rebuilt and becomes a NEW blob users must pull. Until r10 the
# release tag was declared at the top of this file, so every release rebuilt
# apt and the 4 GB torch layer and users re-downloaded ~4.4 GB for a one-line
# patch (measured on GHCR: r8->r9 and r9->r10 both). Order is now: stable and
# heavy first (packages, pinned torch, requirements), the per-build security
# upgrade in its own small layer, app files, and release metadata LAST.
# tests/test_dockerfile_layers.py locks this.

# Apt packages — installed, NOT upgraded here. The upgrade lives in its own
# later layer so refreshing security patches never rebuilds torch.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg python3 python3-pip curl gosu tzdata

# Torch — 4 GB and rarely changes. PINNED: an unpinned `-U` moved torch silently
# and produced a new 4 GB blob on every rebuild. Bump deliberately, in its own
# change, and run scripts/torch-gate.py against the result.
#
# [#27] torch 2.13.0 (fixes CVE-2025-3000) with torchaudio 2.11.0, both from
# cu129. torchaudio's last release is 2.11, so it cannot follow torch's version;
# it MUST follow torch's CUDA build, or it refuses to import ("compiled with
# different CUDA versions") and takes stable_whisper and subgen with it.
# `--no-deps` stops pip resolving torchaudio's own torch pin back to 2.11.
# Gate run 2026-09-17: resample bit-identical, transcripts identical to r10.
#
# setuptools: torch 2.13 requires `setuptools>=77.0.3` (2.11 did not), and the
# pytorch index only mirrors 78.1.0, which trivy flags HIGH three times over:
# CVE-2025-47273 in setuptools itself, plus its vendored jaraco.context 5.3.0
# (CVE-2026-23949) and wheel 0.45.1 (CVE-2026-24049). setuptools 82+ vendors
# the fixed jaraco.context 6.1.0 and wheel 0.46.3; pin 84.0.0 from PyPI in the
# same layer so the vulnerable copy never ships.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129 \
    && python3 -m pip install --no-deps torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu129 \
    && python3 -m pip install setuptools==84.0.0

# App dependencies — only rebuilds when requirements.txt changes. Upstream's
# requirements do not list torch, and pip's default only-if-needed strategy
# leaves the pinned torch alone.
#
# [#73] CONSTRAINED. Upstream's requirements.txt is nine bare names with no
# versions, so this layer used to resolve against whatever PyPI held that day.
# On 2026-09-24 the index moved and pip's resolvelib graph weighting began
# crashing on the result — `assert len(weights) == expected_node_count`, after
# every download had finished — so every UNCACHED build failed. pr.yml and
# security.yml went red, meaning trivy scanned nothing for two days, while
# release.yml stayed green purely on a registry cache hit for this layer. A
# release path that only works while a cache is warm fails exactly when you
# want to cut a release.
#
# constraints.txt is read out of a published, verified image rather than a fresh
# resolve (scripts/gen-constraints.sh), so what is pinned is what has actually
# run. ⚠ The COPY below is load-bearing as a GUARD: the context is staged in
# two places (scripts/build.sh and release.yml's inline stage step), and a COPY
# of a missing file is a hard build failure — so a staging site that forgets
# constraints.txt cannot quietly go back to building unpinned.
COPY requirements.txt constraints.txt /
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -U -r /requirements.txt -c /constraints.txt

# [#27] Refuse to build with a torch pair that cannot run subgen. pip does not
# catch a CUDA-build mismatch, and the requirements step above could in
# principle move either package. Importing both is the check torchaudio itself
# performs; the suffix asserts name the cause if it ever fails.
RUN python3 -c "import torch, torchaudio, stable_whisper, setuptools; \
v, a, st = torch.__version__, torchaudio.__version__, setuptools.__version__; \
print('torch', v, 'torchaudio', a, 'stable_whisper', stable_whisper.__version__, 'setuptools', st); \
assert v.startswith('2.13.0+cu129'), v; \
assert a.startswith('2.11.0+cu129'), a; \
assert st == '84.0.0', st"

# Security upgrade — its own layer, AFTER the heavy ones. `apt-get upgrade`
# pulls base-image security patches (the cuda:12.8 base shipped gnupg
# 2.2.27-3ubuntu2.1, flagged HIGH by trivy as CVE-2025-68973). With a
# persistent build cache this RUN would otherwise be a cache hit for days and
# silently stop shipping patches (the subarr GHA-cache lesson). APT_REFRESH is a
# date passed by every build; the `echo` that REFERENCES it is what makes a new
# date a cache miss. It looks like dead weight. It is not.
ARG APT_REFRESH=0
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    echo "apt-refresh=${APT_REFRESH}" \
    && apt-get update && apt-get upgrade -y \
    && apt-get purge -y --auto-remove python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /subgen

# App files last — changes here don't bust the layers above.
# subgen.py is the PATCHED version (apply-patches.sh has already run).
COPY launcher.py subgen.py language_code.py /subgen/

RUN mkdir -p /cache && chmod 777 /cache

ENV XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    MPLCONFIGDIR=/cache/matplotlib \
    PYTHONUNBUFFERED=1

# [r8] Bake the subarr-tuned subtitle segmentation as the default. "v2-strongpad"
# won the #168 regroup arena (4 audio langs x 6 configs, full files) and held on
# held-out languages (de/sr): median CPS ~13-15, %>25 CPS ~8-14%, zero sub-0.5s
# micro-cues. Overridable at runtime — set CUSTOM_REGROUP="" to disable, or to
# your own DSL. (subgen reads CUSTOM_REGROUP from the environment.)
ENV CUSTOM_REGROUP="cm_sp=.* /。/?/？_sg=.5_mg=.3++84_p=.3+.7+1.5_sl=42++++++1"

# [r9] Bake the Phase-1 kwargs research as the image default. These are the
# hardware-INDEPENDENT anti-failure settings (hallucination/looping fixes,
# each issue-cited in coaxk/subarr-subgen docs): temperature ladder re-enables
# Whisper's fallback, log_prob -0.8 tightens the hallucination gate,
# length_penalty 1.0 stops hallucinated bursts being cheaper than full
# transcriptions (faster-whisper #569), VAD timing fixes over-merging
# (faster-whisper #477). Overridable: any SUBGEN_KWARGS you set wins.
# Contested knobs (condition_on_previous_text=false, VAD timing) are called
# out in the r9 release notes.
#
# [r10] VAD threshold 0.5 -> 0.35 (subarr#543). At 0.5 the Silero pre-filter cut
# whispered and under-score dialogue before Whisper saw it. #171 Phase 2 corpus,
# baseline run twice byte-identical (zero noise): authored-subtitle recall
# 72.63% -> 74.18%, share of captioning outside authored subs flat at ~24.3%,
# 0.30 bought nothing more. no_speech_threshold was ruled out (0.6 -> 0.4 gave
# byte-identical output). Locked by tests/test_baked_kwargs.py.
ENV SUBGEN_KWARGS='{"no_speech_threshold": 0.6, "vad_filter": true, "vad_parameters": {"threshold": 0.35, "min_speech_duration_ms": 250, "min_silence_duration_ms": 500, "speech_pad_ms": 600}, "condition_on_previous_text": false, "beam_size": 5, "patience": 1.0, "length_penalty": 1.0, "repetition_penalty": 1.05, "no_repeat_ngram_size": 3, "compression_ratio_threshold": 2.4, "log_prob_threshold": -0.8, "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}'

# [r9] The one per-language override that survived the 17-override prune
# (JA subtitle convention plausibly matches the longer length_penalty).
ENV SUBGEN_KWARGS_LANG_JA='{"patience": 1.0, "length_penalty": 1.3}'

# [#63] Release metadata LAST. A changed ARG is a cache miss for every later
# instruction, so declaring these at the top rebuilt every layer on every
# release. Here they touch nothing but metadata.
ARG UPSTREAM_VERSION=unknown
ARG PATCH_REV=unknown
# [r9 / subarr#224] The GHCR release tag this image was built from
# (e.g. v2026.05.3-r9). Emitted in the /queue caps body as
# subarr_subgen_release_tag so subarr's update check compares
# like-for-like instead of patch-rev vs date-tag (which made every
# install show a permanent fake "update available").
ARG RELEASE_TAG=dev
ENV SUBARR_SUBGEN_RELEASE_TAG="${RELEASE_TAG}"

LABEL org.opencontainers.image.title="subarr-subgen"
LABEL org.opencontainers.image.description="McCloudS/subgen + subarr-required patches"
LABEL org.opencontainers.image.source="https://github.com/coaxk/subarr-subgen"
LABEL org.opencontainers.image.licenses="MIT"
LABEL subarr.subgen.upstream_version="${UPSTREAM_VERSION}"
LABEL subarr.subgen.patch_rev="${PATCH_REV}"
LABEL subarr.subgen.release_tag="${RELEASE_TAG}"
LABEL subarr.subgen.upstream_source="https://github.com/McCloudS/subgen"

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "launcher.py"]
