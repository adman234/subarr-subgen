#!/usr/bin/env python3
"""Real-model smoke test for the Parakeet backend (patch 0058).

Downloads Parakeet TDT 0.6B v3 (int8), Silero VAD and Whisper tiny into
MODEL_ROOT, then transcribes a known English clip:

    python scripts/parakeet-smoke.py <model_root> <english.wav>

Run after scripts/apply-patches.sh, which puts parakeet_backend.py in upstream/.
"""

import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "upstream"))

import stable_whisper  # noqa: E402

from parakeet_backend import ParakeetHybridModel  # noqa: E402

def words_only(text: str) -> str:
    """Lowercase, punctuation-free text: Parakeet's punctuation varies with quantization."""
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


REGROUP = "cm_sp=.* /。/?/？_sg=.5_mg=.3++84_p=.3+.7+1.5_sl=42++++++1"


def main() -> None:
    root, wav = sys.argv[1], sys.argv[2]

    def whisper_loader():
        return stable_whisper.load_faster_whisper("tiny", download_root=root, device="cpu", compute_type="int8")

    model = ParakeetHybridModel(whisper_loader=whisper_loader, model_root=root, detect_seconds=300)

    t0 = time.time()
    progress = []
    result = model.transcribe(wav, language="en", task="transcribe", regroup=REGROUP,
                              progress_callback=lambda s, t: progress.append((s, t)), beam_size=5)
    srt = result.to_srt_vtt(filepath=None, word_level=False)
    print(srt)
    print(f"transcribed in {time.time() - t0:.1f}s (incl. download), language={result.language}")
    text = words_only(result.text)
    assert "ask not what your country can do for you" in text, result.text
    assert result.language == "en"
    assert srt.startswith("1\n00:00:0"), srt[:40]
    assert all(w.end >= w.start for s in result.segments for w in s.words)
    assert progress and progress[-1][0] == progress[-1][1], progress
    assert model._whisper is None, "Whisper must not load when the caller supplies the language"

    det = model.detect_only(wav)
    print(f"detect_only: {det}")
    assert det.language == "en", det

    auto = model.transcribe(wav, language=None)
    assert auto.language == "en" and "country" in auto.text.lower(), (auto.language, auto.text)

    fallback = model.transcribe(wav, language="en", task="translate", regroup=True)
    print(f"translate fallback via Whisper: {fallback.text.strip()[:80]}")

    model.model.unload_model()
    assert model._asr is None and model._whisper is None

    # Second load must come from the completed local copy, not a new download.
    marker = os.path.join(root, "onnx-asr", "nemo-parakeet-tdt-0.6b-v3-int8", ".subgen-complete")
    assert os.path.isfile(marker), marker
    t1 = time.time()
    again = model.transcribe(wav, language="en")
    assert "country" in again.text.lower()
    print(f"reload + transcribe from cache: {time.time() - t1:.1f}s")
    print("PARAKEET SMOKE: ok")


if __name__ == "__main__":
    main()
