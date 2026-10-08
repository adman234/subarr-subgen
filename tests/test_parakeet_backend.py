"""Unit tests for upstream/parakeet_backend.py (patch 0058), with fake models.

Skipped where numpy / stable-ts are not installed (the PR workflow's pytest
step installs pytest only); scripts/parakeet-smoke.py covers the real models.
"""

import importlib.util
import pathlib
import sys

import pytest

np = pytest.importorskip("numpy")

BACKEND = pathlib.Path(__file__).resolve().parent.parent / "upstream" / "parakeet_backend.py"
if not BACKEND.exists():
    pytest.skip("run scripts/apply-patches.sh first", allow_module_level=True)

spec = importlib.util.spec_from_file_location("parakeet_backend", BACKEND)
pb = importlib.util.module_from_spec(spec)
sys.modules["parakeet_backend"] = pb
spec.loader.exec_module(pb)


class _Seg:
    def __init__(self, start, end, text, tokens, timestamps):
        self.start, self.end, self.text = start, end, text
        self.tokens, self.timestamps = tokens, timestamps
        self.logprobs = [-0.1] * len(tokens)


class _FakeAsr:
    def recognize(self, wave, sample_rate):
        return iter([_Seg(0.5, 3.0, "Hello there.", [" Hello", " there", "."], [0.0, 0.4, 0.8])])


class _FakeWhisper:
    loaded = 0

    def __init__(self):
        _FakeWhisper.loaded += 1

    def detect_language(self, audio=None, **kwargs):
        return ("de", 0.9, [])

    def transcribe(self, audio, **kwargs):
        return ("whisper", kwargs.get("language"), kwargs.get("task"))


def _model():
    _FakeWhisper.loaded = 0
    m = pb.ParakeetHybridModel(whisper_loader=_FakeWhisper, model_root="/nonexistent", detect_seconds=300)
    m._asr = _FakeAsr()
    return m


def test_words_group_subword_tokens_and_offset_by_segment():
    words = pb.words_from_segment(10.0, 14.0, [" Hello", ",", " wor", "ld", "."], [0.0, 0.4, 0.8, 1.0, 1.3], None)
    assert [w["word"] for w in words] == [" Hello,", " world."]
    assert words[0]["start"] == 10.0 and words[0]["end"] == 10.8
    assert words[1]["start"] == 10.8 and words[1]["end"] <= 14.0


def test_supplied_language_never_loads_whisper():
    pytest.importorskip("stable_whisper")
    m = _model()
    result = m.transcribe(np.zeros(16000 * 4, np.float32), language="en", beam_size=5, vad_filter=True)
    assert result.language == "en" and "Hello there." in result.text
    assert _FakeWhisper.loaded == 0


def test_missing_language_is_detected_then_transcribed_by_parakeet():
    pytest.importorskip("stable_whisper")
    m = _model()
    result = m.transcribe(np.zeros(16000 * 4, np.int16), language=None, input_sr=16000)
    assert result.language == "de"
    assert _FakeWhisper.loaded == 1


def test_unsupported_language_and_translate_fall_back_to_whisper():
    m = _model()
    assert m.transcribe(np.zeros(16000, np.float32), language="ja") == ("whisper", "ja", "transcribe")
    assert m.transcribe(np.zeros(16000, np.float32), language="en", task="translate") == ("whisper", "en", "translate")


def test_fallback_none_raises():
    m = _model()
    m.fallback = "none"
    with pytest.raises(RuntimeError):
        m.transcribe(np.zeros(16000, np.float32), language="ja")


def test_detect_only_uses_up_to_ten_30s_windows():
    m = _model()
    det = m.detect_only(np.zeros(16000 * 300, np.float32), input_sr=16000)
    assert det.language == "de"


def test_unload_drops_both_models():
    m = _model()
    m._ensure_whisper()
    m.model.unload_model()
    assert m._asr is None and m._whisper is None
