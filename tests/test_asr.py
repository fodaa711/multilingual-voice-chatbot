"""
tests/test_asr.py — Unit tests for the Whisper ASR module.

What we test in Phase 1:
  1. WhisperASR loads without crashing
  2. Transcribing an English sample returns non-empty text
  3. Transcribing an Arabic sample detects language = "ar"
  4. Transcribing a French sample detects language = "fr"
  5. Passing a non-existent file raises FileNotFoundError
  6. Passing a .txt file raises ValueError (unsupported format)

Run with:
    pytest tests/test_asr.py -v
"""

import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

from app.asr import WhisperASR


# ── Fixture: load the model once for all tests in this file ──────────────────
#
# @pytest.fixture(scope="module") means the model is loaded ONCE
# and reused across all tests — saving ~5 seconds per test.

@pytest.fixture(scope="module")
def asr():
    """Load Whisper base model once for all tests."""
    return WhisperASR(model_size="tiny")   # use tiny in tests — fast enough


# ── Fixture: generate a simple sine-wave tone as a .wav file ─────────────────
#
# We use a synthetic tone for most tests so they don't depend on
# external downloads. For language-detection tests we use the real samples.

@pytest.fixture
def sine_wav(tmp_path):
    """Create a 2-second 440Hz tone .wav — valid audio, no speech."""
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, dtype=np.float32)
    wave = 0.3 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "sine.wav"
    sf.write(str(path), wave, sr)
    return path


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestWhisperASRLoads:
    def test_model_loads(self, asr):
        """WhisperASR should initialise without raising."""
        assert asr is not None
        assert asr.model is not None

    def test_device_is_set(self, asr):
        """Device should be 'cpu' or 'cuda'."""
        assert asr.device in ("cpu", "cuda")

    def test_model_size_stored(self, asr):
        assert asr.model_size == "tiny"


class TestTranscribeResponse:
    def test_returns_transcribe_response(self, asr, sine_wav):
        """transcribe() should return a TranscribeResponse, not raise."""
        from app.schemas import TranscribeResponse
        result = asr.transcribe(sine_wav)
        assert isinstance(result, TranscribeResponse)

    def test_text_is_string(self, asr, sine_wav):
        result = asr.transcribe(sine_wav)
        assert isinstance(result.text, str)

    def test_language_is_string(self, asr, sine_wav):
        result = asr.transcribe(sine_wav)
        assert isinstance(result.language, str)
        assert len(result.language) == 2  # e.g. "en", "ar", "fr"

    def test_language_probability_range(self, asr, sine_wav):
        result = asr.transcribe(sine_wav)
        assert 0.0 <= result.language_probability <= 1.0

    def test_duration_is_positive(self, asr, sine_wav):
        result = asr.transcribe(sine_wav)
        assert result.duration_seconds >= 0.0


class TestRealSamples:
    """
    These tests use real speech from samples/.
    They are skipped automatically if the sample files don't exist yet.
    Run:  python samples/download_samples.py   to download them first.
    """

    @pytest.mark.skipif(
        not Path("samples/en_sample.wav").exists(),
        reason="samples/en_sample.wav not found — run samples/download_samples.py"
    )
    def test_english_sample_transcription(self, asr):
        result = asr.transcribe("samples/en_sample.wav")
        assert result.language == "en"
        assert len(result.text) > 0

    @pytest.mark.skipif(
        not Path("samples/ar_sample.wav").exists(),
        reason="samples/ar_sample.wav not found — run samples/download_samples.py"
    )
    def test_arabic_sample_language_detected(self, asr):
        result = asr.transcribe("samples/ar_sample.wav")
        assert result.language == "ar", (
            f"Expected 'ar', got '{result.language}' — "
            f"try upgrading to whisper model 'base' or 'small' for better Arabic detection"
        )
        assert len(result.text) > 0

    @pytest.mark.skipif(
        not Path("samples/fr_sample.wav").exists(),
        reason="samples/fr_sample.wav not found — run samples/download_samples.py"
    )
    def test_french_sample_language_detected(self, asr):
        result = asr.transcribe("samples/fr_sample.wav")
        assert result.language == "fr"


class TestErrorHandling:
    def test_missing_file_raises(self, asr):
        with pytest.raises(FileNotFoundError):
            asr.transcribe("does_not_exist.wav")

    def test_unsupported_format_raises(self, asr, tmp_path):
        bad_file = tmp_path / "audio.txt"
        bad_file.write_text("this is not audio")
        with pytest.raises(ValueError, match="Unsupported format"):
            asr.transcribe(bad_file)