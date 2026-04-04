"""
app/tts.py — Text-to-Speech engine using gTTS.

HOW TTS WORKS IN THIS PROJECT:
  1. The LLM reply comes in as a plain string e.g. "أنا بخير شكراً"
  2. gTTS sends that text to Google's TTS API
  3. Google returns an MP3 audio stream
  4. We save it to temp_audio/ with a unique filename
  5. The API returns the URL so the client can play it

WHY gTTS:
  - Zero setup — no model to download, no GPU needed
  - Supports Arabic, English, French and 70+ languages natively
  - Google's voices are natural and clear
  - Free for reasonable usage

LANGUAGE SUPPORT:
  gTTS uses the same language codes as Whisper (ar, en, fr, de...)
  so we can pass the detected language straight through with no mapping.
"""

import uuid
import logging
from pathlib import Path
from gtts import gTTS, lang as gtts_lang

log = logging.getLogger(__name__)

# Languages gTTS supports — we check against this before calling the API
SUPPORTED_LANGS = set(gtts_lang.tts_langs().keys())

# Fallback language if detected language is not supported by gTTS
FALLBACK_LANG = "en"


class TTSEngine:
    """
    Wraps gTTS for use in FastAPI.

    Usage:
        tts = TTSEngine()
        audio_path = tts.speak("مرحبا كيف حالك", language="ar")
        print(audio_path)  # temp_audio/reply_abc123.mp3
    """

    def __init__(self, output_dir: str | Path = "temp_audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        log.info("TTS engine ready (gTTS).")

    # ── Public method ─────────────────────────────────────────────────────────

    def speak(self, text: str, language: str = "en") -> Path:
        """
        Convert text to speech and save as an MP3 file.

        Args:
            text:     the reply text from the LLM
            language: language code from Whisper e.g. "ar", "en", "fr"

        Returns:
            Path to the saved MP3 file

        Raises:
            ValueError: if text is empty
            RuntimeError: if gTTS API call fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot synthesise empty text.")

        # Use fallback if gTTS doesn't support this language
        lang = language if language in SUPPORTED_LANGS else FALLBACK_LANG
        if lang != language:
            log.warning(
                f"Language '{language}' not supported by gTTS — "
                f"falling back to '{FALLBACK_LANG}'"
            )

        # Generate a unique filename for each reply
        filename = f"reply_{uuid.uuid4().hex[:12]}.mp3"
        out_path = self.output_dir / filename

        log.info(f"Synthesising [{lang}]: {text[:60]}...")

        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(str(out_path))
            log.info(f"Audio saved → {out_path.name}")
            return out_path

        except Exception as e:
            raise RuntimeError(f"TTS failed: {e}") from e

    def cleanup(self, path: Path) -> None:
        """Delete a generated audio file after it has been served."""
        try:
            if path.exists():
                path.unlink()
        except Exception as e:
            log.warning(f"Could not delete {path}: {e}")