"""
app/asr.py — Whisper ASR wrapper using faster-whisper.

faster-whisper is a drop-in replacement for openai-whisper.
It uses CTranslate2 under the hood which makes it:
  - 3-4x faster on CPU
  - Uses less memory
  - Installs cleanly on Windows with no build issues
"""

import logging
from pathlib import Path

from faster_whisper import WhisperModel

from app.schemas import TranscribeResponse, SegmentModel

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"}


class WhisperASR:
    def __init__(self, model_size: str = "base"):
        log.info(f"Loading Whisper '{model_size}' ...")
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )
        self.model_size = model_size
        self.device = "cpu"
        log.info(f"Whisper '{model_size}' ready.")

    def transcribe(self, audio_path: str | Path) -> TranscribeResponse:
        audio_path = Path(audio_path)
        self._validate(audio_path)
        log.info(f"Transcribing: {audio_path.name}")

        segments_gen, info = self.model.transcribe(
            str(audio_path),
            task="transcribe",
            language=None,
            beam_size=5,
            vad_filter=True,
        )

        segments = list(segments_gen)
        full_text = " ".join(s.text.strip() for s in segments)
        duration = segments[-1].end if segments else 0.0

        log.info(
            f"Done — lang={info.language} "
            f"({info.language_probability:.2f}), "
            f"duration={duration:.1f}s"
        )

        return TranscribeResponse(
            text=full_text.strip(),
            language=info.language,
            language_probability=round(info.language_probability, 4),
            duration_seconds=round(duration, 2),
            segments=[
                SegmentModel(
                    id=i,
                    start=round(s.start, 2),
                    end=round(s.end, 2),
                    text=s.text.strip(),
                )
                for i, s in enumerate(segments)
            ],
        )

    def _validate(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )