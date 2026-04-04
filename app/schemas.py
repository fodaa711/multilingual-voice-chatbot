"""
app/schemas.py — Pydantic request / response models.

These are the "contracts" for your API.
Every endpoint says exactly what comes in and what goes out.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Response: what /transcribe returns ────────────────────────────────────────

class SegmentModel(BaseModel):
    """One timestamped chunk of speech — useful for subtitles or long audio."""
    id: int
    start: float = Field(description="Start time in seconds")
    end: float   = Field(description="End time in seconds")
    text: str    = Field(description="Transcribed text for this segment")


class TranscribeResponse(BaseModel):
    """
    Returned by POST /transcribe

    Example:
    {
        "text": "مرحبا كيف حالك",
        "language": "ar",
        "language_probability": 0.98,
        "duration_seconds": 3.2,
        "segments": [{"id": 0, "start": 0.0, "end": 3.2, "text": "مرحبا كيف حالك"}]
    }
    """
    text: str = Field(description="Full transcribed text")
    language: str = Field(description="Detected language code, e.g. 'ar', 'en', 'fr'")
    language_probability: float = Field(description="Confidence score 0.0 → 1.0")
    duration_seconds: float = Field(description="Length of the audio clip")
    segments: list[SegmentModel] = Field(
        default=[],
        description="Per-segment timestamps (empty for short clips)"
    )


# ── Response: /health ─────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """
    Returned by GET /health

    Example:
    {
        "status": "ok",
        "whisper_model": "base",
        "device": "cpu"
    }
    """
    status: str
    whisper_model: str
    device: str


# ── Response: errors ─────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str


# ── Request / Response: /chat ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Sent to POST /chat

    Example:
    {
        "text": "مرحبا كيف حالك",
        "language": "ar",
        "session_id": "user_123"
    }
    """
    text: str = Field(description="Transcribed user message")
    language: str = Field(default="en", description="Language code e.g. 'ar', 'en'")
    session_id: str = Field(default="default", description="Unique session ID per user")


class ChatResponse(BaseModel):
    """
    Returned by POST /chat

    Example:
    {
        "reply": "أنا بخير شكراً، كيف يمكنني مساعدتك؟",
        "session_id": "user_123"
    }
    """
    reply: str = Field(description="LLM reply text")
    session_id: str


# ── Request / Response: /voice-loop ──────────────────────────────────────────

class VoiceLoopResponse(BaseModel):
    """
    Returned by POST /voice-loop (full pipeline: audio in → audio out)

    Example:
    {
        "transcript": "مرحبا كيف حالك",
        "language": "ar",
        "reply_text": "أنا بخير، كيف يمكنني مساعدتك؟",
        "audio_url": "/audio/reply_abc123.mp3"
    }
    """
    transcript: str
    language: str
    language_probability: float
    reply_text: str
    audio_url: str
    session_id: str